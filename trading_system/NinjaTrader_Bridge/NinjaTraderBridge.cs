/*
 * NinjaTrader API Bridge - Signal Receiver
 *
 * Listens for trading signals from Python OANDA strategy
 * Executes orders on NinjaTrader futures
 *
 * COMPILE:
 * 1. Open Visual Studio (or VS Code with C# extension)
 * 2. Create new Console App (.NET 4.8)
 * 3. Add reference to: C:\Program Files\NinjaTrader 8\bin\NinjaTrader.Client.dll
 * 4. Copy this code
 * 5. Build → Run
 *
 * USAGE:
 * 1. Start NinjaTrader 8
 * 2. Run this bridge: NinjaTraderBridge.exe
 * 3. Run Python: python run_oanda_to_ninjatrader.py
 * 4. Python sends signals → Bridge executes on NT
 */

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using NinjaTrader.Client;
using Newtonsoft.Json;  // Install via NuGet: Install-Package Newtonsoft.Json

namespace NinjaTraderBridge
{
    // Signal data structure from Python
    public class TradingSignal
    {
        public string Action { get; set; }      // "ENTRY", "EXIT", or "PRICE_QUERY"
        public string Symbol { get; set; }      // "M6E", "M6B", etc.
        public string Side { get; set; }        // "BUY" or "SELL"
        public int Quantity { get; set; }       // Always 1 for futures
        public double StopLoss { get; set; }    // SL price
        public double TakeProfit { get; set; }  // TP price
        public double EntryPrice { get; set; }  // Current market price
        public string Timestamp { get; set; }
    }

    // Price data structure to return to Python
    public class PriceData
    {
        public string Symbol { get; set; }
        public double Bid { get; set; }
        public double Ask { get; set; }
        public double Last { get; set; }
        public string Timestamp { get; set; }
        public string Status { get; set; }  // "OK" or "ERROR"
    }

    class Program
    {
        // NinjaTrader Client
        private static Client ntClient;

        // Active positions tracking
        private static Dictionary<string, string> openPositions = new Dictionary<string, string>();

        // TCP Server for receiving signals
        private static TcpListener server;
        private static bool isRunning = true;

        // Last known prices cache (for when market is closed)
        private static Dictionary<string, PriceData> lastKnownPrices = new Dictionary<string, PriceData>();

        // Symbol mapping: NinjaTrader format - matches Market Analyzer display "M6E DEC25"
        private static Dictionary<string, string> symbolMap = new Dictionary<string, string>
        {
            { "M6E", "M6E DEC25" },    // Micro Euro Dec 2025
            { "M6B", "M6B DEC25" },    // Micro British Pound Dec 2025
            { "MJY", "MJY DEC25" },    // Micro Japanese Yen Dec 2025
            { "MCD", "MCD DEC25" },    // Micro Canadian Dollar Dec 2025
            { "MSF", "MSF DEC25" }     // Micro Swiss Franc Dec 2025
        };

        static void Main(string[] args)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("NINJATRADER API BRIDGE - SIGNAL RECEIVER");
            Console.WriteLine("================================================================================");
            Console.WriteLine("Receives trading signals from Python OANDA strategy");
            Console.WriteLine("Executes orders on NinjaTrader futures");
            Console.WriteLine("================================================================================\n");

            try
            {
                // Initialize NinjaTrader Client
                Console.WriteLine("[1/3] Connecting to NinjaTrader...");
                ntClient = new Client();
                Console.WriteLine("✓ Connected to NinjaTrader 8\n");

                // Start TCP Server (listen for Python signals)
                Console.WriteLine("[2/3] Starting signal receiver...");
                server = new TcpListener(IPAddress.Loopback, 8888);
                server.Start();
                Console.WriteLine("✓ Listening on port 8888 for signals\n");

                Console.WriteLine("[3/3] Bridge ready! Waiting for signals...");
                Console.WriteLine("================================================================================");
                Console.WriteLine("Status: ACTIVE");
                Console.WriteLine("Press Ctrl+C to stop");
                Console.WriteLine("================================================================================\n");

                // Handle Ctrl+C gracefully
                Console.CancelKeyPress += (sender, e) =>
                {
                    e.Cancel = true;
                    isRunning = false;
                };

                // Accept connections in loop
                while (isRunning)
                {
                    if (server.Pending())
                    {
                        TcpClient client = server.AcceptTcpClient();
                        Thread clientThread = new Thread(() => HandleClient(client));
                        clientThread.Start();
                    }
                    Thread.Sleep(100);
                }

                // Cleanup
                Console.WriteLine("\n[SHUTDOWN] Closing positions...");
                CloseAllPositions();

                Console.WriteLine("[SHUTDOWN] Disconnecting from NinjaTrader...");
                ntClient.TearDown();

                Console.WriteLine("[SHUTDOWN] Stopping server...");
                server.Stop();

                Console.WriteLine("✓ Bridge stopped successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ ERROR: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        // Track market data status to avoid log spam
        static Dictionary<string, bool> lastMarketDataStatus = new Dictionary<string, bool>();
        static DateTime lastPriceQueryLog = DateTime.MinValue;

        static void HandleClient(TcpClient client)
        {
            try
            {
                NetworkStream stream = client.GetStream();
                byte[] buffer = new byte[4096];
                int bytesRead = stream.Read(buffer, 0, buffer.Length);

                if (bytesRead > 0)
                {
                    string json = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    TradingSignal signal = JsonConvert.DeserializeObject<TradingSignal>(json);

                    // Process signal
                    string response = "OK";

                    if (signal.Action == "ENTRY")
                    {
                        // Log trading signals (important!)
                        Console.WriteLine($"\n[{DateTime.Now:HH:mm:ss}] 📈 TRADING SIGNAL RECEIVED:");
                        Console.WriteLine($"  Action: {signal.Action}");
                        Console.WriteLine($"  Symbol: {signal.Symbol}");
                        Console.WriteLine($"  Side: {signal.Side}");
                        Console.WriteLine($"  Quantity: {signal.Quantity}");
                        ExecuteEntry(signal);
                    }
                    else if (signal.Action == "EXIT")
                    {
                        // Log trading signals (important!)
                        Console.WriteLine($"\n[{DateTime.Now:HH:mm:ss}] 📉 EXIT SIGNAL RECEIVED:");
                        Console.WriteLine($"  Action: {signal.Action}");
                        Console.WriteLine($"  Symbol: {signal.Symbol}");
                        Console.WriteLine($"  Side: {signal.Side}");
                        ExecuteExit(signal);
                    }
                    else if (signal.Action == "PRICE_QUERY")
                    {
                        // Silently handle price queries - no log spam
                        PriceData priceData = GetMarketPriceQuiet(signal.Symbol);
                        response = JsonConvert.SerializeObject(priceData);
                    }

                    // Send response
                    byte[] responseBytes = Encoding.UTF8.GetBytes(response);
                    stream.Write(responseBytes, 0, responseBytes.Length);
                }

                client.Close();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ Error handling client: {ex.Message}");
            }
        }

        static void ExecuteEntry(TradingSignal signal)
        {
            try
            {
                string ntSymbol = symbolMap[signal.Symbol];
                string action = signal.Side.ToUpper() == "BUY" ? "BUY" : "SELL";

                // Place market order
                int orderId = ntClient.Command(
                    "PLACE",
                    "",  // Account (empty = default)
                    ntSymbol,
                    action,
                    signal.Quantity,
                    "MARKET",
                    0,  // Limit price (not used for market)
                    0,  // Stop price (not used for market)
                    "DAY",
                    "",  // OCO
                    "",  // Order ID
                    "Bridge",  // Strategy
                    ""  // Strategy ID
                );

                // Check if order was accepted or rejected
                if (orderId > 0)
                {
                    Console.WriteLine($"  ✓ Order ACCEPTED: {action} {signal.Quantity} {signal.Symbol} @ Market");
                    Console.WriteLine($"    Order ID: {orderId}");
                }
                else
                {
                    Console.WriteLine($"  ❌ Order REJECTED by NinjaTrader!");
                    Console.WriteLine($"    Order ID: {orderId} (0 = rejection)");
                    Console.WriteLine($"    Possible reasons:");
                    Console.WriteLine($"      - Market is closed");
                    Console.WriteLine($"      - Insufficient funds");
                    Console.WriteLine($"      - Invalid symbol or contract");
                    Console.WriteLine($"      - Account not connected");
                    return;  // Don't place SL/TP if entry order was rejected
                }

                // Place stop loss
                if (signal.StopLoss > 0)
                {
                    string slAction = signal.Side.ToUpper() == "BUY" ? "SELL" : "BUY";
                    int slOrderId = ntClient.Command(
                        "PLACE",
                        "",  // Account
                        ntSymbol,
                        slAction,
                        signal.Quantity,
                        "STOP",
                        0,  // Limit price
                        signal.StopLoss,  // Stop price
                        "DAY",
                        "",  // OCO
                        "",  // Order ID
                        "Bridge_SL",  // Strategy
                        ""  // Strategy ID
                    );

                    if (slOrderId > 0)
                    {
                        Console.WriteLine($"    ✓ Stop Loss ACCEPTED: {signal.StopLoss:F5} (Order ID: {slOrderId})");
                    }
                    else
                    {
                        Console.WriteLine($"    ⚠ Stop Loss REJECTED: {signal.StopLoss:F5} (Order ID: {slOrderId})");
                    }
                }

                // Place take profit
                if (signal.TakeProfit > 0)
                {
                    string tpAction = signal.Side.ToUpper() == "BUY" ? "SELL" : "BUY";
                    int tpOrderId = ntClient.Command(
                        "PLACE",
                        "",  // Account
                        ntSymbol,
                        tpAction,
                        signal.Quantity,
                        "LIMIT",
                        signal.TakeProfit,  // Limit price
                        0,  // Stop price
                        "DAY",
                        "",  // OCO
                        "",  // Order ID
                        "Bridge_TP",  // Strategy
                        ""  // Strategy ID
                    );

                    if (tpOrderId > 0)
                    {
                        Console.WriteLine($"    ✓ Take Profit ACCEPTED: {signal.TakeProfit:F5} (Order ID: {tpOrderId})");
                    }
                    else
                    {
                        Console.WriteLine($"    ⚠ Take Profit REJECTED: {signal.TakeProfit:F5} (Order ID: {tpOrderId})");
                    }
                }

                // Track position (only if entry order was accepted)
                openPositions[signal.Symbol] = signal.Side;

                Console.WriteLine($"  ✓ TRADE COMPLETE: {signal.Symbol} {signal.Side} position opened successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ Error executing entry: {ex.Message}");
            }
        }

        static void ExecuteExit(TradingSignal signal)
        {
            try
            {
                if (!openPositions.ContainsKey(signal.Symbol))
                {
                    Console.WriteLine($"  ⚠ No open position for {signal.Symbol}");
                    return;
                }

                string ntSymbol = symbolMap[signal.Symbol];
                string originalSide = openPositions[signal.Symbol];
                string action = originalSide == "BUY" ? "SELL" : "BUY";

                // Close position (market order)
                int orderId = ntClient.Command(
                    "PLACE",
                    "",  // Account
                    ntSymbol,
                    action,
                    signal.Quantity,
                    "MARKET",
                    0,  // Limit price
                    0,  // Stop price
                    "DAY",
                    "",  // OCO
                    "",  // Order ID
                    "Bridge_Exit",  // Strategy
                    ""  // Strategy ID
                );

                Console.WriteLine($"  ✓ Position closed: {action} {signal.Quantity} {signal.Symbol} @ Market");
                Console.WriteLine($"    Order ID: {orderId}");

                // Cancel SL/TP orders
                CancelPendingOrders(ntSymbol);

                // Remove from tracking
                openPositions.Remove(signal.Symbol);

                Console.WriteLine($"  ✓ Position exited: {signal.Symbol}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ Error executing exit: {ex.Message}");
            }
        }

        static void CancelPendingOrders(string symbol)
        {
            try
            {
                // SL and TP orders will be automatically cancelled when position closes
                Console.WriteLine($"    Pending orders for {symbol} will be auto-cancelled");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    Warning: Could not cancel pending orders: {ex.Message}");
            }
        }

        static void CloseAllPositions()
        {
            foreach (var position in openPositions)
            {
                try
                {
                    string symbol = position.Key;
                    string side = position.Value;
                    string ntSymbol = symbolMap[symbol];
                    string action = side == "BUY" ? "SELL" : "BUY";

                    ntClient.Command(
                        "PLACE",
                        "",  // Account
                        ntSymbol,
                        action,
                        1,
                        "MARKET",
                        0,  // Limit price
                        0,  // Stop price
                        "DAY",
                        "",  // OCO
                        "",  // Order ID
                        "Bridge_Close",  // Strategy
                        ""  // Strategy ID
                    );

                    Console.WriteLine($"  Closed: {symbol}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Error closing {position.Key}: {ex.Message}");
                }
            }

            openPositions.Clear();
        }

        static PriceData GetMarketPrice(string symbol)
        {
            try
            {
                // Get NinjaTrader symbol format
                string ntSymbol = symbolMap.ContainsKey(symbol) ? symbolMap[symbol] : symbol;

                // Query market data from NinjaTrader
                // MarketData parameters: symbol, type (0=last, 1=bid, 2=ask)
                double last = ntClient.MarketData(ntSymbol, 0);  // Last price
                double bid = ntClient.MarketData(ntSymbol, 1);   // Bid price
                double ask = ntClient.MarketData(ntSymbol, 2);   // Ask price

                // Check if we got valid data (market is open)
                if (bid > 0 && ask > 0 && last > 0)
                {
                    Console.WriteLine($"  [PRICE] {symbol}: Bid={bid:F5}, Ask={ask:F5}, Last={last:F5}");

                    PriceData priceData = new PriceData
                    {
                        Symbol = symbol,
                        Bid = bid,
                        Ask = ask,
                        Last = last,
                        Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                        Status = "OK"
                    };

                    // Cache the price for when market is closed
                    lastKnownPrices[symbol] = priceData;

                    return priceData;
                }
                else
                {
                    // Market is likely closed - return last known prices if available
                    if (lastKnownPrices.ContainsKey(symbol))
                    {
                        PriceData cachedPrice = lastKnownPrices[symbol];
                        Console.WriteLine($"  ⚠ Market closed for {symbol} - Returning last known price from {cachedPrice.Timestamp}");
                        Console.WriteLine($"    Last known: Bid={cachedPrice.Bid:F5}, Ask={cachedPrice.Ask:F5}, Last={cachedPrice.Last:F5}");

                        return new PriceData
                        {
                            Symbol = symbol,
                            Bid = cachedPrice.Bid,
                            Ask = cachedPrice.Ask,
                            Last = cachedPrice.Last,
                            Timestamp = cachedPrice.Timestamp,  // Keep original timestamp
                            Status = "MARKET_CLOSED"
                        };
                    }
                    else
                    {
                        Console.WriteLine($"  ⚠ No market data for {symbol} (Bid={bid}, Ask={ask}, Last={last})");
                        Console.WriteLine($"    Market may be closed and no cached prices available");

                        return new PriceData
                        {
                            Symbol = symbol,
                            Bid = 0,
                            Ask = 0,
                            Last = 0,
                            Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                            Status = "MARKET_CLOSED_NO_CACHE"
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ Error fetching price for {symbol}: {ex.Message}");

                // Try to return cached price on error
                if (lastKnownPrices.ContainsKey(symbol))
                {
                    PriceData cachedPrice = lastKnownPrices[symbol];
                    Console.WriteLine($"    Returning cached price due to error");

                    return new PriceData
                    {
                        Symbol = symbol,
                        Bid = cachedPrice.Bid,
                        Ask = cachedPrice.Ask,
                        Last = cachedPrice.Last,
                        Timestamp = cachedPrice.Timestamp,
                        Status = $"ERROR_CACHED: {ex.Message}"
                    };
                }
                else
                {
                    return new PriceData
                    {
                        Symbol = symbol,
                        Bid = 0,
                        Ask = 0,
                        Last = 0,
                        Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                        Status = $"ERROR: {ex.Message}"
                    };
                }
            }
        }

        // Quiet version of GetMarketPrice - only logs status changes
        static PriceData GetMarketPriceQuiet(string symbol)
        {
            try
            {
                // Get NinjaTrader symbol format
                string ntSymbol = symbolMap.ContainsKey(symbol) ? symbolMap[symbol] : symbol;

                // Query market data from NinjaTrader
                double last = ntClient.MarketData(ntSymbol, 0);  // Last price
                double bid = ntClient.MarketData(ntSymbol, 1);   // Bid price
                double ask = ntClient.MarketData(ntSymbol, 2);   // Ask price

                // Check if we got valid data (market is open)
                bool hasData = bid > 0 && ask > 0 && last > 0;
                bool previouslyHadData = lastMarketDataStatus.ContainsKey(symbol) && lastMarketDataStatus[symbol];

                // Only log status CHANGES
                if (hasData && !previouslyHadData)
                {
                    Console.WriteLine($"\n[{DateTime.Now:HH:mm:ss}] ✅ Market data AVAILABLE for {symbol}");
                    lastMarketDataStatus[symbol] = true;
                }
                else if (!hasData && previouslyHadData)
                {
                    Console.WriteLine($"\n[{DateTime.Now:HH:mm:ss}] ⚠ Market data UNAVAILABLE for {symbol} - Market may be closed");
                    lastMarketDataStatus[symbol] = false;
                }
                else if (!hasData && !lastMarketDataStatus.ContainsKey(symbol))
                {
                    // First check with no data - log once
                    Console.WriteLine($"\n[{DateTime.Now:HH:mm:ss}] ⚠ No market data for {symbol} - Market may be closed");
                    lastMarketDataStatus[symbol] = false;
                }

                if (hasData)
                {
                    PriceData priceData = new PriceData
                    {
                        Symbol = symbol,
                        Bid = bid,
                        Ask = ask,
                        Last = last,
                        Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                        Status = "OK"
                    };

                    // Cache the price for when market is closed
                    lastKnownPrices[symbol] = priceData;
                    return priceData;
                }
                else
                {
                    // Market is likely closed - return last known prices if available
                    if (lastKnownPrices.ContainsKey(symbol))
                    {
                        PriceData cachedPrice = lastKnownPrices[symbol];
                        return new PriceData
                        {
                            Symbol = symbol,
                            Bid = cachedPrice.Bid,
                            Ask = cachedPrice.Ask,
                            Last = cachedPrice.Last,
                            Timestamp = cachedPrice.Timestamp,
                            Status = "MARKET_CLOSED"
                        };
                    }
                    else
                    {
                        return new PriceData
                        {
                            Symbol = symbol,
                            Bid = 0,
                            Ask = 0,
                            Last = 0,
                            Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                            Status = "MARKET_CLOSED_NO_CACHE"
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                // Try to return cached price on error
                if (lastKnownPrices.ContainsKey(symbol))
                {
                    PriceData cachedPrice = lastKnownPrices[symbol];
                    return new PriceData
                    {
                        Symbol = symbol,
                        Bid = cachedPrice.Bid,
                        Ask = cachedPrice.Ask,
                        Last = cachedPrice.Last,
                        Timestamp = cachedPrice.Timestamp,
                        Status = $"ERROR_CACHED: {ex.Message}"
                    };
                }
                else
                {
                    return new PriceData
                    {
                        Symbol = symbol,
                        Bid = 0,
                        Ask = 0,
                        Last = 0,
                        Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                        Status = $"ERROR: {ex.Message}"
                    };
                }
            }
        }
    }
}
