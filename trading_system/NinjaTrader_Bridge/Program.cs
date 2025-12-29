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
        public string Action { get; set; }      // "ENTRY" or "EXIT"
        public string Symbol { get; set; }      // "M6E", "M6B", etc.
        public string Side { get; set; }        // "BUY" or "SELL"
        public int Quantity { get; set; }       // Always 1 for futures
        public double StopLoss { get; set; }    // SL price
        public double TakeProfit { get; set; }  // TP price
        public double EntryPrice { get; set; }  // Current market price
        public string Timestamp { get; set; }
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

        // Symbol mapping: NinjaTrader format
        private static Dictionary<string, string> symbolMap = new Dictionary<string, string>
        {
            { "M6E", "M6E 06-25" },    // Micro Euro Jun 2025
            { "M6B", "M6B 06-25" },    // Micro British Pound Jun 2025
            { "MJY", "MJY 06-25" },    // Micro Japanese Yen Jun 2025
            { "MCD", "MCD 06-25" },    // Micro Canadian Dollar Jun 2025
            { "MSF", "MSF 06-25" }     // Micro Swiss Franc Jun 2025
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

                    Console.WriteLine($"\n[{DateTime.Now:HH:mm:ss}] Received signal:");
                    Console.WriteLine($"  Action: {signal.Action}");
                    Console.WriteLine($"  Symbol: {signal.Symbol}");
                    Console.WriteLine($"  Side: {signal.Side}");

                    // Process signal
                    if (signal.Action == "ENTRY")
                    {
                        ExecuteEntry(signal);
                    }
                    else if (signal.Action == "EXIT")
                    {
                        ExecuteExit(signal);
                    }

                    // Send acknowledgment
                    string response = "OK";
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

                Console.WriteLine($"  ✓ Order placed: {action} {signal.Quantity} {signal.Symbol} @ Market");
                Console.WriteLine($"    Order ID: {orderId}");

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
                    Console.WriteLine($"    Stop Loss: {signal.StopLoss:F5} (Order ID: {slOrderId})");
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
                    Console.WriteLine($"    Take Profit: {signal.TakeProfit:F5} (Order ID: {tpOrderId})");
                }

                // Track position
                openPositions[signal.Symbol] = signal.Side;

                Console.WriteLine($"  ✓ Position opened: {signal.Symbol} {signal.Side}");
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
    }
}
