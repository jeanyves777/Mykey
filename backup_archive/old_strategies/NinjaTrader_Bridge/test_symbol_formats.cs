using System;
using NinjaTrader.Client;

class TestSymbolFormats
{
    static void Main()
    {
        Console.WriteLine("Testing NinjaTrader Symbol Formats for M6E");
        Console.WriteLine("==========================================\n");

        var client = new Client();
        int connected = client.Connected(1);

        if (connected != 0)
        {
            Console.WriteLine("ERROR: Not connected to NinjaTrader!");
            Console.WriteLine("Make sure NinjaTrader is running and ATI is enabled.");
            return;
        }

        Console.WriteLine("Connected to NinjaTrader!\n");

        // Test different symbol formats
        string[] formats = new string[] {
            "M6E DEC25",
            "M6E 12-25",
            "M6E DEC 25",
            "M6E Z25",      // Z = December futures code
            "M6E Z5",
            "M6EZ25",
            "M6EZ5",
            "M6E 12-2025",
            "M6E",
            "@M6E",
            "M6E MAR26",
            "M6E H26",      // H = March futures code
            "M6E 03-26"
        };

        Console.WriteLine("Format               | Bid      | Ask      | Last     | Status");
        Console.WriteLine("---------------------|----------|----------|----------|--------");

        foreach (string symbol in formats)
        {
            try
            {
                double bid = client.MarketData(symbol, 1);
                double ask = client.MarketData(symbol, 2);
                double last = client.MarketData(symbol, 0);

                string status = (bid > 0 && ask > 0) ? "OK" : "NO DATA";
                Console.WriteLine($"{symbol,-20} | {bid,8:F5} | {ask,8:F5} | {last,8:F5} | {status}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{symbol,-20} | ERROR: {ex.Message}");
            }
        }

        Console.WriteLine("\nPress Enter to exit...");
        Console.ReadLine();
    }
}
