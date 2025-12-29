# NinjaTrader Bridge - Price Caching System

## Overview

The NinjaTrader bridge now includes an intelligent **price caching system** that stores the last known prices when the market is open, and returns them when the market is closed. This matches the behavior of other platforms like OANDA that display last known prices even when markets are closed.

## The Problem

Previously, when the market was closed (weekends, holidays), NinjaTrader's `MarketData()` API would return zeros:
```
Bid = 0.0
Ask = 0.0
Last = 0.0
```

This caused the bridge to return `"ERROR: No market data"`, giving users no visibility into what prices were before the market closed.

OANDA, by contrast, returns the last known prices even when the market is closed, which is more user-friendly for monitoring and comparison.

## The Solution

The bridge now implements a **price cache** that:

1. **Stores prices when market is open** - Every successful price query is cached
2. **Returns cached prices when market is closed** - If NinjaTrader returns zeros, the bridge returns the last known prices with a `"MARKET_CLOSED"` status
3. **Preserves original timestamps** - Cached prices retain their original timestamp so you know when they were valid
4. **Handles errors gracefully** - If an error occurs, the bridge tries to return cached prices instead of failing completely

## Status Codes

The bridge now returns different status codes to indicate the state of the price data:

### `OK` - Live Market Data
```json
{
  "Symbol": "M6E",
  "Bid": 1.05123,
  "Ask": 1.05145,
  "Last": 1.05134,
  "Timestamp": "2025-12-13 15:30:45",
  "Status": "OK"
}
```
**Meaning**: Market is open, these are current live prices from NinjaTrader.

### `MARKET_CLOSED` - Cached Prices (Market Closed)
```json
{
  "Symbol": "M6E",
  "Bid": 1.05123,
  "Ask": 1.05145,
  "Last": 1.05134,
  "Timestamp": "2025-12-13 17:00:00",
  "Status": "MARKET_CLOSED"
}
```
**Meaning**: Market is closed, these are the last known prices before the market closed. The timestamp shows when these prices were valid.

### `MARKET_CLOSED_NO_CACHE` - No Data Available
```json
{
  "Symbol": "M6E",
  "Bid": 0,
  "Ask": 0,
  "Last": 0,
  "Timestamp": "2025-12-13 15:30:45",
  "Status": "MARKET_CLOSED_NO_CACHE"
}
```
**Meaning**: Market is closed AND the bridge has never received prices for this symbol (bridge just started, or symbol never queried before).

### `ERROR_CACHED: <error message>` - Error with Cached Fallback
```json
{
  "Symbol": "M6E",
  "Bid": 1.05123,
  "Ask": 1.05145,
  "Last": 1.05134,
  "Timestamp": "2025-12-13 17:00:00",
  "Status": "ERROR_CACHED: Connection timeout"
}
```
**Meaning**: An error occurred while fetching prices, but cached prices are available as a fallback.

### `ERROR: <error message>` - Error without Cache
```json
{
  "Symbol": "M6E",
  "Bid": 0,
  "Ask": 0,
  "Last": 0,
  "Timestamp": "2025-12-13 15:30:45",
  "Status": "ERROR: Connection timeout"
}
```
**Meaning**: An error occurred and no cached prices are available.

## How It Works

### C# Bridge Implementation

```csharp
// Price cache - stores last known prices for each symbol
private static Dictionary<string, PriceData> lastKnownPrices = new Dictionary<string, PriceData>();

static PriceData GetMarketPrice(string symbol)
{
    // Query NinjaTrader
    double bid = ntClient.MarketData(ntSymbol, 1);
    double ask = ntClient.MarketData(ntSymbol, 2);
    double last = ntClient.MarketData(ntSymbol, 0);

    // If we got valid data (market is open)
    if (bid > 0 && ask > 0 && last > 0)
    {
        // Create price data
        PriceData priceData = new PriceData { ... };

        // Cache it for when market closes
        lastKnownPrices[symbol] = priceData;

        return priceData;  // Status = "OK"
    }
    else
    {
        // Market is closed - return cached prices if available
        if (lastKnownPrices.ContainsKey(symbol))
        {
            PriceData cachedPrice = lastKnownPrices[symbol];
            return new PriceData
            {
                Bid = cachedPrice.Bid,
                Ask = cachedPrice.Ask,
                Last = cachedPrice.Last,
                Timestamp = cachedPrice.Timestamp,  // Original timestamp
                Status = "MARKET_CLOSED"
            };
        }
        else
        {
            // No cache available
            return new PriceData { Status = "MARKET_CLOSED_NO_CACHE" };
        }
    }
}
```

### Python Client Handling

```python
def get_ninjatrader_price(self, nt_symbol: str) -> Optional[Dict]:
    # Query bridge
    price_data = json.loads(response)
    status = price_data.get('Status', '')

    if status == 'OK':
        # Live market data
        return {
            'symbol': nt_symbol,
            'bid': price_data['Bid'],
            'ask': price_data['Ask'],
            'last': price_data['Last'],
            'source': 'NinjaTrader',
            'status': 'LIVE'
        }
    elif status == 'MARKET_CLOSED':
        # Cached prices from when market was open
        return {
            'symbol': nt_symbol,
            'bid': price_data['Bid'],
            'ask': price_data['Ask'],
            'last': price_data['Last'],
            'source': 'NinjaTrader (cached)',
            'timestamp': price_data['Timestamp'],
            'status': 'CACHED'
        }
    elif status == 'MARKET_CLOSED_NO_CACHE':
        # No data available
        print(f"⚠ {nt_symbol}: Market closed, no cached prices")
        return None
    else:
        # Error
        print(f"⚠ Error: {status}")
        return None
```

## Display Examples

### When Market is Open (Status: OK)
```
Market Prices - OANDA vs NinjaTrader Comparison:
Symbol Source                 Bid          Ask          Mid          Diff (pips)
----------------------------------------------------------------------------------
M6E    OANDA                  1.05123      1.05145      1.05134
       NinjaTrader            1.05118      1.05143      1.05131      0.6
----------------------------------------------------------------------------------
```

### When Market is Closed (Status: MARKET_CLOSED)
```
Market Prices - OANDA vs NinjaTrader Comparison:
Symbol Source                      Bid          Ask          Mid          Diff (pips)
--------------------------------------------------------------------------------------
M6E    OANDA                       1.05123      1.05145      1.05134
       NinjaTrader (cached)        1.05118      1.05143      1.05131      0.6
       [2025-12-13 17:00:00]
--------------------------------------------------------------------------------------
```
**Note**: The timestamp shows when the cached prices were last valid (market close time).

### When Market is Closed (No Cache)
```
Market Prices - OANDA vs NinjaTrader Comparison:
Symbol Source                 Bid          Ask          Mid          Diff (pips)
----------------------------------------------------------------------------------
M6E    OANDA                  1.05123      1.05145      1.05134
       ⚠ NinjaTrader: Market closed, no cached prices available
----------------------------------------------------------------------------------
```

## Benefits

### Before (No Caching)
- ❌ Returns "ERROR: No market data" when market is closed
- ❌ No visibility into last known prices
- ❌ Cannot compare OANDA vs NinjaTrader when market is closed
- ❌ Users have no reference point for price differences

### After (With Caching)
- ✅ Returns last known prices when market is closed
- ✅ Shows timestamp of when prices were valid
- ✅ Allows OANDA vs NinjaTrader comparison even when market is closed
- ✅ Users can see historical price differences
- ✅ Better user experience matching OANDA's behavior

## Use Cases

### 1. Weekend Testing
When testing on weekends, you can:
- See the last prices before market closed (Friday 5pm ET)
- Compare OANDA vs NinjaTrader pricing from Friday close
- Validate the dual pricing system is working correctly
- Ensure the bridge is connected and functional

### 2. Pre-Market Preparation
Before market opens (Sunday 5pm ET), you can:
- Review last week's closing prices
- Check price differences between OANDA and NinjaTrader
- Verify the bridge is ready for live trading

### 3. Post-Market Analysis
After market closes (Friday 5pm ET), you can:
- Review closing prices
- Analyze final price differences
- Prepare for next week's trading

### 4. Error Recovery
If NinjaTrader connection is temporarily lost:
- Bridge returns cached prices with `ERROR_CACHED` status
- Trading can continue using cached prices as reference
- Users are alerted to the error but still have data

## Testing the System

### Test When Market is Open (Sunday 5pm - Friday 5pm ET)

```powershell
python test_dual_pricing_system.py
```

**Expected Output**:
```
DUAL PRICING SYSTEM TEST - OANDA vs NinjaTrader
================================================================================
[1/3] Connecting to OANDA...
✓ Connected to OANDA

[2/3] Testing NinjaTrader bridge connection...
✓ Connected to NinjaTrader bridge

[3/3] Fetching prices from both sources...

PRICE COMPARISON - OANDA vs NinjaTrader
================================================================================
Symbol Source                 Bid          Ask          Mid          Diff (pips)
----------------------------------------------------------------------------------
M6E    OANDA                  1.05123      1.05145      1.05134
       NinjaTrader            1.05118      1.05143      1.05131      0.6
----------------------------------------------------------------------------------
```
**Status**: `OK` - Live prices

### Test When Market is Closed (After First Run)

```powershell
python test_dual_pricing_system.py
```

**Expected Output**:
```
DUAL PRICING SYSTEM TEST - OANDA vs NinjaTrader
================================================================================
[1/3] Connecting to OANDA...
✓ Connected to OANDA

[2/3] Testing NinjaTrader bridge connection...
✓ Connected to NinjaTrader bridge

[3/3] Fetching prices from both sources...

PRICE COMPARISON - OANDA vs NinjaTrader
================================================================================
Symbol Source                      Bid          Ask          Mid          Diff (pips)
--------------------------------------------------------------------------------------
M6E    OANDA                       1.05123      1.05145      1.05134
       NinjaTrader (cached)        1.05118      1.05143      1.05131      0.6
       [2025-12-13 17:00:00]
--------------------------------------------------------------------------------------
```
**Status**: `MARKET_CLOSED` - Cached prices from last market close

### Test When Bridge Just Started (Market Closed, No Cache)

```powershell
# Start bridge for first time when market is closed
.\NinjaTraderBridge.exe

# Then test
python test_dual_pricing_system.py
```

**Expected Output**:
```
DUAL PRICING SYSTEM TEST - OANDA vs NinjaTrader
================================================================================
[1/3] Connecting to OANDA...
✓ Connected to OANDA

[2/3] Testing NinjaTrader bridge connection...
✓ Connected to NinjaTrader bridge

[3/3] Fetching prices from both sources...

PRICE COMPARISON - OANDA vs NinjaTrader
================================================================================
❌ M6E (Micro EUR/USD) - NinjaTrader error: Market closed, no cached prices available
❌ M6B (Micro GBP/USD) - NinjaTrader error: Market closed, no cached prices available
...
```
**Status**: `MARKET_CLOSED_NO_CACHE` - No prices available

## Upgrading to the New Bridge

### Prerequisites
1. Stop the old bridge (Ctrl+C in the bridge window)
2. Make sure NinjaTrader 8 is running

### Upgrade Steps

```powershell
cd "C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge"

# Backup old bridge (optional)
copy NinjaTraderBridge.exe NinjaTraderBridge_OLD.exe

# Install new bridge
copy NinjaTraderBridge_V3.exe NinjaTraderBridge.exe

# Start new bridge
.\NinjaTraderBridge.exe
```

**You should see**:
```
================================================================================
NINJATRADER API BRIDGE - SIGNAL RECEIVER
================================================================================
Receives trading signals from Python OANDA strategy
Executes orders on NinjaTrader futures
================================================================================

[1/3] Connecting to NinjaTrader...
✓ Connected to NinjaTrader 8

[2/3] Starting signal receiver...
✓ Listening on port 8888 for signals

[3/3] Bridge ready! Waiting for signals...
================================================================================
Status: ACTIVE
Press Ctrl+C to stop
================================================================================
```

### Test the New Bridge

```powershell
# Open a new PowerShell window
python test_dual_pricing_system.py
```

## Bridge Console Output Examples

### When Querying Prices (Market Open)
```
[15:30:45] Received signal:
  Action: PRICE_QUERY
  Symbol: M6E
  Side:
  [PRICE] M6E: Bid=1.05118, Ask=1.05143, Last=1.05131
```

### When Querying Prices (Market Closed, Cache Available)
```
[10:15:30] Received signal:
  Action: PRICE_QUERY
  Symbol: M6E
  Side:
  ⚠ Market closed for M6E - Returning last known price from 2025-12-13 17:00:00
    Last known: Bid=1.05118, Ask=1.05143, Last=1.05131
```

### When Querying Prices (Market Closed, No Cache)
```
[10:15:30] Received signal:
  Action: PRICE_QUERY
  Symbol: M6E
  Side:
  ⚠ No market data for M6E (Bid=0, Ask=0, Last=0)
    Market may be closed and no cached prices available
```

## Important Notes

### Cache Persistence
- **Cache is in-memory only** - When you stop the bridge, the cache is cleared
- **Cache is built on first query** - Prices are cached the first time you query them while market is open
- **Cache is per-symbol** - Each symbol has its own cached prices

### Cache Lifetime
- Cached prices remain valid until:
  1. Bridge is restarted (cache is cleared)
  2. New prices are received (cache is updated)

### When to Use Cached Prices
✅ **Good for**:
- Monitoring and comparison when market is closed
- Verifying bridge connectivity
- Pre-market preparation
- Post-market analysis

❌ **NOT good for**:
- Live trading decisions (only use `OK` status prices)
- Real-time execution (check status == 'OK' before trading)
- TP/SL calculations when placing trades (verify market is open)

### Trading Safety
The Python live trading script should ALWAYS check the status before placing trades:

```python
nt_price_data = self.get_ninjatrader_price(nt_symbol)

if nt_price_data and nt_price_data['status'] == 'LIVE':
    # Safe to trade - market is open, prices are current
    entry_price = nt_price_data['ask']
    sl_price, tp_price = self.calculate_sl_tp(nt_symbol, 'BUY', entry_price)
    # Send trade...
else:
    # DO NOT TRADE - market is closed or error
    print(f"Cannot trade {nt_symbol} - market closed or no live data")
```

## Troubleshooting

### "Market closed, no cached prices available"
**Cause**: Bridge was just started when market is closed, or symbol never queried before

**Fix**: Wait until market opens (Sunday 5pm ET), then query prices. After first successful query, prices will be cached.

### Cached prices are from many hours/days ago
**Cause**: Bridge has been running continuously without being restarted

**Fix**: This is normal. The timestamp shows when the prices were last valid. Restart the bridge when market opens to get fresh prices.

### Bridge returns cached prices but market should be open
**Cause**: NinjaTrader is disconnected or data feed is not active

**Fix**:
1. Check NinjaTrader connection status
2. Verify you're connected to an account (Sim101 or FundedNext)
3. Check that market data subscription is active
4. Try disconnecting and reconnecting in NinjaTrader

### Cached prices differ significantly from OANDA
**Cause**: Cached prices are from market close (Friday 5pm), OANDA shows current spot prices

**Fix**: This is normal. Forex spot continues trading 24/5, futures close at specific times. The difference shows the gap between Friday close and current spot.

## Summary

The price caching system provides:
- **Better user experience** - See last known prices when market is closed
- **OANDA parity** - Matches OANDA's behavior of showing last prices
- **Clear status indicators** - Know when you're looking at live vs cached data
- **Graceful error handling** - Fallback to cached prices on errors
- **Original timestamps** - Know exactly when cached prices were valid

This allows you to:
- Test and monitor the system even when market is closed
- Compare OANDA vs NinjaTrader pricing at any time
- Prepare for trading before market opens
- Analyze results after market closes

**Remember**: Only use `OK` (live) prices for actual trading decisions!
