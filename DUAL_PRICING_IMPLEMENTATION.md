# Dual Pricing System Implementation

## Overview

The OANDA → NinjaTrader live trading bridge now uses a **dual pricing system** that fetches prices from both OANDA (forex spot) and NinjaTrader (futures) to ensure accurate trade execution.

## The Problem

Previously, the system used OANDA forex spot prices as a proxy for NinjaTrader futures prices. However:

- **OANDA**: Spot forex market (OTC, decentralized)
- **NinjaTrader**: Micro E-mini FX futures (CME, exchange-traded)
- **Price Difference**: Typically 1-5 pips due to:
  - Basis risk (futures vs spot)
  - Funding costs embedded in futures
  - Different liquidity and market participants
  - Exchange-traded (CME) vs OTC pricing

This meant trade logs showed OANDA prices, but actual execution happened at different NinjaTrader futures prices, leading to TP/SL calculation errors.

## The Solution

### 1. NinjaTrader Bridge Price Query

**Updated: [NinjaTraderBridge.cs](trading_system/NinjaTrader_Bridge/NinjaTraderBridge.cs)**

Added new `PRICE_QUERY` action to the C# bridge:

```csharp
// New price data structure
public class PriceData
{
    public string Symbol { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
    public double Last { get; set; }
    public string Timestamp { get; set; }
    public string Status { get; set; }
}

// New method to fetch NinjaTrader market prices
static PriceData GetMarketPrice(string symbol)
{
    string ntSymbol = symbolMap[symbol];
    string marketData = ntClient.MarketData(ntSymbol);
    // Parse and return bid, ask, last prices
}
```

The bridge now handles three actions:
- `ENTRY`: Place new position
- `EXIT`: Close position
- `PRICE_QUERY`: Query current market price

### 2. Python Price Query Methods

**Updated: [run_oanda_ninjatrader_live.py](trading_system/NinjaTrader_Bridge/run_oanda_ninjatrader_live.py)**

Added three new methods:

```python
def get_current_prices(self) -> Dict:
    """Fetch OANDA market prices (for comparison)"""
    # Returns OANDA spot forex prices

def get_ninjatrader_price(self, nt_symbol: str) -> Optional[Dict]:
    """Query actual NinjaTrader futures price for a specific symbol"""
    # Sends PRICE_QUERY to bridge
    # Returns NinjaTrader bid, ask, last prices

def get_ninjatrader_prices(self) -> Dict:
    """Fetch actual NinjaTrader futures prices for all symbols"""
    # Queries all 5 NinjaTrader futures symbols
```

### 3. Trade Validation with NinjaTrader Prices

**Key Change**: Trades now use **actual NinjaTrader prices** for entry and TP/SL calculation:

```python
# OLD (incorrect):
price_data = self.oanda_client.get_current_price(oanda_symbol)
entry_price = price_data['ask']  # OANDA price
sl_price, tp_price = self.calculate_sl_tp(nt_symbol, 'BUY', entry_price)

# NEW (correct):
nt_price_data = self.get_ninjatrader_price(nt_symbol)  # Query NinjaTrader
entry_price = nt_price_data['ask']  # NinjaTrader actual price
sl_price, tp_price = self.calculate_sl_tp(nt_symbol, 'BUY', entry_price)
```

**Benefits**:
- TP/SL calculated from actual execution price
- No slippage from OANDA → NinjaTrader conversion
- Accurate trade logs

### 4. Dual Price Comparison Display

**Status display now shows both prices side-by-side:**

```
Market Prices - OANDA vs NinjaTrader Comparison:
Symbol Source       Bid          Ask          Mid          Diff (pips)
------------------------------------------------------------------------
M6E    OANDA        1.05123      1.05145      1.05134
       NinjaTrader  1.05118      1.05143      1.05131      0.6
------------------------------------------------------------------------
M6B    OANDA        1.27456      1.27489      1.27473
       NinjaTrader  1.27451      1.27484      1.27468      0.5
------------------------------------------------------------------------
```

**Features**:
- Shows both OANDA and NinjaTrader prices
- Calculates price difference in pips
- Helps detect data connection issues
- Displayed every 5 loops during live trading

### 5. Trade Execution with Price Difference

**When a signal is found:**

```
[SIGNAL] M6E BUY @ 1.05131 (NinjaTrader)
  TP: 1.05331, SL: 1.04971
  Price difference (NT vs OANDA): 0.6 pips
  Reason: Multi-timeframe momentum
  ✓ Signal sent to NinjaTrader
```

**Key Information**:
- Entry price is NinjaTrader actual price
- TP/SL calculated from NinjaTrader price
- Price difference shown for monitoring
- "(NinjaTrader)" label confirms source

### 6. Exit Price Accuracy

**Position exits also use NinjaTrader prices:**

```python
# Query actual NinjaTrader price for exit
nt_price_data = self.get_ninjatrader_price(symbol)

if nt_price_data:
    if pos['side'] == 'BUY':
        exit_price = nt_price_data['bid']  # Exit at NinjaTrader bid
    else:
        exit_price = nt_price_data['ask']  # Cover at NinjaTrader ask
```

## Testing the System

### Test Script: `test_dual_pricing_system.py`

Run this before live trading to validate both price sources:

```powershell
python test_dual_pricing_system.py
```

**What it tests**:
1. OANDA API connection
2. NinjaTrader bridge connection
3. Price query functionality
4. Price comparison for all 5 symbols
5. Price difference calculation
6. Data quality assessment

**Expected Results**:
- Average price difference: 0.5-2 pips (good)
- Average price difference: 2-5 pips (acceptable)
- Average price difference: > 10 pips (warning - check connections)

## Files Modified

### C# Bridge
- `trading_system/NinjaTrader_Bridge/NinjaTraderBridge.cs`
  - Added `PriceData` class
  - Added `PRICE_QUERY` action handler
  - Added `GetMarketPrice()` method

### Python Live Trading
- `trading_system/NinjaTrader_Bridge/run_oanda_ninjatrader_live.py`
  - Added `get_current_prices()` (OANDA)
  - Added `get_ninjatrader_price()` (single symbol)
  - Added `get_ninjatrader_prices()` (all symbols)
  - Updated trade validation to use NinjaTrader prices
  - Updated exit price logging to use NinjaTrader prices
  - Updated status display with dual price comparison

### Documentation
- `trading_system/NinjaTrader_Bridge/HOW_TO_RUN.md`
  - Updated status display examples
  - Updated trade execution examples
  - Added dual pricing test section
  - Updated testing checklist

### Test Scripts
- `test_dual_pricing_system.py` (NEW)
  - Comprehensive dual pricing validation
  - Price difference analysis
  - Connection testing

## Workflow

### Signal Generation (OANDA)
1. Fetch 1min, 5min, 15min, 30min candles from OANDA
2. Run ForexScalpingStrategy on OANDA data
3. Generate BUY/SELL signals based on OANDA data

### Trade Execution (NinjaTrader)
1. **Signal found** → Query NinjaTrader for actual price
2. **Get NinjaTrader bid/ask** → Use for entry price
3. **Calculate TP/SL** → Based on NinjaTrader price
4. **Send trade to NinjaTrader** → Execute at actual futures price
5. **Log trade** → Record NinjaTrader price (not OANDA)

### Position Exit (NinjaTrader)
1. **Exit signal** → Query NinjaTrader for current price
2. **Get NinjaTrader bid/ask** → Use for exit price
3. **Send exit to NinjaTrader** → Close at actual futures price
4. **Log exit** → Record NinjaTrader price

## Benefits

### Before (OANDA Proxy)
- ❌ Used OANDA spot prices as proxy
- ❌ TP/SL calculated from OANDA (1-5 pips off)
- ❌ Trade logs showed OANDA prices
- ❌ Actual execution at different NinjaTrader prices
- ❌ No visibility into price differences

### After (Dual Pricing)
- ✅ Uses actual NinjaTrader futures prices
- ✅ TP/SL calculated from execution price
- ✅ Trade logs show real execution prices
- ✅ Accurate P&L tracking
- ✅ Price difference monitoring
- ✅ Data quality validation

## Price Difference Monitoring

### Typical Differences
- **EUR/USD (M6E)**: 0.5-1.5 pips
- **GBP/USD (M6B)**: 0.5-1.5 pips
- **USD/JPY (MJY)**: 1-3 pips
- **USD/CHF (MSF)**: 0.5-2 pips
- **USD/CAD (MCD)**: 0.5-2 pips

### Warning Signs
- Price difference > 10 pips → Check data connection
- No NinjaTrader prices → Bridge not running or NT disconnected
- No OANDA prices → API issue or market closed
- Wildly fluctuating differences → Data quality issue

## Implementation Notes

### Why Keep OANDA?
- **Signal generation**: OANDA provides historical candle data
- **Strategy logic**: Multi-timeframe analysis uses OANDA data
- **Market scanning**: Check all 5 symbols for signals
- **Comparison**: Validate NinjaTrader data quality

### Why Add NinjaTrader Prices?
- **Trade execution**: Actual futures prices for TP/SL
- **Accurate logging**: Record real execution prices
- **P&L tracking**: Calculate profits from actual fills
- **Risk management**: True position sizing based on futures prices

## Next Steps

1. **Recompile C# Bridge**:
   ```powershell
   # Rebuild NinjaTraderBridge.exe with new price query functionality
   ```

2. **Test Dual Pricing**:
   ```powershell
   python test_dual_pricing_system.py
   ```

3. **Test on Sim101**:
   ```powershell
   # Run live trading in simple mode
   python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py
   ```

4. **Monitor Price Differences**:
   - Watch the dual price display every 5 loops
   - Ensure differences stay < 5 pips
   - Alert if differences spike > 10 pips

5. **Validate Trade Logs**:
   - Check that logged prices match NinjaTrader execution
   - Verify TP/SL distances are correct
   - Confirm P&L calculations are accurate

## Troubleshooting

### Cannot fetch NinjaTrader prices
- **Cause**: Bridge not running or NinjaTrader disconnected
- **Fix**: Start NinjaTraderBridge.exe, connect to account in NT

### High price differences (> 10 pips)
- **Cause**: Data feed issue, market volatility, or connection lag
- **Fix**: Restart bridge, check NT data feed, verify internet connection

### Trade logs still show OANDA prices
- **Cause**: Old code running, bridge not updated
- **Fix**: Restart Python script, ensure latest version running

### Price query timeout
- **Cause**: Bridge slow to respond, NT overloaded
- **Fix**: Increase timeout in `get_ninjatrader_price()`, restart bridge

## Summary

The dual pricing system ensures that:
- **OANDA** provides signal generation (strategy logic)
- **NinjaTrader** provides execution prices (actual fills)
- **Both** are displayed for comparison and validation
- **Trade logs** reflect actual execution prices
- **TP/SL** are calculated from real futures prices

This eliminates the 1-5 pip discrepancy that was causing inaccurate trade logging and TP/SL calculations.
