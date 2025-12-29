# How to Run OANDA → NinjaTrader Live Trading

## Quick Start

### Prerequisites (Do these FIRST)
1. **Start NinjaTrader 8**
2. **Connect to your account** (Sim101 for testing, FundedNext for live)
3. **Start the bridge**: `NinjaTraderBridge.exe`
4. **Then run the Python script** (see modes below)

---

## Running Modes

### Mode 1: SIMPLE MODE (Default - Recommended)
**No consistency rule, just -$500 daily loss limit**

```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py
```

**Runs continuously** - Press Ctrl+C to stop when needed. All FundedNext safeguards are active, no need to specify duration!

**Rules enforced:**
- ✅ EOD Balance Trailing ($1,000 max loss)
- ✅ Buffer Protection ($200 minimum)
- ✅ Daily Loss Limit (-$500)
- ✅ Profit Target (+$1,250)
- ❌ NO consistency rule
- ✅ Position Limits (1 contract, max 5 concurrent)
- ✅ Trade Limits (50/day, 10/symbol)

**Use this for:**
- Testing on Sim101
- Simple trading without consistency restrictions
- When you want maximum daily profit potential

---

### Mode 2: CONSISTENCY MODE (FundedNext Challenge)
**Enables 40% consistency rule + $400 daily cap**

```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py --consistency
```

**Runs continuously** with consistency rule enforced. Use this if FundedNext explicitly requires the consistency rule.

**Rules enforced:**
- ✅ EOD Balance Trailing ($1,000 max loss)
- ✅ Buffer Protection ($200 minimum)
- ✅ Daily Loss Limit (-$500)
- ✅ Profit Target (+$1,250)
- ✅ Consistency Rule (40% max per day, $400 cap)
- ✅ Position Limits (1 contract, max 5 concurrent)
- ✅ Trade Limits (50/day, 10/symbol)

**Use this for:**
- FundedNext challenge (if they require consistency rule)
- When you want to enforce steady, consistent gains

---

### Mode 3: FUNDED MODE (After Challenge Passes)
**For funded accounts (no consistency rule)**

```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py --funded
```

**Runs continuously** for funded account trading.

**Rules enforced:**
- ✅ EOD Balance Trailing ($1,000 max loss)
- ✅ Buffer Protection ($200 minimum)
- ✅ Daily Loss Limit (-$500)
- ✅ Profit Target (+$1,250)
- ❌ NO consistency rule
- ✅ Position Limits (1 contract, max 5 concurrent)
- ✅ Trade Limits (50/day, 10/symbol)

**Use this for:**
- After you pass the FundedNext challenge
- Funded account trading

---

## Command-Line Options

| Flag | Description |
|------|-------------|
| *(no flags)* | Run continuously (default - recommended) |
| `--hours 8` | Run for specific hours (optional, use if needed) |
| `--consistency` | Enable consistency rule (40%/day, $400 cap) |
| `--funded` | Funded account mode (no consistency) |
| `--yes` or `-y` | Skip confirmation prompt |

---

## Examples

### Start trading (simple mode, continuous)
```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py
```

### Start with consistency rule (FundedNext Challenge)
```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py --consistency -y
```

### Funded account (continuous, no confirmation)
```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py --funded -y
```

### Run for specific hours (if needed)
```powershell
python trading_system\NinjaTrader_Bridge\run_oanda_ninjatrader_live.py --hours 4
```

---

## What You'll See

### When starting:
```
================================================================================
OANDA → NINJATRADER LIVE TRADING BRIDGE
================================================================================

⚠️  IMPORTANT: This will execute REAL trades on NinjaTrader!

Prerequisites:
1. NinjaTrader 8 is running
2. Connected to your account (Sim101 for testing, FundedNext for challenge)
3. NinjaTraderBridge.exe is running

Duration: Continuous (Ctrl+C to stop)
Mode: CHALLENGE
Consistency Rule: DISABLED

================================================================================

Continue? (y/n):
```

### During trading:

**Every 5 minutes, you'll see a detailed status update:**
```
[14:32:15] Loop 5 - Checking market...
  Market Status: Market open
  Balance: $25,200.00
  Threshold: $24,000.00 (EOD trailing stop)
  Buffer: $1,200.00 above threshold
  Daily P&L: +$200.00 (Daily loss limit: $-500)
  Total Profit: +$200.00 (Target: $1,250)
  Trades today: 3/50
  Open positions: 2/5

  Market Prices - OANDA vs NinjaTrader Comparison:
  Symbol Source       Bid          Ask          Mid          Diff (pips)
  ------------------------------------------------------------------------
  M6E    OANDA        1.05123      1.05145      1.05134
         NinjaTrader  1.05118      1.05143      1.05131      0.6
  ------------------------------------------------------------------------
  M6B    OANDA        1.27456      1.27489      1.27473
         NinjaTrader  1.27451      1.27484      1.27468      0.5
  ------------------------------------------------------------------------
  MJY    OANDA        149.82300    149.85100    149.83700
         NinjaTrader  149.82100    149.84900    149.83500    2.0
  ------------------------------------------------------------------------
  MSF    OANDA        0.89234      0.89267      0.89251
         NinjaTrader  0.89229      0.89262      0.89246      1.0
  ------------------------------------------------------------------------
  MCD    OANDA        1.36789      1.36812      1.36801
         NinjaTrader  1.36784      1.36807      1.36796      1.0
  ------------------------------------------------------------------------

  [EUR_USD] No trade: Weak momentum
  [GBP_USD] No trade: No clear trend
  [USD_JPY] SIGNAL FOUND: BUY - Multi-timeframe momentum alignment
  [USD_CHF] SKIP - Already have open position
  [USD_CAD] No trade: Choppy price action

[SUMMARY] Loop 5 complete:
  Symbols checked: 5
  Signals found: 1
  Signals skipped: 4
  Common skip reasons:
    - No signal: 3
    - Position exists: 1
```

**What each value means:**
- **Balance**: Current account balance
- **Threshold**: Account failure point (if balance drops to this, account fails)
- **Buffer**: How much room you have above the threshold
- **Daily P&L**: Today's profit/loss (resets each day, limit: -$500)
- **Total Profit**: Cumulative profit since start (target: +$1,250 to pass challenge)
- **Trades today**: Number of trades executed today / daily limit
- **Open positions**: Currently open positions / max concurrent allowed
- **Market Prices**:
  - **OANDA**: Forex spot prices (used for signal generation)
  - **NinjaTrader**: Actual futures prices (used for trade execution)
  - **Diff (pips)**: Price difference between OANDA and NinjaTrader (helps detect data issues)
```

**When a trade is executed:**
```
[SIGNAL] M6E BUY @ 1.05131 (NinjaTrader)
  TP: 1.05331, SL: 1.04971
  Price difference (NT vs OANDA): 0.6 pips
  Reason: Multi-timeframe momentum
  ✓ Signal sent to NinjaTrader
  Trade ID: M6E_20250115_143215
  Trades today: 1/50
```

**Important Notes:**
- Entry price is the ACTUAL NinjaTrader futures price (not OANDA proxy)
- TP/SL calculated using NinjaTrader price for accuracy
- Price difference shows how much NinjaTrader differs from OANDA (typical: 0.5-2 pips)

### When consistency rule triggers (if enabled):
```
[STOP] Consistency rule: Today $395.00 >= 40% of total $980.00
Waiting 60 seconds...
```

### When daily profit cap hit (if consistency enabled):
```
[STOP] Daily profit cap ($400.00 >= $400)
Waiting 60 seconds...
```

---

## Stopping the Script

- Press **Ctrl+C** to stop at any time
- **You'll be asked about open positions:**

```
================================================================================
You have 2 open positions:
  - M6E BUY @ 1.05000
  - M6B SELL @ 1.28500
================================================================================

Close all positions? (y/n):
```

**Choose:**
- **y** = Close all positions now (via NinjaTrader)
- **n** = Leave positions open (they'll be managed automatically on next run)

**The script will show session summary:**

```
================================================================================
SESSION COMPLETE
================================================================================
Final Balance: $25,350.00
Total Profit: +$350.00
Trades Today: 12
Open Positions: 0 (or number if left open)
================================================================================
```

---

## Testing the Dual Pricing System

Before running live trading, test that both OANDA and NinjaTrader prices are working:

```powershell
python test_dual_pricing_system.py
```

**This test will:**
1. Connect to OANDA API
2. Connect to NinjaTrader bridge
3. Fetch prices from both sources
4. Display side-by-side comparison
5. Calculate price differences in pips
6. Provide interpretation and recommendations

**Expected output:**
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
Symbol Source       Bid          Ask          Mid          Diff (pips)
------------------------------------------------------------------------
M6E    OANDA        1.05123      1.05145      1.05134
       NinjaTrader  1.05118      1.05143      1.05131      0.6
------------------------------------------------------------------------

ANALYSIS
================================================================================
  Average price difference: 1.2 pips
  Maximum difference: 2.0 pips (MJY)
  Minimum difference: 0.5 pips (M6B)

INTERPRETATION:
  ✓ Good - Price differences are acceptable (2-5 pips)

RECOMMENDATION:
  Use NinjaTrader prices for trade execution (TP/SL calculation)
  Use OANDA for signal generation (strategy logic)
  Monitor price differences to detect data issues
```

**What to look for:**
- Price differences < 5 pips = Good (typical for forex vs futures)
- Price differences > 10 pips = Warning (check data connection)
- Both sources showing prices = System ready
- Errors connecting = Fix NinjaTrader or OANDA connection

---

## Testing Checklist

Before running on FundedNext:

- [ ] **Test dual pricing system** (`python test_dual_pricing_system.py`)
- [ ] Verify price differences are reasonable (< 5 pips average)
- [ ] Test on Sim101 in simple mode (let it run Mon-Fri)
- [ ] Test with consistency rule enabled (if required)
- [ ] Verify all signals execute correctly in NinjaTrader
- [ ] Check TP/SL orders are placed automatically
- [ ] Confirm position sizing is always 1 contract
- [ ] Validate daily profit tracking works
- [ ] **Confirm trade logs show NinjaTrader prices (not OANDA)**
- [ ] Test that consistency rule stops trading when $400 hit (if enabled)
- [ ] Ensure buffer protection prevents overtrading
- [ ] Verify it runs continuously without issues

---

## Troubleshooting

### "Cannot connect to NinjaTrader bridge"
- Make sure NinjaTraderBridge.exe is running
- Check that bridge shows "Status: ACTIVE"

### "No trades happening"
- Check OANDA API credentials in .env
- Verify markets are open (forex hours)
- Check if FundedNext rules blocking (buffer, limits, etc.)
- Review strategy signals in console

### "Orders not showing in NinjaTrader"
- Check NinjaTrader is connected (not disconnected)
- Verify bridge window shows order confirmations
- Check Control Panel → Orders tab

---

## Resuming After Stop

When you restart the script, it **automatically resumes** from where you left off:

```
[START] Starting live trading...
[START] Duration: Continuous (until stopped or market closes)
[STATE] Loaded from trading_system/NinjaTrader_Bridge/bridge_state.json
[STATE] Balance: $25,350.00
[STATE] Open Positions: 2
[STATE]   - M6E BUY @ 1.05000
[STATE]   - M6B SELL @ 1.28500

================================================================================
RESUMING FROM PREVIOUS SESSION
================================================================================
Found 2 open positions
These positions will be managed automatically
================================================================================
```

**The script will:**
- Continue monitoring your open positions
- Respect TP/SL levels already set in NinjaTrader
- Track profit/loss from previous session
- Enforce FundedNext rules based on current state

**State file:** `trading_system/NinjaTrader_Bridge/bridge_state.json`
- Saves automatically on every position change
- Saves when you stop the script
- Can be deleted to start fresh

---

## Trade Analytics

Every trade is logged with full details for analysis:

**Trade Logs Location:**
- `trading_system/NinjaTrader_Bridge/trade_logs/trades.json` - All trades (easy to read)
- `trading_system/NinjaTrader_Bridge/trade_logs/trades.csv` - CSV format (Excel compatible)

**What's Logged:**
- Entry/exit timestamps and prices
- TP/SL levels
- Exit reason (TP, SL, MANUAL, RULE_VIOLATION)
- P&L in ticks and USD
- Symbol details (tick size, tick value)
- Account balance before/after
- Daily profit tracking
- FundedNext rule compliance data

**View Analytics:**
```powershell
python trading_system\NinjaTrader_Bridge\view_trade_analytics.py
```

**This shows:**
- Overall P&L statistics
- Win rate and profit factor
- Per-symbol performance
- Exit reason analysis
- Daily performance
- Last 10 trades
- Current open positions

**Export to Excel:**
```python
from trading_system.analytics.forex_trade_logger import ForexTradeLogger
logger = ForexTradeLogger()
logger.export_to_excel()  # Creates trades.xlsx with multiple sheets
```

---

## Important Notes

1. **Runs continuously by default** - All FundedNext safeguards will stop trading automatically when needed
2. **Position persistence** - Open positions are saved and resumed automatically on restart
3. **Always test on Sim101 first** before running on FundedNext
4. **The consistency rule is OPTIONAL** - only enable if FundedNext requires it
5. **Simple mode is recommended** for most trading (no artificial profit caps)
6. **Bridge must be running** before starting Python script
7. **Check .env file** has correct OANDA API credentials
8. **Press Ctrl+C to stop** anytime - you'll be asked about open positions

---

**Questions? Issues? Check the bridge window and NinjaTrader logs for details!**
