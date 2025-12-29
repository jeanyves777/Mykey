# Risk Management - OANDA → NinjaTrader Bridge

## Overview

This document details ALL risk management parameters for the FundedNext challenge trading system.

## ✅ PROVEN STRATEGY

**Using: ForexScalpingStrategy (COMBINED V2)**

**Backtest Results (1 month, 7 pairs):**
- ✅ **Return: +25.82%**
- ✅ **Win Rate: 51.9%**
- ✅ **Profit Factor: 1.34**
- ✅ **Max Drawdown: 5.0%**
- ✅ **Total Trades: 291**

**Entry Logic:** Any 2 of 3 signals agree:
1. RSI Oversold/Overbought (<30 / >70)
2. Bollinger Bands (price touches bands)
3. Range Scalping (price near support/resistance + RSI+Stoch confirmation)

**Key Features:**
- Pair-specific TP/SL based on volatility (12-25 pips)
- 15% margin per position with 20:1 leverage
- No pullback requirement (tested, proven better)
- Max 5 concurrent positions across all pairs

---

## Account-Level Rules (FundedNext Challenge)

### 1. Initial Balance
- **Amount**: $25,000
- **Type**: Account starting balance

### 2. EOD Balance Trailing Stop
- **Max Loss**: $1,000 from highest EOD (End of Day) balance
- **Current Threshold**: $24,000 (account fails if balance drops to this level)
- **Type**: CUMULATIVE (not daily - this is your total max loss)
- **Example**:
  - Day 1 EOD: $25,500 → New threshold: $24,500
  - Day 2 EOD: $26,000 → New threshold: $25,000
  - If balance ever drops to threshold → ACCOUNT FAILS

### 3. Daily Loss Limit
- **Amount**: -$500 per day
- **Type**: DAILY (resets each day)
- **Action**: Stop trading for the day if hit

### 4. Profit Target
- **Amount**: +$1,250 total
- **Type**: CUMULATIVE (challenge passes when reached)

### 5. Daily Profit Cap (OPTIONAL - Consistency Mode Only)
- **Amount**: $400 per day
- **Rule**: Can't make more than 40% of total profit in one day
- **Enabled**: Only with `--consistency` flag
- **Action**: Stop trading for the day if hit

### 6. Buffer Protection
- **Amount**: $200 minimum buffer above threshold
- **Formula**: `current_balance - threshold >= $200`
- **Action**: Stop trading if buffer drops below $200

---

## Position-Level Rules

### 1. Position Size
- **Contracts per Trade**: 1 contract ALWAYS
- **Max Concurrent Positions**: 5 (one per symbol max)
- **Type**: FIXED (never changes)

### 2. Symbol Mapping (OANDA → NinjaTrader Futures)

| OANDA Symbol | NinjaTrader Symbol | Contract Name | Exchange |
|--------------|-------------------|---------------|----------|
| EUR_USD | M6E | Micro EUR/USD | CME |
| GBP_USD | M6B | Micro GBP/USD | CME |
| USD_JPY | MJY | Micro USD/JPY | CME |
| USD_CAD | MCD | Micro USD/CAD | CME |
| USD_CHF | MSF | Micro USD/CHF | CME |

**IMPORTANT**: These are MICRO E-mini FX futures (not standard E-mini). Micro contracts are 1/10th the size.

---

## Take Profit & Stop Loss Settings

### M6E (Micro EUR/USD)
- **Take Profit**: 20 pips = 40 ticks
- **Stop Loss**: 16 pips = 32 ticks
- **Tick Size**: 0.00005 (0.5 pips)
- **Tick Value**: $6.25 per tick
- **Risk:Reward Ratio**: 1:1.25
- **Max Loss per Trade**: 32 ticks × $6.25 = **$200**
- **Max Profit per Trade**: 40 ticks × $6.25 = **$250**

### M6B (Micro GBP/USD)
- **Take Profit**: 30 pips = 30 ticks
- **Stop Loss**: 25 pips = 25 ticks
- **Tick Size**: 0.0001 (1 pip)
- **Tick Value**: $6.25 per tick
- **Risk:Reward Ratio**: 1:1.2
- **Max Loss per Trade**: 25 ticks × $6.25 = **$156.25**
- **Max Profit per Trade**: 30 ticks × $6.25 = **$187.50**

### MJY (Micro USD/JPY)
- **Take Profit**: 18 pips = 180 ticks
- **Stop Loss**: 15 pips = 150 ticks
- **Tick Size**: 0.000001 (0.01 pips)
- **Tick Value**: $1.25 per tick
- **Risk:Reward Ratio**: 1:1.2
- **Max Loss per Trade**: 150 ticks × $1.25 = **$187.50**
- **Max Profit per Trade**: 180 ticks × $1.25 = **$225**

### MCD (Micro USD/CAD)
- **Take Profit**: 20 pips = 40 ticks
- **Stop Loss**: 16 pips = 32 ticks
- **Tick Size**: 0.00005 (0.5 pips)
- **Tick Value**: $5.00 per tick
- **Risk:Reward Ratio**: 1:1.25
- **Max Loss per Trade**: 32 ticks × $5.00 = **$160**
- **Max Profit per Trade**: 40 ticks × $5.00 = **$200**

### MSF (Micro USD/CHF)
- **Take Profit**: 15 pips = 30 ticks
- **Stop Loss**: 12 pips = 24 ticks
- **Tick Size**: 0.00005 (0.5 pips)
- **Tick Value**: $6.25 per tick
- **Risk:Reward Ratio**: 1:1.25
- **Max Loss per Trade**: 24 ticks × $6.25 = **$150**
- **Max Profit per Trade**: 30 ticks × $6.25 = **$187.50**

---

## Trade Limits

### 1. Daily Trade Limits
- **Max Trades per Day**: 50 (across all symbols)
- **Max Trades per Symbol**: 10 per day
- **Type**: DAILY (resets each day)

### 2. Position Limits
- **Max Concurrent Positions**: 5 (one per symbol)
- **Contracts per Position**: 1 (always)

---

## Trailing Stop

**Currently**: NO trailing stop implemented

The system uses:
- **Fixed Stop Loss**: Set at entry, sent to NinjaTrader
- **Fixed Take Profit**: Set at entry, sent to NinjaTrader

**To Add Trailing Stop** (if needed):
- Would require NinjaTrader bridge to support trailing stop orders
- Would need to specify trailing distance in ticks
- Would need to handle partial profits

---

## Risk Per Trade Analysis

| Symbol | Max Loss | Max Profit | R:R Ratio | % of Account (Risk) | % of Account (Profit) |
|--------|----------|------------|-----------|---------------------|----------------------|
| M6E | $200 | $250 | 1:1.25 | 0.80% | 1.00% |
| M6B | $156.25 | $187.50 | 1:1.20 | 0.63% | 0.75% |
| MJY | $187.50 | $225 | 1:1.20 | 0.75% | 0.90% |
| MCD | $160 | $200 | 1:1.25 | 0.64% | 0.80% |
| MSF | $150 | $187.50 | 1:1.25 | 0.60% | 0.75% |

**Average Risk per Trade**: ~0.68% of $25,000 account

**Maximum Risk** (5 concurrent positions, all hit SL):
- Worst case: $200 + $156.25 + $187.50 + $160 + $150 = **$853.75**
- Percentage: 3.42% of account
- Within daily loss limit: YES (less than $500? NO - exceeds by $353.75)

⚠️ **WARNING**: If all 5 positions hit stop loss simultaneously, you would exceed the daily loss limit!

---

## Safety Recommendations

### 1. Reduce Concurrent Positions
- Current: 5 max concurrent
- Recommended: 3 max concurrent (to stay under $500 daily loss limit)
- Max loss with 3 positions: ~$550 (still close to limit)

### 2. Tighter Stop Losses
- Consider reducing SL by 20-25% to ensure 5 positions can't exceed $500 daily loss
- Example: M6E SL from 32 ticks → 25 ticks ($156.25 max loss)

### 3. Dynamic Position Sizing Based on Daily P&L
- If daily P&L is -$300, reduce max concurrent to 2 positions
- If daily P&L is -$400, stop all new trades

---

## Status Display Explanation

When you see:
```
Balance: $25,000.00, Threshold: $24,000.00
```

This means:
- **Balance**: Current account balance
- **Threshold**: Account failure point (EOD trailing stop)
- **Buffer**: $1,000 above threshold
- **Daily P&L**: Calculated from today's starting balance (not shown here, but tracked)

**To see daily P&L**, the log should show:
```
Balance: $25,200.00, Threshold: $24,000.00, Daily P&L: +$200.00
```

---

## FundedNext Rules Summary

| Rule | Value | Type | Action if Hit |
|------|-------|------|--------------|
| Initial Balance | $25,000 | Static | N/A |
| Max Loss (EOD Trailing) | $1,000 | Cumulative | Account fails |
| Daily Loss Limit | -$500 | Daily | Stop trading for day |
| Profit Target | +$1,250 | Cumulative | Challenge passed! |
| Buffer Protection | $200 | Real-time | Stop trading |
| Consistency (optional) | 40%/day, $400 cap | Daily | Stop trading for day |
| Max Concurrent | 5 positions | Real-time | No new trades |
| Max Trades/Day | 50 | Daily | No new trades |
| Max Trades/Symbol | 10 | Daily | No trades for symbol |

---

## Logging Status Display

The system should show:
```
[14:32:15] Loop 5 - Checking market...
  Market Status: Market open
  Balance: $25,200.00
  Threshold: $24,000.00 (EOD trailing stop)
  Buffer: $1,200.00
  Daily P&L: +$200.00 (Daily loss limit: -$500)
  Trades today: 3/50
  Open positions: 2/5
```

**Currently Missing**: Daily P&L display in status logs

---

## Questions to Confirm

1. **Should we reduce max concurrent positions from 5 to 3?** (to ensure we can't exceed $500 daily loss)
2. **Should we add trailing stop functionality?** (requires NinjaTrader bridge changes)
3. **Should we add daily P&L to the status display?** (recommended)
4. **Are the TP/SL settings correct for your risk tolerance?** (currently risking ~0.6-0.8% per trade)
5. **Do you want dynamic position sizing based on daily P&L?** (e.g., reduce positions if losing day)

---

## Next Steps

Let me know if you want me to:
1. ✅ Add daily P&L to status display
2. ✅ Reduce max concurrent positions to 3 (safer for $500 daily loss limit)
3. ✅ Add trailing stop functionality
4. ✅ Add dynamic position sizing
5. ✅ Adjust TP/SL values

**All of these are quick changes!**
