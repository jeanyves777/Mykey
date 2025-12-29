# Forex Trade Logging System

## Overview

Complete trade and market data logging system for evaluating strategy performance after 1 month of live trading.

## What Gets Logged

### 1. Trade Entries (`trades_YYYYMMDD.jsonl`)
Every trade entry logs:
- **Trade Details**: Instrument, direction, price, units, position size
- **Risk Management**: Stop loss, take profit, trailing stop distance
- **Signal Analysis**: Momentum score, HTF trend, RSI, pullback data, confidence
- **Risk Metrics**: SL pips, TP pips, R:R ratio, position size %
- **Account State**: Balance at entry

### 2. Trade Exits (`trades_YYYYMMDD.jsonl`)
Every trade exit logs:
- **Exit Details**: Exit price, exit reason (TP/SL/TRAIL), timestamp
- **Performance**: P&L ($), P&L (%), pips gained/lost
- **Duration**: How long trade was open (minutes)
- **Account State**: Balance after exit

### 3. Skipped Signals (`market_data_YYYYMMDD.jsonl`)
Every rejected signal logs:
- **Why skipped**: No momentum, HTF not aligned, waiting for pullback, etc.
- **Analysis data**: What the indicators showed
- **Timestamp**: When signal was evaluated

### 4. Daily Summary (`daily_summary_YYYYMMDD.json`)
End-of-day summary with:
- Total trades, winners, losers, win rate
- Total P&L, avg win, avg loss, profit factor
- Per-instrument breakdown
- Per-direction breakdown (long vs short)
- Exit reasons breakdown
- Signals skipped count

### 5. CSV Export (`trades_YYYYMMDD.csv`)
Excel-ready format with all trades and full analysis for easy spreadsheet analysis.

## Log Files Location

All logs saved to: `forex_logs/`

**File naming:**
- `trades_20250112.jsonl` - All entry/exit data for Jan 12, 2025
- `market_data_20250112.jsonl` - Skipped signals and market snapshots
- `daily_summary_20250112.json` - End-of-day statistics
- `trades_20250112.csv` - Excel export
- `monthly_summary_202501.json` - Full month aggregated stats

## How to Use

### During Trading

The logging happens **automatically** while running:
```bash
py run_forex_multi_symbol_live.py
```

**What you'll see:**
```
[LOGGER] Initialized trade logger
[LOGGER] Trade log: forex_logs/trades_20250112.jsonl
[LOGGER] Market log: forex_logs/market_data_20250112.jsonl

[LOGGER] Logged ENTRY: EUR_USD BUY @ 1.16500
[LOGGER] Logged EXIT: EUR_USD TP P&L: $+12.50 (+1.2%)
```

### End of Day

Automatically generates:
- Daily summary JSON
- CSV export for Excel
- Shows monthly stats if available

### After 1 Month

Run the analysis script:
```bash
py analyze_monthly_performance.py
```

**Output includes:**
```
MONTHLY FOREX TRADING PERFORMANCE ANALYSIS
==========================================

Total Trades: 127
Win Rate: 54%
Total P&L: $+347.82
Profit Factor: 1.67

PER-INSTRUMENT BREAKDOWN
EUR_USD: 24 trades | 58% WR | $+87.34
GBP_USD: 18 trades | 50% WR | $+42.11
USD_CAD: 15 trades | 67% WR | $+123.45

DIRECTIONAL ANALYSIS
EUR_USD:
  LONG:  15 trades | 73% WR | $+95.23
  SHORT: 9 trades  | 33% WR | $-7.89
  --> LONG BIAS (73% WR)

RECOMMENDATIONS
1. PAIR SELECTION:
   [KEEP] EUR_USD: 58% WR | $+87.34
   [KEEP] USD_CAD: 67% WR | $+123.45
   [REVIEW] GBP_USD: 50% WR | $+42.11

2. CONSIDER DIRECTIONAL FILTERS:
   EUR_USD: Trade LONG only (73% vs 33%)
   USD_CAD: Trade SHORT only (80% vs 45%)

3. OVERALL VERDICT:
   [GOOD] Strategy is profitable but could be improved.
```

## What to Look For After 1 Month

### 1. **Win Rate by Pair**
- Keep pairs with 45%+ win rate
- Remove pairs with <40% win rate

### 2. **Directional Bias**
- If LONG has 60%+ win rate, trade LONG only
- If SHORT has 60%+ win rate, trade SHORT only

### 3. **Exit Analysis**
- How many TP vs SL?
- If mostly SL, targets too aggressive
- If trailing stops working, keep them

### 4. **Profit Factor**
- 1.5+ = Excellent
- 1.2-1.5 = Good
- 1.0-1.2 = Needs improvement
- <1.0 = Stop trading, revise strategy

### 5. **Duration**
- Are trades lasting too long?
- Quick exits may indicate poor entries

## Monthly Stats Tracking

The logger automatically tracks:
- **Consistency**: Are you profitable every week?
- **Drawdown**: Largest peak-to-trough decline
- **Recovery**: How fast do you recover from losses?
- **Best pairs**: Which instruments are most profitable?
- **Best sessions**: London open, NY open, Asian session?

## Data-Driven Decisions

After 1 month you can:

1. **Remove underperforming pairs** based on win rate
2. **Add directional filters** (LONG only or SHORT only) based on bias
3. **Adjust TP/SL** based on avg win/loss
4. **Change trading hours** based on session performance
5. **Increase position size** on profitable pairs
6. **Decrease position size** on marginal pairs

## Important Notes

- **3 days is NOT enough** - need minimum 1 month (20+ trading days)
- **Minimum 50 trades** for statistical significance
- **Track monthly** not weekly - forex has weekly cycles
- **Review every month** and make small adjustments
- **Don't overtrade** - quality > quantity

## Files to Keep

Keep ALL log files for historical reference:
```
forex_logs/
  trades_20250101.jsonl
  trades_20250102.jsonl
  ...
  trades_20250131.jsonl
  monthly_summary_202501.json
  monthly_summary_202502.json
  ...
```

This allows you to:
- Compare month-to-month performance
- See if strategy degrades over time
- Identify seasonal patterns
- Track your improvement as a trader

## Current Strategy Status

**Current Configuration:**
- 7 pairs: EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD
- Pullback detection: ENABLED
- HTF strict mode: ENABLED
- Max 10 trades/day per symbol
- 5% position size (15% leverage cap)

**After 1 month, evaluate:**
- Which pairs to keep trading
- Whether to add directional filters
- If pullback detection helps or hurts
- Whether to relax HTF requirements
- Position sizing adjustments

---

**Start date**: Track from first live trade
**Review date**: After 1 month (minimum 20 trading days)
**Next action**: Keep trading, gather data, review monthly
