# OPTIMIZED FOREX TRADING SYSTEM V2
## Comprehensive Documentation

**Version:** 2.0
**Optimization Date:** December 18, 2025
**Total Backtest Results:** +11,070 pips across 7 pairs
**Overall Win Rate:** 86%+
**Walk-Forward Validated:** 7/7 pairs

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Performance Summary](#2-performance-summary)
3. [File Structure & Dependencies](#3-file-structure--dependencies)
4. [Configuration Files](#4-configuration-files)
5. [Strategy Logic](#5-strategy-logic)
6. [Trading Engine](#6-trading-engine)
7. [OANDA API Client](#7-oanda-api-client)
8. [Trade Logger](#8-trade-logger)
9. [Session-Based Loss Cooldown](#9-session-based-loss-cooldown)
10. [How to Run](#10-how-to-run)
11. [Environment Setup](#11-environment-setup)
12. [Risk Management](#12-risk-management)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Overview

The Optimized Forex Trading System V2 is an automated trading system designed for the OANDA forex platform. It uses technical analysis strategies optimized through extensive backtesting on 100,000+ candles per pair.

### Key Features

- **Multi-Pair Trading:** Trades 7 optimized currency pairs simultaneously
- **Session Filtering:** Trades only during optimal hours for each pair
- **Loss Protection:** Session-based cooldown after losses
- **Friday Close Protection:** Auto-closes positions before weekend
- **Comprehensive Logging:** Full trade history and session statistics
- **Paper & Live Modes:** Separate configurations for practice and real trading

### Architecture

```
User -> run_optimized_forex_paper.py / run_optimized_forex_live.py
           |
           v
    [Config] optimized_paper_config.py / optimized_live_config.py
           |
           v
    [Strategy] optimized_strategy.py
           |
           v
    [Engine] oanda_client.py
           |
           v
    [Logger] trade_logger.py
           |
           v
      OANDA API
```

---

## 2. Performance Summary

### V2 Optimization Results (100K Candles Backtest)

| Pair     | Strategy      | TP   | SL   | Session | Win Rate | Pips    |
|----------|---------------|------|------|---------|----------|---------|
| EUR_USD  | RSI_REVERSAL  | 8    | 20   | ASIAN   | 90.0%    | +1,313  |
| GBP_USD  | RSI_REVERSAL  | 8    | 20   | ASIAN   | 86.0%    | +1,021  |
| USD_CHF  | MACD_CROSS    | 8    | 20   | ASIAN   | 92.7%    | +1,015  |
| USD_CAD  | RSI_REVERSAL  | 8    | 20   | ASIAN   | 96.4%    | +1,279  |
| NZD_USD  | RSI_REVERSAL  | 8    | 20   | ALL     | 94.0%    | +3,429  |
| AUD_CHF  | RSI_REVERSAL  | 8    | 20   | ALL     | 84.4%    | +1,004  |
| EUR_GBP  | RSI_REVERSAL  | 8    | 20   | ALL     | 92.5%    | +2,009  |
| **TOTAL**|               |      |      |         | **86%**  | **+11,070** |

### Key Settings

- **Take Profit:** 8 pips
- **Stop Loss:** 20 pips
- **Risk:Reward:** 1:2.5
- **Position Size:** $1 per pip (10,000 units for major pairs)

---

## 3. File Structure & Dependencies

### Core Files Required

```
thevolumeainative/
├── run_optimized_forex_paper.py      # Paper trading runner
├── run_optimized_forex_live.py       # Live trading runner
├── .env                              # OANDA credentials
│
└── trading_system/
    └── Forex_Trading/
        ├── config/
        │   ├── optimized_paper_config.py   # Paper trading config
        │   └── optimized_live_config.py    # Live trading config
        │
        ├── strategies/
        │   └── optimized_strategy.py       # Strategy logic
        │
        ├── engine/
        │   └── oanda_client.py             # OANDA API wrapper
        │
        └── utils/
            └── trade_logger.py             # Trade logging system
```

### Python Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
requests>=2.28.0
python-dotenv>=0.21.0
pytz>=2022.7
```

Install with:
```bash
pip install pandas numpy requests python-dotenv pytz
```

---

## 4. Configuration Files

### optimized_paper_config.py / optimized_live_config.py

Both config files share the same structure with different account settings:

```python
# Account Settings
ACCOUNT_TYPE = 'practice'  # or 'live'
LOG_DIR = 'forex_logs'     # or 'forex_logs_live'

# Trading Parameters
MAX_CONCURRENT_POSITIONS = 10
TIMEFRAME = 'M1'           # 1-minute candles
CANDLE_COUNT = 100         # Candles for analysis
COOLDOWN_MINUTES = 0       # No cooldown between trades

# Session Definitions (UTC)
SESSIONS = {
    'ASIAN': list(range(0, 8)),      # 00:00-08:00 UTC
    'LONDON': list(range(8, 16)),    # 08:00-16:00 UTC
    'NEW_YORK': list(range(13, 21)), # 13:00-21:00 UTC
    'ALL': list(range(0, 24)),       # All hours
}

# Trading Sessions per Pair
TRADING_SESSIONS = {
    'EUR_USD': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'GBP_USD': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'USD_CHF': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'USD_CAD': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'NZD_USD': {'allowed_hours': SESSIONS['ALL'], 'session_filter': False},
    'AUD_CHF': {'allowed_hours': SESSIONS['ALL'], 'session_filter': False},
    'EUR_GBP': {'allowed_hours': SESSIONS['ALL'], 'session_filter': False},
}

# Pair-Specific Settings
PAIR_SETTINGS = {
    'EUR_USD': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ASIAN',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 90.0,
        'backtest_pips': 1313,
    },
    # ... other pairs
}
```

### Key Functions in Config

| Function | Description |
|----------|-------------|
| `calculate_position_size(instrument, price)` | Returns units for $1/pip |
| `get_pair_settings(instrument)` | Returns strategy settings for pair |
| `is_allowed_hour(instrument, hour_utc)` | Checks session filter |
| `calculate_tp_sl(instrument, entry, direction)` | Returns TP/SL prices |
| `get_cooldown_minutes(instrument)` | Returns cooldown setting |

---

## 5. Strategy Logic

### optimized_strategy.py

The strategy file contains all trading signal logic.

### Available Strategies

| Strategy | Description | Entry Conditions |
|----------|-------------|-----------------|
| **RSI_REVERSAL** | RSI crosses 35/65 levels | BUY: RSI crosses up through 35 + green candle |
| **RSI_30_70** | RSI crosses 30/70 levels | BUY: RSI crosses up through 30 + green candle |
| **RSI_EXTREME** | RSI at extreme levels | BUY: RSI < 25 turning up + green candle |
| **MACD_CROSS** | MACD/Signal crossover | BUY: MACD crosses above signal + green candle |
| **MACD_ZERO** | MACD crosses zero | BUY: MACD crosses above 0 + green candle |
| **EMA_PULLBACK** | Trend pullback | BUY: EMA9>EMA21>EMA50 + RSI 35-50 + green |
| **RSI_MACD_COMBO** | Combined signals | BUY: RSI<40 + MACD>Signal + green candle |
| **TRIPLE_CONFIRM** | Triple confirmation | BUY: Uptrend + RSI<50 + MACD bullish |

### Technical Indicators Calculated

```python
# EMAs
ema9, ema21, ema50

# RSI (14-period)
rsi

# MACD (12, 26, 9)
macd, macd_signal

# Volume
volume_avg (20-period), volume_ratio

# Candle
is_green, is_red
```

### Filters (Optional)

- **Volume Filter:** Requires volume > 120% of 20-period average
- **Trend Filter:** Requires EMAs aligned in trade direction

### Main Function

```python
def get_signal(instrument: str, df: pd.DataFrame, config=None) -> Tuple[Optional[str], str]:
    """
    Returns: (signal, reason)
    - signal: 'BUY', 'SELL', or None
    - reason: Human-readable explanation
    """
```

---

## 6. Trading Engine

### run_optimized_forex_paper.py / run_optimized_forex_live.py

The main trading loop that orchestrates everything.

### Main Loop Flow

1. **Initialize** - Connect to OANDA, load config, start logger
2. **Check Open Positions** - Track existing trades
3. **Main Loop** (every 5 seconds):
   - Check for closed trades and log results
   - Apply session-based loss cooldown
   - Display open position status (every 10 iterations)
   - Check for new entry signals (every 12 iterations / 60 seconds)
   - Apply all filters and place orders

### Friday Market Close Protection

```python
# No new trades after 21:00 UTC on Friday
friday_no_new_trades = is_friday and hour_utc >= 21

# Force close all positions at 21:50 UTC on Friday
friday_close_all = is_friday and ((hour_utc == 21 and minute_utc >= 50) or hour_utc >= 22)
```

### Session Statistics Tracking

```python
session_stats = {
    'total_trades': 0,
    'wins': 0,
    'losses': 0,
    'total_pnl': 0.0,
    'by_pair': {pair: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0} for pair}
}
```

---

## 7. OANDA API Client

### oanda_client.py

A wrapper for the OANDA v20 REST API with retry logic and error handling.

### Key Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_account_info()` | Account details | Dict |
| `get_balance()` | Account balance | Float |
| `get_candles(instrument, granularity, count)` | Historical OHLCV | List[Dict] |
| `get_current_price(instrument)` | Bid/Ask/Spread | Dict |
| `get_open_trades()` | Open positions | List[Dict] or None |
| `place_market_order(instrument, units, sl, tp)` | Place order | Dict |
| `close_trade(trade_id)` | Close position | Dict |
| `modify_trade(trade_id, sl, tp)` | Modify SL/TP | Dict |
| `get_trade_history(count)` | Closed trades | List[Dict] |

### Price Precision

```python
# JPY pairs: 3 decimal places
# All others: 5 decimal places
PRICE_PRECISION = {
    "USD_JPY": 3, "EUR_JPY": 3, "GBP_JPY": 3,
    "AUD_JPY": 3, "NZD_JPY": 3, "CAD_JPY": 3, "CHF_JPY": 3,
    # Default: 5
}
```

### Retry Logic

- 3 retries with 1.0s backoff factor
- Handles 500, 502, 503, 504 errors automatically

---

## 8. Trade Logger

### trade_logger.py

Comprehensive logging system for all trades and market data.

### Log Files Created

```
forex_logs/
├── trades_YYYYMMDD_HHMMSS.jsonl     # Trade entries/exits
├── market_data_YYYYMMDD_HHMMSS.jsonl # Market snapshots
├── session_summary_YYYYMMDD_HHMMSS.json # Session stats
├── trades_YYYYMMDD_HHMMSS.csv       # Excel export
└── monthly_summary_YYYYMM.json      # Monthly aggregation
```

### Logged Data

**Trade Entry:**
- Timestamp, instrument, direction
- Entry price, units, position value
- Stop loss, take profit, trailing distance
- Signal analysis (momentum, RSI, HTF trend)
- Risk metrics (SL pips, TP pips, R:R ratio)

**Trade Exit:**
- Exit price, exit reason (TP/SL/TRAIL/MANUAL)
- P&L (dollars and percentage)
- Pips won/lost
- Duration in minutes

### Key Methods

| Method | Description |
|--------|-------------|
| `log_trade_entry(...)` | Log trade open |
| `log_trade_exit(...)` | Log trade close |
| `log_signal_skipped(...)` | Log skipped signals |
| `generate_daily_summary()` | Create session summary |
| `export_to_csv()` | Export to Excel format |
| `get_monthly_stats()` | Aggregate monthly stats |

---

## 9. Session-Based Loss Cooldown

### Feature Description

After 1 loss on any pair, that pair is **blocked** until the next trading session opens.

### Session Times (UTC)

| Session | Start | End |
|---------|-------|-----|
| ASIAN | 00:00 | 08:00 |
| LONDON | 08:00 | 13:00 |
| NEW_YORK | 13:00 | 21:00 |

### How It Works

1. Trade on EUR_USD hits stop loss at 14:00 UTC (NEW_YORK session)
2. EUR_USD is blocked until ASIAN session at 00:00 UTC next day
3. Display: `EUR_USD: BLOCKED after loss (wait 10.0h for ASIAN)`
4. At 00:00 UTC: `EUR_USD: Session block expired, trading enabled`

### Implementation

```python
# Tracking dictionary
pair_blocked_until = {pair: None for pair in config.OPTIMIZED_PAIRS}

# After loss detected
if pnl < 0:
    next_session, next_start = get_next_session_start(now)
    pair_blocked_until[instrument] = next_start

# Before entry check
if pair_blocked_until[instrument]:
    if now < pair_blocked_until[instrument]:
        continue  # Skip this pair
    else:
        pair_blocked_until[instrument] = None  # Clear block
```

---

## 10. How to Run

### Paper Trading (Recommended First)

```bash
cd c:\Users\Jean-Yves\thevolumeainative
python run_optimized_forex_paper.py
```

**Output:**
```
================================================================================
OPTIMIZED FOREX STRATEGY V2 - 14-STRATEGY MULTI-STAGE OPTIMIZATION
================================================================================
Account Type: PRACTICE
Timeframe: M1 (1-minute)
Max Concurrent Positions: 10

Trading 7 Optimized Pairs:
PAIR       STRATEGY           TP   SL   SESSION    WR       PIPS
---------------------------------------------------------------------------
EUR_USD    RSI_REVERSAL       8    20   ASIAN      90.0%   +1313p
GBP_USD    RSI_REVERSAL       8    20   ASIAN      86.0%   +1021p
...
---------------------------------------------------------------------------
Total: +11070 pips (100K candle backtest) | 86% Avg WR | Walk-Forward Validated
================================================================================

[1/4] Connecting to OANDA PRACTICE Account...
      Account: XXX-XXX-XXXXXXXX-XXX
      Balance: $XX,XXX.XX
      Status: CONNECTED
...
```

### Live Trading (Real Money)

```bash
cd c:\Users\Jean-Yves\thevolumeainative
python run_optimized_forex_live.py
```

**Requires two confirmations:**
1. Type `I ACCEPT THE RISK`
2. Type `START`

### Stopping

Press `Ctrl+C` to stop gracefully. You'll be prompted to:
1. Close all positions now
2. Keep positions open (TP/SL remain active on OANDA)

---

## 11. Environment Setup

### .env File Required

Create a `.env` file in the project root:

```env
# OANDA Practice Account
OANDA_PRACTICE_API_KEY=your-practice-api-key-here
OANDA_PRACTICE_ACCOUNT_ID=XXX-XXX-XXXXXXXX-XXX

# OANDA Live Account
OANDA_LIVE_API_KEY=your-live-api-key-here
OANDA_LIVE_ACCOUNT_ID=XXX-XXX-XXXXXXXX-XXX
```

### Getting OANDA API Keys

1. Log in to OANDA at https://www.oanda.com
2. Go to **Manage API Access** in your account settings
3. Generate a new API token
4. Copy your Account ID from the account page

### Account Types

| Type | API URL | Use For |
|------|---------|---------|
| Practice | api-fxpractice.oanda.com | Testing, paper trading |
| Live | api-fxtrade.oanda.com | Real money trading |

---

## 12. Risk Management

### Position Sizing

The system uses **$1 per pip** sizing:

| Pair | Units | Calculation |
|------|-------|-------------|
| EUR_USD, GBP_USD, NZD_USD | 10,000 | Standard |
| USD_CHF | ~11,000 | 10000 / current_price |
| USD_CAD | ~14,350 | 10000 * current_price |
| EUR_GBP | 12,700 | Fixed (GBP quote) |
| AUD_CHF | 11,000 | Fixed (CHF quote) |

### Risk per Trade

- **Stop Loss:** 20 pips = $20 risk
- **Take Profit:** 8 pips = $8 reward
- **Risk:Reward:** 1:2.5 (favorable for high win rate)

### Maximum Risk

- **Max Concurrent Positions:** 10
- **Maximum Exposure:** $200 (10 positions x $20 SL)

### Protections

1. **Session Filtering:** Trade only during optimal hours
2. **Loss Cooldown:** Blocked until next session after loss
3. **Friday Close:** All positions closed before weekend
4. **No Cooldown Between Trades:** Fast re-entry allowed (0 min)

---

## 13. Troubleshooting

### Common Issues

**"OANDA credentials not found"**
- Check .env file exists in project root
- Verify API key and Account ID are correct
- Ensure no extra spaces in .env values

**"Order not filled"**
- Check if market is open (forex closed weekends)
- Verify account has sufficient margin
- Check if pair is tradeable on OANDA

**"Unknown strategy: XXX"**
Valid strategies: RSI_30_70, RSI_REVERSAL, RSI_EXTREME, MACD_CROSS, MACD_ZERO, STRONG_TREND, EMA_PULLBACK, RSI_MACD_COMBO, TRIPLE_CONFIRM

**No signals detected**
- Verify market is open
- Check session filter (ASIAN pairs only trade 00:00-08:00 UTC)
- RSI/MACD conditions may not be met

**API rate limiting**
- The client has built-in retry logic
- If persistent, increase sleep time between iterations

### Log Analysis

Check trade logs for debugging:
```bash
# View recent trades
cat forex_logs/trades_*.jsonl | tail -20

# View session summary
cat forex_logs/session_summary_*.json
```

### Performance Monitoring

The system displays live statistics:
```
SESSION STATS: 5W/1L (83.3%) | Total P&L: $+18.50
---------------------------------------------------------------
PAIR       TRADES   WINS   LOSSES   WIN%     P&L
---------------------------------------------------------------
EUR_USD    2        2      0        100.0%   $+16.00
GBP_USD    1        1      0        100.0%   $+8.00
USD_CHF    2        1      1        50.0%    $-6.00
---------------------------------------------------------------
TOTAL      5        4      1        80.0%    $+18.00
```

---

## Quick Reference

### Commands

| Command | Description |
|---------|-------------|
| `python run_optimized_forex_paper.py` | Start paper trading |
| `python run_optimized_forex_live.py` | Start live trading |
| `Ctrl+C` | Stop gracefully |

### Key Settings

| Setting | Value | Location |
|---------|-------|----------|
| TP/SL | 8/20 pips | Config files |
| Timeframe | M1 (1-min) | Config files |
| Session Filter | Per-pair | TRADING_SESSIONS |
| Max Positions | 10 | Config files |
| Loss Cooldown | Until next session | Trading engine |

### Session Hours (UTC)

| Session | Hours | Pairs |
|---------|-------|-------|
| ASIAN | 00:00-08:00 | EUR_USD, GBP_USD, USD_CHF, USD_CAD |
| ALL | 00:00-24:00 | NZD_USD, AUD_CHF, EUR_GBP |

---

**Document Version:** 2.0
**Last Updated:** December 19, 2025
**System Status:** Production Ready
