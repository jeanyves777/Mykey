# OANDA Forex Trading System

Multi-timeframe momentum trading strategy for major forex pairs using OANDA's v20 API.

## Features

### Strategy: Multi-Timeframe Momentum
This strategy implements a comprehensive 11-step validation pipeline similar to your MARA options strategy:

1. **Market Timing Check** - 24/5 forex market with optimal session detection
2. **Position & Risk Check** - Max trades per day, profit targets, position limits
3. **Technical Scoring** (1-MIN bars) - 17+ indicators including EMA stack, RSI, MACD, Bollinger Bands
4. **Price Action Analysis** (5-MIN bars) - Candle patterns, higher highs/lower lows
5. **Real-Time Momentum** (Last 5 1-MIN bars) - HIGHEST WEIGHT, can override other signals
6. **V3 Weighted Decision** - Combines all methods (Technical=1x, Price Action=1x, Momentum=2x)
7. **Higher Timeframe Trend Filter** (30-MIN + 1-HOUR) - STRICT alignment required
8. **Pullback Detection** (5-MIN) - Wait for better entry timing
9. **Position Sizing** - Percentage-based with account protection
10. **Risk Management** - Automatic SL, TP, and trailing stops
11. **Order Execution** - Market orders with slippage and commission simulation

## Major Forex Pairs Traded

- EUR/USD - Euro / US Dollar
- GBP/USD - British Pound / US Dollar
- USD/JPY - US Dollar / Japanese Yen
- USD/CHF - US Dollar / Swiss Franc
- AUD/USD - Australian Dollar / US Dollar
- USD/CAD - US Dollar / Canadian Dollar
- NZD/USD - New Zealand Dollar / US Dollar

## Setup

### 1. Get OANDA Account

1. Sign up for free practice account: https://www.oanda.com/
2. Get your API credentials:
   - Go to "Manage API Access"
   - Generate API token
   - Note your Account ID

### 2. Configure Environment

Add to your `.env` file:

```bash
# OANDA Practice Account
OANDA_PRACTICE_API_KEY=your_practice_api_key_here
OANDA_PRACTICE_ACCOUNT_ID=your_practice_account_id_here

# OANDA Live Account (optional)
OANDA_LIVE_API_KEY=your_live_api_key_here
OANDA_LIVE_ACCOUNT_ID=your_live_account_id_here
```

### 3. Install Dependencies

The system uses standard packages already in your environment:
- pandas
- numpy
- pytz
- requests
- python-dotenv

No additional packages needed!

## Usage

### Paper Trading (Live Simulation)

Run paper trading with real OANDA data but simulated trades:

```bash
# Run for 8 hours (default)
python trading_system/Forex_Trading/run_paper_trading.py -y

# Run for 24 hours
python trading_system/Forex_Trading/run_paper_trading.py --hours 24 -y

# Run for 1 hour (testing)
python trading_system/Forex_Trading/run_paper_trading.py --hours 1 -y
```

### Backtesting

Run historical backtests on OANDA data:

```bash
# Backtest last 30 days (default: EUR/USD, GBP/USD)
python trading_system/Forex_Trading/run_backtest.py -y

# Backtest last 60 days
python trading_system/Forex_Trading/run_backtest.py --days 60 -y

# Backtest specific date range
python trading_system/Forex_Trading/run_backtest.py --start 2024-01-01 --end 2024-12-31 -y

# Backtest multiple pairs with custom capital
python trading_system/Forex_Trading/run_backtest.py --instruments EUR_USD GBP_USD USD_JPY --capital 25000 -y

# Backtest all major pairs
python trading_system/Forex_Trading/run_backtest.py --instruments EUR_USD GBP_USD USD_JPY USD_CHF AUD_USD USD_CAD NZD_USD --days 90 -y
```

### Test OANDA Connection

Test your API credentials:

```bash
python trading_system/Forex_Trading/engine/oanda_client.py
```

## Strategy Configuration

Edit `trading_system/Forex_Trading/config/forex_trading_config.py`:

```python
STRATEGY_CONFIG = {
    "max_trades_per_day": 3,           # Max trades per day
    "daily_profit_target": 0.02,       # 2% daily profit target
    "trade_size_pct": 0.10,            # 10% of account per trade
    "take_profit_pct": 0.015,          # 1.5% TP (150 pips)
    "stop_loss_pct": 0.01,             # 1% SL (100 pips)
    "trailing_stop_trigger": 0.006,    # Start trailing at 0.6%
    "trailing_stop_distance": 0.004,   # Trail 0.4% behind
}

RISK_CONFIG = {
    "max_position_size_pct": 0.20,     # Max 20% per position
    "max_daily_loss_pct": 0.03,        # Stop at -3% daily
    "max_total_positions": 3,          # Max concurrent positions
}
```

## Project Structure

```
trading_system/Forex_Trading/
├── config/
│   └── forex_trading_config.py      # Strategy & risk parameters
├── engine/
│   ├── oanda_client.py               # OANDA API wrapper
│   ├── forex_paper_trading_engine.py # Paper trading engine
│   └── forex_backtest_engine.py      # Backtest engine
├── strategies/
│   └── multi_timeframe_momentum.py   # Main strategy logic
├── logs/                             # Trade logs (auto-created)
├── run_paper_trading.py              # Paper trading script
├── run_backtest.py                   # Backtest script
└── README.md                         # This file
```

## How It Works

### Entry Signal Generation

The strategy analyzes multiple timeframes and indicators:

**1-Minute Bars (Entry Signals):**
- EMA stack (9, 20, 50)
- RSI (14)
- MACD with histogram
- Bollinger Bands
- Price momentum

**5-Minute Bars (Price Action):**
- Candle color patterns
- Higher highs / Lower lows
- 5-bar moving average
- Pullback detection

**30-Minute & 1-Hour Bars (Trend Filter):**
- **STRICT**: Both must agree on trend direction
- Blocks trades against higher timeframe trend
- Blocks trades when timeframes conflict

### Decision Logic

1. **All 3 methods agree** → HIGH CONFIDENCE trade
2. **Strong momentum override** → APPROVED (even if others disagree)
3. **Momentum + 1 other agree** → APPROVED
4. **Weighted score decision** → Based on 1x, 1x, 2x weights
5. **No consensus** → SKIP TRADE

### Exit Management

- **Take Profit**: 1.5% gain (150 pips on standard lot)
- **Stop Loss**: 1% loss (100 pips on standard lot)
- **Trailing Stop**: Triggers at +0.6%, trails 0.4% behind high/low

## Trading Sessions

Forex market is open 24/5, but best liquidity during:

- **London Session**: 08:00-17:00 UTC
- **New York Session**: 13:00-22:00 UTC
- **London/NY Overlap**: 13:00-17:00 UTC (BEST for this strategy)

The strategy monitors all sessions but performs best during high-liquidity periods.

## Risk Management

### Position Sizing
- Each trade: 10% of account balance
- Maximum position size: 20% of account
- Minimum: 1,000 units (1 micro lot)

### Daily Limits
- Max 3 trades per day
- Stop trading at +2% daily profit
- Stop trading at -3% daily loss

### Maximum Positions
- Max 3 concurrent positions across all pairs
- One position per instrument

## Output & Logs

### Paper Trading Output
- Real-time position updates
- Entry/exit notifications with full reasoning
- P&L tracking
- Daily summary
- Trade logs saved to `logs/forex_paper_trades_YYYYMMDD.json`

### Backtest Output
- Complete trade history
- Win rate and profit factor
- Maximum drawdown
- Equity curve
- Results saved to `logs/forex_backtest_YYYYMMDD_HHMMSS.json`

## Example Output

```
================================================================================
[TRADE] ENTERED LONG position on EUR_USD
[TRADE] Entry Price: 1.08450
[TRADE] Units: 10,000
[TRADE] Position Size: $10,845.00
[TRADE] Stop Loss: 1.07365
[TRADE] Take Profit: 1.10078
[TRADE] Commission: $2.17
[TRADE] Confidence: HIGH
[TRADE] Reason: ✅ ALL CHECKS PASSED - PULLBACK MET - Dip + Green recovery
[TRADE] Balance: $9,997.83
================================================================================

================================================================================
[TRADE] CLOSED LONG position on EUR_USD
[TRADE] Entry: 1.08450 -> Exit: 1.10078
[TRADE] P&L: $162.80 (+1.50%)
[TRADE] Reason: TAKE PROFIT at 1.10078
[TRADE] New Balance: $10,160.63
[TRADE] Total Return: +1.61%
================================================================================
```

## Why OANDA?

- ✅ **No minimum deposit** for practice account
- ✅ **Excellent API** - Clean REST + streaming
- ✅ **Tight spreads** - EUR/USD typically 0.9-1.2 pips
- ✅ **Reliable execution** - Institutional-grade infrastructure
- ✅ **Python SDK available** - Easy integration
- ✅ **Regulated** - Multiple jurisdictions
- ✅ **Free practice account** - Perfect for testing

## Next Steps

1. **Test Connection**: Run `python trading_system/Forex_Trading/engine/oanda_client.py`
2. **Run Backtest**: Test strategy on historical data
3. **Paper Trade**: Run live simulation to verify performance
4. **Optimize**: Adjust parameters based on results
5. **Scale Up**: Add more pairs or increase position sizes
6. **Go Live**: Switch to live account when confident

## Performance Tips

### For Better Results:
- Trade during London/NY overlap (13:00-17:00 UTC)
- Start with EUR/USD and GBP/USD (highest liquidity)
- Respect daily profit/loss limits
- Let trailing stops work - don't override
- Review trade logs to identify patterns

### For Safer Trading:
- Start with micro lots (1,000 units)
- Use lower position sizing (5% instead of 10%)
- Reduce max trades per day (2 instead of 3)
- Tighter stop losses (0.5% instead of 1%)
- Only trade major pairs with tight spreads

## Troubleshooting

### "OANDA credentials not found"
- Check your `.env` file exists in project root
- Verify variable names: `OANDA_PRACTICE_API_KEY` and `OANDA_PRACTICE_ACCOUNT_ID`
- Make sure no extra spaces or quotes

### "Insufficient data"
- OANDA practice accounts need to be active
- Try running during London/NY sessions
- Check internet connection

### "No trades executed"
- Strategy is strict - this is normal
- Try longer backtest periods (60-90 days)
- Lower session has lower signal frequency

## Comparison to Your Options Strategies

| Feature | Options (MARA) | Forex (OANDA) |
|---------|---------------|---------------|
| **Market Hours** | 9:30-16:00 EST | 24/5 |
| **Validation Steps** | 11 steps | 11 steps (same pipeline) |
| **HTF Filter** | 30min + 1hour STRICT | 30min + 1hour STRICT |
| **Pullback Detection** | 5-min HTF | 5-min HTF |
| **Momentum Weight** | 2x | 2x |
| **Take Profit** | +25% | +1.5% (150 pips) |
| **Stop Loss** | Trailing from +10% | 1% with trailing from +0.6% |
| **Max Trades/Day** | 3 | 3 |
| **Position Sizing** | $2000 target | 10% of account |

The forex strategy uses the **same validation logic** as your successful options strategies, adapted for 24/5 forex market dynamics and pip-based pricing.

## License

Part of thevolumeainative trading system.
