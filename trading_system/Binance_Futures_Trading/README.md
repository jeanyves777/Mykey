# Binance Futures Trading System

Pure Momentum Trading Strategy for Binance Futures (NO ML)

## Overview

This trading system implements a momentum-based strategy for Binance Futures trading. It uses technical analysis indicators to identify high-probability entry points when **Trend + Momentum ALIGN**.

**Key Features:**
- Pure momentum-based signals (no machine learning)
- Multi-timeframe confirmation (optional)
- DCA (Dollar Cost Averaging) support
- Trailing stop functionality
- Paper trading & live trading modes
- Historical backtesting

---

## Quick Start

### 1. Test Connection
```bash
python test_connection.py
```

### 2. Run Paper Trading (Recommended First)
```bash
python run_paper_trading.py --hours 8
```

### 3. Run Backtest
```bash
python run_backtest.py --days 30
```

### 4. Run Live Trading (TESTNET)
```bash
python run_live_trading.py --hours 24
```

---

## Strategy Logic

### Master Momentum Entry System

Entry signals trigger when **ALL 4 conditions align**:

| Condition | Description | Default |
|-----------|-------------|---------|
| **Momentum Spike** | N-bar price change > threshold | >0.15% in 3 bars |
| **Trend Aligned** | EMA fast vs slow matches direction | EMA8 > EMA21 for BUY |
| **RSI OK** | Not at extreme levels | 30 < RSI < 70 |
| **ADX Strong** | Trend exists | ADX > 20 |

### Signal Types

1. **MOMENTUM** - Pure momentum spike with all confirmations
2. **MOMENTUM_STRONG** - Momentum spike >= 2x threshold (extra strong)

### Entry Flow
```
1. Check Momentum (3-bar move)
   │
   ├─ >= 0.15%? ──► Check Trend Alignment
   │                 │
   │                 ├─ EMA8 vs EMA21 aligned?
   │                 │   │
   │                 │   ├─ RSI not extreme?
   │                 │   │   │
   │                 │   │   └─ ADX > 20?
   │                 │   │       │
   │                 │   │       └─► MOMENTUM SIGNAL (enter!)
   │                 │   │
   │                 │   └─► Skip (RSI extreme)
   │                 │
   │                 └─► Skip (trend not aligned)
   │
   └─ < 0.15%? ──► No signal (wait for momentum)
```

---

## Configuration

### Trading Parameters (`config/trading_config.py`)

```python
# Momentum Settings
MOMENTUM_CONFIG = {
    "momentum_period": 3,           # Look back 3 candles
    "momentum_threshold": 0.15,     # 0.15% move required
    "ema_fast_period": 8,           # Fast EMA
    "ema_slow_period": 21,          # Slow EMA
    "rsi_period": 14,
    "rsi_max_for_buy": 70.0,
    "rsi_min_for_sell": 30.0,
    "adx_period": 14,
    "min_adx": 20.0,
    "cooldown_bars": 5,
}

# Strategy Settings
STRATEGY_CONFIG = {
    "max_trades_per_day": 10,
    "take_profit_pct": 0.02,        # 2% TP
    "stop_loss_pct": 0.01,          # 1% SL
    "trailing_stop_trigger": 0.008, # Activate at 0.8% profit
    "trailing_stop_distance": 0.005,# Trail 0.5% behind
    "leverage": 1,                   # No leverage (safe)
}
```

### DCA (Dollar Cost Averaging)

```python
DCA_CONFIG = {
    "enabled": True,
    "position_divisor": 4,          # Initial = 25% of normal size
    "levels": [
        {"trigger": -0.014, "multiplier": 1.75},   # Level 1
        {"trigger": -0.015, "multiplier": 1.25},   # Level 2
        {"trigger": -0.025, "multiplier": 1.50},   # Level 3
        {"trigger": -0.020, "multiplier": 0.75},   # Level 4
    ],
    "max_exposure_multiplier": 2.50,
    "sl_after_dca_pct": 0.012,      # Tighten SL after DCA
}
```

### Trading Symbols

```python
FUTURES_SYMBOLS = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "BNBUSDT",   # Binance Coin
    "SOLUSDT",   # Solana
    "XRPUSDT",   # Ripple
    "DOGEUSDT",  # Dogecoin
]
```

---

## File Structure

```
Binance_Futures_Trading/
├── config/
│   ├── __init__.py
│   └── trading_config.py          # All configuration
├── engine/
│   ├── __init__.py
│   ├── binance_client.py          # Binance API wrapper
│   ├── momentum_signal.py         # Signal generation (NO ML)
│   ├── paper_trading_engine.py    # Paper trading simulation
│   ├── live_trading_engine.py     # Real trading engine
│   └── backtest_engine.py         # Historical backtesting
├── strategies/
│   ├── __init__.py
│   └── momentum_strategy.py       # Strategy wrapper
├── logs/                          # Trade logs (auto-created)
├── run_paper_trading.py           # Paper trading entry point
├── run_live_trading.py            # Live trading entry point
├── run_backtest.py                # Backtest entry point
├── test_connection.py             # Connection test
└── README.md                      # This file
```

---

## Usage Examples

### Paper Trading
```bash
# Run for 24 hours (default)
python run_paper_trading.py

# Run for 8 hours
python run_paper_trading.py --hours 8

# Single timeframe mode (faster, less confirmation)
python run_paper_trading.py --no-mtf

# Skip confirmation prompt
python run_paper_trading.py -y
```

### Backtesting
```bash
# Backtest last 30 days (default)
python run_backtest.py

# Backtest last 60 days
python run_backtest.py --days 60

# Backtest specific symbol
python run_backtest.py --symbol BTCUSDT

# Backtest date range
python run_backtest.py --start 2024-01-01 --end 2024-02-01

# Save results to file
python run_backtest.py --days 30 --save
```

### Live Trading (TESTNET)
```bash
# Run on testnet (default - SAFE)
python run_live_trading.py

# Run for 8 hours
python run_live_trading.py --hours 8

# REAL MONEY MODE (DANGEROUS!)
python run_live_trading.py --live
```

---

## Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_trades_per_day` | 10 | Daily trade limit |
| `max_daily_loss_pct` | 5% | Stop trading at -5% daily |
| `max_total_positions` | 3 | Max concurrent positions |
| `max_position_size_pct` | 25% | Max per position |
| `stop_loss_pct` | 1% | Default stop loss |
| `take_profit_pct` | 2% | Default take profit |

---

## API Configuration

### Demo/Testnet (Already Configured)
The system comes pre-configured with demo API credentials for testing.

### Using Your Own Keys
1. Get API keys from Binance Testnet: https://testnet.binancefuture.com
2. Update `config/trading_config.py`:

```python
BINANCE_CONFIG = {
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "testnet": True,  # Set to False for mainnet (DANGEROUS!)
}
```

Or use environment variables:
```bash
export BINANCE_DEMO_API_KEY="your_key"
export BINANCE_DEMO_API_SECRET="your_secret"
```

---

## Important Notes

1. **Always test on paper trading first** before using real money
2. **Start with testnet** when using live trading engine
3. **Use 1x leverage** until you understand the strategy
4. **Monitor positions** - automated trading requires supervision
5. **Check logs** for trade history and performance metrics

---

## Comparison: Forex vs Binance System

| Feature | Forex (OANDA) | Binance Futures |
|---------|---------------|-----------------|
| ML Signals | Yes (optional) | No |
| Momentum Signals | Yes | Yes |
| Multi-Timeframe | Yes | Yes (optional) |
| DCA | Yes | Yes |
| Trailing Stop | Yes | Yes |
| Leverage | Up to 50x | 1x default |
| Commission | ~0.2 pips | 0.02-0.04% |
| Markets | Forex pairs | Crypto futures |

---

## Troubleshooting

### Connection Failed
```bash
python test_connection.py
```
- Check internet connection
- Verify API credentials
- Confirm testnet/mainnet setting

### No Signals Generated
- Market may be ranging (low ADX)
- Momentum below threshold
- RSI at extremes
- Check logs for rejection reasons

### Orders Not Filling
- Check minimum order sizes
- Verify account balance
- Confirm symbol is tradable

---

## License

For personal use only. Use at your own risk.

**DISCLAIMER**: Trading cryptocurrencies involves substantial risk of loss. This software is provided as-is without any guarantees. Always use proper risk management and never trade with money you cannot afford to lose.
