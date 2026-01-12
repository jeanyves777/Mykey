# Forex Trading System

HTF Confluence Strategy for Forex markets using OANDA API v20.

## Overview

This is a **direct port** of the successful Binance Futures trading strategy to Forex markets:

- **Strategy**: HTF (Higher Timeframe) Trend + Confluence
- **Entry**: 8/8 confluence requirement (HTF trend + LTF signals)
- **Exit**: Smart profit lock, trailing profit lock, fakeout protection
- **Results**: Proven +32% ROI on crypto markets (90-day backtest)

## Strategy Details

### Multi-Timeframe Analysis
- **4H/1H**: HTF trend detection (21/50 EMA crossover)
- **15m**: Entry signals (MACD + RSI + EMA confluence)
- **5m**: Confirmation (EMA alignment + volume)
- **1m**: Pullback detection (smart entry filter)

### Entry Conditions (8/8 Confluence)
All 8 conditions must be met:
1. **HTF Trend**: 1H EMA 21 > 50 (bullish) or < (bearish)
2. **LTF Alignment**: 15m EMA 9 > 21 (bullish) or < (bearish)
3. **RSI Zone**: 35-75 for longs, 25-65 for shorts
4. **MACD**: Line > Signal (bullish) or < (bearish)
5. **5m Confirmation**: EMA 9 > 21 (bullish) or < (bearish)
6. **ADX Trending**: ADX > 20 (not choppy)
7. **Momentum**: MACD histogram increasing (bullish) or decreasing (bearish)
8. **Volume**: 5m volume > 1.2x average (confirmation)

### Risk Management
- **Leverage**: 50x (standard for forex)
- **Take Profit**: 80 pips (default)
- **Stop Loss**: 200 pips (default)
- **Risk/Reward**: 2.5:1 ratio
- **Risk per trade**: 2% of capital

### Smart Exit System
1. **Profit Lock**: Close if in profit and HTF trend reverses
2. **Trailing Lock**: Lock profits after 60 pips (floor 30 pips)
3. **Fakeout Protection**: Exit early on suspected fakeouts
4. **Damage Control**: Close immediately if loss > 40 pips after reversal

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config/trading_config.py`:

```python
# OANDA API Configuration
OANDA_API_KEY = "your-api-key-here"
OANDA_ACCOUNT_ID = "your-account-id-here"
USE_PRACTICE = True  # Set to False for live trading
```

### 3. Run the Trading System

**Practice Account (Recommended)**:
```bash
python main.py
```

**Live Account** (CAUTION):
```bash
python main.py --live
```

**Custom Symbols**:
```bash
python main.py --symbols EUR_USD,GBP_USD,USD_JPY
```

**Custom Risk**:
```bash
python main.py --risk 1.5  # 1.5% risk per trade
```

## Project Structure

```
forex_trading_system/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── config/
│   ├── __init__.py
│   └── trading_config.py            # Strategy configuration
└── engine/
    ├── __init__.py
    ├── oanda_client.py              # OANDA API v20 client
    ├── htf_confluence_strategy.py   # Strategy logic
    └── htf_confluence_live_engine.py  # Live trading engine
```

## Forex Pairs

Default symbols (from config):
- EUR_USD (Euro / US Dollar)
- GBP_USD (British Pound / US Dollar)
- USD_JPY (US Dollar / Japanese Yen)
- AUD_USD (Australian Dollar / US Dollar)
- USD_CAD (US Dollar / Canadian Dollar)
- NZD_USD (New Zealand Dollar / US Dollar)

## Risk Warning

**FOREX TRADING INVOLVES SUBSTANTIAL RISK OF LOSS.**

- Start with a **practice account** to test the strategy
- Never risk more than 1-2% of capital per trade
- Understand the strategy before using real money
- Past performance does not guarantee future results

## Differences from Binance Version

### API Differences
- **OANDA v20 REST API** vs Binance REST/WebSocket
- Pip-based TP/SL vs ROI-based
- Position model: Aggregated vs individual orders

### Market Differences
- **Forex**: 24/5 trading (closed weekends)
- **Leverage**: 50x standard (vs 20x on crypto)
- **Spreads**: Variable spreads (vs fixed fees)
- **Symbols**: Currency pairs (vs crypto pairs)

### Strategy Adaptations
- TP/SL in pips (80/200) instead of ROI % (60%/20%)
- Pip location handling (JPY pairs = -2, others = -4)
- OANDA-specific order management
- Identical confluence logic (8/8 conditions)

## Monitoring

The system logs:
- **Entry signals**: All 8/8 confluence conditions met
- **Active positions**: Current P&L in pips
- **Smart exits**: Profit lock, trailing lock, fakeout protection
- **Session stats**: Wins/losses, total P&L

Example output:
```
============================================================
HTF CONFLUENCE FOREX LIVE TRADING ENGINE
============================================================
Symbols: EUR_USD, GBP_USD, USD_JPY
Mode: PRACTICE
Risk per trade: 2%
------------------------------------------------------------
[EUR_USD] 📊 Signal: BUY | Score: 8/8 | 1H: Bullish | 15m: EMA 9>21 | RSI 52 | MACD>0 | 5m: Confirmed | ADX 24 | Momentum ↑ | Vol 1.3x
[EUR_USD] Executing BUY | Size: 10000 | Entry: 1.09543 | TP: 1.10343 | SL: 1.07543
[EUR_USD] ✓ Trade executed
------------------------------------------------------------
Stats: 3W/1L | PnL: $+187.50
Active positions: 1
------------------------------------------------------------
```

## Support

For questions or issues:
1. Check the configuration in `config/trading_config.py`
2. Review the strategy logic in `engine/htf_confluence_strategy.py`
3. Compare with Binance version at `/workspaces/Mykey/trading_system/Binance_Futures_Trading/`

## License

This is a private trading system. Not for distribution.
