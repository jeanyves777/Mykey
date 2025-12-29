# High-Frequency Crypto Trading System with Ensemble ML Voting

A sophisticated ML-powered high-frequency trading system for cryptocurrency markets using Alpaca Paper Trading API.

## Overview

This system uses an **Ensemble Voting Model** combining 5 powerful machine learning models to generate trading signals:

1. **Random Forest** - Captures non-linear patterns in price data
2. **XGBoost** - Gradient boosting for optimal feature importance
3. **LightGBM** - Fast training with leaf-wise tree growth
4. **CatBoost** - Handles categorical features with built-in overfitting prevention
5. **Neural Network** - Deep learning for complex pattern recognition

The final trading decision is made through **weighted voting**, with weights dynamically adjusted based on each model's recent performance.

## Features

- **100+ Engineered Features** from raw OHLCV data
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Market microstructure features
  - Candlestick patterns
  - Volume analysis
  - Volatility metrics

- **Walk-Forward Validation** for robust out-of-sample testing

- **Realistic Backtesting** with:
  - Slippage modeling
  - Commission costs
  - Stop loss / Take profit
  - Trailing stops
  - Position sizing

- **Live Paper Trading** via Alpaca API

## Project Structure

```
High_frequency_crypto_tradin/
├── Crypto_Data_from_Binance/    # Raw 1-minute OHLCV data
├── features/                     # Feature engineering
│   ├── feature_engineer.py       # Main feature pipeline
│   ├── technical_features.py     # Technical indicators
│   └── microstructure_features.py # Market microstructure
├── ml_models/                    # Individual ML models
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── catboost_model.py
│   └── neural_network_model.py
├── ensemble/                     # Ensemble voting system
│   ├── ensemble_voting.py        # Main ensemble class
│   └── dynamic_weighting.py      # Dynamic weight management
├── backtest/                     # Backtesting framework
│   ├── backtest_engine.py        # Backtest execution
│   └── walk_forward.py           # Walk-forward validation
├── engine/                       # Trading engines
│   ├── hf_trading_strategy.py    # HFT strategy logic
│   └── alpaca_live_engine.py     # Alpaca API integration
├── config/                       # Configuration
│   └── trading_config.py         # All system settings
├── saved_models/                 # Trained models
├── logs/                         # Trading logs
├── train_ensemble.py             # Training script
├── run_backtest.py               # Backtest script
└── run_paper_trading.py          # Live trading script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Alpaca Credentials

Create a `.env` file in the project root:

```env
ALPACA_PAPER_KEY=your_paper_api_key
ALPACA_PAPER_SECRET=your_paper_api_secret
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Train the Ensemble Model

```bash
python train_ensemble.py
```

This will:
- Load historical data from `Crypto_Data_from_Binance/`
- Engineer 100+ features
- Train all 5 ML models
- Run walk-forward validation
- Save trained models to `saved_models/`

### 4. Run Backtest

```bash
python run_backtest.py
```

This will:
- Load the trained ensemble model
- Generate ML signals on historical data
- Run backtest with realistic execution
- Generate performance report

### 5. Run Paper Trading

```bash
python run_paper_trading.py
```

This will:
- Connect to Alpaca Paper Trading API
- Start real-time trading with ML signals
- Log all trades and performance

## Configuration

All settings are in `config/trading_config.py`:

```python
# Trading Strategy
position_size_pct = 0.02      # 2% per trade
stop_loss_pct = 0.015         # 1.5% stop loss
take_profit_pct = 0.025       # 2.5% take profit
trailing_stop_pct = 0.012     # 1.2% trailing stop

# Ensemble
confidence_threshold = 0.65   # Minimum confidence for trade
min_model_agreement = 3       # At least 3 models must agree
voting_method = "soft"        # Probability-weighted voting

# Risk Management
max_trades_per_day = 50
max_daily_loss_pct = 0.05     # 5% max daily loss
```

## Model Weights

Initial weights (automatically adjusted based on performance):

| Model | Weight |
|-------|--------|
| XGBoost | 25% |
| Random Forest | 20% |
| LightGBM | 20% |
| CatBoost | 20% |
| Neural Network | 15% |

## Available Symbols

The system supports these crypto pairs (configurable):

- BTC/USD
- ETH/USD
- SOL/USD
- DOGE/USD
- AVAX/USD

## Performance Metrics

The system tracks:

- **Returns**: Total, Annual, Sharpe Ratio, Sortino Ratio
- **Risk**: Max Drawdown, Calmar Ratio
- **Trading**: Win Rate, Profit Factor, Average Win/Loss
- **Costs**: Commission, Slippage

## Walk-Forward Validation

The system uses walk-forward validation to ensure robust out-of-sample testing:

1. Divide data into N windows
2. Train on each window, test on the next
3. Aggregate results across all windows
4. Detect model degradation over time

## License

For educational and research purposes only. Trading involves risk.

## Disclaimer

This system is for paper trading and educational purposes only. Past performance does not guarantee future results. Always understand the risks before trading real money.
