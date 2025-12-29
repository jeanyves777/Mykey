"""
Forex Trading ML Version
========================

ML-powered Forex trading system with 5-model ensemble.
Based on the High-Frequency Crypto Trading system architecture.

Components:
- config/: Trading configuration and parameters
- data/: OANDA data handlers and downloaders
- features/: Feature engineering (100+ features)
- ml_models/: Individual ML models (RF, XGBoost, LightGBM, CatBoost, NN)
- ensemble/: Ensemble voting system with dynamic weighting
- risk_management/: Position sizing, risk limits, trade execution
- engine/: Paper and live trading engines
- backtest/: Backtesting and walk-forward validation
"""

__version__ = "1.0.0"
__author__ = "Trading System"
