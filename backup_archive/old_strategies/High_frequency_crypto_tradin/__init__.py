"""
High-Frequency Crypto Trading System with Ensemble ML Voting
============================================================

A sophisticated ML-powered trading system using 5 powerful models:
1. Random Forest - Captures non-linear patterns
2. XGBoost - Gradient boosting for feature importance
3. LightGBM - Fast gradient boosting with leaf-wise growth
4. CatBoost - Handles categorical features and reduces overfitting
5. Neural Network - Deep learning for complex pattern recognition

The ensemble combines predictions using dynamic weighted voting
based on recent model performance.

Features:
- 100+ engineered features from raw OHLCV data
- Walk-forward validation for robust backtesting
- Real-time Alpaca paper trading integration
- Risk management and position sizing
- Performance analytics and logging
"""

__version__ = "1.0.0"
__author__ = "Trading System"
