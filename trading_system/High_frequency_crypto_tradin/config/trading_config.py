"""
Trading Configuration for High-Frequency Crypto Trading System
================================================================

Centralized configuration for all system components.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
import json


@dataclass
class AlpacaCredentials:
    """Alpaca API credentials."""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lookback_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    forward_period: int = 5
    target_threshold_pct: float = 0.1
    normalize: bool = True
    normalization_method: str = "zscore"


@dataclass
class ModelConfig:
    """ML model configuration."""
    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 20
    rf_min_samples_leaf: int = 10

    # XGBoost
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05

    # LightGBM
    lgb_n_estimators: int = 300
    lgb_max_depth: int = 10
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 50

    # CatBoost
    cat_iterations: int = 300
    cat_depth: int = 8
    cat_learning_rate: float = 0.05

    # Neural Network
    nn_hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    nn_dropout_rate: float = 0.3
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 256
    nn_epochs: int = 100


@dataclass
class EnsembleConfig:
    """Ensemble configuration for BUY-only spot trading."""
    voting_method: str = "soft"  # "soft" or "hard"
    confidence_threshold: float = 0.45  # Confidence for trade execution
    min_model_agreement: int = 2       # 2/5 - let TP/SL filter, not ML

    # BUY probability threshold for binary classification
    # DON'T USE argmax (always picks NO_TRADE at ~80%)
    # Instead, predict BUY if P(BUY) > this threshold
    #
    # Analysis of model probability distribution:
    # - BUY proba range: 0.17 - 0.42 (mean 0.34, median 0.36)
    # - Threshold 0.35 → 74% BUYs (too many, random)
    # - Threshold 0.38 → 15% BUYs (selective, good)
    # - Threshold 0.40 → 1% BUYs (too few)
    #
    # With 0.40% TP and 0.30% SL, need ~43% win rate to break even
    buy_probability_threshold: float = 0.38  # Selective: ~15% of signals are BUY

    # IMPORTANT: For spot crypto trading, only BUY (1) signals are used
    # SELL (-1) signals are ignored (no short selling)
    # HOLD (0) signals result in no action
    buy_only_mode: bool = True  # Enable BUY-only signal filtering

    # Initial weights
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        'random_forest': 0.20,
        'xgboost': 0.25,
        'lightgbm': 0.20,
        'catboost': 0.20,
        'neural_network': 0.15
    })

    # Dynamic weighting
    use_dynamic_weights: bool = True
    weight_learning_rate: float = 0.1
    min_weight: float = 0.05
    max_weight: float = 0.50


@dataclass
class StrategyConfig:
    """HIGH-FREQUENCY SCALPING - Quick entries/exits, no DCA."""
    # STRATEGY: Pure HF scalping - tight TP/SL, quick exits
    # - NO DCA (causes long hold times)
    # - Tight take profit (0.3-0.5%)
    # - Tight stop loss (0.2-0.3%)
    # - BUY-only (spot crypto)

    position_size_pct: float = 0.02
    max_position_value: float = 10000

    # HF SCALPING settings - ADJUSTED for higher win rate
    # Previous: TP=0.4%, SL=0.3% needed 43% WR to break even (got 15-19%)
    # New: TP=0.25%, SL=0.50% needs only 33% WR to break even
    # Smaller TP = more likely to hit = higher win rate
    take_profit_pct: float = 0.0025       # 0.25% TP - easier to hit
    stop_loss_pct: float = 0.005          # 0.50% SL - more room to breathe
    trailing_stop_pct: float = 0.002      # 0.2% trailing - lock in gains
    max_trades_per_day: int = 500         # Allow many trades
    max_daily_loss_pct: float = 0.03      # 3% max daily loss
    cooldown_seconds: int = 10            # 10 second cooldown only

    # DCA DISABLED for true HF scalping
    dca_enabled: bool = False
    max_dca_entries: int = 0
    dca_level_1_trigger_pct: float = 0.01
    dca_level_1_multiplier: float = 1.0
    dca_level_2_trigger_pct: float = 0.015
    dca_level_2_multiplier: float = 1.0
    dca_level_3_trigger_pct: float = 0.02
    dca_level_3_multiplier: float = 1.0
    dca_level_4_trigger_pct: float = 0.025
    dca_level_4_multiplier: float = 1.0
    dca_profit_target_pct: float = 0.01
    sl_after_last_dca_pct: float = 0.01


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_pct: float = 0.0005
    walk_forward_splits: int = 5
    min_train_size: int = 5000


@dataclass
class TradingConfig:
    """Master configuration for the trading system."""
    # Component configs
    alpaca: AlpacaCredentials = field(default_factory=AlpacaCredentials)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"
    ])

    # Data paths
    data_dir: str = ""
    model_save_dir: str = ""
    log_dir: str = ""

    # Runtime settings
    trading_interval_seconds: int = 60
    max_runtime_hours: float = 24
    verbose: bool = True

    def __post_init__(self):
        """Set up default paths."""
        base_dir = Path(__file__).parent.parent

        if not self.data_dir:
            # Use fresh data with synthetic + real 1-minute candles
            self.data_dir = str(base_dir / "Crypto_Data_Fresh")

        if not self.model_save_dir:
            # V18: Use improved models trained on all 5 symbols with optimal config
            self.model_save_dir = str(base_dir / "saved_models_improved")

        if not self.log_dir:
            self.log_dir = str(base_dir / "logs")

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'alpaca': {
                'api_key': self.alpaca.api_key[:8] + "..." if self.alpaca.api_key else "",
                'base_url': self.alpaca.base_url
            },
            'features': {
                'lookback_periods': self.features.lookback_periods,
                'forward_period': self.features.forward_period,
                'target_threshold_pct': self.features.target_threshold_pct
            },
            'ensemble': {
                'voting_method': self.ensemble.voting_method,
                'confidence_threshold': self.ensemble.confidence_threshold,
                'min_model_agreement': self.ensemble.min_model_agreement,
                'use_dynamic_weights': self.ensemble.use_dynamic_weights
            },
            'strategy': {
                'position_size_pct': self.strategy.position_size_pct,
                'stop_loss_pct': self.strategy.stop_loss_pct,
                'take_profit_pct': self.strategy.take_profit_pct,
                'max_trades_per_day': self.strategy.max_trades_per_day
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'walk_forward_splits': self.backtest.walk_forward_splits
            },
            'symbols': self.symbols,
            'data_dir': self.data_dir,
            'model_save_dir': self.model_save_dir
        }

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'TradingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = cls()

        # Update from loaded values
        if 'symbols' in config_dict:
            config.symbols = config_dict['symbols']

        if 'strategy' in config_dict:
            for key, value in config_dict['strategy'].items():
                if hasattr(config.strategy, key):
                    setattr(config.strategy, key, value)

        if 'ensemble' in config_dict:
            for key, value in config_dict['ensemble'].items():
                if hasattr(config.ensemble, key):
                    setattr(config.ensemble, key, value)

        return config


def load_config() -> TradingConfig:
    """Load configuration from environment variables and defaults."""
    config = TradingConfig()

    # Load Alpaca credentials from environment
    # Using ALPACA_CRYPTO_KEY for crypto trading account
    config.alpaca.api_key = os.getenv('ALPACA_CRYPTO_KEY', '')
    config.alpaca.api_secret = os.getenv('ALPACA_CRYPTO_SECRET', '')
    config.alpaca.base_url = os.getenv('ALPACA_CRYPTO_BASE_URL', 'https://paper-api.alpaca.markets')

    return config


def get_default_config() -> TradingConfig:
    """Get default configuration."""
    return TradingConfig()


if __name__ == "__main__":
    # Test configuration
    config = load_config()
    print("Configuration loaded:")
    print(json.dumps(config.to_dict(), indent=2))
