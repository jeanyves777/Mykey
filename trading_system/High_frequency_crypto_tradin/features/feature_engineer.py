"""
Feature Engineering Pipeline for High-Frequency Crypto Trading
================================================================

Master feature engineering class that:
1. Loads raw OHLCV data from CSV files
2. Computes 100+ technical and microstructure features
3. Creates target labels for ML training
4. Handles missing data and normalization
5. Prepares data for model training and inference
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from .technical_features import TechnicalFeatures
from .microstructure_features import MicrostructureFeatures


class FeatureEngineer:
    """
    Master class for feature engineering pipeline.

    Extracts 100+ features from raw OHLCV data for ML model training.
    """

    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize the FeatureEngineer.

        Args:
            lookback_periods: List of periods for lagged features
        """
        self.technical = TechnicalFeatures()
        self.microstructure = MicrostructureFeatures()
        self.lookback_periods = lookback_periods or [1, 2, 3, 5, 10]
        self.feature_columns = []
        self.scaler_params = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(filepath)

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Parse datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Sort by time
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)

        return df

    def load_multiple_symbols(self, data_dir: str, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            data_dir: Directory containing CSV files
            symbols: List of symbol names (without _1m.csv suffix)

        Returns:
            Dictionary of symbol -> DataFrame
        """
        data_path = Path(data_dir)
        data = {}

        if symbols is None:
            # Load all CSV files
            csv_files = list(data_path.glob("*_1m.csv"))
            symbols = [f.stem.replace('_1m', '') for f in csv_files]

        for symbol in symbols:
            filepath = data_path / f"{symbol}_1m.csv"
            if filepath.exists():
                df = self.load_data(str(filepath))
                data[symbol] = df
                print(f"Loaded {symbol}: {len(df)} rows")
            else:
                print(f"Warning: {filepath} not found")

        return data

    def create_target(self, df: pd.DataFrame,
                      forward_period: int = 5,
                      threshold_pct: float = 0.1,
                      use_adaptive_threshold: bool = True,
                      target_buy_pct: float = 33.0,
                      target_sell_pct: float = 33.0,
                      use_asymmetric_targets: bool = False,
                      extreme_pct: float = 10.0) -> pd.DataFrame:
        """
        Create target variable for ML training.

        Args:
            df: DataFrame with OHLCV data
            forward_period: Number of bars to look ahead
            threshold_pct: Minimum % move to consider significant (used if adaptive=False)
            use_adaptive_threshold: Use percentile-based thresholds for balanced classes
            target_buy_pct: Target percentage of BUY signals (top percentile of returns)
            target_sell_pct: Target percentage of SELL signals (bottom percentile of returns)
            use_asymmetric_targets: If True, only predict EXTREME moves (top/bottom X%)
            extreme_pct: Percentage for extreme moves (default 10% = top/bottom 10%)

        Returns:
            DataFrame with target column added
        """
        # Forward returns
        df['forward_return'] = df['close'].shift(-forward_period) / df['close'] - 1

        valid_returns = df['forward_return'].dropna()

        if use_asymmetric_targets:
            # ASYMMETRIC TARGETS: Only predict extreme moves
            # This dramatically reduces noise - we ignore the middle 80% of moves
            # and only train on the top 10% (BUY) and bottom 10% (SELL) of moves
            #
            # Why this works:
            # - Middle moves (small returns) are essentially random noise
            # - Extreme moves often have detectable patterns (volume spikes, momentum)
            # - By focusing on extremes, we increase signal-to-noise ratio

            buy_threshold = valid_returns.quantile(1 - extreme_pct / 100)
            sell_threshold = valid_returns.quantile(extreme_pct / 100)

            # Ensure meaningful thresholds
            buy_threshold = max(buy_threshold, 0.001)   # At least 0.1%
            sell_threshold = min(sell_threshold, -0.001)  # At least -0.1%

            # IMPORTANT: Middle returns become "NO TRADE" (target=0)
            # These samples can be excluded from training OR trained as HOLD
            df['target'] = 0  # Default: no trade / hold
            df.loc[df['forward_return'] >= buy_threshold, 'target'] = 1   # Strong BUY
            df.loc[df['forward_return'] <= sell_threshold, 'target'] = -1  # Strong SELL

            # Track which samples are "extreme" vs "noise"
            df['is_extreme_move'] = (df['target'] != 0).astype(int)

            self._buy_threshold = buy_threshold
            self._sell_threshold = sell_threshold
            self._target_mode = 'asymmetric'

        elif use_adaptive_threshold:
            # Use percentile-based thresholds for balanced classes
            # This ensures approximately equal BUY/SELL/HOLD distribution
            buy_threshold = valid_returns.quantile(1 - target_buy_pct / 100)
            sell_threshold = valid_returns.quantile(target_sell_pct / 100)

            # Ensure buy_threshold > 0 and sell_threshold < 0 for meaningful signals
            buy_threshold = max(buy_threshold, 0.0001)  # At least 0.01%
            sell_threshold = min(sell_threshold, -0.0001)  # At least -0.01%

            df['target'] = 0
            df.loc[df['forward_return'] >= buy_threshold, 'target'] = 1
            df.loc[df['forward_return'] <= sell_threshold, 'target'] = -1

            # Store thresholds for reference
            self._buy_threshold = buy_threshold
            self._sell_threshold = sell_threshold
            self._target_mode = 'adaptive'
        else:
            # Use fixed threshold
            df['target'] = 0
            df.loc[df['forward_return'] > threshold_pct / 100, 'target'] = 1
            df.loc[df['forward_return'] < -threshold_pct / 100, 'target'] = -1
            self._buy_threshold = threshold_pct / 100
            self._sell_threshold = -threshold_pct / 100
            self._target_mode = 'fixed'

        # Binary target for simpler models
        df['target_binary'] = (df['forward_return'] > 0).astype(int)

        # Forward max gain/loss (for risk-adjusted targets)
        df['forward_max_gain'] = df['high'].rolling(forward_period).max().shift(-forward_period) / df['close'] - 1
        df['forward_max_loss'] = df['low'].rolling(forward_period).min().shift(-forward_period) / df['close'] - 1

        # Risk-reward target
        gain_potential = df['forward_max_gain'].abs()
        loss_potential = df['forward_max_loss'].abs()
        df['risk_reward'] = gain_potential / (loss_potential + 1e-10)

        return df

    def add_lagged_features(self, df: pd.DataFrame,
                            feature_cols: List[str]) -> pd.DataFrame:
        """
        Add lagged versions of features.

        Args:
            df: DataFrame with features
            feature_cols: Columns to create lags for

        Returns:
            DataFrame with lagged features
        """
        for col in feature_cols:
            for lag in self.lookback_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    def add_rolling_stats(self, df: pd.DataFrame,
                          feature_cols: List[str],
                          windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling statistics for features.

        Args:
            df: DataFrame with features
            feature_cols: Columns to compute stats for
            windows: Rolling window sizes

        Returns:
            DataFrame with rolling stats
        """
        for col in feature_cols:
            for window in windows:
                df[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window).max()

        return df

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for the dataset.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all features
        """
        df = df.copy()

        # Compute technical features
        df = self.technical.compute_all(df)

        # Compute microstructure features
        df = self.microstructure.compute_all(df)

        # Add interaction features
        df = self._add_interaction_features(df)

        # Add lagged features for key indicators
        key_features = ['rsi_14', 'macd', 'return_1', 'volume_ratio', 'order_imbalance']
        available_features = [f for f in key_features if f in df.columns]
        df = self.add_lagged_features(df, available_features)

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interaction terms."""

        # RSI * Volume interactions
        if 'rsi_14' in df.columns and 'volume_ratio' in df.columns:
            df['rsi_volume_interaction'] = (df['rsi_14'] - 50) * df['volume_ratio']

        # MACD * Trend interactions
        if 'macd' in df.columns and 'adx' in df.columns:
            df['macd_trend_strength'] = df['macd'] * df['adx'] / 100

        # Volatility * Momentum
        if 'volatility_10' in df.columns and 'roc_10' in df.columns:
            df['vol_momentum_interaction'] = df['volatility_10'] * abs(df['roc_10'])

        # Volume * Price action
        if 'volume_spike' in df.columns and 'breakout_up' in df.columns:
            df['volume_breakout_up'] = df['volume_spike'] * df['breakout_up']
            df['volume_breakout_down'] = df['volume_spike'] * df['breakout_down']

        # Mean reversion * RSI
        if 'zscore_20' in df.columns and 'rsi_14' in df.columns:
            df['mean_reversion_signal'] = -df['zscore_20'] * (100 - df['rsi_14']) / 100

        # Order flow * Volume
        if 'order_imbalance' in df.columns and 'volume_ratio' in df.columns:
            df['order_flow_strength'] = df['order_imbalance'] * df['volume_ratio']

        return df

    def normalize_features(self, df: pd.DataFrame,
                          feature_cols: List[str],
                          method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features for ML models.

        Args:
            df: DataFrame with features
            feature_cols: Columns to normalize
            method: 'zscore' or 'minmax'

        Returns:
            Normalized DataFrame
        """
        for col in feature_cols:
            if col not in df.columns:
                continue

            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                self.scaler_params[col] = {'mean': mean, 'std': std}
                df[col] = (df[col] - mean) / (std + 1e-10)
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                self.scaler_params[col] = {'min': min_val, 'max': max_val}
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)

        return df

    def handle_missing_values(self, df: pd.DataFrame,
                              method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame with potential missing values
            method: 'ffill', 'bfill', 'zero', or 'drop'

        Returns:
            DataFrame with handled missing values
        """
        if method == 'ffill':
            df = df.ffill().bfill()
        elif method == 'bfill':
            df = df.bfill().ffill()
        elif method == 'zero':
            df = df.fillna(0)
        elif method == 'drop':
            df = df.dropna()

        # Replace infinities with large finite values
        df = df.replace([np.inf, -np.inf], [1e10, -1e10])

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding target and metadata).

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'timestamp', 'close_time', 'datetime', 'date',
            'open', 'high', 'low', 'close', 'volume',
            'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
            'target', 'target_binary', 'forward_return',
            'forward_max_gain', 'forward_max_loss', 'risk_reward'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        return feature_cols

    def prepare_training_data(self, df: pd.DataFrame,
                              forward_period: int = 5,
                              threshold_pct: float = 0.1,
                              normalize: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for ML model training.

        Args:
            df: Raw OHLCV DataFrame
            forward_period: Bars to look ahead for target
            threshold_pct: Threshold for significant moves
            normalize: Whether to normalize features

        Returns:
            Tuple of (features_df, target_multiclass, target_binary)
        """
        # Compute all features
        df = self.compute_all_features(df)

        # Create targets
        df = self.create_target(df, forward_period, threshold_pct)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Get feature columns
        feature_cols = self.get_feature_columns(df)

        # Normalize if requested
        if normalize:
            df = self.normalize_features(df, feature_cols)

        # Remove rows with NaN targets (at end of dataset)
        df = df.dropna(subset=['target'])

        X = df[feature_cols]
        y_multi = df['target']
        y_binary = df['target_binary']

        print(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        print(f"Target distribution: {y_multi.value_counts().to_dict()}")

        return X, y_multi, y_binary

    def prepare_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for real-time inference.

        Args:
            df: Recent OHLCV DataFrame

        Returns:
            Features DataFrame ready for prediction
        """
        # Compute all features
        df = self.compute_all_features(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Apply saved normalization parameters
        for col, params in self.scaler_params.items():
            if col in df.columns:
                if 'mean' in params:
                    df[col] = (df[col] - params['mean']) / (params['std'] + 1e-10)
                elif 'min' in params:
                    df[col] = (df[col] - params['min']) / (params['max'] - params['min'] + 1e-10)

        # Return only feature columns
        return df[self.feature_columns] if self.feature_columns else df

    def save_feature_config(self, filepath: str):
        """Save feature configuration for later use."""
        import json
        config = {
            'feature_columns': self.feature_columns,
            'scaler_params': {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in self.scaler_params.items()},
            'lookback_periods': self.lookback_periods
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def load_feature_config(self, filepath: str):
        """Load feature configuration."""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        self.feature_columns = config['feature_columns']
        # Handle optional fields with defaults
        self.scaler_params = config.get('scaler_params', {})
        self.lookback_periods = config.get('lookback_periods', [1, 2, 3, 5, 10])


if __name__ == "__main__":
    # Test the feature engineering pipeline
    fe = FeatureEngineer()

    # Load sample data
    data_dir = Path(__file__).parent.parent / "Crypto_Data_from_Binance"

    # Load BTC data
    btc_data = fe.load_data(str(data_dir / "BTCUSDT_1m.csv"))
    print(f"Loaded BTC data: {len(btc_data)} rows")

    # Prepare training data
    X, y_multi, y_binary = fe.prepare_training_data(btc_data)
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y_multi.shape}")
    print(f"\nSample features:\n{X.head()}")
