"""
Feature Engineer for Forex ML System
=====================================

Main feature engineering pipeline combining all feature sources.
"""

import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import os

# Suppress pandas fragmentation and deprecation warnings
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')
warnings.filterwarnings('ignore', category=FutureWarning)

from trading_system.Forex_Trading.Forex_Trading_ML_Version.features.technical_features import TechnicalFeatures
from trading_system.Forex_Trading.Forex_Trading_ML_Version.features.microstructure_features import MicrostructureFeatures


class FeatureEngineer:
    """Main feature engineering class."""

    def __init__(self, config=None):
        """Initialize feature engineer."""
        self.config = config
        self.technical = TechnicalFeatures(config.features if config else None)
        self.microstructure = MicrostructureFeatures(config.features if config else None)

        # Feature lists
        self.feature_names: List[str] = []
        self.target_col = 'target'

        # Normalization parameters
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Engineer all features from raw OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            fit: Whether to fit normalization parameters (True for training)

        Returns:
            DataFrame with all features calculated
        """
        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate technical features
        df = self.technical.calculate_all(df)

        # Calculate microstructure features
        df = self.microstructure.calculate_all(df)

        # Add lagged features
        df = self._add_lagged_features(df)

        # Add time-based features
        if 'time' in df.columns or df.index.name == 'time' or isinstance(df.index, pd.DatetimeIndex):
            df = self._add_time_features(df)

        # Get feature columns
        self._identify_feature_columns(df)

        # Normalize features
        if fit:
            df = self._fit_normalize(df)
        else:
            df = self._transform_normalize(df)

        # Handle infinities and NaNs
        df = self._handle_missing_values(df)

        return df

    def create_target(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0002) -> pd.DataFrame:
        """
        Create target variable for classification.

        Args:
            df: DataFrame with price data
            horizon: Number of bars to look ahead
            threshold: Minimum price change for BUY/SELL signal (0.02% = 2 pips)

        Returns:
            DataFrame with 'target' column (0=SELL, 1=HOLD, 2=BUY)
        """
        df = df.copy()

        # Calculate future return
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

        # Create 3-class target
        df[self.target_col] = 1  # Default: HOLD

        # BUY: Future return > threshold
        df.loc[df['future_return'] > threshold, self.target_col] = 2

        # SELL: Future return < -threshold
        df.loc[df['future_return'] < -threshold, self.target_col] = 0

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features."""
        lag_periods = [1, 2, 3, 5, 10]

        # Key columns to lag
        key_cols = ['returns', 'rsi_14', 'macd', 'macd_histogram', 'atr_14_pct']

        for col in key_cols:
            if col in df.columns:
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Get datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        elif 'time' in df.columns:
            dt = pd.to_datetime(df['time'])
        else:
            return df

        # Hour of day (cyclical encoding)
        df['hour'] = dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (cyclical encoding)
        df['day_of_week'] = dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Session indicators
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)  # London/NY overlap

        # Clean up
        df.drop(['hour', 'day_of_week'], axis=1, inplace=True, errors='ignore')

        return df

    def _identify_feature_columns(self, df: pd.DataFrame) -> None:
        """Identify which columns are features."""
        # Exclude non-feature columns
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'time', 'datetime', 'date', 'timestamp',
            'target', 'future_return',
            'symbol', 'instrument'
        ]

        self.feature_names = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

    def _fit_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit normalization parameters and transform."""
        for col in self.feature_names:
            if col in df.columns:
                self.feature_means[col] = df[col].mean()
                self.feature_stds[col] = df[col].std()

                # Avoid division by zero
                if self.feature_stds[col] < 1e-10:
                    self.feature_stds[col] = 1.0

                df[col] = (df[col] - self.feature_means[col]) / self.feature_stds[col]

        return df

    def _transform_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted parameters."""
        for col in self.feature_names:
            if col in df.columns and col in self.feature_means:
                df[col] = (df[col] - self.feature_means[col]) / self.feature_stds[col]

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing and infinite values."""
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill
        df = df.ffill()
        df = df.bfill()

        # Any remaining NaN -> 0
        df = df.fillna(0)

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Get feature matrix as numpy array."""
        available_features = [f for f in self.feature_names if f in df.columns]
        return df[available_features].values

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def save_config(self, path: str) -> None:
        """Save feature configuration to JSON."""
        config = {
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'target_col': self.target_col
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: str) -> None:
        """Load feature configuration from JSON."""
        with open(path, 'r') as f:
            config = json.load(f)

        self.feature_names = config['feature_names']
        self.feature_means = config['feature_means']
        self.feature_stds = config['feature_stds']
        self.target_col = config.get('target_col', 'target')

    def prepare_training_data(self, df: pd.DataFrame, horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.

        Args:
            df: Raw OHLCV DataFrame
            horizon: Prediction horizon in bars

        Returns:
            Tuple of (X features, y targets)
        """
        # Engineer features (fit normalization)
        df = self.engineer_features(df, fit=True)

        # Create targets
        df = self.create_target(df, horizon=horizon)

        # Drop rows with missing targets (end of dataset)
        df = df.dropna(subset=[self.target_col])

        # Get feature matrix and targets
        X = self.get_feature_matrix(df)
        y = df[self.target_col].values

        return X, y

    def prepare_inference_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare data for inference (no targets needed).

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Feature matrix as numpy array
        """
        # Engineer features (use fitted normalization)
        df = self.engineer_features(df, fit=False)

        # Get feature matrix
        X = self.get_feature_matrix(df)

        return X
