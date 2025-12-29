"""
Microstructure Features for Forex ML System
============================================

Market microstructure analysis features for high-frequency signals.
"""

import pandas as pd
import numpy as np
from typing import Optional


class MicrostructureFeatures:
    """Calculate market microstructure features."""

    def __init__(self, config=None):
        """Initialize with optional config."""
        self.config = config

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all microstructure features."""
        df = df.copy()

        df = self._calculate_price_momentum(df)
        df = self._calculate_volatility_regime(df)
        df = self._calculate_mean_reversion(df)
        df = self._calculate_price_acceleration(df)
        df = self._calculate_relative_strength(df)
        df = self._calculate_market_regime(df)

        return df

    def _calculate_price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price momentum features."""
        # Short-term momentum (1-5 bars)
        df['momentum_1'] = df['close'].diff(1)
        df['momentum_3'] = df['close'].diff(3)
        df['momentum_5'] = df['close'].diff(5)

        # Momentum percentile (where is current momentum relative to recent history)
        for period in [1, 3, 5]:
            df[f'momentum_{period}_percentile'] = df[f'momentum_{period}'].rolling(
                window=100
            ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

        # Price velocity (rate of price change)
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1)
        df['price_velocity_sma'] = df['price_velocity'].rolling(window=10).mean()

        # Momentum divergence (price momentum vs volume momentum if available)
        if 'volume' in df.columns:
            df['volume_momentum'] = df['volume'].pct_change(5)
            df['momentum_divergence'] = df['momentum_5'] * df['volume_momentum']

        return df

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility regime features."""
        # Short-term vs long-term volatility
        df['vol_short'] = df['close'].pct_change().rolling(window=10).std()
        df['vol_long'] = df['close'].pct_change().rolling(window=50).std()
        df['vol_ratio'] = df['vol_short'] / (df['vol_long'] + 1e-10)

        # Volatility expansion/contraction
        df['vol_expanding'] = (df['vol_ratio'] > 1).astype(int)
        df['vol_contracting'] = (df['vol_ratio'] < 1).astype(int)

        # Volatility percentile
        df['vol_percentile'] = df['vol_short'].rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Parkinson volatility (high-low based)
        df['parkinson_vol'] = np.sqrt(
            np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        ).rolling(window=20).mean()

        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        df['gk_volatility'] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=20).mean()

        return df

    def _calculate_mean_reversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion features."""
        # Z-score of price relative to moving averages
        for period in [20, 50, 100]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'zscore_{period}'] = (df['close'] - sma) / (std + 1e-10)

        # Distance from rolling mean (percentage)
        df['dist_from_mean_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()

        # Mean reversion signal (extreme z-scores)
        df['mean_reversion_long'] = (df['zscore_20'] < -2).astype(int)
        df['mean_reversion_short'] = (df['zscore_20'] > 2).astype(int)

        # Hurst exponent proxy (trending vs mean-reverting)
        returns = df['close'].pct_change()
        df['autocorr_1'] = returns.rolling(window=50).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        df['autocorr_5'] = returns.rolling(window=50).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )

        return df

    def _calculate_price_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price acceleration features."""
        # First derivative (velocity)
        df['velocity'] = df['close'].diff()

        # Second derivative (acceleration)
        df['acceleration'] = df['velocity'].diff()

        # Jerk (rate of change of acceleration)
        df['jerk'] = df['acceleration'].diff()

        # Smoothed acceleration
        df['acceleration_sma'] = df['acceleration'].rolling(window=5).mean()

        # Acceleration sign changes (inflection points)
        df['acceleration_sign'] = np.sign(df['acceleration'])
        df['inflection_point'] = (df['acceleration_sign'] != df['acceleration_sign'].shift(1)).astype(int)

        return df

    def _calculate_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative strength features."""
        # Price relative to recent range
        rolling_high = df['high'].rolling(window=20).max()
        rolling_low = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

        # Price relative to VWAP (if volume available)
        if 'volume' in df.columns:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['rolling_vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['price_vs_vwap'] = (df['close'] - df['rolling_vwap']) / df['rolling_vwap']

        # Up/Down day ratio
        df['up_day'] = (df['close'] > df['open']).astype(int)
        df['up_day_ratio'] = df['up_day'].rolling(window=20).mean()

        # Average up move vs average down move
        df['up_move'] = np.where(df['close'] > df['open'], df['close'] - df['open'], 0)
        df['down_move'] = np.where(df['close'] < df['open'], df['open'] - df['close'], 0)
        df['avg_up_move'] = df['up_move'].rolling(window=20).mean()
        df['avg_down_move'] = df['down_move'].rolling(window=20).mean()
        df['up_down_ratio'] = df['avg_up_move'] / (df['avg_down_move'] + 1e-10)

        return df

    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features."""
        # Trend regime based on MA slopes
        df['sma_20_slope'] = df['close'].rolling(20).mean().diff(5) / 5
        df['sma_50_slope'] = df['close'].rolling(50).mean().diff(10) / 10

        # Regime: trending up, trending down, or ranging
        df['uptrend'] = (
            (df['sma_20_slope'] > 0) &
            (df['close'] > df['close'].rolling(20).mean())
        ).astype(int)

        df['downtrend'] = (
            (df['sma_20_slope'] < 0) &
            (df['close'] < df['close'].rolling(20).mean())
        ).astype(int)

        df['ranging'] = (~df['uptrend'].astype(bool) & ~df['downtrend'].astype(bool)).astype(int)

        # Trend duration (bars in current trend)
        df['trend_duration'] = (
            df['uptrend'].astype(int).groupby(
                (~df['uptrend'].astype(bool)).cumsum()
            ).cumsum() -
            df['downtrend'].astype(int).groupby(
                (~df['downtrend'].astype(bool)).cumsum()
            ).cumsum()
        )

        # Range detection (Bollinger Band width based)
        bb_width = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        df['bb_squeeze'] = (bb_width < bb_width.rolling(100).quantile(0.25)).astype(int)

        # Breakout potential (squeeze + volume increase)
        if 'volume' in df.columns:
            df['breakout_potential'] = (
                df['bb_squeeze'] &
                (df['volume'] > df['volume'].rolling(20).mean())
            ).astype(int)

        return df
