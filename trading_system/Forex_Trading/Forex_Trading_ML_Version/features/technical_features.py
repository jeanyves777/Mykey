"""
Technical Features for Forex ML System
======================================

Calculates 50+ technical indicators for ML model input.
"""

import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Suppress pandas fragmentation warnings
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')


class TechnicalFeatures:
    """Calculate technical analysis features."""

    def __init__(self, config=None):
        """Initialize with optional config."""
        self.config = config
        # Default periods if no config
        self.ma_periods = [5, 10, 20, 50, 100, 200] if config is None else config.ma_periods
        self.rsi_periods = [7, 14, 21] if config is None else config.rsi_periods

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical features."""
        # Use copy and consolidate to prevent fragmentation
        df = df.copy()

        # Collect new features in a dict, then assign at once
        features = {}

        # Price-based features
        df = self._calculate_returns(df)
        df = self._calculate_moving_averages(df)
        df = self._calculate_ma_crossovers(df)

        # Momentum indicators
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_stochastic(df)
        df = self._calculate_roc(df)
        df = self._calculate_momentum(df)

        # Volatility indicators
        df = self._calculate_atr(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_volatility(df)

        # Trend indicators
        df = self._calculate_adx(df)
        df = self._calculate_trend_strength(df)

        # Price patterns
        df = self._calculate_candle_patterns(df)
        df = self._calculate_support_resistance(df)

        # Volume features (if available)
        if 'volume' in df.columns:
            df = self._calculate_volume_features(df)

        # Consolidate DataFrame to reduce fragmentation
        return pd.DataFrame(df)

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and log returns."""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Multi-period returns
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)

        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages."""
        for period in self.ma_periods:
            # Simple Moving Average
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            # Exponential Moving Average
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            # Weighted Moving Average
            weights = np.arange(1, period + 1)
            df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )

        # Price distance from MAs (normalized)
        for period in self.ma_periods:
            df[f'price_sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'price_ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

        return df

    def _calculate_ma_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MA crossover signals."""
        # EMA crossovers
        df['ema_9_21_cross'] = np.where(
            df['ema_5'].shift(1) < df['ema_20'].shift(1),
            np.where(df['ema_5'] > df['ema_20'], 1, 0),
            np.where(df['ema_5'] < df['ema_20'], -1, 0)
        )

        df['ema_20_50_cross'] = np.where(
            df['ema_20'].shift(1) < df['ema_50'].shift(1),
            np.where(df['ema_20'] > df['ema_50'], 1, 0),
            np.where(df['ema_20'] < df['ema_50'], -1, 0)
        )

        # MA alignment score (-1 to 1)
        df['ma_alignment'] = (
            (df['ema_5'] > df['ema_10']).astype(int) +
            (df['ema_10'] > df['ema_20']).astype(int) +
            (df['ema_20'] > df['ema_50']).astype(int) +
            (df['ema_50'] > df['ema_100']).astype(int)
        ) / 4 * 2 - 1  # Normalize to -1 to 1

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI for multiple periods."""
        for period in self.rsi_periods:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # RSI momentum (change in RSI)
            df[f'rsi_{period}_momentum'] = df[f'rsi_{period}'].diff()

            # RSI zones
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and related features."""
        # Standard MACD (12, 26, 9)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD momentum
        df['macd_momentum'] = df['macd'].diff()
        df['macd_hist_momentum'] = df['macd_histogram'].diff()

        # MACD crossovers
        df['macd_cross'] = np.where(
            df['macd'].shift(1) < df['macd_signal'].shift(1),
            np.where(df['macd'] > df['macd_signal'], 1, 0),
            np.where(df['macd'] < df['macd_signal'], -1, 0)
        )

        # MACD zero line cross
        df['macd_zero_cross'] = np.where(
            df['macd'].shift(1) < 0,
            np.where(df['macd'] > 0, 1, 0),
            np.where(df['macd'] < 0, -1, 0)
        )

        # Normalized MACD
        df['macd_normalized'] = df['macd'] / df['close'] * 100

        return df

    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        period = 14
        smooth_k = 3
        smooth_d = 3

        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        # Fast %K
        df['stoch_k_raw'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        # Slow %K (smoothed)
        df['stoch_k'] = df['stoch_k_raw'].rolling(window=smooth_k).mean()

        # %D (signal)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth_d).mean()

        # Stochastic crossover
        df['stoch_cross'] = np.where(
            df['stoch_k'].shift(1) < df['stoch_d'].shift(1),
            np.where(df['stoch_k'] > df['stoch_d'], 1, 0),
            np.where(df['stoch_k'] < df['stoch_d'], -1, 0)
        )

        return df

    def _calculate_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Rate of Change."""
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

        return df

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Williams %R
        period = 14
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)

        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range."""
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()

            # ATR percent (normalized)
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close'] * 100

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        period = 20
        std = 2

        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std)

        # Bollinger Band width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Bollinger %B (position within bands)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Distance from bands
        df['bb_upper_dist'] = (df['bb_upper'] - df['close']) / df['close']
        df['bb_lower_dist'] = (df['close'] - df['bb_lower']) / df['close']

        return df

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility measures."""
        # Historical volatility (standard deviation of returns)
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()

        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_avg'] = df['hl_range'].rolling(window=14).mean()

        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

        return df

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        period = 14

        # +DM and -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()

        df['plus_dm'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'], 0
        )
        df['minus_dm'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'], 0
        )

        # True Range
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / atr)
        minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / atr)

        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()

        # Trend strength
        df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)

        # Clean up temp columns
        df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm'], axis=1, inplace=True)

        return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend strength indicators."""
        # Price vs MAs (trend direction)
        df['price_above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
        df['price_above_sma_100'] = (df['close'] > df['sma_100']).astype(int)

        # Trend score
        df['trend_score'] = (
            df['price_above_sma_20'] +
            df['price_above_sma_50'] +
            df['price_above_sma_100'] +
            (df['ema_5'] > df['ema_20']).astype(int) +
            (df['ema_20'] > df['ema_50']).astype(int)
        ) / 5 * 2 - 1  # Normalize to -1 to 1

        # Consecutive closes in same direction
        df['up_streak'] = df['returns'].gt(0).astype(int).groupby(
            df['returns'].le(0).astype(int).cumsum()
        ).cumsum()
        df['down_streak'] = df['returns'].lt(0).astype(int).groupby(
            df['returns'].ge(0).astype(int).cumsum()
        ).cumsum()

        return df

    def _calculate_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick pattern features."""
        # Candle body and shadow sizes
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']

        # Candle type
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_doji'] = (df['body_size'] < df['total_range'] * 0.1).astype(int)

        # Body to range ratio
        df['body_range_ratio'] = df['body_size'] / (df['total_range'] + 1e-10)

        # Shadow ratios
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['total_range'] + 1e-10)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['total_range'] + 1e-10)

        # Hammer pattern (long lower shadow, small body at top)
        df['hammer'] = (
            (df['lower_shadow'] > df['body_size'] * 2) &
            (df['upper_shadow'] < df['body_size'] * 0.5)
        ).astype(int)

        # Shooting star (long upper shadow, small body at bottom)
        df['shooting_star'] = (
            (df['upper_shadow'] > df['body_size'] * 2) &
            (df['lower_shadow'] < df['body_size'] * 0.5)
        ).astype(int)

        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous red
            (df['close'] > df['open']) &  # Current green
            (df['close'] > df['open'].shift(1)) &  # Close above prev open
            (df['open'] < df['close'].shift(1))  # Open below prev close
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous green
            (df['close'] < df['open']) &  # Current red
            (df['close'] < df['open'].shift(1)) &  # Close below prev open
            (df['open'] > df['close'].shift(1))  # Open above prev close
        ).astype(int)

        return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support/resistance levels."""
        # Recent highs and lows
        for period in [10, 20, 50]:
            df[f'highest_{period}'] = df['high'].rolling(window=period).max()
            df[f'lowest_{period}'] = df['low'].rolling(window=period).min()

            # Distance from recent high/low
            df[f'dist_from_high_{period}'] = (df[f'highest_{period}'] - df['close']) / df['close']
            df[f'dist_from_low_{period}'] = (df['close'] - df[f'lowest_{period}']) / df['close']

        # Pivot points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))

        # Distance from pivot levels
        df['dist_from_pivot'] = (df['close'] - df['pivot']) / df['pivot']
        df['dist_from_r1'] = (df['r1'] - df['close']) / df['close']
        df['dist_from_s1'] = (df['close'] - df['s1']) / df['close']

        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_momentum'] = df['obv'] - df['obv_sma']

        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['returns']).fillna(0).cumsum()

        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        period = 14
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        df['mfi'] = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))

        # Volume-weighted average price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']

        return df
