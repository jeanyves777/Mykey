"""
Technical Features for High-Frequency Crypto Trading
=====================================================

Extracts technical analysis features from OHLCV data including:
- Moving averages and crossovers
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (ATR, Bollinger Bands)
- Volume indicators (OBV, VWAP, Volume Profile)
- Trend indicators (ADX, Ichimoku Cloud)
"""

import numpy as np
import pandas as pd
from typing import Optional


class TechnicalFeatures:
    """Compute technical analysis features from OHLCV data."""

    def __init__(self):
        """Initialize the TechnicalFeatures calculator."""
        pass

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
            raw=True
        )

    def compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute various moving averages."""
        close = df['close']

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = self.sma(close, period)

        # Exponential Moving Averages
        for period in [5, 10, 20, 50, 100]:
            df[f'ema_{period}'] = self.ema(close, period)

        # Moving Average Crossovers
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)

        # Price distance from MAs (normalized)
        for period in [10, 20, 50]:
            df[f'price_dist_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'price_dist_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}']

        return df

    def compute_rsi(self, df: pd.DataFrame, periods: list = [7, 14, 21]) -> pd.DataFrame:
        """Compute RSI for multiple periods."""
        close = df['close']

        for period in periods:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta.where(delta < 0, 0))

            avg_gain = gain.ewm(span=period, adjust=False, min_periods=1).mean()
            avg_loss = loss.ewm(span=period, adjust=False, min_periods=1).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # RSI zones
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)

        # RSI divergence signals
        df['rsi_momentum'] = df['rsi_14'].diff(3)

        return df

    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD indicator."""
        close = df['close']

        # Standard MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD crossovers
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) &
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) &
                                  (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

        # MACD momentum
        df['macd_momentum'] = df['macd_histogram'].diff()

        # Fast MACD for HFT
        exp1_fast = close.ewm(span=5, adjust=False).mean()
        exp2_fast = close.ewm(span=13, adjust=False).mean()
        df['macd_fast'] = exp1_fast - exp2_fast
        df['macd_signal_fast'] = df['macd_fast'].ewm(span=5, adjust=False).mean()

        return df

    def compute_stochastic(self, df: pd.DataFrame, periods: list = [14, 21]) -> pd.DataFrame:
        """Compute Stochastic Oscillator."""
        high, low, close = df['high'], df['low'], df['close']

        for period in periods:
            lowest_low = low.rolling(window=period, min_periods=1).min()
            highest_high = high.rolling(window=period, min_periods=1).max()

            df[f'stoch_k_{period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3, min_periods=1).mean()

            # Stochastic signals
            df[f'stoch_{period}_oversold'] = (df[f'stoch_k_{period}'] < 20).astype(int)
            df[f'stoch_{period}_overbought'] = (df[f'stoch_k_{period}'] > 80).astype(int)

        return df

    def compute_bollinger_bands(self, df: pd.DataFrame, periods: list = [20, 50]) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        close = df['close']

        for period in periods:
            sma = close.rolling(window=period, min_periods=1).mean()
            std = close.rolling(window=period, min_periods=1).std()

            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma

            # Price position within bands (0-1)
            df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

            # Band squeeze detection
            df[f'bb_squeeze_{period}'] = (df[f'bb_width_{period}'] <
                                          df[f'bb_width_{period}'].rolling(50).mean() * 0.5).astype(int)

        return df

    def compute_atr(self, df: pd.DataFrame, periods: list = [7, 14, 21]) -> pd.DataFrame:
        """Compute Average True Range."""
        high, low, close = df['high'], df['low'], df['close']

        for period in periods:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.ewm(span=period, adjust=False, min_periods=1).mean()

            # Normalized ATR (as percentage of price)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close * 100

        # ATR expansion/contraction
        df['atr_expansion'] = (df['atr_14'] > df['atr_14'].shift(1)).astype(int)

        return df

    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average Directional Index (ADX)."""
        high, low, close = df['high'], df['low'], df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth TR, +DM, -DM
        atr = tr.ewm(span=period, adjust=False, min_periods=1).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False, min_periods=1).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False, min_periods=1).mean() / (atr + 1e-10))

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(span=period, adjust=False, min_periods=1).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # ADX signals
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)
        df['di_cross'] = (df['plus_di'] > df['minus_di']).astype(int)

        return df

    def compute_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute On-Balance Volume."""
        close, volume = df['close'], df['volume']

        obv = (np.sign(close.diff()) * volume).cumsum()
        df['obv'] = obv
        df['obv_sma_20'] = obv.rolling(window=20, min_periods=1).mean()
        df['obv_momentum'] = obv.diff(5)

        # OBV divergence from price
        price_change = close.pct_change(20)
        obv_change = obv.pct_change(20)
        df['obv_divergence'] = np.sign(price_change) != np.sign(obv_change)

        return df

    def compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Rolling VWAP (since we don't have session boundaries in crypto)
        for period in [20, 50, 100]:
            cum_vol = df['volume'].rolling(window=period, min_periods=1).sum()
            cum_vol_price = (typical_price * df['volume']).rolling(window=period, min_periods=1).sum()
            df[f'vwap_{period}'] = cum_vol_price / (cum_vol + 1e-10)

            # Price distance from VWAP
            df[f'price_vwap_dist_{period}'] = (df['close'] - df[f'vwap_{period}']) / df[f'vwap_{period}']

        return df

    def compute_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Williams %R."""
        high, low, close = df['high'], df['low'], df['close']

        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()

        df['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        df['williams_r_oversold'] = (df['williams_r'] < -80).astype(int)
        df['williams_r_overbought'] = (df['williams_r'] > -20).astype(int)

        return df

    def compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
        mad = typical_price.rolling(window=period, min_periods=1).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        df['cci_oversold'] = (df['cci'] < -100).astype(int)
        df['cci_overbought'] = (df['cci'] > 100).astype(int)

        return df

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute various momentum indicators."""
        close = df['close']

        # Rate of Change (ROC)
        for period in [5, 10, 20, 50]:
            df[f'roc_{period}'] = close.pct_change(period) * 100

        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = close - close.shift(period)

        # Price acceleration
        df['price_acceleration'] = df['roc_10'].diff()

        # Trend Strength
        df['trend_strength'] = abs(df['roc_20']) / (df['atr_pct_14'] + 1e-10)

        return df

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features."""
        df = df.copy()

        df = self.compute_moving_averages(df)
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_stochastic(df)
        df = self.compute_bollinger_bands(df)
        df = self.compute_atr(df)
        df = self.compute_adx(df)
        df = self.compute_obv(df)
        df = self.compute_vwap(df)
        df = self.compute_williams_r(df)
        df = self.compute_cci(df)
        df = self.compute_momentum(df)

        return df
