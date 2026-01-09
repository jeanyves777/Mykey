"""
Microstructure Features for High-Frequency Crypto Trading
==========================================================

Extracts market microstructure features including:
- Price action patterns (candlesticks, gaps, wicks)
- Volume analysis (volume profile, accumulation/distribution)
- Order flow approximations from OHLCV data
- Volatility clustering
- Mean reversion and momentum signals
"""

import numpy as np
import pandas as pd
from typing import Optional


class MicrostructureFeatures:
    """Compute market microstructure features from OHLCV data."""

    def __init__(self):
        """Initialize the MicrostructureFeatures calculator."""
        pass

    def compute_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute candlestick pattern features."""
        open_p, high, low, close = df['open'], df['high'], df['low'], df['close']

        # Basic candle metrics
        df['body_size'] = abs(close - open_p)
        df['body_size_pct'] = df['body_size'] / (high - low + 1e-10)
        df['upper_wick'] = high - np.maximum(open_p, close)
        df['lower_wick'] = np.minimum(open_p, close) - low
        df['upper_wick_pct'] = df['upper_wick'] / (high - low + 1e-10)
        df['lower_wick_pct'] = df['lower_wick'] / (high - low + 1e-10)

        # Candle direction
        df['is_bullish'] = (close > open_p).astype(int)
        df['is_bearish'] = (close < open_p).astype(int)
        df['is_doji'] = (df['body_size_pct'] < 0.1).astype(int)

        # Candle patterns
        # Hammer (bullish reversal)
        df['is_hammer'] = ((df['lower_wick'] > 2 * df['body_size']) &
                           (df['upper_wick'] < df['body_size'] * 0.3)).astype(int)

        # Shooting Star (bearish reversal)
        df['is_shooting_star'] = ((df['upper_wick'] > 2 * df['body_size']) &
                                   (df['lower_wick'] < df['body_size'] * 0.3)).astype(int)

        # Engulfing patterns
        df['bullish_engulfing'] = ((df['is_bullish'] == 1) &
                                    (df['is_bearish'].shift(1) == 1) &
                                    (close > open_p.shift(1)) &
                                    (open_p < close.shift(1))).astype(int)

        df['bearish_engulfing'] = ((df['is_bearish'] == 1) &
                                    (df['is_bullish'].shift(1) == 1) &
                                    (close < open_p.shift(1)) &
                                    (open_p > close.shift(1))).astype(int)

        # Morning/Evening Star approximation (3-candle patterns)
        df['morning_star'] = ((df['is_bearish'].shift(2) == 1) &
                               (df['is_doji'].shift(1) == 1) &
                               (df['is_bullish'] == 1) &
                               (close > (open_p.shift(2) + close.shift(2)) / 2)).astype(int)

        df['evening_star'] = ((df['is_bullish'].shift(2) == 1) &
                               (df['is_doji'].shift(1) == 1) &
                               (df['is_bearish'] == 1) &
                               (close < (open_p.shift(2) + close.shift(2)) / 2)).astype(int)

        # Three white soldiers / Three black crows
        bullish_streak = df['is_bullish'].rolling(3).sum()
        bearish_streak = df['is_bearish'].rolling(3).sum()
        df['three_white_soldiers'] = (bullish_streak == 3).astype(int)
        df['three_black_crows'] = (bearish_streak == 3).astype(int)

        return df

    def compute_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price action features."""
        open_p, high, low, close = df['open'], df['high'], df['low'], df['close']

        # Returns at different scales
        for period in [1, 2, 3, 5, 10, 20, 60]:
            df[f'return_{period}'] = close.pct_change(period)
            df[f'log_return_{period}'] = np.log(close / close.shift(period))

        # High-Low range
        df['hl_range'] = (high - low) / close
        df['hl_range_avg'] = df['hl_range'].rolling(20).mean()
        df['hl_range_ratio'] = df['hl_range'] / (df['hl_range_avg'] + 1e-10)

        # Gap analysis
        df['gap_up'] = (open_p > high.shift(1)).astype(int)
        df['gap_down'] = (open_p < low.shift(1)).astype(int)
        df['gap_size'] = (open_p - close.shift(1)) / close.shift(1)

        # Close position within bar
        df['close_position'] = (close - low) / (high - low + 1e-10)

        # Higher highs, lower lows
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)
        df['higher_close'] = (close > close.shift(1)).astype(int)

        # Swing detection
        df['swing_high'] = ((high > high.shift(1)) & (high > high.shift(-1))).astype(int)
        df['swing_low'] = ((low < low.shift(1)) & (low < low.shift(-1))).astype(int)

        # Support/Resistance proximity
        rolling_high_20 = high.rolling(20).max()
        rolling_low_20 = low.rolling(20).min()
        df['near_resistance'] = ((rolling_high_20 - close) / close < 0.005).astype(int)
        df['near_support'] = ((close - rolling_low_20) / close < 0.005).astype(int)

        # Breakout signals
        df['breakout_up'] = (close > rolling_high_20.shift(1)).astype(int)
        df['breakout_down'] = (close < rolling_low_20.shift(1)).astype(int)

        return df

    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        volume = df['volume']
        close = df['close']

        # Volume metrics
        df['volume_sma_10'] = volume.rolling(10).mean()
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)

        # Volume spikes
        df['volume_spike'] = (volume > df['volume_sma_20'] * 2).astype(int)
        df['volume_dry'] = (volume < df['volume_sma_20'] * 0.5).astype(int)

        # Volume momentum
        df['volume_momentum'] = volume.pct_change(5)

        # Price-Volume relationship
        df['pv_trend'] = (df['return_1'].rolling(5).corr(volume.pct_change())).fillna(0)

        # Volume-weighted metrics
        df['volume_close_position'] = (volume * df['close_position']).rolling(10).mean()

        # Accumulation/Distribution approximation
        clv = ((2 * close - df['low'] - df['high']) / (df['high'] - df['low'] + 1e-10))
        df['accumulation_distribution'] = (clv * volume).cumsum()
        df['ad_momentum'] = df['accumulation_distribution'].diff(10)

        # Money Flow Index components
        typical_price = (df['high'] + df['low'] + close) / 3
        raw_money_flow = typical_price * volume

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf_sum = positive_flow.rolling(14).sum()
        negative_mf_sum = negative_flow.rolling(14).sum()

        df['mfi'] = 100 - (100 / (1 + positive_mf_sum / (negative_mf_sum + 1e-10)))

        # Chaikin Money Flow
        df['cmf'] = ((2 * close - df['low'] - df['high']) / (df['high'] - df['low'] + 1e-10) *
                     volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)

        return df

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based features."""
        close = df['close']

        # Realized volatility at different scales
        for period in [5, 10, 20, 50]:
            returns = np.log(close / close.shift(1))
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252 * 24 * 60)

        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)

        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20'].rolling(20).std()

        # Parkinson volatility (more efficient estimator)
        log_hl = np.log(df['high'] / df['low'])
        df['parkinson_vol'] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252 * 24 * 60)

        # Garman-Klass volatility
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        df['gk_vol'] = np.sqrt(
            (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).rolling(20).mean()
        ) * np.sqrt(252 * 24 * 60)

        # Volatility regime
        vol_median = df['volatility_20'].rolling(100).median()
        df['high_vol_regime'] = (df['volatility_20'] > vol_median * 1.5).astype(int)
        df['low_vol_regime'] = (df['volatility_20'] < vol_median * 0.5).astype(int)

        # Volatility clustering (GARCH-like)
        df['vol_autocorr'] = df['volatility_10'].rolling(20).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=False
        ).fillna(0)

        return df

    def compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean reversion features."""
        close = df['close']

        # Z-score based on different lookback periods
        for period in [10, 20, 50]:
            mean = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'zscore_{period}'] = (close - mean) / (std + 1e-10)

        # Hurst exponent approximation (rolling)
        def hurst_approx(series):
            if len(series) < 10:
                return 0.5
            n = len(series)
            mean = series.mean()
            std = series.std()
            if std == 0:
                return 0.5
            cumdev = (series - mean).cumsum()
            r = cumdev.max() - cumdev.min()
            return np.log(r / std + 1e-10) / np.log(n)

        df['hurst_50'] = close.rolling(50).apply(hurst_approx, raw=True).fillna(0.5)

        # Mean reversion vs momentum indicator
        df['mr_momentum_indicator'] = df['zscore_20'] * df['return_5'] * -1

        # Distance from VWAP (mean reversion anchor)
        if 'vwap_50' in df.columns:
            df['vwap_zscore'] = (close - df['vwap_50']) / (df['volatility_20'] * close + 1e-10)

        # Oversold/Overbought extremes
        df['extreme_oversold'] = (df['zscore_20'] < -2).astype(int)
        df['extreme_overbought'] = (df['zscore_20'] > 2).astype(int)

        return df

    def compute_order_flow_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute order flow proxy features from OHLCV data."""
        open_p, high, low, close = df['open'], df['high'], df['low'], df['close']
        volume = df['volume']

        # Trade-based features if trades column available
        if 'trades' in df.columns and 'quote_volume' in df.columns:
            trades = df['trades'].fillna(0).replace(0, 1)  # Avoid div by zero
            df['avg_trade_size'] = df['quote_volume'] / (trades + 1e-10)
            df['trade_intensity'] = trades / (volume + 1e-10)
        else:
            # Estimate from volume/quote_volume if available
            if 'quote_volume' in df.columns:
                # Estimate trade size from average price
                avg_price = df['quote_volume'] / (volume + 1e-10)
                df['avg_trade_size'] = avg_price
                df['trade_intensity'] = 1.0  # Default
            else:
                df['avg_trade_size'] = 0
                df['trade_intensity'] = 0

        # Taker buy/sell ratio if available
        if 'taker_buy_base' in df.columns:
            df['taker_buy_ratio'] = df['taker_buy_base'] / (volume + 1e-10)
            df['taker_sell_ratio'] = 1 - df['taker_buy_ratio']
            df['order_imbalance'] = df['taker_buy_ratio'] - 0.5

            # Order imbalance momentum
            df['oi_momentum'] = df['order_imbalance'].diff(5)
            df['oi_sma'] = df['order_imbalance'].rolling(20).mean()
        else:
            df['taker_buy_ratio'] = 0.5
            df['taker_sell_ratio'] = 0.5
            df['order_imbalance'] = 0
            df['oi_momentum'] = 0
            df['oi_sma'] = 0

        # Buy/Sell pressure approximation from price action
        df['buy_pressure'] = (close - low) / (high - low + 1e-10)
        df['sell_pressure'] = (high - close) / (high - low + 1e-10)
        df['net_pressure'] = df['buy_pressure'] - df['sell_pressure']

        # Volume Delta approximation
        df['volume_delta'] = volume * df['net_pressure']
        df['cum_volume_delta'] = df['volume_delta'].cumsum()
        df['cvd_momentum'] = df['cum_volume_delta'].diff(10)

        return df

    def compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-based features."""
        if 'datetime' in df.columns:
            dt = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            # Create synthetic time index
            df['hour'] = 0
            df['minute'] = 0
            df['day_of_week'] = 0
            df['is_weekend'] = 0
            return df

        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['day_of_week'] = dt.dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Trading sessions (crypto trades 24/7 but activity varies)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 14) & (df['hour'] < 22)).astype(int)

        return df

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all microstructure features."""
        df = df.copy()

        df = self.compute_candlestick_patterns(df)
        df = self.compute_price_action(df)
        df = self.compute_volume_features(df)
        df = self.compute_volatility_features(df)
        df = self.compute_mean_reversion_features(df)
        df = self.compute_order_flow_proxies(df)
        df = self.compute_time_features(df)

        return df
