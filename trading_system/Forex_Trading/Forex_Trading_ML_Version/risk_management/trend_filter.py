"""
Trend Filter for Smart DCA System
=================================

Analyzes market conditions to determine if DCA entries are safe.
Prevents adding to losing positions when trend is strongly against us.

Key Features:
- ADX (Average Directional Index) for trend strength measurement
- EMA crossover for reversal detection
- Reversal candle pattern recognition
- Validates both DCA entries and initial ML trade entries
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    adx: float = 0.0
    adx_strength: str = "unknown"  # "strong", "moderate", "weak"
    trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_crossover: str = "none"  # "bullish", "bearish", "none"
    reversal_candle: bool = False
    reversal_type: str = "none"  # "bullish", "bearish", "none"
    can_dca: bool = False
    can_enter: bool = False
    reason: str = ""


class TrendFilter:
    """
    Filter trades and DCA based on market trend conditions.

    Logic:
    - If ADX > 30 (strong trend) and trend is AGAINST our position, DON'T DCA
    - If ADX < 20 (weak trend), safer to DCA (market consolidating)
    - Look for reversal signals (EMA cross, reversal candles) before DCA
    - Same logic applies to initial ML trade entries
    """

    def __init__(self, config=None):
        """
        Initialize trend filter.

        Args:
            config: TradingConfig or DCAConfig with trend filter settings
        """
        self.config = config

        # Get settings from config or use defaults
        if config and hasattr(config, 'dca'):
            dca = config.dca
            self.adx_strong = getattr(dca, 'adx_strong_trend', 30.0)
            self.adx_weak = getattr(dca, 'adx_weak_trend', 20.0)
            self.ema_fast_period = getattr(dca, 'ema_fast_period', 8)
            self.ema_slow_period = getattr(dca, 'ema_slow_period', 21)
            self.require_reversal = getattr(dca, 'require_reversal_candle', True)
            self.min_reversal_strength = getattr(dca, 'min_reversal_strength', 0.3)
            self.use_trend_filter = getattr(dca, 'use_trend_filter', True)
        elif config:
            self.adx_strong = getattr(config, 'adx_strong_trend', 30.0)
            self.adx_weak = getattr(config, 'adx_weak_trend', 20.0)
            self.ema_fast_period = getattr(config, 'ema_fast_period', 8)
            self.ema_slow_period = getattr(config, 'ema_slow_period', 21)
            self.require_reversal = getattr(config, 'require_reversal_candle', True)
            self.min_reversal_strength = getattr(config, 'min_reversal_strength', 0.3)
            self.use_trend_filter = getattr(config, 'use_trend_filter', True)
        else:
            # Defaults
            self.adx_strong = 30.0
            self.adx_weak = 20.0
            self.ema_fast_period = 8
            self.ema_slow_period = 21
            self.require_reversal = True
            self.min_reversal_strength = 0.3
            self.use_trend_filter = True

    def analyze_trend(self, df: pd.DataFrame, position_direction: str = None) -> TrendAnalysis:
        """
        Analyze current market trend.

        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            position_direction: 'BUY' or 'SELL' (for our existing position)

        Returns:
            TrendAnalysis with all indicators and recommendations
        """
        analysis = TrendAnalysis()

        if df is None or len(df) < max(self.ema_slow_period + 5, 20):
            analysis.reason = "Insufficient data for trend analysis"
            analysis.can_dca = True  # Allow DCA if no data (fallback)
            analysis.can_enter = True
            return analysis

        # Ensure we have required columns
        df = df.copy()
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                col_upper = col.capitalize()
                if col_upper in df.columns:
                    df[col] = df[col_upper]

        # Calculate ADX
        analysis.adx = self._calculate_adx(df)

        # Determine ADX strength
        if analysis.adx >= self.adx_strong:
            analysis.adx_strength = "strong"
        elif analysis.adx <= self.adx_weak:
            analysis.adx_strength = "weak"
        else:
            analysis.adx_strength = "moderate"

        # Calculate EMAs
        analysis.ema_fast = self._calculate_ema(df['close'], self.ema_fast_period)
        analysis.ema_slow = self._calculate_ema(df['close'], self.ema_slow_period)

        # Determine trend direction based on EMAs
        if analysis.ema_fast > analysis.ema_slow:
            analysis.trend_direction = "bullish"
        elif analysis.ema_fast < analysis.ema_slow:
            analysis.trend_direction = "bearish"
        else:
            analysis.trend_direction = "neutral"

        # Check for EMA crossover (recent)
        analysis.ema_crossover = self._check_ema_crossover(df)

        # Check for reversal candle
        reversal = self._check_reversal_candle(df)
        analysis.reversal_candle = reversal['is_reversal']
        analysis.reversal_type = reversal['type']

        # Determine if DCA is safe
        analysis.can_dca, analysis.can_enter, analysis.reason = self._evaluate_conditions(
            analysis, position_direction
        )

        return analysis

    def can_dca(self, df: pd.DataFrame, position_direction: str, dca_level: int = 1) -> Tuple[bool, str]:
        """
        Check if DCA is allowed based on market conditions.

        Args:
            df: OHLCV DataFrame
            position_direction: 'BUY' or 'SELL' (our position)
            dca_level: DCA level (1-4), higher levels require stronger reversal

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.use_trend_filter:
            return True, "Trend filter disabled"

        analysis = self.analyze_trend(df, position_direction)

        # Higher DCA levels require more confirmation
        if dca_level >= 3 and analysis.adx_strength == "strong":
            # DCA Level 3-4 requires reversal signal in strong trends
            if not analysis.reversal_candle:
                return False, f"DCA L{dca_level}: Strong trend (ADX={analysis.adx:.1f}), no reversal signal"

        return analysis.can_dca, analysis.reason

    def validate_entry(self, df: pd.DataFrame, signal_direction: str) -> Tuple[bool, str]:
        """
        Validate initial ML trade entry against current trend.

        Args:
            df: OHLCV DataFrame
            signal_direction: 'BUY' or 'SELL' (signal from ML)

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.use_trend_filter:
            return True, "Trend filter disabled"

        analysis = self.analyze_trend(df, None)

        # Check if signal aligns with trend
        is_aligned = False
        if signal_direction == 'BUY':
            is_aligned = (analysis.trend_direction == 'bullish' or
                         analysis.trend_direction == 'neutral' or
                         analysis.ema_crossover == 'bullish')
        else:  # SELL
            is_aligned = (analysis.trend_direction == 'bearish' or
                         analysis.trend_direction == 'neutral' or
                         analysis.ema_crossover == 'bearish')

        # Strong counter-trend
        if analysis.adx_strength == "strong" and not is_aligned:
            return False, f"Strong {analysis.trend_direction} trend (ADX={analysis.adx:.1f}), signal is {signal_direction}"

        # Moderate trend - allow with caution
        if analysis.adx_strength == "moderate" and not is_aligned:
            # Check for reversal signals
            if signal_direction == 'BUY' and analysis.reversal_type == 'bullish':
                return True, f"Bullish reversal detected in moderate trend"
            elif signal_direction == 'SELL' and analysis.reversal_type == 'bearish':
                return True, f"Bearish reversal detected in moderate trend"
            else:
                return False, f"Moderate {analysis.trend_direction} trend, no reversal for {signal_direction}"

        return True, f"Entry aligned with {analysis.adx_strength} {analysis.trend_direction} trend"

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength (not direction):
        - ADX > 30: Strong trend
        - ADX 20-30: Moderate trend
        - ADX < 20: Weak/no trend (consolidation)
        """
        if len(df) < period + 1:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = abs(high[1:] - close[:-1])
        tr3 = abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages using Wilder's method
        atr = self._wilder_smooth(tr, period)
        plus_di = 100 * self._wilder_smooth(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._wilder_smooth(minus_dm, period) / (atr + 1e-10)

        # DX and ADX
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * di_diff / (di_sum + 1e-10)

        adx = self._wilder_smooth(dx, period)

        return float(adx[-1]) if len(adx) > 0 else 0.0

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing method for ADX calculation."""
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result

    def _calculate_ema(self, series: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(series) < period:
            return float(series.iloc[-1]) if len(series) > 0 else 0.0
        ema = series.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])

    def _check_ema_crossover(self, df: pd.DataFrame) -> str:
        """
        Check for recent EMA crossover (bullish or bearish).

        Returns:
            'bullish' if fast crossed above slow recently
            'bearish' if fast crossed below slow recently
            'none' if no recent crossover
        """
        if len(df) < self.ema_slow_period + 3:
            return "none"

        close = df['close']
        ema_fast = close.ewm(span=self.ema_fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow_period, adjust=False).mean()

        # Check last 3 bars for crossover
        for i in range(-3, 0):
            if i - 1 < -len(ema_fast):
                continue

            prev_diff = ema_fast.iloc[i-1] - ema_slow.iloc[i-1]
            curr_diff = ema_fast.iloc[i] - ema_slow.iloc[i]

            # Bullish crossover: fast crosses above slow
            if prev_diff <= 0 and curr_diff > 0:
                return "bullish"
            # Bearish crossover: fast crosses below slow
            elif prev_diff >= 0 and curr_diff < 0:
                return "bearish"

        return "none"

    def _check_reversal_candle(self, df: pd.DataFrame) -> Dict:
        """
        Check for reversal candle patterns on recent bars.

        Looks for:
        - Bullish: Large bullish candle after downtrend (hammer, engulfing)
        - Bearish: Large bearish candle after uptrend (shooting star, engulfing)
        """
        result = {'is_reversal': False, 'type': 'none'}

        if len(df) < 5:
            return result

        # Get last few candles
        candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']

        prev_open = prev_candle['open']
        prev_close = prev_candle['close']

        # Calculate candle metrics
        body = abs(close_price - open_price)
        range_size = high_price - low_price
        body_ratio = body / (range_size + 1e-10)

        is_bullish = close_price > open_price
        is_bearish = close_price < open_price

        # Check for minimum body strength
        if body_ratio < self.min_reversal_strength:
            return result

        # Bullish reversal patterns
        if is_bullish:
            # Bullish engulfing: current body engulfs previous bearish candle
            if prev_close < prev_open and close_price > prev_open and open_price < prev_close:
                result['is_reversal'] = True
                result['type'] = 'bullish'
                return result

            # Strong bullish candle after decline (last 3 bars were down)
            if len(df) >= 5:
                recent_trend = df['close'].iloc[-5:-1]
                if recent_trend.iloc[-1] < recent_trend.iloc[0]:  # Recent decline
                    if body_ratio >= 0.5:  # Strong body
                        result['is_reversal'] = True
                        result['type'] = 'bullish'
                        return result

        # Bearish reversal patterns
        if is_bearish:
            # Bearish engulfing: current body engulfs previous bullish candle
            if prev_close > prev_open and close_price < prev_open and open_price > prev_close:
                result['is_reversal'] = True
                result['type'] = 'bearish'
                return result

            # Strong bearish candle after rally (last 3 bars were up)
            if len(df) >= 5:
                recent_trend = df['close'].iloc[-5:-1]
                if recent_trend.iloc[-1] > recent_trend.iloc[0]:  # Recent rally
                    if body_ratio >= 0.5:  # Strong body
                        result['is_reversal'] = True
                        result['type'] = 'bearish'
                        return result

        return result

    def _evaluate_conditions(self, analysis: TrendAnalysis,
                            position_direction: str) -> Tuple[bool, bool, str]:
        """
        Evaluate all conditions and determine if DCA/entry is allowed.

        Args:
            analysis: TrendAnalysis object
            position_direction: 'BUY' or 'SELL' (or None for new entry)

        Returns:
            Tuple of (can_dca, can_enter, reason)
        """
        # Weak trend = safe to DCA (market consolidating)
        if analysis.adx_strength == "weak":
            return True, True, f"Weak trend (ADX={analysis.adx:.1f}), safe to DCA"

        # No position direction = new entry validation
        if position_direction is None:
            return True, True, f"Trend: {analysis.trend_direction}, ADX: {analysis.adx:.1f}"

        # Check if trend is WITH or AGAINST our position
        is_trend_against = False
        if position_direction == 'BUY' and analysis.trend_direction == 'bearish':
            is_trend_against = True
        elif position_direction == 'SELL' and analysis.trend_direction == 'bullish':
            is_trend_against = True

        # Strong trend AGAINST us = DON'T DCA unless reversal signal
        if analysis.adx_strength == "strong" and is_trend_against:
            if analysis.reversal_candle or analysis.ema_crossover != 'none':
                reversal_signal = analysis.reversal_type or analysis.ema_crossover
                return True, True, f"Strong counter-trend BUT {reversal_signal} reversal detected"
            else:
                return False, True, f"Strong {analysis.trend_direction} trend (ADX={analysis.adx:.1f}) against {position_direction}, wait for reversal"

        # Moderate trend AGAINST us = DCA with caution
        if analysis.adx_strength == "moderate" and is_trend_against:
            if analysis.reversal_candle or analysis.ema_crossover != 'none':
                return True, True, f"Moderate counter-trend with reversal signal"
            else:
                return False, True, f"Moderate {analysis.trend_direction} trend against {position_direction}, prefer reversal"

        # Trend WITH us = safe to DCA
        return True, True, f"Trend ({analysis.trend_direction}) aligned with {position_direction}"

    def get_trend_summary(self, df: pd.DataFrame) -> str:
        """Get a one-line summary of current trend conditions."""
        analysis = self.analyze_trend(df)
        return (f"ADX={analysis.adx:.1f} ({analysis.adx_strength}), "
                f"Trend={analysis.trend_direction}, "
                f"EMA Cross={analysis.ema_crossover}, "
                f"Reversal={analysis.reversal_type}")


# Export
__all__ = ['TrendFilter', 'TrendAnalysis']
