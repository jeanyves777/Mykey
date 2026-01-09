"""
Master Momentum Signal Generator
================================

Fast entry signals when trend and momentum ALIGN.
Bypasses heavy ML consensus for quick scalping with DCA.

Entry Logic:
1. Momentum spike detected (>threshold in 2-3 bars)
2. Trend aligns (EMA8 vs EMA21 matches momentum direction)
3. RSI not extreme (30-70 range)
4. ADX shows trend exists (>15)

>>> When all 4 align = ENTER IMMEDIATELY <<<
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class MasterMomentumSignal:
    """Fast momentum-based signal generator."""

    def __init__(self, config=None):
        """
        Initialize momentum signal generator.

        Args:
            config: TradingConfig with MasterMomentumConfig
        """
        self.config = config

        # Default settings (overridden by config if provided)
        self.momentum_period = 3
        self.momentum_threshold = 0.25  # 0.25% move
        self.ema_fast = 8
        self.ema_slow = 21
        self.rsi_period = 14
        self.rsi_max_buy = 70.0
        self.rsi_min_sell = 30.0
        self.adx_period = 14
        self.min_adx = 15.0
        self.cooldown_bars = 5

        # Load from config if provided
        if config and hasattr(config, 'momentum'):
            mom = config.momentum
            self.momentum_period = mom.momentum_period
            self.momentum_threshold = mom.momentum_threshold
            self.ema_fast = mom.ema_fast_period
            self.ema_slow = mom.ema_slow_period
            self.rsi_period = mom.rsi_period
            self.rsi_max_buy = mom.rsi_max_for_buy
            self.rsi_min_sell = mom.rsi_min_for_sell
            self.adx_period = mom.adx_period
            self.min_adx = mom.min_adx
            self.cooldown_bars = mom.cooldown_bars

        # Signal cooldown tracking
        self.last_signal_bar: Dict[str, int] = {}
        self.bar_count: Dict[str, int] = {}

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[Optional[str], float, str]:
        """
        Generate fast momentum signal when trend + momentum align.

        Args:
            symbol: Trading pair
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (signal, confidence, reason)
            signal: 'BUY', 'SELL', or None
            confidence: 0.0-1.0 (always 0.80+ for momentum signals)
            reason: Explanation of signal
        """
        if df is None or len(df) < max(self.ema_slow + 10, 50):
            return None, 0.0, "Insufficient data"

        # Track bar count for cooldown
        if symbol not in self.bar_count:
            self.bar_count[symbol] = 0
        self.bar_count[symbol] += 1

        # Check cooldown
        if symbol in self.last_signal_bar:
            bars_since = self.bar_count[symbol] - self.last_signal_bar[symbol]
            if bars_since < self.cooldown_bars:
                return None, 0.0, f"Cooldown ({bars_since}/{self.cooldown_bars} bars)"

        try:
            # Calculate indicators
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # 1. MOMENTUM: Check for spike in last N bars
            momentum, mom_direction = self._calculate_momentum(close)

            if abs(momentum) < self.momentum_threshold:
                return None, 0.0, f"No momentum spike ({momentum:+.2f}% < {self.momentum_threshold}%)"

            # 2. TREND: Check EMA alignment
            ema_fast = self._ema(close, self.ema_fast)
            ema_slow = self._ema(close, self.ema_slow)
            trend_bullish = ema_fast[-1] > ema_slow[-1]

            # Trend must align with momentum
            if mom_direction == 'UP' and not trend_bullish:
                return None, 0.0, f"Momentum UP but trend bearish (EMA8 < EMA21)"
            if mom_direction == 'DOWN' and trend_bullish:
                return None, 0.0, f"Momentum DOWN but trend bullish (EMA8 > EMA21)"

            # 3. RSI: Not extreme
            rsi = self._calculate_rsi(close, self.rsi_period)

            if mom_direction == 'UP' and rsi > self.rsi_max_buy:
                return None, 0.0, f"RSI overbought ({rsi:.1f} > {self.rsi_max_buy})"
            if mom_direction == 'DOWN' and rsi < self.rsi_min_sell:
                return None, 0.0, f"RSI oversold ({rsi:.1f} < {self.rsi_min_sell})"

            # 4. ADX: Trend must exist
            adx = self._calculate_adx(high, low, close, self.adx_period)

            if adx < self.min_adx:
                return None, 0.0, f"Weak trend (ADX {adx:.1f} < {self.min_adx})"

            # ALL CONDITIONS MET - GENERATE SIGNAL
            if mom_direction == 'UP':
                signal = 'BUY'
                confidence = min(0.95, 0.80 + (adx / 100))  # Higher ADX = higher confidence
                reason = f"MOMENTUM BUY: Mom={momentum:+.2f}%, EMA8>EMA21, RSI={rsi:.1f}, ADX={adx:.1f}"
            else:
                signal = 'SELL'
                confidence = min(0.95, 0.80 + (adx / 100))
                reason = f"MOMENTUM SELL: Mom={momentum:+.2f}%, EMA8<EMA21, RSI={rsi:.1f}, ADX={adx:.1f}"

            # Record signal for cooldown
            self.last_signal_bar[symbol] = self.bar_count[symbol]

            return signal, confidence, reason

        except Exception as e:
            return None, 0.0, f"Error: {str(e)}"

    def _calculate_momentum(self, close: np.ndarray) -> Tuple[float, str]:
        """Calculate momentum over N bars."""
        if len(close) < self.momentum_period + 1:
            return 0.0, 'NONE'

        # Momentum = % change over period
        current = close[-1]
        past = close[-(self.momentum_period + 1)]

        momentum = ((current - past) / past) * 100

        direction = 'UP' if momentum > 0 else 'DOWN' if momentum < 0 else 'NONE'
        return momentum, direction

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        if len(close) < period + 1:
            return 0.0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # +DM and -DM
        plus_dm = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0
        )
        minus_dm = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0
        )

        # Smooth with EMA
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / np.where(atr > 0, atr, 1)
        minus_di = 100 * self._ema(minus_dm, period) / np.where(atr > 0, atr, 1)

        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
        adx = self._ema(dx, period)

        return float(adx[-1]) if len(adx) > 0 else 0.0

    def get_trend_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of current trend indicators."""
        if df is None or len(df) < 50:
            return {'error': 'Insufficient data'}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        ema_fast = self._ema(close, self.ema_fast)
        ema_slow = self._ema(close, self.ema_slow)
        rsi = self._calculate_rsi(close, self.rsi_period)
        adx = self._calculate_adx(high, low, close, self.adx_period)
        momentum, direction = self._calculate_momentum(close)

        return {
            'momentum': momentum,
            'momentum_direction': direction,
            'ema_fast': float(ema_fast[-1]),
            'ema_slow': float(ema_slow[-1]),
            'trend': 'bullish' if ema_fast[-1] > ema_slow[-1] else 'bearish',
            'rsi': rsi,
            'adx': adx
        }

    def reset_cooldown(self, symbol: str = None) -> None:
        """Reset cooldown for symbol(s)."""
        if symbol:
            if symbol in self.last_signal_bar:
                del self.last_signal_bar[symbol]
        else:
            self.last_signal_bar.clear()
