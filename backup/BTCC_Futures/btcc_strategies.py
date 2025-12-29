"""
BTCC Futures Trading Strategies
===============================
Trading strategies for BTCC Futures platform.

Strategies:
- SCALPING_MOMENTUM: RSI + MACD + EMA for quick scalps
- RSI_REVERSAL: RSI oversold/overbought reversals
- BREAKOUT: Price breakout with volume confirmation
- EMA_CROSSOVER: Classic EMA crossover strategy
- BOLLINGER_SQUEEZE: Volatility breakout from Bollinger squeeze
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TechnicalIndicators:
    """Technical indicator calculations"""

    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)

        multiplier = 2 / (period + 1)
        ema_values = [np.nan] * (period - 1)

        # First EMA is SMA
        ema_values.append(sum(data[:period]) / period)

        for i in range(period, len(data)):
            ema_values.append(data[i] * multiplier + ema_values[-1] * (1 - multiplier))

        return ema_values

    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)

        sma_values = [np.nan] * (period - 1)
        for i in range(period - 1, len(data)):
            sma_values.append(sum(data[i - period + 1:i + 1]) / period)

        return sma_values

    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(data) < period + 1:
            return [np.nan] * len(data)

        deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        rsi_values = [np.nan] * period

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)

        macd_line = [f - s if not (np.isnan(f) or np.isnan(s)) else np.nan
                     for f, s in zip(ema_fast, ema_slow)]

        # Filter out NaN for signal line calculation
        valid_macd = [m for m in macd_line if not np.isnan(m)]
        signal_line_raw = TechnicalIndicators.ema(valid_macd, signal)

        # Pad signal line to match original length
        signal_line = [np.nan] * (len(macd_line) - len(signal_line_raw)) + signal_line_raw

        histogram = [m - s if not (np.isnan(m) or np.isnan(s)) else np.nan
                     for m, s in zip(macd_line, signal_line)]

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data: List[float], period: int = 20, std_mult: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands (middle, upper, lower)"""
        middle = TechnicalIndicators.sma(data, period)

        upper = []
        lower = []

        for i in range(len(data)):
            if np.isnan(middle[i]):
                upper.append(np.nan)
                lower.append(np.nan)
            else:
                std = np.std(data[max(0, i - period + 1):i + 1])
                upper.append(middle[i] + std_mult * std)
                lower.append(middle[i] - std_mult * std)

        return middle, upper, lower

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate Average True Range"""
        if len(closes) < 2:
            return [np.nan] * len(closes)

        true_ranges = [highs[0] - lows[0]]

        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)

        return TechnicalIndicators.sma(true_ranges, period)

    @staticmethod
    def keltner_channel(highs: List[float], lows: List[float], closes: List[float],
                        period: int = 20, mult: float = 1.5) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Keltner Channel"""
        middle = TechnicalIndicators.ema(closes, period)
        atr = TechnicalIndicators.atr(highs, lows, closes, period)

        upper = [m + mult * a if not (np.isnan(m) or np.isnan(a)) else np.nan
                 for m, a in zip(middle, atr)]
        lower = [m - mult * a if not (np.isnan(m) or np.isnan(a)) else np.nan
                 for m, a in zip(middle, atr)]

        return middle, upper, lower


class BTCCStrategy:
    """Base class for BTCC trading strategies"""

    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        """
        Evaluate strategy and return signal if any.

        Returns:
            Dict with 'direction' (1=buy, 2=sell), 'strength', 'sl', 'tp'
            or None if no signal
        """
        raise NotImplementedError


class ScalpingMomentumStrategy(BTCCStrategy):
    """
    Scalping Momentum Strategy
    --------------------------
    Combines RSI, MACD, and EMA for quick scalping trades.

    Entry conditions:
    - Long: RSI < oversold + MACD histogram turning positive + price above EMA
    - Short: RSI > overbought + MACD histogram turning negative + price below EMA
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ema_fast': 9,
            'ema_slow': 21,
            'volume_threshold': 1.5,
        }
        if params:
            default_params.update(params)
        super().__init__('SCALPING_MOMENTUM', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < 50:
            return None

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        # Calculate indicators
        rsi = TechnicalIndicators.rsi(closes, self.params['rsi_period'])
        macd, signal, hist = TechnicalIndicators.macd(
            closes,
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        ema_fast = TechnicalIndicators.ema(closes, self.params['ema_fast'])
        ema_slow = TechnicalIndicators.ema(closes, self.params['ema_slow'])

        # Get current values
        current_rsi = rsi[-1]
        current_hist = hist[-1]
        prev_hist = hist[-2] if len(hist) > 1 else 0
        current_ema_fast = ema_fast[-1]
        current_ema_slow = ema_slow[-1]

        # Check volume
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ok = current_volume > avg_volume * self.params['volume_threshold']

        if np.isnan(current_rsi) or np.isnan(current_hist):
            return None

        # Long signal
        if (current_rsi < self.params['rsi_oversold'] and
            current_hist > prev_hist and current_hist > 0 and
            current_ema_fast > current_ema_slow and
            volume_ok):

            return {
                'direction': 1,
                'strength': min(1.0, (self.params['rsi_oversold'] - current_rsi) / 10),
                'reason': f"RSI oversold ({current_rsi:.1f}), MACD bullish, EMA bullish",
            }

        # Short signal
        if (current_rsi > self.params['rsi_overbought'] and
            current_hist < prev_hist and current_hist < 0 and
            current_ema_fast < current_ema_slow and
            volume_ok):

            return {
                'direction': 2,
                'strength': min(1.0, (current_rsi - self.params['rsi_overbought']) / 10),
                'reason': f"RSI overbought ({current_rsi:.1f}), MACD bearish, EMA bearish",
            }

        return None


class RSIReversalStrategy(BTCCStrategy):
    """
    RSI Reversal Strategy
    ---------------------
    Trades RSI oversold/overbought reversals with confirmation.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'confirmation_bars': 2,
            'min_reversal_pct': 0.5,
        }
        if params:
            default_params.update(params)
        super().__init__('RSI_REVERSAL', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < 30:
            return None

        closes = [c.close for c in candles]
        rsi = TechnicalIndicators.rsi(closes, self.params['rsi_period'])

        if len(rsi) < self.params['confirmation_bars'] + 1:
            return None

        current_rsi = rsi[-1]
        prev_rsi = rsi[-2]

        if np.isnan(current_rsi) or np.isnan(prev_rsi):
            return None

        # Check for reversal from oversold
        was_oversold = any(r < self.params['rsi_oversold']
                          for r in rsi[-(self.params['confirmation_bars'] + 1):-1]
                          if not np.isnan(r))

        if was_oversold and current_rsi > self.params['rsi_oversold'] and current_rsi > prev_rsi:
            return {
                'direction': 1,
                'strength': min(1.0, (current_rsi - self.params['rsi_oversold']) / 15),
                'reason': f"RSI reversal from oversold ({prev_rsi:.1f} -> {current_rsi:.1f})",
            }

        # Check for reversal from overbought
        was_overbought = any(r > self.params['rsi_overbought']
                            for r in rsi[-(self.params['confirmation_bars'] + 1):-1]
                            if not np.isnan(r))

        if was_overbought and current_rsi < self.params['rsi_overbought'] and current_rsi < prev_rsi:
            return {
                'direction': 2,
                'strength': min(1.0, (self.params['rsi_overbought'] - current_rsi) / 15),
                'reason': f"RSI reversal from overbought ({prev_rsi:.1f} -> {current_rsi:.1f})",
            }

        return None


class BreakoutStrategy(BTCCStrategy):
    """
    Breakout Strategy
    ----------------
    Trades breakouts from consolidation ranges.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'lookback_period': 20,
            'breakout_threshold': 1.5,
            'volume_confirmation': True,
        }
        if params:
            default_params.update(params)
        super().__init__('BREAKOUT', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < self.params['lookback_period'] + 5:
            return None

        lookback = self.params['lookback_period']
        recent_candles = candles[-lookback - 1:-1]  # Exclude current
        current_candle = candles[-1]

        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        resistance = max(highs)
        support = min(lows)

        # Calculate ATR for threshold
        atr = TechnicalIndicators.atr(
            [c.high for c in candles],
            [c.low for c in candles],
            closes,
            14
        )[-1]

        if np.isnan(atr):
            return None

        # Check volume
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ok = not self.params['volume_confirmation'] or current_volume > avg_volume * 1.5

        # Breakout above resistance
        if current_price > resistance + atr * 0.5 and volume_ok:
            return {
                'direction': 1,
                'strength': min(1.0, (current_price - resistance) / (atr * 2)),
                'reason': f"Breakout above {resistance:.2f} (ATR: {atr:.2f})",
            }

        # Breakdown below support
        if current_price < support - atr * 0.5 and volume_ok:
            return {
                'direction': 2,
                'strength': min(1.0, (support - current_price) / (atr * 2)),
                'reason': f"Breakdown below {support:.2f} (ATR: {atr:.2f})",
            }

        return None


class EMACrossoverStrategy(BTCCStrategy):
    """
    EMA Crossover Strategy
    ----------------------
    Classic EMA crossover with trend filter.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'ema_fast': 9,
            'ema_slow': 21,
            'ema_trend': 50,
            'min_separation': 0.1,
        }
        if params:
            default_params.update(params)
        super().__init__('EMA_CROSSOVER', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < 60:
            return None

        closes = [c.close for c in candles]

        ema_fast = TechnicalIndicators.ema(closes, self.params['ema_fast'])
        ema_slow = TechnicalIndicators.ema(closes, self.params['ema_slow'])
        ema_trend = TechnicalIndicators.ema(closes, self.params['ema_trend'])

        current_fast = ema_fast[-1]
        current_slow = ema_slow[-1]
        current_trend = ema_trend[-1]
        prev_fast = ema_fast[-2]
        prev_slow = ema_slow[-2]

        if any(np.isnan(v) for v in [current_fast, current_slow, current_trend, prev_fast, prev_slow]):
            return None

        # Calculate separation
        separation = abs(current_fast - current_slow) / current_slow * 100

        # Bullish crossover (fast crosses above slow)
        if prev_fast <= prev_slow and current_fast > current_slow:
            if current_price > current_trend:  # Trend filter
                return {
                    'direction': 1,
                    'strength': min(1.0, separation / 1.0),
                    'reason': f"EMA bullish crossover (sep: {separation:.2f}%)",
                }

        # Bearish crossover (fast crosses below slow)
        if prev_fast >= prev_slow and current_fast < current_slow:
            if current_price < current_trend:  # Trend filter
                return {
                    'direction': 2,
                    'strength': min(1.0, separation / 1.0),
                    'reason': f"EMA bearish crossover (sep: {separation:.2f}%)",
                }

        return None


class BollingerSqueezeStrategy(BTCCStrategy):
    """
    Bollinger Squeeze Strategy
    --------------------------
    Trades breakouts from Bollinger Band squeezes.
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'kc_period': 20,
            'kc_mult': 1.5,
            'squeeze_threshold': 0.05,
        }
        if params:
            default_params.update(params)
        super().__init__('BOLLINGER_SQUEEZE', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < 30:
            return None

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(
            closes, self.params['bb_period'], self.params['bb_std']
        )

        # Keltner Channel
        kc_mid, kc_upper, kc_lower = TechnicalIndicators.keltner_channel(
            highs, lows, closes, self.params['kc_period'], self.params['kc_mult']
        )

        if any(np.isnan(v) for v in [bb_upper[-1], bb_lower[-1], kc_upper[-1], kc_lower[-1]]):
            return None

        # Check for squeeze (BB inside KC)
        bb_width = bb_upper[-1] - bb_lower[-1]
        kc_width = kc_upper[-1] - kc_lower[-1]

        was_squeeze = bb_lower[-2] > kc_lower[-2] and bb_upper[-2] < kc_upper[-2]
        is_squeeze = bb_lower[-1] > kc_lower[-1] and bb_upper[-1] < kc_upper[-1]

        # Squeeze release
        if was_squeeze and not is_squeeze:
            # Check momentum direction
            momentum = closes[-1] - closes[-5]

            if momentum > 0:
                return {
                    'direction': 1,
                    'strength': min(1.0, abs(momentum) / (bb_width * 0.5)),
                    'reason': f"Squeeze release bullish (momentum: +{momentum:.2f})",
                }
            else:
                return {
                    'direction': 2,
                    'strength': min(1.0, abs(momentum) / (bb_width * 0.5)),
                    'reason': f"Squeeze release bearish (momentum: {momentum:.2f})",
                }

        return None


class HighWinRateScalpMomentumStrategy(BTCCStrategy):
    """
    HIGH WIN RATE Scalp Momentum Strategy (84.8% win rate)
    -------------------------------------------------------
    Uses EMA5 > EMA13 with RSI 50-70 for momentum confirmation.
    Small TP (1%) + Wide SL (3%) = High win rate, inverted risk/reward.

    Best on: ETHUSDT, BTCUSDT at 20x leverage
    Backtest: 84.8% win rate, +51.5% return, 8.6% max DD (30 days)
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 7,
            'rsi_long_min': 50,
            'rsi_long_max': 70,
            'rsi_short_min': 30,
            'rsi_short_max': 50,
        }
        if params:
            default_params.update(params)
        super().__init__('SCALP_MOMENTUM', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < 25:
            return None

        closes = [c.close for c in candles]

        # Calculate indicators
        ema_fast = TechnicalIndicators.ema(closes, self.params['ema_fast'])
        ema_slow = TechnicalIndicators.ema(closes, self.params['ema_slow'])
        rsi = TechnicalIndicators.rsi(closes, self.params['rsi_period'])

        current_ema_fast = ema_fast[-1]
        current_ema_slow = ema_slow[-1]
        current_rsi = rsi[-1]

        if any(np.isnan(v) for v in [current_ema_fast, current_ema_slow, current_rsi]):
            return None

        # Long signal: EMA5 > EMA13 + RSI between 50-70 (momentum confirmation)
        if (current_ema_fast > current_ema_slow and
            current_rsi > self.params['rsi_long_min'] and
            current_rsi < self.params['rsi_long_max']):

            strength = (current_rsi - 50) / 20  # Strength based on RSI distance from 50
            return {
                'direction': 1,
                'strength': min(1.0, strength),
                'reason': f"SCALP_MOMENTUM Long: EMA5>EMA13, RSI={current_rsi:.1f}",
            }

        # Short signal: EMA5 < EMA13 + RSI between 30-50 (momentum confirmation)
        if (current_ema_fast < current_ema_slow and
            current_rsi < self.params['rsi_short_max'] and
            current_rsi > self.params['rsi_short_min']):

            strength = (50 - current_rsi) / 20
            return {
                'direction': 2,
                'strength': min(1.0, strength),
                'reason': f"SCALP_MOMENTUM Short: EMA5<EMA13, RSI={current_rsi:.1f}",
            }

        return None


class HighWinRateBollingerMeanStrategy(BTCCStrategy):
    """
    HIGH WIN RATE Bollinger Mean Reversion Strategy (80.4% win rate)
    -----------------------------------------------------------------
    Mean reversion when price touches Bollinger Bands.
    Small TP (1%) + Wide SL (3%) = High win rate.

    Best on: XRPUSDT at 20x leverage
    Backtest: 80.4% win rate, +17.2% return, 11.0% max DD (30 days)
    """

    def __init__(self, params: Dict = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'touch_threshold': 0.01,  # 1% buffer from band
        }
        if params:
            default_params.update(params)
        super().__init__('BOLLINGER_MEAN', default_params)

    def evaluate(self, candles: List[OHLCV], current_price: float) -> Optional[Dict]:
        if len(candles) < 30:
            return None

        closes = [c.close for c in candles]

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(
            closes, self.params['bb_period'], self.params['bb_std']
        )

        if any(np.isnan(v) for v in [bb_upper[-1], bb_lower[-1], bb_mid[-1]]):
            return None

        current_close = closes[-1]
        lower_threshold = bb_lower[-1] * (1 + self.params['touch_threshold'])
        upper_threshold = bb_upper[-1] * (1 - self.params['touch_threshold'])

        # Long signal: Price near/below lower band (mean reversion up)
        if current_close < lower_threshold:
            distance = (lower_threshold - current_close) / current_close * 100
            return {
                'direction': 1,
                'strength': min(1.0, distance / 2),
                'reason': f"BOLLINGER_MEAN Long: Price at lower band ({distance:.2f}% below)",
            }

        # Short signal: Price near/above upper band (mean reversion down)
        if current_close > upper_threshold:
            distance = (current_close - upper_threshold) / current_close * 100
            return {
                'direction': 2,
                'strength': min(1.0, distance / 2),
                'reason': f"BOLLINGER_MEAN Short: Price at upper band ({distance:.2f}% above)",
            }

        return None


# Strategy factory
STRATEGIES = {
    'SCALPING_MOMENTUM': ScalpingMomentumStrategy,
    'RSI_REVERSAL': RSIReversalStrategy,
    'BREAKOUT': BreakoutStrategy,
    'EMA_CROSSOVER': EMACrossoverStrategy,
    'BOLLINGER_SQUEEZE': BollingerSqueezeStrategy,
    # HIGH WIN RATE STRATEGIES (from find_high_winrate.py backtest)
    'SCALP_MOMENTUM': HighWinRateScalpMomentumStrategy,
    'BOLLINGER_MEAN': HighWinRateBollingerMeanStrategy,
}


def get_strategy(name: str, params: Dict = None) -> Optional[BTCCStrategy]:
    """Get strategy instance by name."""
    strategy_class = STRATEGIES.get(name)
    if strategy_class:
        return strategy_class(params)
    return None


def evaluate_all_strategies(candles: List[OHLCV], current_price: float,
                           strategies: List[str] = None) -> List[Dict]:
    """Evaluate multiple strategies and return all signals."""
    signals = []

    strategies = strategies or list(STRATEGIES.keys())

    for name in strategies:
        strategy = get_strategy(name)
        if strategy:
            signal = strategy.evaluate(candles, current_price)
            if signal:
                signal['strategy'] = name
                signals.append(signal)

    return signals
