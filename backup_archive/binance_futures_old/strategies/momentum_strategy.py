"""
Momentum Trading Strategy
=========================
High-level strategy wrapper for momentum-based trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import MOMENTUM_CONFIG, STRATEGY_CONFIG


class TrendDirection(Enum):
    """Market trend direction"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class StrategySignal:
    """Strategy signal output"""
    action: Optional[str]           # "BUY", "SELL", or None
    confidence: float               # 0.0 to 1.0
    trend: TrendDirection
    momentum_score: float           # -100 to 100
    reason: str
    indicators: Dict                # All calculated indicators


class MomentumStrategy:
    """
    Pure Momentum Trading Strategy

    Entry Conditions (ALL must be met):
    1. Momentum spike > threshold in N bars
    2. EMA trend aligned with momentum direction
    3. RSI not at extremes
    4. ADX confirms trending market

    Optional HTF Confirmation:
    - 15-min EMA trend aligned
    - 1-hour EMA trend aligned
    """

    def __init__(self):
        self.config = MOMENTUM_CONFIG

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators

        Args:
            df: OHLCV DataFrame

        Returns:
            Dict with all indicator values
        """
        if df is None or len(df) < 30:
            return {}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        indicators = {}

        # Momentum (N-bar % change)
        period = self.config["momentum_period"]
        indicators["momentum"] = ((close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]) * 100

        # EMAs
        ema_fast = close.ewm(span=self.config["ema_fast_period"], adjust=False).mean()
        ema_slow = close.ewm(span=self.config["ema_slow_period"], adjust=False).mean()
        indicators["ema_fast"] = ema_fast.iloc[-1]
        indicators["ema_slow"] = ema_slow.iloc[-1]
        indicators["ema_diff"] = indicators["ema_fast"] - indicators["ema_slow"]
        indicators["price"] = close.iloc[-1]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=self.config["rsi_period"], adjust=False).mean()
        avg_loss = loss.ewm(span=self.config["rsi_period"], adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        indicators["rsi"] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        # ADX
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.ewm(span=self.config["adx_period"], adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.config["adx_period"], adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.config["adx_period"], adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=self.config["adx_period"], adjust=False).mean()
        indicators["adx"] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
        indicators["plus_di"] = plus_di.iloc[-1]
        indicators["minus_di"] = minus_di.iloc[-1]

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        indicators["macd"] = macd_line.iloc[-1]
        indicators["macd_signal"] = signal_line.iloc[-1]
        indicators["macd_histogram"] = indicators["macd"] - indicators["macd_signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        indicators["bb_upper"] = (sma20 + 2 * std20).iloc[-1]
        indicators["bb_lower"] = (sma20 - 2 * std20).iloc[-1]
        indicators["bb_middle"] = sma20.iloc[-1]
        indicators["bb_width"] = (indicators["bb_upper"] - indicators["bb_lower"]) / indicators["bb_middle"]

        # Price position in BB
        if indicators["bb_upper"] - indicators["bb_lower"] > 0:
            indicators["bb_position"] = (close.iloc[-1] - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
        else:
            indicators["bb_position"] = 0.5

        # Volume analysis (if available)
        if "volume" in df.columns:
            vol = df["volume"]
            indicators["volume"] = vol.iloc[-1]
            indicators["volume_sma"] = vol.rolling(20).mean().iloc[-1]
            indicators["volume_ratio"] = indicators["volume"] / indicators["volume_sma"] if indicators["volume_sma"] > 0 else 1.0

        # Recent candle patterns
        indicators["last_candle_bullish"] = close.iloc[-1] > df["open"].iloc[-1]
        indicators["last_3_bullish"] = sum([close.iloc[-i] > df["open"].iloc[-i] for i in range(1, 4)])
        indicators["last_3_bearish"] = 3 - indicators["last_3_bullish"]

        return indicators

    def determine_trend(self, indicators: Dict) -> TrendDirection:
        """Determine overall trend direction"""
        if not indicators:
            return TrendDirection.NEUTRAL

        bullish_count = 0
        bearish_count = 0

        # EMA alignment
        if indicators["ema_fast"] > indicators["ema_slow"]:
            bullish_count += 1
        else:
            bearish_count += 1

        # Price vs EMAs
        if indicators["price"] > indicators["ema_fast"]:
            bullish_count += 1
        else:
            bearish_count += 1

        # MACD
        if indicators.get("macd_histogram", 0) > 0:
            bullish_count += 1
        else:
            bearish_count += 1

        # DI+/DI-
        if indicators.get("plus_di", 0) > indicators.get("minus_di", 0):
            bullish_count += 1
        else:
            bearish_count += 1

        if bullish_count >= 3:
            return TrendDirection.BULLISH
        elif bearish_count >= 3:
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    def calculate_momentum_score(self, indicators: Dict) -> float:
        """
        Calculate momentum score (-100 to 100)

        Positive = bullish momentum
        Negative = bearish momentum
        """
        if not indicators:
            return 0.0

        score = 0.0

        # Raw momentum (scaled)
        momentum = indicators.get("momentum", 0)
        score += momentum * 10  # Scale momentum to reasonable range

        # MACD histogram contribution
        macd_hist = indicators.get("macd_histogram", 0)
        score += macd_hist * 5

        # RSI contribution (deviation from 50)
        rsi = indicators.get("rsi", 50)
        score += (rsi - 50) * 0.5

        # ADX strength multiplier
        adx = indicators.get("adx", 0)
        if adx > 25:
            score *= 1.2
        elif adx < 15:
            score *= 0.8

        # Clamp to range
        return max(-100, min(100, score))

    def should_enter(self, df: pd.DataFrame, htf_data: Dict[str, pd.DataFrame] = None) -> StrategySignal:
        """
        Determine if should enter a position

        Args:
            df: 1-minute OHLCV data
            htf_data: Optional dict with "15m" and "1h" DataFrames

        Returns:
            StrategySignal with entry decision
        """
        indicators = self.calculate_indicators(df)

        if not indicators:
            return StrategySignal(
                action=None,
                confidence=0.0,
                trend=TrendDirection.NEUTRAL,
                momentum_score=0.0,
                reason="Insufficient data",
                indicators={}
            )

        trend = self.determine_trend(indicators)
        momentum_score = self.calculate_momentum_score(indicators)

        # Check momentum threshold
        momentum = indicators.get("momentum", 0)
        threshold = self.config["momentum_threshold"]

        if abs(momentum) < threshold:
            return StrategySignal(
                action=None,
                confidence=0.0,
                trend=trend,
                momentum_score=momentum_score,
                reason=f"No momentum spike ({abs(momentum):.3f}% < {threshold}%)",
                indicators=indicators
            )

        # Determine direction
        direction = "BUY" if momentum > 0 else "SELL"

        # Check trend alignment
        expected_trend = TrendDirection.BULLISH if direction == "BUY" else TrendDirection.BEARISH
        if trend != expected_trend:
            return StrategySignal(
                action=None,
                confidence=0.0,
                trend=trend,
                momentum_score=momentum_score,
                reason=f"Trend not aligned ({trend.value} vs {expected_trend.value})",
                indicators=indicators
            )

        # Check RSI extremes
        rsi = indicators.get("rsi", 50)
        if direction == "BUY" and rsi > self.config["rsi_max_for_buy"]:
            return StrategySignal(
                action=None,
                confidence=0.0,
                trend=trend,
                momentum_score=momentum_score,
                reason=f"RSI overbought ({rsi:.1f} > {self.config['rsi_max_for_buy']})",
                indicators=indicators
            )
        if direction == "SELL" and rsi < self.config["rsi_min_for_sell"]:
            return StrategySignal(
                action=None,
                confidence=0.0,
                trend=trend,
                momentum_score=momentum_score,
                reason=f"RSI oversold ({rsi:.1f} < {self.config['rsi_min_for_sell']})",
                indicators=indicators
            )

        # Check ADX
        adx = indicators.get("adx", 0)
        if adx < self.config["min_adx"]:
            return StrategySignal(
                action=None,
                confidence=0.0,
                trend=trend,
                momentum_score=momentum_score,
                reason=f"Weak trend (ADX {adx:.1f} < {self.config['min_adx']})",
                indicators=indicators
            )

        # Check HTF alignment if provided
        if htf_data:
            htf_aligned = self._check_htf_alignment(htf_data, direction)
            if not htf_aligned:
                return StrategySignal(
                    action=None,
                    confidence=0.0,
                    trend=trend,
                    momentum_score=momentum_score,
                    reason="HTF trend not aligned",
                    indicators=indicators
                )

        # All conditions met - calculate confidence
        confidence = self._calculate_confidence(indicators, direction)

        return StrategySignal(
            action=direction,
            confidence=confidence,
            trend=trend,
            momentum_score=momentum_score,
            reason=f"MOMENTUM {direction}: {momentum:+.3f}% | ADX:{adx:.1f} | RSI:{rsi:.1f}",
            indicators=indicators
        )

    def _check_htf_alignment(self, htf_data: Dict[str, pd.DataFrame], direction: str) -> bool:
        """Check if higher timeframes support the direction"""
        for tf, df in htf_data.items():
            if df is None or len(df) < 30:
                continue

            indicators = self.calculate_indicators(df)
            if not indicators:
                continue

            trend = self.determine_trend(indicators)

            if direction == "BUY" and trend != TrendDirection.BULLISH:
                return False
            if direction == "SELL" and trend != TrendDirection.BEARISH:
                return False

        return True

    def _calculate_confidence(self, indicators: Dict, direction: str) -> float:
        """Calculate signal confidence"""
        base = 0.65

        # Momentum strength bonus
        momentum = abs(indicators.get("momentum", 0))
        threshold = self.config["momentum_threshold"]
        momentum_bonus = min(0.15, (momentum / threshold - 1) * 0.05)

        # ADX bonus
        adx = indicators.get("adx", 0)
        adx_bonus = min(0.10, (adx - self.config["min_adx"]) / 30 * 0.10)

        # RSI room bonus
        rsi = indicators.get("rsi", 50)
        if direction == "BUY":
            rsi_room = (self.config["rsi_max_for_buy"] - rsi) / 40
        else:
            rsi_room = (rsi - self.config["rsi_min_for_sell"]) / 40
        rsi_bonus = min(0.10, max(0, rsi_room * 0.10))

        return min(0.95, base + momentum_bonus + adx_bonus + rsi_bonus)

    def get_exit_levels(self, entry_price: float, side: str) -> Tuple[float, float]:
        """
        Get stop loss and take profit levels

        Args:
            entry_price: Entry price
            side: "LONG" or "SHORT"

        Returns:
            (stop_loss, take_profit)
        """
        if side == "LONG":
            sl = entry_price * (1 - STRATEGY_CONFIG["stop_loss_pct"])
            tp = entry_price * (1 + STRATEGY_CONFIG["take_profit_pct"])
        else:
            sl = entry_price * (1 + STRATEGY_CONFIG["stop_loss_pct"])
            tp = entry_price * (1 - STRATEGY_CONFIG["take_profit_pct"])

        return sl, tp


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1min")

    # Simulate trending market
    prices = [50000]
    for i in range(99):
        change = np.random.uniform(-0.001, 0.002)  # Slight uptrend
        if i > 90:  # Momentum spike
            change = np.random.uniform(0.001, 0.003)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.001 for p in prices],
        "low": [p * 0.999 for p in prices],
        "close": prices,
        "volume": np.random.uniform(100, 1000, 100)
    }, index=dates)

    # Test strategy
    strategy = MomentumStrategy()
    signal = strategy.should_enter(df)

    print("\n" + "="*60)
    print("MOMENTUM STRATEGY TEST")
    print("="*60)
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Trend: {signal.trend.value}")
    print(f"Momentum Score: {signal.momentum_score:.1f}")
    print(f"Reason: {signal.reason}")

    if signal.action:
        sl, tp = strategy.get_exit_levels(prices[-1], "LONG" if signal.action == "BUY" else "SHORT")
        print(f"\nExit Levels:")
        print(f"  Stop Loss: ${sl:,.2f}")
        print(f"  Take Profit: ${tp:,.2f}")
