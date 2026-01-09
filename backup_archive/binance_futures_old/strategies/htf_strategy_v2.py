"""
HTF Trend + Confluence Strategy V2 - IMPROVED ENTRIES
======================================================
Fixes for live trading issues:
1. ATR-based dynamic SL/TP - adapts to actual volatility
2. Pullback entries - enter on retracements, not chasing
3. Volatility filter - skip high volatility (choppy) periods
4. Candle confirmation - require proper candle structure

Problem with V1: Fixed % SL gets hit in normal volatility
Solution: Use ATR to set SL outside normal price noise
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


class MarketCondition(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"  # Too volatile - skip


@dataclass
class ConfluenceSignal:
    action: Optional[str]
    confidence: float
    strength: SignalStrength
    trend: TrendDirection
    confluence_score: int
    reason: str
    indicators: Dict
    entry_price: float
    stop_loss: float
    take_profit: float
    market_condition: MarketCondition
    atr_value: float  # For reference


class HTFConfluenceStrategyV2:
    """
    IMPROVED VERSION with:
    - ATR-based dynamic SL/TP
    - Pullback entry detection
    - Volatility filtering
    - Candle pattern confirmation
    """

    def __init__(
        self,
        leverage: int = 20,
        atr_sl_multiplier: float = 1.5,  # SL = 1.5x ATR from entry
        atr_tp_multiplier: float = 3.0,  # TP = 3x ATR (2:1 R:R)
        max_atr_percent: float = 2.0,    # Skip if ATR > 2% of price (too volatile)
        min_atr_percent: float = 0.3,    # Skip if ATR < 0.3% (no movement)
    ):
        """
        Args:
            leverage: Trading leverage
            atr_sl_multiplier: ATR multiplier for stop loss
            atr_tp_multiplier: ATR multiplier for take profit
            max_atr_percent: Maximum ATR as % of price (volatility cap)
            min_atr_percent: Minimum ATR as % of price (ensure movement)
        """
        self.leverage = leverage
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.max_atr_percent = max_atr_percent
        self.min_atr_percent = min_atr_percent

        # ATR period
        self.atr_period = 14

        # HTF trend - 50 EMA (more responsive)
        self.htf_ema_period = 50

        # LTF EMAs
        self.ema_fast = 9
        self.ema_slow = 21

        # RSI settings - WIDER RANGE for more signals
        self.rsi_period = 14
        self.rsi_long_min = 35   # Was 40 - allow slightly oversold
        self.rsi_long_max = 60   # Was 65 - not overbought
        self.rsi_short_min = 40  # Was 35 - not oversold
        self.rsi_short_max = 65  # Was 60 - allow slightly overbought

        # MACD
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # ADX
        self.adx_period = 14
        self.adx_trending_threshold = 20   # Above = trending
        self.adx_strong_threshold = 25     # Above = strong trend

        # Cooldown
        self.min_bars_between_signals = 8  # Reduced from 16 - allow more trades
        self.last_signal_bar = -999

        # Confluence
        self.min_confluence_score = 3

        # Pullback settings
        self.pullback_ema = 21  # Look for pullback to 21 EMA
        self.pullback_tolerance = 0.003  # Within 0.3% of EMA

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range - measures volatility"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def get_market_condition(self, df: pd.DataFrame) -> Tuple[MarketCondition, float, float]:
        """
        Assess market condition based on ATR and ADX.
        
        Returns:
            (condition, atr_value, atr_percent)
        """
        if df is None or len(df) < self.atr_period + 5:
            return MarketCondition.RANGING, 0.0, 0.0

        close = df["close"]
        current_price = close.iloc[-1]

        # Calculate ATR
        atr = self.calculate_atr(df, self.atr_period)
        atr_value = atr.iloc[-1]
        atr_percent = (atr_value / current_price) * 100

        # Calculate ADX
        adx = self.calculate_adx(df, self.adx_period)
        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

        # Determine condition
        if atr_percent > self.max_atr_percent:
            return MarketCondition.VOLATILE, atr_value, atr_percent
        elif atr_percent < self.min_atr_percent:
            return MarketCondition.RANGING, atr_value, atr_percent
        elif adx_value >= self.adx_trending_threshold:
            return MarketCondition.TRENDING, atr_value, atr_percent
        else:
            return MarketCondition.RANGING, atr_value, atr_percent

    def is_pullback_entry(self, df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
        """
        Check if price has pulled back to EMA - better entry than chasing.
        
        For LONG: Price should be near/touching 21 EMA from above (dip buy)
        For SHORT: Price should be near/touching 21 EMA from below (rally sell)
        """
        if df is None or len(df) < self.pullback_ema + 5:
            return False, "Insufficient data"

        close = df["close"]
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]

        ema = self.calculate_ema(close, self.pullback_ema)
        ema_value = ema.iloc[-1]

        distance_from_ema = (current_price - ema_value) / ema_value

        if direction == "LONG":
            # For LONG: Price should be slightly above EMA (just bounced) or touching it
            # Good: Price dipped to EMA and bouncing (distance 0% to 0.5%)
            # Bad: Price far above EMA (chasing)
            if -self.pullback_tolerance <= distance_from_ema <= 0.005:
                # Check if bouncing (current > prev and near EMA)
                if current_price >= prev_price:
                    return True, f"Pullback to 21 EMA ({distance_from_ema*100:.2f}% from EMA)"
            return False, f"No pullback ({distance_from_ema*100:.2f}% from EMA)"

        elif direction == "SHORT":
            # For SHORT: Price should be slightly below EMA (just rejected) or touching it
            if -0.005 <= distance_from_ema <= self.pullback_tolerance:
                if current_price <= prev_price:
                    return True, f"Rally to 21 EMA ({distance_from_ema*100:.2f}% from EMA)"
            return False, f"No rally ({distance_from_ema*100:.2f}% from EMA)"

        return False, "Invalid direction"

    def is_candle_confirmed(self, df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
        """
        Check candle pattern confirmation.
        
        For LONG: Current candle should be bullish (close > open)
        For SHORT: Current candle should be bearish (close < open)
        
        Also checks for rejection wicks (good) vs indecision (bad)
        """
        if df is None or len(df) < 3:
            return False, "Insufficient data"

        current = df.iloc[-1]
        prev = df.iloc[-2]

        body = abs(current["close"] - current["open"])
        upper_wick = current["high"] - max(current["close"], current["open"])
        lower_wick = min(current["close"], current["open"]) - current["low"]
        total_range = current["high"] - current["low"]

        if total_range == 0:
            return False, "No range (doji)"

        body_ratio = body / total_range

        if direction == "LONG":
            is_bullish = current["close"] > current["open"]
            has_rejection = lower_wick > upper_wick  # Lower wick = buyers stepping in

            if is_bullish and body_ratio > 0.4:
                if has_rejection:
                    return True, "Bullish candle with rejection wick"
                return True, "Bullish candle confirmed"
            elif is_bullish:
                return False, "Weak bullish candle (small body)"
            return False, "Bearish candle - no confirmation"

        elif direction == "SHORT":
            is_bearish = current["close"] < current["open"]
            has_rejection = upper_wick > lower_wick  # Upper wick = sellers stepping in

            if is_bearish and body_ratio > 0.4:
                if has_rejection:
                    return True, "Bearish candle with rejection wick"
                return True, "Bearish candle confirmed"
            elif is_bearish:
                return False, "Weak bearish candle (small body)"
            return False, "Bullish candle - no confirmation"

        return False, "Invalid direction"

    def calculate_atr_based_exits(
        self, 
        entry_price: float, 
        atr_value: float, 
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate SL and TP based on ATR - adapts to volatility.
        
        SL = 1.5x ATR (outside normal noise)
        TP = 3x ATR (2:1 reward to risk)
        """
        sl_distance = atr_value * self.atr_sl_multiplier
        tp_distance = atr_value * self.atr_tp_multiplier

        if side in ["LONG", "BUY"]:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        if df is None or len(df) < 50:
            return {}

        close = df["close"]
        indicators = {}

        # Price
        indicators["price"] = close.iloc[-1]

        # ATR
        atr = self.calculate_atr(df, self.atr_period)
        indicators["atr"] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        indicators["atr_percent"] = (indicators["atr"] / indicators["price"]) * 100

        # EMAs
        ema_fast = self.calculate_ema(close, self.ema_fast)
        ema_slow = self.calculate_ema(close, self.ema_slow)
        indicators["ema_fast"] = ema_fast.iloc[-1]
        indicators["ema_slow"] = ema_slow.iloc[-1]
        indicators["ema_fast_prev"] = ema_fast.iloc[-2]
        indicators["ema_slow_prev"] = ema_slow.iloc[-2]

        # EMA states
        indicators["ema_bullish_cross"] = (
            ema_fast.iloc[-1] > ema_slow.iloc[-1] and
            ema_fast.iloc[-2] <= ema_slow.iloc[-2]
        )
        indicators["ema_bearish_cross"] = (
            ema_fast.iloc[-1] < ema_slow.iloc[-1] and
            ema_fast.iloc[-2] >= ema_slow.iloc[-2]
        )
        indicators["ema_bullish"] = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        indicators["ema_bearish"] = ema_fast.iloc[-1] < ema_slow.iloc[-1]

        # RSI
        rsi = self.calculate_rsi(close, self.rsi_period)
        indicators["rsi"] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(close)
        indicators["macd"] = macd_line.iloc[-1]
        indicators["macd_signal"] = signal_line.iloc[-1]
        indicators["macd_histogram"] = histogram.iloc[-1]
        indicators["macd_histogram_prev"] = histogram.iloc[-2]

        indicators["macd_bullish_cross"] = (
            macd_line.iloc[-1] > signal_line.iloc[-1] and
            macd_line.iloc[-2] <= signal_line.iloc[-2]
        )
        indicators["macd_bearish_cross"] = (
            macd_line.iloc[-1] < signal_line.iloc[-1] and
            macd_line.iloc[-2] >= signal_line.iloc[-2]
        )
        indicators["macd_bullish"] = histogram.iloc[-1] > 0
        indicators["macd_bearish"] = histogram.iloc[-1] < 0

        # ADX
        adx = self.calculate_adx(df, self.adx_period)
        indicators["adx"] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        indicators["adx_trending"] = indicators["adx"] >= self.adx_trending_threshold

        return indicators

    def get_htf_trend(self, htf_df: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Get HTF trend using dual EMA"""
        if htf_df is None or len(htf_df) < self.htf_ema_period:
            return TrendDirection.NEUTRAL, 0.0

        close = htf_df["close"]
        ema_fast_htf = self.calculate_ema(close, 21)
        ema_slow_htf = self.calculate_ema(close, self.htf_ema_period)

        current_price = close.iloc[-1]
        ema_fast_value = ema_fast_htf.iloc[-1]
        ema_slow_value = ema_slow_htf.iloc[-1]

        if pd.isna(ema_fast_value) or pd.isna(ema_slow_value):
            return TrendDirection.NEUTRAL, 0.0

        distance_pct = (current_price - ema_slow_value) / ema_slow_value * 100

        if ema_fast_value > ema_slow_value and current_price > ema_fast_value:
            return TrendDirection.BULLISH, distance_pct
        elif ema_fast_value < ema_slow_value and current_price < ema_fast_value:
            return TrendDirection.BEARISH, distance_pct
        elif ema_fast_value > ema_slow_value:
            return TrendDirection.BULLISH, distance_pct
        elif ema_fast_value < ema_slow_value:
            return TrendDirection.BEARISH, distance_pct
        else:
            return TrendDirection.NEUTRAL, distance_pct

    def check_long_conditions(
        self, 
        indicators: Dict, 
        htf_trend: TrendDirection,
        ltf_df: pd.DataFrame
    ) -> Tuple[int, List[str], bool, bool]:
        """
        Check LONG conditions with additional filters.
        
        Returns:
            (score, conditions, pullback_ok, candle_ok)
        """
        conditions_met = []

        # 1. HTF Trend
        if htf_trend == TrendDirection.BULLISH:
            if indicators.get("adx_trending", False):
                conditions_met.append(f"HTF Bullish + ADX {indicators.get('adx', 0):.1f}")
            else:
                conditions_met.append(f"HTF Bullish (weak ADX {indicators.get('adx', 0):.1f})")

        # 2. EMA alignment
        if indicators.get("ema_bullish_cross", False):
            conditions_met.append("EMA 9/21 bullish cross")
        elif indicators.get("ema_bullish", False):
            conditions_met.append("EMA 9 > 21")

        # 3. RSI
        rsi = indicators.get("rsi", 50)
        if self.rsi_long_min <= rsi <= self.rsi_long_max:
            conditions_met.append(f"RSI {rsi:.1f} OK")

        # 4. MACD
        if indicators.get("macd_bullish_cross", False):
            conditions_met.append("MACD bullish cross")
        elif indicators.get("macd_bullish", False):
            conditions_met.append("MACD histogram > 0")

        # Additional checks (not counted in score but required)
        pullback_ok, pullback_reason = self.is_pullback_entry(ltf_df, "LONG")
        candle_ok, candle_reason = self.is_candle_confirmed(ltf_df, "LONG")

        return len(conditions_met), conditions_met, pullback_ok, candle_ok

    def check_short_conditions(
        self, 
        indicators: Dict, 
        htf_trend: TrendDirection,
        ltf_df: pd.DataFrame
    ) -> Tuple[int, List[str], bool, bool]:
        """Check SHORT conditions with additional filters"""
        conditions_met = []

        # 1. HTF Trend
        if htf_trend == TrendDirection.BEARISH:
            if indicators.get("adx_trending", False):
                conditions_met.append(f"HTF Bearish + ADX {indicators.get('adx', 0):.1f}")
            else:
                conditions_met.append(f"HTF Bearish (weak ADX {indicators.get('adx', 0):.1f})")

        # 2. EMA alignment
        if indicators.get("ema_bearish_cross", False):
            conditions_met.append("EMA 9/21 bearish cross")
        elif indicators.get("ema_bearish", False):
            conditions_met.append("EMA 9 < 21")

        # 3. RSI
        rsi = indicators.get("rsi", 50)
        if self.rsi_short_min <= rsi <= self.rsi_short_max:
            conditions_met.append(f"RSI {rsi:.1f} OK")

        # 4. MACD
        if indicators.get("macd_bearish_cross", False):
            conditions_met.append("MACD bearish cross")
        elif indicators.get("macd_bearish", False):
            conditions_met.append("MACD histogram < 0")

        # Additional checks
        pullback_ok, pullback_reason = self.is_pullback_entry(ltf_df, "SHORT")
        candle_ok, candle_reason = self.is_candle_confirmed(ltf_df, "SHORT")

        return len(conditions_met), conditions_met, pullback_ok, candle_ok

    def get_signal_strength(self, confluence_score: int) -> SignalStrength:
        if confluence_score >= 4:
            return SignalStrength.STRONG
        elif confluence_score >= 3:
            return SignalStrength.MODERATE
        elif confluence_score >= 2:
            return SignalStrength.WEAK
        return SignalStrength.NONE

    def should_enter(
        self,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        current_bar: int = 0,
        require_pullback: bool = True,   # NEW: Require pullback entry
        require_candle: bool = True,      # NEW: Require candle confirmation
    ) -> ConfluenceSignal:
        """
        Main entry logic with improved filters.
        """
        # Get market condition first
        market_condition, atr_value, atr_percent = self.get_market_condition(ltf_df)

        # Calculate indicators
        indicators = self.calculate_indicators(ltf_df)

        if not indicators:
            return ConfluenceSignal(
                action=None, confidence=0.0, strength=SignalStrength.NONE,
                trend=TrendDirection.NEUTRAL, confluence_score=0,
                reason="Insufficient data", indicators={},
                entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                market_condition=market_condition, atr_value=0.0
            )

        # Get HTF trend
        htf_trend, htf_distance = self.get_htf_trend(htf_df)
        indicators["htf_trend"] = htf_trend.value
        indicators["htf_distance_pct"] = htf_distance
        indicators["market_condition"] = market_condition.value

        # FILTER 1: Skip volatile markets
        if market_condition == MarketCondition.VOLATILE:
            return ConfluenceSignal(
                action=None, confidence=0.0, strength=SignalStrength.NONE,
                trend=htf_trend, confluence_score=0,
                reason=f"TOO VOLATILE: ATR {atr_percent:.2f}% > {self.max_atr_percent}%",
                indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                market_condition=market_condition, atr_value=atr_value
            )

        # FILTER 2: Skip ranging (low volatility) markets
        if market_condition == MarketCondition.RANGING and not indicators.get("adx_trending", False):
            return ConfluenceSignal(
                action=None, confidence=0.0, strength=SignalStrength.NONE,
                trend=htf_trend, confluence_score=0,
                reason=f"RANGING: ATR {atr_percent:.2f}%, ADX {indicators.get('adx', 0):.1f} < {self.adx_trending_threshold}",
                indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                market_condition=market_condition, atr_value=atr_value
            )

        # Cooldown check
        if current_bar - self.last_signal_bar < self.min_bars_between_signals:
            return ConfluenceSignal(
                action=None, confidence=0.0, strength=SignalStrength.NONE,
                trend=htf_trend, confluence_score=0,
                reason=f"Cooldown: {self.min_bars_between_signals - (current_bar - self.last_signal_bar)} bars left",
                indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                market_condition=market_condition, atr_value=atr_value
            )

        # Neutral trend = no trade
        if htf_trend == TrendDirection.NEUTRAL:
            return ConfluenceSignal(
                action=None, confidence=0.0, strength=SignalStrength.NONE,
                trend=htf_trend, confluence_score=0,
                reason="HTF Neutral - no clear trend",
                indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                market_condition=market_condition, atr_value=atr_value
            )

        entry_price = indicators["price"]

        # Check LONG
        if htf_trend == TrendDirection.BULLISH:
            score, conditions, pullback_ok, candle_ok = self.check_long_conditions(
                indicators, htf_trend, ltf_df
            )

            # Check additional filters
            filters_passed = []
            filters_failed = []

            if require_pullback:
                if pullback_ok:
                    filters_passed.append("Pullback ✓")
                else:
                    filters_failed.append("No pullback")

            if require_candle:
                if candle_ok:
                    filters_passed.append("Candle ✓")
                else:
                    filters_failed.append("Bad candle")

            # Need confluence + filters
            if score >= self.min_confluence_score:
                if filters_failed and (require_pullback or require_candle):
                    return ConfluenceSignal(
                        action=None, confidence=0.0, strength=SignalStrength.WEAK,
                        trend=htf_trend, confluence_score=score,
                        reason=f"LONG {score}/4 but: {', '.join(filters_failed)}",
                        indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                        market_condition=market_condition, atr_value=atr_value
                    )

                # ALL GOOD - Generate signal
                strength = self.get_signal_strength(score)
                confidence = min(0.95, 0.5 + score * 0.1)
                stop_loss, take_profit = self.calculate_atr_based_exits(entry_price, atr_value, "LONG")

                self.last_signal_bar = current_bar

                return ConfluenceSignal(
                    action="BUY",
                    confidence=confidence,
                    strength=strength,
                    trend=htf_trend,
                    confluence_score=score,
                    reason=" | ".join(conditions + filters_passed),
                    indicators=indicators,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    market_condition=market_condition,
                    atr_value=atr_value
                )
            else:
                return ConfluenceSignal(
                    action=None, confidence=0.0, strength=SignalStrength.WEAK,
                    trend=htf_trend, confluence_score=score,
                    reason=f"LONG: {score}/{self.min_confluence_score} conditions",
                    indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                    market_condition=market_condition, atr_value=atr_value
                )

        # Check SHORT
        if htf_trend == TrendDirection.BEARISH:
            score, conditions, pullback_ok, candle_ok = self.check_short_conditions(
                indicators, htf_trend, ltf_df
            )

            filters_passed = []
            filters_failed = []

            if require_pullback:
                if pullback_ok:
                    filters_passed.append("Pullback ✓")
                else:
                    filters_failed.append("No pullback")

            if require_candle:
                if candle_ok:
                    filters_passed.append("Candle ✓")
                else:
                    filters_failed.append("Bad candle")

            if score >= self.min_confluence_score:
                if filters_failed and (require_pullback or require_candle):
                    return ConfluenceSignal(
                        action=None, confidence=0.0, strength=SignalStrength.WEAK,
                        trend=htf_trend, confluence_score=score,
                        reason=f"SHORT {score}/4 but: {', '.join(filters_failed)}",
                        indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                        market_condition=market_condition, atr_value=atr_value
                    )

                strength = self.get_signal_strength(score)
                confidence = min(0.95, 0.5 + score * 0.1)
                stop_loss, take_profit = self.calculate_atr_based_exits(entry_price, atr_value, "SHORT")

                self.last_signal_bar = current_bar

                return ConfluenceSignal(
                    action="SELL",
                    confidence=confidence,
                    strength=strength,
                    trend=htf_trend,
                    confluence_score=score,
                    reason=" | ".join(conditions + filters_passed),
                    indicators=indicators,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    market_condition=market_condition,
                    atr_value=atr_value
                )
            else:
                return ConfluenceSignal(
                    action=None, confidence=0.0, strength=SignalStrength.WEAK,
                    trend=htf_trend, confluence_score=score,
                    reason=f"SHORT: {score}/{self.min_confluence_score} conditions",
                    indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                    market_condition=market_condition, atr_value=atr_value
                )

        return ConfluenceSignal(
            action=None, confidence=0.0, strength=SignalStrength.NONE,
            trend=htf_trend, confluence_score=0, reason="No signal",
            indicators=indicators, entry_price=0.0, stop_loss=0.0, take_profit=0.0,
            market_condition=market_condition, atr_value=atr_value
        )


# =============================================================================
# V2 CONFIGURATIONS - ATR-based
# =============================================================================
V2_CONSERVATIVE = {
    "leverage": 20,
    "atr_sl_multiplier": 2.0,   # 2x ATR SL (very safe)
    "atr_tp_multiplier": 4.0,   # 4x ATR TP (2:1 R:R)
    "max_atr_percent": 1.5,     # Skip if ATR > 1.5%
    "min_atr_percent": 0.3,
}

V2_MODERATE = {
    "leverage": 20,
    "atr_sl_multiplier": 1.5,   # 1.5x ATR SL
    "atr_tp_multiplier": 3.0,   # 3x ATR TP (2:1 R:R)
    "max_atr_percent": 2.0,     # Skip if ATR > 2%
    "min_atr_percent": 0.3,
}

V2_AGGRESSIVE = {
    "leverage": 20,
    "atr_sl_multiplier": 1.2,   # 1.2x ATR SL (tighter)
    "atr_tp_multiplier": 3.6,   # 3.6x ATR TP (3:1 R:R)
    "max_atr_percent": 2.5,     # Allow more volatility
    "min_atr_percent": 0.2,
}


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("HTF CONFLUENCE STRATEGY V2 - TEST")
    print("="*60)

    np.random.seed(42)

    # Simulate trending market with pullbacks
    print("\nSimulating BULLISH trend with pullback...")

    # 4H data
    htf_prices = [50000]
    for i in range(99):
        change = np.random.uniform(0.001, 0.003)
        htf_prices.append(htf_prices[-1] * (1 + change))

    htf_df = pd.DataFrame({
        "open": htf_prices,
        "high": [p * 1.003 for p in htf_prices],
        "low": [p * 0.997 for p in htf_prices],
        "close": htf_prices,
    })

    # 15m data with pullback pattern
    base_price = htf_prices[-1]
    ltf_prices = []
    for i in range(100):
        if i < 70:
            # Uptrend
            ltf_prices.append(base_price * (1 + i * 0.0005))
        elif i < 85:
            # Pullback to EMA
            ltf_prices.append(base_price * (1 + 70 * 0.0005 - (i - 70) * 0.001))
        else:
            # Bounce
            ltf_prices.append(base_price * (1 + 70 * 0.0005 - 15 * 0.001 + (i - 85) * 0.0008))

    ltf_df = pd.DataFrame({
        "open": [p * 0.999 for p in ltf_prices],
        "high": [p * 1.002 for p in ltf_prices],
        "low": [p * 0.998 for p in ltf_prices],
        "close": ltf_prices,
    })

    # Test V2 strategy
    strategy = HTFConfluenceStrategyV2(**V2_MODERATE)
    signal = strategy.should_enter(ltf_df, htf_df, require_pullback=True, require_candle=True)

    print(f"\n{'='*40}")
    print("SIGNAL RESULTS")
    print(f"{'='*40}")
    print(f"Action: {signal.action}")
    print(f"Strength: {signal.strength.value}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"HTF Trend: {signal.trend.value}")
    print(f"Market Condition: {signal.market_condition.value}")
    print(f"Confluence: {signal.confluence_score}/4")
    print(f"Reason: {signal.reason}")

    if signal.action:
        print(f"\nEntry: ${signal.entry_price:,.2f}")
        print(f"Stop Loss: ${signal.stop_loss:,.2f} ({(signal.stop_loss/signal.entry_price - 1)*100:+.2f}%)")
        print(f"Take Profit: ${signal.take_profit:,.2f} ({(signal.take_profit/signal.entry_price - 1)*100:+.2f}%)")
        print(f"ATR: ${signal.atr_value:,.2f} ({signal.indicators.get('atr_percent', 0):.2f}%)")

    print(f"\nKey Indicators:")
    print(f"  RSI: {signal.indicators.get('rsi', 0):.1f}")
    print(f"  ADX: {signal.indicators.get('adx', 0):.1f}")
    print(f"  ATR%: {signal.indicators.get('atr_percent', 0):.2f}%")

    print("\n" + "="*60)
