"""
HTF Trend + Confluence Strategy
================================
Combines high win-rate indicators with higher timeframe trend detection.

Key Components:
1. HTF (1H) Trend Detection - 21/50 EMA crossover (fast for scalping)
2. Entry Signals - MACD + RSI + EMA (9/21) confluence with MOMENTUM + VOLUME filters
3. Single Direction Trading - Follow the trend, no hedging
4. Moderate Leverage - 20x with strict 8/8 confluence requirement

Research Base:
- MACD + RSI: 73% win rate in backtests
- EMA Crossover (9/21): 55-60% win rate, 2.65% avg gain on BTC
- MACD Momentum: Filters weak signals by requiring increasing histogram
- Volume Confirmation: Ensures strong market participation
- HTF trend filter: Proven to increase win rates by filtering counter-trend trades

Entry Conditions (ALL 8 must be met - MAXIMUM STRENGTH):
LONG:
  1. HTF: 1H EMA 21 > 50 (Bullish trend)
  2. LTF: 15m EMA 9 > 21 (Bullish alignment)
  3. RSI: Between 35-75 (not extreme overbought)
  4. MACD: Line > Signal (bullish momentum)
  5. 5m: EMA 9 > 21 (confirmation)
  6. ADX: > 20 (trending, not choppy)
  7. Momentum: MACD histogram increasing (building momentum)
  8. Volume: 5m volume > 1.2x average (strong participation)

SHORT:
  1. HTF: 1H EMA 21 < 50 (Bearish trend)
  2. LTF: 15m EMA 9 < 21 (Bearish alignment)
  3. RSI: Between 25-65 (not extreme oversold)
  4. MACD: Line < Signal (bearish momentum)
  5. 5m: EMA 9 < 21 (confirmation)
  6. ADX: > 20 (trending, not choppy)
  7. Momentum: MACD histogram decreasing (building momentum)
  8. Volume: 5m volume > 1.2x average (strong participation)

Risk Management:
  - TP: 40% ROI (2% price move at 20x leverage)
  - SL: 10% ROI (0.5% price move at 20x leverage)
  - Reward:Risk = 4:1
  - Leverage: 20x
  - Risk per trade: 1-2% of balance
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class TrendDirection(Enum):
    """Market trend direction"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal strength levels"""
    STRONG = "STRONG"      # 4/4 confluence
    MODERATE = "MODERATE"  # 3/4 confluence
    WEAK = "WEAK"          # 2/4 confluence
    NONE = "NONE"          # Less than 2


@dataclass
class ConfluenceSignal:
    """Strategy signal output with confluence score"""
    action: Optional[str]           # "BUY", "SELL", or None
    confidence: float               # 0.0 to 1.0
    strength: SignalStrength        # Signal strength
    trend: TrendDirection           # HTF trend direction
    confluence_score: int           # 0-4 (how many conditions met)
    reason: str                     # Human-readable reason
    indicators: Dict                # All calculated indicators
    entry_price: float              # Current price for entry
    stop_loss: float                # Calculated SL price
    take_profit: float              # Calculated TP price


class HTFConfluenceStrategy:
    """
    HTF Trend + Confluence Trading Strategy

    Combines 200 EMA trend filter with MACD + RSI + EMA crossover for entries.
    Single direction trading - only trade in direction of HTF trend.
    """

    def __init__(self, leverage: int = 10, tp_roi: float = 0.02, sl_roi: float = 0.01, min_confluence_score: int = 8):
        """
        Initialize strategy.

        Args:
            leverage: Trading leverage (5-10x recommended)
            tp_roi: Take profit as ROI (0.02 = 2% ROI)
            sl_roi: Stop loss as ROI (0.01 = 1% ROI)
            min_confluence_score: Minimum confluence score required (asset-specific: 4-8)
        """
        # Risk settings
        self.leverage = leverage
        self.tp_roi = tp_roi
        self.sl_roi = sl_roi

        # HTF trend settings
        # Use 50 EMA instead of 200 - more responsive and accurate with limited data
        self.htf_ema_period = 50           # 50 EMA on 4H for trend (200 needs too much history)

        # LTF entry settings (15m or 1H)
        self.ema_fast = 9                   # Fast EMA
        self.ema_slow = 21                  # Slow EMA

        # RSI settings - WIDENED for trending markets (RSI stays elevated in trends)
        self.rsi_period = 14
        self.rsi_long_min = 35              # Minimum RSI for LONG (was 40)
        self.rsi_long_max = 75              # Maximum RSI for LONG (was 65) - RSI can stay high in uptrend
        self.rsi_short_min = 25             # Minimum RSI for SHORT (was 35) - RSI can stay low in downtrend  
        self.rsi_short_max = 65             # Maximum RSI for SHORT (was 60)

        # MACD settings (standard 12, 26, 9)
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # ADX settings for trend strength filter
        self.adx_period = 14
        self.adx_threshold = 20             # Minimum ADX for trending market

        # ATR settings for dynamic SL/TP (DISABLED - backtests showed worse results)
        self.atr_period = 14
        self.atr_sl_multiplier = 1.5        # SL = ATR * multiplier
        self.atr_tp_multiplier = 3.0        # TP = ATR * multiplier (2:1 R:R based on ATR)
        self.use_dynamic_sl = False         # DISABLED - fixed ROI works better

        # Pullback entry filter - enter on dips, not chasing (DISABLED)
        self.use_pullback_filter = True     # ENABLED - wait for price near EMA (avoid chasing)
        self.pullback_ema_period = 21       # Check pullback to this EMA
        self.max_distance_from_ema = 2.0    # Max 2.0% from EMA (avoid over-extension)

        # Volatility filter - skip when ATR is too high (ENABLED - blocks choppy markets)
        self.use_volatility_filter = True   # ENABLED - blocks high volatility losses
        self.atr_spike_threshold = 1.5      # ATR > 1.5x average = spiking (choppy)
        self.atr_pct_max = 1.2              # Max ATR as % of price (1.2% = allows normal volatility)
        self.atr_avg_period = 20            # Period to calculate average ATR
        
        # Momentum spike filter (NEW - avoid parabolic moves that reverse)
        self.use_momentum_filter = False    # DISABLED - too restrictive
        self.momentum_spike_ratio = 2.0     # Block if candle moved >2x ATR (parabolic)

        # Volume confirmation filter (NEW - blocks low conviction entries)
        self.use_volume_filter = False      # DISABLED - too restrictive
        self.volume_min_ratio = 1.5         # Require 1.5x average volume

        # RSI zone filter (NEW - avoid extreme overbought/oversold)
        self.use_rsi_zone_filter = False    # DISABLED - RSI can stay overbought in trends
        self.rsi_entry_min = 30             # Don't buy if RSI < 30 (oversold bounce risk)
        self.rsi_entry_max = 70             # Don't buy if RSI > 70 (overbought reversal risk)

        # Cooldown - DISABLED (trade whenever signal appears)
        self.min_bars_between_signals = 0   # No cooldown
        self.last_signal_bar = -999

        # Minimum confluence score - Asset-specific based on historical performance
        # BTC/SOL/ETH/DOT=8 (premium), XRP=6 (never reaches 8), ADA/BNB=4 (mostly 3-4)
        self.min_confluence_score = min_confluence_score

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.

        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) for dynamic SL/TP.
        ATR measures volatility and adapts to market conditions.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA of True Range
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) for trend strength.
        ADX > 20 indicates trending market, ADX < 20 indicates ranging market.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smooth TR, +DM, -DM
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators for LTF data.

        Args:
            df: OHLCV DataFrame (15m or 1H timeframe)

        Returns:
            Dict with all indicator values
        """
        if df is None or len(df) < 50:
            return {}

        close = df["close"]

        indicators = {}

        # Current price
        indicators["price"] = close.iloc[-1]

        # EMAs (9 and 21)
        ema_fast = self.calculate_ema(close, self.ema_fast)
        ema_slow = self.calculate_ema(close, self.ema_slow)
        indicators["ema_fast"] = ema_fast.iloc[-1]
        indicators["ema_slow"] = ema_slow.iloc[-1]
        indicators["ema_fast_prev"] = ema_fast.iloc[-2]
        indicators["ema_slow_prev"] = ema_slow.iloc[-2]

        # EMA crossover detection
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
        indicators["rsi_prev"] = rsi.iloc[-2] if not pd.isna(rsi.iloc[-2]) else 50.0

        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(close)
        indicators["macd"] = macd_line.iloc[-1]
        indicators["macd_signal"] = signal_line.iloc[-1]
        indicators["macd_histogram"] = histogram.iloc[-1]
        indicators["macd_prev"] = macd_line.iloc[-2]
        indicators["macd_signal_prev"] = signal_line.iloc[-2]
        indicators["macd_histogram_prev"] = histogram.iloc[-2]

        # MACD crossover detection
        indicators["macd_bullish_cross"] = (
            macd_line.iloc[-1] > signal_line.iloc[-1] and
            macd_line.iloc[-2] <= signal_line.iloc[-2]
        )
        indicators["macd_bearish_cross"] = (
            macd_line.iloc[-1] < signal_line.iloc[-1] and
            macd_line.iloc[-2] >= signal_line.iloc[-2]
        )
        # FIXED: Use MACD line > signal (more responsive to trend changes)
        indicators["macd_bullish"] = macd_line.iloc[-1] > signal_line.iloc[-1]
        indicators["macd_bearish"] = macd_line.iloc[-1] < signal_line.iloc[-1]

        # FIXED: MACD momentum building (histogram increasing, can start from negative)
        # This catches momentum BUILDING, not just already positive
        indicators["macd_momentum_bullish"] = (
            indicators["macd_histogram"] > indicators["macd_histogram_prev"]
        )
        indicators["macd_momentum_bearish"] = (
            indicators["macd_histogram"] < indicators["macd_histogram_prev"]
        )

        # ADX for trend strength
        adx = self.calculate_adx(df, self.adx_period)
        indicators["adx"] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
        indicators["adx_trending"] = indicators["adx"] >= self.adx_threshold

        # ATR for dynamic SL/TP and volatility filter
        atr = self.calculate_atr(df, self.atr_period)
        indicators["atr"] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
        indicators["atr_pct"] = (indicators["atr"] / indicators["price"]) * 100  # ATR as % of price

        # Average ATR for volatility spike detection
        atr_avg = atr.rolling(window=self.atr_avg_period).mean()
        indicators["atr_avg"] = atr_avg.iloc[-1] if not pd.isna(atr_avg.iloc[-1]) else indicators["atr"]
        indicators["atr_ratio"] = indicators["atr"] / indicators["atr_avg"] if indicators["atr_avg"] > 0 else 1.0
        indicators["volatility_spiking"] = indicators["atr_ratio"] > self.atr_spike_threshold

        # Pullback filter - distance from EMA
        pullback_ema = self.calculate_ema(close, self.pullback_ema_period)
        indicators["pullback_ema"] = pullback_ema.iloc[-1]
        indicators["distance_from_ema"] = ((indicators["price"] - indicators["pullback_ema"]) / indicators["pullback_ema"]) * 100
        indicators["is_pullback"] = abs(indicators["distance_from_ema"]) <= self.max_distance_from_ema

        # VOLUME CONFIRMATION (5m timeframe) - Strong volume supports real moves
        volume = df['volume']
        vol_sma20 = volume.rolling(window=20).mean()
        indicators["volume"] = volume.iloc[-1]
        indicators["volume_sma20"] = vol_sma20.iloc[-1] if not pd.isna(vol_sma20.iloc[-1]) else volume.iloc[-1]
        indicators["volume_ratio"] = indicators["volume"] / indicators["volume_sma20"] if indicators["volume_sma20"] > 0 else 1.0
        # Volume must be at least 1.2x average (20% above average)
        indicators["volume_confirmed"] = indicators["volume_ratio"] >= 1.2

        return indicators

    def get_htf_trend(self, htf_df: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """
        Determine higher timeframe trend using dual EMA crossover (21/50).
        More responsive than single EMA approach.

        Args:
            htf_df: 4H OHLCV DataFrame

        Returns:
            (trend_direction, distance_from_ema_pct)
        """
        if htf_df is None or len(htf_df) < self.htf_ema_period:
            return TrendDirection.NEUTRAL, 0.0

        close = htf_df["close"]

        # Use dual EMA crossover for trend detection (more responsive)
        ema_fast_htf = self.calculate_ema(close, 21)   # Fast EMA on HTF
        ema_slow_htf = self.calculate_ema(close, self.htf_ema_period)  # 50 EMA on HTF

        current_price = close.iloc[-1]
        ema_fast_value = ema_fast_htf.iloc[-1]
        ema_slow_value = ema_slow_htf.iloc[-1]

        if pd.isna(ema_fast_value) or pd.isna(ema_slow_value):
            return TrendDirection.NEUTRAL, 0.0

        # Distance from slow EMA for reference
        distance_pct = (current_price - ema_slow_value) / ema_slow_value * 100

        # EMA crossover difference
        ema_diff_pct = (ema_fast_value - ema_slow_value) / ema_slow_value * 100

        # Trend determination based on:
        # 1. Fast EMA above/below slow EMA (crossover)
        # 2. Price position relative to both EMAs
        if ema_fast_value > ema_slow_value and current_price > ema_fast_value:
            # Strong bullish: price above both EMAs, fast > slow
            return TrendDirection.BULLISH, distance_pct
        elif ema_fast_value < ema_slow_value and current_price < ema_fast_value:
            # Strong bearish: price below both EMAs, fast < slow
            return TrendDirection.BEARISH, distance_pct
        elif ema_fast_value > ema_slow_value:
            # Weak bullish: fast > slow but price below fast EMA
            return TrendDirection.BULLISH, distance_pct
        elif ema_fast_value < ema_slow_value:
            # Weak bearish: fast < slow but price above fast EMA
            return TrendDirection.BEARISH, distance_pct
        else:
            return TrendDirection.NEUTRAL, distance_pct

    def calculate_exit_levels(self, entry_price: float, side: str, atr: float = 0.0) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        Uses ATR-based dynamic levels if enabled, otherwise falls back to fixed ROI.

        Args:
            entry_price: Entry price
            side: "LONG" or "SHORT"
            atr: Current ATR value (for dynamic SL/TP)

        Returns:
            (stop_loss, take_profit)
        """
        if self.use_dynamic_sl and atr > 0:
            # ATR-based dynamic SL/TP - adapts to actual volatility
            sl_distance = atr * self.atr_sl_multiplier
            tp_distance = atr * self.atr_tp_multiplier

            if side in ["LONG", "BUY"]:
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # SHORT or SELL
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
        else:
            # Fixed ROI-based SL/TP (fallback)
            tp_move = self.tp_roi / self.leverage
            sl_move = self.sl_roi / self.leverage

            if side in ["LONG", "BUY"]:
                stop_loss = entry_price * (1 - sl_move)
                take_profit = entry_price * (1 + tp_move)
            else:  # SHORT or SELL
                stop_loss = entry_price * (1 + sl_move)
                take_profit = entry_price * (1 - tp_move)

        return stop_loss, take_profit

    def check_long_conditions(self, indicators: Dict, htf_trend: TrendDirection) -> Tuple[int, List[str]]:
        """
        Check LONG entry conditions - 8/8 CONFLUENCE REQUIRED.

        Conditions:
        1. 1H Trend: EMA 21 > 50 (Bullish)
        2. 15m EMA: 9 > 21 (Bullish alignment)
        3. 15m RSI: 35-75 range (trending up)
        4. 15m MACD: Line > Signal (bullish momentum)
        5. 5m EMA: 9 > 21 (Confirmation)
        6. ADX > 20 (Trending, not choppy)
        7. MACD Momentum: Histogram increasing (building bullish momentum)
        8. Volume: 5m volume > 1.2x average (strong volume confirmation)

        Returns:
            (confluence_score, list of met conditions)
        """
        conditions_met = []

        # Condition 1: 1H Trend bullish (from htf_trend which now uses 1H data)
        if htf_trend == TrendDirection.BULLISH:
            conditions_met.append("1H: Bullish")

        # Condition 2: 15m EMA alignment (9 > 21)
        if indicators.get("ema_bullish", False):
            conditions_met.append("15m: EMA 9>21")

        # Condition 3: 15m RSI in valid range
        rsi = indicators.get("rsi", 50)
        if self.rsi_long_min <= rsi <= self.rsi_long_max:
            conditions_met.append(f"RSI {rsi:.0f}")

        # Condition 4: 15m MACD bullish (histogram > 0)
        if indicators.get("macd_bullish", False):
            conditions_met.append("MACD>0")

        # Condition 5: 5m EMA confirmation (9 > 21)
        if indicators.get("5m_ema_bullish", False):
            conditions_met.append("5m: Confirmed")

        # Condition 6: ADX > 20 (NOT choppy - trending market)
        adx = indicators.get("adx", 0)
        if indicators.get("adx_trending", False):
            conditions_met.append(f"ADX {adx:.0f}")

        # Condition 7: MACD Momentum building (histogram increasing)
        if indicators.get("macd_momentum_bullish", False):
            conditions_met.append("Momentum ↑")

        # Condition 8: Volume confirmation (5m volume > 1.2x average)
        volume_ratio = indicators.get("volume_ratio", 0)
        if indicators.get("volume_confirmed", False):
            conditions_met.append(f"Vol {volume_ratio:.1f}x")

        return len(conditions_met), conditions_met

    def check_short_conditions(self, indicators: Dict, htf_trend: TrendDirection) -> Tuple[int, List[str]]:
        """
        Check SHORT entry conditions - 8/8 CONFLUENCE REQUIRED.

        Conditions:
        1. 1H Trend: EMA 21 < 50 (Bearish)
        2. 15m EMA: 9 < 21 (Bearish alignment)
        3. 15m RSI: 25-65 range (trending down)
        4. 15m MACD: Line < Signal (bearish momentum)
        5. 5m EMA: 9 < 21 (Confirmation)
        6. ADX > 20 (Trending, not choppy)
        7. MACD Momentum: Histogram decreasing (building bearish momentum)
        8. Volume: 5m volume > 1.2x average (strong volume confirmation)

        Returns:
            (confluence_score, list of met conditions)
        """
        conditions_met = []

        # Condition 1: 1H Trend bearish (from htf_trend which now uses 1H data)
        if htf_trend == TrendDirection.BEARISH:
            conditions_met.append("1H: Bearish")

        # Condition 2: 15m EMA alignment (9 < 21)
        if indicators.get("ema_bearish", False):
            conditions_met.append("15m: EMA 9<21")

        # Condition 3: 15m RSI in valid range
        rsi = indicators.get("rsi", 50)
        if self.rsi_short_min <= rsi <= self.rsi_short_max:
            conditions_met.append(f"RSI {rsi:.0f}")

        # Condition 4: 15m MACD bearish (histogram < 0)
        if indicators.get("macd_bearish", False):
            conditions_met.append("MACD<0")

        # Condition 5: 5m EMA confirmation (9 < 21)
        if indicators.get("5m_ema_bearish", False):
            conditions_met.append("5m: Confirmed")

        # Condition 6: ADX > 20 (NOT choppy - trending market)
        adx = indicators.get("adx", 0)
        if indicators.get("adx_trending", False):
            conditions_met.append(f"ADX {adx:.0f}")

        # Condition 7: MACD Momentum building (histogram decreasing)
        if indicators.get("macd_momentum_bearish", False):
            conditions_met.append("Momentum ↓")

        # Condition 8: Volume confirmation (5m volume > 1.2x average)
        volume_ratio = indicators.get("volume_ratio", 0)
        if indicators.get("volume_confirmed", False):
            conditions_met.append(f"Vol {volume_ratio:.1f}x")

        return len(conditions_met), conditions_met

    def get_signal_strength(self, confluence_score: int) -> SignalStrength:
        """Convert confluence score to signal strength"""
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
        current_bar: int = 0
    ) -> ConfluenceSignal:
        """
        Determine if should enter a position.

        Args:
            ltf_df: Lower timeframe OHLCV data (15m or 1H)
            htf_df: Higher timeframe OHLCV data (4H)
            current_bar: Current bar index for cooldown tracking

        Returns:
            ConfluenceSignal with entry decision
        """
        # Calculate indicators
        indicators = self.calculate_indicators(ltf_df)

        if not indicators:
            return ConfluenceSignal(
                action=None,
                confidence=0.0,
                strength=SignalStrength.NONE,
                trend=TrendDirection.NEUTRAL,
                confluence_score=0,
                reason="Insufficient LTF data",
                indicators={},
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0
            )

        # Get HTF trend
        htf_trend, htf_distance = self.get_htf_trend(htf_df)
        indicators["htf_trend"] = htf_trend.value
        indicators["htf_distance_pct"] = htf_distance

        # Check cooldown
        if current_bar - self.last_signal_bar < self.min_bars_between_signals:
            return ConfluenceSignal(
                action=None,
                confidence=0.0,
                strength=SignalStrength.NONE,
                trend=htf_trend,
                confluence_score=0,
                reason=f"Cooldown: {self.min_bars_between_signals - (current_bar - self.last_signal_bar)} bars remaining",
                indicators=indicators,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0
            )

        # If HTF trend is neutral, no trading
        if htf_trend == TrendDirection.NEUTRAL:
            return ConfluenceSignal(
                action=None,
                confidence=0.0,
                strength=SignalStrength.NONE,
                trend=htf_trend,
                confluence_score=0,
                reason=f"HTF Neutral: Price near 200 EMA ({htf_distance:+.2f}%)",
                indicators=indicators,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0
            )

        entry_price = indicators["price"]
        atr = indicators.get("atr", 0.0)
        
        # SMART ENTRY FILTER 1: Momentum spike filter - avoid parabolic moves (NEW)
        if self.use_momentum_filter and atr > 0:
            # Calculate last candle % move
            candle_range = abs(indicators.get("high", entry_price) - indicators.get("low", entry_price))
            candle_pct = (candle_range / entry_price) * 100
            atr_pct = (atr / entry_price) * 100
            
            # If candle moved >2x ATR, it's a momentum spike (likely to reverse)
            if candle_pct > (atr_pct * self.momentum_spike_ratio):
                return ConfluenceSignal(
                    action=None,
                    confidence=0.0,
                    strength=SignalStrength.NONE,
                    trend=htf_trend,
                    confluence_score=0,
                    reason=f"Momentum spike: {candle_pct:.2f}% > {atr_pct * self.momentum_spike_ratio:.2f}% ({self.momentum_spike_ratio}x ATR) - wait for consolidation",
                    indicators=indicators,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )

        # SMART ENTRY FILTER 2: Volatility filter - skip when ATR is spiking (choppy market)
        if self.use_volatility_filter:
            atr_pct = indicators.get("atr_pct", 0.0)
            if atr_pct > self.atr_pct_max:
                return ConfluenceSignal(
                    action=None,
                    confidence=0.0,
                    strength=SignalStrength.NONE,
                    trend=htf_trend,
                    confluence_score=0,
                    reason=f"Volatility filter: ATR {atr_pct:.2f}% > {self.atr_pct_max}% - too choppy",
                    indicators=indicators,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )
            # Also check ATR spike ratio
            if indicators.get("volatility_spiking", False):
                atr_ratio = indicators.get("atr_ratio", 1.0)
                return ConfluenceSignal(
                    action=None,
                    confidence=0.0,
                    strength=SignalStrength.NONE,
                    trend=htf_trend,
                    confluence_score=0,
                    reason=f"Volatility filter: ATR spiking ({atr_ratio:.2f}x avg) - choppy market",
                    indicators=indicators,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )

        # SMART ENTRY FILTER 2: Pullback filter - only enter on dips to EMA
        if self.use_pullback_filter and not indicators.get("is_pullback", True):
            distance = indicators.get("distance_from_ema", 0.0)
            return ConfluenceSignal(
                action=None,
                confidence=0.0,
                strength=SignalStrength.NONE,
                trend=htf_trend,
                confluence_score=0,
                reason=f"Pullback filter: Price {distance:+.2f}% from EMA (max {self.max_distance_from_ema}%) - not a dip entry",
                indicators=indicators,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0
            )
        
        # SMART ENTRY FILTER 3: Volume confirmation - require strong volume
        if self.use_volume_filter:
            volume_ratio = indicators.get("volume_ratio", 0.0)
            if volume_ratio < self.volume_min_ratio:
                return ConfluenceSignal(
                    action=None,
                    confidence=0.0,
                    strength=SignalStrength.NONE,
                    trend=htf_trend,
                    confluence_score=0,
                    reason=f"Volume filter: {volume_ratio:.2f}x < {self.volume_min_ratio}x required - low conviction",
                    indicators=indicators,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )
        
        # SMART ENTRY FILTER 4: RSI zone filter - avoid extreme overbought/oversold
        if self.use_rsi_zone_filter:
            rsi = indicators.get("rsi", 50)
            if htf_trend == TrendDirection.BULLISH:
                if rsi < self.rsi_entry_min:
                    return ConfluenceSignal(
                        action=None,
                        confidence=0.0,
                        strength=SignalStrength.NONE,
                        trend=htf_trend,
                        confluence_score=0,
                        reason=f"RSI filter: RSI {rsi:.1f} < {self.rsi_entry_min} - too oversold (bounce risk)",
                        indicators=indicators,
                        entry_price=0.0,
                        stop_loss=0.0,
                        take_profit=0.0
                    )
                if rsi > self.rsi_entry_max:
                    return ConfluenceSignal(
                        action=None,
                        confidence=0.0,
                        strength=SignalStrength.NONE,
                        trend=htf_trend,
                        confluence_score=0,
                        reason=f"RSI filter: RSI {rsi:.1f} > {self.rsi_entry_max} - too overbought (reversal risk)",
                        indicators=indicators,
                        entry_price=0.0,
                        stop_loss=0.0,
                        take_profit=0.0
                    )
            elif htf_trend == TrendDirection.BEARISH:
                if rsi > self.rsi_entry_max:
                    return ConfluenceSignal(
                        action=None,
                        confidence=0.0,
                        strength=SignalStrength.NONE,
                        trend=htf_trend,
                        confluence_score=0,
                        reason=f"RSI filter: RSI {rsi:.1f} > {self.rsi_entry_max} - too overbought (bounce risk)",
                        indicators=indicators,
                        entry_price=0.0,
                        stop_loss=0.0,
                        take_profit=0.0
                    )
                if rsi < self.rsi_entry_min:
                    return ConfluenceSignal(
                        action=None,
                        confidence=0.0,
                        strength=SignalStrength.NONE,
                        trend=htf_trend,
                        confluence_score=0,
                        reason=f"RSI filter: RSI {rsi:.1f} < {self.rsi_entry_min} - too oversold (reversal risk)",
                        indicators=indicators,
                        entry_price=0.0,
                        stop_loss=0.0,
                        take_profit=0.0
                    )

        # Check LONG conditions if HTF is bullish
        if htf_trend == TrendDirection.BULLISH:
            long_score, long_conditions = self.check_long_conditions(indicators, htf_trend)

            # STRENGTHENED: Require min_confluence_score (default 4 = ALL conditions)
            if long_score >= self.min_confluence_score:
                strength = self.get_signal_strength(long_score)
                confidence = min(0.95, 0.5 + long_score * 0.1)
                # Pass ATR for dynamic SL/TP calculation
                stop_loss, take_profit = self.calculate_exit_levels(entry_price, "LONG", atr)

                self.last_signal_bar = current_bar

                return ConfluenceSignal(
                    action="BUY",
                    confidence=confidence,
                    strength=strength,
                    trend=htf_trend,
                    confluence_score=long_score,
                    reason=" | ".join(long_conditions),
                    indicators=indicators,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            else:
                return ConfluenceSignal(
                    action=None,
                    confidence=0.0,
                    strength=SignalStrength.WEAK,
                    trend=htf_trend,
                    confluence_score=long_score,
                    reason=f"LONG: Only {long_score}/{self.min_confluence_score} conditions met",
                    indicators=indicators,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )

        # Check SHORT conditions if HTF is bearish
        if htf_trend == TrendDirection.BEARISH:
            short_score, short_conditions = self.check_short_conditions(indicators, htf_trend)

            # STRENGTHENED: Require min_confluence_score (default 4 = ALL conditions)
            if short_score >= self.min_confluence_score:
                strength = self.get_signal_strength(short_score)
                confidence = min(0.95, 0.5 + short_score * 0.1)
                # Pass ATR for dynamic SL/TP calculation
                stop_loss, take_profit = self.calculate_exit_levels(entry_price, "SHORT", atr)

                self.last_signal_bar = current_bar

                return ConfluenceSignal(
                    action="SELL",
                    confidence=confidence,
                    strength=strength,
                    trend=htf_trend,
                    confluence_score=short_score,
                    reason=" | ".join(short_conditions),
                    indicators=indicators,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            else:
                return ConfluenceSignal(
                    action=None,
                    confidence=0.0,
                    strength=SignalStrength.WEAK,
                    trend=htf_trend,
                    confluence_score=short_score,
                    reason=f"SHORT: Only {short_score}/{self.min_confluence_score} conditions met",
                    indicators=indicators,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0
                )

        return ConfluenceSignal(
            action=None,
            confidence=0.0,
            strength=SignalStrength.NONE,
            trend=htf_trend,
            confluence_score=0,
            reason="No valid signal",
            indicators=indicators,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0
        )


# =============================================================================
# Configuration for different risk profiles
# =============================================================================
# =============================================================================
# IMPORTANT: ROI vs Price Move
# =============================================================================
# ROI = Return on Margin (what you see in Binance)
# Price Move = ROI / Leverage
#
# Example with 20x leverage:
#   - 10% ROI = 0.5% price move
#   - 20% ROI = 1% price move
#   - 100% ROI = 5% price move (double your money)
#   - 400% ROI = 20% price move (possible in strong trends!)
#
# SL should be tight (10-20% ROI) to protect capital
# TP can be wide (30-100%+ ROI) in trending markets
# =============================================================================

CONSERVATIVE_CONFIG = {
    "leverage": 10,
    "tp_roi": 0.20,     # 20% ROI (TP) = 2% price move
    "sl_roi": 0.10,     # 10% ROI (SL) = 1% price move | 2:1 R:R
}

MODERATE_CONFIG = {
    "leverage": 20,
    "tp_roi": 0.30,     # 30% ROI (TP) = 1.5% price move
    "sl_roi": 0.10,     # 10% ROI (SL) = 0.5% price move | 3:1 R:R
}

AGGRESSIVE_CONFIG = {
    "leverage": 20,
    "tp_roi": 0.50,     # 50% ROI (TP) = 2.5% price move
    "sl_roi": 0.15,     # 15% ROI (SL) = 0.75% price move | 3.3:1 R:R
}

# OPTIMIZED CONFIG - Based on user's live trading (20x leverage)
# Wider TP to catch trends, tight SL to protect capital
OPTIMIZED_CONFIG = {
    "leverage": 20,
    "tp_roi": 0.40,     # 40% ROI (TP) = 2% price move
    "sl_roi": 0.10,     # 10% ROI (SL) = 0.5% price move | 4:1 R:R
}

# SWING CONFIG - For catching bigger moves in trending markets
# Can hit 100%+ ROI in strong trends
SWING_CONFIG = {
    "leverage": 20,
    "tp_roi": 1.00,     # 100% ROI (TP) = 5% price move (double your money)
    "sl_roi": 0.20,     # 20% ROI (SL) = 1% price move | 5:1 R:R
}

# =============================================================================
# ASSET-SPECIFIC OPTIMIZED CONFIGS (from 60-day backtest optimization)
# =============================================================================
# Each asset has different volatility characteristics:
# - DOT: More volatile, needs wider SL (15% ROI = 0.75% price move)
# - BNB: Less volatile, tight SL works (10% ROI = 0.5% price move)
# - AVAX: Trending well, tight SL works (10% ROI = 0.5% price move)
#
# Optimization results (60-day backtest):
# Optimized with REAL Binance Futures data (60-day backtest Jan 2026)
# Total: +165.8% return over 60 days
# =============================================================================
ASSET_SPECIFIC_CONFIG = {
    "DOTUSDT": {
        "leverage": 20,
        "tp_roi": 0.75,     # 75% ROI (TP) = 3.75% price move
        "sl_roi": 0.35,     # 35% ROI (SL) = 1.75% price move | 2.14:1 R:R
        "min_confluence_score": 4,  # Lowered from 5 to allow more trades
    },
    "BNBUSDT": {
        "leverage": 20,
        "tp_roi": 0.30,     # 30% ROI (TP) = 1.5% price move
        "sl_roi": 0.15,     # 15% ROI (SL) = 0.75% price move | 2:1 R:R
        "min_confluence_score": 4,  # Lowered from 5 to allow more trades
    },
    "XRPUSDT": {
        "leverage": 20,
        "tp_roi": 0.50,     # 50% ROI (TP) = 2.5% price move (was 40%, +10%)
        "sl_roi": 0.35,     # 35% ROI (SL) = 1.75% price move | 1.43:1 R:R (was 25%, +10%)
        "min_confluence_score": 5,  # Lowered from 6 - already struggles to reach 6
    },
    "ADAUSDT": {
        "leverage": 20,
        "tp_roi": 0.60,     # 60% ROI (TP) = 3% price move
        "sl_roi": 0.25,     # 25% ROI (SL) = 1.25% price move | 2.4:1 R:R
        "min_confluence_score": 4,  # Lowered from 5 to allow more trades
    },
    "BTCUSDT": {
        "leverage": 20,
        "tp_roi": 0.40,     # 40% ROI (TP) = 2% price move
        "sl_roi": 0.20,     # 20% ROI (SL) = 1% price move | 2:1 R:R
        "min_confluence_score": 6,  # Lowered from 8 to 6 (still high quality)
    },
    "ETHUSDT": {
        "leverage": 20,
        "tp_roi": 0.45,     # 45% ROI (TP) = 2.25% price move
        "sl_roi": 0.20,     # 20% ROI (SL) = 1% price move | 2.25:1 R:R
        "min_confluence_score": 5,  # Lowered from 7 to 5
    },
    "SOLUSDT": {
        "leverage": 20,
        "tp_roi": 0.50,     # 50% ROI (TP) = 2.5% price move
        "sl_roi": 0.25,     # 25% ROI (SL) = 1.25% price move | 2:1 R:R
        "min_confluence_score": 5,  # Lowered from 7 to 5
    },
    "AVAXUSDT": {
        "leverage": 20,
        "tp_roi": 0.75,     # 75% ROI (TP) = 3.75% price move
        "sl_roi": 0.35,     # 35% ROI (SL) = 1.75% price move | 2.14:1 R:R
        "min_confluence_score": 5,  # Lowered from 7 to 5
    },
    "LINKUSDT": {
        "leverage": 20,
        "tp_roi": 0.45,     # 45% ROI (TP) = 2.25% price move
        "sl_roi": 0.20,     # 20% ROI (SL) = 1% price move | 2.25:1 R:R
        "min_confluence_score": 5,  # Lowered from 7 to 5
    },
    "LTCUSDT": {
        "leverage": 20,
        "tp_roi": 0.40,     # 40% ROI (TP) = 2% price move (like BTC)
        "sl_roi": 0.20,     # 20% ROI (SL) = 1% price move | 2:1 R:R
        "min_confluence_score": 6,  # Lowered from 8 to 6 (still high quality)
    },
}

def get_config_for_symbol(symbol: str) -> dict:
    """
    Get the optimized config for a specific symbol.
    Falls back to MODERATE_CONFIG if symbol not in asset-specific configs.
    """
    return ASSET_SPECIFIC_CONFIG.get(symbol, MODERATE_CONFIG)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("\n" + "="*60)
    print("HTF CONFLUENCE STRATEGY - TEST")
    print("="*60)

    # Create sample data
    np.random.seed(42)

    # Simulate bullish trending market
    print("\nSimulating BULLISH trend scenario...")

    # 4H data (HTF) - 250 candles for 200 EMA
    htf_prices = [50000]
    for i in range(249):
        change = np.random.uniform(0.001, 0.003)  # Uptrend
        htf_prices.append(htf_prices[-1] * (1 + change))

    htf_df = pd.DataFrame({
        "open": htf_prices,
        "high": [p * 1.002 for p in htf_prices],
        "low": [p * 0.998 for p in htf_prices],
        "close": htf_prices,
    })

    # 15m data (LTF) - 100 candles
    ltf_prices = [htf_prices[-1]]
    for i in range(99):
        change = np.random.uniform(-0.001, 0.002)  # Slight uptrend
        if i > 90:  # Recent momentum
            change = np.random.uniform(0.001, 0.003)
        ltf_prices.append(ltf_prices[-1] * (1 + change))

    ltf_df = pd.DataFrame({
        "open": ltf_prices,
        "high": [p * 1.001 for p in ltf_prices],
        "low": [p * 0.999 for p in ltf_prices],
        "close": ltf_prices,
    })

    # Test strategy
    strategy = HTFConfluenceStrategy(**MODERATE_CONFIG)
    signal = strategy.should_enter(ltf_df, htf_df)

    print(f"\nSignal Results:")
    print(f"  Action: {signal.action}")
    print(f"  Strength: {signal.strength.value}")
    print(f"  Confidence: {signal.confidence:.1%}")
    print(f"  HTF Trend: {signal.trend.value}")
    print(f"  Confluence Score: {signal.confluence_score}/4")
    print(f"  Reason: {signal.reason}")

    if signal.action:
        print(f"\n  Entry Price: ${signal.entry_price:,.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"  Take Profit: ${signal.take_profit:,.2f}")

    print(f"\n  Key Indicators:")
    print(f"    RSI: {signal.indicators.get('rsi', 0):.1f}")
    print(f"    MACD Histogram: {signal.indicators.get('macd_histogram', 0):.4f}")
    print(f"    EMA Fast: {signal.indicators.get('ema_fast', 0):,.2f}")
    print(f"    EMA Slow: {signal.indicators.get('ema_slow', 0):,.2f}")
    print(f"    HTF Distance: {signal.indicators.get('htf_distance_pct', 0):+.2f}%")

    print("\n" + "="*60)
