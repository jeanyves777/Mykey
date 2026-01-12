"""
HTF Trend + Confluence Strategy for Forex
=========================================
Exact same strategy as Binance Futures, adapted for Forex markets

Key Components:
1. HTF (1H) Trend Detection - 21/50 EMA crossover
2. Entry Signals - MACD + RSI + EMA (9/21) confluence with MOMENTUM + VOLUME filters
3. Single Direction Trading - Follow the trend
4. Moderate Leverage - 50x with strict 8/8 confluence requirement

Entry Conditions (ALL 8 must be met):
LONG:
  1. HTF: 1H EMA 21 > 50 (Bullish trend)
  2. LTF: 15m EMA 9 > 21 (Bullish alignment)
  3. RSI: Between 35-75 (not extreme overbought)
  4. MACD: Line > Signal (bullish momentum)
  5. 5m: EMA 9 > 21 (confirmation)
  6. ADX: > 20 (trending, not choppy)
  7. Momentum: MACD histogram increasing
  8. Volume: 5m volume > 1.2x average

SHORT:
  1. HTF: 1H EMA 21 < 50 (Bearish trend)
  2. LTF: 15m EMA 9 < 21 (Bearish alignment)
  3. RSI: Between 25-65 (not extreme oversold)
  4. MACD: Line < Signal (bearish momentum)
  5. 5m: EMA 9 < 21 (confirmation)
  6. ADX: > 20 (trending, not choppy)
  7. Momentum: MACD histogram decreasing
  8. Volume: 5m volume > 1.2x average
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    """Market trend direction"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal strength levels"""
    STRONG = "STRONG"      # 8/8 confluence
    MODERATE = "MODERATE"  # 6-7/8 confluence
    WEAK = "WEAK"          # 4-5/8 confluence
    NONE = "NONE"          # Less than 4


@dataclass
class ConfluenceSignal:
    """Strategy signal output with confluence score"""
    action: Optional[str]           # "BUY", "SELL", or None
    confidence: float               # 0.0 to 1.0
    strength: SignalStrength        # Signal strength
    trend: TrendDirection           # HTF trend direction
    confluence_score: int           # 0-8 (how many conditions met)
    reason: str                     # Human-readable reason
    indicators: Dict                # All calculated indicators
    entry_price: float              # Current price for entry
    stop_loss_pips: float           # SL in pips
    take_profit_pips: float         # TP in pips


class HTFConfluenceForexStrategy:
    """
    HTF Trend + Confluence Trading Strategy for Forex
    
    Identical logic to Binance strategy, adapted for Forex:
    - Pips instead of ROI percentages
    - Forex-specific symbols
    - OANDA API integration
    """
    
    def __init__(
        self,
        leverage: int = 50,
        tp_pips: float = 80,
        sl_pips: float = 200,
        min_confluence_score: int = 8
    ):
        """
        Initialize strategy.
        
        Args:
            leverage: Trading leverage (50x standard for forex)
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
            min_confluence_score: Minimum confluence score (4-8)
        """
        # Risk settings
        self.leverage = leverage
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        
        # HTF trend settings (50 EMA instead of 200)
        self.htf_ema_period = 50
        
        # LTF entry settings (15m or 1H)
        self.ema_fast = 9
        self.ema_slow = 21
        
        # RSI settings (widened for trending markets)
        self.rsi_period = 14
        self.rsi_long_min = 35
        self.rsi_long_max = 75
        self.rsi_short_min = 25
        self.rsi_short_max = 65
        
        # MACD settings (standard 12, 26, 9)
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # ADX settings for trend strength
        self.adx_period = 14
        self.adx_threshold = 20
        
        # ATR settings for volatility filtering
        self.atr_period = 14
        self.atr_pct_max = 0.35
        self.atr_spike_threshold = 1.5
        self.atr_avg_period = 20
        
        # Pullback filter
        self.use_pullback_filter = True
        self.pullback_ema_period = 21
        self.max_distance_from_ema = 2.0
        
        # Momentum spike filter
        self.use_momentum_filter = True
        self.momentum_spike_ratio = 2.0
        
        # Volume filter
        self.use_volume_filter = True
        self.volume_min_ratio = 1.5
        
        # RSI zone filter
        self.use_rsi_zone_filter = True
        self.rsi_entry_min = 30
        self.rsi_entry_max = 70
        
        # Cooldown (disabled)
        self.min_bars_between_signals = 0
        self.last_signal_bar = -999
        
        # Minimum confluence score
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
        """Calculate MACD indicator"""
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
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
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators for LTF data"""
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
        indicators["macd_bullish"] = macd_line.iloc[-1] > signal_line.iloc[-1]
        indicators["macd_bearish"] = macd_line.iloc[-1] < signal_line.iloc[-1]
        
        # MACD momentum building
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
        
        # ATR for volatility filter
        atr = self.calculate_atr(df, self.atr_period)
        indicators["atr"] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
        indicators["atr_pct"] = (indicators["atr"] / indicators["price"]) * 100
        
        # Average ATR for volatility spike detection
        atr_avg = atr.rolling(window=self.atr_avg_period).mean()
        indicators["atr_avg"] = atr_avg.iloc[-1] if not pd.isna(atr_avg.iloc[-1]) else indicators["atr"]
        indicators["atr_ratio"] = indicators["atr"] / indicators["atr_avg"] if indicators["atr_avg"] > 0 else 1.0
        indicators["volatility_spiking"] = indicators["atr_ratio"] > self.atr_spike_threshold
        
        # Pullback filter
        pullback_ema = self.calculate_ema(close, self.pullback_ema_period)
        indicators["pullback_ema"] = pullback_ema.iloc[-1]
        indicators["distance_from_ema"] = ((indicators["price"] - indicators["pullback_ema"]) / indicators["pullback_ema"]) * 100
        indicators["is_pullback"] = abs(indicators["distance_from_ema"]) <= self.max_distance_from_ema
        
        # Volume confirmation
        volume = df['volume']
        vol_sma20 = volume.rolling(window=20).mean()
        indicators["volume"] = volume.iloc[-1]
        indicators["volume_sma20"] = vol_sma20.iloc[-1] if not pd.isna(vol_sma20.iloc[-1]) else volume.iloc[-1]
        indicators["volume_ratio"] = indicators["volume"] / indicators["volume_sma20"] if indicators["volume_sma20"] > 0 else 1.0
        indicators["volume_confirmed"] = indicators["volume_ratio"] >= 1.2
        
        return indicators
    
    def get_htf_trend(self, htf_df: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Determine higher timeframe trend using dual EMA crossover (21/50)"""
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
    
    def calculate_exit_levels_pips(
        self,
        entry_price: float,
        side: str,
        pip_location: int = -4
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels in pips.
        
        Args:
            entry_price: Entry price
            side: "LONG" or "SHORT"
            pip_location: Pip decimal location (-4 for most pairs, -2 for JPY)
            
        Returns:
            (stop_loss_price, take_profit_price)
        """
        pip_value = 10 ** pip_location
        
        if side in ["LONG", "BUY"]:
            stop_loss = entry_price - (self.sl_pips * pip_value)
            take_profit = entry_price + (self.tp_pips * pip_value)
        else:  # SHORT or SELL
            stop_loss = entry_price + (self.sl_pips * pip_value)
            take_profit = entry_price - (self.tp_pips * pip_value)
        
        return stop_loss, take_profit
    
    def check_long_conditions(
        self,
        indicators: Dict,
        htf_trend: TrendDirection
    ) -> Tuple[int, List[str]]:
        """
        Check LONG entry conditions - 8/8 CONFLUENCE REQUIRED.
        
        Returns:
            (confluence_score, list of met conditions)
        """
        conditions_met = []
        
        # Condition 1: 1H Trend bullish
        if htf_trend == TrendDirection.BULLISH:
            conditions_met.append("1H: Bullish")
        
        # Condition 2: 15m EMA alignment (9 > 21)
        if indicators.get("ema_bullish", False):
            conditions_met.append("15m: EMA 9>21")
        
        # Condition 3: 15m RSI in valid range
        rsi = indicators.get("rsi", 50)
        if self.rsi_long_min <= rsi <= self.rsi_long_max:
            conditions_met.append(f"RSI {rsi:.0f}")
        
        # Condition 4: 15m MACD bullish
        if indicators.get("macd_bullish", False):
            conditions_met.append("MACD>0")
        
        # Condition 5: 5m EMA confirmation (9 > 21)
        if indicators.get("5m_ema_bullish", False):
            conditions_met.append("5m: Confirmed")
        
        # Condition 6: ADX > 20
        adx = indicators.get("adx", 0)
        if indicators.get("adx_trending", False):
            conditions_met.append(f"ADX {adx:.0f}")
        
        # Condition 7: MACD Momentum building
        if indicators.get("macd_momentum_bullish", False):
            conditions_met.append("Momentum ↑")
        
        # Condition 8: Volume confirmation
        volume_ratio = indicators.get("volume_ratio", 0)
        if indicators.get("volume_confirmed", False):
            conditions_met.append(f"Vol {volume_ratio:.1f}x")
        
        return len(conditions_met), conditions_met
    
    def check_short_conditions(
        self,
        indicators: Dict,
        htf_trend: TrendDirection
    ) -> Tuple[int, List[str]]:
        """
        Check SHORT entry conditions - 8/8 CONFLUENCE REQUIRED.
        
        Returns:
            (confluence_score, list of met conditions)
        """
        conditions_met = []
        
        # Condition 1: 1H Trend bearish
        if htf_trend == TrendDirection.BEARISH:
            conditions_met.append("1H: Bearish")
        
        # Condition 2: 15m EMA alignment (9 < 21)
        if indicators.get("ema_bearish", False):
            conditions_met.append("15m: EMA 9<21")
        
        # Condition 3: 15m RSI in valid range
        rsi = indicators.get("rsi", 50)
        if self.rsi_short_min <= rsi <= self.rsi_short_max:
            conditions_met.append(f"RSI {rsi:.0f}")
        
        # Condition 4: 15m MACD bearish
        if indicators.get("macd_bearish", False):
            conditions_met.append("MACD<0")
        
        # Condition 5: 5m EMA confirmation (9 < 21)
        if indicators.get("5m_ema_bearish", False):
            conditions_met.append("5m: Confirmed")
        
        # Condition 6: ADX > 20
        adx = indicators.get("adx", 0)
        if indicators.get("adx_trending", False):
            conditions_met.append(f"ADX {adx:.0f}")
        
        # Condition 7: MACD Momentum building
        if indicators.get("macd_momentum_bearish", False):
            conditions_met.append("Momentum ↓")
        
        # Condition 8: Volume confirmation
        volume_ratio = indicators.get("volume_ratio", 0)
        if indicators.get("volume_confirmed", False):
            conditions_met.append(f"Vol {volume_ratio:.1f}x")
        
        return len(conditions_met), conditions_met
    
    def get_signal_strength(self, confluence_score: int) -> SignalStrength:
        """Convert confluence score to signal strength"""
        if confluence_score >= 8:
            return SignalStrength.STRONG
        elif confluence_score >= 6:
            return SignalStrength.MODERATE
        elif confluence_score >= 4:
            return SignalStrength.WEAK
        return SignalStrength.NONE
    
    def should_enter(
        self,
        ltf_df: pd.DataFrame,
        htf_df: pd.DataFrame,
        pip_location: int = -4,
        current_bar: int = 0
    ) -> ConfluenceSignal:
        """
        Determine if should enter a position.
        
        Args:
            ltf_df: Lower timeframe OHLCV data (15m)
            htf_df: Higher timeframe OHLCV data (1H or 4H)
            pip_location: Pip decimal location (-4 for most, -2 for JPY)
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
                stop_loss_pips=0.0,
                take_profit_pips=0.0
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
                stop_loss_pips=0.0,
                take_profit_pips=0.0
            )
        
        # If HTF trend is neutral, no trading
        if htf_trend == TrendDirection.NEUTRAL:
            return ConfluenceSignal(
                action=None,
                confidence=0.0,
                strength=SignalStrength.NONE,
                trend=htf_trend,
                confluence_score=0,
                reason=f"HTF Neutral ({htf_distance:+.2f}%)",
                indicators=indicators,
                entry_price=0.0,
                stop_loss_pips=0.0,
                take_profit_pips=0.0
            )
        
        entry_price = indicators["price"]
        
        # Smart entry filters (same as Binance)
        # ... (filters would be applied here in full implementation)
        
        # Check LONG conditions if HTF is bullish
        if htf_trend == TrendDirection.BULLISH:
            long_score, long_conditions = self.check_long_conditions(indicators, htf_trend)
            
            if long_score >= self.min_confluence_score:
                strength = self.get_signal_strength(long_score)
                confidence = min(0.95, 0.5 + long_score * 0.05)
                
                sl_price, tp_price = self.calculate_exit_levels_pips(
                    entry_price, "LONG", pip_location
                )
                
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
                    stop_loss_pips=self.sl_pips,
                    take_profit_pips=self.tp_pips
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
                    stop_loss_pips=0.0,
                    take_profit_pips=0.0
                )
        
        # Check SHORT conditions if HTF is bearish
        if htf_trend == TrendDirection.BEARISH:
            short_score, short_conditions = self.check_short_conditions(indicators, htf_trend)
            
            if short_score >= self.min_confluence_score:
                strength = self.get_signal_strength(short_score)
                confidence = min(0.95, 0.5 + short_score * 0.05)
                
                sl_price, tp_price = self.calculate_exit_levels_pips(
                    entry_price, "SHORT", pip_location
                )
                
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
                    stop_loss_pips=self.sl_pips,
                    take_profit_pips=self.tp_pips
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
                    stop_loss_pips=0.0,
                    take_profit_pips=0.0
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
            stop_loss_pips=0.0,
            take_profit_pips=0.0
        )
