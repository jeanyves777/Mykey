"""
Master Momentum Signal Generator
================================
Pure momentum-based signal generation (NO ML)
Based on Forex Master Momentum System
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.trading_config import MOMENTUM_CONFIG


class SignalType(Enum):
    """Types of signals generated"""
    NONE = "NONE"
    MOMENTUM = "MOMENTUM"           # Pure momentum spike
    MOMENTUM_STRONG = "MOMENTUM_STRONG"  # Strong momentum + all confirmations


@dataclass
class TradingSignal:
    """Trading signal container"""
    signal: Optional[str]           # "BUY", "SELL", or None
    signal_type: SignalType
    confidence: float               # 0.0 to 1.0
    reason: str                     # Human-readable reason
    momentum_value: float           # Actual momentum %
    ema_trend: str                  # "BULLISH", "BEARISH", "NEUTRAL"
    rsi_value: float
    adx_value: float


class MasterMomentumSignal:
    """
    Master Momentum Signal Generator

    Generates trading signals based on:
    1. Momentum spike detection (N-bar price change)
    2. Trend alignment (EMA fast vs slow)
    3. RSI filter (avoid overbought/oversold)
    4. ADX filter (require trending market)
    """

    def __init__(self):
        # Load configuration
        self.momentum_period = MOMENTUM_CONFIG["momentum_period"]
        self.momentum_threshold = MOMENTUM_CONFIG["momentum_threshold"]

        self.ema_fast_period = MOMENTUM_CONFIG["ema_fast_period"]
        self.ema_slow_period = MOMENTUM_CONFIG["ema_slow_period"]

        self.rsi_period = MOMENTUM_CONFIG["rsi_period"]
        self.rsi_max_buy = MOMENTUM_CONFIG["rsi_max_for_buy"]
        self.rsi_min_sell = MOMENTUM_CONFIG["rsi_min_for_sell"]

        self.adx_period = MOMENTUM_CONFIG["adx_period"]
        self.min_adx = MOMENTUM_CONFIG["min_adx"]

        self.cooldown_bars = MOMENTUM_CONFIG["cooldown_bars"]

        # State tracking
        self.last_signal_bar = {}  # {symbol: bar_index}

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_adx(self, high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)

        ADX measures trend strength (not direction):
        - ADX < 20: Weak/no trend
        - ADX 20-40: Trending
        - ADX > 40: Strong trend
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed values
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    def calculate_momentum(self, close: pd.Series, period: int = 3) -> pd.Series:
        """
        Calculate momentum as percentage change over N bars

        Args:
            close: Close prices
            period: Number of bars to look back

        Returns:
            Momentum as percentage
        """
        return ((close - close.shift(period)) / close.shift(period)) * 100

    def check_cooldown(self, symbol: str, current_bar: int) -> bool:
        """
        Check if enough time has passed since last signal.
        Uses time-based cooldown for live trading, bar-based for backtest.

        Returns:
            True if cooldown period has passed (can trade)
        """
        from datetime import datetime, timedelta

        last_signal = self.last_signal_bar.get(symbol)

        if last_signal is None:
            return True  # No previous signal, can trade

        # Bar-based cooldown (for backtest)
        if isinstance(last_signal, int):
            return current_bar - last_signal >= self.cooldown_bars

        # Time-based cooldown (for live trading)
        cooldown_seconds = self.cooldown_bars * 60  # 5 bars = 5 minutes
        elapsed = (datetime.now() - last_signal).total_seconds()

        return elapsed >= cooldown_seconds

    def set_cooldown_bar(self, symbol: str, bar_index: int):
        """Set cooldown using bar index (for backtest)"""
        self.last_signal_bar[symbol] = bar_index

    def set_cooldown_time(self, symbol: str):
        """Set cooldown using current time (for live trading)"""
        from datetime import datetime
        self.last_signal_bar[symbol] = datetime.now()

    def generate_signal_no_cooldown(self, symbol: str, df: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal WITHOUT cooldown check (for backtest where cooldown is handled externally).
        """
        return self._generate_signal_internal(symbol, df, check_cooldown=False)

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> TradingSignal:
        """
        Generate trading signal based on momentum analysis (with cooldown check).
        """
        return self._generate_signal_internal(symbol, df, check_cooldown=True)

    def _generate_signal_internal(self, symbol: str, df: pd.DataFrame, check_cooldown: bool = True) -> TradingSignal:
        """
        Internal signal generation logic.

        Signal Logic:
        1. Calculate momentum (3-bar % change)
        2. Check if momentum exceeds threshold
        3. Verify trend alignment (EMA fast vs slow)
        4. Filter with RSI (avoid extremes)
        5. Confirm with ADX (require trend)

        Args:
            symbol: Trading pair
            df: DataFrame with OHLCV data (1-min timeframe preferred)
            check_cooldown: Whether to check cooldown (False for backtest)

        Returns:
            TradingSignal with signal details
        """
        if df is None or len(df) < max(self.ema_slow_period, self.adx_period) + 5:
            return TradingSignal(
                signal=None,
                signal_type=SignalType.NONE,
                confidence=0.0,
                reason="Insufficient data",
                momentum_value=0.0,
                ema_trend="NEUTRAL",
                rsi_value=50.0,
                adx_value=0.0
            )

        close = df["close"]
        high = df["high"]
        low = df["low"]

        current_bar = len(df) - 1

        # Check cooldown (only if requested - backtest handles externally)
        if check_cooldown and not self.check_cooldown(symbol, current_bar):
            return TradingSignal(
                signal=None,
                signal_type=SignalType.NONE,
                confidence=0.0,
                reason=f"Cooldown active ({self.cooldown_bars} bars)",
                momentum_value=0.0,
                ema_trend="NEUTRAL",
                rsi_value=50.0,
                adx_value=0.0
            )

        # =================================================================
        # Step 1: Calculate Momentum
        # =================================================================
        momentum = self.calculate_momentum(close, self.momentum_period)
        current_momentum = momentum.iloc[-1]

        if np.isnan(current_momentum):
            return TradingSignal(
                signal=None,
                signal_type=SignalType.NONE,
                confidence=0.0,
                reason="Invalid momentum calculation",
                momentum_value=0.0,
                ema_trend="NEUTRAL",
                rsi_value=50.0,
                adx_value=0.0
            )

        # Determine direction
        direction = "UP" if current_momentum > 0 else "DOWN"

        # Check momentum threshold
        momentum_met = abs(current_momentum) >= self.momentum_threshold

        # =================================================================
        # Step 2: Calculate Trend (EMA)
        # =================================================================
        ema_fast = self.calculate_ema(close, self.ema_fast_period)
        ema_slow = self.calculate_ema(close, self.ema_slow_period)

        ema_fast_val = ema_fast.iloc[-1]
        ema_slow_val = ema_slow.iloc[-1]
        current_price = close.iloc[-1]

        if ema_fast_val > ema_slow_val and current_price > ema_fast_val:
            ema_trend = "BULLISH"
        elif ema_fast_val < ema_slow_val and current_price < ema_fast_val:
            ema_trend = "BEARISH"
        else:
            ema_trend = "NEUTRAL"

        # Check trend alignment
        trend_aligned = (
            (direction == "UP" and ema_trend == "BULLISH") or
            (direction == "DOWN" and ema_trend == "BEARISH")
        )

        # =================================================================
        # Step 3: Calculate RSI
        # =================================================================
        rsi = self.calculate_rsi(close, self.rsi_period)
        current_rsi = rsi.iloc[-1]

        if np.isnan(current_rsi):
            current_rsi = 50.0

        # Check RSI filter
        rsi_ok_for_buy = current_rsi < self.rsi_max_buy
        rsi_ok_for_sell = current_rsi > self.rsi_min_sell

        # =================================================================
        # Step 4: Calculate ADX
        # =================================================================
        adx = self.calculate_adx(high, low, close, self.adx_period)
        current_adx = adx.iloc[-1]

        if np.isnan(current_adx):
            current_adx = 0.0

        # Check ADX threshold
        adx_ok = current_adx >= self.min_adx

        # =================================================================
        # Step 5: Generate Signal
        # =================================================================

        # No momentum spike - no signal
        if not momentum_met:
            return TradingSignal(
                signal=None,
                signal_type=SignalType.NONE,
                confidence=0.0,
                reason=f"No momentum spike (|{current_momentum:.3f}%| < {self.momentum_threshold}%)",
                momentum_value=current_momentum,
                ema_trend=ema_trend,
                rsi_value=current_rsi,
                adx_value=current_adx
            )

        # Momentum spike detected - check all conditions
        if direction == "UP":
            # BUY Signal Conditions
            if not trend_aligned:
                return TradingSignal(
                    signal=None,
                    signal_type=SignalType.NONE,
                    confidence=0.0,
                    reason=f"Trend not aligned (EMA trend: {ema_trend})",
                    momentum_value=current_momentum,
                    ema_trend=ema_trend,
                    rsi_value=current_rsi,
                    adx_value=current_adx
                )

            if not rsi_ok_for_buy:
                return TradingSignal(
                    signal=None,
                    signal_type=SignalType.NONE,
                    confidence=0.0,
                    reason=f"RSI overbought ({current_rsi:.1f} > {self.rsi_max_buy})",
                    momentum_value=current_momentum,
                    ema_trend=ema_trend,
                    rsi_value=current_rsi,
                    adx_value=current_adx
                )

            if not adx_ok:
                return TradingSignal(
                    signal=None,
                    signal_type=SignalType.NONE,
                    confidence=0.0,
                    reason=f"Weak trend (ADX {current_adx:.1f} < {self.min_adx})",
                    momentum_value=current_momentum,
                    ema_trend=ema_trend,
                    rsi_value=current_rsi,
                    adx_value=current_adx
                )

            # ALL CONDITIONS MET - BUY SIGNAL!
            # Calculate confidence based on signal strength
            confidence = self._calculate_confidence(
                current_momentum, current_adx, current_rsi, direction
            )

            # Determine signal type
            signal_type = (SignalType.MOMENTUM_STRONG
                          if abs(current_momentum) >= self.momentum_threshold * 2
                          else SignalType.MOMENTUM)

            # NOTE: Cooldown is now set by the trading engine AFTER trade execution
            # This prevents cooldown from blocking trades when HTF is not aligned

            return TradingSignal(
                signal="BUY",
                signal_type=signal_type,
                confidence=confidence,
                reason=f"MOMENTUM BUY: +{current_momentum:.3f}% | EMA:{ema_trend} | RSI:{current_rsi:.1f} | ADX:{current_adx:.1f}",
                momentum_value=current_momentum,
                ema_trend=ema_trend,
                rsi_value=current_rsi,
                adx_value=current_adx
            )

        else:
            # SELL Signal Conditions (direction == "DOWN")
            if not trend_aligned:
                return TradingSignal(
                    signal=None,
                    signal_type=SignalType.NONE,
                    confidence=0.0,
                    reason=f"Trend not aligned (EMA trend: {ema_trend})",
                    momentum_value=current_momentum,
                    ema_trend=ema_trend,
                    rsi_value=current_rsi,
                    adx_value=current_adx
                )

            if not rsi_ok_for_sell:
                return TradingSignal(
                    signal=None,
                    signal_type=SignalType.NONE,
                    confidence=0.0,
                    reason=f"RSI oversold ({current_rsi:.1f} < {self.rsi_min_sell})",
                    momentum_value=current_momentum,
                    ema_trend=ema_trend,
                    rsi_value=current_rsi,
                    adx_value=current_adx
                )

            if not adx_ok:
                return TradingSignal(
                    signal=None,
                    signal_type=SignalType.NONE,
                    confidence=0.0,
                    reason=f"Weak trend (ADX {current_adx:.1f} < {self.min_adx})",
                    momentum_value=current_momentum,
                    ema_trend=ema_trend,
                    rsi_value=current_rsi,
                    adx_value=current_adx
                )

            # ALL CONDITIONS MET - SELL SIGNAL!
            confidence = self._calculate_confidence(
                current_momentum, current_adx, current_rsi, direction
            )

            signal_type = (SignalType.MOMENTUM_STRONG
                          if abs(current_momentum) >= self.momentum_threshold * 2
                          else SignalType.MOMENTUM)

            # NOTE: Cooldown is now set by the trading engine AFTER trade execution
            # This prevents cooldown from blocking trades when HTF is not aligned

            return TradingSignal(
                signal="SELL",
                signal_type=signal_type,
                confidence=confidence,
                reason=f"MOMENTUM SELL: {current_momentum:.3f}% | EMA:{ema_trend} | RSI:{current_rsi:.1f} | ADX:{current_adx:.1f}",
                momentum_value=current_momentum,
                ema_trend=ema_trend,
                rsi_value=current_rsi,
                adx_value=current_adx
            )

    def can_dca(self, df: pd.DataFrame, position_side: str, dca_level: int) -> Tuple[bool, str]:
        """
        SMART DCA: Check if trend/momentum supports adding to position.

        STRICT REVERSAL DETECTION - Don't just add on noise!
        DCA should only execute when market shows REAL signs of reversal:
        - Momentum flipping in our direction
        - EMA starting to converge (for higher levels)
        - RSI not at extremes

        Args:
            df: DataFrame with OHLCV data
            position_side: "LONG" or "SHORT"
            dca_level: Current DCA level (1-4), higher levels need stronger confirmation

        Returns:
            Tuple of (can_dca: bool, reason: str)
        """
        if df is None or len(df) < max(self.ema_slow_period, self.adx_period) + 5:
            return False, "Insufficient data for DCA validation"

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate indicators
        ema_fast = self.calculate_ema(close, self.ema_fast_period)
        ema_slow = self.calculate_ema(close, self.ema_slow_period)
        momentum = self.calculate_momentum(close, self.momentum_period)
        rsi = self.calculate_rsi(close, self.rsi_period)
        adx = self.calculate_adx(high, low, close, self.adx_period)

        current_price = close.iloc[-1]
        ema_fast_val = ema_fast.iloc[-1]
        ema_slow_val = ema_slow.iloc[-1]
        prev_ema_fast = ema_fast.iloc[-2] if len(ema_fast) > 1 else ema_fast_val
        current_momentum = momentum.iloc[-1]
        prev_momentum = momentum.iloc[-2] if len(momentum) > 1 else current_momentum
        current_rsi = rsi.iloc[-1]
        current_adx = adx.iloc[-1]

        if np.isnan(current_momentum):
            current_momentum = 0.0
        if np.isnan(prev_momentum):
            prev_momentum = 0.0
        if np.isnan(current_rsi):
            current_rsi = 50.0
        if np.isnan(current_adx):
            current_adx = 0.0

        # Calculate momentum direction (is it improving?)
        momentum_improving = current_momentum > prev_momentum

        # Calculate EMA convergence (is fast EMA catching up to slow?)
        ema_gap_pct = (ema_fast_val - ema_slow_val) / ema_slow_val * 100

        # =================================================================
        # STRICTER DCA REQUIREMENTS BY LEVEL
        # Level 1: Basic reversal signs
        # Level 2: Momentum must be improving + not too strong against us
        # Level 3-4: Need clear reversal - momentum flipping OR EMA converging
        # =================================================================

        if position_side == "LONG":
            # For LONG positions, we need bullish reversal signs

            # Check 1: Strong bearish trend = BLOCK ALL DCA
            strong_bearish_trend = (
                ema_fast_val < ema_slow_val * 0.992 and  # Fast EMA 0.8%+ below slow
                current_adx > 20  # Decent trend strength
            )
            if strong_bearish_trend:
                return False, f"Strong bearish (EMA gap:{ema_gap_pct:.2f}% ADX:{current_adx:.1f})"

            # Check 2: RSI should show room to grow
            if current_rsi > 70:
                return False, f"RSI overbought ({current_rsi:.1f})"

            # Level-specific requirements
            if dca_level == 1:
                # Level 1: Just need momentum not dropping hard + slight improvement
                if current_momentum < -0.08:
                    return False, f"Still dropping hard ({current_momentum:.3f}%)"
                if not momentum_improving and current_momentum < 0:
                    return False, f"Momentum not improving ({prev_momentum:.3f}% -> {current_momentum:.3f}%)"
                return True, f"L1 OK: Mom improving {prev_momentum:.3f}%->{current_momentum:.3f}%"

            elif dca_level == 2:
                # Level 2: Need momentum improving AND not strongly negative
                if current_momentum < -0.05:
                    return False, f"Momentum too negative for L2 ({current_momentum:.3f}%)"
                if not momentum_improving:
                    return False, f"L2 needs momentum improvement ({prev_momentum:.3f}% -> {current_momentum:.3f}%)"
                # EMA should not be diverging further
                if ema_gap_pct < -0.5 and current_adx > 25:
                    return False, f"EMA still diverging ({ema_gap_pct:.2f}%)"
                return True, f"L2 OK: Mom {current_momentum:.3f}% improving, EMA gap:{ema_gap_pct:.2f}%"

            else:  # Level 3-4: STRICT - need clear reversal
                # Must have positive momentum OR EMA starting to converge
                has_positive_momentum = current_momentum > 0.05
                ema_converging = (ema_fast_val > prev_ema_fast) and (ema_gap_pct > -0.8)

                if not has_positive_momentum and not ema_converging:
                    return False, f"L{dca_level} needs reversal: Mom:{current_momentum:.3f}% EMA gap:{ema_gap_pct:.2f}%"

                # ADX should be weakening (trend losing strength) OR momentum clearly positive
                if current_adx > 30 and not has_positive_momentum:
                    return False, f"L{dca_level} trend too strong (ADX:{current_adx:.1f})"

                reason = []
                if has_positive_momentum:
                    reason.append(f"Mom:{current_momentum:.3f}%")
                if ema_converging:
                    reason.append(f"EMA converging")
                return True, f"L{dca_level} reversal: {', '.join(reason)}"

        else:  # SHORT position
            # For SHORT positions, we need bearish reversal signs

            # Check 1: Strong bullish trend = BLOCK ALL DCA
            strong_bullish_trend = (
                ema_fast_val > ema_slow_val * 1.008 and  # Fast EMA 0.8%+ above slow
                current_adx > 20  # Decent trend strength
            )
            if strong_bullish_trend:
                return False, f"Strong bullish (EMA gap:{ema_gap_pct:.2f}% ADX:{current_adx:.1f})"

            # Check 2: RSI should show room to drop
            if current_rsi < 30:
                return False, f"RSI oversold ({current_rsi:.1f})"

            # Level-specific requirements
            if dca_level == 1:
                # Level 1: Just need momentum not rallying hard + slight improvement
                # 0.3% is real rally, below that is noise/consolidation - ALLOW DCA
                if current_momentum > 0.30:
                    return False, f"Still rallying hard ({current_momentum:.3f}%)"
                if not (current_momentum < prev_momentum) and current_momentum > 0.15:
                    return False, f"Momentum not weakening ({prev_momentum:.3f}% -> {current_momentum:.3f}%)"
                return True, f"L1 OK: Mom weakening {prev_momentum:.3f}%->{current_momentum:.3f}%"

            elif dca_level == 2:
                # Level 2: Need momentum weakening AND not strongly positive
                # 0.2% is acceptable, only block if clearly rallying
                if current_momentum > 0.20:
                    return False, f"Momentum too positive for L2 ({current_momentum:.3f}%)"
                if not (current_momentum < prev_momentum) and current_momentum > 0.10:
                    return False, f"L2 needs momentum weakening ({prev_momentum:.3f}% -> {current_momentum:.3f}%)"
                # EMA should not be diverging further
                if ema_gap_pct > 0.5 and current_adx > 25:
                    return False, f"EMA still diverging ({ema_gap_pct:.2f}%)"
                return True, f"L2 OK: Mom {current_momentum:.3f}% weakening, EMA gap:{ema_gap_pct:.2f}%"

            else:  # Level 3-4: STRICT - need clear reversal
                # Must have negative momentum OR EMA starting to converge
                has_negative_momentum = current_momentum < -0.05
                ema_converging = (ema_fast_val < prev_ema_fast) and (ema_gap_pct < 0.8)

                if not has_negative_momentum and not ema_converging:
                    return False, f"L{dca_level} needs reversal: Mom:{current_momentum:.3f}% EMA gap:{ema_gap_pct:.2f}%"

                # ADX should be weakening (trend losing strength) OR momentum clearly negative
                if current_adx > 30 and not has_negative_momentum:
                    return False, f"L{dca_level} trend too strong (ADX:{current_adx:.1f})"

                reason = []
                if has_negative_momentum:
                    reason.append(f"Mom:{current_momentum:.3f}%")
                if ema_converging:
                    reason.append(f"EMA converging")
                return True, f"L{dca_level} reversal: {', '.join(reason)}"

    def _calculate_confidence(self, momentum: float, adx: float,
                               rsi: float, direction: str) -> float:
        """
        Calculate signal confidence (0.0 to 1.0)

        Factors:
        1. Momentum strength (higher = more confident)
        2. ADX strength (higher = more confident)
        3. RSI position (closer to 50 = more room to run)
        """
        # Base confidence
        base = 0.60

        # Momentum bonus (up to +0.15)
        momentum_bonus = min(0.15, abs(momentum) / self.momentum_threshold * 0.05)

        # ADX bonus (up to +0.15)
        adx_bonus = min(0.15, (adx - self.min_adx) / 30 * 0.15)

        # RSI bonus (closer to 50 = more room)
        if direction == "UP":
            rsi_room = (self.rsi_max_buy - rsi) / (self.rsi_max_buy - 50)
        else:
            rsi_room = (rsi - self.rsi_min_sell) / (50 - self.rsi_min_sell)
        rsi_bonus = min(0.10, max(0, rsi_room * 0.10))

        confidence = base + momentum_bonus + adx_bonus + rsi_bonus

        return min(0.95, max(0.50, confidence))


class MultiTimeframeMomentumSignal(MasterMomentumSignal):
    """
    Multi-Timeframe Momentum Signal Generator

    Adds higher timeframe confirmation to base momentum signals
    """

    def __init__(self):
        super().__init__()

    def check_htf_alignment(self, df_htf: pd.DataFrame, direction: str) -> bool:
        """
        Check if higher timeframe supports the direction

        Args:
            df_htf: Higher timeframe DataFrame (15m or 1h)
            direction: "UP" or "DOWN"

        Returns:
            True if HTF supports the direction
        """
        if df_htf is None or len(df_htf) < self.ema_slow_period + 5:
            return True  # Assume OK if no data

        close = df_htf["close"]

        ema_fast = self.calculate_ema(close, self.ema_fast_period)
        ema_slow = self.calculate_ema(close, self.ema_slow_period)

        ema_fast_val = ema_fast.iloc[-1]
        ema_slow_val = ema_slow.iloc[-1]

        if direction == "UP":
            return ema_fast_val > ema_slow_val
        else:
            return ema_fast_val < ema_slow_val

    def generate_signal_mtf(self, symbol: str, market_data: Dict[str, pd.DataFrame]) -> TradingSignal:
        """
        Generate signal with multi-timeframe confirmation

        Args:
            symbol: Trading pair
            market_data: Dict with timeframe keys ("1m", "5m", "15m", "1h")

        Returns:
            TradingSignal with MTF confirmation
        """
        # Get base signal from 1-minute data
        df_1m = market_data.get("1m")
        signal = self.generate_signal(symbol, df_1m)

        if signal.signal is None:
            return signal

        # Check HTF alignment
        direction = "UP" if signal.signal == "BUY" else "DOWN"

        df_15m = market_data.get("15m")
        df_1h = market_data.get("1h")

        htf_15m_ok = self.check_htf_alignment(df_15m, direction)
        htf_1h_ok = self.check_htf_alignment(df_1h, direction)

        if not htf_15m_ok or not htf_1h_ok:
            return TradingSignal(
                signal=None,
                signal_type=SignalType.NONE,
                confidence=0.0,
                reason=f"HTF not aligned (15m: {htf_15m_ok}, 1h: {htf_1h_ok})",
                momentum_value=signal.momentum_value,
                ema_trend=signal.ema_trend,
                rsi_value=signal.rsi_value,
                adx_value=signal.adx_value
            )

        # Both HTFs aligned - boost confidence slightly
        boosted_confidence = min(0.95, signal.confidence + 0.05)

        return TradingSignal(
            signal=signal.signal,
            signal_type=signal.signal_type,
            confidence=boosted_confidence,
            reason=signal.reason + " | HTF ALIGNED",
            momentum_value=signal.momentum_value,
            ema_trend=signal.ema_trend,
            rsi_value=signal.rsi_value,
            adx_value=signal.adx_value
        )


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)

    # Simulate trending market with momentum spike
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1min")
    base_price = 50000  # BTC price

    # Create trending data with a momentum spike at the end
    closes = [base_price]
    for i in range(99):
        if i > 90:  # Momentum spike at end
            change = np.random.uniform(0.001, 0.003)  # Strong up move
        else:
            change = np.random.uniform(-0.001, 0.0015)  # Slight uptrend
        closes.append(closes[-1] * (1 + change))

    df = pd.DataFrame({
        "open": closes,
        "high": [c * 1.001 for c in closes],
        "low": [c * 0.999 for c in closes],
        "close": closes,
        "volume": np.random.uniform(100, 1000, 100)
    }, index=dates)

    # Test signal generation
    signal_gen = MasterMomentumSignal()
    signal = signal_gen.generate_signal("BTCUSDT", df)

    print("\n" + "="*60)
    print("MOMENTUM SIGNAL TEST")
    print("="*60)
    print(f"Signal: {signal.signal}")
    print(f"Type: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Reason: {signal.reason}")
    print(f"Momentum: {signal.momentum_value:.3f}%")
    print(f"EMA Trend: {signal.ema_trend}")
    print(f"RSI: {signal.rsi_value:.1f}")
    print(f"ADX: {signal.adx_value:.1f}")
