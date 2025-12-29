"""
Forex Scalping Strategy - TREND-FOLLOWING V3 (Breakout + Pullback)

STRATEGY PHILOSOPHY:
TRADE WITH THE TREND, NOT AGAINST IT!
- Old strategy: Buy RSI<30 (catching falling knives) - WRONG
- New strategy: Buy pullbacks in UPTRENDS, Sell rallies in DOWNTRENDS - RIGHT

TWO ENTRY METHODS (both require trend alignment):

1. BREAKOUT ENTRY (Early momentum catch):
   - Price breaks above resistance/recent high + EMA alignment
   - Price breaks below support/recent low + EMA alignment
   - Get in at the START of a move

2. PULLBACK ENTRY (Better price in established trend):
   - UPTREND: Wait for RSI to pull back to 40-50, then buy
   - DOWNTREND: Wait for RSI to bounce to 50-60, then sell
   - Get in at a BETTER PRICE within the trend

TREND DETERMINATION (Higher Timeframe):
- 15min EMA9 > EMA21 + 30min EMA9 > EMA21 = UPTREND (only BUY)
- 15min EMA9 < EMA21 + 30min EMA9 < EMA21 = DOWNTREND (only SELL)
- Conflicting = NO TRADE

Key Features:
- ONLY trade WITH the higher timeframe trend
- Pair-specific TP/SL based on volatility
- Reduced position sizing (10% margin per trade for safety)
- Session-based cooldown
- Max 5 concurrent positions

TRADING SESSIONS (UTC):
- Tokyo:   00:00 - 09:00 UTC (Asia)
- London:  07:00 - 16:00 UTC (Europe)
- New York: 12:00 - 21:00 UTC (America)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from typing import Dict, List, Optional, Tuple
import sys
import os

# Import pair-specific settings
try:
    from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params
except ImportError:
    # Fallback if import fails
    def get_scalping_params(pair, account_balance=1000):
        return {
            "take_profit_pct": 0.002,
            "stop_loss_pct": 0.0012,
            "trailing_stop_trigger": 0.0012,
            "trailing_stop_distance": 0.0006,
            "tp_pips": 20,
            "sl_pips": 12
        }


class ForexScalpingStrategy:
    """
    Scalping strategy for forex - generates more frequent trades
    With session-based cooldown and enhanced pullback confirmation
    """

    # Trading sessions (UTC hours)
    SESSIONS = {
        "TOKYO": {"start": 0, "end": 9},      # 00:00 - 09:00 UTC
        "LONDON": {"start": 7, "end": 16},    # 07:00 - 16:00 UTC
        "NEW_YORK": {"start": 12, "end": 21}  # 12:00 - 21:00 UTC
    }

    # Session order for cooldown (after Tokyo -> wait for London, after London -> wait for NY)
    SESSION_ORDER = ["TOKYO", "LONDON", "NEW_YORK"]

    def __init__(
        self,
        instruments: List[str],
        max_trades_per_day: int = 10,
        daily_profit_target: float = 0.03,
        trade_size_pct: float = 0.15,
        take_profit_pct: float = 0.005,  # 50 pips
        stop_loss_pct: float = 0.003,  # 30 pips
        trailing_stop_trigger: float = 0.003,
        trailing_stop_distance: float = 0.002,
        require_htf_strict: bool = False,  # KEY: Allow HTF NEUTRAL
        pullback_required: bool = False,  # KEY: Don't wait for pullback
        min_consensus_score: int = 1,  # KEY: Only need 1 method
        session_cooldown: bool = True  # NEW: Enable session-based cooldown
    ):
        self.instruments = instruments
        self.max_trades_per_day = max_trades_per_day
        self.daily_profit_target = daily_profit_target
        self.trade_size_pct = trade_size_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_trigger = trailing_stop_trigger
        self.trailing_stop_distance = trailing_stop_distance
        self.require_htf_strict = require_htf_strict
        self.pullback_required = pullback_required
        self.min_consensus_score = min_consensus_score
        self.session_cooldown = session_cooldown

        # Track last trade session per symbol (for cooldown)
        # Format: {instrument: {"session": "TOKYO", "date": date_obj}}
        self.last_trade_session = {}

        print("[STRATEGY] ========================================")
        print("[STRATEGY] TREND-FOLLOWING V3 (Breakout + Pullback)")
        print("[STRATEGY] TRADE WITH THE TREND, NOT AGAINST IT!")
        print("[STRATEGY] ========================================")
        print(f"[STRATEGY] Max trades/day: {max_trades_per_day}")
        print(f"[STRATEGY] Entry Methods:")
        print(f"[STRATEGY]   1. BREAKOUT: Price breaks key level + trend aligned")
        print(f"[STRATEGY]   2. PULLBACK: RSI retraces to 40-60 zone in trend")
        print(f"[STRATEGY] Position size: 10% margin @ 20:1 leverage")
        print(f"[STRATEGY] RULE: Only BUY in uptrends, only SELL in downtrends")
        if session_cooldown:
            print(f"[STRATEGY] Time-based cooldown: ENABLED")
            print(f"[STRATEGY]   - Tokyo: 30min | London: 20min | New York: 15min")
            print(f"[STRATEGY]   - Multiple trades per session allowed!")

    def get_current_session(self, now: datetime = None) -> str:
        """
        Get the current trading session based on UTC time.

        Sessions (UTC):
        - Tokyo:    00:00 - 09:00 (but we use 00:00-07:00 as pure Tokyo)
        - London:   07:00 - 16:00 (but we use 09:00-12:00 as pure London)
        - New York: 12:00 - 21:00 (12:00-21:00 as pure NY)
        - Overlap periods are assigned to the "newer" session

        Returns:
            "TOKYO", "LONDON", "NEW_YORK", or "CLOSED"
        """
        if now is None:
            now = datetime.now(pytz.UTC)

        hour = now.hour

        # Determine primary session (no overlap logic, cleaner boundaries)
        if 0 <= hour < 7:
            return "TOKYO"
        elif 7 <= hour < 12:
            return "LONDON"
        elif 12 <= hour < 21:
            return "NEW_YORK"
        else:  # 21-24
            return "CLOSED"

    def is_session_cooldown_active(self, instrument: str, now: datetime = None) -> Tuple[bool, str]:
        """
        Check if symbol is in time-based cooldown (NOT session-based anymore).

        NEW RULE: Allow multiple trades per session with a TIME-BASED cooldown:
        - After a trade, wait X minutes before trading the same symbol again
        - Tokyo/London: 30 min cooldown (less volatility)
        - New York: 15 min cooldown (more setups, higher volatility)

        This allows capturing multiple good setups during active sessions!

        Args:
            instrument: Currency pair
            now: Current datetime (UTC)

        Returns:
            (is_in_cooldown, reason_string)
        """
        if not self.session_cooldown:
            return False, ""

        if now is None:
            now = datetime.now(pytz.UTC)

        current_session = self.get_current_session(now)

        if current_session == "CLOSED":
            return True, "Market closed (21:00-00:00 UTC)"

        # Check if we have a record for this instrument
        if instrument not in self.last_trade_session:
            return False, ""

        last_trade = self.last_trade_session[instrument]
        last_time = last_trade.get("time")
        last_date = last_trade["date"]

        # If traded on a different day, no cooldown
        if last_date != now.date():
            return False, ""

        # If no timestamp recorded (old format), allow trade
        if last_time is None:
            return False, ""

        # Calculate time since last trade
        time_since_trade = (now - last_time).total_seconds() / 60  # in minutes

        # Session-specific cooldown periods (in minutes)
        cooldown_periods = {
            "TOKYO": 30,      # 30 min cooldown - less volatile
            "LONDON": 20,     # 20 min cooldown - moderate
            "NEW_YORK": 15    # 15 min cooldown - most active, more setups
        }

        cooldown_minutes = cooldown_periods.get(current_session, 20)

        if time_since_trade < cooldown_minutes:
            remaining = cooldown_minutes - time_since_trade
            return True, f"Cooldown: {remaining:.0f}min remaining (traded {time_since_trade:.0f}min ago)"

        # Cooldown expired - can trade again
        return False, ""

    def record_trade_session(self, instrument: str, now: datetime = None):
        """
        Record that a trade was made for time-based cooldown tracking.

        Call this after placing a successful trade.
        Now stores the exact timestamp for time-based cooldown.
        """
        if now is None:
            now = datetime.now(pytz.UTC)

        session = self.get_current_session(now)
        self.last_trade_session[instrument] = {
            "session": session,
            "date": now.date(),
            "time": now  # Store exact timestamp for time-based cooldown
        }

    def reset_daily_cooldowns(self):
        """Reset all session cooldowns (call at start of new trading day)"""
        self.last_trade_session = {}

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze_momentum(self, df_1min: pd.DataFrame) -> Dict:
        """
        Quick momentum analysis on 1-min data
        More lenient than multi-timeframe version
        """
        if len(df_1min) < 20:
            return {"signal": "NEUTRAL", "score": 0}

        close = df_1min['close']
        last_10 = df_1min.tail(10)

        score = 0

        # 1. Last 5 candles color
        last_5 = last_10.tail(5)
        green_count = (last_5['close'] > last_5['open']).sum()
        red_count = (last_5['close'] < last_5['open']).sum()

        if green_count >= 4:
            score += 2
        elif green_count >= 3:
            score += 1
        elif red_count >= 4:
            score -= 2
        elif red_count >= 3:
            score -= 1

        # 2. Price momentum
        price_change = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11]
        if price_change > 0.0005:  # 0.05% = 5 pips
            score += 2
        elif price_change > 0.0002:  # 0.02% = 2 pips
            score += 1
        elif price_change < -0.0005:
            score -= 2
        elif price_change < -0.0002:
            score -= 1

        # 3. EMA trend
        ema9 = self.calculate_ema(close, 9).iloc[-1]
        if close.iloc[-1] > ema9:
            score += 1
        else:
            score -= 1

        # Determine signal
        if score >= 2:
            signal = "BULLISH"
        elif score <= -2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {"signal": signal, "score": score, "max_score": 5}

    def check_htf_trend(self, df_15min: pd.DataFrame, df_30min: pd.DataFrame) -> Dict:
        """
        Check higher timeframe trend using EMA9 vs EMA21 crossover.
        This is the PRIMARY trend filter - we ONLY trade in this direction!

        UPTREND: EMA9 > EMA21 on both 15min AND 30min → Only take BUY signals
        DOWNTREND: EMA9 < EMA21 on both 15min AND 30min → Only take SELL signals
        NEUTRAL/CONFLICTING: No trade
        """
        if len(df_15min) < 25 or len(df_30min) < 25:
            return {"trend": "NEUTRAL", "15min": "UNKNOWN", "30min": "UNKNOWN", "strength": 0}

        def get_trend_with_strength(df: pd.DataFrame) -> Tuple[str, float]:
            close = df['close']
            ema9 = self.calculate_ema(close, 9)
            ema21 = self.calculate_ema(close, 21)

            current_ema9 = ema9.iloc[-1]
            current_ema21 = ema21.iloc[-1]
            current_price = close.iloc[-1]

            # Calculate EMA separation as percentage
            ema_separation = ((current_ema9 - current_ema21) / current_ema21) * 100

            # Check if EMAs are crossed and price confirms
            if current_ema9 > current_ema21 and current_price > current_ema9:
                return "BULLISH", abs(ema_separation)
            elif current_ema9 < current_ema21 and current_price < current_ema9:
                return "BEARISH", abs(ema_separation)
            elif current_ema9 > current_ema21:
                return "WEAK_BULLISH", abs(ema_separation)
            elif current_ema9 < current_ema21:
                return "WEAK_BEARISH", abs(ema_separation)
            else:
                return "NEUTRAL", 0

        trend_15min, strength_15 = get_trend_with_strength(df_15min)
        trend_30min, strength_30 = get_trend_with_strength(df_30min)

        # Both timeframes must agree for a valid trend
        if trend_15min in ["BULLISH", "WEAK_BULLISH"] and trend_30min in ["BULLISH", "WEAK_BULLISH"]:
            # Strong bullish if both are full BULLISH
            if trend_15min == "BULLISH" and trend_30min == "BULLISH":
                overall_trend = "STRONG_UPTREND"
            else:
                overall_trend = "UPTREND"
        elif trend_15min in ["BEARISH", "WEAK_BEARISH"] and trend_30min in ["BEARISH", "WEAK_BEARISH"]:
            # Strong bearish if both are full BEARISH
            if trend_15min == "BEARISH" and trend_30min == "BEARISH":
                overall_trend = "STRONG_DOWNTREND"
            else:
                overall_trend = "DOWNTREND"
        else:
            overall_trend = "NEUTRAL"

        return {
            "trend": overall_trend,
            "15min": trend_15min,
            "30min": trend_30min,
            "strength": (strength_15 + strength_30) / 2,
            "tradeable_direction": "BUY" if "UPTREND" in overall_trend else ("SELL" if "DOWNTREND" in overall_trend else None)
        }

    def detect_strong_trend(self, df_15min: pd.DataFrame, df_5min: pd.DataFrame) -> Dict:
        """
        Detect if market is in a STRONG trend (avoid early entries).

        Strong downtrend indicators:
        - 5+ consecutive red candles on 15min
        - Price far below EMA20 (>0.3%)
        - RSI < 25 (extremely oversold but still dropping)
        - ADX > 30 (strong trend)

        Strong uptrend indicators:
        - 5+ consecutive green candles on 15min
        - Price far above EMA20 (>0.3%)
        - RSI > 75 (extremely overbought but still rising)

        Returns:
            {
                "is_strong_trend": bool,
                "trend_direction": "STRONG_DOWN" / "STRONG_UP" / "NORMAL",
                "consecutive_candles": int,
                "ema_distance_pct": float,
                "rsi": float,
                "requires_pullback": bool,
                "reason": str
            }
        """
        if len(df_15min) < 30 or len(df_5min) < 20:
            return {
                "is_strong_trend": False,
                "trend_direction": "NORMAL",
                "consecutive_candles": 0,
                "ema_distance_pct": 0,
                "rsi": 50,
                "requires_pullback": False,
                "reason": "Insufficient data"
            }

        close_15 = df_15min['close']
        open_15 = df_15min['open']
        close_5 = df_5min['close']

        # Calculate indicators
        ema20 = self.calculate_ema(close_15, 20).iloc[-1]
        rsi = self.calculate_rsi(close_15, 14).iloc[-1]
        current_price = close_15.iloc[-1]

        # EMA distance (percentage)
        ema_distance_pct = ((current_price - ema20) / ema20) * 100

        # Count consecutive candles
        consecutive_red = 0
        consecutive_green = 0

        for i in range(-1, -11, -1):  # Last 10 candles
            if close_15.iloc[i] < open_15.iloc[i]:
                if consecutive_green == 0:
                    consecutive_red += 1
                else:
                    break
            elif close_15.iloc[i] > open_15.iloc[i]:
                if consecutive_red == 0:
                    consecutive_green += 1
                else:
                    break
            else:
                break

        # Check for STRONG DOWNTREND
        strong_down_score = 0
        reasons = []

        if consecutive_red >= 4:
            strong_down_score += 2
            reasons.append(f"{consecutive_red} consecutive red candles")
        if consecutive_red >= 6:
            strong_down_score += 1

        if ema_distance_pct < -0.25:
            strong_down_score += 2
            reasons.append(f"Price {ema_distance_pct:.2f}% below EMA20")
        if ema_distance_pct < -0.4:
            strong_down_score += 1

        # AGGRESSIVE RSI CHECK - extreme RSI values alone should trigger strong trend
        if rsi < 25:
            strong_down_score += 2
            reasons.append(f"RSI extremely oversold at {rsi:.1f}")
        elif rsi < 30:
            strong_down_score += 1
            reasons.append(f"RSI oversold at {rsi:.1f}")

        if rsi < 20:
            strong_down_score += 2  # Extra weight for extreme values
            reasons.append(f"RSI CRITICAL LOW ({rsi:.1f}) - high risk of catching falling knife")

        # Check for STRONG UPTREND
        strong_up_score = 0
        up_reasons = []

        if consecutive_green >= 4:
            strong_up_score += 2
            up_reasons.append(f"{consecutive_green} consecutive green candles")
        if consecutive_green >= 6:
            strong_up_score += 1

        if ema_distance_pct > 0.25:
            strong_up_score += 2
            up_reasons.append(f"Price {ema_distance_pct:.2f}% above EMA20")
        if ema_distance_pct > 0.4:
            strong_up_score += 1

        # AGGRESSIVE RSI CHECK - extreme RSI values alone should trigger strong trend
        if rsi > 75:
            strong_up_score += 2
            up_reasons.append(f"RSI extremely overbought at {rsi:.1f}")
        elif rsi > 70:
            strong_up_score += 1
            up_reasons.append(f"RSI overbought at {rsi:.1f}")

        if rsi > 80:
            strong_up_score += 2  # Extra weight for extreme values
            up_reasons.append(f"RSI CRITICAL HIGH ({rsi:.1f}) - high risk of catching falling knife")

        # Determine result
        if strong_down_score >= 3:
            return {
                "is_strong_trend": True,
                "trend_direction": "STRONG_DOWN",
                "consecutive_candles": consecutive_red,
                "ema_distance_pct": ema_distance_pct,
                "rsi": rsi,
                "requires_pullback": True,
                "reason": f"STRONG DOWNTREND: {', '.join(reasons)} - WAIT FOR PULLBACK"
            }
        elif strong_up_score >= 3:
            return {
                "is_strong_trend": True,
                "trend_direction": "STRONG_UP",
                "consecutive_candles": consecutive_green,
                "ema_distance_pct": ema_distance_pct,
                "rsi": rsi,
                "requires_pullback": True,
                "reason": f"STRONG UPTREND: {', '.join(up_reasons)} - WAIT FOR PULLBACK"
            }
        else:
            return {
                "is_strong_trend": False,
                "trend_direction": "NORMAL",
                "consecutive_candles": max(consecutive_red, consecutive_green),
                "ema_distance_pct": ema_distance_pct,
                "rsi": rsi,
                "requires_pullback": False,
                "reason": "Normal market conditions"
            }

    def check_pullback_in_trend(self, df_5min: pd.DataFrame, df_1min: pd.DataFrame, trend_direction: str) -> Dict:
        """
        Check if a pullback has occurred in a strong trend, making it safe to enter.

        For STRONG_DOWN (looking to BUY):
        - Wait for at least 2-3 green candles (bounce)
        - RSI should recover above 30
        - Price should retrace at least 30% of the recent drop

        For STRONG_UP (looking to SELL):
        - Wait for at least 2-3 red candles (pullback)
        - RSI should drop below 70
        - Price should retrace at least 30% of the recent rally

        Args:
            df_5min: 5-minute candle data
            df_1min: 1-minute candle data
            trend_direction: "STRONG_DOWN" or "STRONG_UP"

        Returns:
            {
                "pullback_confirmed": bool,
                "recovery_candles": int,
                "retracement_pct": float,
                "rsi_recovered": bool,
                "safe_to_enter": bool,
                "reason": str
            }
        """
        if len(df_5min) < 20 or len(df_1min) < 20:
            return {
                "pullback_confirmed": False,
                "recovery_candles": 0,
                "retracement_pct": 0,
                "rsi_recovered": False,
                "safe_to_enter": False,
                "reason": "Insufficient data"
            }

        close_5 = df_5min['close']
        open_5 = df_5min['open']
        close_1 = df_1min['close']
        open_1 = df_1min['open']

        rsi_5 = self.calculate_rsi(close_5, 14).iloc[-1]
        rsi_1 = self.calculate_rsi(close_1, 14).iloc[-1]

        # Recent price extremes (last 20 candles on 5min)
        recent_high = df_5min['high'].tail(20).max()
        recent_low = df_5min['low'].tail(20).min()
        current_price = close_5.iloc[-1]
        price_range = recent_high - recent_low

        if trend_direction == "STRONG_DOWN":
            # Looking to BUY after pullback in downtrend
            # Count recovery (green) candles
            recovery_candles = 0
            for i in range(-1, -6, -1):
                if close_5.iloc[i] > open_5.iloc[i]:
                    recovery_candles += 1
                else:
                    break

            # Check retracement from low
            retracement = (current_price - recent_low) / price_range if price_range > 0 else 0
            retracement_pct = retracement * 100

            # RSI recovery
            rsi_recovered = rsi_5 > 32 and rsi_1 > 35

            # Check 1-min confirmation (last candle should be green)
            last_1min_green = close_1.iloc[-1] > open_1.iloc[-1]

            # Safe to enter conditions
            conditions_met = 0
            reasons = []

            if recovery_candles >= 2:
                conditions_met += 1
                reasons.append(f"{recovery_candles} green candles")

            if retracement_pct >= 25:
                conditions_met += 1
                reasons.append(f"{retracement_pct:.1f}% retracement")

            if rsi_recovered:
                conditions_met += 1
                reasons.append(f"RSI recovered to {rsi_5:.1f}")

            if last_1min_green:
                conditions_met += 1
                reasons.append("1min confirmation (green)")

            safe_to_enter = conditions_met >= 3

            return {
                "pullback_confirmed": recovery_candles >= 2,
                "recovery_candles": recovery_candles,
                "retracement_pct": retracement_pct,
                "rsi_recovered": rsi_recovered,
                "safe_to_enter": safe_to_enter,
                "reason": f"PULLBACK CHECK: {', '.join(reasons)}" if reasons else "No pullback yet"
            }

        elif trend_direction == "STRONG_UP":
            # Looking to SELL after pullback in uptrend
            # Count pullback (red) candles
            pullback_candles = 0
            for i in range(-1, -6, -1):
                if close_5.iloc[i] < open_5.iloc[i]:
                    pullback_candles += 1
                else:
                    break

            # Check retracement from high
            retracement = (recent_high - current_price) / price_range if price_range > 0 else 0
            retracement_pct = retracement * 100

            # RSI pullback
            rsi_recovered = rsi_5 < 68 and rsi_1 < 65

            # Check 1-min confirmation (last candle should be red)
            last_1min_red = close_1.iloc[-1] < open_1.iloc[-1]

            # Safe to enter conditions
            conditions_met = 0
            reasons = []

            if pullback_candles >= 2:
                conditions_met += 1
                reasons.append(f"{pullback_candles} red candles")

            if retracement_pct >= 25:
                conditions_met += 1
                reasons.append(f"{retracement_pct:.1f}% retracement")

            if rsi_recovered:
                conditions_met += 1
                reasons.append(f"RSI pulled back to {rsi_5:.1f}")

            if last_1min_red:
                conditions_met += 1
                reasons.append("1min confirmation (red)")

            safe_to_enter = conditions_met >= 3

            return {
                "pullback_confirmed": pullback_candles >= 2,
                "recovery_candles": pullback_candles,
                "retracement_pct": retracement_pct,
                "rsi_recovered": rsi_recovered,
                "safe_to_enter": safe_to_enter,
                "reason": f"PULLBACK CHECK: {', '.join(reasons)}" if reasons else "No pullback yet"
            }

        return {
            "pullback_confirmed": False,
            "recovery_candles": 0,
            "retracement_pct": 0,
            "rsi_recovered": False,
            "safe_to_enter": False,
            "reason": "Unknown trend direction"
        }

    def detect_breakout(self, df_5min: pd.DataFrame, df_15min: pd.DataFrame, trend_direction: str) -> Dict:
        """
        Detect breakout opportunities that align with the higher timeframe trend.

        BREAKOUT BUY (in UPTREND):
        - Price breaks above recent swing high (last 20 candles)
        - Volume/momentum confirmation (current candle is strong green)
        - EMA9 > EMA21 on 5min confirms momentum

        BREAKOUT SELL (in DOWNTREND):
        - Price breaks below recent swing low (last 20 candles)
        - Volume/momentum confirmation (current candle is strong red)
        - EMA9 < EMA21 on 5min confirms momentum

        Args:
            df_5min: 5-minute candle data
            df_15min: 15-minute candle data for key levels
            trend_direction: "BUY" or "SELL" (from HTF trend)

        Returns:
            {
                "breakout_detected": bool,
                "breakout_type": "RESISTANCE_BREAK" / "SUPPORT_BREAK" / None,
                "breakout_level": float,
                "momentum_confirmed": bool,
                "entry_signal": bool,
                "reason": str
            }
        """
        if len(df_5min) < 25 or len(df_15min) < 25:
            return {
                "breakout_detected": False,
                "breakout_type": None,
                "breakout_level": 0,
                "momentum_confirmed": False,
                "entry_signal": False,
                "reason": "Insufficient data"
            }

        close_5 = df_5min['close']
        open_5 = df_5min['open']
        high_5 = df_5min['high']
        low_5 = df_5min['low']

        current_price = close_5.iloc[-1]
        current_open = open_5.iloc[-1]
        prev_close = close_5.iloc[-2]

        # Calculate EMAs on 5min for momentum confirmation
        ema9_5 = self.calculate_ema(close_5, 9).iloc[-1]
        ema21_5 = self.calculate_ema(close_5, 21).iloc[-1]

        # Find recent swing high/low (last 20 candles, excluding last 2)
        lookback_high = high_5.iloc[-22:-2]
        lookback_low = low_5.iloc[-22:-2]
        recent_high = lookback_high.max()
        recent_low = lookback_low.min()

        # Current candle characteristics
        candle_body = abs(current_price - current_open)
        candle_range = high_5.iloc[-1] - low_5.iloc[-1]
        is_strong_green = current_price > current_open and candle_body > candle_range * 0.5
        is_strong_red = current_price < current_open and candle_body > candle_range * 0.5

        reasons = []

        if trend_direction == "BUY":
            # Looking for BREAKOUT above resistance in uptrend
            breakout_detected = current_price > recent_high and prev_close <= recent_high
            momentum_confirmed = ema9_5 > ema21_5 and is_strong_green

            if breakout_detected:
                reasons.append(f"Price broke above {recent_high:.5f}")
            if ema9_5 > ema21_5:
                reasons.append("5min EMA9 > EMA21")
            if is_strong_green:
                reasons.append("Strong green candle")

            entry_signal = breakout_detected and momentum_confirmed

            return {
                "breakout_detected": breakout_detected,
                "breakout_type": "RESISTANCE_BREAK" if breakout_detected else None,
                "breakout_level": recent_high,
                "momentum_confirmed": momentum_confirmed,
                "entry_signal": entry_signal,
                "reason": f"BREAKOUT BUY: {', '.join(reasons)}" if reasons else "No breakout"
            }

        elif trend_direction == "SELL":
            # Looking for BREAKOUT below support in downtrend
            breakout_detected = current_price < recent_low and prev_close >= recent_low
            momentum_confirmed = ema9_5 < ema21_5 and is_strong_red

            if breakout_detected:
                reasons.append(f"Price broke below {recent_low:.5f}")
            if ema9_5 < ema21_5:
                reasons.append("5min EMA9 < EMA21")
            if is_strong_red:
                reasons.append("Strong red candle")

            entry_signal = breakout_detected and momentum_confirmed

            return {
                "breakout_detected": breakout_detected,
                "breakout_type": "SUPPORT_BREAK" if breakout_detected else None,
                "breakout_level": recent_low,
                "momentum_confirmed": momentum_confirmed,
                "entry_signal": entry_signal,
                "reason": f"BREAKOUT SELL: {', '.join(reasons)}" if reasons else "No breakout"
            }

        return {
            "breakout_detected": False,
            "breakout_type": None,
            "breakout_level": 0,
            "momentum_confirmed": False,
            "entry_signal": False,
            "reason": "Invalid trend direction"
        }

    def detect_pullback_entry(self, df_5min: pd.DataFrame, df_1min: pd.DataFrame, trend_direction: str) -> Dict:
        """
        Detect pullback entry opportunities within an established trend.

        PULLBACK BUY (in UPTREND):
        - RSI pulled back to 40-55 zone (not oversold, just retraced)
        - Price near or touching EMA21 (dynamic support)
        - Current candle turning green (reversal signal)
        - Higher lows pattern forming

        PULLBACK SELL (in DOWNTREND):
        - RSI bounced to 45-60 zone (not overbought, just bounced)
        - Price near or touching EMA21 (dynamic resistance)
        - Current candle turning red (reversal signal)
        - Lower highs pattern forming

        Args:
            df_5min: 5-minute candle data
            df_1min: 1-minute candle data for precise entry
            trend_direction: "BUY" or "SELL" (from HTF trend)

        Returns:
            {
                "pullback_entry": bool,
                "rsi_in_zone": bool,
                "price_at_ema": bool,
                "reversal_candle": bool,
                "pattern_confirmed": bool,
                "entry_signal": bool,
                "rsi_value": float,
                "reason": str
            }
        """
        if len(df_5min) < 25 or len(df_1min) < 10:
            return {
                "pullback_entry": False,
                "rsi_in_zone": False,
                "price_at_ema": False,
                "reversal_candle": False,
                "pattern_confirmed": False,
                "entry_signal": False,
                "rsi_value": 50,
                "reason": "Insufficient data"
            }

        close_5 = df_5min['close']
        open_5 = df_5min['open']
        high_5 = df_5min['high']
        low_5 = df_5min['low']
        close_1 = df_1min['close']
        open_1 = df_1min['open']

        current_price = close_5.iloc[-1]
        current_open = open_5.iloc[-1]

        # Calculate indicators
        rsi = self.calculate_rsi(close_5, 14).iloc[-1]
        ema21 = self.calculate_ema(close_5, 21).iloc[-1]
        ema9 = self.calculate_ema(close_5, 9).iloc[-1]

        # Distance from EMA21 (percentage)
        ema_distance = abs(current_price - ema21) / ema21 * 100

        # 1min candle for precise entry
        last_1min_green = close_1.iloc[-1] > open_1.iloc[-1]
        last_1min_red = close_1.iloc[-1] < open_1.iloc[-1]

        # 5min candle color
        candle_5min_green = current_price > current_open
        candle_5min_red = current_price < current_open

        reasons = []
        score = 0

        if trend_direction == "BUY":
            # PULLBACK BUY in UPTREND
            # RSI should be in 35-55 zone (pulled back but not oversold)
            rsi_in_zone = 35 <= rsi <= 55

            # Price should be near EMA21 (within 0.15%) or between EMA9 and EMA21
            price_at_ema = ema_distance < 0.15 or (ema21 <= current_price <= ema9)

            # Reversal candle (turning green)
            reversal_candle = candle_5min_green and last_1min_green

            # Higher lows pattern (last 3 candle lows ascending)
            if len(low_5) >= 3:
                higher_lows = low_5.iloc[-1] > low_5.iloc[-2] and low_5.iloc[-2] > low_5.iloc[-3]
            else:
                higher_lows = False

            if rsi_in_zone:
                score += 2
                reasons.append(f"RSI in buy zone ({rsi:.1f})")
            if price_at_ema:
                score += 2
                reasons.append(f"Price near EMA21 support")
            if reversal_candle:
                score += 2
                reasons.append("Green reversal candle")
            if higher_lows:
                score += 1
                reasons.append("Higher lows pattern")

            entry_signal = score >= 4 and rsi_in_zone and reversal_candle

            return {
                "pullback_entry": True,
                "rsi_in_zone": rsi_in_zone,
                "price_at_ema": price_at_ema,
                "reversal_candle": reversal_candle,
                "pattern_confirmed": higher_lows,
                "entry_signal": entry_signal,
                "rsi_value": rsi,
                "score": score,
                "reason": f"PULLBACK BUY: {', '.join(reasons)}" if reasons else "No pullback setup"
            }

        elif trend_direction == "SELL":
            # PULLBACK SELL in DOWNTREND
            # RSI should be in 45-65 zone (bounced but not overbought)
            rsi_in_zone = 45 <= rsi <= 65

            # Price should be near EMA21 (within 0.15%) or between EMA21 and EMA9
            price_at_ema = ema_distance < 0.15 or (ema9 <= current_price <= ema21)

            # Reversal candle (turning red)
            reversal_candle = candle_5min_red and last_1min_red

            # Lower highs pattern (last 3 candle highs descending)
            if len(high_5) >= 3:
                lower_highs = high_5.iloc[-1] < high_5.iloc[-2] and high_5.iloc[-2] < high_5.iloc[-3]
            else:
                lower_highs = False

            if rsi_in_zone:
                score += 2
                reasons.append(f"RSI in sell zone ({rsi:.1f})")
            if price_at_ema:
                score += 2
                reasons.append(f"Price near EMA21 resistance")
            if reversal_candle:
                score += 2
                reasons.append("Red reversal candle")
            if lower_highs:
                score += 1
                reasons.append("Lower highs pattern")

            entry_signal = score >= 4 and rsi_in_zone and reversal_candle

            return {
                "pullback_entry": True,
                "rsi_in_zone": rsi_in_zone,
                "price_at_ema": price_at_ema,
                "reversal_candle": reversal_candle,
                "pattern_confirmed": lower_highs,
                "entry_signal": entry_signal,
                "rsi_value": rsi,
                "score": score,
                "reason": f"PULLBACK SELL: {', '.join(reasons)}" if reasons else "No pullback setup"
            }

        return {
            "pullback_entry": False,
            "rsi_in_zone": False,
            "price_at_ema": False,
            "reversal_candle": False,
            "pattern_confirmed": False,
            "entry_signal": False,
            "rsi_value": 50,
            "reason": "Invalid trend direction"
        }

    # List of HIGH VOLATILITY pairs that need extra validation
    HIGH_VOLATILITY_PAIRS = ["GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD"]

    def validate_high_volatility_entry(
        self,
        instrument: str,
        df_5min: pd.DataFrame,
        df_15min: pd.DataFrame,
        trend_direction: str
    ) -> Tuple[bool, str]:
        """
        EXTRA VALIDATION for high volatility pairs (GBP, JPY, AUD, NZD).

        These pairs are prone to:
        - Sudden spikes/whipsaws
        - False breakouts
        - Wide spreads during news

        EXTRA REQUIREMENTS for entry:
        1. STRONG trend (not just weak) - both timeframes must show full BULLISH/BEARISH
        2. ADX > 25 (trending market, not choppy)
        3. No recent large spike (last 3 candles not > 2x average)
        4. Multiple candle confirmation (2+ candles in direction)

        Returns:
            (is_valid, reason)
        """
        if instrument not in self.HIGH_VOLATILITY_PAIRS:
            return True, "Not a high volatility pair"

        if len(df_5min) < 25 or len(df_15min) < 25:
            return False, "Insufficient data for volatility check"

        close_5 = df_5min['close']
        high_5 = df_5min['high']
        low_5 = df_5min['low']
        open_5 = df_5min['open']

        # Check 1: Calculate ATR-like measure (average candle range)
        candle_ranges = high_5 - low_5
        avg_range = candle_ranges.iloc[-20:-3].mean()  # Average of last 20 candles (excluding last 3)
        recent_max_range = candle_ranges.iloc[-3:].max()  # Max of last 3 candles

        # If recent candle is > 2.5x average, market is too volatile
        if recent_max_range > avg_range * 2.5:
            return False, f"HIGH VOLATILITY: Recent spike detected ({recent_max_range:.5f} > 2.5x avg {avg_range:.5f})"

        # Check 2: Multiple candle confirmation in trend direction
        if trend_direction == "BUY":
            # Need at least 2 of last 3 candles to be green
            green_count = sum(1 for i in range(-3, 0) if close_5.iloc[i] > open_5.iloc[i])
            if green_count < 2:
                return False, f"HIGH VOLATILITY: Need 2+ green candles, only {green_count}"

            # Last candle must be green
            if close_5.iloc[-1] <= open_5.iloc[-1]:
                return False, "HIGH VOLATILITY: Last candle must be green for BUY"

        elif trend_direction == "SELL":
            # Need at least 2 of last 3 candles to be red
            red_count = sum(1 for i in range(-3, 0) if close_5.iloc[i] < open_5.iloc[i])
            if red_count < 2:
                return False, f"HIGH VOLATILITY: Need 2+ red candles, only {red_count}"

            # Last candle must be red
            if close_5.iloc[-1] >= open_5.iloc[-1]:
                return False, "HIGH VOLATILITY: Last candle must be red for SELL"

        # Check 3: EMA spread on 15min must be significant (not just barely crossed)
        ema9_15 = self.calculate_ema(df_15min['close'], 9).iloc[-1]
        ema21_15 = self.calculate_ema(df_15min['close'], 21).iloc[-1]
        ema_spread_pct = abs(ema9_15 - ema21_15) / ema21_15 * 100

        if ema_spread_pct < 0.05:  # Less than 0.05% spread
            return False, f"HIGH VOLATILITY: EMA spread too tight ({ema_spread_pct:.3f}%)"

        # Check 4: RSI not at extremes (avoid overbought/oversold in volatile pairs)
        rsi = self.calculate_rsi(close_5, 14).iloc[-1]
        if trend_direction == "BUY" and rsi > 70:
            return False, f"HIGH VOLATILITY: RSI too high for BUY ({rsi:.1f})"
        if trend_direction == "SELL" and rsi < 30:
            return False, f"HIGH VOLATILITY: RSI too low for SELL ({rsi:.1f})"

        return True, f"HIGH VOLATILITY VALIDATED: {instrument} passed all checks"

    def should_enter_trade(
        self,
        instrument: str,
        df_1min: pd.DataFrame,
        df_5min: pd.DataFrame,
        df_15min: pd.DataFrame,
        df_30min: pd.DataFrame,
        current_positions: int,
        trades_today: int,
        daily_pl_pct: float,
        now: datetime = None
    ) -> Dict:
        """
        TREND-FOLLOWING V3: Breakout + Pullback Entry (WITH the trend!)

        CORE PRINCIPLE: Only trade in the direction of the higher timeframe trend!
        - UPTREND (EMA9 > EMA21 on 15min+30min) → Only take BUY signals
        - DOWNTREND (EMA9 < EMA21 on 15min+30min) → Only take SELL signals

        TWO ENTRY METHODS:
        1. BREAKOUT: Price breaks key level WITH trend → Early momentum entry
        2. PULLBACK: RSI retraces to 40-60 zone WITH trend → Better price entry

        EXTRA VALIDATION for high volatility pairs (GBP, JPY, AUD, NZD):
        - Requires stronger trend confirmation
        - Multiple candle confirmation
        - No recent spikes

        NO MORE counter-trend trades (no buying oversold in downtrends!)
        """
        if now is None:
            now = datetime.now(pytz.UTC)

        current_session = self.get_current_session(now)

        # STEP 0: Session cooldown check
        cooldown_active, cooldown_reason = self.is_session_cooldown_active(instrument, now)
        if cooldown_active:
            return {
                "action": "SKIP",
                "reason": f"Session cooldown: {cooldown_reason} (current: {current_session})",
                "session": current_session
            }

        # STEP 1: Basic checks
        if current_positions > 0:
            return {"action": "SKIP", "reason": "Position already open"}

        if trades_today >= self.max_trades_per_day:
            return {"action": "SKIP", "reason": f"Max trades reached ({self.max_trades_per_day})"}

        if daily_pl_pct >= self.daily_profit_target:
            return {"action": "SKIP", "reason": f"Daily profit target reached"}

        if daily_pl_pct <= -0.05:  # -5% daily loss limit
            return {"action": "SKIP", "reason": "Daily loss limit reached"}

        # STEP 2: Data validation
        if len(df_15min) < 30 or len(df_30min) < 30 or len(df_5min) < 25:
            return {"action": "SKIP", "reason": "Insufficient data"}

        # STEP 3: DETERMINE HIGHER TIMEFRAME TREND (THIS IS THE KEY!)
        htf_trend = self.check_htf_trend(df_15min, df_30min)
        tradeable_direction = htf_trend.get("tradeable_direction")

        # If no clear trend, DO NOT TRADE
        if tradeable_direction is None:
            return {
                "action": "SKIP",
                "reason": f"No clear trend - 15min: {htf_trend['15min']}, 30min: {htf_trend['30min']}",
                "analysis": {
                    "htf_trend": htf_trend,
                    "session": current_session
                }
            }

        # Calculate current RSI for display
        close_5 = df_5min['close']
        current_rsi = self.calculate_rsi(close_5, 14).iloc[-1]

        # STEP 3.5: EXTRA VALIDATION for high volatility pairs (GBP, JPY, AUD, NZD)
        # These pairs need additional confirmation to avoid false signals
        is_high_vol_pair = instrument in self.HIGH_VOLATILITY_PAIRS
        if is_high_vol_pair:
            vol_valid, vol_reason = self.validate_high_volatility_entry(
                instrument, df_5min, df_15min, tradeable_direction
            )
            if not vol_valid:
                return {
                    "action": "SKIP",
                    "reason": vol_reason,
                    "analysis": {
                        "htf_trend": htf_trend,
                        "tradeable_direction": tradeable_direction,
                        "rsi": current_rsi,
                        "session": current_session,
                        "high_volatility_check": "FAILED"
                    }
                }

        # STEP 4: Check for BREAKOUT entry (Method 1)
        breakout = self.detect_breakout(df_5min, df_15min, tradeable_direction)

        if breakout["entry_signal"]:
            return {
                "action": tradeable_direction,
                "confidence": "HIGH",
                "entry_type": "BREAKOUT",
                "reason": f"BREAKOUT {tradeable_direction}: {breakout['reason']} | Trend: {htf_trend['trend']}" + (" [HV VALIDATED]" if is_high_vol_pair else ""),
                "analysis": {
                    "htf_trend": htf_trend,
                    "breakout": breakout,
                    "rsi": current_rsi,
                    "session": current_session,
                    "high_volatility_validated": is_high_vol_pair
                }
            }

        # STEP 5: Check for PULLBACK entry (Method 2)
        pullback = self.detect_pullback_entry(df_5min, df_1min, tradeable_direction)

        if pullback["entry_signal"]:
            return {
                "action": tradeable_direction,
                "confidence": "HIGH" if pullback.get("score", 0) >= 6 else "MEDIUM",
                "entry_type": "PULLBACK",
                "reason": f"PULLBACK {tradeable_direction}: {pullback['reason']} | Trend: {htf_trend['trend']}" + (" [HV VALIDATED]" if is_high_vol_pair else ""),
                "analysis": {
                    "htf_trend": htf_trend,
                    "pullback": pullback,
                    "rsi": current_rsi,
                    "session": current_session,
                    "high_volatility_validated": is_high_vol_pair
                }
            }

        # STEP 6: No entry signal - provide detailed feedback
        skip_reasons = []

        # Breakout status
        if not breakout["breakout_detected"]:
            skip_reasons.append("No breakout")
        elif not breakout["momentum_confirmed"]:
            skip_reasons.append("Breakout without momentum")

        # Pullback status
        if not pullback.get("rsi_in_zone", False):
            if tradeable_direction == "BUY":
                skip_reasons.append(f"RSI not in buy zone ({current_rsi:.1f}, need 35-55)")
            else:
                skip_reasons.append(f"RSI not in sell zone ({current_rsi:.1f}, need 45-65)")
        elif not pullback.get("reversal_candle", False):
            skip_reasons.append("Waiting for reversal candle")

        return {
            "action": "SKIP",
            "reason": f"Trend: {htf_trend['trend']} ({tradeable_direction} only) | {', '.join(skip_reasons)}",
            "analysis": {
                "htf_trend": htf_trend,
                "tradeable_direction": tradeable_direction,
                "breakout": breakout,
                "pullback": pullback,
                "rsi": current_rsi,
                "session": current_session
            }
        }

    def calculate_position_size(self, account_balance: float, current_price: float, instrument: str) -> int:
        """
        Calculate position size for CONSISTENT $1 PER PIP across ALL pairs.

        This ensures uniform risk management - every pip gained/lost = $1
        Makes it easy to calculate risk: 20 pip SL = $20 risk

        FORMULA BY PAIR TYPE:

        1. XXX/USD pairs (EUR/USD, GBP/USD, AUD/USD, NZD/USD):
           - Quote currency is USD, so pip value is straightforward
           - 1 pip = 0.0001 price move
           - Pip value in USD = Units × 0.0001
           - For $1/pip: Units = 10,000

        2. USD/JPY:
           - 1 pip = 0.01 price move
           - Pip value in USD = Units × 0.01 / USD_JPY_rate
           - For $1/pip: Units = 1 × USD_JPY_rate / 0.01 = rate × 100
           - At 155.00: Units = 15,500

        3. USD/CHF:
           - 1 pip = 0.0001 price move
           - Pip value in USD = Units × 0.0001 / USD_CHF_rate
           - For $1/pip: Units = 1 × USD_CHF_rate / 0.0001 = rate × 10,000
           - At 0.80: Units = 8,000

        4. USD/CAD:
           - 1 pip = 0.0001 price move
           - Pip value in USD = Units × 0.0001 / USD_CAD_rate
           - For $1/pip: Units = 1 × USD_CAD_rate / 0.0001 = rate × 10,000
           - At 1.38: Units = 13,800

        TARGET: $1 per pip for consistent risk management
        """
        # Target pip value in USD
        target_pip_value = 1.0  # $1 per pip

        # Calculate units based on pair type
        if instrument in ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"]:
            # XXX/USD pairs: Quote is USD
            # Pip value = Units × 0.0001
            # Units for $1/pip = 1 / 0.0001 = 10,000
            units = int(target_pip_value / 0.0001)  # 10,000 units

        elif instrument == "USD_JPY":
            # JPY pair: 1 pip = 0.01
            # Pip value in USD = Units × 0.01 / rate
            # Units for $1/pip = target × rate / 0.01
            units = int(target_pip_value * current_price / 0.01)

        elif instrument == "USD_CHF":
            # USD/CHF: Quote is CHF
            # Pip value in USD = Units × 0.0001 / rate
            # Units for $1/pip = target × rate / 0.0001
            units = int(target_pip_value * current_price / 0.0001)

        elif instrument == "USD_CAD":
            # USD/CAD: Quote is CAD
            # Pip value in USD = Units × 0.0001 / rate
            # Units for $1/pip = target × rate / 0.0001
            units = int(target_pip_value * current_price / 0.0001)

        else:
            # Default: assume XXX/USD structure
            units = int(target_pip_value / 0.0001)  # 10,000 units

        # OANDA minimum is 1 unit
        if units < 1:
            units = 1

        return units

    def calculate_stop_loss_take_profit(self, entry_price: float, direction: str, instrument: str = None) -> Tuple[float, float]:
        """
        Calculate SL and TP with pair-specific settings

        Args:
            entry_price: Entry price
            direction: "BUY" or "SELL"
            instrument: Currency pair (e.g., "EUR_USD")

        Returns:
            (stop_loss, take_profit) tuple
        """
        # Get pair-specific settings if instrument provided
        if instrument:
            params = get_scalping_params(instrument)
            # These are ABSOLUTE price movements (e.g. 0.0020 for EUR, 0.18 for JPY)
            sl_abs = params["stop_loss_pct"]
            tp_abs = params["take_profit_pct"]
        else:
            # Fallback to default percentage
            sl_abs = entry_price * self.stop_loss_pct
            tp_abs = entry_price * self.take_profit_pct

        if direction == "BUY":
            stop_loss = entry_price - sl_abs  # Subtract absolute distance
            take_profit = entry_price + tp_abs  # Add absolute distance
        else:
            stop_loss = entry_price + sl_abs  # Add absolute distance
            take_profit = entry_price - tp_abs  # Subtract absolute distance

        # Round to proper precision for OANDA
        # JPY pairs: 3 decimals, others: 5 decimals
        precision = 3 if "JPY" in instrument else 5
        stop_loss = round(stop_loss, precision)
        take_profit = round(take_profit, precision)

        return stop_loss, take_profit

    def get_pair_trailing_settings(self, instrument: str) -> Tuple[float, float]:
        """
        Get trailing stop settings for specific pair

        Returns:
            (trailing_trigger_pct, trailing_distance_pct)
        """
        params = get_scalping_params(instrument)
        return params["trailing_stop_trigger"], params["trailing_stop_distance"]

    def check_pullback_entry(self, df_1min: pd.DataFrame, direction: str) -> Dict:
        """
        Check if conditions are right for entry based on pullback analysis.

        For BULLISH (BUY): Wait for pullback (dip) then recovery (green candle)
        For BEARISH (SELL): Wait for bounce (rally) then rejection (red candle)

        This helps get better entry points instead of chasing momentum.

        Args:
            df_1min: 1-minute DataFrame with OHLC data
            direction: 'BULLISH' or 'BEARISH'

        Returns:
            dict with ready_to_enter, pullback_detected, recovery_detected, scores, reasons
        """
        if len(df_1min) < 20:
            return {
                'ready_to_enter': False,
                'pullback_detected': False,
                'recovery_detected': False,
                'pullback_score': 0,
                'recovery_score': 0,
                'reasons': ['Not enough data for pullback analysis']
            }

        # Get recent price data
        closes = df_1min['close'].values[-20:]
        opens = df_1min['open'].values[-20:]
        highs = df_1min['high'].values[-20:]
        lows = df_1min['low'].values[-20:]
        current_price = closes[-1]

        # Calculate indicators
        ema_9 = self.calculate_ema(df_1min['close'].tail(20), 9).iloc[-1]
        ema_20 = self.calculate_ema(df_1min['close'].tail(20), 20).iloc[-1]
        rsi = self.calculate_rsi(df_1min['close'].tail(20), 14).iloc[-1]

        # VWAP approximation (using mid-price)
        vwap = ((df_1min['high'].tail(20) + df_1min['low'].tail(20) + df_1min['close'].tail(20)) / 3).mean()

        # Recent extremes
        recent_high = max(highs[-20:])
        recent_low = min(lows[-10:])
        pullback_from_high = ((recent_high - current_price) / recent_high) * 100 if recent_high > 0 else 0
        bounce_from_low = ((current_price - recent_low) / recent_low) * 100 if recent_low > 0 else 0

        reasons = []
        pullback_score = 0
        recovery_score = 0

        last_candle_green = closes[-1] > opens[-1]
        last_candle_red = closes[-1] < opens[-1]

        if direction == 'BULLISH':
            # For BUY: Wait for pullback (dip) + recovery

            # PULLBACK DETECTION (price pulled back)
            if rsi < 40:
                pullback_score += 3
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi < 50:
                pullback_score += 1
                reasons.append(f"RSI neutral-low ({rsi:.1f})")

            if current_price < vwap:
                pullback_score += 2
                reasons.append("Below VWAP")

            if pullback_from_high >= 0.3:
                pullback_score += 2
                reasons.append(f"Pulled {pullback_from_high:.2f}% from high")

            # Count red candles in last 5
            red_count = sum(1 for i in range(-5, 0) if closes[i] < opens[i])
            if red_count >= 3:
                pullback_score += 2
                reasons.append(f"{red_count}/5 red candles")

            # RECOVERY DETECTION (turning bullish)
            if last_candle_green:
                recovery_score += 3
                reasons.append("Green candle - reversal!")
            else:
                reasons.append("Waiting for green candle")

            if bounce_from_low >= 0.05:
                recovery_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")

            if rsi > 35:
                recovery_score += 1

            # 3-bar momentum
            if len(closes) >= 3:
                momentum_3 = ((closes[-1] - closes[-3]) / closes[-3]) * 100
                if momentum_3 > 0:
                    recovery_score += 2
                    reasons.append("3-bar momentum +")

            # Higher low pattern
            if len(lows) >= 2 and lows[-1] > lows[-2]:
                recovery_score += 1
                reasons.append("Higher low")

            pullback_detected = pullback_score >= 4
            recovery_detected = recovery_score >= 4

            # Safety: Don't enter if RSI too high
            if rsi > 65:
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': recovery_detected,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': [f"RSI too high ({rsi:.1f})"]
                }

            # Require green candle
            if not last_candle_green:
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': False,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons
                }

            ready = pullback_detected and recovery_detected
            if ready:
                reasons.append(">>> PULLBACK + RECOVERY: BUY NOW!")

        else:  # BEARISH
            # For SELL: Wait for bounce (rally) + rejection

            # BOUNCE DETECTION (price bounced up)
            if rsi > 60:
                pullback_score += 3
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif rsi > 50:
                pullback_score += 1
                reasons.append(f"RSI neutral-high ({rsi:.1f})")

            if current_price > vwap:
                pullback_score += 2
                reasons.append("Above VWAP")

            if bounce_from_low >= 0.3:
                pullback_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")

            # Count green candles in last 5
            green_count = sum(1 for i in range(-5, 0) if closes[i] > opens[i])
            if green_count >= 3:
                pullback_score += 2
                reasons.append(f"{green_count}/5 green candles")

            # REJECTION DETECTION (turning bearish)
            if last_candle_red:
                recovery_score += 3
                reasons.append("Red candle - reversal!")
            else:
                reasons.append("Waiting for red candle")

            if pullback_from_high >= 0.05:
                recovery_score += 2
                reasons.append(f"Rejecting {pullback_from_high:.2f}% from high")

            if rsi < 65:
                recovery_score += 1

            # 3-bar momentum
            if len(closes) >= 3:
                momentum_3 = ((closes[-1] - closes[-3]) / closes[-3]) * 100
                if momentum_3 < 0:
                    recovery_score += 2
                    reasons.append("3-bar momentum -")

            # Lower high pattern
            if len(highs) >= 2 and highs[-1] < highs[-2]:
                recovery_score += 1
                reasons.append("Lower high")

            pullback_detected = pullback_score >= 4
            recovery_detected = recovery_score >= 4

            # Safety: Don't enter if RSI too low
            if rsi < 35:
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': recovery_detected,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': [f"RSI too low ({rsi:.1f})"]
                }

            # Require red candle
            if not last_candle_red:
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': False,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons
                }

            ready = pullback_detected and recovery_detected
            if ready:
                reasons.append(">>> BOUNCE + REJECTION: SELL NOW!")

        return {
            'ready_to_enter': ready,
            'pullback_detected': pullback_detected,
            'recovery_detected': recovery_detected,
            'pullback_score': pullback_score,
            'recovery_score': recovery_score,
            'reasons': reasons
        }
