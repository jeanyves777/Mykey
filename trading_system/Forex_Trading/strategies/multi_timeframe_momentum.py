"""
Multi-Timeframe Momentum Forex Strategy
Similar to MARA continuous trading strategy with HTF filters and pullback detection

VALIDATION PIPELINE:
1. Market Timing Check (Forex is 24/5)
2. Position & Risk Check
3. Technical Scoring (1-MIN bars)
4. Price Action Analysis (5-MIN bars)
5. Real-Time Momentum (Last 5 1-MIN bars) - HIGHEST WEIGHT
6. V3 Final Decision (Momentum-Weighted)
7. Higher Timeframe Trend Filter (30-MIN + 1-HOUR) - STRICT
8. Pullback Detection (5-MIN HTF) - QUALITY FILTER
9. Position Sizing & Execution
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from typing import Dict, List, Optional, Tuple


class MultiTimeframeMomentumStrategy:
    """
    Multi-timeframe momentum strategy for forex pairs
    Combines technical analysis, price action, and momentum with strict HTF filters
    """

    def __init__(
        self,
        instruments: List[str],
        max_trades_per_day: int = 3,
        daily_profit_target: float = 0.02,  # 2% daily profit target
        trade_size_pct: float = 0.10,  # 10% of account per trade
        take_profit_pct: float = 0.015,  # 1.5% TP (150 pips on standard lot)
        stop_loss_pct: float = 0.01,  # 1% SL (100 pips on standard lot)
        trailing_stop_trigger: float = 0.006,  # Start trailing at 0.6% (60 pips)
        trailing_stop_distance: float = 0.004  # Trail 0.4% behind (40 pips)
    ):
        self.instruments = instruments
        self.max_trades_per_day = max_trades_per_day
        self.daily_profit_target = daily_profit_target
        self.trade_size_pct = trade_size_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_trigger = trailing_stop_trigger
        self.trailing_stop_distance = trailing_stop_distance

        # Trading session times (24/5 market)
        self.london_open = time(8, 0)  # 08:00 UTC
        self.ny_open = time(13, 0)  # 13:00 UTC (8 AM EST)
        self.ny_close = time(21, 0)  # 21:00 UTC (4 PM EST)

        print("[STRATEGY] Multi-Timeframe Momentum initialized")
        print(f"[STRATEGY] Instruments: {', '.join(instruments)}")
        print(f"[STRATEGY] Max trades/day: {max_trades_per_day}")
        print(f"[STRATEGY] Profit target: {daily_profit_target*100}%")

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def analyze_technical_indicators(self, df_1min: pd.DataFrame) -> Dict:
        """
        STEP 3: Technical Scoring (1-MIN bars)
        Analyzes 17+ technical indicators
        """
        if len(df_1min) < 50:
            return {"signal": "NEUTRAL", "confidence": 0, "reason": "Insufficient data"}

        close = df_1min['close']
        high = df_1min['high']
        low = df_1min['low']

        # Calculate indicators
        ema9 = self.calculate_ema(close, 9)
        ema20 = self.calculate_ema(close, 20)
        ema50 = self.calculate_ema(close, 50)
        rsi = self.calculate_rsi(close, 14)
        macd_line, signal_line, histogram = self.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close, 20, 2)
        atr = self.calculate_atr(df_1min, 14)

        # Get latest values
        current_price = close.iloc[-1]
        current_ema9 = ema9.iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2]

        score = 0
        max_score = 17

        # 1. EMA Stack (3 points)
        if current_price > current_ema9 > current_ema20:
            score += 3  # Bullish stack
        elif current_price < current_ema9 < current_ema20:
            score -= 3  # Bearish stack

        # 2. Price vs EMA50 (2 points)
        if current_price > current_ema50:
            score += 2
        else:
            score -= 2

        # 3. RSI (3 points)
        if 40 < current_rsi < 60:
            score += 1  # Neutral zone
        elif current_rsi > 70:
            score -= 2  # Overbought
        elif current_rsi < 30:
            score += 2  # Oversold (bullish reversal)

        # 4. MACD Crossover (3 points)
        if current_macd > current_signal and prev_histogram < 0:
            score += 3  # Bullish crossover
        elif current_macd < current_signal and prev_histogram > 0:
            score -= 3  # Bearish crossover
        elif current_macd > current_signal:
            score += 1
        else:
            score -= 1

        # 5. MACD Histogram (2 points)
        if current_histogram > 0 and current_histogram > prev_histogram:
            score += 2  # Increasing bullish momentum
        elif current_histogram < 0 and current_histogram < prev_histogram:
            score -= 2  # Increasing bearish momentum

        # 6. Bollinger Bands (2 points)
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        if bb_position < 0.2:
            score += 2  # Near lower band (bullish)
        elif bb_position > 0.8:
            score -= 2  # Near upper band (bearish)

        # 7. Price momentum (2 points)
        price_change_5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
        if price_change_5 > 0.001:  # 0.1% in 5 bars
            score += 2
        elif price_change_5 < -0.001:
            score -= 2

        # Determine signal
        confidence = abs(score) / max_score

        if score >= 8:
            signal = "BULLISH"
        elif score <= -8:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "confidence": confidence,
            "score": score,
            "max_score": max_score,
            "rsi": current_rsi,
            "macd": current_macd,
            "price": current_price,
            "ema9": current_ema9
        }

    def analyze_price_action(self, df_5min: pd.DataFrame) -> Dict:
        """
        STEP 4: Price Action Analysis (5-MIN bars)
        Pattern recognition on 5-minute timeframe
        """
        if len(df_5min) < 10:
            return {"signal": "NEUTRAL", "strength": 0, "reason": "Insufficient data"}

        close = df_5min['close']
        high = df_5min['high']
        low = df_5min['low']
        open_price = df_5min['open']

        # Last 5 bars
        last_5_close = close.tail(5)
        last_5_high = high.tail(5)
        last_5_low = low.tail(5)
        last_5_open = open_price.tail(5)

        score = 0
        max_score = 10

        # 1. Candle color pattern (3 points)
        green_candles = (last_5_close > last_5_open).sum()
        red_candles = (last_5_close < last_5_open).sum()

        if green_candles >= 4:
            score += 3
        elif green_candles >= 3:
            score += 1
        elif red_candles >= 4:
            score -= 3
        elif red_candles >= 3:
            score -= 1

        # 2. Higher highs / Lower lows (3 points)
        highs = last_5_high.values
        lows = last_5_low.values

        higher_highs = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
        lower_lows = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))

        if higher_highs >= 3:
            score += 3
        elif lower_lows >= 3:
            score -= 3

        # 3. 5-bar SMA (2 points)
        sma5 = close.rolling(5).mean().iloc[-1]
        current_price = close.iloc[-1]

        if current_price > sma5:
            score += 2
        else:
            score -= 2

        # 4. 5-bar momentum (2 points)
        momentum = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]
        if momentum > 0.002:  # 0.2%
            score += 2
        elif momentum < -0.002:
            score -= 2

        strength = abs(score) / max_score

        if score >= 5:
            signal = "BULLISH"
        elif score <= -5:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "strength": strength,
            "score": score,
            "green_candles": green_candles,
            "red_candles": red_candles
        }

    def analyze_realtime_momentum(self, df_1min: pd.DataFrame) -> Dict:
        """
        STEP 5: Real-Time Momentum (Last 5 1-MIN bars)
        HIGHEST WEIGHT - can override other methods
        """
        if len(df_1min) < 6:
            return {"signal": "NEUTRAL", "momentum": 0, "strong": False}

        last_5 = df_1min.tail(5)
        close = last_5['close']
        open_price = last_5['open']
        high = last_5['high']
        low = last_5['low']

        score = 0
        max_score = 12

        # 1. Green vs red candles (4 points)
        green_count = (close > open_price).sum()
        red_count = (close < open_price).sum()

        if green_count >= 4:
            score += 4
            strong_momentum = True
        elif green_count >= 3:
            score += 2
            strong_momentum = False
        elif red_count >= 4:
            score -= 4
            strong_momentum = True
        elif red_count >= 3:
            score -= 2
            strong_momentum = False
        else:
            strong_momentum = False

        # 2. 5-bar price change (4 points)
        price_change = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
        if price_change > 0.003:  # 0.3%
            score += 4
        elif price_change > 0.001:
            score += 2
        elif price_change < -0.003:
            score -= 4
        elif price_change < -0.001:
            score -= 2

        # 3. Last bar direction (2 points)
        last_candle_bullish = close.iloc[-1] > open_price.iloc[-1]
        if last_candle_bullish:
            score += 2
        else:
            score -= 2

        # 4. Higher highs / Lower lows (2 points)
        highs = high.values
        lows = low.values

        hh_count = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
        ll_count = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))

        if hh_count >= 3:
            score += 2
        elif ll_count >= 3:
            score -= 2

        momentum_strength = abs(score) / max_score

        if score >= 7:
            signal = "BULLISH"
        elif score <= -7:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "momentum": momentum_strength,
            "score": score,
            "strong": strong_momentum and abs(score) >= 8,
            "green_count": green_count,
            "red_count": red_count
        }

    def make_weighted_decision(
        self,
        technical: Dict,
        price_action: Dict,
        momentum: Dict
    ) -> Dict:
        """
        STEP 6: V3 Final Decision (Momentum-Weighted)
        Weights: Technical=1x, Price Action=1x, Momentum=2x
        """
        # Map signals to numeric values
        signal_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}

        tech_signal = signal_map[technical["signal"]]
        pa_signal = signal_map[price_action["signal"]]
        mom_signal = signal_map[momentum["signal"]]

        # Apply weights
        weighted_score = (tech_signal * 1) + (pa_signal * 1) + (mom_signal * 2)

        # Check for strong momentum override
        if momentum["strong"]:
            if mom_signal > 0:
                return {
                    "decision": "BUY",
                    "confidence": "HIGH",
                    "reason": "STRONG MOMENTUM OVERRIDE",
                    "weighted_score": weighted_score,
                    "technical": technical["signal"],
                    "price_action": price_action["signal"],
                    "momentum": momentum["signal"]
                }
            elif mom_signal < 0:
                return {
                    "decision": "SELL",
                    "confidence": "HIGH",
                    "reason": "STRONG MOMENTUM OVERRIDE",
                    "weighted_score": weighted_score,
                    "technical": technical["signal"],
                    "price_action": price_action["signal"],
                    "momentum": momentum["signal"]
                }

        # Check if all agree
        signals = [tech_signal, pa_signal, mom_signal]
        if all(s > 0 for s in signals):
            return {
                "decision": "BUY",
                "confidence": "HIGH",
                "reason": "ALL METHODS AGREE",
                "weighted_score": weighted_score,
                "technical": technical["signal"],
                "price_action": price_action["signal"],
                "momentum": momentum["signal"]
            }
        elif all(s < 0 for s in signals):
            return {
                "decision": "SELL",
                "confidence": "HIGH",
                "reason": "ALL METHODS AGREE",
                "weighted_score": weighted_score,
                "technical": technical["signal"],
                "price_action": price_action["signal"],
                "momentum": momentum["signal"]
            }

        # Weighted score decision
        if weighted_score >= 2:
            return {
                "decision": "BUY",
                "confidence": "MEDIUM",
                "reason": f"WEIGHTED SCORE: {weighted_score}/4",
                "weighted_score": weighted_score,
                "technical": technical["signal"],
                "price_action": price_action["signal"],
                "momentum": momentum["signal"]
            }
        elif weighted_score <= -2:
            return {
                "decision": "SELL",
                "confidence": "MEDIUM",
                "reason": f"WEIGHTED SCORE: {weighted_score}/4",
                "weighted_score": weighted_score,
                "technical": technical["signal"],
                "price_action": price_action["signal"],
                "momentum": momentum["signal"]
            }
        else:
            return {
                "decision": "SKIP",
                "confidence": "LOW",
                "reason": "NO CONSENSUS",
                "weighted_score": weighted_score,
                "technical": technical["signal"],
                "price_action": price_action["signal"],
                "momentum": momentum["signal"]
            }

    def check_htf_trend_alignment(
        self,
        df_30min: pd.DataFrame,
        df_1hour: pd.DataFrame,
        signal: str
    ) -> Dict:
        """
        STEP 7: Higher Timeframe Trend Filter (STRICT)
        Both 30-MIN and 1-HOUR must agree, otherwise BLOCKED
        """
        if len(df_30min) < 20 or len(df_1hour) < 20:
            return {"approved": False, "reason": "Insufficient HTF data"}

        def analyze_htf(df: pd.DataFrame) -> str:
            close = df['close']
            ema9 = self.calculate_ema(close, 9)

            # Check trend
            price_above_ema = close.iloc[-1] > ema9.iloc[-1]

            # Check higher highs / lower lows
            last_5_high = df['high'].tail(5)
            last_5_low = df['low'].tail(5)

            highs = last_5_high.values
            lows = last_5_low.values

            hh_count = sum(highs[i] > highs[i-1] for i in range(1, len(highs)))
            ll_count = sum(lows[i] < lows[i-1] for i in range(1, len(lows)))

            # Candle colors
            green_count = (df['close'].tail(5) > df['open'].tail(5)).sum()

            if price_above_ema and hh_count >= 2 and green_count >= 3:
                return "BULLISH"
            elif not price_above_ema and ll_count >= 2 and green_count <= 2:
                return "BEARISH"
            else:
                return "NEUTRAL"

        trend_30min = analyze_htf(df_30min)
        trend_1hour = analyze_htf(df_1hour)

        # STRICT: Both must agree
        if trend_30min == trend_1hour and trend_30min != "NEUTRAL":
            htf_trend = trend_30min
        else:
            htf_trend = "NEUTRAL"

        # Check alignment
        if signal == "BUY":
            if htf_trend == "BULLISH":
                return {
                    "approved": True,
                    "htf_trend": htf_trend,
                    "30min": trend_30min,
                    "1hour": trend_1hour,
                    "reason": "HTF BULLISH - APPROVED"
                }
            elif htf_trend == "NEUTRAL":
                return {
                    "approved": False,
                    "htf_trend": htf_trend,
                    "30min": trend_30min,
                    "1hour": trend_1hour,
                    "reason": "HTF NEUTRAL (conflicting) - BLOCKED"
                }
            else:
                return {
                    "approved": False,
                    "htf_trend": htf_trend,
                    "30min": trend_30min,
                    "1hour": trend_1hour,
                    "reason": "HTF BEARISH - BLOCKED"
                }

        elif signal == "SELL":
            if htf_trend == "BEARISH":
                return {
                    "approved": True,
                    "htf_trend": htf_trend,
                    "30min": trend_30min,
                    "1hour": trend_1hour,
                    "reason": "HTF BEARISH - APPROVED"
                }
            elif htf_trend == "NEUTRAL":
                return {
                    "approved": False,
                    "htf_trend": htf_trend,
                    "30min": trend_30min,
                    "1hour": trend_1hour,
                    "reason": "HTF NEUTRAL (conflicting) - BLOCKED"
                }
            else:
                return {
                    "approved": False,
                    "htf_trend": htf_trend,
                    "30min": trend_30min,
                    "1hour": trend_1hour,
                    "reason": "HTF BULLISH - BLOCKED"
                }

        return {"approved": False, "reason": "Unknown signal"}

    def check_pullback_conditions(
        self,
        df_5min: pd.DataFrame,
        signal: str
    ) -> Dict:
        """
        STEP 8: V4 Pullback Detection (5-MIN HTF)
        Wait for pullback/bounce before entry
        """
        if len(df_5min) < 10:
            return {"ready": False, "reason": "Insufficient data"}

        close = df_5min['close']
        open_price = df_5min['open']
        high = df_5min['high']
        low = df_5min['low']

        last_4 = df_5min.tail(4)
        last_candle = df_5min.iloc[-1]

        rsi = self.calculate_rsi(close, 14).iloc[-1]

        if signal == "BUY":
            # For CALLS: Wait for dip, then recovery
            red_count = (last_4['close'] < last_4['open']).sum()
            last_is_green = last_candle['close'] > last_candle['open']

            # Check pullback from recent high
            recent_high = high.tail(10).max()
            pullback_pct = (recent_high - close.iloc[-1]) / recent_high

            if red_count >= 3 and last_is_green and rsi < 65 and pullback_pct > 0.002:
                return {
                    "ready": True,
                    "reason": "PULLBACK MET - Dip + Green recovery",
                    "red_count": red_count,
                    "rsi": rsi,
                    "pullback_pct": pullback_pct
                }
            else:
                return {
                    "ready": False,
                    "reason": "WAITING FOR PULLBACK...",
                    "red_count": red_count,
                    "last_green": last_is_green,
                    "rsi": rsi
                }

        elif signal == "SELL":
            # For PUTS: Wait for bounce, then rejection
            green_count = (last_4['close'] > last_4['open']).sum()
            last_is_red = last_candle['close'] < last_candle['open']

            # Check bounce from recent low
            recent_low = low.tail(10).min()
            bounce_pct = (close.iloc[-1] - recent_low) / recent_low

            if green_count >= 3 and last_is_red and rsi > 35 and bounce_pct > 0.002:
                return {
                    "ready": True,
                    "reason": "PULLBACK MET - Bounce + Red rejection",
                    "green_count": green_count,
                    "rsi": rsi,
                    "bounce_pct": bounce_pct
                }
            else:
                return {
                    "ready": False,
                    "reason": "WAITING FOR PULLBACK...",
                    "green_count": green_count,
                    "last_red": last_is_red,
                    "rsi": rsi
                }

        return {"ready": False, "reason": "Unknown signal"}

    def should_enter_trade(
        self,
        instrument: str,
        df_1min: pd.DataFrame,
        df_5min: pd.DataFrame,
        df_30min: pd.DataFrame,
        df_1hour: pd.DataFrame,
        current_positions: int,
        trades_today: int,
        daily_pl_pct: float
    ) -> Dict:
        """
        Complete trade entry validation pipeline
        Returns trade decision with full reasoning
        """
        # STEP 1: Market Timing (Forex is 24/5, check for best sessions)
        now_utc = datetime.now(pytz.UTC)
        current_time = now_utc.time()

        # Best trading during London/NY overlap (13:00-17:00 UTC)
        in_best_session = self.london_open <= current_time <= self.ny_close

        # STEP 2: Position & Risk Check
        if current_positions > 0:
            return {"action": "SKIP", "reason": "Position already open"}

        if trades_today >= self.max_trades_per_day:
            return {"action": "SKIP", "reason": f"Max trades reached ({self.max_trades_per_day})"}

        if daily_pl_pct >= self.daily_profit_target:
            return {"action": "SKIP", "reason": f"Daily profit target reached ({self.daily_profit_target*100}%)"}

        # STEP 3: Technical Analysis
        technical = self.analyze_technical_indicators(df_1min)
        if technical["signal"] == "NEUTRAL":
            return {"action": "SKIP", "reason": "Technical: NEUTRAL", "analysis": technical}

        # STEP 4: Price Action
        price_action = self.analyze_price_action(df_5min)

        # STEP 5: Real-Time Momentum
        momentum = self.analyze_realtime_momentum(df_1min)

        # STEP 6: Weighted Decision
        decision = self.make_weighted_decision(technical, price_action, momentum)

        if decision["decision"] == "SKIP":
            return {
                "action": "SKIP",
                "reason": decision["reason"],
                "analysis": {
                    "technical": technical,
                    "price_action": price_action,
                    "momentum": momentum,
                    "decision": decision
                }
            }

        # STEP 7: HTF Trend Alignment (STRICT)
        htf_check = self.check_htf_trend_alignment(df_30min, df_1hour, decision["decision"])

        if not htf_check["approved"]:
            return {
                "action": "SKIP",
                "reason": f"HTF FILTER: {htf_check['reason']}",
                "analysis": {
                    "technical": technical,
                    "price_action": price_action,
                    "momentum": momentum,
                    "decision": decision,
                    "htf": htf_check
                }
            }

        # STEP 8: Pullback Detection
        pullback = self.check_pullback_conditions(df_5min, decision["decision"])

        if not pullback["ready"]:
            return {
                "action": "WAIT",
                "reason": f"⏳ {pullback['reason']}",
                "analysis": {
                    "technical": technical,
                    "price_action": price_action,
                    "momentum": momentum,
                    "decision": decision,
                    "htf": htf_check,
                    "pullback": pullback
                }
            }

        # ALL CHECKS PASSED
        return {
            "action": decision["decision"],
            "confidence": decision["confidence"],
            "reason": f"✅ ALL CHECKS PASSED - {pullback['reason']}",
            "in_best_session": in_best_session,
            "analysis": {
                "technical": technical,
                "price_action": price_action,
                "momentum": momentum,
                "decision": decision,
                "htf": htf_check,
                "pullback": pullback
            }
        }

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        instrument: str
    ) -> int:
        """
        Calculate position size in units
        Forex units: 100,000 = 1 standard lot, 10,000 = 1 mini lot, 1,000 = 1 micro lot
        """
        trade_value = account_balance * self.trade_size_pct
        units = int(trade_value / current_price)

        # Round to nearest 1000 (micro lot)
        units = (units // 1000) * 1000

        return max(units, 1000)  # Minimum 1 micro lot

    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        direction: str
    ) -> Tuple[float, float]:
        """Calculate SL and TP prices"""
        if direction == "BUY":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        return stop_loss, take_profit
