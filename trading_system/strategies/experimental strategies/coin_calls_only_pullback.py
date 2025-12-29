"""
COIN CALLS-ONLY Pullback Strategy

This strategy ONLY buys CALLS - never puts.
It waits for pullbacks (dips) before entering, then rides the recovery.

Key Concept:
- Calls tend to be more profitable over time
- Buying on pullbacks gives better entry prices
- Wait for oversold conditions or dip, then buy calls on the bounce

Entry Conditions (MUST ALL BE MET):
1. Market has pulled back (RSI oversold, below VWAP, or red candles)
2. Signs of recovery (green candle, momentum turning)
3. Not at extreme overbought levels

This is safer than buying puts in a bearish market because:
- Even in bearish markets, there are bounces
- Calls on bounces have defined risk (premium paid)
- No need to perfectly time the bottom
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class COINCallsOnlyPullbackConfig:
    """Configuration for COIN Calls-Only Pullback Strategy."""
    underlying_symbol: str = "COIN"
    fixed_position_value: float = 2000.0
    target_profit_pct: float = 7.5
    stop_loss_pct: float = 25.0
    max_hold_minutes: int = 30
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "15:45:00"
    force_exit_time: str = "15:50:00"
    max_trades_per_day: int = 1

    # Pullback detection settings
    min_pullback_pct: float = 0.3  # Minimum pullback from recent high (%)
    rsi_oversold_threshold: float = 40  # RSI below this = oversold (pullback)
    rsi_recovery_threshold: float = 35  # RSI must be above this for recovery
    min_red_candles_for_pullback: int = 2  # Need at least 2 red candles

    # Recovery detection settings
    require_green_candle: bool = True  # Last bar must be green for entry
    min_bounce_pct: float = 0.05  # Minimum bounce from low (%)

    # Safety limits
    max_rsi_for_entry: float = 65  # Don't buy if RSI too high (overbought)
    max_spread_pct: float = 30.0  # Max bid-ask spread


class COINCallsOnlyPullbackStrategy:
    """
    CALLS-ONLY strategy that waits for pullbacks before entering.

    Logic:
    1. Detect pullback (oversold RSI, red candles, below VWAP)
    2. Wait for recovery signal (green candle, momentum shift)
    3. Enter CALL position
    4. Exit at TP, SL, or force exit time
    """

    def __init__(self, config: COINCallsOnlyPullbackConfig):
        self.config = config

    @staticmethod
    def analyze_pullback_and_recovery(bars: List[Any], config: COINCallsOnlyPullbackConfig = None) -> Dict[str, Any]:
        """
        Analyze bars to detect pullback and recovery conditions.

        Returns dict with:
        - should_enter: bool - True if conditions met for CALL entry
        - pullback_detected: bool - True if price has pulled back
        - recovery_detected: bool - True if showing recovery
        - reasons: list - List of reasons for decision
        - indicators: dict - All calculated indicators
        """
        if config is None:
            config = COINCallsOnlyPullbackConfig()

        result = {
            'should_enter': False,
            'pullback_detected': False,
            'recovery_detected': False,
            'reasons': [],
            'pullback_signals': [],
            'recovery_signals': [],
            'indicators': {}
        }

        if not bars or len(bars) < 20:
            result['reasons'].append("Not enough data")
            return result

        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [getattr(b, 'volume', 0) for b in bars]
        current_price = closes[-1]

        # ========== CALCULATE INDICATORS ==========

        # EMA calculations
        def calc_ema(prices, period):
            if len(prices) < period:
                return sum(prices) / len(prices)
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            return ema

        # RSI calculation
        def calc_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            gains, losses = [], []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(0, change))
                losses.append(max(0, -change))
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        # VWAP calculation
        def calc_vwap(bars_list):
            total_vol = sum(getattr(b, 'volume', 1) for b in bars_list)
            if total_vol == 0:
                return bars_list[-1].close
            total_vwap = sum(((b.high + b.low + b.close) / 3) * getattr(b, 'volume', 1) for b in bars_list)
            return total_vwap / total_vol

        ema_9 = calc_ema(closes, 9)
        ema_20 = calc_ema(closes, 20)
        rsi = calc_rsi(closes, 14)
        vwap = calc_vwap(bars)

        # Recent high (last 20 bars)
        recent_high = max(highs[-20:])
        recent_low = min(lows[-10:])
        pullback_from_high = ((recent_high - current_price) / recent_high) * 100
        bounce_from_low = ((current_price - recent_low) / recent_low) * 100 if recent_low > 0 else 0

        result['indicators'] = {
            'price': current_price,
            'ema_9': ema_9,
            'ema_20': ema_20,
            'rsi': rsi,
            'vwap': vwap,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'pullback_from_high_pct': pullback_from_high,
            'bounce_from_low_pct': bounce_from_low,
        }

        # ========== DETECT PULLBACK ==========
        pullback_score = 0

        # 1. RSI oversold
        if rsi < config.rsi_oversold_threshold:
            pullback_score += 3
            result['pullback_signals'].append(f"RSI oversold ({rsi:.1f} < {config.rsi_oversold_threshold})")
        elif rsi < 45:
            pullback_score += 1
            result['pullback_signals'].append(f"RSI low ({rsi:.1f})")

        # 2. Price below VWAP
        if current_price < vwap:
            pullback_score += 2
            result['pullback_signals'].append(f"Below VWAP (${current_price:.2f} < ${vwap:.2f})")

        # 3. Pullback from recent high
        if pullback_from_high >= config.min_pullback_pct:
            pullback_score += 2
            result['pullback_signals'].append(f"Pulled back {pullback_from_high:.2f}% from high ${recent_high:.2f}")

        # 4. Red candles count (last 5 bars)
        last_5 = bars[-5:]
        red_count = sum(1 for b in last_5 if b.close < b.open)
        if red_count >= config.min_red_candles_for_pullback:
            pullback_score += 2
            result['pullback_signals'].append(f"{red_count}/5 red candles (pullback confirmed)")
        elif red_count >= 2:
            pullback_score += 1
            result['pullback_signals'].append(f"{red_count}/5 red candles")

        # 5. Price below EMAs
        if current_price < ema_9 < ema_20:
            pullback_score += 2
            result['pullback_signals'].append("Price below both EMAs (bearish structure)")
        elif current_price < ema_20:
            pullback_score += 1
            result['pullback_signals'].append("Price below EMA20")

        result['pullback_detected'] = pullback_score >= 4
        result['indicators']['pullback_score'] = pullback_score

        # ========== DETECT RECOVERY ==========
        recovery_score = 0

        # 1. Last bar is GREEN (most important!)
        last_bar = bars[-1]
        if last_bar.close > last_bar.open:
            recovery_score += 3
            bar_change = ((last_bar.close - last_bar.open) / last_bar.open) * 100
            result['recovery_signals'].append(f"Last bar GREEN (+{bar_change:.2f}%)")
        else:
            result['recovery_signals'].append("Last bar is RED - waiting for green candle")

        # 2. Bouncing from low
        if bounce_from_low >= config.min_bounce_pct:
            recovery_score += 2
            result['recovery_signals'].append(f"Bounced {bounce_from_low:.2f}% from low ${recent_low:.2f}")

        # 3. RSI turning up (above recovery threshold)
        if rsi > config.rsi_recovery_threshold:
            recovery_score += 1
            result['recovery_signals'].append(f"RSI recovering ({rsi:.1f} > {config.rsi_recovery_threshold})")

        # 4. Price momentum positive (last 3 bars)
        if len(bars) >= 3:
            momentum_3 = ((closes[-1] - closes[-3]) / closes[-3]) * 100
            if momentum_3 > 0:
                recovery_score += 2
                result['recovery_signals'].append(f"3-bar momentum positive (+{momentum_3:.2f}%)")
            elif momentum_3 > -0.1:
                recovery_score += 1
                result['recovery_signals'].append(f"3-bar momentum stabilizing ({momentum_3:.2f}%)")

        # 5. Higher low pattern (last bar low > previous bar low)
        if len(bars) >= 2 and bars[-1].low > bars[-2].low:
            recovery_score += 1
            result['recovery_signals'].append("Higher low pattern (reversal sign)")

        result['recovery_detected'] = recovery_score >= 4
        result['indicators']['recovery_score'] = recovery_score

        # ========== FINAL ENTRY DECISION ==========

        # Safety checks
        if rsi > config.max_rsi_for_entry:
            result['reasons'].append(f"RSI too high ({rsi:.1f} > {config.max_rsi_for_entry}) - overbought, skip")
            return result

        # Need green candle for entry (configurable)
        if config.require_green_candle and last_bar.close <= last_bar.open:
            result['reasons'].append("Waiting for green candle confirmation")
            return result

        # Entry decision
        if result['pullback_detected'] and result['recovery_detected']:
            result['should_enter'] = True
            result['reasons'].append(f"PULLBACK + RECOVERY DETECTED")
            result['reasons'].append(f"Pullback Score: {pullback_score}/10 | Recovery Score: {recovery_score}/9")
            result['reasons'].append(">>> BUY CALL on the bounce!")
        elif result['pullback_detected']:
            result['reasons'].append(f"Pullback detected (score {pullback_score}) but no recovery yet")
            result['reasons'].append("Waiting for green candle / momentum shift")
        elif result['recovery_detected']:
            result['reasons'].append(f"Recovery signals present (score {recovery_score}) but no clear pullback")
            result['reasons'].append("May be chasing - wait for better entry")
        else:
            result['reasons'].append("No clear pullback or recovery pattern")
            result['reasons'].append("Waiting for dip to buy")

        return result

    @staticmethod
    def get_entry_signal(bars: List[Any], config: COINCallsOnlyPullbackConfig = None) -> Dict[str, Any]:
        """
        Main entry point - returns trading signal.

        Returns:
            dict with:
            - signal: 'BULLISH' (buy call) or 'NEUTRAL' (no trade)
            - option_type: Always 'CALL'
            - should_enter: bool
            - analysis: full analysis dict
        """
        analysis = COINCallsOnlyPullbackStrategy.analyze_pullback_and_recovery(bars, config)

        return {
            'signal': 'BULLISH' if analysis['should_enter'] else 'NEUTRAL',
            'option_type': 'CALL',  # ALWAYS CALLS
            'should_enter': analysis['should_enter'],
            'analysis': analysis
        }
