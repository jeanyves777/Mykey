#!/usr/bin/env python3
"""
THE VOLUME AI - MARA Continuous Momentum Strategy

A continuous trading strategy that:
- Picks ATM contracts at time of each entry
- Uses weekly expiry options (not 0DTE)
- Re-enters after positions close when conditions align
- Requires volume confirmation for entries
- Validates momentum with dual-signal approach

Key Features:
- Unlimited trades per day (with cooldown between trades)
- ATM strike selection based on current underlying price
- Weekly Friday expiry for better time value
- Volume spike detection for entry timing
- Bid/ask spread validation for liquidity
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional, Dict, List, Tuple, Any
import pytz

EST = pytz.timezone('US/Eastern')


@dataclass
class MARAContinuousMomentumConfig:
    """
    Configuration for MARA Continuous Momentum Strategy.
    """
    underlying_symbol: str = "MARA"

    # Position sizing
    fixed_position_value: float = 200.0  # $200 per trade

    # Profit targets
    target_profit_pct: float = 7.5   # Take profit at 7.5%
    stop_loss_pct: float = 25.0      # Stop loss at 25%

    # Trading window
    entry_time_start: str = "09:35:00"  # 5 min after open for stability
    entry_time_end: str = "15:30:00"    # Stop new entries 30 min before close
    force_exit_time: str = "15:50:00"   # Force exit before close

    # Continuous trading settings
    max_hold_minutes: int = 30          # Max hold time per trade
    cooldown_minutes: int = 5           # Wait time after closing a position
    max_trades_per_day: int = 10        # Safety limit

    # Volume requirements
    volume_spike_multiplier: float = 1.5  # Volume must be 1.5x average
    min_volume_threshold: int = 10000     # Minimum absolute volume

    # ATM contract selection
    use_weekly_expiry: bool = True       # Use weekly (Friday) expiry
    max_days_to_expiry: int = 7          # Max DTE for contract selection
    min_days_to_expiry: int = 1          # Min DTE (avoid 0DTE if desired)

    # Liquidity requirements
    max_bid_ask_spread_pct: float = 15.0  # Max spread as % of mid price
    min_option_volume: int = 10           # Minimum option volume
    min_open_interest: int = 50           # Minimum open interest

    # Signal thresholds
    min_signal_score: int = 3            # Minimum technical score (out of 6)
    require_dual_confirmation: bool = True  # Both methods must agree

    # Polling
    poll_interval_seconds: int = 10


def check_btc_alignment(mara_direction: str, btc_change_pct: float) -> bool:
    """
    Validate MARA trade aligns with BTC momentum.

    Args:
        mara_direction: "BULLISH" or "BEARISH"
        btc_change_pct: BTC percentage change (e.g., 2.5 for +2.5%)

    Returns:
        True if trade is aligned with BTC, False if divergence detected
    """
    if mara_direction == "BULLISH" and btc_change_pct > 1.5:
        return True  # BTC up, MARA calls = aligned
    if mara_direction == "BEARISH" and btc_change_pct < -1.5:
        return True  # BTC down, MARA puts = aligned
    if abs(btc_change_pct) < 1.0:
        return True  # BTC flat, MARA moving on own = OK
    return False  # Divergence = skip trade


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values."""
    ema_9: float = 0.0
    ema_20: float = 0.0
    vwap: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    volume: float = 0.0
    avg_volume: float = 0.0
    atr: float = 0.0


@dataclass
class PriceActionSignal:
    """5-minute price action analysis."""
    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    score: int = 0
    candle_trend: str = ""
    higher_highs: bool = False
    lower_lows: bool = False
    momentum: float = 0.0
    last_bar_strength: float = 0.0


@dataclass
class VolumeAnalysis:
    """Volume analysis for entry validation."""
    current_volume: float = 0.0
    avg_volume: float = 0.0
    volume_ratio: float = 0.0
    is_spike: bool = False
    trend: str = "NORMAL"  # SPIKE, HIGH, NORMAL, LOW


@dataclass
class ATMContractInfo:
    """Information about selected ATM contract."""
    symbol: str = ""
    strike: float = 0.0
    expiration: datetime = None
    days_to_expiry: int = 0
    option_type: str = ""  # 'call' or 'put'
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread_pct: float = 0.0
    volume: int = 0
    open_interest: int = 0
    delta: float = 0.0
    is_valid: bool = False
    rejection_reason: str = ""


class MARAContinuousMomentumStrategy:
    """
    MARA Continuous Momentum Strategy

    This strategy continuously monitors MARA for trading opportunities:
    1. Detects volume spikes as entry triggers
    2. Validates momentum with dual-signal approach
    3. Selects ATM contract at time of entry
    4. Uses weekly expiry for better time value
    5. Re-enters after positions close (with cooldown)
    """

    def __init__(self, config: MARAContinuousMomentumConfig):
        self.config = config
        self.last_trade_exit_time: Optional[datetime] = None
        self.trades_today: int = 0

    def is_in_cooldown(self, current_time: datetime) -> Tuple[bool, int]:
        """
        Check if we're in cooldown period after last trade.
        Returns (is_in_cooldown, minutes_remaining)
        """
        if self.last_trade_exit_time is None:
            return False, 0

        elapsed = (current_time - self.last_trade_exit_time).total_seconds() / 60
        remaining = self.config.cooldown_minutes - elapsed

        if remaining > 0:
            return True, int(remaining)
        return False, 0

    def record_trade_exit(self, exit_time: datetime):
        """Record when a trade was exited for cooldown tracking."""
        self.last_trade_exit_time = exit_time
        self.trades_today += 1

    def reset_daily_state(self):
        """Reset daily tracking state."""
        self.last_trade_exit_time = None
        self.trades_today = 0

    def can_trade(self, current_time: datetime) -> Tuple[bool, str]:
        """
        Check if trading is allowed right now.
        Returns (can_trade, reason)
        """
        # Check max trades per day
        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Max trades reached ({self.config.max_trades_per_day})"

        # Check cooldown
        in_cooldown, minutes_left = self.is_in_cooldown(current_time)
        if in_cooldown:
            return False, f"In cooldown ({minutes_left}m remaining)"

        # Check trading window
        entry_start = datetime.strptime(self.config.entry_time_start, "%H:%M:%S").time()
        entry_end = datetime.strptime(self.config.entry_time_end, "%H:%M:%S").time()
        current = current_time.time()

        if current < entry_start:
            return False, f"Before entry window ({self.config.entry_time_start})"
        if current > entry_end:
            return False, f"After entry window ({self.config.entry_time_end})"

        return True, "OK"

    def analyze_volume(self, current_volume: float, avg_volume: float) -> VolumeAnalysis:
        """
        Analyze volume for entry validation.
        """
        analysis = VolumeAnalysis(
            current_volume=current_volume,
            avg_volume=avg_volume
        )

        if avg_volume > 0:
            analysis.volume_ratio = current_volume / avg_volume
        else:
            analysis.volume_ratio = 1.0

        # Classify volume
        if analysis.volume_ratio >= self.config.volume_spike_multiplier:
            analysis.is_spike = True
            analysis.trend = "SPIKE"
        elif analysis.volume_ratio >= 1.2:
            analysis.trend = "HIGH"
        elif analysis.volume_ratio >= 0.8:
            analysis.trend = "NORMAL"
        else:
            analysis.trend = "LOW"

        return analysis

    def calculate_technical_score(
        self,
        price: float,
        indicators: TechnicalIndicators
    ) -> Tuple[int, str, List[str]]:
        """
        Calculate technical score for entry signal.
        Returns (score, direction, reasons)

        Score components (6 total):
        1. EMA Stack alignment
        2. VWAP position
        3. RSI momentum
        4. MACD crossover
        5. Bollinger Band position
        6. Volume confirmation
        """
        bullish_points = 0
        bearish_points = 0
        reasons = []

        # 1. EMA Stack (Price vs EMA9 vs EMA20)
        if price > indicators.ema_9 > indicators.ema_20:
            bullish_points += 1
            reasons.append("EMA_BULL")
        elif price < indicators.ema_9 < indicators.ema_20:
            bearish_points += 1
            reasons.append("EMA_BEAR")

        # 2. VWAP Position
        if indicators.vwap > 0:
            if price > indicators.vwap * 1.002:  # Above VWAP by 0.2%
                bullish_points += 1
                reasons.append("VWAP_BULL")
            elif price < indicators.vwap * 0.998:  # Below VWAP by 0.2%
                bearish_points += 1
                reasons.append("VWAP_BEAR")

        # 3. RSI Momentum
        if indicators.rsi > 55:
            bullish_points += 1
            reasons.append(f"RSI_BULL({indicators.rsi:.0f})")
        elif indicators.rsi < 45:
            bearish_points += 1
            reasons.append(f"RSI_BEAR({indicators.rsi:.0f})")

        # 4. MACD Crossover
        macd_diff = indicators.macd - indicators.macd_signal
        if macd_diff > 0.01:
            bullish_points += 1
            reasons.append("MACD_BULL")
        elif macd_diff < -0.01:
            bearish_points += 1
            reasons.append("MACD_BEAR")

        # 5. Bollinger Band Position
        if indicators.bb_upper > 0 and indicators.bb_lower > 0:
            bb_range = indicators.bb_upper - indicators.bb_lower
            if bb_range > 0:
                bb_pct = (price - indicators.bb_lower) / bb_range
                if bb_pct > 0.7:
                    bullish_points += 1
                    reasons.append(f"BB_BULL({bb_pct:.0%})")
                elif bb_pct < 0.3:
                    bearish_points += 1
                    reasons.append(f"BB_BEAR({bb_pct:.0%})")

        # 6. Volume Confirmation
        if indicators.avg_volume > 0:
            vol_ratio = indicators.volume / indicators.avg_volume
            if vol_ratio >= self.config.volume_spike_multiplier:
                # Volume confirms whatever direction is stronger
                if bullish_points > bearish_points:
                    bullish_points += 1
                    reasons.append(f"VOL_CONFIRM({vol_ratio:.1f}x)")
                elif bearish_points > bullish_points:
                    bearish_points += 1
                    reasons.append(f"VOL_CONFIRM({vol_ratio:.1f}x)")

        # Determine direction
        if bullish_points >= self.config.min_signal_score and bullish_points > bearish_points:
            return bullish_points, "BULLISH", reasons
        elif bearish_points >= self.config.min_signal_score and bearish_points > bullish_points:
            return bearish_points, "BEARISH", reasons
        else:
            return max(bullish_points, bearish_points), "NEUTRAL", reasons

    def analyze_price_action(self, bars: List[Dict], verbose: bool = False) -> PriceActionSignal:
        """
        Analyze 5-minute price action for momentum.
        Expects list of OHLCV bars (most recent last).

        PA Score Components (5 total, need ≥3 for directional signal):
        1. Candle Trend: ≥3 green candles (bullish) or ≥3 red (bearish)
        2. Higher Highs (bullish) / Lower Lows (bearish)
        3. Momentum: >0.3% (bullish) or <-0.3% (bearish)
        4. Last Bar Strength: close near high >0.7 (bullish) or near low <0.3 (bearish)
        5. Above/Below 5-bar average by 0.2%
        """
        signal = PriceActionSignal()

        if len(bars) < 5:
            if verbose:
                print(f"    [PA] Not enough bars: {len(bars)}/5 required")
            return signal

        recent_bars = bars[-5:]

        # Count green vs red candles (support both dict and object access)
        def get_bar_val(bar, key):
            return bar[key] if isinstance(bar, dict) else getattr(bar, key)

        green_count = sum(1 for b in recent_bars if get_bar_val(b, 'close') > get_bar_val(b, 'open'))
        red_count = 5 - green_count

        # Higher highs / Lower lows
        highs = [get_bar_val(b, 'high') for b in recent_bars]
        lows = [get_bar_val(b, 'low') for b in recent_bars]

        higher_highs = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
        lower_lows = all(lows[i] <= lows[i-1] for i in range(1, len(lows)))

        signal.higher_highs = higher_highs
        signal.lower_lows = lower_lows

        # 5-bar momentum
        first_close = get_bar_val(recent_bars[0], 'close')
        last_close = get_bar_val(recent_bars[-1], 'close')
        if first_close > 0:
            signal.momentum = (last_close - first_close) / first_close * 100

        # Last bar strength
        last_bar = recent_bars[-1]
        bar_range = get_bar_val(last_bar, 'high') - get_bar_val(last_bar, 'low')
        if bar_range > 0:
            signal.last_bar_strength = (get_bar_val(last_bar, 'close') - get_bar_val(last_bar, 'low')) / bar_range

        # 5-bar average comparison
        avg_close = sum(get_bar_val(b, 'close') for b in recent_bars) / 5

        # Score calculation with verbose logging
        bullish_score = 0
        bearish_score = 0
        pa_details = []

        # 1. Candle Trend
        if green_count >= 3:
            bullish_score += 1
            signal.candle_trend = f"{green_count}/5 GREEN"
            pa_details.append(f"Candles:{green_count}/5 GREEN +1B")
        elif red_count >= 3:
            bearish_score += 1
            signal.candle_trend = f"{red_count}/5 RED"
            pa_details.append(f"Candles:{red_count}/5 RED +1S")
        else:
            pa_details.append(f"Candles:{green_count}G/{red_count}R (mixed)")

        # 2. Higher Highs / Lower Lows
        if higher_highs:
            bullish_score += 1
            pa_details.append("HH +1B")
        elif lower_lows:
            bearish_score += 1
            pa_details.append("LL +1S")
        else:
            pa_details.append("No HH/LL")

        # 3. Momentum
        if signal.momentum > 0.3:
            bullish_score += 1
            pa_details.append(f"Mom:{signal.momentum:+.2f}% +1B")
        elif signal.momentum < -0.3:
            bearish_score += 1
            pa_details.append(f"Mom:{signal.momentum:+.2f}% +1S")
        else:
            pa_details.append(f"Mom:{signal.momentum:+.2f}% (flat)")

        # 4. Last Bar Strength
        if signal.last_bar_strength > 0.7:
            bullish_score += 1
            pa_details.append(f"LastBar:{signal.last_bar_strength:.0%} +1B")
        elif signal.last_bar_strength < 0.3:
            bearish_score += 1
            pa_details.append(f"LastBar:{signal.last_bar_strength:.0%} +1S")
        else:
            pa_details.append(f"LastBar:{signal.last_bar_strength:.0%} (mid)")

        # 5. Above/Below 5-bar average
        pct_from_avg = (last_close - avg_close) / avg_close * 100
        if last_close > avg_close * 1.002:
            bullish_score += 1
            pa_details.append(f"vsAvg:{pct_from_avg:+.2f}% +1B")
        elif last_close < avg_close * 0.998:
            bearish_score += 1
            pa_details.append(f"vsAvg:{pct_from_avg:+.2f}% +1S")
        else:
            pa_details.append(f"vsAvg:{pct_from_avg:+.2f}% (near)")

        signal.score = max(bullish_score, bearish_score)

        if bullish_score >= 3 and bullish_score > bearish_score:
            signal.direction = "BULLISH"
        elif bearish_score >= 3 and bearish_score > bullish_score:
            signal.direction = "BEARISH"
        else:
            signal.direction = "NEUTRAL"

        # Store details for verbose output
        signal.candle_trend = f"{green_count}G/{red_count}R | " + " | ".join(pa_details)

        if verbose:
            print(f"    [PA] {signal.direction}({signal.score}) Bull={bullish_score} Bear={bearish_score}")
            for detail in pa_details:
                print(f"        {detail}")

        return signal

    def validate_contract(
        self,
        contract: ATMContractInfo,
        underlying_price: float
    ) -> ATMContractInfo:
        """
        Validate that a contract meets liquidity requirements.
        """
        contract.is_valid = True

        # Check bid/ask spread
        if contract.mid > 0:
            contract.spread_pct = (contract.ask - contract.bid) / contract.mid * 100
        else:
            contract.is_valid = False
            contract.rejection_reason = "No mid price"
            return contract

        if contract.spread_pct > self.config.max_bid_ask_spread_pct:
            contract.is_valid = False
            contract.rejection_reason = f"Spread too wide ({contract.spread_pct:.1f}%)"
            return contract

        # Check volume
        if contract.volume < self.config.min_option_volume:
            contract.is_valid = False
            contract.rejection_reason = f"Low volume ({contract.volume})"
            return contract

        # Check open interest
        if contract.open_interest < self.config.min_open_interest:
            contract.is_valid = False
            contract.rejection_reason = f"Low OI ({contract.open_interest})"
            return contract

        # Check days to expiry
        if contract.days_to_expiry < self.config.min_days_to_expiry:
            contract.is_valid = False
            contract.rejection_reason = f"DTE too low ({contract.days_to_expiry})"
            return contract

        if contract.days_to_expiry > self.config.max_days_to_expiry:
            contract.is_valid = False
            contract.rejection_reason = f"DTE too high ({contract.days_to_expiry})"
            return contract

        return contract

    def get_entry_signal(
        self,
        price: float,
        indicators: TechnicalIndicators,
        price_action: PriceActionSignal,
        volume_analysis: VolumeAnalysis,
        current_time: datetime,
        btc_change_pct: float = 0.0
    ) -> Tuple[bool, str, str, List[str]]:
        """
        Determine if we should enter a trade.
        Returns (should_enter, direction, option_type, reasons)

        Args:
            btc_change_pct: BTC percentage change for alignment filter
        """
        reasons = []

        # Check if we can trade
        can_trade, trade_reason = self.can_trade(current_time)
        if not can_trade:
            return False, "NONE", "", [trade_reason]

        # Check volume requirement
        if not volume_analysis.is_spike:
            if volume_analysis.current_volume < self.config.min_volume_threshold:
                return False, "NONE", "", ["Volume below threshold"]
            reasons.append(f"Vol: {volume_analysis.trend}")
        else:
            reasons.append(f"Vol SPIKE: {volume_analysis.volume_ratio:.1f}x")

        # Get technical score
        tech_score, tech_direction, tech_reasons = self.calculate_technical_score(
            price, indicators
        )
        reasons.extend(tech_reasons)

        # V2: AGGRESSIVE MODE - Trade on technical signal alone
        # When dual confirmation is disabled, use technical signal as primary
        if self.config.require_dual_confirmation:
            # Strict mode: both must agree
            if tech_direction != price_action.direction:
                return False, "CONFLICT", "", [
                    f"Tech={tech_direction}, PA={price_action.direction}"
                ]
            if tech_direction == "NEUTRAL" or price_action.direction == "NEUTRAL":
                return False, "NEUTRAL", "", ["No clear direction"]
            direction = tech_direction
        else:
            # Aggressive mode: use technical signal, allow neutral PA
            if tech_direction == "NEUTRAL":
                # If tech is neutral, try price action
                if price_action.direction in ("BULLISH", "BEARISH"):
                    direction = price_action.direction
                    reasons.append(f"PA direction: {direction}")
                else:
                    return False, "NEUTRAL", "", ["Both signals neutral"]
            else:
                direction = tech_direction
                reasons.append(f"Tech direction: {direction}")

        # Check BTC alignment (skip if BTC data not available)
        if abs(btc_change_pct) > 0.01:  # Only check if we have BTC data
            if direction in ("BULLISH", "BEARISH"):
                if not check_btc_alignment(direction, btc_change_pct):
                    # V2: Log but don't block - just warn
                    reasons.append(f"BTC divergence warning ({btc_change_pct:+.1f}%)")
                else:
                    reasons.append(f"BTC aligned ({btc_change_pct:+.1f}%)")

        # Determine option type based on direction
        if direction == "BULLISH":
            return True, "BULLISH", "call", reasons
        elif direction == "BEARISH":
            return True, "BEARISH", "put", reasons

        return False, "NONE", "", reasons

    def calculate_position_size(self, option_price: float) -> int:
        """Calculate number of contracts based on position value."""
        if option_price <= 0:
            return 0

        contract_value = option_price * 100  # Options are 100 shares
        qty = int(self.config.fixed_position_value / contract_value)

        return max(1, qty)  # At least 1 contract

    @staticmethod
    def check_pullback_entry_htf(bars_5min: list, direction: str) -> dict:
        """
        V4 PULLBACK DETECTION LAYER using 5-minute bars (higher timeframe = less noise).

        For BULLISH signals: Wait for dip/pullback + recovery (green candle)
        For BEARISH signals: Wait for bounce/rally + rejection (red candle)

        Using 5-min bars instead of 1-min reduces noise and gives more reliable signals.

        Parameters
        ----------
        bars_5min : list
            List of 5-minute bar objects with: open, close, high, low, volume attributes
        direction : str
            'BULLISH' or 'BEARISH' - the intended trade direction

        Returns
        -------
        dict
            {
                'ready_to_enter': bool,
                'pullback_detected': bool,
                'recovery_detected': bool,
                'pullback_score': int,
                'recovery_score': int,
                'reasons': list,
                'indicators': dict
            }
        """
        if not bars_5min or len(bars_5min) < 10:
            return {
                'ready_to_enter': False,
                'pullback_detected': False,
                'recovery_detected': False,
                'pullback_score': 0,
                'recovery_score': 0,
                'reasons': ['Not enough 5-min data for pullback analysis'],
                'indicators': {}
            }

        # Support both dict and object access
        def get_val(bar, key):
            return bar[key] if isinstance(bar, dict) else getattr(bar, key)

        closes = [get_val(b, 'close') for b in bars_5min]
        opens = [get_val(b, 'open') for b in bars_5min]
        highs = [get_val(b, 'high') for b in bars_5min]
        lows = [get_val(b, 'low') for b in bars_5min]
        current_price = closes[-1]

        # Calculate key indicators
        def calc_ema(prices, period):
            if len(prices) < period:
                return sum(prices) / len(prices)
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            return ema

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

        def calc_vwap(bars_list):
            total_vol = sum(get_val(b, 'volume') if hasattr(b, 'volume') or (isinstance(b, dict) and 'volume' in b) else 1 for b in bars_list)
            if total_vol == 0:
                return get_val(bars_list[-1], 'close')
            total_vwap = sum(((get_val(b, 'high') + get_val(b, 'low') + get_val(b, 'close')) / 3) *
                           (get_val(b, 'volume') if hasattr(b, 'volume') or (isinstance(b, dict) and 'volume' in b) else 1)
                           for b in bars_list)
            return total_vwap / total_vol

        ema_9 = calc_ema(closes, 9)
        ema_20 = calc_ema(closes, min(20, len(closes)))
        rsi = calc_rsi(closes, min(14, len(closes) - 1))
        vwap = calc_vwap(bars_5min)

        # Recent extremes (using 5-min bars = less noise)
        recent_high = max(highs[-10:])  # Last 50 minutes
        recent_low = min(lows[-6:])     # Last 30 minutes
        pullback_from_high = ((recent_high - current_price) / recent_high) * 100
        bounce_from_low = ((current_price - recent_low) / recent_low) * 100 if recent_low > 0 else 0

        indicators = {
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

        reasons = []
        pullback_score = 0
        recovery_score = 0

        last_bar = bars_5min[-1]
        last_3_bars = bars_5min[-3:]  # 15 minutes of data

        if direction == 'BULLISH':
            # ========== PULLBACK DETECTION (for calls - price pulled back) ==========
            # 1. RSI in oversold/neutral zone (more relaxed for 5-min)
            if rsi < 45:
                pullback_score += 3
                reasons.append(f"RSI low zone ({rsi:.1f})")
            elif rsi < 55:
                pullback_score += 1
                reasons.append(f"RSI neutral ({rsi:.1f})")

            # 2. Price below VWAP (indicates selling pressure)
            if current_price < vwap:
                pullback_score += 2
                reasons.append(f"Below VWAP (${current_price:.2f} < ${vwap:.2f})")

            # 3. Pullback from recent high (use larger % for 5-min)
            if pullback_from_high >= 0.5:
                pullback_score += 2
                reasons.append(f"Pulled back {pullback_from_high:.2f}% from high")
            elif pullback_from_high >= 0.25:
                pullback_score += 1
                reasons.append(f"Small pullback {pullback_from_high:.2f}%")

            # 4. Red candles in last 3 bars (5-min = 15 min of selling)
            red_count = sum(1 for b in last_3_bars if get_val(b, 'close') < get_val(b, 'open'))
            if red_count >= 2:
                pullback_score += 2
                reasons.append(f"{red_count}/3 red 5-min candles (selling)")
            elif red_count >= 1:
                pullback_score += 1
                reasons.append(f"{red_count}/3 red candles")

            # ========== RECOVERY DETECTION (for calls - turning bullish) ==========
            # 1. Last 5-min bar is GREEN (key signal!)
            if get_val(last_bar, 'close') > get_val(last_bar, 'open'):
                recovery_score += 3
                bar_change = ((get_val(last_bar, 'close') - get_val(last_bar, 'open')) / get_val(last_bar, 'open')) * 100
                reasons.append(f"Last 5-min bar GREEN (+{bar_change:.2f}%) - reversal")
            else:
                reasons.append("Waiting for green 5-min candle...")

            # 2. Bouncing from low
            if bounce_from_low >= 0.15:
                recovery_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")

            # 3. RSI recovering (not oversold anymore)
            if rsi > 40:
                recovery_score += 1
                reasons.append(f"RSI recovering ({rsi:.1f})")

            # 4. Positive 2-bar momentum (10 minutes)
            if len(bars_5min) >= 2:
                momentum_2 = ((closes[-1] - closes[-2]) / closes[-2]) * 100
                if momentum_2 > 0:
                    recovery_score += 2
                    reasons.append(f"10-min momentum positive (+{momentum_2:.2f}%)")

            # 5. Higher low pattern
            if len(bars_5min) >= 2 and get_val(bars_5min[-1], 'low') > get_val(bars_5min[-2], 'low'):
                recovery_score += 1
                reasons.append("Higher low (reversal)")

            indicators['pullback_score'] = pullback_score
            indicators['recovery_score'] = recovery_score

            pullback_detected = pullback_score >= 3  # Slightly relaxed for 5-min
            recovery_detected = recovery_score >= 4

            # Safety: Don't enter if RSI too high
            if rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.1f}) - skip")
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': recovery_detected,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons,
                    'indicators': indicators
                }

            # Require green candle for call entries
            if get_val(last_bar, 'close') <= get_val(last_bar, 'open'):
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': False,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons,
                    'indicators': indicators
                }

            ready = pullback_detected and recovery_detected
            if ready:
                reasons.append(">>> PULLBACK + RECOVERY: BUY CALL NOW!")

        else:  # BEARISH
            # ========== BOUNCE DETECTION (for puts - price bounced up) ==========
            # 1. RSI in overbought/high zone
            if rsi > 55:
                pullback_score += 3
                reasons.append(f"RSI high zone ({rsi:.1f})")
            elif rsi > 45:
                pullback_score += 1
                reasons.append(f"RSI neutral-high ({rsi:.1f})")

            # 2. Price above VWAP
            if current_price > vwap:
                pullback_score += 2
                reasons.append(f"Above VWAP (${current_price:.2f} > ${vwap:.2f})")

            # 3. Bounce from recent low
            if bounce_from_low >= 0.5:
                pullback_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")
            elif bounce_from_low >= 0.25:
                pullback_score += 1
                reasons.append(f"Small bounce {bounce_from_low:.2f}%")

            # 4. Green candles in last 3 bars (buying pressure)
            green_count = sum(1 for b in last_3_bars if get_val(b, 'close') > get_val(b, 'open'))
            if green_count >= 2:
                pullback_score += 2
                reasons.append(f"{green_count}/3 green 5-min candles (buying)")
            elif green_count >= 1:
                pullback_score += 1
                reasons.append(f"{green_count}/3 green candles")

            # ========== REJECTION DETECTION (for puts - turning bearish) ==========
            # 1. Last 5-min bar is RED (key signal!)
            if get_val(last_bar, 'close') < get_val(last_bar, 'open'):
                recovery_score += 3
                bar_change = ((get_val(last_bar, 'close') - get_val(last_bar, 'open')) / get_val(last_bar, 'open')) * 100
                reasons.append(f"Last 5-min bar RED ({bar_change:.2f}%) - rejection")
            else:
                reasons.append("Waiting for red 5-min candle...")

            # 2. Pulling back from high
            if pullback_from_high >= 0.15:
                recovery_score += 2
                reasons.append(f"Pulled back {pullback_from_high:.2f}% from high")

            # 3. RSI turning down
            if rsi < 60:
                recovery_score += 1
                reasons.append(f"RSI turning down ({rsi:.1f})")

            # 4. Negative 2-bar momentum
            if len(bars_5min) >= 2:
                momentum_2 = ((closes[-1] - closes[-2]) / closes[-2]) * 100
                if momentum_2 < 0:
                    recovery_score += 2
                    reasons.append(f"10-min momentum negative ({momentum_2:.2f}%)")

            # 5. Lower high pattern
            if len(bars_5min) >= 2 and get_val(bars_5min[-1], 'high') < get_val(bars_5min[-2], 'high'):
                recovery_score += 1
                reasons.append("Lower high (rejection)")

            indicators['pullback_score'] = pullback_score
            indicators['recovery_score'] = recovery_score

            pullback_detected = pullback_score >= 3
            recovery_detected = recovery_score >= 4

            # Safety: Don't enter if RSI too low
            if rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.1f}) - skip")
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': recovery_detected,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons,
                    'indicators': indicators
                }

            # Require red candle for put entries
            if get_val(last_bar, 'close') >= get_val(last_bar, 'open'):
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': False,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons,
                    'indicators': indicators
                }

            ready = pullback_detected and recovery_detected
            if ready:
                reasons.append(">>> BOUNCE + REJECTION: BUY PUT NOW!")

        return {
            'ready_to_enter': ready,
            'pullback_detected': pullback_detected,
            'recovery_detected': recovery_detected,
            'pullback_score': pullback_score,
            'recovery_score': recovery_score,
            'reasons': reasons,
            'indicators': indicators
        }

    def get_weekly_expiry(self, from_date: datetime) -> datetime:
        """
        Get the next weekly (Friday) expiry date.
        If today is Friday and before market close, use today.
        Otherwise use next Friday.
        """
        # Find next Friday
        days_until_friday = (4 - from_date.weekday()) % 7

        # If it's Friday and we're in trading hours, use today
        if days_until_friday == 0:
            return from_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Otherwise next Friday
        if days_until_friday == 0:
            days_until_friday = 7

        next_friday = from_date + timedelta(days=days_until_friday)
        return next_friday.replace(hour=0, minute=0, second=0, microsecond=0)
