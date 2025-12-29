"""
MARA Daily 0DTE Momentum Strategy - Alpaca Implementation

MARA (Marathon Digital) 0DTE Options Strategy using the same
dual-confirmation signal system as COIN.

# Find THIS WEEK's Friday - options expire same week at 4PM Friday
# Monday (0) -> Friday in 4 days
# Tuesday (1) -> Friday in 3 days
# Wednesday (2) -> Friday in 2 days
# Thursday (3) -> Friday in 1 day
# Friday (4) -> Friday TODAY (0DTE)

CRITICAL REQUIREMENTS:
======================
1. Allows up to 3 trades per day (configurable)
2. FIXED position size: $200 per trade (configurable)
3. Indicators determine DIRECTION only (CALL vs PUT)
4. If indicators neutral/weak -> NO TRADE (MARA is more volatile)
5. Entry window: 9:30-15:45 EST (extended for MARA)
6. Force exit: 15:50 PM EST (before 4 PM 0DTE expiration)
7. NEVER holds overnight
8. Daily profit target: +15% stops trading for the day

STRATEGY LOGIC:
===============
- Uses EMA, RSI, MACD, Bollinger Bands, Volume for direction analysis
- Buys ATM (at-the-money) weekly options based on market direction
- Target profit: 25% (high target to let trailing stop work)
- Stop loss: 25% (configurable)
- Trailing Stop: Activates at +10%, trails 15% below high water mark
- Pullback detection: Waits for pullback + recovery for better entries
- MARA has smaller option premiums, so we can buy more contracts
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import pytz

from ..strategy.base import Strategy, StrategyConfig
from ..strategy.logger import LogColor
from ..core.models import (
    Bar, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Position, Instrument, OptionContract, InstrumentType, OptionType
)
from ..core.events import FillEvent
from ..indicators import (
    ExponentialMovingAverage,
    RelativeStrengthIndex,
    MACD,
    BollingerBands,
    AverageTrueRange,
)


@dataclass
class MARADaily0DTEMomentumConfig(StrategyConfig):
    """
    Configuration for MARA Daily 0DTE Momentum Strategy.
    """
    underlying_symbol: str = "MARA"
    instrument_id: str = ""
    bar_type: str = ""
    fixed_position_value: float = 200.0
    target_profit_pct: Decimal = Decimal("25.0")  # Increased to allow trailing stop to work
    stop_loss_pct: Decimal = Decimal("25.0")
    # Trailing Stop Configuration
    trailing_stop_enabled: bool = True
    trailing_trigger_pct: Decimal = Decimal("10.0")  # Start trailing after 10% profit
    trailing_distance_pct: Decimal = Decimal("5.0")  # Trail 5% below high water mark (TIGHT to lock profit!)
    min_hold_minutes: int = 5
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "15:45:00"
    force_exit_time: str = "15:50:00"
    max_hold_minutes: int = 30
    max_trades_per_day: int = 3  # Allow up to 3 trades per day
    daily_profit_target_pct: float = 15.0  # Stop trading if daily P&L reaches +15%
    poll_interval_seconds: int = 10

    # Indicator settings
    fast_ema_period: int = 9
    slow_ema_period: int = 20
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2.0
    min_volume_ratio: float = 1.0
    max_bid_ask_spread_pct: float = 30.0
    min_option_premium: float = 0.10  # Lower for MARA options
    max_option_premium: float = 5.0   # Lower for MARA options
    request_bars: bool = True


class MARADaily0DTEMomentumStrategy(Strategy):
    """
    MARA Daily 0DTE Momentum Strategy - Backtest Compatible

    This class extends Strategy base class for backtest engine compatibility.
    Uses the same logic as COIN strategy but configured for MARA.
    """

    def __init__(self, config: MARADaily0DTEMomentumConfig):
        super().__init__(config)
        self.config: MARADaily0DTEMomentumConfig = config

        self.log.info("=" * 80, LogColor.BLUE)
        self.log.info("INITIALIZING MARA DAILY 0DTE MOMENTUM STRATEGY", LogColor.BLUE)
        self.log.info("=" * 80, LogColor.BLUE)

        # Initialize indicators
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.macd = MACD(config.macd_fast_period, config.macd_slow_period, config.macd_signal_period)
        self.bb = BollingerBands(config.bb_period, config.bb_std_dev)
        self.atr = AverageTrueRange(14)

        self.log.info(f"   EMA: Fast={config.fast_ema_period}, Slow={config.slow_ema_period}", LogColor.BLUE)
        self.log.info(f"   RSI: Period={config.rsi_period}", LogColor.BLUE)
        self.log.info(f"   MACD: {config.macd_fast_period}/{config.macd_slow_period}/{config.macd_signal_period}", LogColor.BLUE)

        # Trading state
        self.traded_today = False
        self.last_trade_date: Optional[datetime] = None
        self.entry_price: Optional[float] = None
        self.entry_timestamp: Optional[int] = None
        self.entry_bar_datetime: Optional[datetime] = None
        self.current_position: Optional[Position] = None
        self.current_option_symbol: Optional[str] = None
        self.bars_history: List[Bar] = []

        # Multi-trade tracking
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.trades_today: List[Dict] = []

        # Parse times
        self.entry_start = datetime.strptime(config.entry_time_start, "%H:%M:%S").time()
        self.entry_end = datetime.strptime(config.entry_time_end, "%H:%M:%S").time()
        self.force_exit = datetime.strptime(config.force_exit_time, "%H:%M:%S").time()

        self.est_tz = pytz.timezone('America/New_York')

        self.log.info(f"   Entry window: {config.entry_time_start} - {config.entry_time_end}", LogColor.BLUE)
        self.log.info(f"   Force exit: {config.force_exit_time}", LogColor.BLUE)
        self.log.info(f"   Position size: ${config.fixed_position_value}", LogColor.BLUE)
        self.log.info(f"   Max trades/day: {config.max_trades_per_day}", LogColor.BLUE)

    def on_bar(self, bar: Bar):
        """Process incoming bar data."""
        # Only process underlying bars
        if bar.symbol != self.config.underlying_symbol:
            return

        # Store bar history
        self.bars_history.append(bar)
        if len(self.bars_history) > 200:
            self.bars_history = self.bars_history[-200:]

        # Update indicators
        self.fast_ema.update(bar.close)
        self.slow_ema.update(bar.close)
        self.rsi.update(bar.close)
        self.macd.update(bar.close)
        self.bb.update(bar.close)
        self.atr.update_from_bar(bar)

        # Get current time in EST
        bar_dt = bar.timestamp
        if bar_dt.tzinfo is None:
            bar_dt = pytz.utc.localize(bar_dt)
        bar_est = bar_dt.astimezone(self.est_tz)
        current_time = bar_est.time()
        current_date = bar_est.date()

        # Reset daily tracking
        if self.last_trade_date != current_date:
            self.traded_today = False
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.trades_today = []
            self.last_trade_date = current_date

        # Check for force exit time
        if current_time >= self.force_exit:
            if self.current_position:
                self._exit_position("FORCE_EXIT", bar)
            return

        # Check daily profit target
        if self.daily_pnl >= self.config.daily_profit_target_pct:
            if self.current_position:
                self._exit_position("DAILY_TARGET", bar)
            return

        # Check if we have a position to manage
        if self.current_position or self.current_option_symbol:
            self._manage_position(bar)
            return

        # Check entry conditions
        if self.daily_trades_count >= self.config.max_trades_per_day:
            return

        if not (self.entry_start <= current_time <= self.entry_end):
            return

        # Need enough bars for indicators
        if len(self.bars_history) < 30:
            return

        # Calculate signal
        signal = self._calculate_signal()
        if signal['signal'] == 'NEUTRAL':
            return

        # Find ATM option and enter
        self._enter_position(signal, bar)

    def _calculate_signal(self) -> dict:
        """Calculate trading signal based on indicators."""
        if not self.fast_ema.initialized or not self.slow_ema.initialized:
            return {'signal': 'NEUTRAL', 'bullish_score': 0, 'bearish_score': 0}

        bullish_score = 0
        bearish_score = 0

        # EMA crossover
        if self.fast_ema.value > self.slow_ema.value:
            bullish_score += 2
        else:
            bearish_score += 2

        # RSI
        if self.rsi.initialized:
            if self.rsi.value < 40:
                bullish_score += 2
            elif self.rsi.value > 60:
                bearish_score += 2

        # MACD
        if self.macd.initialized:
            if self.macd.histogram > 0:
                bullish_score += 2
            else:
                bearish_score += 2

        # Bollinger Bands
        if self.bb.initialized:
            current_price = self.bars_history[-1].close
            if current_price > self.bb.middle:
                bullish_score += 1
            else:
                bearish_score += 1

        # Determine signal (lower threshold for more signals)
        if bullish_score >= 4 and bullish_score > bearish_score:
            return {'signal': 'BULLISH', 'bullish_score': bullish_score, 'bearish_score': bearish_score}
        elif bearish_score >= 4 and bearish_score > bullish_score:
            return {'signal': 'BEARISH', 'bullish_score': bullish_score, 'bearish_score': bearish_score}
        else:
            return {'signal': 'NEUTRAL', 'bullish_score': bullish_score, 'bearish_score': bearish_score}

    def _enter_position(self, signal: dict, bar: Bar):
        """Enter a new position."""
        current_price = bar.close
        is_call = signal['signal'] == 'BULLISH'

        # Find ATM option
        option = self._find_atm_option(current_price, is_call, bar)
        if not option:
            return

        # Get option price
        option_bar = self._engine.current_bars.get(option.symbol)
        if not option_bar:
            return

        option_price = option_bar.close
        if option_price < self.config.min_option_premium or option_price > self.config.max_option_premium:
            return

        # Calculate quantity
        quantity = max(1, int(self.config.fixed_position_value / (option_price * 100)))

        # Create order
        order = Order(
            instrument=option,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            time_in_force=TimeInForce.DAY,
        )

        # Submit order
        self.submit_order(order)
        self.current_option_symbol = option.symbol
        self.entry_price = option_price
        self.entry_bar_datetime = bar.timestamp

        self.log.info(f"ENTRY: {option.symbol} @ ${option_price:.2f} x {quantity}", LogColor.GREEN)

    def _find_atm_option(self, current_price: float, is_call: bool, bar: Bar) -> Optional[OptionContract]:
        """Find the at-the-money option."""
        # Get bar date for expiry calculation
        bar_dt = bar.timestamp
        if bar_dt.tzinfo is None:
            bar_dt = pytz.utc.localize(bar_dt)
        bar_est = bar_dt.astimezone(self.est_tz)
        current_date = bar_est.date()

        # Find this week's Friday
        weekday = current_date.weekday()
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = (4 - weekday) % 7
        expiry_date = current_date + timedelta(days=days_to_friday)

        # Round price to nearest strike (MARA uses $0.50 increments)
        strike = round(current_price * 2) / 2  # Round to nearest 0.50

        # Format option symbol
        expiry_str = expiry_date.strftime('%y%m%d')
        type_char = 'C' if is_call else 'P'
        option_symbol = f"MARA{expiry_str}{type_char}{int(strike*1000):08d}"

        # Check if option exists in engine
        instrument = self._engine.instrument(option_symbol)
        if instrument and isinstance(instrument, OptionContract):
            return instrument

        return None

    def _manage_position(self, bar: Bar):
        """Manage existing position - check SL/TP."""
        if not self.current_option_symbol or not self.entry_price:
            return

        option_bar = self._engine.current_bars.get(self.current_option_symbol)
        if not option_bar:
            return

        current_price = option_bar.close
        pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100

        # Check stop loss
        if pnl_pct <= -float(self.config.stop_loss_pct):
            self._exit_position("STOP_LOSS", bar)
            return

        # Check take profit
        if pnl_pct >= float(self.config.target_profit_pct):
            self._exit_position("TAKE_PROFIT", bar)
            return

        # Check max hold time
        if self.entry_bar_datetime:
            hold_time = (bar.timestamp - self.entry_bar_datetime).total_seconds() / 60
            if hold_time >= self.config.max_hold_minutes:
                self._exit_position("TIME_EXIT", bar)
                return

    def _exit_position(self, reason: str, bar: Bar):
        """Exit current position."""
        if not self.current_position or not self.current_option_symbol:
            return

        option_bar = self._engine.current_bars.get(self.current_option_symbol)
        exit_price = option_bar.close if option_bar else self.entry_price

        # Calculate P&L
        pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100 if self.entry_price else 0

        # Create exit order
        order = Order(
            symbol=self.current_option_symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=self.current_position.quantity,
            time_in_force=TimeInForce.DAY,
        )

        self.submit_order(order)

        self.log.info(f"EXIT: {self.current_option_symbol} @ ${exit_price:.2f} ({reason}) P&L: {pnl_pct:.1f}%", LogColor.YELLOW)

        # Update tracking
        self.daily_pnl += pnl_pct
        self.daily_trades_count += 1
        self.current_position = None
        self.current_option_symbol = None
        self.entry_price = None
        self.entry_bar_datetime = None

    def on_fill(self, event: FillEvent):
        """Handle fill events."""
        if event.side == OrderSide.BUY:
            self.current_position = Position(
                symbol=event.symbol,
                quantity=event.quantity,
                avg_price=event.price,
            )
        else:
            self.current_position = None


class MARADaily0DTEMomentum:
    """
    MARA Daily 0DTE Momentum Strategy - Signal Calculator

    This class provides standalone methods for signal calculation
    that can be used by any trading engine.
    """

    @staticmethod
    def calculate_signal_from_bars(bars: list, config: MARADaily0DTEMomentumConfig = None) -> dict:
        """
        Calculate trading signal from raw bar data.

        Parameters
        ----------
        bars : list
            List of bar objects with: close, high, low, volume attributes
        config : MARADaily0DTEMomentumConfig, optional
            Strategy configuration

        Returns
        -------
        dict
            {
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'bullish_score': int,
                'bearish_score': int,
                'confidence': str,  # 'HIGH' | 'MEDIUM' | 'LOW'
                'signals': list,    # List of signal reasons
                'indicators': dict  # All indicator values
            }
        """
        if not bars or len(bars) < 30:
            return {
                'signal': 'NEUTRAL',
                'bullish_score': 0,
                'bearish_score': 0,
                'confidence': 'LOW',
                'signals': ['Not enough data'],
                'indicators': {}
            }

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [getattr(b, 'volume', 0) for b in bars]
        current_price = closes[-1]

        # Calculate indicators
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

        def calc_bb(prices, period=20, std_dev=2.0):
            if len(prices) < period:
                return None, None, None
            recent = prices[-period:]
            middle = sum(recent) / period
            variance = sum((p - middle) ** 2 for p in recent) / period
            std = variance ** 0.5
            return middle - std_dev * std, middle, middle + std_dev * std

        def calc_vwap(bars_list):
            total_vol = sum(getattr(b, 'volume', 0) for b in bars_list)
            if total_vol == 0:
                return bars_list[-1].close
            total_vwap = sum(((b.high + b.low + b.close) / 3) * getattr(b, 'volume', 0) for b in bars_list)
            return total_vwap / total_vol

        def calc_atr(bars_list, period=14):
            if len(bars_list) < 2:
                return 0
            trs = []
            for i in range(1, len(bars_list)):
                tr = max(
                    bars_list[i].high - bars_list[i].low,
                    abs(bars_list[i].high - bars_list[i-1].close),
                    abs(bars_list[i].low - bars_list[i-1].close)
                )
                trs.append(tr)
            return sum(trs[-period:]) / min(period, len(trs))

        # Calculate all indicators
        ema_9 = calc_ema(closes, 9)
        ema_20 = calc_ema(closes, 20)
        ema_50 = calc_ema(closes, min(50, len(closes)))
        rsi = calc_rsi(closes, 14)
        bb_lower, bb_middle, bb_upper = calc_bb(closes, 20, 2.0)
        vwap = calc_vwap(bars)
        atr = calc_atr(bars, 14)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        # MACD
        ema_12 = calc_ema(closes, 12)
        ema_26 = calc_ema(closes, 26)
        macd_line = ema_12 - ema_26
        macd_signal = macd_line * 0.9  # Approximation
        macd_hist = macd_line - macd_signal

        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / max(1, len(volumes))
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # ========== SCORING SYSTEM ==========
        bullish_score = 0
        bearish_score = 0
        bullish_signals = []
        bearish_signals = []

        # 1. EMA STACK (up to 4 points)
        if current_price > ema_9 > ema_20:
            bullish_score += 4
            bullish_signals.append("EMA Stack Bullish")
        elif current_price < ema_9 < ema_20:
            bearish_score += 4
            bearish_signals.append("EMA Stack Bearish")
        elif current_price > ema_20:
            bullish_score += 2
            bullish_signals.append("Above EMA20")
        elif current_price < ema_20:
            bearish_score += 2
            bearish_signals.append("Below EMA20")

        # 2. VWAP (up to 3 points)
        vwap_dist = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
        if current_price > vwap:
            bullish_score += 2
            bullish_signals.append(f"Above VWAP (+{vwap_dist:.2f}%)")
            if abs(vwap_dist) < 0.3:
                bullish_score += 1
                bullish_signals.append("Near VWAP support")
        else:
            bearish_score += 2
            bearish_signals.append(f"Below VWAP ({vwap_dist:.2f}%)")
            if abs(vwap_dist) < 0.3:
                bearish_score += 1
                bearish_signals.append("Near VWAP resistance")

        # 3. RSI (up to 3 points)
        if 30 <= rsi <= 50:
            bullish_score += 3
            bullish_signals.append(f"RSI bullish zone ({rsi:.1f})")
        elif 50 < rsi <= 65:
            bullish_score += 2
            bullish_signals.append(f"RSI momentum ({rsi:.1f})")
        elif rsi > 70:
            bearish_score += 3
            bearish_signals.append(f"RSI OVERBOUGHT ({rsi:.1f})")
        elif rsi < 30:
            bullish_score += 3
            bullish_signals.append(f"RSI OVERSOLD ({rsi:.1f})")
        elif 50 <= rsi <= 70:
            bearish_score += 2
            bearish_signals.append(f"RSI neutral-high ({rsi:.1f})")

        # 4. MACD (up to 3 points)
        if macd_hist > 0 and macd_line > macd_signal:
            bullish_score += 3
            bullish_signals.append("MACD Bullish")
        elif macd_hist < 0 and macd_line < macd_signal:
            bearish_score += 3
            bearish_signals.append("MACD Bearish")

        # 5. Bollinger Bands (up to 2 points)
        if bb_middle:
            if current_price > bb_middle:
                bullish_score += 2
                bullish_signals.append("Above BB Middle")
            else:
                bearish_score += 2
                bearish_signals.append("Below BB Middle")

        # 6. Volume (up to 2 points)
        if volume_ratio > 1.5:
            price_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
            if price_change > 0:
                bullish_score += 2
                bullish_signals.append(f"High volume bullish ({volume_ratio:.1f}x)")
            else:
                bearish_score += 2
                bearish_signals.append(f"High volume bearish ({volume_ratio:.1f}x)")

        # Volatility adjustment (MARA is more volatile, so less adjustment)
        if atr_pct > 3.0:  # Higher threshold for MARA
            bullish_score = int(bullish_score * 0.8)
            bearish_score = int(bearish_score * 0.8)

        # ========== DECISION ==========
        MIN_SCORE = 8
        MIN_LEAD = 4

        if bullish_score >= MIN_SCORE and bullish_score >= bearish_score + MIN_LEAD:
            signal = 'BULLISH'
            confidence = 'HIGH'
        elif bearish_score >= MIN_SCORE and bearish_score >= bullish_score + MIN_LEAD:
            signal = 'BEARISH'
            confidence = 'HIGH'
        elif bullish_score >= 6 or bearish_score >= 6:
            signal = 'BULLISH' if bullish_score >= bearish_score else 'BEARISH'
            confidence = 'MEDIUM'
        else:
            signal = 'NEUTRAL'
            confidence = 'LOW'

        return {
            'signal': signal,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'indicators': {
                'price': current_price,
                'ema_9': ema_9,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_upper': bb_upper,
                'vwap': vwap,
                'atr': atr,
                'atr_pct': atr_pct,
                'volume_ratio': volume_ratio,
            }
        }

    @staticmethod
    def calculate_price_action_signal(bars: list) -> dict:
        """
        Calculate trading signal based on PRICE ACTION patterns.

        This is METHOD 2 for dual-confirmation signal validation.

        Parameters
        ----------
        bars : list
            List of bar objects with: open, close, high, low attributes

        Returns
        -------
        dict
            {
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'strength': str,  # 'STRONG' | 'MODERATE' | 'WEAK'
                'bullish_points': int,
                'bearish_points': int,
                'reasons': list
            }
        """
        if not bars or len(bars) < 10:
            return {
                'signal': 'NEUTRAL',
                'strength': 'WEAK',
                'bullish_points': 0,
                'bearish_points': 0,
                'reasons': ['Not enough bars for price action analysis']
            }

        # Use last 10 bars for price action analysis
        recent_bars = bars[-10:]
        closes = [b.close for b in recent_bars]
        opens = [b.open for b in recent_bars]
        highs = [b.high for b in recent_bars]
        lows = [b.low for b in recent_bars]

        bullish_points = 0
        bearish_points = 0
        reasons = []

        # ========== 1. CANDLE COLOR COUNT (last 5 bars) ==========
        last_5 = recent_bars[-5:]
        green_candles = sum(1 for b in last_5 if b.close > b.open)
        red_candles = 5 - green_candles

        if green_candles >= 4:
            bullish_points += 3
            reasons.append(f"{green_candles}/5 bars bullish (green)")
        elif green_candles >= 3:
            bullish_points += 2
            reasons.append(f"{green_candles}/5 bars bullish")
        elif red_candles >= 4:
            bearish_points += 3
            reasons.append(f"{red_candles}/5 bars bearish (red)")
        elif red_candles >= 3:
            bearish_points += 2
            reasons.append(f"{red_candles}/5 bars bearish")

        # ========== 2. HIGHER HIGHS / LOWER LOWS (last 5 bars) ==========
        higher_highs = sum(1 for i in range(1, 5) if highs[-i] > highs[-i-1])
        lower_lows = sum(1 for i in range(1, 5) if lows[-i] < lows[-i-1])
        higher_lows = sum(1 for i in range(1, 5) if lows[-i] > lows[-i-1])
        lower_highs = sum(1 for i in range(1, 5) if highs[-i] < highs[-i-1])

        # Bullish: Higher highs AND higher lows (uptrend)
        if higher_highs >= 3 and higher_lows >= 2:
            bullish_points += 3
            reasons.append(f"Uptrend: {higher_highs} higher highs, {higher_lows} higher lows")
        elif higher_highs >= 2:
            bullish_points += 2
            reasons.append(f"Higher highs pattern ({higher_highs}/4)")

        # Bearish: Lower lows AND lower highs (downtrend)
        if lower_lows >= 3 and lower_highs >= 2:
            bearish_points += 3
            reasons.append(f"Downtrend: {lower_lows} lower lows, {lower_highs} lower highs")
        elif lower_lows >= 2:
            bearish_points += 2
            reasons.append(f"Lower lows pattern ({lower_lows}/4)")

        # ========== 3. PRICE vs 5-BAR AVERAGE ==========
        avg_5 = sum(closes[-5:]) / 5
        current_price = closes[-1]
        price_vs_avg = ((current_price - avg_5) / avg_5) * 100

        if price_vs_avg > 0.1:
            bullish_points += 2
            reasons.append(f"Price above 5-bar avg (+{price_vs_avg:.2f}%)")
        elif price_vs_avg < -0.1:
            bearish_points += 2
            reasons.append(f"Price below 5-bar avg ({price_vs_avg:.2f}%)")

        # ========== 4. MOMENTUM (5-bar change) ==========
        momentum_5 = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0

        if momentum_5 > 0.15:
            bullish_points += 2
            reasons.append(f"Strong 5-bar momentum (+{momentum_5:.2f}%)")
        elif momentum_5 > 0.05:
            bullish_points += 1
            reasons.append(f"Positive momentum (+{momentum_5:.2f}%)")
        elif momentum_5 < -0.15:
            bearish_points += 2
            reasons.append(f"Strong bearish momentum ({momentum_5:.2f}%)")
        elif momentum_5 < -0.05:
            bearish_points += 1
            reasons.append(f"Negative momentum ({momentum_5:.2f}%)")

        # ========== 5. LAST BAR ANALYSIS ==========
        last_bar = recent_bars[-1]
        bar_range = last_bar.high - last_bar.low
        bar_body = abs(last_bar.close - last_bar.open)
        body_ratio = bar_body / bar_range if bar_range > 0 else 0

        # Strong bullish bar: closes near high with good body
        if last_bar.close > last_bar.open and body_ratio > 0.6:
            upper_wick = last_bar.high - last_bar.close
            if upper_wick < bar_body * 0.3:  # Small upper wick
                bullish_points += 2
                reasons.append("Strong bullish last bar (closes near high)")

        # Strong bearish bar: closes near low with good body
        if last_bar.close < last_bar.open and body_ratio > 0.6:
            lower_wick = last_bar.close - last_bar.low
            if lower_wick < bar_body * 0.3:  # Small lower wick
                bearish_points += 2
                reasons.append("Strong bearish last bar (closes near low)")

        # ========== DECISION ==========
        if bullish_points >= 5 and bullish_points >= bearish_points + 3:
            signal = 'BULLISH'
            strength = 'STRONG'
        elif bearish_points >= 5 and bearish_points >= bullish_points + 3:
            signal = 'BEARISH'
            strength = 'STRONG'
        elif bullish_points >= 4 and bullish_points > bearish_points:
            signal = 'BULLISH'
            strength = 'MODERATE'
        elif bearish_points >= 4 and bearish_points > bullish_points:
            signal = 'BEARISH'
            strength = 'MODERATE'
        else:
            signal = 'NEUTRAL'
            strength = 'WEAK'

        return {
            'signal': signal,
            'strength': strength,
            'bullish_points': bullish_points,
            'bearish_points': bearish_points,
            'reasons': reasons
        }

    @staticmethod
    def check_pullback_entry(bars: list, direction: str) -> dict:
        """
        Check if conditions are right for entry based on pullback analysis.

        This is the PULLBACK DETECTION layer that waits for better entry points.
        - For BULLISH signals: Wait for a dip/pullback before buying CALL
        - For BEARISH signals: Wait for a bounce/rally before buying PUT

        Parameters
        ----------
        bars : list
            List of bar objects with: open, close, high, low attributes (1-min bars)
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
        if not bars or len(bars) < 20:
            return {
                'ready_to_enter': False,
                'pullback_detected': False,
                'recovery_detected': False,
                'pullback_score': 0,
                'recovery_score': 0,
                'reasons': ['Not enough data for pullback analysis'],
                'indicators': {}
            }

        closes = [b.close for b in bars]
        opens = [b.open for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
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
            total_vol = sum(getattr(b, 'volume', 1) for b in bars_list)
            if total_vol == 0:
                return bars_list[-1].close
            total_vwap = sum(((b.high + b.low + b.close) / 3) * getattr(b, 'volume', 1) for b in bars_list)
            return total_vwap / total_vol

        ema_9 = calc_ema(closes, 9)
        ema_20 = calc_ema(closes, 20)
        rsi = calc_rsi(closes, 14)
        vwap = calc_vwap(bars)

        # Recent extremes
        recent_high = max(highs[-20:])
        recent_low = min(lows[-10:])
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

        last_bar = bars[-1]
        last_5_bars = bars[-5:]

        if direction == 'BULLISH':
            # For BULLISH: We want to see a pullback (dip) followed by recovery

            # ========== PULLBACK DETECTION (for calls - price pulled back) ==========
            if rsi < 40:
                pullback_score += 3
                reasons.append(f"RSI oversold zone ({rsi:.1f})")
            elif rsi < 50:
                pullback_score += 1
                reasons.append(f"RSI neutral-low ({rsi:.1f})")

            if current_price < vwap:
                pullback_score += 2
                reasons.append(f"Below VWAP (${current_price:.2f} < ${vwap:.2f})")

            if pullback_from_high >= 0.3:
                pullback_score += 2
                reasons.append(f"Pulled back {pullback_from_high:.2f}% from high")

            red_count = sum(1 for b in last_5_bars if b.close < b.open)
            if red_count >= 3:
                pullback_score += 2
                reasons.append(f"{red_count}/5 red candles (selling)")
            elif red_count >= 2:
                pullback_score += 1
                reasons.append(f"{red_count}/5 red candles")

            # ========== RECOVERY DETECTION (for calls - turning bullish) ==========
            if last_bar.close > last_bar.open:
                recovery_score += 3
                bar_change = ((last_bar.close - last_bar.open) / last_bar.open) * 100
                reasons.append(f"Last bar GREEN (+{bar_change:.2f}%) - reversal signal")
            else:
                reasons.append("Waiting for green candle...")

            if bounce_from_low >= 0.05:
                recovery_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")

            if rsi > 35:
                recovery_score += 1
                reasons.append(f"RSI recovering ({rsi:.1f})")

            if len(bars) >= 3:
                momentum_3 = ((closes[-1] - closes[-3]) / closes[-3]) * 100
                if momentum_3 > 0:
                    recovery_score += 2
                    reasons.append(f"3-bar momentum positive (+{momentum_3:.2f}%)")

            if len(bars) >= 2 and bars[-1].low > bars[-2].low:
                recovery_score += 1
                reasons.append("Higher low (reversal)")

            indicators['pullback_score'] = pullback_score
            indicators['recovery_score'] = recovery_score

            pullback_detected = pullback_score >= 4
            recovery_detected = recovery_score >= 4

            if rsi > 65:
                reasons.append(f"RSI too high ({rsi:.1f}) - skip entry")
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': recovery_detected,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons,
                    'indicators': indicators
                }

            if last_bar.close <= last_bar.open:
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
            # For BEARISH: We want to see a bounce (rally) followed by rejection

            # ========== BOUNCE DETECTION (for puts - price bounced up) ==========
            if rsi > 60:
                pullback_score += 3
                reasons.append(f"RSI overbought zone ({rsi:.1f})")
            elif rsi > 50:
                pullback_score += 1
                reasons.append(f"RSI neutral-high ({rsi:.1f})")

            if current_price > vwap:
                pullback_score += 2
                reasons.append(f"Above VWAP (${current_price:.2f} > ${vwap:.2f})")

            if bounce_from_low >= 0.3:
                pullback_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")

            green_count = sum(1 for b in last_5_bars if b.close > b.open)
            if green_count >= 3:
                pullback_score += 2
                reasons.append(f"{green_count}/5 green candles (buying)")
            elif green_count >= 2:
                pullback_score += 1
                reasons.append(f"{green_count}/5 green candles")

            # ========== REJECTION DETECTION (for puts - turning bearish) ==========
            if last_bar.close < last_bar.open:
                recovery_score += 3
                bar_change = ((last_bar.close - last_bar.open) / last_bar.open) * 100
                reasons.append(f"Last bar RED ({bar_change:.2f}%) - rejection signal")
            else:
                reasons.append("Waiting for red candle...")

            if pullback_from_high >= 0.05:
                recovery_score += 2
                reasons.append(f"Pulled back {pullback_from_high:.2f}% from high")

            if rsi < 65:
                recovery_score += 1
                reasons.append(f"RSI turning down ({rsi:.1f})")

            if len(bars) >= 3:
                momentum_3 = ((closes[-1] - closes[-3]) / closes[-3]) * 100
                if momentum_3 < 0:
                    recovery_score += 2
                    reasons.append(f"3-bar momentum negative ({momentum_3:.2f}%)")

            if len(bars) >= 2 and bars[-1].high < bars[-2].high:
                recovery_score += 1
                reasons.append("Lower high (rejection)")

            indicators['pullback_score'] = pullback_score
            indicators['recovery_score'] = recovery_score

            pullback_detected = pullback_score >= 4
            recovery_detected = recovery_score >= 4

            if rsi < 35:
                reasons.append(f"RSI too low ({rsi:.1f}) - skip entry")
                return {
                    'ready_to_enter': False,
                    'pullback_detected': pullback_detected,
                    'recovery_detected': recovery_detected,
                    'pullback_score': pullback_score,
                    'recovery_score': recovery_score,
                    'reasons': reasons,
                    'indicators': indicators
                }

            if last_bar.close >= last_bar.open:
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

    @staticmethod
    def check_pullback_entry_htf(bars_5min: list, direction: str) -> dict:
        """
        V4 PULLBACK DETECTION LAYER using 5-minute bars (Higher TimeFrame = less noise).

        For BULLISH signals: Wait for dip/pullback + recovery (green 5-min candle)
        For BEARISH signals: Wait for bounce/rally + rejection (red 5-min candle)

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
            if isinstance(bar, dict):
                return bar[key]
            return getattr(bar, key)

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
            total_vol = sum(getattr(b, 'volume', 1) if hasattr(b, 'volume') else b.get('volume', 1) if isinstance(b, dict) else 1 for b in bars_list)
            if total_vol == 0:
                return get_val(bars_list[-1], 'close')
            total_vwap = sum(((get_val(b, 'high') + get_val(b, 'low') + get_val(b, 'close')) / 3) *
                           (getattr(b, 'volume', 1) if hasattr(b, 'volume') else b.get('volume', 1) if isinstance(b, dict) else 1)
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
            if rsi < 45:
                pullback_score += 3
                reasons.append(f"RSI low zone ({rsi:.1f})")
            elif rsi < 55:
                pullback_score += 1
                reasons.append(f"RSI neutral ({rsi:.1f})")

            if current_price < vwap:
                pullback_score += 2
                reasons.append(f"Below VWAP (${current_price:.2f} < ${vwap:.2f})")

            if pullback_from_high >= 0.5:
                pullback_score += 2
                reasons.append(f"Pulled back {pullback_from_high:.2f}% from high")
            elif pullback_from_high >= 0.25:
                pullback_score += 1
                reasons.append(f"Small pullback {pullback_from_high:.2f}%")

            red_count = sum(1 for b in last_3_bars if get_val(b, 'close') < get_val(b, 'open'))
            if red_count >= 2:
                pullback_score += 2
                reasons.append(f"{red_count}/3 red 5-min candles (selling)")
            elif red_count >= 1:
                pullback_score += 1
                reasons.append(f"{red_count}/3 red candles")

            # ========== RECOVERY DETECTION (for calls - turning bullish) ==========
            if get_val(last_bar, 'close') > get_val(last_bar, 'open'):
                recovery_score += 3
                bar_change = ((get_val(last_bar, 'close') - get_val(last_bar, 'open')) / get_val(last_bar, 'open')) * 100
                reasons.append(f"Last 5-min bar GREEN (+{bar_change:.2f}%) - reversal")
            else:
                reasons.append("Waiting for green 5-min candle...")

            if bounce_from_low >= 0.15:
                recovery_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")

            if rsi > 40:
                recovery_score += 1
                reasons.append(f"RSI recovering ({rsi:.1f})")

            if len(bars_5min) >= 2:
                momentum_2 = ((closes[-1] - closes[-2]) / closes[-2]) * 100
                if momentum_2 > 0:
                    recovery_score += 2
                    reasons.append(f"10-min momentum positive (+{momentum_2:.2f}%)")

            if len(bars_5min) >= 2 and get_val(bars_5min[-1], 'low') > get_val(bars_5min[-2], 'low'):
                recovery_score += 1
                reasons.append("Higher low (reversal)")

            indicators['pullback_score'] = pullback_score
            indicators['recovery_score'] = recovery_score

            pullback_detected = pullback_score >= 3
            recovery_detected = recovery_score >= 4

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
            if rsi > 55:
                pullback_score += 3
                reasons.append(f"RSI high zone ({rsi:.1f})")
            elif rsi > 45:
                pullback_score += 1
                reasons.append(f"RSI neutral ({rsi:.1f})")

            if current_price > vwap:
                pullback_score += 2
                reasons.append(f"Above VWAP (${current_price:.2f} > ${vwap:.2f})")

            if bounce_from_low >= 0.5:
                pullback_score += 2
                reasons.append(f"Bounced {bounce_from_low:.2f}% from low")
            elif bounce_from_low >= 0.25:
                pullback_score += 1
                reasons.append(f"Small bounce {bounce_from_low:.2f}%")

            green_count = sum(1 for b in last_3_bars if get_val(b, 'close') > get_val(b, 'open'))
            if green_count >= 2:
                pullback_score += 2
                reasons.append(f"{green_count}/3 green 5-min candles (buying)")
            elif green_count >= 1:
                pullback_score += 1
                reasons.append(f"{green_count}/3 green candles")

            # ========== REJECTION DETECTION (for puts - turning bearish) ==========
            if get_val(last_bar, 'close') < get_val(last_bar, 'open'):
                recovery_score += 3
                bar_change = ((get_val(last_bar, 'close') - get_val(last_bar, 'open')) / get_val(last_bar, 'open')) * 100
                reasons.append(f"Last 5-min bar RED ({bar_change:.2f}%) - rejection")
            else:
                reasons.append("Waiting for red 5-min candle...")

            if pullback_from_high >= 0.15:
                recovery_score += 2
                reasons.append(f"Pulled back {pullback_from_high:.2f}% from high")

            if rsi < 60:
                recovery_score += 1
                reasons.append(f"RSI turning down ({rsi:.1f})")

            if len(bars_5min) >= 2:
                momentum_2 = ((closes[-1] - closes[-2]) / closes[-2]) * 100
                if momentum_2 < 0:
                    recovery_score += 2
                    reasons.append(f"10-min momentum negative ({momentum_2:.2f}%)")

            if len(bars_5min) >= 2 and get_val(bars_5min[-1], 'high') < get_val(bars_5min[-2], 'high'):
                recovery_score += 1
                reasons.append("Lower high (rejection)")

            indicators['pullback_score'] = pullback_score
            indicators['recovery_score'] = recovery_score

            pullback_detected = pullback_score >= 3
            recovery_detected = recovery_score >= 4

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
