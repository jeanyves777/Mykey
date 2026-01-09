"""
Paper Trading Engine for 0DTE Options

Real-time paper trading execution engine with COMPREHENSIVE LOGGING that:
- Connects to Alpaca for live market data
- Executes trades using the paper trading API
- Manages positions with SL/TP orders
- Follows the COIN 0DTE momentum strategy
- Shows all market data and decisions in real-time
"""

import os
import signal
import sys
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import pytz
import threading

from ..config import PaperTradingConfig
from .alpaca_client import AlpacaClient, Quote, Bar, ALPACA_AVAILABLE
from ..strategies import COINDaily0DTEMomentum, COINDaily0DTEMomentumConfig
from ..analytics.options_trade_logger import OptionsTradeLogger, get_options_trade_logger
from ..data.options_market_collector import OptionsMarketCollector, get_options_market_collector


EST = pytz.timezone('America/New_York')


@dataclass
class PaperPosition:
    """Tracks an open paper trading position."""
    symbol: str  # OCC option symbol
    underlying: str
    qty: int
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiration: datetime
    signal: str  # 'BULLISH' or 'BEARISH'

    # SL/TP tracking
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    highest_price_since_entry: float = 0.0

    # Order IDs
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""


@dataclass
class TradingSession:
    """Tracks current trading session state."""
    date: datetime = field(default_factory=lambda: datetime.now(EST).date())
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    pnl_today: float = 0.0
    position: Optional[PaperPosition] = None
    has_traded_today: bool = False


class PaperTradingEngine:
    """
    Real-time paper trading execution engine with comprehensive logging.
    """

    def __init__(self, config: PaperTradingConfig):
        """Initialize paper trading engine."""
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Run: pip install alpaca-py")

        self.config = config
        self.client = AlpacaClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            paper=True
        )

        # Session state
        self.session = TradingSession()
        self.running = False
        self._stop_event = threading.Event()
        self._status_counter = 0

        # Market data
        self.latest_underlying_quote: Optional[Quote] = None
        self.latest_underlying_bar: Optional[Bar] = None
        self.latest_option_quote: Optional[Quote] = None

        # Strategy config - Load all parameters from config file
        self.strategy_config = COINDaily0DTEMomentumConfig(
            underlying_symbol=config.underlying_symbol,
            fixed_position_value=config.fixed_position_value,
            target_profit_pct=Decimal(str(config.target_profit_pct)),
            stop_loss_pct=Decimal(str(config.stop_loss_pct)),
            # Trailing stop settings
            trailing_stop_enabled=getattr(config, 'trailing_stop_enabled', True),
            trailing_trigger_pct=Decimal(str(getattr(config, 'trailing_trigger_pct', 10.0))),
            trailing_distance_pct=Decimal(str(getattr(config, 'trailing_distance_pct', 15.0))),
            # Hold time settings
            min_hold_minutes=getattr(config, 'min_hold_minutes', 5),
            max_hold_minutes=config.max_hold_minutes,
            entry_time_start=config.entry_time_start,
            entry_time_end=config.entry_time_end,
            force_exit_time=config.force_exit_time,
            # Technical indicator settings
            fast_ema_period=getattr(config, 'fast_ema_period', 9),
            slow_ema_period=getattr(config, 'slow_ema_period', 20),
            rsi_period=getattr(config, 'rsi_period', 14),
            macd_fast_period=getattr(config, 'macd_fast_period', 12),
            macd_slow_period=getattr(config, 'macd_slow_period', 26),
            macd_signal_period=getattr(config, 'macd_signal_period', 9),
            bb_period=getattr(config, 'bb_period', 20),
            bb_std_dev=getattr(config, 'bb_std_dev', 2.0),
            # Option filtering settings
            min_volume_ratio=getattr(config, 'min_volume_ratio', 1.0),
            max_bid_ask_spread_pct=getattr(config, 'max_bid_ask_spread_pct', 30.0),
            min_option_premium=getattr(config, 'min_option_premium', 2.0),
            max_option_premium=getattr(config, 'max_option_premium', 30.0),
            max_trades_per_day=getattr(config, 'max_trades_per_day', 1),
        )

        # Parse trading times
        self.entry_start = datetime.strptime(config.entry_time_start, "%H:%M:%S").time()
        self.entry_end = datetime.strptime(config.entry_time_end, "%H:%M:%S").time()
        self.force_exit = datetime.strptime(config.force_exit_time, "%H:%M:%S").time()

        # Max trades per day from strategy config
        self.max_trades_per_day = self.strategy_config.max_trades_per_day

        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_quote: Optional[Callable] = None

        # Trade logging
        self.trade_logger: OptionsTradeLogger = get_options_trade_logger()
        self.market_collector: OptionsMarketCollector = get_options_market_collector()

    def _log(self, msg: str, level: str = "INFO"):
        """Log message with timestamp and color."""
        now = datetime.now(EST)
        color = ""
        reset = ""

        # Enable colors if possible
        if sys.platform != 'win32' or 'TERM' in os.environ:
            if level == "ERROR":
                color = "\033[91m"  # Red
            elif level == "WARN":
                color = "\033[93m"  # Yellow
            elif level == "SUCCESS":
                color = "\033[92m"  # Green
            elif level == "TRADE":
                color = "\033[96m"  # Cyan
            reset = "\033[0m"

        print(f"{color}[{now.strftime('%H:%M:%S')}] [{level}] {msg}{reset}")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EST)
        if now.weekday() >= 5:  # Weekend
            return False
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= now.time() <= market_close

    def _is_entry_window(self) -> bool:
        """Check if within entry time window."""
        now = datetime.now(EST).time()
        return self.entry_start <= now <= self.entry_end

    def _should_force_exit(self) -> bool:
        """Check if should force exit position."""
        now = datetime.now(EST).time()
        return now >= self.force_exit

    def _get_this_weeks_friday(self) -> datetime:
        """
        Get the optimal Friday expiry to avoid rapid theta decay.

        Strategy:
        - Monday/Tuesday (days 0-1): Use THIS week's Friday (3-4 DTE)
        - Wednesday/Thursday/Friday (days 2-4): Use NEXT week's Friday (9-11 DTE)

        This avoids the steepest theta decay in the final 48-72 hours.
        """
        now = datetime.now(EST)
        weekday = now.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri

        # Calculate this week's Friday
        if weekday <= 4:
            days_to_this_friday = 4 - weekday
        else:
            days_to_this_friday = (4 - weekday) % 7

        # SMART EXPIRY SELECTION:
        # Mon-Tue: Use this week's Friday
        # Wed-Fri: Use next week's Friday (to avoid rapid theta decay)
        if weekday <= 1:  # Monday or Tuesday
            target_friday = now + timedelta(days=days_to_this_friday)
            self._log(f"  Using THIS week's Friday expiry (Mon/Tue rule)", "INFO")
        else:  # Wednesday, Thursday, or Friday
            target_friday = now + timedelta(days=days_to_this_friday + 7)
            self._log(f"  Using NEXT week's Friday expiry (Wed-Fri rule)", "INFO")

        return target_friday.replace(hour=16, minute=0, second=0, microsecond=0)

    def _recover_existing_positions(self):
        """Check for existing positions in Alpaca on startup and recover them.

        Uses multiple methods to detect positions:
        1. Direct positions API (with retries)
        2. Open SELL orders as backup (if TP order exists, position must exist)
        """
        self._log("Checking for existing positions in Alpaca...", "INFO")

        import time

        # METHOD 1: Try positions API with retries (API can be slow)
        alpaca_positions = []
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                alpaca_positions = self.client.get_options_positions()
                if alpaca_positions:
                    self._log(f"Positions API returned {len(alpaca_positions)} position(s)", "INFO")
                    break
                else:
                    self._log(f"Positions API empty (attempt {attempt}/{max_retries})", "WARN")
                    if attempt < max_retries:
                        time.sleep(2)  # Wait before retry
            except Exception as e:
                self._log(f"Positions API error (attempt {attempt}/{max_retries}): {e}", "WARN")
                if attempt < max_retries:
                    time.sleep(2)

        # METHOD 2: If positions API is empty, check open SELL orders as backup
        # If there's a TP SELL order for COIN options, we have a position
        if not alpaca_positions:
            self._log("Positions API empty - checking open orders as backup...", "INFO")
            try:
                # Use filtered method to get only orders for our underlying
                open_orders = self.client.get_open_orders(self.config.underlying_symbol)
                self._log(f"Found {len(open_orders)} open orders for {self.config.underlying_symbol}", "INFO")
                for order in open_orders:
                    symbol = order.get('symbol', '')
                    side = order.get('side', '')
                    # Look for SELL orders (these are TP orders)
                    if side == 'sell':
                        self._log(f"FOUND TP ORDER (position must exist): {symbol}", "WARN")
                        self._log(f"  Order ID: {order.get('id')}", "WARN")
                        self._log(f"  Limit Price: ${float(order.get('limit_price', 0)):.2f}", "WARN")
                        self._log(f"  Qty: {order.get('qty')}", "WARN")

                        # Recover position from order info
                        qty = int(float(order.get('qty', 0)))
                        tp_price = float(order.get('limit_price', 0))
                        # Estimate entry price from TP price (TP = entry * 1.075)
                        estimated_entry = tp_price / (1 + self.config.target_profit_pct / 100)

                        # Parse OCC symbol
                        try:
                            option_type = 'CALL' if 'C' in symbol[6:13] else 'PUT'
                            strike_str = symbol[-8:]
                            strike = float(strike_str) / 1000
                            expiry_str = symbol[4:10]
                            expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
                        except Exception:
                            option_type = 'CALL'
                            strike = 0
                            expiry = self._get_this_weeks_friday()

                        # Create position object
                        self.session.position = PaperPosition(
                            symbol=symbol,
                            underlying=self.config.underlying_symbol,
                            qty=qty,
                            side='long',
                            entry_price=estimated_entry,
                            entry_time=datetime.now(EST),
                            option_type=option_type,
                            strike=strike,
                            expiration=expiry,
                            signal='BULLISH' if option_type == 'CALL' else 'BEARISH',
                            stop_loss_price=estimated_entry * (1 - self.config.stop_loss_pct / 100),
                            take_profit_price=tp_price,
                            highest_price_since_entry=estimated_entry,
                        )
                        self.session.position.tp_order_id = order.get('id', '')

                        self._log(f"Position RECOVERED from TP order:", "SUCCESS")
                        self._log(f"  Estimated Entry: ${estimated_entry:.2f}", "SUCCESS")
                        self._log(f"  TP: ${tp_price:.2f}", "SUCCESS")
                        self._log(f"  SL: ${self.session.position.stop_loss_price:.2f}", "SUCCESS")
                        self._log(f"  TP Order ID: {order.get('id')}", "SUCCESS")

                        self.session.has_traded_today = True
                        self.session.trades_today = 1
                        return

            except Exception as e:
                self._log(f"Error checking open orders: {e}", "WARN")

            self._log("No existing positions or TP orders found", "INFO")
            return

        # Process positions found via API
        for pos in alpaca_positions:
            symbol = pos.get('symbol', '')
            if self.config.underlying_symbol in symbol:
                self._log(f"FOUND EXISTING POSITION: {symbol}", "WARN")
                self._log(f"  Qty: {pos.get('qty', 0)}", "WARN")
                self._log(f"  Entry: ${pos.get('avg_entry_price', 0):.2f}", "WARN")
                self._log(f"  Current: ${pos.get('current_price', 0):.2f}", "WARN")
                self._log(f"  P&L: ${pos.get('unrealized_pl', 0):.2f}", "WARN")

                # Recover the position
                entry_price = pos.get('avg_entry_price', 0)
                qty = int(pos.get('qty', 0))

                # Parse OCC symbol for strike and expiry
                # Format: COIN251212C00275000
                try:
                    option_type = 'CALL' if 'C' in symbol[6:13] else 'PUT'
                    strike_str = symbol[-8:]  # Last 8 chars are strike * 1000
                    strike = float(strike_str) / 1000
                    expiry_str = symbol[4:10]  # YYMMDD
                    expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
                except Exception as parse_err:
                    self._log(f"Could not parse OCC symbol: {parse_err}", "WARN")
                    option_type = 'CALL'
                    strike = 0
                    expiry = self._get_this_weeks_friday()

                # Create position object
                self.session.position = PaperPosition(
                    symbol=symbol,
                    underlying=self.config.underlying_symbol,
                    qty=qty,
                    side='long',
                    entry_price=entry_price,
                    entry_time=datetime.now(EST),  # Approximation
                    option_type=option_type,
                    strike=strike,
                    expiration=expiry,
                    signal='BULLISH' if option_type == 'CALL' else 'BEARISH',
                    stop_loss_price=entry_price * (1 - self.config.stop_loss_pct / 100),
                    take_profit_price=entry_price * (1 + self.config.target_profit_pct / 100),
                    highest_price_since_entry=pos.get('current_price', entry_price),
                )

                self._log(f"Position RECOVERED:", "SUCCESS")
                self._log(f"  TP: ${self.session.position.take_profit_price:.2f}", "SUCCESS")
                self._log(f"  SL: ${self.session.position.stop_loss_price:.2f}", "SUCCESS")

                # Mark as traded today
                self.session.has_traded_today = True
                self.session.trades_today = 1

                # Check for existing TP order
                try:
                    orders = self.client.get_open_orders()
                    for order in orders:
                        if order.get('symbol') == symbol and order.get('side') == 'sell':
                            self.session.position.tp_order_id = order.get('id', '')
                            self._log(f"  Found existing TP order: {order.get('id')}", "SUCCESS")
                            break
                except Exception as order_err:
                    self._log(f"Could not check for TP orders: {order_err}", "WARN")

                break  # Only recover first matching position

    def _calculate_signal(self) -> str:
        """
        Calculate trading signal using IMPROVED MOMENTUM-WEIGHTED ANALYSIS.

        NEW V3 APPROACH:
        - METHOD 1: Technical Scoring (EMA, VWAP, RSI, MACD, BB) - 1-MIN BARS
        - METHOD 2: Price Action (candle patterns, higher highs/lows) - 5-MIN BARS
        - METHOD 3: REAL-TIME MOMENTUM (last 3-5 bars direction) - HIGHEST WEIGHT!

        The real-time momentum can OVERRIDE conflicting signals when strong.
        This fixes the issue where lagging indicators give wrong direction.

        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        from datetime import timedelta

        # Get 1-MINUTE bars for Technical Scoring (Method 1)
        start_time = datetime.now(EST) - timedelta(hours=3)
        bars_1min = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='1Min',
            start=start_time,
            limit=180
        )

        if not bars_1min or len(bars_1min) < 30:
            bars_count = len(bars_1min) if bars_1min else 0
            self._log(f"Not enough 1-min bar data for analysis ({bars_count} bars, need 30) - SKIPPING TRADE", "WARN")
            return 'NEUTRAL'

        # Use last 60 bars for 1-min
        bars_1min = bars_1min[-60:]

        # Update the latest bar from fetched data (more accurate than get_latest_stock_bar)
        if bars_1min:
            self.latest_underlying_bar = bars_1min[-1]

        # Get 5-MINUTE bars for Price Action (Method 2)
        start_time_5min = datetime.now(EST) - timedelta(hours=6)
        bars_5min = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='5Min',
            start=start_time_5min,
            limit=100
        )

        if not bars_5min or len(bars_5min) < 12:
            bars_count = len(bars_5min) if bars_5min else 0
            self._log(f"Not enough 5-min bar data for Price Action ({bars_count} bars, need 12) - SKIPPING TRADE", "WARN")
            return 'NEUTRAL'

        # Use last 30 bars for 5-min (2.5 hours of data)
        bars_5min = bars_5min[-30:]

        # ============================================================
        # V3: IMPROVED SIGNAL ANALYSIS WITH REAL-TIME MOMENTUM
        # ============================================================
        print()
        self._log("=" * 65, "INFO")
        self._log("  V3 MOMENTUM-WEIGHTED SIGNAL ANALYSIS", "INFO")
        self._log("=" * 65, "INFO")

        # ========== METHOD 1: Technical Scoring (1-MIN BARS) ==========
        tech_result = COINDaily0DTEMomentum.calculate_signal_from_bars(bars_1min)
        tech_signal = tech_result['signal']
        tech_confidence = tech_result['confidence']
        tech_bull_score = tech_result['bullish_score']
        tech_bear_score = tech_result['bearish_score']
        indicators = tech_result.get('indicators', {})

        self._log("  METHOD 1 - Technical Scoring (1-MIN):", "INFO")
        if indicators:
            self._log(f"    Price: ${indicators.get('price', 0):.2f} | VWAP: ${indicators.get('vwap', 0):.2f}", "INFO")
            self._log(f"    EMA9: ${indicators.get('ema_9', 0):.2f} | EMA20: ${indicators.get('ema_20', 0):.2f}", "INFO")
            self._log(f"    RSI: {indicators.get('rsi', 0):.1f} | MACD: {indicators.get('macd_line', 0):.3f}", "INFO")

        self._log(f"    Signal: {tech_signal} | Score: {tech_bull_score}/17 vs {tech_bear_score}/17 | Confidence: {tech_confidence}", "INFO")

        for sig in tech_result.get('bullish_signals', []):
            self._log(f"      + {sig}", "SUCCESS")
        for sig in tech_result.get('bearish_signals', []):
            self._log(f"      - {sig}", "WARN")

        self._log("-" * 65, "INFO")

        # ========== METHOD 2: Price Action (5-MIN BARS) ==========
        pa_result = COINDaily0DTEMomentum.calculate_price_action_signal(bars_5min)
        pa_signal = pa_result['signal']
        pa_strength = pa_result['strength']
        pa_bull_points = pa_result['bullish_points']
        pa_bear_points = pa_result['bearish_points']

        self._log("  METHOD 2 - Price Action (5-MIN):", "INFO")
        self._log(f"    Signal: {pa_signal} | Strength: {pa_strength} | Points: {pa_bull_points} bull vs {pa_bear_points} bear", "INFO")

        for reason in pa_result.get('reasons', []):
            if 'bullish' in reason.lower() or 'green' in reason.lower() or 'higher' in reason.lower() or 'above' in reason.lower() or 'uptrend' in reason.lower() or '+' in reason:
                self._log(f"      + {reason}", "SUCCESS")
            elif 'bearish' in reason.lower() or 'red' in reason.lower() or 'lower' in reason.lower() or 'below' in reason.lower() or 'downtrend' in reason.lower():
                self._log(f"      - {reason}", "WARN")
            else:
                self._log(f"      * {reason}", "INFO")

        self._log("-" * 65, "INFO")

        # ========== METHOD 3: REAL-TIME MOMENTUM (LAST 5 1-MIN BARS) ==========
        # This is the KEY FIX: Use the most recent price action to determine direction
        # Lagging indicators can be wrong, but recent bars show what's ACTUALLY happening NOW
        self._log("  METHOD 3 - REAL-TIME MOMENTUM (Last 5 bars):", "INFO")

        last_5_bars = bars_1min[-5:]
        momentum_bull_points = 0
        momentum_bear_points = 0
        momentum_reasons = []

        # 1. Count green vs red candles (last 5 bars)
        green_count = sum(1 for b in last_5_bars if b.close > b.open)
        red_count = 5 - green_count

        if green_count >= 4:
            momentum_bull_points += 4
            momentum_reasons.append(f"{green_count}/5 bars GREEN (strong bullish)")
        elif green_count >= 3:
            momentum_bull_points += 2
            momentum_reasons.append(f"{green_count}/5 bars green")
        elif red_count >= 4:
            momentum_bear_points += 4
            momentum_reasons.append(f"{red_count}/5 bars RED (strong bearish)")
        elif red_count >= 3:
            momentum_bear_points += 2
            momentum_reasons.append(f"{red_count}/5 bars red")

        # 2. Price change over last 5 bars
        price_change_5 = ((last_5_bars[-1].close - last_5_bars[0].open) / last_5_bars[0].open) * 100
        if price_change_5 > 0.2:
            momentum_bull_points += 3
            momentum_reasons.append(f"5-bar momentum +{price_change_5:.2f}% (bullish)")
        elif price_change_5 > 0.1:
            momentum_bull_points += 2
            momentum_reasons.append(f"5-bar momentum +{price_change_5:.2f}%")
        elif price_change_5 < -0.2:
            momentum_bear_points += 3
            momentum_reasons.append(f"5-bar momentum {price_change_5:.2f}% (bearish)")
        elif price_change_5 < -0.1:
            momentum_bear_points += 2
            momentum_reasons.append(f"5-bar momentum {price_change_5:.2f}%")

        # 3. Last bar direction (most recent = most important)
        last_bar = last_5_bars[-1]
        last_bar_change = ((last_bar.close - last_bar.open) / last_bar.open) * 100
        if last_bar.close > last_bar.open:
            momentum_bull_points += 2
            momentum_reasons.append(f"Last bar GREEN (+{last_bar_change:.2f}%)")
        elif last_bar.close < last_bar.open:
            momentum_bear_points += 2
            momentum_reasons.append(f"Last bar RED ({last_bar_change:.2f}%)")

        # 4. Higher highs / Lower lows in last 5 bars
        higher_highs = sum(1 for i in range(1, 5) if last_5_bars[i].high > last_5_bars[i-1].high)
        lower_lows = sum(1 for i in range(1, 5) if last_5_bars[i].low < last_5_bars[i-1].low)

        if higher_highs >= 3:
            momentum_bull_points += 2
            momentum_reasons.append(f"{higher_highs}/4 higher highs")
        if lower_lows >= 3:
            momentum_bear_points += 2
            momentum_reasons.append(f"{lower_lows}/4 lower lows")

        # Determine momentum signal
        if momentum_bull_points >= 6 and momentum_bull_points >= momentum_bear_points + 3:
            momentum_signal = 'BULLISH'
            momentum_strength = 'STRONG'
        elif momentum_bear_points >= 6 and momentum_bear_points >= momentum_bull_points + 3:
            momentum_signal = 'BEARISH'
            momentum_strength = 'STRONG'
        elif momentum_bull_points >= 4 and momentum_bull_points > momentum_bear_points:
            momentum_signal = 'BULLISH'
            momentum_strength = 'MODERATE'
        elif momentum_bear_points >= 4 and momentum_bear_points > momentum_bull_points:
            momentum_signal = 'BEARISH'
            momentum_strength = 'MODERATE'
        else:
            momentum_signal = 'NEUTRAL'
            momentum_strength = 'WEAK'

        self._log(f"    Signal: {momentum_signal} | Strength: {momentum_strength} | Points: {momentum_bull_points} bull vs {momentum_bear_points} bear", "INFO")
        for reason in momentum_reasons:
            if 'bull' in reason.lower() or 'green' in reason.lower() or 'higher' in reason.lower() or '+' in reason:
                self._log(f"      + {reason}", "SUCCESS")
            elif 'bear' in reason.lower() or 'red' in reason.lower() or 'lower' in reason.lower():
                self._log(f"      - {reason}", "WARN")
            else:
                self._log(f"      * {reason}", "INFO")

        self._log("-" * 65, "INFO")

        # ========== METHOD 4: HIGHER TIMEFRAME TREND FILTER (30-MIN + 1-HOUR) ==========
        # This prevents counter-trend trades (e.g., taking CALLS in a bearish market)
        self._log("  METHOD 4 - Higher Timeframe Trend Filter:", "INFO")

        # Get 30-MINUTE bars (last 6 hours = 12 bars)
        start_time_30min = datetime.now(EST) - timedelta(hours=6)
        bars_30min = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='30Min',
            start=start_time_30min,
            limit=20
        )

        # Get 1-HOUR bars (last 12 hours = 12 bars)
        start_time_1h = datetime.now(EST) - timedelta(hours=12)
        bars_1h = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='1Hour',
            start=start_time_1h,
            limit=15
        )

        htf_trend = 'NEUTRAL'
        htf_reasons = []
        htf_30min_bull = 0
        htf_30min_bear = 0
        htf_1h_bull = 0
        htf_1h_bear = 0

        # Analyze 30-MIN bars
        if bars_30min and len(bars_30min) >= 6:
            last_6_bars_30min = bars_30min[-6:]

            # 1. EMA trend on 30-min
            if len(bars_30min) >= 9:
                prices_30min = [b.close for b in bars_30min[-9:]]
                ema_9_30min = sum(prices_30min[-9:]) / 9  # Simple MA approximation
                current_price_30min = prices_30min[-1]

                if current_price_30min > ema_9_30min * 1.002:
                    htf_30min_bull += 2
                    htf_reasons.append(f"30-min: Price above EMA9 (${current_price_30min:.2f} > ${ema_9_30min:.2f})")
                elif current_price_30min < ema_9_30min * 0.998:
                    htf_30min_bear += 2
                    htf_reasons.append(f"30-min: Price below EMA9 (${current_price_30min:.2f} < ${ema_9_30min:.2f})")

            # 2. Higher highs / Lower lows on 30-min
            higher_highs_30min = sum(1 for i in range(1, 6) if last_6_bars_30min[i].high > last_6_bars_30min[i-1].high)
            lower_lows_30min = sum(1 for i in range(1, 6) if last_6_bars_30min[i].low < last_6_bars_30min[i-1].low)

            if higher_highs_30min >= 4:
                htf_30min_bull += 2
                htf_reasons.append(f"30-min: {higher_highs_30min}/5 higher highs (uptrend)")
            if lower_lows_30min >= 4:
                htf_30min_bear += 2
                htf_reasons.append(f"30-min: {lower_lows_30min}/5 lower lows (downtrend)")

            # 3. 30-min candle color pattern
            green_count_30min = sum(1 for b in last_6_bars_30min if b.close > b.open)
            if green_count_30min >= 5:
                htf_30min_bull += 2
                htf_reasons.append(f"30-min: {green_count_30min}/6 green candles")
            elif green_count_30min <= 1:
                htf_30min_bear += 2
                htf_reasons.append(f"30-min: {6-green_count_30min}/6 red candles")

        # Analyze 1-HOUR bars
        if bars_1h and len(bars_1h) >= 6:
            last_6_bars_1h = bars_1h[-6:]

            # 1. EMA trend on 1-hour
            if len(bars_1h) >= 9:
                prices_1h = [b.close for b in bars_1h[-9:]]
                ema_9_1h = sum(prices_1h[-9:]) / 9  # Simple MA approximation
                current_price_1h = prices_1h[-1]

                if current_price_1h > ema_9_1h * 1.003:
                    htf_1h_bull += 2
                    htf_reasons.append(f"1-hour: Price above EMA9 (${current_price_1h:.2f} > ${ema_9_1h:.2f})")
                elif current_price_1h < ema_9_1h * 0.997:
                    htf_1h_bear += 2
                    htf_reasons.append(f"1-hour: Price below EMA9 (${current_price_1h:.2f} < ${ema_9_1h:.2f})")

            # 2. Higher highs / Lower lows on 1-hour
            higher_highs_1h = sum(1 for i in range(1, 6) if last_6_bars_1h[i].high > last_6_bars_1h[i-1].high)
            lower_lows_1h = sum(1 for i in range(1, 6) if last_6_bars_1h[i].low < last_6_bars_1h[i-1].low)

            if higher_highs_1h >= 4:
                htf_1h_bull += 2
                htf_reasons.append(f"1-hour: {higher_highs_1h}/5 higher highs (uptrend)")
            if lower_lows_1h >= 4:
                htf_1h_bear += 2
                htf_reasons.append(f"1-hour: {lower_lows_1h}/5 lower lows (downtrend)")

            # 3. 1-hour candle color pattern
            green_count_1h = sum(1 for b in last_6_bars_1h if b.close > b.open)
            if green_count_1h >= 5:
                htf_1h_bull += 2
                htf_reasons.append(f"1-hour: {green_count_1h}/6 green candles")
            elif green_count_1h <= 1:
                htf_1h_bear += 2
                htf_reasons.append(f"1-hour: {6-green_count_1h}/6 red candles")

        # Determine HTF trend (both timeframes must agree for strong signal)
        htf_30min_signal = 'NEUTRAL'
        if htf_30min_bull >= 4 and htf_30min_bull > htf_30min_bear + 2:
            htf_30min_signal = 'BULLISH'
        elif htf_30min_bear >= 4 and htf_30min_bear > htf_30min_bull + 2:
            htf_30min_signal = 'BEARISH'

        htf_1h_signal = 'NEUTRAL'
        if htf_1h_bull >= 4 and htf_1h_bull > htf_1h_bear + 2:
            htf_1h_signal = 'BULLISH'
        elif htf_1h_bear >= 4 and htf_1h_bear > htf_1h_bull + 2:
            htf_1h_signal = 'BEARISH'

        # Final HTF trend (both timeframes should agree)
        if htf_30min_signal == htf_1h_signal and htf_30min_signal != 'NEUTRAL':
            htf_trend = htf_30min_signal
            self._log(f"    HTF Trend: {htf_trend} ✅ (BOTH 30-min and 1-hour AGREE)", "SUCCESS" if htf_trend == 'BULLISH' else "WARN")
        elif htf_30min_signal != 'NEUTRAL' and htf_1h_signal == 'NEUTRAL':
            htf_trend = htf_30min_signal
            self._log(f"    HTF Trend: {htf_trend} (30-min only, 1-hour neutral)", "INFO")
        elif htf_1h_signal != 'NEUTRAL' and htf_30min_signal == 'NEUTRAL':
            htf_trend = htf_1h_signal
            self._log(f"    HTF Trend: {htf_trend} (1-hour only, 30-min neutral)", "INFO")
        elif htf_30min_signal != htf_1h_signal:
            htf_trend = 'NEUTRAL'
            self._log(f"    HTF Trend: CONFLICTING (30-min={htf_30min_signal}, 1-hour={htf_1h_signal})", "WARN")
        else:
            htf_trend = 'NEUTRAL'
            self._log(f"    HTF Trend: NEUTRAL (no clear trend)", "INFO")

        self._log(f"    Scores: 30-min ({htf_30min_bull}b/{htf_30min_bear}r) | 1-hour ({htf_1h_bull}b/{htf_1h_bear}r)", "INFO")

        for reason in htf_reasons:
            if 'above' in reason.lower() or 'higher' in reason.lower() or 'green' in reason.lower() or 'uptrend' in reason.lower():
                self._log(f"      + {reason}", "SUCCESS")
            elif 'below' in reason.lower() or 'lower' in reason.lower() or 'red' in reason.lower() or 'downtrend' in reason.lower():
                self._log(f"      - {reason}", "WARN")
            else:
                self._log(f"      * {reason}", "INFO")

        self._log("-" * 65, "INFO")

        # ========== V3 FINAL DECISION: MOMENTUM-WEIGHTED ==========
        self._log("  V3 FINAL DECISION (Momentum-Weighted + HTF Filter):", "INFO")

        # Calculate weighted scores (momentum has highest weight)
        # Technical: 1x, Price Action: 1x, Real-Time Momentum: 2x (most important!)
        total_bull = tech_bull_score + pa_bull_points + (momentum_bull_points * 2)
        total_bear = tech_bear_score + pa_bear_points + (momentum_bear_points * 2)

        self._log(f"    Weighted Scores: BULL={total_bull} vs BEAR={total_bear}", "INFO")
        self._log(f"    (Tech: {tech_bull_score}b/{tech_bear_score}r + PA: {pa_bull_points}b/{pa_bear_points}r + Mom×2: {momentum_bull_points*2}b/{momentum_bear_points*2}r)", "INFO")

        # DECISION LOGIC V3:
        # 1. If momentum is STRONG, it can OVERRIDE conflicting weaker signals
        # 2. Otherwise, require 2 of 3 methods to agree
        # 3. With conflict and weak momentum, default to momentum direction (it's real-time!)

        final_signal = 'NEUTRAL'
        high_confidence_signal = False  # Flag for strong signals that can bypass HTF neutral

        # Case 1: All three agree - HIGH CONFIDENCE
        if tech_signal == momentum_signal == pa_signal and momentum_signal != 'NEUTRAL':
            final_signal = momentum_signal
            high_confidence_signal = True  # Can bypass HTF neutral
            self._log(f"    ✅ ALL THREE METHODS AGREE: {final_signal}", "SUCCESS")
            self._log(f"    >>> V3 SIGNAL: {'CALLS' if final_signal == 'BULLISH' else 'PUTS'} (checking pullback...)", "INFO")

        # Case 2: Strong momentum OVERRIDES conflicting signals
        elif momentum_strength == 'STRONG' and momentum_signal != 'NEUTRAL':
            final_signal = momentum_signal
            high_confidence_signal = True  # Can bypass HTF neutral
            self._log(f"    ⚡ STRONG MOMENTUM OVERRIDE: {final_signal}", "SUCCESS")
            self._log(f"    Real-time momentum is strong ({momentum_bull_points} vs {momentum_bear_points}) - overriding other signals", "SUCCESS")
            self._log(f"    >>> V3 SIGNAL: {'CALLS' if final_signal == 'BULLISH' else 'PUTS'} (checking pullback...)", "INFO")

        # Case 3: Momentum + one other method agree
        elif momentum_signal != 'NEUTRAL':
            if momentum_signal == tech_signal or momentum_signal == pa_signal:
                final_signal = momentum_signal
                agreeing_method = "Technical" if momentum_signal == tech_signal else "Price Action"
                self._log(f"    ✅ MOMENTUM + {agreeing_method.upper()} AGREE: {final_signal}", "SUCCESS")
                self._log(f"    >>> V3 SIGNAL: {'CALLS' if final_signal == 'BULLISH' else 'PUTS'} (checking pullback...)", "INFO")
            else:
                # Momentum disagrees with both - use weighted score as tiebreaker
                if total_bull > total_bear + 5:
                    final_signal = 'BULLISH'
                    self._log(f"    📊 WEIGHTED SCORE DECISION: BULLISH ({total_bull} vs {total_bear})", "SUCCESS")
                    self._log(f"    >>> V3 SIGNAL: CALLS (checking pullback...)", "INFO")
                elif total_bear > total_bull + 5:
                    final_signal = 'BEARISH'
                    self._log(f"    📊 WEIGHTED SCORE DECISION: BEARISH ({total_bear} vs {total_bull})", "WARN")
                    self._log(f"    >>> V3 SIGNAL: PUTS (checking pullback...)", "INFO")
                else:
                    final_signal = 'NEUTRAL'
                    self._log(f"    ⚠️ CONFLICTING SIGNALS - NO CLEAR EDGE", "WARN")
                    self._log(f"    >>> SKIPPING (scores too close: {total_bull} vs {total_bear})", "WARN")

        # Case 4: Weak/neutral momentum - require 2 of 3 methods to agree
        else:
            if tech_signal == pa_signal and tech_signal != 'NEUTRAL':
                final_signal = tech_signal
                self._log(f"    ✅ TECH + PA AGREE: {final_signal}", "SUCCESS")
                self._log(f"    >>> V3 SIGNAL: {'CALLS' if final_signal == 'BULLISH' else 'PUTS'} (checking pullback...)", "INFO")
            else:
                final_signal = 'NEUTRAL'
                self._log(f"    ⚠️ NO CONSENSUS - WEAK MOMENTUM & DISAGREEMENT", "WARN")
                self._log(f"    >>> SKIPPING (no clear direction)", "WARN")

        # ========== APPLY STRICT HTF TREND FILTER ==========
        # STRICT RULE: HTF must ALIGN with signal direction (neutral = not aligned)
        # This prevents counter-trend trades and whipsaw entries
        if final_signal != 'NEUTRAL':
            if htf_trend == final_signal:
                # Perfect alignment - APPROVED
                self._log("", "INFO")
                self._log(f"    ✅ HTF FILTER: Trade ALIGNED with {htf_trend} HTF trend - APPROVED", "SUCCESS")
            elif htf_trend == 'BULLISH' and final_signal == 'BEARISH':
                # Counter-trend - BLOCK
                self._log("", "INFO")
                self._log("    🛑 HTF FILTER: BLOCKING BEARISH trade in BULLISH HTF trend", "WARN")
                self._log(f"    Signal would be PUTS, but HTF is BULLISH - SKIPPING TRADE", "WARN")
                final_signal = 'NEUTRAL'
            elif htf_trend == 'BEARISH' and final_signal == 'BULLISH':
                # Counter-trend - BLOCK
                self._log("", "INFO")
                self._log("    🛑 HTF FILTER: BLOCKING BULLISH trade in BEARISH HTF trend", "WARN")
                self._log(f"    Signal would be CALLS, but HTF is BEARISH - SKIPPING TRADE", "WARN")
                final_signal = 'NEUTRAL'
            else:
                # HTF is NEUTRAL - NO CLEAR TREND
                # BUT: Allow high confidence signals (all 3 agree OR strong momentum) to bypass
                if high_confidence_signal:
                    self._log("", "INFO")
                    self._log("    ⚡ HTF FILTER: HTF is NEUTRAL, but HIGH CONFIDENCE signal detected", "SUCCESS")
                    self._log(f"    BYPASS: All methods agree OR strong momentum - ALLOWING TRADE", "SUCCESS")
                    # Keep final_signal as-is (don't block)
                else:
                    self._log("", "INFO")
                    self._log("    🛑 HTF FILTER: HTF is NEUTRAL (conflicting timeframes)", "WARN")
                    self._log(f"    STRICT MODE: Trades require HTF ALIGNMENT - SKIPPING TRADE", "WARN")
                    self._log(f"    Reason: Cannot confirm {final_signal} direction on higher timeframes", "WARN")
                    final_signal = 'NEUTRAL'

        self._log("=" * 65, "INFO")

        return final_signal

    def _calculate_ema(self, prices: list, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices)  # Simple average if not enough data

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        # Calculate average gain and loss
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: list) -> tuple:
        """Calculate MACD (12, 26, 9)."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26

        # Calculate signal line (9-period EMA of MACD)
        # For simplicity, just use the current MACD as approximation
        signal_line = self._calculate_ema(prices[-9:], 9) if len(prices) >= 9 else macd_line
        signal_line = macd_line * 0.9  # Approximate signal line

        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_vwap(self, bars: list) -> float:
        """Calculate Volume Weighted Average Price."""
        if not bars:
            return 0.0

        total_volume = 0
        total_vwap = 0

        for bar in bars:
            typical_price = (bar.high + bar.low + bar.close) / 3
            total_vwap += typical_price * bar.volume
            total_volume += bar.volume

        if total_volume == 0:
            return bars[-1].close  # Fallback to current price

        return total_vwap / total_volume

    def _calculate_atr(self, bars: list, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        if not true_ranges:
            return 0.0

        # Return average of last 'period' true ranges
        return sum(true_ranges[-period:]) / min(period, len(true_ranges))

    def _find_atm_option(self, option_type: str) -> Optional[str]:
        """Find ATM option contract for the underlying."""
        if not self.latest_underlying_quote:
            return None

        underlying_price = self.latest_underlying_quote.mid
        expiry = self._get_this_weeks_friday()
        strike = round(underlying_price / 5) * 5

        occ_symbol = self.client.format_occ_symbol(
            underlying=self.config.underlying_symbol,
            expiration=expiry,
            strike=strike,
            option_type=option_type
        )
        return occ_symbol

    def _calculate_position_size(self, option_price: float) -> int:
        """
        Calculate number of contracts based on fixed position value and available buying power.

        Ensures we don't exceed account's options buying power.
        """
        if option_price <= 0:
            return 0

        # Get current account buying power
        try:
            account = self.client.get_account()
            if isinstance(account, dict):
                options_bp = float(account.get('options_buying_power', account.get('buying_power', 0)))
            else:
                options_bp = float(getattr(account, 'options_buying_power', getattr(account, 'buying_power', 0)))
        except Exception as e:
            self._log(f"Warning: Could not get buying power ({e}), using config value", "WARN")
            options_bp = self.config.fixed_position_value

        contract_value = option_price * 100

        # Calculate based on config target
        contracts_by_target = int(self.config.fixed_position_value / contract_value)

        # Calculate based on available buying power (leave small buffer)
        contracts_by_bp = int((options_bp * 0.95) / contract_value)

        # Use the smaller of the two to ensure we don't exceed buying power
        contracts = min(contracts_by_target, contracts_by_bp)

        # Log the calculation for transparency
        self._log(f"  Position Sizing: Target={contracts_by_target}, BP-limited={contracts_by_bp}, Final={contracts}", "INFO")
        self._log(f"  Available BP: ${options_bp:.2f} | Contract cost: ${contract_value:.2f}", "INFO")

        if contracts < 1:
            self._log(f"Insufficient buying power: Need ${contract_value:.2f}, have ${options_bp:.2f}", "ERROR")
            return 0

        return contracts

    def _check_pullback_conditions(self, signal: str) -> dict:
        """
        Check if pullback conditions are met for better entry.

        This is the V4 PULLBACK DETECTION layer using 5-MIN BARS (HTF = less noise):
        - For BULLISH: Wait for dip/pullback + recovery (green 5-min candle)
        - For BEARISH: Wait for bounce/rally + rejection (red 5-min candle)

        Using 5-min bars instead of 1-min reduces noise and gives more reliable signals.

        Returns:
            dict with ready_to_enter, pullback_detected, recovery_detected, reasons
        """
        from datetime import timedelta

        # Get recent 5-min bars for pullback analysis (HTF = less noise)
        try:
            start_time = datetime.now(EST) - timedelta(hours=3)
            bars_5min = self.client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe='5Min',
                start=start_time,
                limit=40
            )

            if not bars_5min or len(bars_5min) < 10:
                self._log("  Not enough 5-min bars for pullback analysis", "WARN")
                # If we don't have enough data, allow entry to not miss opportunities
                return {
                    'ready_to_enter': True,  # Default to allowing entry
                    'pullback_detected': False,
                    'recovery_detected': False,
                    'pullback_score': 0,
                    'recovery_score': 0,
                    'reasons': ['Not enough 5-min data - allowing entry']
                }

            # Use the strategy's HTF (5-min) pullback detection method
            result = COINDaily0DTEMomentum.check_pullback_entry_htf(bars_5min, signal)

            # Log the analysis
            self._log(f"  Direction: {signal} (5-min HTF)", "INFO")
            indicators = result.get('indicators', {})
            if indicators:
                self._log(f"  Price: ${indicators.get('price', 0):.2f} | RSI: {indicators.get('rsi', 0):.1f}")
                self._log(f"  VWAP: ${indicators.get('vwap', 0):.2f} | Pullback from high: {indicators.get('pullback_from_high_pct', 0):.2f}%")

            return result

        except Exception as e:
            self._log(f"  Error in pullback check: {e}", "ERROR")
            # On error, allow entry to not miss opportunities
            return {
                'ready_to_enter': True,
                'pullback_detected': False,
                'recovery_detected': False,
                'pullback_score': 0,
                'recovery_score': 0,
                'reasons': [f'Error: {e} - allowing entry']
            }

    def _enter_position(self, signal: str):
        """Enter a new position with comprehensive logging."""
        if self.session.position is not None:
            self._log("Already in position, skipping entry", "WARN")
            return

        if self.session.trades_today >= self.max_trades_per_day:
            self._log(f"Max trades per day reached ({self.max_trades_per_day})", "INFO")
            return

        # SAFETY CHECK: Double-check Alpaca for existing positions/orders before entering
        # This prevents duplicate trades when API is slow/inconsistent
        try:
            # Check for existing options positions
            alpaca_positions = self.client.get_options_positions()
            for pos in alpaca_positions:
                if self.config.underlying_symbol in pos.get('symbol', ''):
                    self._log(f"BLOCKING ENTRY: Found existing position in Alpaca: {pos.get('symbol')}", "WARN")
                    self._log("  Recovering position instead of creating new one...", "WARN")
                    self._recover_existing_positions()
                    return

            # Check for existing SELL orders (TP orders = we have a position)
            # Use filtered method for efficiency
            open_orders = self.client.get_open_orders(self.config.underlying_symbol)
            for order in open_orders:
                if order.get('side') == 'sell':
                    self._log(f"BLOCKING ENTRY: Found existing TP order in Alpaca: {order.get('symbol')}", "WARN")
                    self._log("  Recovering position from TP order instead...", "WARN")
                    self._recover_existing_positions()
                    return
        except Exception as e:
            self._log(f"Warning: Could not verify Alpaca state before entry: {e}", "WARN")
            # Continue with entry but log warning

        option_type = 'C' if signal == 'BULLISH' else 'P'
        option_type_name = 'CALL' if signal == 'BULLISH' else 'PUT'
        occ_symbol = self._find_atm_option(option_type)

        if not occ_symbol:
            self._log("Could not find ATM option", "ERROR")
            return

        # Get option quote
        option_quote = self.client.get_latest_option_quote(occ_symbol)
        if not option_quote:
            self._log(f"Could not get quote for {occ_symbol}", "ERROR")
            return

        qty = self._calculate_position_size(option_quote.ask)
        if qty == 0:
            self._log("Position size would be 0, skipping", "WARN")
            return

        # Parse option details
        option_details = self.client.parse_occ_symbol(occ_symbol)

        print()  # New line for visibility
        self._log("=" * 70, "TRADE")
        self._log(f">>> ENTRY SIGNAL: {signal} - {option_type_name} OPTION", "TRADE")
        self._log("=" * 70, "TRADE")
        self._log(f"    Underlying: {self.config.underlying_symbol} @ ${self.latest_underlying_quote.mid:.2f}", "TRADE")
        self._log(f"    Option: {occ_symbol}", "TRADE")
        self._log(f"    Strike: ${option_details['strike']:.0f} | Expiry: {option_details['expiration']}", "TRADE")
        self._log(f"    Option Bid: ${option_quote.bid:.2f} | Ask: ${option_quote.ask:.2f} | Mid: ${option_quote.mid:.2f}", "TRADE")
        self._log(f"    Qty: {qty} contracts | Value: ${self.config.fixed_position_value:.2f}", "TRADE")
        self._log(f"    Submitting BUY order to Alpaca...", "TRADE")

        try:
            order = self.client.submit_market_order(
                symbol=occ_symbol,
                qty=qty,
                side='buy'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')

            self._log(f"    ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            # Wait for order to fill and get actual fill price
            actual_fill_price = option_quote.ask  # Default to ask price
            self._log(f"    Waiting for fill confirmation...", "TRADE")

            import time as time_module
            for wait_attempt in range(10):  # Wait up to 10 seconds
                time_module.sleep(1)
                filled_order = self.client.get_order(order_id)
                if filled_order:
                    fill_status = filled_order.get('status', '')
                    fill_price = filled_order.get('filled_avg_price')
                    if fill_status == 'filled' and fill_price:
                        actual_fill_price = float(fill_price)
                        self._log(f"    ORDER FILLED @ ${actual_fill_price:.2f}", "SUCCESS")
                        break
                    elif fill_status in ['filled', 'partially_filled']:
                        if fill_price:
                            actual_fill_price = float(fill_price)
                        break
            else:
                self._log(f"    Using estimated fill price: ${actual_fill_price:.2f}", "WARN")

            # Create position tracking with ACTUAL fill price
            self.session.position = PaperPosition(
                symbol=occ_symbol,
                underlying=self.config.underlying_symbol,
                qty=qty,
                side='long',
                entry_price=actual_fill_price,  # Use actual fill price
                entry_time=datetime.now(EST),
                option_type=option_type_name,
                strike=option_details['strike'],
                expiration=option_details['expiration'],
                signal=signal,
                entry_order_id=order_id,
                highest_price_since_entry=actual_fill_price,
            )

            # Calculate SL/TP based on ACTUAL fill price
            sl_price = actual_fill_price * (1 - self.config.stop_loss_pct / 100)
            tp_price = actual_fill_price * (1 + self.config.target_profit_pct / 100)
            self.session.position.stop_loss_price = sl_price
            self.session.position.take_profit_price = tp_price

            self._log(f"    POSITION OPENED:", "SUCCESS")
            self._log(f"      Entry Price: ${actual_fill_price:.2f} (actual fill)", "SUCCESS")
            self._log(f"      Take Profit: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", "SUCCESS")
            self._log(f"      Stop Loss: ${sl_price:.2f} (-{self.config.stop_loss_pct}%)", "SUCCESS")
            self._log(f"      Force Exit: {self.config.force_exit_time} EST (if no TP/SL hit)", "SUCCESS")

            # Place TAKE PROFIT limit order on the exchange
            # NOTE: Alpaca only supports MARKET and LIMIT orders for options (no STOP orders)
            # So we place TP as limit order, and monitor SL internally
            self._log(f"    Submitting TAKE PROFIT limit order to Alpaca...", "TRADE")
            tp_placed = False
            max_retries = 3

            for attempt in range(1, max_retries + 1):
                try:
                    tp_order = self.client.submit_option_limit_order(
                        symbol=occ_symbol,
                        qty=qty,
                        side='sell',
                        limit_price=round(tp_price, 2),  # Round to 2 decimal places for options
                    )
                    tp_order_id = tp_order.get('id', 'N/A')
                    tp_order_status = tp_order.get('status', 'unknown')
                    self.session.position.tp_order_id = tp_order_id
                    tp_placed = True

                    self._log(f"    TAKE PROFIT ORDER PLACED:", "SUCCESS")
                    self._log(f"      TP Order ID: {tp_order_id}", "SUCCESS")
                    self._log(f"      TP Status: {tp_order_status}", "SUCCESS")
                    self._log(f"      TP Trigger: ${tp_price:.2f}", "SUCCESS")
                    break

                except Exception as tp_err:
                    if attempt < max_retries:
                        self._log(f"    Attempt {attempt}/{max_retries} failed: {tp_err}", "WARN")
                        self._log(f"    Retrying in 2 seconds...", "WARN")
                        import time
                        time.sleep(2)
                    else:
                        self._log(f"    FAILED after {max_retries} attempts: {tp_err}", "ERROR")

            # Show protection status
            print()
            if tp_placed:
                self._log("*" * 60, "SUCCESS")
                self._log("***  TP LIMIT ORDER ON EXCHANGE | SL MONITORED INTERNALLY ***", "SUCCESS")
                self._log("*" * 60, "SUCCESS")
            else:
                self._log("*" * 60, "WARN")
                self._log("***  TP ORDER FAILED - MONITORING BOTH TP/SL INTERNALLY  ***", "WARN")
                self._log("*" * 60, "WARN")

            self._log("=" * 70, "TRADE")
            print()

            self.session.has_traded_today = True
            self.session.trades_today += 1

            # Log trade entry
            try:
                trade_id = f"PAPER_{occ_symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"
                expiry_str = self.session.position.expiration.strftime("%Y-%m-%d") if isinstance(self.session.position.expiration, datetime) else str(self.session.position.expiration)[:10]

                # Fetch Greeks from Alpaca snapshot
                greeks_dict = {}
                try:
                    option_greeks = self.client.get_option_greeks(occ_symbol)
                    if option_greeks:
                        greeks_dict = {
                            'delta': option_greeks.delta,
                            'gamma': option_greeks.gamma,
                            'theta': option_greeks.theta,
                            'vega': option_greeks.vega,
                            'iv': option_greeks.implied_volatility
                        }
                        self._log(f"  Greeks: Δ={option_greeks.delta:.3f} Γ={option_greeks.gamma:.4f} Θ={option_greeks.theta:.3f} V={option_greeks.vega:.3f} IV={option_greeks.implied_volatility:.1%}", "TRADE")
                except Exception as greek_err:
                    self._log(f"  Could not fetch Greeks: {greek_err}", "WARN")

                self.trade_logger.log_entry(
                    trade_id=trade_id,
                    underlying_symbol=self.config.underlying_symbol,
                    option_symbol=occ_symbol,
                    option_type='call' if option_type == 'C' else 'put',
                    strike_price=self.session.position.strike,
                    expiration_date=expiry_str,
                    entry_time=datetime.now(EST),
                    entry_price=option_quote.ask,
                    entry_qty=qty,
                    entry_order_id=order_id,
                    entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    greeks=greeks_dict,
                    target_profit_pct=self.config.target_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    notes="PAPER TRADING"
                )
                self.session.position.entry_order_id = trade_id
            except Exception as log_err:
                self._log(f"Error logging trade entry: {log_err}", "WARN")

            if self.on_trade:
                self.on_trade('ENTRY', self.session.position)

        except Exception as e:
            self._log(f"Error entering position: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _check_exit_conditions(self):
        """Check if position should be exited."""
        if self.session.position is None:
            return

        pos = self.session.position

        # FIRST: Check if TP order has filled (Alpaca limit order)
        if pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                if tp_order and tp_order.get('status') == 'filled':
                    # TP order filled! Log the exit and clear position
                    fill_price = float(tp_order.get('filled_avg_price', pos.take_profit_price))
                    pnl_dollars = (fill_price - pos.entry_price) * pos.qty * 100
                    self._log(f"TP LIMIT ORDER FILLED @ ${fill_price:.2f}!", "SUCCESS")
                    self._handle_tp_filled(fill_price, pnl_dollars, tp_order.get('id', ''))
                    return
            except Exception as e:
                self._log(f"Error checking TP order status: {e}", "WARN")

        # FETCH POSITION DATA DIRECTLY FROM ALPACA for accurate real-time data
        # Use direct symbol lookup (more reliable than get_all_positions)
        alpaca_pos = self.client.get_position_by_symbol(pos.symbol)

        # If position is gone from Alpaca but we have a TP order, check if it filled
        if not alpaca_pos and pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                tp_status = tp_order.get('status', 'unknown') if tp_order else 'unknown'

                if tp_status == 'filled':
                    fill_price = float(tp_order.get('filled_avg_price', pos.take_profit_price))
                    pnl_dollars = (fill_price - pos.entry_price) * pos.qty * 100
                    self._log(f"TP LIMIT ORDER FILLED @ ${fill_price:.2f} (position closed)!", "SUCCESS")
                    self._handle_tp_filled(fill_price, pnl_dollars, tp_order.get('id', ''))
                    return
                elif tp_status in ['canceled', 'expired', 'rejected']:
                    self._log(f"Position gone but TP order not filled (status: {tp_status})", "WARN")
                    self._log(f"WARNING: Position disappeared from Alpaca unexpectedly!", "WARN")
                    self.session.position = None
                    return
                elif tp_status in ['pending_new', 'accepted', 'new', 'partially_filled']:
                    # TP order still pending but Alpaca positions API returned empty
                    # This can happen due to API delay - DON'T clear the position!
                    # Only log this warning once per minute to avoid spam
                    if not hasattr(self, '_last_api_delay_warn') or \
                       (datetime.now(EST) - self._last_api_delay_warn).total_seconds() > 60:
                        self._log(f"Note: Using quote pricing (Alpaca positions API slow)", "INFO")
                        self._last_api_delay_warn = datetime.now(EST)
                    # Fall through to use quote-based pricing
                else:
                    self._log(f"Position not in Alpaca, TP order status: {tp_status} - using quote pricing", "INFO")
                    # Fall through to use quote-based pricing
            except Exception as e:
                self._log(f"Position check error (keeping position): {e}", "WARN")
                # Don't clear position on error - fall through to quote-based pricing

        # Use Alpaca data if available, otherwise fall back to quote
        if alpaca_pos:
            current_price = alpaca_pos['current_price']
            # Update local position with Alpaca's actual entry price (in case it differs)
            if abs(alpaca_pos['avg_entry_price'] - pos.entry_price) > 0.01:
                self._log(f"Updating entry price from Alpaca: ${pos.entry_price:.2f} -> ${alpaca_pos['avg_entry_price']:.2f}", "INFO")
                pos.entry_price = alpaca_pos['avg_entry_price']
                # Recalculate SL/TP based on actual entry
                pos.stop_loss_price = pos.entry_price * (1 - self.config.stop_loss_pct / 100)
                pos.take_profit_price = pos.entry_price * (1 + self.config.target_profit_pct / 100)

            pnl_pct = alpaca_pos['unrealized_plpc']
            pnl_dollars = alpaca_pos['unrealized_pl']

            # Create a fake quote object for compatibility
            self.latest_option_quote = Quote(
                symbol=pos.symbol,
                bid=current_price,
                ask=current_price,
                mid=current_price,
                timestamp=datetime.now(EST)
            )
        else:
            # Fallback to quote-based pricing
            option_quote = self.client.get_latest_option_quote(pos.symbol)
            if not option_quote:
                self._log(f"Could not get quote for position {pos.symbol}", "WARN")
                return

            current_price = option_quote.mid
            self.latest_option_quote = option_quote
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
            pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

        # Update highest price for tracking
        old_hwm = pos.highest_price_since_entry
        if current_price > pos.highest_price_since_entry:
            pos.highest_price_since_entry = current_price

            # Log HWM update if trailing is active
            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                self._log(f"📈 New High Water Mark: ${old_hwm:.2f} -> ${current_price:.2f}", "SUCCESS")

        exit_reason = None

        # ========== TRAILING STOP LOGIC ==========
        if self.strategy_config.trailing_stop_enabled:
            profit_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100

            # Initialize trailing stop attributes if not present
            if not hasattr(pos, 'trailing_stop_active'):
                pos.trailing_stop_active = False
                pos.trailing_stop_price = None

            # Check if we should ACTIVATE trailing stop
            if not pos.trailing_stop_active and profit_pct >= float(self.strategy_config.trailing_trigger_pct):
                pos.trailing_stop_active = True
                pos.trailing_stop_price = pos.highest_price_since_entry * (1 - float(self.strategy_config.trailing_distance_pct) / 100)

                print()  # New line for visibility
                self._log("=" * 70, "SUCCESS")
                self._log("🔄 TRAILING STOP ACTIVATED!", "SUCCESS")
                self._log("=" * 70, "SUCCESS")
                self._log(f"    Current Price: ${current_price:.2f}", "SUCCESS")
                self._log(f"    Entry Price: ${pos.entry_price:.2f}", "SUCCESS")
                self._log(f"    Profit: +{profit_pct:.1f}% (trigger: +{self.strategy_config.trailing_trigger_pct}%)", "SUCCESS")
                self._log(f"    High Water Mark: ${pos.highest_price_since_entry:.2f}", "SUCCESS")
                self._log(f"    Trailing Stop: ${pos.trailing_stop_price:.2f} ({self.strategy_config.trailing_distance_pct}% below HWM)", "SUCCESS")
                self._log("=" * 70, "SUCCESS")
                print()

            # Update trailing stop price if active and HWM increased
            elif pos.trailing_stop_active and current_price > old_hwm:
                old_trail = pos.trailing_stop_price
                pos.trailing_stop_price = pos.highest_price_since_entry * (1 - float(self.strategy_config.trailing_distance_pct) / 100)
                self._log(f"📊 Trailing Stop Updated: ${old_trail:.2f} -> ${pos.trailing_stop_price:.2f}", "SUCCESS")

            # Check if trailing stop is HIT
            if pos.trailing_stop_active and pos.trailing_stop_price and current_price <= pos.trailing_stop_price:
                locked_profit_pct = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price) * 100

                print()  # New line for visibility
                self._log("=" * 70, "SUCCESS")
                self._log("🛑 TRAILING STOP HIT!", "SUCCESS")
                self._log("=" * 70, "SUCCESS")
                self._log(f"    Entry: ${pos.entry_price:.2f}", "INFO")
                self._log(f"    High Water Mark: ${pos.highest_price_since_entry:.2f}", "INFO")
                self._log(f"    Trail Stop: ${pos.trailing_stop_price:.2f}", "INFO")
                self._log(f"    Current Price: ${current_price:.2f}", "INFO")
                self._log(f"    Locked Profit: +{locked_profit_pct:.1f}%", "SUCCESS")
                self._log("=" * 70, "SUCCESS")
                print()

                exit_reason = "TRAILING_STOP"

        # Check take profit
        if not exit_reason and current_price >= pos.take_profit_price:
            exit_reason = "TAKE_PROFIT"

        # Check stop loss
        elif not exit_reason and current_price <= pos.stop_loss_price:
            exit_reason = "STOP_LOSS"

        # Check force exit time (no max hold time - only TP, SL, or Force Exit)
        elif not exit_reason and self._should_force_exit():
            exit_reason = "FORCE_EXIT"

        if exit_reason:
            self._exit_position(exit_reason, current_price, pnl_dollars)

    def _handle_tp_filled(self, fill_price: float, pnl: float, order_id: str):
        """Handle when TP limit order fills on exchange - no need to submit sell order."""
        if self.session.position is None:
            return

        pos = self.session.position
        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60
        pnl_pct = ((fill_price - pos.entry_price) / pos.entry_price) * 100

        print()  # New line for visibility
        self._log("=" * 70, "TRADE")
        self._log("<<< TAKE PROFIT FILLED (Limit Order on Exchange)", "SUCCESS")
        self._log("=" * 70, "TRADE")
        self._log(f"    Option: {pos.symbol}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.2f}", "TRADE")
        self._log(f"    Exit Price: ${fill_price:.2f} (TP LIMIT FILL)", "SUCCESS")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        # Update session stats
        self.session.pnl_today += pnl
        if pnl > 0:
            self.session.wins += 1
        else:
            self.session.losses += 1

        # Display P&L
        log_level = "SUCCESS" if pnl >= 0 else "WARN"
        self._log(f"    RESULT:", log_level)
        self._log(f"      P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)
        self._log(f"      Reason: TAKE_PROFIT (Limit Order Filled)", log_level)

        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_today:+.2f}", "INFO")
        self._log("=" * 70, "TRADE")
        print()

        # Log trade exit with Greeks
        try:
            exit_greeks_dict = {}
            try:
                option_greeks = self.client.get_option_greeks(pos.symbol)
                if option_greeks:
                    exit_greeks_dict = {
                        'delta': option_greeks.delta,
                        'gamma': option_greeks.gamma,
                        'theta': option_greeks.theta,
                        'vega': option_greeks.vega,
                        'iv': option_greeks.implied_volatility
                    }
                    self._log(f"  Exit Greeks: Δ={option_greeks.delta:.3f} Γ={option_greeks.gamma:.4f} Θ={option_greeks.theta:.3f} V={option_greeks.vega:.3f} IV={option_greeks.implied_volatility:.1%}", "INFO")
            except Exception:
                pass  # Greeks are optional for exit

            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=datetime.now(EST),
                exit_price=fill_price,
                exit_qty=pos.qty,
                exit_order_id=order_id,
                exit_reason="TAKE_PROFIT",
                exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                exit_greeks=exit_greeks_dict,
                notes=f"PAPER TRADING - TP Limit Order Filled - Hold time: {hold_time:.1f}m"
            )
        except Exception as log_err:
            self._log(f"Error logging trade exit: {log_err}", "WARN")

        if self.on_trade:
            self.on_trade('EXIT', pos, "TAKE_PROFIT", fill_price, pnl)

        self.session.position = None

    def _exit_position(self, reason: str, exit_price: float, pnl: float):
        """Exit the current position with comprehensive logging."""
        if self.session.position is None:
            return

        pos = self.session.position

        # Check if position is locked due to PDT protection
        if hasattr(pos, 'pdt_locked') and pos.pdt_locked:
            self._log("⚠️  Position locked due to Pattern Day Trading (PDT) protection - cannot exit programmatically", "WARN")
            return

        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100

        print()  # New line for visibility
        self._log("=" * 70, "TRADE")
        self._log(f"<<< EXIT SIGNAL: {reason}", "TRADE")
        self._log("=" * 70, "TRADE")
        self._log(f"    Option: {pos.symbol}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.2f}", "TRADE")
        self._log(f"    Exit Price: ${exit_price:.2f}", "TRADE")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        # Cancel the take profit order before exiting (unless exit is due to TP filling)
        # We have TP limit order on exchange, SL is monitored internally
        if pos.tp_order_id and reason != "TAKE_PROFIT":
            self._log(f"    Cancelling take profit order {pos.tp_order_id}...", "TRADE")
            try:
                cancelled = self.client.cancel_order(pos.tp_order_id)
                if cancelled:
                    self._log(f"    Take profit order cancelled", "SUCCESS")
                else:
                    self._log(f"    Could not cancel TP order (may have already filled)", "WARN")
            except Exception as cancel_err:
                self._log(f"    Warning: Error cancelling TP order: {cancel_err}", "WARN")

        self._log(f"    Submitting SELL order to Alpaca...", "TRADE")

        try:
            order = self.client.submit_market_order(
                symbol=pos.symbol,
                qty=pos.qty,
                side='sell'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')

            self._log(f"    ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            # Update session stats
            self.session.pnl_today += pnl
            if pnl > 0:
                self.session.wins += 1
            else:
                self.session.losses += 1

            # Display P&L
            log_level = "SUCCESS" if pnl >= 0 else "WARN"
            self._log(f"    RESULT:", log_level)
            self._log(f"      P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)
            self._log(f"      Reason: {reason}", log_level)

            win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
            self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_today:+.2f}", "INFO")
            self._log("=" * 70, "TRADE")
            print()

            # Log trade exit with Greeks
            try:
                exit_greeks_dict = {}
                try:
                    option_greeks = self.client.get_option_greeks(pos.symbol)
                    if option_greeks:
                        exit_greeks_dict = {
                            'delta': option_greeks.delta,
                            'gamma': option_greeks.gamma,
                            'theta': option_greeks.theta,
                            'vega': option_greeks.vega,
                            'iv': option_greeks.implied_volatility
                        }
                        self._log(f"  Exit Greeks: Δ={option_greeks.delta:.3f} Γ={option_greeks.gamma:.4f} Θ={option_greeks.theta:.3f} V={option_greeks.vega:.3f} IV={option_greeks.implied_volatility:.1%}", "INFO")
                except Exception:
                    pass  # Greeks are optional for exit

                self.trade_logger.log_exit(
                    trade_id=pos.entry_order_id,
                    exit_time=datetime.now(EST),
                    exit_price=exit_price,
                    exit_qty=pos.qty,
                    exit_order_id=order_id,
                    exit_reason=reason,
                    exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    exit_greeks=exit_greeks_dict,
                    notes=f"PAPER TRADING - Hold time: {hold_time:.1f}m"
                )
            except Exception as log_err:
                self._log(f"Error logging trade exit: {log_err}", "WARN")

            if self.on_trade:
                self.on_trade('EXIT', pos, reason, exit_price, pnl)

            self.session.position = None

        except Exception as e:
            error_msg = str(e)

            # Check if this is a PDT (Pattern Day Trading) error
            if "40310100" in error_msg or "pattern day trading" in error_msg.lower():
                # Mark position as PDT-locked to prevent future retry attempts
                pos.pdt_locked = True

                # Display clear, one-time warning message
                print()
                self._log("=" * 70, "ERROR")
                self._log("🚫 PATTERN DAY TRADING (PDT) PROTECTION ERROR", "ERROR")
                self._log("=" * 70, "ERROR")
                self._log("", "ERROR")
                self._log("Your Alpaca account is protected by PDT rules and cannot execute this trade.", "ERROR")
                self._log("", "ERROR")
                self._log("ACTIONABLE SOLUTIONS:", "ERROR")
                self._log("  1. Manual Close: Log into Alpaca dashboard and manually close this position", "ERROR")
                self._log("  2. Contact Alpaca: Request to disable PDT protection (requires $25k+ account)", "ERROR")
                self._log("  3. Wait: Position can be closed tomorrow (next trading day)", "ERROR")
                self._log("  4. Upgrade: Switch to a cash account (no PDT rules, but slower settlement)", "ERROR")
                self._log("", "ERROR")
                self._log(f"Position Details:", "ERROR")
                self._log(f"  Symbol: {pos.symbol}", "ERROR")
                self._log(f"  Entry: ${pos.entry_price:.2f}", "ERROR")
                self._log(f"  Current: ${exit_price:.2f}", "ERROR")
                self._log(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", "ERROR")
                self._log("", "ERROR")
                self._log("⚠️  This position will NOT be closed automatically. Manual action required.", "ERROR")
                self._log("=" * 70, "ERROR")
                print()
            else:
                # Generic error handling for non-PDT errors
                self._log(f"Error exiting position: {e}", "ERROR")
                import traceback
                traceback.print_exc()

    def _print_status(self):
        """Print current status with comprehensive monitoring."""
        now = datetime.now(EST)
        is_entry = self._is_entry_window()
        is_open = self._is_market_open()

        # Status parts
        parts = [
            f"{now.strftime('%H:%M:%S')} EST",
            f"Market: {'OPEN' if is_open else 'CLOSED'}",
        ]

        # Entry window status
        if is_entry:
            parts.append("ENTRY WINDOW")
        elif now.time() < self.entry_start:
            minutes_to_entry = (datetime.combine(now.date(), self.entry_start) - now).total_seconds() / 60
            parts.append(f"Entry in {minutes_to_entry:.0f}m")
        else:
            parts.append("Entry closed")

        # Underlying price
        if self.latest_underlying_quote:
            parts.append(f"{self.config.underlying_symbol}: ${self.latest_underlying_quote.mid:.2f}")

        # Position info and P&L
        if self.session.position:
            pos = self.session.position
            if self.latest_option_quote:
                current = self.latest_option_quote.mid
                pnl_pct = ((current - pos.entry_price) / pos.entry_price) * 100
                pnl_dollars = (current - pos.entry_price) * pos.qty * 100
                parts.append(f"{pos.option_type}: ${current:.2f} ({pnl_pct:+.1f}%)")
                # Show unrealized P&L when in position
                parts.append(f"P&L: ${pnl_dollars:+.2f}")
            else:
                parts.append(f"{pos.option_type}")
                parts.append(f"P&L: ${self.session.pnl_today:+.2f}")
        else:
            parts.append("No Position")
            # Session P&L (realized) when no position
            parts.append(f"P&L: ${self.session.pnl_today:+.2f}")

        status = " | ".join(parts)
        print(f"\r{status}    ", end='', flush=True)

        # Print detailed info every 6 iterations (60 seconds)
        self._status_counter += 1
        if self._status_counter % 6 == 0:
            self._print_monitoring_table()

    def _print_monitoring_table(self):
        """Print detailed monitoring table."""
        now = datetime.now(EST)
        print()  # New line after status
        print()
        print("-" * 70)
        print(f"MONITORING @ {now.strftime('%H:%M:%S')} EST")
        print("-" * 70)

        # Market status
        print(f"  Market: {'OPEN' if self._is_market_open() else 'CLOSED'}")
        print(f"  Entry Window: {self.entry_start} - {self.entry_end} EST")
        print(f"  Force Exit: {self.force_exit} EST")
        print(f"  Currently: {'IN ENTRY WINDOW' if self._is_entry_window() else 'OUTSIDE ENTRY WINDOW'}")

        # Underlying info
        if self.latest_underlying_quote:
            q = self.latest_underlying_quote
            spread = q.ask - q.bid
            spread_pct = (spread / q.mid * 100) if q.mid > 0 else 0
            print(f"\n  {self.config.underlying_symbol}:")
            print(f"    Bid: ${q.bid:.2f} | Ask: ${q.ask:.2f} | Mid: ${q.mid:.2f}")
            print(f"    Spread: ${spread:.2f} ({spread_pct:.2f}%)")

        # Bar info
        if self.latest_underlying_bar:
            bar = self.latest_underlying_bar
            direction = "BULLISH" if bar.close > bar.open else "BEARISH" if bar.close < bar.open else "NEUTRAL"
            print(f"    Bar: O=${bar.open:.2f} H=${bar.high:.2f} L=${bar.low:.2f} C=${bar.close:.2f} -> {direction}")

        # Position monitoring - FETCH DIRECTLY FROM ALPACA
        if self.session.position:
            pos = self.session.position

            # Get position data directly from Alpaca (use direct symbol lookup - more reliable)
            alpaca_pos = self.client.get_position_by_symbol(pos.symbol)

            print(f"\n  OPEN POSITION:")
            print(f"    {pos.option_type} @ Strike ${pos.strike:.0f}")

            # Use Alpaca data if available
            if alpaca_pos:
                print(f"    [DATA FROM ALPACA]")
                print(f"    Entry: ${alpaca_pos['avg_entry_price']:.2f} | Qty: {alpaca_pos['qty']} | Market Value: ${alpaca_pos['market_value']:.2f}")
                current = alpaca_pos['current_price']
                pnl_pct = alpaca_pos['unrealized_plpc']
                pnl_dollars = alpaca_pos['unrealized_pl']

                # Sync local entry price with Alpaca's
                if abs(alpaca_pos['avg_entry_price'] - pos.entry_price) > 0.01:
                    pos.entry_price = alpaca_pos['avg_entry_price']
                    pos.stop_loss_price = pos.entry_price * (1 - self.config.stop_loss_pct / 100)
                    pos.take_profit_price = pos.entry_price * (1 + self.config.target_profit_pct / 100)
            else:
                print(f"    [LOCAL DATA - Alpaca position not found]")
                print(f"    Entry: ${pos.entry_price:.2f} | Qty: {pos.qty}")
                if self.latest_option_quote:
                    current = self.latest_option_quote.mid
                    pnl_pct = ((current - pos.entry_price) / pos.entry_price) * 100
                    pnl_dollars = (current - pos.entry_price) * pos.qty * 100
                else:
                    current = pos.entry_price
                    pnl_pct = 0
                    pnl_dollars = 0

            # Protection status (TP on exchange, SL monitored internally)
            if pos.tp_order_id:
                print(f"    TP Order: {pos.tp_order_id[:16]}... (ON EXCHANGE)")
                print(f"    SL: Monitored internally")
            else:
                print(f"    TP/SL: Both monitored internally")

            hold_minutes = (now - pos.entry_time).total_seconds() / 60

            # Distance to TP/SL
            dist_to_tp = (pos.take_profit_price - current) / current * 100 if current > 0 else 0
            dist_to_sl = (current - pos.stop_loss_price) / current * 100 if current > 0 else 0

            # Progress bar
            price_range = pos.take_profit_price - pos.stop_loss_price
            if price_range > 0:
                progress = (current - pos.stop_loss_price) / price_range
                progress = max(0, min(1, progress))
                bar_width = 20
                filled = int(progress * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
            else:
                bar = "░" * 20

            print(f"    Current: ${current:.2f} | P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)")
            print(f"    Hold Time: {hold_minutes:.1f}m | Force Exit: {self.config.force_exit_time} EST")

            # Show trailing stop status
            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active and hasattr(pos, 'trailing_stop_price'):
                locked_profit_pct = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price) * 100
                locked_profit_dollars = (pos.trailing_stop_price - pos.entry_price) * pos.qty * 100
                hwm_pct = ((pos.highest_price_since_entry - pos.entry_price) / pos.entry_price) * 100
                print(f"    🔄 TRAILING STOP: ACTIVE @ ${pos.trailing_stop_price:.2f} ({locked_profit_pct:+.1f}% profit LOCKED)")
                print(f"       Peak: ${pos.highest_price_since_entry:.2f} ({hwm_pct:+.1f}%) | Locked: ${locked_profit_dollars:+.2f}")
            elif pnl_pct >= float(self.strategy_config.trailing_trigger_pct):
                trigger_price = pos.entry_price * (1 + float(self.strategy_config.trailing_trigger_pct) / 100)
                print(f"    ⏳ TRAILING STOP: Should be active (profit {pnl_pct:.1f}% >= {self.strategy_config.trailing_trigger_pct}%)")
            else:
                trigger_price = pos.entry_price * (1 + float(self.strategy_config.trailing_trigger_pct) / 100)
                print(f"    ⚪ TRAILING STOP: Inactive (needs ${trigger_price:.2f}, +{self.strategy_config.trailing_trigger_pct}%)")

            print(f"    SL ${pos.stop_loss_price:.2f} [{bar}] TP ${pos.take_profit_price:.2f}")
            print(f"    Distance: SL {dist_to_sl:.1f}% away | TP {dist_to_tp:.1f}% away")
        else:
            print(f"\n  No open position")
            print(f"  Trades today: {self.session.trades_today}/{self.max_trades_per_day}")
            print(f"  Traded: {'YES' if self.session.has_traded_today else 'NO'}")

        # Session stats
        print(f"\n  SESSION STATS:")
        print(f"    Wins: {self.session.wins} | Losses: {self.session.losses}")
        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        print(f"    Win Rate: {win_rate:.0f}%")
        print(f"    P&L: ${self.session.pnl_today:+.2f}")

        print("-" * 70)
        print()

    def run(self):
        """Main trading loop with comprehensive logging."""
        print()
        self._log("=" * 70)
        self._log("THE VOLUME AI - 0DTE Options Paper Trading Engine")
        self._log("=" * 70)
        self._log(f"Symbol: {self.config.underlying_symbol}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Max Trades/Day: {self.max_trades_per_day}")
        self._log(f"Take Profit: {self.config.target_profit_pct}%")
        self._log(f"Stop Loss: {self.config.stop_loss_pct}%")
        self._log(f"Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST")
        self._log(f"Force Exit: {self.config.force_exit_time} EST (if no TP/SL hit)")
        self._log("=" * 70)

        # Test connection
        try:
            account = self.client.get_account()
            self._log(f"Connected to Alpaca Paper Trading", "SUCCESS")
            self._log(f"Account ID: {account['id']}")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")
            self._log(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        except Exception as e:
            self._log(f"Failed to connect to Alpaca: {e}", "ERROR")
            return

        # CHECK FOR EXISTING POSITIONS ON STARTUP
        self._recover_existing_positions()

        self.running = True
        self._log("Starting trading loop... (Ctrl+C to stop)")
        print()

        try:
            while self.running and not self._stop_event.is_set():
                # Check if new trading day
                today = datetime.now(EST).date()
                if today != self.session.date:
                    self._log(f"New trading day: {today}")
                    self.session = TradingSession(date=today)

                # Update market data
                self.latest_underlying_quote = self.client.get_latest_stock_quote(
                    self.config.underlying_symbol
                )
                self.latest_underlying_bar = self.client.get_latest_stock_bar(
                    self.config.underlying_symbol
                )

                if not self._is_market_open():
                    self._print_status()
                    self._stop_event.wait(timeout=60)
                    continue

                # If in position, check exit conditions
                if self.session.position:
                    self._check_exit_conditions()

                # If not in position and within entry window, look for entry
                elif self._is_entry_window() and self.session.trades_today < self.max_trades_per_day:
                    print()  # New line before entry attempt
                    self._log("=" * 50)
                    self._log("IN ENTRY WINDOW - Analyzing signal...")
                    self._log("=" * 50)

                    signal = self._calculate_signal()

                    if self.latest_underlying_bar:
                        bar = self.latest_underlying_bar
                        self._log(f"  Bar: O=${bar.open:.2f} C=${bar.close:.2f} -> {signal}")
                    if self.latest_underlying_quote:
                        q = self.latest_underlying_quote
                        self._log(f"  Quote: Bid=${q.bid:.2f} Ask=${q.ask:.2f} Mid=${q.mid:.2f}")

                    self._log(f"  SIGNAL: {signal}")

                    if signal in ['BULLISH', 'BEARISH']:
                        # ========== PULLBACK DETECTION LAYER (5-MIN HTF) ==========
                        # Wait for better entry by checking pullback conditions on 5-min bars
                        self._log("-" * 50)
                        self._log("  V4 PULLBACK DETECTION (5-min HTF)", "INFO")

                        pullback_result = self._check_pullback_conditions(signal)

                        if pullback_result['ready_to_enter']:
                            self._log(f"  ✅ PULLBACK CONDITIONS MET - Entering {signal}", "SUCCESS")
                            self._enter_position(signal)
                        else:
                            self._log(f"  ⏳ WAITING FOR BETTER ENTRY...", "WARN")
                            self._log(f"     Pullback: {'YES' if pullback_result['pullback_detected'] else 'NO'} (score: {pullback_result['pullback_score']})")
                            self._log(f"     Recovery: {'YES' if pullback_result['recovery_detected'] else 'NO'} (score: {pullback_result['recovery_score']})")
                            for reason in pullback_result.get('reasons', [])[-5:]:  # Show last 5 reasons
                                self._log(f"       - {reason}")
                    else:
                        self._log(f"  Signal is NEUTRAL - waiting...", "WARN")
                    print()

                # Force exit if needed
                if self.session.position and self._should_force_exit():
                    self._check_exit_conditions()

                self._print_status()

                # Poll every 10 seconds
                self._stop_event.wait(timeout=10)

        except KeyboardInterrupt:
            self._log("\nShutdown requested...")
        finally:
            self.running = False
            self._handle_shutdown()

    def _handle_shutdown(self):
        """Handle shutdown with interactive prompts like crypto engine."""
        print()
        self._log("=" * 70)
        self._log("SHUTDOWN OPTIONS")
        self._log("=" * 70)

        # Handle open position
        if self.session.position:
            pos = self.session.position
            option_quote = self.client.get_latest_option_quote(pos.symbol)

            if option_quote:
                current_price = option_quote.mid
                pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

                print()
                self._log(f"OPEN POSITION: {pos.symbol}", "WARN")
                self._log(f"  Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f}", "INFO")
                self._log(f"  P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)", "INFO")
                print()

                # Ask user what to do with position
                while True:
                    try:
                        response = input("Close position before stopping? (y/n): ").strip().lower()
                        if response in ['y', 'yes']:
                            self._log("Closing position...", "TRADE")
                            self._exit_position("USER_SHUTDOWN", current_price, pnl_dollars)
                            break
                        elif response in ['n', 'no']:
                            self._log("Keeping position open (TP order remains active)", "WARN")
                            # Note about the TP order if keeping position
                            if pos.tp_order_id:
                                self._log(f"NOTE: Take profit order {pos.tp_order_id[:16]}... is still active on Alpaca", "INFO")
                            break
                        else:
                            print("Please enter 'y' or 'n'")
                    except (EOFError, KeyboardInterrupt):
                        self._log("Keeping position open", "WARN")
                        break
        else:
            self._log("No open positions", "INFO")

        # Ask if user wants to reset daily trade counter for testing
        print()
        self._log(f"Daily Trades: {self.session.trades_today}/{self.max_trades_per_day}", "INFO")
        self._log(f"Session P&L: ${self.session.pnl_today:+.2f}", "INFO")

        while True:
            try:
                response = input("\nReset daily trade counter for next test? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    self.session.trades_today = 0
                    self.session.has_traded_today = False
                    self._log("Daily trade counter RESET - can trade again!", "SUCCESS")
                    break
                elif response in ['n', 'no']:
                    self._log("Daily trade counter NOT reset", "INFO")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            except (EOFError, KeyboardInterrupt):
                self._log("Daily trade counter NOT reset", "INFO")
                break

        self._log("Paper trading engine stopped.")
        self._print_session_summary()

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()

    def _print_session_summary(self):
        """Print session summary."""
        print()
        self._log("=" * 70)
        self._log("SESSION SUMMARY")
        self._log("=" * 70)
        self._log(f"Date: {self.session.date}")
        self._log(f"Trades: {self.session.trades_today}")
        self._log(f"Wins: {self.session.wins} | Losses: {self.session.losses}")
        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        self._log(f"Win Rate: {win_rate:.0f}%")
        self._log(f"P&L: ${self.session.pnl_today:+.2f}")
        self._log("=" * 70)

        try:
            self.trade_logger.print_summary()
        except Exception as e:
            self._log(f"Error printing trade summary: {e}", "WARN")


def run_paper_trading(config: PaperTradingConfig):
    """Run the paper trading engine with given config."""
    engine = PaperTradingEngine(config)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    engine.run()
