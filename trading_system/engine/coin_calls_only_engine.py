"""
COIN CALLS-ONLY Pullback Trading Engine

This engine ONLY buys CALLS - never puts.
It waits for pullbacks and enters on recovery signals.

Key Features:
- CALLS ONLY - no puts ever
- Waits for pullback (dip) before entering
- Enters when recovery is detected (green candle, momentum shift)
- Uses same TP/SL mechanism as original COIN strategy
"""

import os
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import pytz
import threading

from .alpaca_client import AlpacaClient, Quote, Bar, ALPACA_AVAILABLE
from ..strategies.coin_calls_only_pullback import (
    COINCallsOnlyPullbackStrategy,
    COINCallsOnlyPullbackConfig
)
from ..analytics.options_trade_logger import OptionsTradeLogger, get_options_trade_logger

EST = pytz.timezone('America/New_York')


@dataclass
class PaperPosition:
    """Tracks an open paper trading position."""
    symbol: str
    underlying: str
    qty: int
    side: str
    entry_price: float
    entry_time: datetime
    option_type: str  # Always 'CALL' in this strategy
    strike: float
    expiration: datetime
    signal: str  # Always 'BULLISH'
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    highest_price_since_entry: float = 0.0
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


class COINCallsOnlyEngine:
    """
    CALLS-ONLY trading engine that waits for pullbacks.
    """

    def __init__(self, api_key: str = None, api_secret: str = None, config: COINCallsOnlyPullbackConfig = None):
        """Initialize the CALLS-ONLY engine.

        Args:
            api_key: Alpaca API key (can also be passed via config)
            api_secret: Alpaca API secret (can also be passed via config)
            config: Strategy configuration
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Run: pip install alpaca-py")

        self.config = config or COINCallsOnlyPullbackConfig()

        # Get API credentials (from args or fallback to environment)
        self.api_key = api_key
        self.api_secret = api_secret

        if not self.api_key or not self.api_secret:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.api_key = self.api_key or os.getenv('ALPACA_COIN_KEY') or os.getenv('ALPACA_API_KEY_PAPER')
            self.api_secret = self.api_secret or os.getenv('ALPACA_COIN_SECRET') or os.getenv('ALPACA_SECRET_KEY_PAPER')

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials required")

        self.client = AlpacaClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
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

        # Parse trading times
        self.entry_start = datetime.strptime(self.config.entry_time_start, "%H:%M:%S").time()
        self.entry_end = datetime.strptime(self.config.entry_time_end, "%H:%M:%S").time()
        self.force_exit = datetime.strptime(self.config.force_exit_time, "%H:%M:%S").time()

        # Max trades per day
        self.max_trades_per_day = self.config.max_trades_per_day

        # Trade logging
        self.trade_logger: OptionsTradeLogger = get_options_trade_logger()

    def _log(self, msg: str, level: str = "INFO"):
        """Log message with timestamp."""
        now = datetime.now(EST)
        color = ""
        reset = ""

        if sys.platform != 'win32' or 'TERM' in os.environ:
            if level == "ERROR":
                color = "\033[91m"
            elif level == "WARN":
                color = "\033[93m"
            elif level == "SUCCESS":
                color = "\033[92m"
            elif level == "TRADE":
                color = "\033[96m"
            elif level == "PULLBACK":
                color = "\033[95m"  # Magenta for pullback signals
            reset = "\033[0m"

        print(f"{color}[{now.strftime('%H:%M:%S')}] [{level}] {msg}{reset}")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EST)
        if now.weekday() >= 5:
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
        """Get this week's Friday for 0DTE expiry."""
        now = datetime.now(EST)
        weekday = now.weekday()
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = (4 - weekday) % 7
        friday = now + timedelta(days=days_to_friday)
        return friday.replace(hour=16, minute=0, second=0, microsecond=0)

    def _calculate_signal(self) -> Dict[str, Any]:
        """
        Calculate entry signal using CALLS-ONLY pullback strategy.

        Returns dict with signal info.
        """
        # Get 1-minute bars for analysis
        start_time = datetime.now(EST) - timedelta(hours=3)
        bars = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='1Min',
            start=start_time,
            limit=180
        )

        if not bars or len(bars) < 30:
            bars_count = len(bars) if bars else 0
            self._log(f"Not enough data ({bars_count} bars, need 30) - waiting", "WARN")
            return {'signal': 'NEUTRAL', 'should_enter': False, 'analysis': None}

        # Use last 60 bars
        bars = bars[-60:]

        # Update latest bar
        if bars:
            self.latest_underlying_bar = bars[-1]

        # Get signal from strategy
        result = COINCallsOnlyPullbackStrategy.get_entry_signal(bars, self.config)
        analysis = result['analysis']

        # ========== LOG ANALYSIS ==========
        print()
        self._log("=" * 70, "INFO")
        self._log("  CALLS-ONLY PULLBACK ANALYSIS", "INFO")
        self._log("=" * 70, "INFO")

        indicators = analysis.get('indicators', {})
        if indicators:
            self._log(f"  Price: ${indicators.get('price', 0):.2f} | VWAP: ${indicators.get('vwap', 0):.2f}", "INFO")
            self._log(f"  RSI: {indicators.get('rsi', 0):.1f} | EMA9: ${indicators.get('ema_9', 0):.2f} | EMA20: ${indicators.get('ema_20', 0):.2f}", "INFO")
            self._log(f"  Recent High: ${indicators.get('recent_high', 0):.2f} | Pullback: {indicators.get('pullback_from_high_pct', 0):.2f}%", "INFO")

        self._log("-" * 70, "INFO")

        # Pullback signals
        self._log("  PULLBACK DETECTION:", "PULLBACK")
        pullback_score = indicators.get('pullback_score', 0)
        self._log(f"    Score: {pullback_score}/10 | Detected: {'YES' if analysis['pullback_detected'] else 'NO'}", "PULLBACK")
        for sig in analysis.get('pullback_signals', []):
            self._log(f"      - {sig}", "PULLBACK")

        self._log("-" * 70, "INFO")

        # Recovery signals
        self._log("  RECOVERY DETECTION:", "SUCCESS")
        recovery_score = indicators.get('recovery_score', 0)
        self._log(f"    Score: {recovery_score}/9 | Detected: {'YES' if analysis['recovery_detected'] else 'NO'}", "SUCCESS")
        for sig in analysis.get('recovery_signals', []):
            if 'GREEN' in sig or 'positive' in sig.lower() or 'recovering' in sig.lower() or 'higher' in sig.lower():
                self._log(f"      + {sig}", "SUCCESS")
            else:
                self._log(f"      * {sig}", "INFO")

        self._log("-" * 70, "INFO")

        # Final decision
        self._log("  DECISION:", "INFO")
        for reason in analysis.get('reasons', []):
            if 'BUY CALL' in reason:
                self._log(f"    >>> {reason}", "SUCCESS")
            elif 'Waiting' in reason or 'No' in reason:
                self._log(f"    {reason}", "WARN")
            else:
                self._log(f"    {reason}", "INFO")

        self._log("=" * 70, "INFO")

        return result

    def _find_atm_call(self) -> Optional[str]:
        """Find ATM CALL option contract."""
        if not self.latest_underlying_quote:
            return None

        underlying_price = self.latest_underlying_quote.mid
        expiry = self._get_this_weeks_friday()
        strike = round(underlying_price / 5) * 5

        # Always CALL
        occ_symbol = self.client.format_occ_symbol(
            underlying=self.config.underlying_symbol,
            expiration=expiry,
            strike=strike,
            option_type='C'  # ALWAYS CALL
        )
        return occ_symbol

    def _calculate_position_size(self, option_price: float) -> int:
        """Calculate number of contracts based on fixed position value."""
        if option_price <= 0:
            return 0
        contract_value = option_price * 100
        contracts = int(self.config.fixed_position_value / contract_value)
        return max(1, contracts)

    def _enter_position(self):
        """Enter a CALL position."""
        if self.session.position is not None:
            self._log("Already in position, skipping entry", "WARN")
            return

        if self.session.trades_today >= self.max_trades_per_day:
            self._log(f"Max trades per day reached ({self.max_trades_per_day})", "INFO")
            return

        occ_symbol = self._find_atm_call()
        if not occ_symbol:
            self._log("Could not find ATM CALL", "ERROR")
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

        option_details = self.client.parse_occ_symbol(occ_symbol)

        print()
        self._log("=" * 70, "TRADE")
        self._log(">>> ENTERING CALL POSITION (Pullback Recovery)", "TRADE")
        self._log("=" * 70, "TRADE")
        self._log(f"    Underlying: {self.config.underlying_symbol} @ ${self.latest_underlying_quote.mid:.2f}", "TRADE")
        self._log(f"    Option: {occ_symbol}", "TRADE")
        self._log(f"    Type: CALL (always)", "TRADE")
        self._log(f"    Strike: ${option_details['strike']:.0f} | Expiry: {option_details['expiration']}", "TRADE")
        self._log(f"    Option Bid: ${option_quote.bid:.2f} | Ask: ${option_quote.ask:.2f}", "TRADE")
        self._log(f"    Qty: {qty} contracts | Value: ${self.config.fixed_position_value:.2f}", "TRADE")

        try:
            order = self.client.submit_market_order(
                symbol=occ_symbol,
                qty=qty,
                side='buy'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')

            self._log(f"    ORDER SUBMITTED: {order_id}", "SUCCESS")

            # Wait for fill
            actual_fill_price = option_quote.ask
            import time as time_module
            for _ in range(10):
                time_module.sleep(1)
                filled_order = self.client.get_order(order_id)
                if filled_order:
                    fill_status = filled_order.get('status', '')
                    fill_price = filled_order.get('filled_avg_price')
                    if fill_status == 'filled' and fill_price:
                        actual_fill_price = float(fill_price)
                        self._log(f"    FILLED @ ${actual_fill_price:.2f}", "SUCCESS")
                        break

            # Create position
            self.session.position = PaperPosition(
                symbol=occ_symbol,
                underlying=self.config.underlying_symbol,
                qty=qty,
                side='long',
                entry_price=actual_fill_price,
                entry_time=datetime.now(EST),
                option_type='CALL',  # Always CALL
                strike=option_details['strike'],
                expiration=option_details['expiration'],
                signal='BULLISH',  # Always BULLISH for calls
                entry_order_id=order_id,
                highest_price_since_entry=actual_fill_price,
            )

            # Calculate SL/TP
            sl_price = actual_fill_price * (1 - self.config.stop_loss_pct / 100)
            tp_price = actual_fill_price * (1 + self.config.target_profit_pct / 100)
            self.session.position.stop_loss_price = sl_price
            self.session.position.take_profit_price = tp_price

            self._log(f"    Take Profit: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", "SUCCESS")
            self._log(f"    Stop Loss: ${sl_price:.2f} (-{self.config.stop_loss_pct}%)", "SUCCESS")

            # Place TP limit order
            try:
                tp_order = self.client.submit_option_limit_order(
                    symbol=occ_symbol,
                    qty=qty,
                    side='sell',
                    limit_price=round(tp_price, 2),
                )
                self.session.position.tp_order_id = tp_order.get('id', '')
                self._log(f"    TP Order Placed: {tp_order.get('id', 'N/A')}", "SUCCESS")
            except Exception as tp_err:
                self._log(f"    Warning: Could not place TP order: {tp_err}", "WARN")

            self._log("=" * 70, "TRADE")
            print()

            self.session.has_traded_today = True
            self.session.trades_today += 1

            # Log to trade logger
            try:
                trade_id = f"CALLS_ONLY_{occ_symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"
                expiry_str = self.session.position.expiration.strftime("%Y-%m-%d") if isinstance(self.session.position.expiration, datetime) else str(self.session.position.expiration)[:10]

                self.trade_logger.log_entry(
                    trade_id=trade_id,
                    underlying_symbol=self.config.underlying_symbol,
                    option_symbol=occ_symbol,
                    option_type='call',
                    strike_price=self.session.position.strike,
                    expiration_date=expiry_str,
                    entry_time=datetime.now(EST),
                    entry_price=actual_fill_price,
                    entry_qty=qty,
                    entry_order_id=order_id,
                    entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    target_profit_pct=self.config.target_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    notes="CALLS-ONLY PULLBACK STRATEGY"
                )
                self.session.position.entry_order_id = trade_id
            except Exception as log_err:
                self._log(f"Error logging trade: {log_err}", "WARN")

        except Exception as e:
            self._log(f"Error entering position: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _check_exit_conditions(self):
        """Check if position should be exited."""
        if self.session.position is None:
            return

        pos = self.session.position

        # Check if TP order filled
        if pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                if tp_order and tp_order.get('status') == 'filled':
                    fill_price = float(tp_order.get('filled_avg_price', pos.take_profit_price))
                    pnl = (fill_price - pos.entry_price) * pos.qty * 100
                    self._handle_exit("TAKE_PROFIT", fill_price, pnl)
                    return
            except Exception as e:
                self._log(f"Error checking TP order: {e}", "WARN")

        # Get position from Alpaca
        alpaca_pos = self.client.get_position_by_symbol(pos.symbol)

        if alpaca_pos:
            current_price = alpaca_pos['current_price']
            pnl_pct = alpaca_pos['unrealized_plpc']
            pnl_dollars = alpaca_pos['unrealized_pl']
        else:
            # Fallback to quote
            option_quote = self.client.get_latest_option_quote(pos.symbol)
            if not option_quote:
                return
            current_price = option_quote.mid
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
            pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

        self.latest_option_quote = Quote(
            symbol=pos.symbol,
            bid=current_price,
            ask=current_price,
            mid=current_price,
            timestamp=datetime.now(EST)
        )

        if current_price > pos.highest_price_since_entry:
            pos.highest_price_since_entry = current_price

        exit_reason = None

        # Check TP
        if current_price >= pos.take_profit_price:
            exit_reason = "TAKE_PROFIT"

        # Check SL
        elif current_price <= pos.stop_loss_price:
            exit_reason = "STOP_LOSS"

        # Check force exit
        elif self._should_force_exit():
            exit_reason = "FORCE_EXIT"

        if exit_reason:
            self._handle_exit(exit_reason, current_price, pnl_dollars)

    def _handle_exit(self, reason: str, exit_price: float, pnl: float):
        """Handle position exit."""
        if self.session.position is None:
            return

        pos = self.session.position
        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100

        print()
        self._log("=" * 70, "TRADE")
        self._log(f"<<< EXIT: {reason}", "TRADE")
        self._log("=" * 70, "TRADE")
        self._log(f"    Option: {pos.symbol}", "TRADE")
        self._log(f"    Entry: ${pos.entry_price:.2f} | Exit: ${exit_price:.2f}", "TRADE")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        # Cancel TP if not filled
        if pos.tp_order_id and reason != "TAKE_PROFIT":
            try:
                self.client.cancel_order(pos.tp_order_id)
            except:
                pass

        # Sell if not TP fill
        if reason != "TAKE_PROFIT":
            try:
                self.client.submit_market_order(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    side='sell'
                )
            except Exception as e:
                self._log(f"Error selling: {e}", "ERROR")

        # Update session
        self.session.pnl_today += pnl
        if pnl > 0:
            self.session.wins += 1
        else:
            self.session.losses += 1

        log_level = "SUCCESS" if pnl >= 0 else "WARN"
        self._log(f"    P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)

        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        self._log(f"    Session: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_today:+.2f}", "INFO")
        self._log("=" * 70, "TRADE")
        print()

        # Log exit
        try:
            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=datetime.now(EST),
                exit_price=exit_price,
                exit_qty=pos.qty,
                exit_order_id='',
                exit_reason=reason,
                exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                notes=f"CALLS-ONLY - Hold time: {hold_time:.1f}m"
            )
        except Exception as log_err:
            self._log(f"Error logging exit: {log_err}", "WARN")

        self.session.position = None

    def _print_status(self):
        """Print current status."""
        now = datetime.now(EST)
        is_entry = self._is_entry_window()
        is_open = self._is_market_open()

        parts = [
            f"{now.strftime('%H:%M:%S')} EST",
            f"Market: {'OPEN' if is_open else 'CLOSED'}",
        ]

        if is_entry:
            parts.append("ENTRY WINDOW")

        if self.latest_underlying_quote:
            parts.append(f"{self.config.underlying_symbol}: ${self.latest_underlying_quote.mid:.2f}")

        if self.session.position:
            pos = self.session.position
            if self.latest_option_quote:
                current = self.latest_option_quote.mid
                pnl_pct = ((current - pos.entry_price) / pos.entry_price) * 100
                pnl_dollars = (current - pos.entry_price) * pos.qty * 100
                parts.append(f"CALL: ${current:.2f} ({pnl_pct:+.1f}%)")
                parts.append(f"P&L: ${pnl_dollars:+.2f}")
        else:
            parts.append("No Position (waiting for pullback)")
            parts.append(f"P&L: ${self.session.pnl_today:+.2f}")

        status = " | ".join(parts)
        print(f"\r{status}    ", end='', flush=True)

        self._status_counter += 1
        if self._status_counter % 6 == 0:
            self._print_monitoring_table()

    def _print_monitoring_table(self):
        """Print detailed monitoring info."""
        now = datetime.now(EST)
        print()
        print()
        print("-" * 70)
        print(f"CALLS-ONLY PULLBACK MONITORING @ {now.strftime('%H:%M:%S')} EST")
        print("-" * 70)
        print(f"  Strategy: CALLS ONLY - Wait for pullback, buy on recovery")
        print(f"  Market: {'OPEN' if self._is_market_open() else 'CLOSED'}")
        print(f"  Entry Window: {self.entry_start} - {self.entry_end} EST")

        if self.latest_underlying_quote:
            q = self.latest_underlying_quote
            print(f"\n  {self.config.underlying_symbol}: ${q.mid:.2f} (Bid: ${q.bid:.2f} / Ask: ${q.ask:.2f})")

        if self.latest_underlying_bar:
            bar = self.latest_underlying_bar
            direction = "GREEN" if bar.close > bar.open else "RED" if bar.close < bar.open else "FLAT"
            print(f"  Last Bar: O=${bar.open:.2f} H=${bar.high:.2f} L=${bar.low:.2f} C=${bar.close:.2f} -> {direction}")

        if self.session.position:
            pos = self.session.position
            print(f"\n  OPEN POSITION: {pos.option_type} @ Strike ${pos.strike:.0f}")

            alpaca_pos = self.client.get_position_by_symbol(pos.symbol)
            if alpaca_pos:
                print(f"    [FROM ALPACA]")
                print(f"    Entry: ${alpaca_pos['avg_entry_price']:.2f} | Current: ${alpaca_pos['current_price']:.2f}")
                print(f"    P&L: ${alpaca_pos['unrealized_pl']:+.2f} ({alpaca_pos['unrealized_plpc']:+.1f}%)")
            else:
                print(f"    Entry: ${pos.entry_price:.2f}")

            print(f"    TP: ${pos.take_profit_price:.2f} | SL: ${pos.stop_loss_price:.2f}")
        else:
            print(f"\n  No position - waiting for pullback + recovery")
            print(f"  Trades today: {self.session.trades_today}/{self.max_trades_per_day}")

        print(f"\n  SESSION: W/L {self.session.wins}/{self.session.losses} | P&L: ${self.session.pnl_today:+.2f}")
        print("-" * 70)
        print()

    def run(self):
        """Main trading loop."""
        print()
        self._log("=" * 70)
        self._log("THE VOLUME AI - CALLS-ONLY PULLBACK STRATEGY")
        self._log("=" * 70)
        self._log(f"Strategy: ONLY BUY CALLS - Wait for dip, buy on recovery")
        self._log(f"Symbol: {self.config.underlying_symbol}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Take Profit: {self.config.target_profit_pct}%")
        self._log(f"Stop Loss: {self.config.stop_loss_pct}%")
        self._log(f"Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST")
        self._log("=" * 70)

        # Test connection
        try:
            account = self.client.get_account()
            self._log(f"Connected to Alpaca Paper Trading", "SUCCESS")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")
        except Exception as e:
            self._log(f"Failed to connect: {e}", "ERROR")
            return

        self.running = True
        self._log("Starting CALLS-ONLY pullback trading... (Ctrl+C to stop)")
        print()

        try:
            while self.running and not self._stop_event.is_set():
                # Check new day
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

                # If in position, check exit
                if self.session.position:
                    self._check_exit_conditions()

                # If not in position, look for entry
                elif self._is_entry_window() and self.session.trades_today < self.max_trades_per_day:
                    signal_result = self._calculate_signal()

                    if signal_result['should_enter']:
                        self._log("PULLBACK + RECOVERY DETECTED - Entering CALL!", "SUCCESS")
                        self._enter_position()
                    else:
                        # Just waiting
                        pass

                self._print_status()
                self._stop_event.wait(timeout=10)

        except KeyboardInterrupt:
            self._log("\nShutdown requested...")
        finally:
            self.running = False
            self._handle_shutdown()

    def _handle_shutdown(self):
        """Handle shutdown."""
        print()
        self._log("=" * 70)
        self._log("SHUTDOWN")
        self._log("=" * 70)

        if self.session.position:
            pos = self.session.position
            self._log(f"Open position: {pos.symbol}", "WARN")
            while True:
                try:
                    response = input("Close position? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        option_quote = self.client.get_latest_option_quote(pos.symbol)
                        if option_quote:
                            pnl = (option_quote.mid - pos.entry_price) * pos.qty * 100
                            self._handle_exit("USER_SHUTDOWN", option_quote.mid, pnl)
                        break
                    elif response in ['n', 'no']:
                        self._log("Keeping position open", "WARN")
                        break
                except:
                    break
        else:
            self._log("No open positions")

        self._log(f"Session P&L: ${self.session.pnl_today:+.2f}")
        self._log("Engine stopped.")

        try:
            self.trade_logger.print_summary()
        except:
            pass

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()
