#!/usr/bin/env python3
"""
XRP Verbose Trading Engine
===========================
Comprehensive logging of signal generation, indicators, and trade development.

Shows:
- Real-time indicator values (EMA, RSI)
- Signal generation logic
- Position P&L tracking
- Distance to TP/SL/Liquidation
"""

import logging
import time
import threading
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field

try:
    from .btcc_api_client import BTCCAPIClient
    from .btcc_data_fetcher import BTCCDataFetcher
    from .xrp_95_winrate_config import XRP95WinRateConfig
except ImportError:
    from btcc_api_client import BTCCAPIClient
    from btcc_data_fetcher import BTCCDataFetcher
    from xrp_95_winrate_config import XRP95WinRateConfig

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Active trading position"""
    id: int
    direction: int  # 1=LONG, -1=SHORT
    entry_price: float
    margin: float
    tp_price: float
    sl_price: float
    liq_price: float
    open_time: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """Completed trade record"""
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    result: str
    open_time: datetime
    close_time: datetime


class XRPVerboseEngine:
    """Verbose trading engine with comprehensive logging"""

    def __init__(self, config: XRP95WinRateConfig = None):
        """Initialize the trading engine"""
        self.config = config or XRP95WinRateConfig()
        self.data_fetcher = BTCCDataFetcher(use_binance_com=False)

        if self.config.TRADING_MODE == 'live':
            self.api_client = BTCCAPIClient(
                api_key=self.config.API_KEY,
                secret_key=self.config.SECRET_KEY,
                user_name=self.config.USER_NAME,
                password=self.config.PASSWORD,
            )
        else:
            self.api_client = None

        # Trading state
        self.is_running = False
        self.position: Optional[Position] = None
        self.position_counter = 1000

        # Paper trading
        self.paper_balance = self.config.PAPER_INITIAL_BALANCE
        self.initial_balance = self.paper_balance

        # Performance
        self.trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_balance = self.paper_balance
        self.max_drawdown = 0.0

        # Indicators
        self.close_buffer: List[float] = []
        self.ema_fast_buffer: List[float] = []
        self.ema_slow_buffer: List[float] = []
        self.rsi_buffer: List[float] = []

        self._main_thread: Optional[threading.Thread] = None

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return [prices[-1] if prices else 0.0] * len(prices)
        ema = [0.0] * len(prices)
        alpha = 2 / (period + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_rsi(self, prices: List[float], period: int) -> List[float]:
        """Calculate RSI using Wilder's smoothing method"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)

        rsi = [50.0] * len(prices)

        # Calculate price changes
        deltas = [0.0]
        for i in range(1, len(prices)):
            deltas.append(prices[i] - prices[i-1])

        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]

        # Initial averages using SMA for first period
        if period < len(gains):
            avg_gain = sum(gains[1:period+1]) / period
            avg_loss = sum(losses[1:period+1]) / period
        else:
            return rsi

        # Calculate RSI using Wilder's smoothing
        for i in range(period, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss < 0.0000001:  # Avoid division by zero
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def _update_indicators(self, current_price: float):
        self.close_buffer.append(current_price)
        if len(self.close_buffer) > 200:
            self.close_buffer = self.close_buffer[-200:]
        self.ema_fast_buffer = self._calculate_ema(self.close_buffer, self.config.EMA_FAST)
        self.ema_slow_buffer = self._calculate_ema(self.close_buffer, self.config.EMA_SLOW)
        self.rsi_buffer = self._calculate_rsi(self.close_buffer, self.config.RSI_PERIOD)

    def _get_signal_with_details(self) -> tuple:
        """
        Generate signal with full details about why signal was generated or not.
        Returns: (signal, details_dict)
        """
        if len(self.close_buffer) < 20:
            return 0, {"reason": "Not enough data", "candles": len(self.close_buffer)}

        ema_fast = self.ema_fast_buffer[-1]
        ema_slow = self.ema_slow_buffer[-1]
        rsi = self.rsi_buffer[-1]
        price = self.close_buffer[-1]

        details = {
            "price": price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_diff": ema_fast - ema_slow,
            "ema_diff_pct": ((ema_fast - ema_slow) / ema_slow) * 100 if ema_slow > 0 else 0,
            "rsi": rsi,
            "trend": "BULLISH" if ema_fast > ema_slow else "BEARISH",
        }

        # Check LONG conditions
        ema_bullish = ema_fast > ema_slow
        rsi_long_ok = self.config.RSI_LONG_MIN < rsi < self.config.RSI_LONG_MAX

        # Check SHORT conditions
        ema_bearish = ema_fast < ema_slow
        rsi_short_ok = self.config.RSI_SHORT_MIN < rsi < self.config.RSI_SHORT_MAX

        details["long_conditions"] = {
            "ema_bullish": ema_bullish,
            "rsi_in_range": rsi_long_ok,
            "rsi_range": f"{self.config.RSI_LONG_MIN}-{self.config.RSI_LONG_MAX}",
            "all_met": ema_bullish and rsi_long_ok,
        }

        details["short_conditions"] = {
            "ema_bearish": ema_bearish,
            "rsi_in_range": rsi_short_ok,
            "rsi_range": f"{self.config.RSI_SHORT_MIN}-{self.config.RSI_SHORT_MAX}",
            "all_met": ema_bearish and rsi_short_ok,
        }

        if ema_bullish and rsi_long_ok:
            details["signal"] = "LONG"
            details["reason"] = f"EMA{self.config.EMA_FAST} > EMA{self.config.EMA_SLOW} AND RSI({rsi:.1f}) in {self.config.RSI_LONG_MIN}-{self.config.RSI_LONG_MAX}"
            return 1, details

        if ema_bearish and rsi_short_ok:
            details["signal"] = "SHORT"
            details["reason"] = f"EMA{self.config.EMA_FAST} < EMA{self.config.EMA_SLOW} AND RSI({rsi:.1f}) in {self.config.RSI_SHORT_MIN}-{self.config.RSI_SHORT_MAX}"
            return -1, details

        # No signal - explain why
        reasons = []
        if ema_bullish:
            if not rsi_long_ok:
                if rsi < self.config.RSI_LONG_MIN:
                    reasons.append(f"RSI({rsi:.1f}) < {self.config.RSI_LONG_MIN} (too low for LONG)")
                else:
                    reasons.append(f"RSI({rsi:.1f}) > {self.config.RSI_LONG_MAX} (too high for LONG)")
        else:  # bearish
            if not rsi_short_ok:
                if rsi < self.config.RSI_SHORT_MIN:
                    reasons.append(f"RSI({rsi:.1f}) < {self.config.RSI_SHORT_MIN} (too low for SHORT)")
                else:
                    reasons.append(f"RSI({rsi:.1f}) > {self.config.RSI_SHORT_MAX} (too high for SHORT)")

        details["signal"] = "NONE"
        details["reason"] = "; ".join(reasons) if reasons else "Conditions not met"
        return 0, details

    def _get_balance(self) -> float:
        if self.config.TRADING_MODE == 'paper':
            return self.paper_balance
        return self.api_client.get_balance() if self.api_client else 0

    def _calculate_position_pnl(self, current_price: float) -> dict:
        """Calculate detailed P&L for current position"""
        if not self.position:
            return {}

        direction = self.position.direction
        entry = self.position.entry_price
        margin = self.position.margin

        if direction == 1:  # LONG
            price_change_pct = (current_price - entry) / entry * 100
            distance_to_tp = (self.position.tp_price - current_price) / current_price * 100
            distance_to_sl = (current_price - self.position.sl_price) / current_price * 100
            distance_to_liq = (current_price - self.position.liq_price) / current_price * 100
        else:  # SHORT
            price_change_pct = (entry - current_price) / entry * 100
            distance_to_tp = (current_price - self.position.tp_price) / current_price * 100
            distance_to_sl = (self.position.sl_price - current_price) / current_price * 100
            distance_to_liq = (self.position.liq_price - current_price) / current_price * 100

        gross_pnl = (price_change_pct / 100) * margin * self.config.LEVERAGE
        fees = margin * self.config.LEVERAGE * self.config.TAKER_FEE * 2
        net_pnl = gross_pnl - fees

        return {
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": entry,
            "current": current_price,
            "price_change_pct": price_change_pct,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "margin": margin,
            "tp_price": self.position.tp_price,
            "sl_price": self.position.sl_price,
            "liq_price": self.position.liq_price,
            "distance_to_tp_pct": distance_to_tp,
            "distance_to_sl_pct": distance_to_sl,
            "distance_to_liq_pct": distance_to_liq,
            "hold_time": (datetime.now() - self.position.open_time).total_seconds(),
        }

    def _open_position(self, direction: int, price: float, signal_details: dict) -> bool:
        balance = self._get_balance()
        margin = balance * (self.config.POSITION_SIZE_PCT / 100)

        if margin <= 0:
            logger.warning("Insufficient balance")
            return False

        tp_price = self.config.get_tp_price(price, direction)
        sl_price = self.config.get_sl_price(price, direction)
        liq_price = self.config.get_liquidation_price(price, direction)

        self.position_counter += 1
        self.position = Position(
            id=self.position_counter,
            direction=direction,
            entry_price=price,
            margin=margin,
            tp_price=tp_price,
            sl_price=sl_price,
            liq_price=liq_price,
        )

        dir_str = "LONG" if direction == 1 else "SHORT"

        # Verbose output
        print()
        print("=" * 80)
        print(f"{'[PAPER] ' if self.config.TRADING_MODE == 'paper' else ''}NEW {dir_str} POSITION OPENED")
        print("=" * 80)
        print(f"Signal Reason:    {signal_details.get('reason', 'N/A')}")
        print(f"Entry Price:      ${price:.4f}")
        print(f"Position Size:    ${margin:.2f} ({self.config.POSITION_SIZE_PCT}% of ${balance:.2f})")
        print(f"Leverage:         {self.config.LEVERAGE}x")
        print(f"Notional Value:   ${margin * self.config.LEVERAGE:.2f}")
        print("-" * 80)
        print(f"Take Profit:      ${tp_price:.4f} ({self.config.TAKE_PROFIT_PCT}% from entry)")
        print(f"Stop Loss:        ${sl_price:.4f} ({self.config.STOP_LOSS_PCT}% from entry)")
        print(f"Liquidation:      ${liq_price:.4f}")
        print("-" * 80)
        print("Indicators at Entry:")
        print(f"  EMA{self.config.EMA_FAST}:          ${signal_details.get('ema_fast', 0):.4f}")
        print(f"  EMA{self.config.EMA_SLOW}:         ${signal_details.get('ema_slow', 0):.4f}")
        print(f"  EMA Diff:        {signal_details.get('ema_diff_pct', 0):.3f}%")
        print(f"  RSI({self.config.RSI_PERIOD}):         {signal_details.get('rsi', 0):.1f}")
        print(f"  Trend:           {signal_details.get('trend', 'N/A')}")
        print("=" * 80)
        print()

        return True

    def _close_position(self, exit_price: float, result: str) -> float:
        if not self.position:
            return 0.0

        pnl_info = self._calculate_position_pnl(exit_price)
        pnl = pnl_info.get('net_pnl', 0) if result != 'LIQ' else -self.position.margin

        if self.config.TRADING_MODE == 'paper':
            self.paper_balance += pnl

        trade = Trade(
            direction=pnl_info.get('direction', 'UNKNOWN'),
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            result=result,
            open_time=self.position.open_time,
            close_time=datetime.now(),
        )
        self.trades.append(trade)

        self.daily_pnl += pnl
        self.total_pnl += pnl

        if self.paper_balance > self.max_balance:
            self.max_balance = self.paper_balance
        dd = (self.max_balance - self.paper_balance) / self.max_balance * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        # Verbose output
        print()
        print("=" * 80)
        print(f"{'[PAPER] ' if self.config.TRADING_MODE == 'paper' else ''}POSITION CLOSED - {result}")
        print("=" * 80)
        print(f"Direction:        {pnl_info.get('direction', 'N/A')}")
        print(f"Entry:            ${self.position.entry_price:.4f}")
        print(f"Exit:             ${exit_price:.4f}")
        print(f"Price Change:     {pnl_info.get('price_change_pct', 0):+.2f}%")
        print("-" * 80)
        print(f"Gross P&L:        ${pnl_info.get('gross_pnl', 0):+.2f}")
        print(f"Fees:             ${pnl_info.get('fees', 0):.2f}")
        print(f"Net P&L:          ${pnl:+.2f}")
        print("-" * 80)
        print(f"Hold Time:        {pnl_info.get('hold_time', 0):.0f} seconds")
        print(f"New Balance:      ${self.paper_balance:.2f}")
        print(f"Total P&L:        ${self.total_pnl:+.2f}")
        print("=" * 80)
        print()

        self.position = None
        return pnl

    def _check_position(self, current_price: float):
        if not self.position:
            return

        direction = self.position.direction

        if direction == 1:
            if current_price <= self.position.liq_price:
                self._close_position(self.position.liq_price, 'LIQUIDATION')
            elif current_price >= self.position.tp_price:
                self._close_position(self.position.tp_price, 'TAKE PROFIT')
            elif current_price <= self.position.sl_price:
                self._close_position(self.position.sl_price, 'STOP LOSS')
        else:
            if current_price >= self.position.liq_price:
                self._close_position(self.position.liq_price, 'LIQUIDATION')
            elif current_price <= self.position.tp_price:
                self._close_position(self.position.tp_price, 'TAKE PROFIT')
            elif current_price >= self.position.sl_price:
                self._close_position(self.position.sl_price, 'STOP LOSS')

    def _print_verbose_status(self, current_price: float, signal: int, signal_details: dict):
        """Print comprehensive status"""
        wins = len([t for t in self.trades if t.pnl > 0])
        losses = len([t for t in self.trades if t.pnl <= 0])
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        # Calculate floating balance (balance + unrealized P&L)
        balance = self._get_balance()
        unrealized_pnl = 0.0
        if self.position:
            pnl_info = self._calculate_position_pnl(current_price)
            unrealized_pnl = pnl_info.get('net_pnl', 0)
        floating_balance = balance + unrealized_pnl

        print()
        print("=" * 100)
        print(f"LIVE STATUS @ {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 100)
        print()

        # Price & Account
        print(f"PRICE:           ${current_price:.4f}")
        print(f"BALANCE:         ${balance:.2f} (realized)")
        print(f"UNREALIZED P&L:  ${unrealized_pnl:+.2f}")
        print(f"FLOATING BAL:    ${floating_balance:.2f} (balance if closed now)")
        print(f"TOTAL P&L:       ${self.total_pnl + unrealized_pnl:+.2f}")
        print(f"TRADES:          {total} (W:{wins} L:{losses}) | Win Rate: {win_rate:.1f}%")
        print()

        # Indicators
        print("-" * 50)
        print("INDICATORS")
        print("-" * 50)
        print(f"EMA{self.config.EMA_FAST}:            ${signal_details.get('ema_fast', 0):.4f}")
        print(f"EMA{self.config.EMA_SLOW}:           ${signal_details.get('ema_slow', 0):.4f}")
        print(f"EMA Diff:        {signal_details.get('ema_diff_pct', 0):+.3f}% ({'BULLISH' if signal_details.get('ema_diff', 0) > 0 else 'BEARISH'})")
        print(f"RSI({self.config.RSI_PERIOD}):          {signal_details.get('rsi', 0):.1f}")
        print()

        # Signal Analysis
        print("-" * 50)
        print("SIGNAL ANALYSIS")
        print("-" * 50)
        long_cond = signal_details.get('long_conditions', {})
        short_cond = signal_details.get('short_conditions', {})

        print(f"LONG Conditions:  EMA Bullish: {'YES' if long_cond.get('ema_bullish') else 'NO'} | RSI in {long_cond.get('rsi_range', 'N/A')}: {'YES' if long_cond.get('rsi_in_range') else 'NO'}")
        print(f"SHORT Conditions: EMA Bearish: {'YES' if short_cond.get('ema_bearish') else 'NO'} | RSI in {short_cond.get('rsi_range', 'N/A')}: {'YES' if short_cond.get('rsi_in_range') else 'NO'}")
        print(f"SIGNAL:          {signal_details.get('signal', 'NONE')}")
        if signal_details.get('reason'):
            print(f"REASON:          {signal_details.get('reason')}")
        print()

        # Position
        if self.position:
            pnl_info = self._calculate_position_pnl(current_price)
            print("-" * 50)
            print("OPEN POSITION")
            print("-" * 50)
            print(f"Direction:       {pnl_info.get('direction')}")
            print(f"Entry:           ${pnl_info.get('entry', 0):.4f}")
            print(f"Current:         ${current_price:.4f}")
            print(f"Unrealized P&L:  ${pnl_info.get('net_pnl', 0):+.2f} ({pnl_info.get('price_change_pct', 0):+.2f}%)")
            print()
            print(f"Take Profit:     ${pnl_info.get('tp_price', 0):.4f} ({pnl_info.get('distance_to_tp_pct', 0):+.2f}% away)")
            print(f"Stop Loss:       ${pnl_info.get('sl_price', 0):.4f} ({pnl_info.get('distance_to_sl_pct', 0):+.2f}% away)")
            print(f"Liquidation:     ${pnl_info.get('liq_price', 0):.4f} ({pnl_info.get('distance_to_liq_pct', 0):+.2f}% away)")
            print(f"Hold Time:       {pnl_info.get('hold_time', 0):.0f} seconds")
        else:
            print("-" * 50)
            print("NO OPEN POSITION - Waiting for signal...")

        print()
        print("=" * 100)

    def _trading_loop(self):
        print()
        print("=" * 100)
        print("XRP VERBOSE TRADING ENGINE STARTED")
        print("=" * 100)
        print(f"Symbol:          {self.config.SYMBOL}")
        print(f"Mode:            {self.config.TRADING_MODE.upper()}")
        print(f"Leverage:        {self.config.LEVERAGE}x")
        print(f"Take Profit:     {self.config.TAKE_PROFIT_PCT}%")
        print(f"Stop Loss:       {self.config.STOP_LOSS_PCT}%")
        print(f"Position Size:   {self.config.POSITION_SIZE_PCT}%")
        print(f"EMA Periods:     {self.config.EMA_FAST}/{self.config.EMA_SLOW}")
        print(f"RSI Period:      {self.config.RSI_PERIOD}")
        print(f"RSI Long Range:  {self.config.RSI_LONG_MIN}-{self.config.RSI_LONG_MAX}")
        print(f"RSI Short Range: {self.config.RSI_SHORT_MIN}-{self.config.RSI_SHORT_MAX}")
        print(f"Initial Balance: ${self._get_balance():.2f}")
        print("=" * 100)
        print()

        print("Loading initial price data...")
        candles = self.data_fetcher.get_klines(self.config.SYMBOL, self.config.CANDLE_INTERVAL, 100)
        for candle in candles:
            self.close_buffer.append(candle.close)

        if len(self.close_buffer) < 20:
            print("ERROR: Failed to load enough data")
            self.is_running = False
            return

        self.ema_fast_buffer = self._calculate_ema(self.close_buffer, self.config.EMA_FAST)
        self.ema_slow_buffer = self._calculate_ema(self.close_buffer, self.config.EMA_SLOW)
        self.rsi_buffer = self._calculate_rsi(self.close_buffer, self.config.RSI_PERIOD)

        print(f"Loaded {len(self.close_buffer)} candles. Trading started!")
        print()

        status_interval = 10  # Print status every 10 checks
        check_count = 0

        while self.is_running:
            try:
                current_price = self.data_fetcher.get_current_price(self.config.SYMBOL)
                if not current_price:
                    print("WARNING: Failed to get price, retrying...")
                    time.sleep(5)
                    continue

                self._update_indicators(current_price)
                self._check_position(current_price)

                signal, signal_details = self._get_signal_with_details()

                if self.position is None and signal != 0:
                    self._open_position(signal, current_price, signal_details)

                check_count += 1
                if check_count >= status_interval:
                    check_count = 0
                    self._print_verbose_status(current_price, signal, signal_details)

                time.sleep(self.config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(10)

        if self.position:
            current_price = self.data_fetcher.get_current_price(self.config.SYMBOL)
            if current_price:
                self._close_position(current_price, 'MANUAL')

    def start(self):
        if self.is_running:
            print("Engine already running")
            return

        self.is_running = True
        self._main_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._main_thread.start()

    def stop(self):
        self.is_running = False
        if self._main_thread:
            self._main_thread.join(timeout=10)
        print("Engine stopped")

    def run(self):
        self.start()
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()
        self._print_final_stats()

    def _print_final_stats(self):
        wins = len([t for t in self.trades if t.pnl > 0])
        losses = len([t for t in self.trades if t.pnl <= 0])
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0
        final_balance = self._get_balance()
        return_pct = (final_balance - self.initial_balance) / self.initial_balance * 100

        print()
        print("=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Initial:         ${self.initial_balance:.2f}")
        print(f"Final:           ${final_balance:.2f}")
        print(f"P&L:             ${final_balance - self.initial_balance:+.2f} ({return_pct:+.1f}%)")
        print(f"Max Drawdown:    {self.max_drawdown:.1f}%")
        print(f"Total Trades:    {total}")
        print(f"Wins:            {wins}")
        print(f"Losses:          {losses}")
        print(f"Win Rate:        {win_rate:.1f}%")
        if self.trades:
            print()
            print("Trade History:")
            for i, t in enumerate(self.trades, 1):
                print(f"  {i}. {t.direction} | Entry: ${t.entry_price:.4f} -> Exit: ${t.exit_price:.4f} | P&L: ${t.pnl:+.2f} | {t.result}")
        print("=" * 80)
