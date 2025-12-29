"""
Binance Futures Paper Trading Engine
====================================
Simulates live trading with real-time market data (no real orders)
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import (
    FUTURES_SYMBOLS, STRATEGY_CONFIG, RISK_CONFIG, DCA_CONFIG,
    PAPER_TRADING_CONFIG, LOGGING_CONFIG, MOMENTUM_CONFIG
)
from engine.binance_client import BinanceClient
from engine.momentum_signal import MasterMomentumSignal, MultiTimeframeMomentumSignal, TradingSignal


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str                      # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop_price: Optional[float] = None
    trailing_stop_active: bool = False
    highest_price: float = 0.0     # For trailing stop (LONG)
    lowest_price: float = float("inf")  # For trailing stop (SHORT)
    dca_count: int = 0             # Number of DCA levels triggered
    avg_entry_price: float = 0.0   # Average entry after DCA
    total_cost: float = 0.0        # Total position cost (margin used)
    margin_used: float = 0.0       # Total margin used for this symbol


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str
    dca_count: int = 0


class BinancePaperTradingEngine:
    """
    Paper Trading Engine for Binance Futures

    Features:
    - Real-time market data from Binance
    - Simulated order execution
    - Position management (SL/TP/Trailing Stop)
    - DCA support
    - Performance tracking
    """

    def __init__(self, use_mtf: bool = True):
        """
        Initialize paper trading engine

        Args:
            use_mtf: Use multi-timeframe signal confirmation
        """
        # Initialize client (testnet mode)
        self.client = BinanceClient(testnet=True)

        # Initialize signal generator
        if use_mtf:
            self.signal_generator = MultiTimeframeMomentumSignal()
        else:
            self.signal_generator = MasterMomentumSignal()
        self.use_mtf = use_mtf

        # Paper trading state
        self.balance = PAPER_TRADING_CONFIG["initial_balance"]
        self.initial_balance = self.balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Dynamic Fund Allocation
        self.num_symbols = len(FUTURES_SYMBOLS)
        self.symbol_budgets: Dict[str, float] = {}      # Max budget per symbol
        self.symbol_margin_used: Dict[str, float] = {}  # Current margin used per symbol
        self.leverage = STRATEGY_CONFIG["leverage"]      # Fixed 20x leverage

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_start_balance = self.balance
        self.last_reset_date = datetime.now().date()

        # Symbol tracking
        self.last_check_time: Dict[str, datetime] = {}
        self.symbol_check_interval = PAPER_TRADING_CONFIG["symbol_check_interval"]

        # Data buffer for SMART DCA (stores recent market data per symbol)
        self.data_buffer: Dict[str, any] = {}

        # Logging
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Binance_Futures_Trading",
            LOGGING_CONFIG["log_dir"]
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": self.balance,
        }

        self.running = False

    def log(self, message: str, level: str = "INFO"):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def initialize_dynamic_allocation(self):
        """
        Initialize dynamic fund allocation based on current balance.

        Divides balance equally among all symbols.
        Each symbol gets: balance / num_symbols
        """
        # Apply buffer for fees/safety
        buffer = RISK_CONFIG.get("allocation_buffer_pct", 0.05)
        available_balance = self.balance * (1 - buffer)

        # Calculate budget per symbol
        budget_per_symbol = available_balance / self.num_symbols

        self.log(f"Dynamic Allocation: ${self.balance:.2f} balance")
        self.log(f"  Buffer: {buffer*100:.1f}% reserved for fees")
        self.log(f"  Available: ${available_balance:.2f}")
        self.log(f"  Symbols: {self.num_symbols}")
        self.log(f"  Budget per symbol: ${budget_per_symbol:.2f}")
        self.log(f"  Leverage: {self.leverage}x (ISOLATED)")

        # Initialize budgets for each symbol
        for symbol in FUTURES_SYMBOLS:
            self.symbol_budgets[symbol] = budget_per_symbol
            self.symbol_margin_used[symbol] = 0.0

        return budget_per_symbol

    def get_entry_margin(self, symbol: str) -> float:
        """
        Get margin for initial entry based on symbol budget.

        Returns margin amount (20% of symbol budget for initial entry).
        """
        if symbol not in self.symbol_budgets:
            return 0.0

        budget = self.symbol_budgets[symbol]
        entry_pct = RISK_CONFIG.get("initial_entry_pct", 0.20)

        return budget * entry_pct

    def get_dca_margin(self, symbol: str, dca_level: int) -> float:
        """
        Get margin for DCA level based on symbol budget.

        Args:
            symbol: Trading symbol
            dca_level: DCA level (1-4)

        Returns margin amount for that DCA level.
        """
        if symbol not in self.symbol_budgets:
            return 0.0

        budget = self.symbol_budgets[symbol]

        # Get DCA percentage from config
        dca_pcts = {
            1: RISK_CONFIG.get("dca1_pct", 0.25),
            2: RISK_CONFIG.get("dca2_pct", 0.20),
            3: RISK_CONFIG.get("dca3_pct", 0.20),
            4: RISK_CONFIG.get("dca4_pct", 0.15),
        }

        return budget * dca_pcts.get(dca_level, 0.0)

    def get_remaining_budget(self, symbol: str) -> float:
        """Get remaining budget for a symbol."""
        if symbol not in self.symbol_budgets:
            return 0.0

        return self.symbol_budgets[symbol] - self.symbol_margin_used.get(symbol, 0.0)

    def can_afford_dca(self, symbol: str, dca_level: int) -> bool:
        """Check if we have budget for this DCA level."""
        required_margin = self.get_dca_margin(symbol, dca_level)
        remaining = self.get_remaining_budget(symbol)
        return remaining >= required_margin

    def check_daily_reset(self):
        """Reset daily counters at midnight UTC"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.log(f"Daily reset - Previous day P&L: ${self.daily_pnl:.2f}")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_start_balance = self.balance
            self.last_reset_date = today

    def can_trade(self, symbol: str) -> bool:
        """Check if trading is allowed"""
        # Check daily trade limit
        if self.daily_trades >= STRATEGY_CONFIG["max_trades_per_day"]:
            return False

        # Check daily loss limit
        daily_loss_pct = (self.daily_start_balance - self.balance) / self.daily_start_balance
        if daily_loss_pct >= RISK_CONFIG["max_daily_loss_pct"]:
            return False

        # Check max positions
        if len(self.positions) >= RISK_CONFIG["max_total_positions"]:
            return False

        # Check if already have position in this symbol
        if symbol in self.positions:
            return False

        # Check symbol cooldown
        last_check = self.last_check_time.get(symbol, datetime.min)
        if (datetime.now() - last_check).total_seconds() < self.symbol_check_interval:
            return False

        return True

    def calculate_position_size(self, symbol: str, price: float, is_dca: bool = False, dca_level: int = 0) -> float:
        """
        Calculate position size based on dynamic allocation.

        Uses margin-based allocation:
        - Initial entry: 20% of symbol budget as margin
        - DCA levels: 25%, 20%, 20%, 15% of symbol budget

        With 20x leverage, position value = margin * leverage

        Returns quantity to trade
        """
        # Get margin for this entry type
        if is_dca and dca_level > 0:
            margin = self.get_dca_margin(symbol, dca_level)
        else:
            margin = self.get_entry_margin(symbol)

        # Check if we have enough budget
        remaining = self.get_remaining_budget(symbol)
        if margin > remaining:
            margin = remaining  # Use what's available

        if margin <= 0:
            return 0.0

        # Calculate position value with leverage
        # Position Value = Margin * Leverage
        position_value = margin * self.leverage

        # Convert to quantity
        quantity = position_value / price

        return quantity

    def calculate_sl_tp(self, entry_price: float, side: str) -> tuple:
        """Calculate stop loss and take profit prices using ROI-based calculation"""
        leverage = STRATEGY_CONFIG["leverage"]  # 20x

        if DCA_CONFIG["enabled"]:
            # ROI-based: Convert ROI to price move
            # Formula: price_move = roi / leverage
            tp_roi = DCA_CONFIG["take_profit_roi"]    # 15% ROI
            sl_roi = DCA_CONFIG["stop_loss_roi"]      # 80% ROI
            tp_price_pct = tp_roi / leverage          # 15% / 20 = 0.75% price move
            sl_price_pct = sl_roi / leverage          # 80% / 20 = 4% price move
        else:
            # Legacy price-based
            tp_price_pct = STRATEGY_CONFIG["take_profit_pct"]
            sl_price_pct = STRATEGY_CONFIG["stop_loss_pct"]

        if side == "LONG":
            stop_loss = entry_price * (1 - sl_price_pct)
            take_profit = entry_price * (1 + tp_price_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + sl_price_pct)
            take_profit = entry_price * (1 - tp_price_pct)

        return stop_loss, take_profit

    def enter_position(self, symbol: str, signal: TradingSignal, current_price: float):
        """Enter a new position with dynamic margin allocation"""
        side = "LONG" if signal.signal == "BUY" else "SHORT"

        # Get margin for initial entry
        margin = self.get_entry_margin(symbol)
        remaining = self.get_remaining_budget(symbol)

        if margin > remaining or margin <= 0:
            self.log(f"Skip {symbol}: Insufficient budget (need ${margin:.2f}, have ${remaining:.2f})", level="WARN")
            return

        # Calculate position size based on margin + leverage
        quantity = self.calculate_position_size(symbol, current_price)

        if quantity <= 0:
            self.log(f"Skip {symbol}: Position size too small", level="WARN")
            return

        # Apply slippage
        slippage = PAPER_TRADING_CONFIG["slippage_pct"]
        if side == "LONG":
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)

        # Calculate SL/TP
        stop_loss, take_profit = self.calculate_sl_tp(entry_price, side)

        # Apply commission
        commission = entry_price * quantity * PAPER_TRADING_CONFIG["commission_per_trade"]
        self.balance -= commission

        # Track margin used for this symbol
        self.symbol_margin_used[symbol] = margin

        # Position value = margin * leverage (notional)
        position_value = margin * self.leverage

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=entry_price,
            lowest_price=entry_price,
            avg_entry_price=entry_price,
            total_cost=position_value,  # Notional position value
            margin_used=margin          # Actual margin (collateral)
        )

        self.positions[symbol] = position
        self.daily_trades += 1

        budget = self.symbol_budgets.get(symbol, 0)
        self.log(
            f"ENTER {side} {symbol} @ ${entry_price:,.2f} | "
            f"Qty: {quantity:.6f} | Margin: ${margin:.2f}/{budget:.2f} | "
            f"Leverage: {self.leverage}x | SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}",
            level="TRADE"
        )

    def check_dca(self, position: Position, current_price: float) -> bool:
        """Check if DCA should trigger (ROI-based)"""
        if not DCA_CONFIG["enabled"]:
            return False

        if position.dca_count >= len(DCA_CONFIG["levels"]):
            return False

        # Calculate current ROI (for leveraged positions)
        leverage = STRATEGY_CONFIG["leverage"]  # 20x
        if position.side == "LONG":
            price_drawdown = (position.avg_entry_price - current_price) / position.avg_entry_price
        else:
            price_drawdown = (current_price - position.avg_entry_price) / position.avg_entry_price

        # Convert to ROI: ROI = price_change * leverage
        current_roi_loss = price_drawdown * leverage  # e.g., 0.75% price loss * 20 = 15% ROI loss

        # Check if next DCA level triggered (ROI-based)
        next_level = DCA_CONFIG["levels"][position.dca_count]
        trigger_roi = abs(next_level["trigger_roi"])  # e.g., 0.15 = 15% ROI loss

        return current_roi_loss >= trigger_roi

    def execute_dca(self, symbol: str, current_price: float):
        """Execute DCA order with margin-based allocation and SMART DCA validation"""
        position = self.positions[symbol]
        dca_level = position.dca_count + 1  # Next DCA level (1-4)

        # Check if we can afford this DCA level
        if not self.can_afford_dca(symbol, dca_level):
            remaining = self.get_remaining_budget(symbol)
            required = self.get_dca_margin(symbol, dca_level)
            self.log(f"DCA {dca_level} skipped for {symbol}: Budget exhausted (need ${required:.2f}, have ${remaining:.2f})", level="WARN")
            return

        # =================================================================
        # SMART DCA: Validate signal/trend before executing
        # =================================================================
        if symbol in self.data_buffer and self.data_buffer[symbol] is not None:
            df = self.data_buffer[symbol].get("1m") if isinstance(self.data_buffer[symbol], dict) else self.data_buffer[symbol]
            if df is not None:
                # Use signal generator to check if DCA is safe
                can_dca, reason = self.signal_generator.can_dca(df, position.side, dca_level)

                if not can_dca:
                    # Trend too strong against us - wait for reversal
                    self.log(
                        f"DCA {dca_level} {symbol}: BLOCKED - {reason}",
                        level="DCA"
                    )
                    self.log(
                        f"       >>> Waiting for market reversal before adding to position",
                        level="DCA"
                    )
                    return

                self.log(f"DCA {dca_level} {symbol}: Signal validated - {reason}", level="DCA")

        # Get DCA margin for this level
        dca_margin = self.get_dca_margin(symbol, dca_level)

        # Calculate DCA quantity based on margin + leverage
        dca_qty = self.calculate_position_size(symbol, current_price, is_dca=True, dca_level=dca_level)

        if dca_qty <= 0:
            self.log(f"DCA {dca_level} skipped for {symbol}: Quantity too small", level="WARN")
            return

        # Apply slippage
        slippage = PAPER_TRADING_CONFIG["slippage_pct"]
        if position.side == "LONG":
            fill_price = current_price * (1 + slippage)
        else:
            fill_price = current_price * (1 - slippage)

        # Calculate DCA position value
        dca_position_value = dca_margin * self.leverage

        # Update margin tracking
        self.symbol_margin_used[symbol] += dca_margin
        position.margin_used += dca_margin

        # Update position
        old_cost = position.total_cost
        position.total_cost = old_cost + dca_position_value
        position.quantity += dca_qty
        position.avg_entry_price = position.total_cost / position.quantity
        position.dca_count = dca_level

        # Tighten SL and REDUCE TP after DCA (ROI-based)
        # REDUCED TP: Exit faster to reduce exposure on DCA positions
        leverage = STRATEGY_CONFIG["leverage"]
        sl_roi_after_dca = DCA_CONFIG["sl_after_dca_roi"]  # 20% ROI
        sl_price_pct = sl_roi_after_dca / leverage         # 20% / 20 = 1% price move

        # Get REDUCED TP from DCA level config
        dca_level_config = DCA_CONFIG["levels"][dca_level - 1]  # dca_level is 1-indexed
        tp_roi = dca_level_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])
        tp_price_pct = tp_roi / leverage

        if position.side == "LONG":
            position.stop_loss = position.avg_entry_price * (1 - sl_price_pct)
            position.take_profit = position.avg_entry_price * (1 + tp_price_pct)
        else:
            position.stop_loss = position.avg_entry_price * (1 + sl_price_pct)
            position.take_profit = position.avg_entry_price * (1 - tp_price_pct)

        # Apply commission
        commission = fill_price * dca_qty * PAPER_TRADING_CONFIG["commission_per_trade"]
        self.balance -= commission

        budget = self.symbol_budgets.get(symbol, 0)
        remaining = self.get_remaining_budget(symbol)
        self.log(
            f"DCA {dca_level} {symbol} @ ${fill_price:,.2f} | "
            f"Added: {dca_qty:.6f} | Margin: +${dca_margin:.2f} (${position.margin_used:.2f}/{budget:.2f}) | "
            f"Total Qty: {position.quantity:.6f} | Avg: ${position.avg_entry_price:,.2f}",
            level="DCA"
        )

    def update_trailing_stop(self, position: Position, current_price: float):
        """Update trailing stop if applicable"""
        if not STRATEGY_CONFIG["trailing_stop_enabled"]:
            return

        # Update price extremes
        if position.side == "LONG":
            position.highest_price = max(position.highest_price, current_price)

            # Check if trailing stop should activate
            profit_pct = (position.highest_price - position.avg_entry_price) / position.avg_entry_price
            if profit_pct >= STRATEGY_CONFIG["trailing_stop_trigger"]:
                position.trailing_stop_active = True
                new_trailing = position.highest_price * (1 - STRATEGY_CONFIG["trailing_stop_distance"])
                position.trailing_stop_price = max(
                    position.trailing_stop_price or 0,
                    new_trailing
                )
        else:  # SHORT
            position.lowest_price = min(position.lowest_price, current_price)

            profit_pct = (position.avg_entry_price - position.lowest_price) / position.avg_entry_price
            if profit_pct >= STRATEGY_CONFIG["trailing_stop_trigger"]:
                position.trailing_stop_active = True
                new_trailing = position.lowest_price * (1 + STRATEGY_CONFIG["trailing_stop_distance"])
                position.trailing_stop_price = min(
                    position.trailing_stop_price or float("inf"),
                    new_trailing
                )

    def check_exit(self, position: Position, current_price: float) -> Optional[str]:
        """Check if position should exit"""
        if position.side == "LONG":
            # Check take profit
            if current_price >= position.take_profit:
                return "TAKE_PROFIT"

            # Check stop loss
            if current_price <= position.stop_loss:
                return "STOP_LOSS"

            # Check trailing stop
            if position.trailing_stop_active and position.trailing_stop_price:
                if current_price <= position.trailing_stop_price:
                    return "TRAILING_STOP"
        else:  # SHORT
            # Check take profit
            if current_price <= position.take_profit:
                return "TAKE_PROFIT"

            # Check stop loss
            if current_price >= position.stop_loss:
                return "STOP_LOSS"

            # Check trailing stop
            if position.trailing_stop_active and position.trailing_stop_price:
                if current_price >= position.trailing_stop_price:
                    return "TRAILING_STOP"

        return None

    def close_position(self, symbol: str, current_price: float, reason: str):
        """Close a position and release margin"""
        position = self.positions[symbol]

        # Apply slippage
        slippage = PAPER_TRADING_CONFIG["slippage_pct"]
        if position.side == "LONG":
            exit_price = current_price * (1 - slippage)
        else:
            exit_price = current_price * (1 + slippage)

        # Calculate P&L based on position value
        if position.side == "LONG":
            pnl = (exit_price - position.avg_entry_price) * position.quantity
        else:
            pnl = (position.avg_entry_price - exit_price) * position.quantity

        # P&L percentage based on margin used (actual risk)
        pnl_pct = pnl / position.margin_used if position.margin_used > 0 else 0

        # Apply exit commission
        commission = exit_price * position.quantity * PAPER_TRADING_CONFIG["commission_per_trade"]
        pnl -= commission

        # Update balance: return margin + P&L
        self.balance += position.margin_used + pnl
        self.daily_pnl += pnl

        # Release symbol margin
        self.symbol_margin_used[symbol] = 0.0

        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=position.avg_entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            dca_count=position.dca_count
        )
        self.trades.append(trade)

        # Update statistics
        self.stats["total_trades"] += 1
        self.stats["total_pnl"] += pnl
        if pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"] += 1

        # Update peak and drawdown
        if self.balance > self.stats["peak_balance"]:
            self.stats["peak_balance"] = self.balance
        drawdown = (self.stats["peak_balance"] - self.balance) / self.stats["peak_balance"]
        self.stats["max_drawdown"] = max(self.stats["max_drawdown"], drawdown)

        # Remove position
        del self.positions[symbol]

        pnl_emoji = "+" if pnl > 0 else ""
        self.log(
            f"EXIT {position.side} {symbol} @ ${exit_price:,.2f} | "
            f"P&L: {pnl_emoji}${pnl:.2f} ({pnl_emoji}{pnl_pct:.2%} on margin) | "
            f"Margin: ${position.margin_used:.2f} | Reason: {reason} | DCAs: {position.dca_count}",
            level="TRADE"
        )

    def manage_positions(self):
        """Check and manage all open positions"""
        for symbol in list(self.positions.keys()):
            try:
                # Get current price
                price_data = self.client.get_current_price(symbol)
                current_price = price_data["price"]

                position = self.positions[symbol]

                # Check for DCA
                if self.check_dca(position, current_price):
                    # Refresh market data for SMART DCA validation
                    if symbol not in self.data_buffer or self.data_buffer[symbol] is None:
                        try:
                            market_data = self.client.get_market_data(symbol)
                            if market_data and "1m" in market_data:
                                self.data_buffer[symbol] = market_data
                        except:
                            pass  # Continue with DCA check even without fresh data

                    self.execute_dca(symbol, current_price)

                # Update trailing stop
                self.update_trailing_stop(position, current_price)

                # Check for exit
                exit_reason = self.check_exit(position, current_price)
                if exit_reason:
                    self.close_position(symbol, current_price, exit_reason)

            except Exception as e:
                self.log(f"Error managing position {symbol}: {e}", level="ERROR")

    def check_entry_signals(self):
        """Check for entry signals on all symbols"""
        for symbol in FUTURES_SYMBOLS:
            try:
                # Check if can trade
                if not self.can_trade(symbol):
                    continue

                # Update check time
                self.last_check_time[symbol] = datetime.now()

                # Get market data
                market_data = self.client.get_market_data(symbol)

                if not market_data or "1m" not in market_data:
                    continue

                # Store market data for SMART DCA validation
                self.data_buffer[symbol] = market_data

                # Generate signal
                if self.use_mtf:
                    signal = self.signal_generator.generate_signal_mtf(symbol, market_data)
                else:
                    signal = self.signal_generator.generate_signal(symbol, market_data["1m"])

                # Check for valid signal
                if signal.signal is not None:
                    current_price = self.client.get_current_price(symbol)["price"]
                    self.enter_position(symbol, signal, current_price)

            except Exception as e:
                self.log(f"Error checking signals for {symbol}: {e}", level="ERROR")

    def print_status(self):
        """Print current status with margin allocation info"""
        print("\n" + "="*80)
        print(f"BINANCE FUTURES PAPER TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        print(f"\nBalance: ${self.balance:,.2f} (Start: ${self.initial_balance:,.2f})")
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Daily P&L: ${self.daily_pnl:+.2f}")
        print(f"Daily Trades: {self.daily_trades}/{STRATEGY_CONFIG['max_trades_per_day']}")
        print(f"Leverage: {self.leverage}x (ISOLATED)")

        # Show margin allocation summary
        total_margin_used = sum(self.symbol_margin_used.values())
        total_budget = sum(self.symbol_budgets.values())
        print(f"\nMargin Allocation: ${total_margin_used:,.2f} / ${total_budget:,.2f} used")

        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                try:
                    price_data = self.client.get_current_price(symbol)
                    current_price = price_data["price"]

                    if pos.side == "LONG":
                        unrealized_pnl = (current_price - pos.avg_entry_price) * pos.quantity
                    else:
                        unrealized_pnl = (pos.avg_entry_price - current_price) * pos.quantity

                    pnl_pct = unrealized_pnl / pos.margin_used * 100 if pos.margin_used > 0 else 0
                    budget = self.symbol_budgets.get(symbol, 0)

                    print(f"  {symbol} {pos.side}: Entry ${pos.avg_entry_price:,.2f} | "
                          f"Current ${current_price:,.2f} | P&L: ${unrealized_pnl:+.2f} ({pnl_pct:+.2f}%) | "
                          f"Margin: ${pos.margin_used:.2f}/{budget:.2f} | DCAs: {pos.dca_count}")
                except:
                    print(f"  {symbol} {pos.side}: Entry ${pos.avg_entry_price:,.2f} | Margin: ${pos.margin_used:.2f}")
        else:
            print("\nNo open positions")

        if self.stats["total_trades"] > 0:
            win_rate = self.stats["winning_trades"] / self.stats["total_trades"] * 100
            print(f"\nStatistics:")
            print(f"  Total Trades: {self.stats['total_trades']}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Max Drawdown: {self.stats['max_drawdown']:.2%}")

        print("="*80 + "\n")

    def save_trades(self):
        """Save trade history to file"""
        if not self.trades:
            return

        trade_file = os.path.join(self.log_dir, LOGGING_CONFIG["trade_log_file"])

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "exit_reason": trade.exit_reason,
                "dca_count": trade.dca_count
            })

        with open(trade_file, "w") as f:
            json.dump({
                "trades": trades_data,
                "statistics": self.stats,
                "final_balance": self.balance,
                "initial_balance": self.initial_balance
            }, f, indent=2)

        self.log(f"Trades saved to {trade_file}")

    def run(self, duration_hours: float = 0):
        """
        Run paper trading for specified duration

        Args:
            duration_hours: How long to run (0 = run continuously)
        """
        self.running = True
        start_time = datetime.now()
        continuous = duration_hours == 0
        end_time = None if continuous else start_time + timedelta(hours=duration_hours)

        duration_str = "CONTINUOUSLY (24/7)" if continuous else f"for {duration_hours} hours"
        self.log(f"Starting paper trading {duration_str}")
        self.log(f"Trading symbols: {', '.join(FUTURES_SYMBOLS)}")
        self.log(f"Initial balance: ${self.balance:,.2f}")
        self.log(f"Using {'Multi-Timeframe' if self.use_mtf else 'Single Timeframe'} signals")

        # Test connection
        if not self.client.test_connection():
            self.log("Failed to connect to Binance. Exiting.", level="ERROR")
            return

        # Initialize dynamic fund allocation
        self.initialize_dynamic_allocation()

        check_interval = PAPER_TRADING_CONFIG["check_interval"]
        status_interval = 300  # Print status every 5 minutes
        last_status_time = datetime.now()

        try:
            while self.running and (continuous or datetime.now() < end_time):
                # Check for daily reset
                self.check_daily_reset()

                # Check entry signals
                self.check_entry_signals()

                # Manage open positions
                self.manage_positions()

                # Print status periodically
                if (datetime.now() - last_status_time).total_seconds() >= status_interval:
                    self.print_status()
                    last_status_time = datetime.now()

                # Wait before next check
                time.sleep(check_interval)

        except KeyboardInterrupt:
            self.log("Interrupted by user")
        except Exception as e:
            self.log(f"Error: {e}", level="ERROR")
        finally:
            self.running = False
            self.print_status()
            self.save_trades()
            self.log("Paper trading stopped")

    def stop(self):
        """Stop paper trading"""
        self.running = False


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Futures Paper Trading")
    parser.add_argument("--hours", type=float, default=24, help="Duration in hours")
    parser.add_argument("--no-mtf", action="store_true", help="Disable multi-timeframe")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if not args.yes:
        print("\n" + "="*60)
        print("BINANCE FUTURES PAPER TRADING")
        print("="*60)
        print(f"Duration: {args.hours} hours")
        print(f"Mode: {'Single Timeframe' if args.no_mtf else 'Multi-Timeframe'}")
        print(f"Symbols: {', '.join(FUTURES_SYMBOLS)}")
        print(f"Initial Balance: ${PAPER_TRADING_CONFIG['initial_balance']:,.2f}")
        print("="*60)

        confirm = input("\nStart paper trading? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled")
            sys.exit(0)

    engine = BinancePaperTradingEngine(use_mtf=not args.no_mtf)
    engine.run(duration_hours=args.hours)
