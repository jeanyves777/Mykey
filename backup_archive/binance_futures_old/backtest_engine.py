"""
Binance Futures Backtest Engine
===============================
Historical backtesting with real Binance data
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import (
    FUTURES_SYMBOLS, STRATEGY_CONFIG, RISK_CONFIG, DCA_CONFIG,
    BACKTEST_CONFIG, LOGGING_CONFIG, MOMENTUM_CONFIG
)
from engine.binance_client import BinanceClient
from engine.momentum_signal import MasterMomentumSignal, TradingSignal


@dataclass
class BacktestPosition:
    """Position during backtest"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop_price: Optional[float] = None
    trailing_stop_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = float("inf")
    dca_count: int = 0
    avg_entry_price: float = 0.0
    total_cost: float = 0.0
    margin_used: float = 0.0       # Total margin used for this symbol


@dataclass
class BacktestTrade:
    """Completed trade record"""
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
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Backtest results"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    final_balance: float = 0.0
    initial_balance: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BinanceBacktestEngine:
    """
    Backtesting Engine for Binance Futures

    Simulates trading on historical data to evaluate strategy performance.
    """

    def __init__(self, symbols: List[str] = None):
        """
        Initialize backtest engine

        Args:
            symbols: List of symbols to backtest (default: all from config)
        """
        self.client = BinanceClient(testnet=True)
        self.signal_generator = MasterMomentumSignal()

        self.symbols = symbols or FUTURES_SYMBOLS

        # Backtest state
        self.balance = BACKTEST_CONFIG["initial_balance"]
        self.initial_balance = self.balance
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []

        # Dynamic Fund Allocation
        self.num_symbols = len(self.symbols)
        self.symbol_budgets: Dict[str, float] = {}      # Max budget per symbol
        self.symbol_margin_used: Dict[str, float] = {}  # Current margin used per symbol
        self.leverage = STRATEGY_CONFIG["leverage"]      # Fixed 20x leverage

        # Tracking
        self.equity_curve = []
        self.daily_returns = []
        self.peak_balance = self.balance
        self.max_drawdown = 0.0

        # Daily counters
        self.daily_trades = 0
        self.current_date = None

        # Funding fee tracking (every 8 hours: 00:00, 08:00, 16:00 UTC)
        self.last_funding_hour = None
        self.total_funding_fees = 0.0
        self.funding_rate = 0.0001  # 0.01% default funding rate (typical range: -0.01% to +0.03%)

        # Liquidation tracking
        self.liquidations = 0
        self.liquidation_losses = 0.0

    def fetch_historical_data(self, symbol: str, start_date: str,
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for a symbol

        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict with DataFrames for each timeframe
        """
        print(f"Fetching historical data for {symbol}...")

        data = {}

        # Fetch 1-minute data (primary timeframe)
        data["1m"] = self.client.get_historical_klines(
            symbol, "1m", start_date, end_date
        )

        # Fetch other timeframes
        data["5m"] = self.client.get_historical_klines(
            symbol, "5m", start_date, end_date
        )

        data["15m"] = self.client.get_historical_klines(
            symbol, "15m", start_date, end_date
        )

        data["1h"] = self.client.get_historical_klines(
            symbol, "1h", start_date, end_date
        )

        # Print data info
        for tf, df in data.items():
            if not df.empty:
                print(f"  {tf}: {len(df)} candles ({df.index[0]} to {df.index[-1]})")

        return data

    def initialize_dynamic_allocation(self):
        """Initialize dynamic fund allocation based on balance."""
        buffer = RISK_CONFIG.get("allocation_buffer_pct", 0.05)
        available_balance = self.balance * (1 - buffer)
        budget_per_symbol = available_balance / self.num_symbols

        print(f"Dynamic Allocation: ${self.balance:.2f} balance")
        print(f"  Budget per symbol: ${budget_per_symbol:.2f}")
        print(f"  Leverage: {self.leverage}x (ISOLATED)")

        for symbol in self.symbols:
            self.symbol_budgets[symbol] = budget_per_symbol
            self.symbol_margin_used[symbol] = 0.0

        return budget_per_symbol

    def get_entry_margin(self, symbol: str) -> float:
        """Get margin for initial entry (20% of symbol budget)."""
        if symbol not in self.symbol_budgets:
            return 0.0
        budget = self.symbol_budgets[symbol]
        entry_pct = RISK_CONFIG.get("initial_entry_pct", 0.20)
        return budget * entry_pct

    def get_dca_margin(self, symbol: str, dca_level: int) -> float:
        """Get margin for DCA level (1-4)."""
        if symbol not in self.symbol_budgets:
            return 0.0
        budget = self.symbol_budgets[symbol]
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

    def calculate_position_size(self, symbol: str, price: float, is_dca: bool = False, dca_level: int = 0) -> float:
        """Calculate position size based on dynamic allocation."""
        if is_dca and dca_level > 0:
            margin = self.get_dca_margin(symbol, dca_level)
        else:
            margin = self.get_entry_margin(symbol)

        remaining = self.get_remaining_budget(symbol)
        if margin > remaining:
            margin = remaining

        if margin <= 0:
            return 0.0

        position_value = margin * self.leverage
        return position_value / price

    def calculate_sl_tp(self, entry_price: float, side: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit using ROI-based calculation for leveraged futures"""
        # ROI-based calculation: price_move = roi / leverage
        leverage = STRATEGY_CONFIG["leverage"]  # 20x

        if DCA_CONFIG["enabled"]:
            # ROI-based: 3% ROI TP, 60% ROI SL (below liquidation ~91% ROI)
            tp_roi = DCA_CONFIG["take_profit_roi"]    # 3% ROI
            sl_roi = DCA_CONFIG["stop_loss_roi"]      # 60% ROI
            tp_price_pct = tp_roi / leverage          # 3% / 20 = 0.15% price move
            sl_price_pct = sl_roi / leverage          # 60% / 20 = 3% price move
        else:
            # Price-based for non-DCA (legacy)
            tp_price_pct = STRATEGY_CONFIG["take_profit_pct"]
            sl_price_pct = STRATEGY_CONFIG["stop_loss_pct"]

        if side == "LONG":
            sl = entry_price * (1 - sl_price_pct)
            tp = entry_price * (1 + tp_price_pct)
        else:
            sl = entry_price * (1 + sl_price_pct)
            tp = entry_price * (1 - tp_price_pct)
        return sl, tp

    def enter_position(self, symbol: str, signal: TradingSignal,
                       current_bar: pd.Series, bar_time: datetime):
        """Enter a position during backtest with dynamic margin allocation"""
        side = "LONG" if signal.signal == "BUY" else "SHORT"

        # Get margin for initial entry
        margin = self.get_entry_margin(symbol)
        remaining = self.get_remaining_budget(symbol)

        if margin > remaining or margin <= 0:
            return  # Can't afford entry

        # Use close price with slippage
        slippage = BACKTEST_CONFIG["slippage_pct"]
        if side == "LONG":
            entry_price = current_bar["close"] * (1 + slippage)
        else:
            entry_price = current_bar["close"] * (1 - slippage)

        quantity = self.calculate_position_size(symbol, entry_price)

        if quantity <= 0:
            return  # Position too small

        sl, tp = self.calculate_sl_tp(entry_price, side)

        # Commission
        commission = entry_price * quantity * BACKTEST_CONFIG["commission_per_trade"]
        self.balance -= commission

        # DEDUCT MARGIN from balance (will be returned on close)
        self.balance -= margin

        # Track margin used
        self.symbol_margin_used[symbol] = margin

        # Position value = margin * leverage
        position_value = margin * self.leverage

        position = BacktestPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=bar_time,
            stop_loss=sl,
            take_profit=tp,
            highest_price=entry_price,
            lowest_price=entry_price,
            avg_entry_price=entry_price,
            total_cost=position_value,
            margin_used=margin
        )

        self.positions[symbol] = position
        self.daily_trades += 1

    def check_exit(self, position: BacktestPosition,
                   current_bar: pd.Series) -> Optional[Tuple[str, float]]:
        """Check if position should exit based on bar data"""
        high = current_bar["high"]
        low = current_bar["low"]
        close = current_bar["close"]

        if position.side == "LONG":
            # Check if SL hit (use low)
            if low <= position.stop_loss:
                return "STOP_LOSS", position.stop_loss

            # Check if TP hit (use high)
            if high >= position.take_profit:
                return "TAKE_PROFIT", position.take_profit

            # Update trailing stop
            if STRATEGY_CONFIG["trailing_stop_enabled"]:
                position.highest_price = max(position.highest_price, high)
                profit_pct = (position.highest_price - position.avg_entry_price) / position.avg_entry_price

                if profit_pct >= STRATEGY_CONFIG["trailing_stop_trigger"]:
                    position.trailing_stop_active = True
                    new_trail = position.highest_price * (1 - STRATEGY_CONFIG["trailing_stop_distance"])
                    position.trailing_stop_price = max(
                        position.trailing_stop_price or 0,
                        new_trail
                    )

                    if position.trailing_stop_price and low <= position.trailing_stop_price:
                        return "TRAILING_STOP", position.trailing_stop_price

        else:  # SHORT
            # Check if SL hit (use high)
            if high >= position.stop_loss:
                return "STOP_LOSS", position.stop_loss

            # Check if TP hit (use low)
            if low <= position.take_profit:
                return "TAKE_PROFIT", position.take_profit

            # Update trailing stop
            if STRATEGY_CONFIG["trailing_stop_enabled"]:
                position.lowest_price = min(position.lowest_price, low)
                profit_pct = (position.avg_entry_price - position.lowest_price) / position.avg_entry_price

                if profit_pct >= STRATEGY_CONFIG["trailing_stop_trigger"]:
                    position.trailing_stop_active = True
                    new_trail = position.lowest_price * (1 + STRATEGY_CONFIG["trailing_stop_distance"])
                    position.trailing_stop_price = min(
                        position.trailing_stop_price or float("inf"),
                        new_trail
                    )

                    if position.trailing_stop_price and high >= position.trailing_stop_price:
                        return "TRAILING_STOP", position.trailing_stop_price

        return None

    def check_dca(self, position: BacktestPosition, current_price: float) -> bool:
        """Check if DCA should trigger (ROI-based)"""
        if not DCA_CONFIG["enabled"]:
            return False

        if position.dca_count >= len(DCA_CONFIG["levels"]):
            return False

        dca_level = position.dca_count + 1
        if not self.can_afford_dca(position.symbol, dca_level):
            return False

        # Calculate current ROI (for leveraged positions)
        leverage = STRATEGY_CONFIG["leverage"]  # 20x
        if position.side == "LONG":
            price_drawdown = (position.avg_entry_price - current_price) / position.avg_entry_price
        else:
            price_drawdown = (current_price - position.avg_entry_price) / position.avg_entry_price

        # Convert to ROI: ROI = price_change * leverage
        current_roi_loss = price_drawdown * leverage

        # Check if DCA level triggered (ROI-based)
        level = DCA_CONFIG["levels"][position.dca_count]
        trigger_roi = abs(level["trigger_roi"])

        return current_roi_loss >= trigger_roi

    def check_liquidation(self, position: BacktestPosition, current_price: float) -> bool:
        """
        Check if position should be liquidated.
        With 20x leverage and ISOLATED margin, liquidation happens at ~4.5% price move against position.
        """
        leverage = STRATEGY_CONFIG["leverage"]
        # Liquidation threshold is approximately when ROI loss reaches ~95% (leaving some buffer for fees)
        # With 20x: 95% ROI / 20 = 4.75% price move
        liq_threshold = 0.95 / leverage  # ~4.75% price move

        if position.side == "LONG":
            price_drop = (position.avg_entry_price - current_price) / position.avg_entry_price
            return price_drop >= liq_threshold
        else:  # SHORT
            price_rise = (current_price - position.avg_entry_price) / position.avg_entry_price
            return price_rise >= liq_threshold

    def apply_funding_fee(self, bar_time: datetime, position: BacktestPosition, current_price: float):
        """
        Apply funding fee to position.
        Funding is paid/received every 8 hours (00:00, 08:00, 16:00 UTC).
        Long positions PAY funding when rate is positive (typical).
        Short positions RECEIVE funding when rate is positive.
        """
        hour = bar_time.hour
        funding_hours = [0, 8, 16]

        if hour in funding_hours and self.last_funding_hour != hour:
            self.last_funding_hour = hour

            # Position notional value
            position_value = position.quantity * current_price

            # Funding fee = position_value * funding_rate
            # Positive rate: Longs pay shorts
            # Negative rate: Shorts pay longs
            funding_fee = position_value * self.funding_rate

            if position.side == "LONG":
                # Longs PAY when rate is positive
                self.balance -= funding_fee
                self.total_funding_fees += funding_fee
            else:
                # Shorts RECEIVE when rate is positive
                self.balance += funding_fee
                self.total_funding_fees -= funding_fee  # Negative = received

    def execute_dca(self, position: BacktestPosition, current_price: float,
                    lookback_df: pd.DataFrame = None):
        """Execute DCA during backtest with margin-based allocation and SMART DCA validation"""
        dca_level = position.dca_count + 1

        # =================================================================
        # SMART DCA: Validate signal/trend before executing
        # =================================================================
        if lookback_df is not None and len(lookback_df) >= 30:
            # Use signal generator to check if DCA is safe
            can_dca, reason = self.signal_generator.can_dca(lookback_df, position.side, dca_level)

            if not can_dca:
                # Trend too strong against us - wait for reversal
                # (In backtest, we just skip this bar and will check again next bar)
                return

        # Get DCA margin for this level
        dca_margin = self.get_dca_margin(position.symbol, dca_level)

        # Calculate DCA quantity based on margin + leverage
        dca_qty = self.calculate_position_size(position.symbol, current_price, is_dca=True, dca_level=dca_level)

        if dca_qty <= 0:
            return

        # Apply slippage
        slippage = BACKTEST_CONFIG["slippage_pct"]
        if position.side == "LONG":
            fill_price = current_price * (1 + slippage)
        else:
            fill_price = current_price * (1 - slippage)

        # Update margin tracking
        self.symbol_margin_used[position.symbol] += dca_margin
        position.margin_used += dca_margin

        # DEDUCT DCA MARGIN from balance (will be returned on close)
        self.balance -= dca_margin

        # Calculate DCA position value
        dca_position_value = dca_margin * self.leverage

        # Update position
        old_cost = position.total_cost
        position.total_cost = old_cost + dca_position_value
        position.quantity += dca_qty
        position.avg_entry_price = position.total_cost / position.quantity
        position.dca_count = dca_level

        # Tighten SL and REDUCE TP after DCA (ROI-based)
        # REDUCED TP: Exit faster to reduce exposure on DCA positions
        leverage = STRATEGY_CONFIG["leverage"]
        sl_roi_after_dca = DCA_CONFIG["sl_after_dca_roi"]  # 20% ROI SL
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

        # Commission
        commission = fill_price * dca_qty * BACKTEST_CONFIG["commission_per_trade"]
        self.balance -= commission

    def close_position(self, symbol: str, exit_price: float,
                       exit_reason: str, exit_time: datetime, bars_held: int):
        """Close position and release margin"""
        position = self.positions[symbol]

        # Apply slippage
        slippage = BACKTEST_CONFIG["slippage_pct"]
        if position.side == "LONG":
            exit_price = exit_price * (1 - slippage)
            pnl = (exit_price - position.avg_entry_price) * position.quantity
        else:
            exit_price = exit_price * (1 + slippage)
            pnl = (position.avg_entry_price - exit_price) * position.quantity

        # P&L percentage based on margin used
        pnl_pct = pnl / position.margin_used if position.margin_used > 0 else 0

        # Commission
        commission = exit_price * position.quantity * BACKTEST_CONFIG["commission_per_trade"]
        pnl -= commission

        # Update balance: return margin + P&L
        self.balance += position.margin_used + pnl

        # Release symbol margin
        self.symbol_margin_used[symbol] = 0.0

        # Record trade
        trade = BacktestTrade(
            symbol=symbol,
            side=position.side,
            entry_price=position.avg_entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            dca_count=position.dca_count,
            bars_held=bars_held
        )
        self.trades.append(trade)

        # Update tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

        del self.positions[symbol]

    def can_trade(self, symbol: str) -> bool:
        """Check if can enter trade"""
        if self.daily_trades >= STRATEGY_CONFIG["max_trades_per_day"]:
            return False

        if len(self.positions) >= RISK_CONFIG["max_total_positions"]:
            return False

        if symbol in self.positions:
            return False

        return True

    def run_backtest(self, start_date: str, end_date: str,
                     symbol: str = None) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Single symbol to backtest (default: all)

        Returns:
            BacktestResult with performance metrics
        """
        symbols = [symbol] if symbol else self.symbols

        print("\n" + "="*60)
        print("BINANCE FUTURES BACKTEST")
        print("="*60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print("="*60)

        # Fetch all data first
        all_data = {}
        for sym in symbols:
            all_data[sym] = self.fetch_historical_data(sym, start_date, end_date)

        # Get primary timeframe data for iteration
        primary_data = {}
        for sym in symbols:
            if "1m" in all_data[sym] and not all_data[sym]["1m"].empty:
                primary_data[sym] = all_data[sym]["1m"]

        if not primary_data:
            print("No data available for backtesting!")
            return BacktestResult()

        # Find common date range
        all_dates = set()
        for df in primary_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        print(f"\nBacktesting {len(all_dates)} bars...")

        # Reset state
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.balance]
        self.peak_balance = self.balance
        self.max_drawdown = 0.0

        # Initialize dynamic fund allocation
        self.initialize_dynamic_allocation()

        # Track position entry bars for holding period
        entry_bars = {}

        # Iterate through each bar
        for bar_idx, bar_time in enumerate(all_dates):
            # Check for daily reset
            bar_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time.to_pydatetime().date()
            if bar_date != self.current_date:
                self.current_date = bar_date
                self.daily_trades = 0

            # Process each symbol
            for sym in symbols:
                if sym not in primary_data:
                    continue

                df = primary_data[sym]
                if bar_time not in df.index:
                    continue

                current_bar = df.loc[bar_time]

                # Check open positions first
                if sym in self.positions:
                    position = self.positions[sym]
                    current_price = current_bar["close"]

                    # ============================================
                    # 1. CHECK LIQUIDATION FIRST (highest priority)
                    # ============================================
                    if self.check_liquidation(position, current_price):
                        bars_held = bar_idx - entry_bars.get(sym, bar_idx)
                        # Liquidation = lose entire margin
                        liq_loss = position.margin_used
                        self.liquidations += 1
                        self.liquidation_losses += liq_loss
                        self.close_position(sym, current_price, "LIQUIDATION", bar_time, bars_held)
                        if sym in entry_bars:
                            del entry_bars[sym]
                        continue

                    # ============================================
                    # 2. APPLY FUNDING FEE (every 8 hours)
                    # ============================================
                    self.apply_funding_fee(bar_time, position, current_price)

                    # Build lookback data for SMART DCA validation
                    lookback_start = max(0, bar_idx - 100)
                    lookback_df = df.iloc[lookback_start:bar_idx+1].copy()

                    # Check DCA (with SMART DCA validation using lookback data)
                    if self.check_dca(position, current_price):
                        self.execute_dca(position, current_price, lookback_df)

                    # Check exit
                    exit_result = self.check_exit(position, current_bar)
                    if exit_result:
                        exit_reason, exit_price = exit_result
                        bars_held = bar_idx - entry_bars.get(sym, bar_idx)
                        self.close_position(sym, exit_price, exit_reason, bar_time, bars_held)
                        if sym in entry_bars:
                            del entry_bars[sym]

                # Check for new entry (only if can trade)
                elif self.can_trade(sym):
                    # Check cooldown using actual bar index (not lookback index)
                    if not self.signal_generator.check_cooldown(sym, bar_idx):
                        continue  # Skip if in cooldown

                    # Build lookback data for signal
                    lookback_start = max(0, bar_idx - 100)
                    lookback_df = df.iloc[lookback_start:bar_idx+1].copy()

                    if len(lookback_df) >= 30:
                        signal = self.signal_generator.generate_signal_no_cooldown(sym, lookback_df)

                        if signal.signal is not None:
                            self.enter_position(sym, signal, current_bar, bar_time)
                            entry_bars[sym] = bar_idx
                            # Set bar-based cooldown for backtest
                            self.signal_generator.set_cooldown_bar(sym, bar_idx)

            # Update equity curve
            self.equity_curve.append(self.balance)

            # Progress update
            if bar_idx % 10000 == 0:
                print(f"  Processed {bar_idx}/{len(all_dates)} bars... Balance: ${self.balance:,.2f}")

        # Close any remaining positions at end
        for sym in list(self.positions.keys()):
            if sym in primary_data:
                last_bar = primary_data[sym].iloc[-1]
                bars_held = len(all_dates) - entry_bars.get(sym, len(all_dates))
                self.close_position(sym, last_bar["close"], "END_OF_BACKTEST",
                                   all_dates[-1], bars_held)

        # Calculate results
        result = self._calculate_results()

        # Print summary
        self._print_summary(result)

        return result

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest metrics"""
        result = BacktestResult()
        result.initial_balance = self.initial_balance
        result.final_balance = self.balance
        result.trades = self.trades
        result.equity_curve = self.equity_curve

        result.total_trades = len(self.trades)

        if result.total_trades == 0:
            return result

        # Win/Loss counts
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        result.winning_trades = len(winners)
        result.losing_trades = len(losers)

        # P&L
        result.total_pnl = sum(t.pnl for t in self.trades)
        result.total_pnl_pct = (self.balance - self.initial_balance) / self.initial_balance

        # Win rate
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        # Average win/loss
        if winners:
            result.avg_win = sum(t.pnl for t in winners) / len(winners)
        if losers:
            result.avg_loss = abs(sum(t.pnl for t in losers) / len(losers))

        # Profit factor
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Drawdown
        result.max_drawdown = self.max_drawdown
        result.max_drawdown_pct = self.max_drawdown

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)

        return result

    def _print_summary(self, result: BacktestResult):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        print(f"\nCapital:")
        print(f"  Initial: ${result.initial_balance:,.2f}")
        print(f"  Final:   ${result.final_balance:,.2f}")
        print(f"  P&L:     ${result.total_pnl:+,.2f} ({result.total_pnl_pct:+.2%})")

        print(f"\nTrades:")
        print(f"  Total:   {result.total_trades}")
        print(f"  Winners: {result.winning_trades} ({result.win_rate:.1%})")
        print(f"  Losers:  {result.losing_trades}")

        if result.avg_win > 0 or result.avg_loss > 0:
            print(f"\nAverage:")
            print(f"  Avg Win:  ${result.avg_win:,.2f}")
            print(f"  Avg Loss: ${result.avg_loss:,.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown:  {result.max_drawdown_pct:.2%}")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")

        # Funding fees and liquidations
        print(f"\nFees & Liquidations:")
        funding_sign = "-" if self.total_funding_fees > 0 else "+"
        print(f"  Funding Fees:  {funding_sign}${abs(self.total_funding_fees):,.2f}")
        print(f"  Liquidations:  {self.liquidations} (${self.liquidation_losses:,.2f} lost)")

        print("="*60 + "\n")

    def save_results(self, result: BacktestResult, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Binance_Futures_Trading",
            LOGGING_CONFIG["log_dir"],
            filename
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "summary": {
                "initial_balance": result.initial_balance,
                "final_balance": result.final_balance,
                "total_pnl": result.total_pnl,
                "total_pnl_pct": result.total_pnl_pct,
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "dca_count": t.dca_count,
                    "bars_held": t.bars_held
                }
                for t in result.trades
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Futures Backtest")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save results")

    args = parser.parse_args()

    # Calculate dates
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Run backtest
    engine = BinanceBacktestEngine()
    result = engine.run_backtest(start_date, end_date, symbol=args.symbol)

    if args.save and result.total_trades > 0:
        engine.save_results(result)
