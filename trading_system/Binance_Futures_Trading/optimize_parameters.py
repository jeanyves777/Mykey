#!/usr/bin/env python3
"""
Parameter Optimization System
==============================
Find the most profitable settings per asset using backtesting.

Features:
- Grid search over parameter combinations
- Per-asset optimization (BTC, ETH, BNB each get optimal settings)
- TRADING SESSION ANALYSIS (Asian, Tokyo, London, New York)
- Per-session profitability tracking
- Results tracking and export
- Best parameters auto-save to config

Usage:
    python optimize_parameters.py                    # Optimize all symbols
    python optimize_parameters.py --symbol BTCUSDT   # Single symbol
    python optimize_parameters.py --days 60          # Custom period
    python optimize_parameters.py --quick            # Quick mode (fewer combos)
    python optimize_parameters.py --sessions         # Analyze by trading session
"""

import os
import sys
import json
import time
import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.trading_config import (
    FUTURES_SYMBOLS_LIVE, STRATEGY_CONFIG, RISK_CONFIG, DCA_CONFIG,
    BACKTEST_CONFIG, SYMBOL_SETTINGS
)
from engine.binance_client import BinanceClient
from engine.momentum_signal import MasterMomentumSignal, TradingSignal


# =============================================================================
# PARAMETER SEARCH SPACE
# =============================================================================
# These are the parameters we'll optimize for each asset

PARAMETER_SPACE = {
    # Take Profit ROI (what % ROI to close at)
    "tp_roi": [0.05, 0.06, 0.08, 0.10, 0.12, 0.15],

    # Leverage (affects position size and liquidation risk)
    "leverage": [10, 15, 20, 25],

    # Boost Mode settings
    "boost_trigger_roi": [-0.15, -0.20, -0.25, -0.30],  # When to boost opposite side
    "boost_multiplier": [1.25, 1.5, 1.75, 2.0],         # How much to boost

    # Trailing TP settings
    "trailing_activation_roi": [0.02, 0.03, 0.04, 0.05],
    "trailing_distance_roi": [0.02, 0.03, 0.04, 0.05],

    # Half-close settings
    "half_close_enabled": [True, False],
}

# Quick mode - fewer combinations for faster testing
PARAMETER_SPACE_QUICK = {
    "tp_roi": [0.06, 0.08, 0.10],
    "leverage": [15, 20],
    "boost_trigger_roi": [-0.20, -0.25],
    "boost_multiplier": [1.5, 1.75],
    "trailing_activation_roi": [0.02, 0.04],
    "trailing_distance_roi": [0.03, 0.04],
    "half_close_enabled": [True],
}

# =============================================================================
# TRADING SESSIONS (UTC times)
# =============================================================================
# Each session has different volatility and market dynamics
# Crypto trades 24/7 but follows traditional forex session patterns

TRADING_SESSIONS = {
    "ASIAN": {
        "name": "Asian Session",
        "start_hour": 0,   # 00:00 UTC (Tokyo 09:00, Sydney 11:00)
        "end_hour": 8,     # 08:00 UTC
        "description": "Lower volatility, range-bound trading",
        "characteristics": ["consolidation", "low_volume", "range_trading"],
    },
    "TOKYO": {
        "name": "Tokyo Session",
        "start_hour": 0,   # 00:00 UTC (Tokyo 09:00)
        "end_hour": 6,     # 06:00 UTC (Tokyo 15:00)
        "description": "JPY pairs active, moderate volatility",
        "characteristics": ["yen_pairs", "moderate_volume"],
    },
    "LONDON": {
        "name": "London Session",
        "start_hour": 7,   # 07:00 UTC (London 08:00 summer)
        "end_hour": 16,    # 16:00 UTC (London 17:00 summer)
        "description": "Highest volume, breakouts common",
        "characteristics": ["high_volume", "breakouts", "trend_start"],
    },
    "NEW_YORK": {
        "name": "New York Session",
        "start_hour": 13,  # 13:00 UTC (NY 08:00 EST)
        "end_hour": 22,    # 22:00 UTC (NY 17:00 EST)
        "description": "High volatility, especially during London overlap",
        "characteristics": ["high_volume", "news_events", "reversals"],
    },
    "LONDON_NY_OVERLAP": {
        "name": "London/NY Overlap",
        "start_hour": 13,  # 13:00 UTC
        "end_hour": 16,    # 16:00 UTC
        "description": "HIGHEST volatility period of the day",
        "characteristics": ["extreme_volume", "major_moves", "best_opportunities"],
    },
    "WEEKEND": {
        "name": "Weekend (Low Liquidity)",
        "start_hour": 0,
        "end_hour": 24,
        "days": [5, 6],    # Saturday=5, Sunday=6
        "description": "Low liquidity, avoid trading",
        "characteristics": ["low_liquidity", "erratic_moves", "gaps"],
    },
}

# Session-specific parameter adjustments
SESSION_PARAM_ADJUSTMENTS = {
    "ASIAN": {
        "tp_roi_mult": 0.8,       # Lower TP target (less volatility)
        "leverage_mult": 1.0,     # Same leverage
        "description": "Tighter targets for range-bound market"
    },
    "TOKYO": {
        "tp_roi_mult": 0.8,
        "leverage_mult": 1.0,
        "description": "Conservative during Tokyo"
    },
    "LONDON": {
        "tp_roi_mult": 1.2,       # Higher TP target (more volatility)
        "leverage_mult": 1.0,
        "description": "Wider targets for breakout moves"
    },
    "NEW_YORK": {
        "tp_roi_mult": 1.0,
        "leverage_mult": 1.0,
        "description": "Standard parameters"
    },
    "LONDON_NY_OVERLAP": {
        "tp_roi_mult": 1.5,       # Much higher TP (extreme volatility)
        "leverage_mult": 0.8,     # Slightly lower leverage (higher risk)
        "description": "Wide targets but careful sizing"
    },
}


@dataclass
class SessionResult:
    """Result for a specific trading session"""
    session_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    best_hour: int
    worst_hour: int
    hourly_pnl: Dict[int, float] = field(default_factory=dict)

    def is_profitable(self) -> bool:
        return self.total_pnl > 0 and self.win_rate > 0.5


@dataclass
class OptimizationResult:
    """Result from a single parameter combination test"""
    symbol: str
    params: Dict[str, Any]
    total_return_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    liquidations: int
    avg_trade_pnl: float
    final_balance: float
    session_results: Dict[str, SessionResult] = field(default_factory=dict)

    def score(self) -> float:
        """
        Calculate optimization score.
        Higher is better. Prioritizes:
        1. Return (40%)
        2. Win rate (20%)
        3. Low drawdown (20%)
        4. Profit factor (10%)
        5. Zero liquidations (10%)
        """
        # Normalize components
        return_score = min(self.total_return_pct * 100, 100) / 100  # Cap at 100%
        win_score = self.win_rate
        drawdown_score = max(0, 1 - self.max_drawdown_pct * 2)  # Penalize >50% DD
        pf_score = min(self.profit_factor, 3) / 3  # Cap at 3x
        liq_score = 1.0 if self.liquidations == 0 else 0.5 if self.liquidations < 3 else 0

        # Weighted score
        score = (
            return_score * 0.40 +
            win_score * 0.20 +
            drawdown_score * 0.20 +
            pf_score * 0.10 +
            liq_score * 0.10
        )

        return score


def get_trading_session(bar_time: datetime) -> str:
    """
    Determine which trading session a bar belongs to.

    Args:
        bar_time: The timestamp of the bar (should be UTC)

    Returns:
        Session name (ASIAN, TOKYO, LONDON, NEW_YORK, LONDON_NY_OVERLAP, WEEKEND)
    """
    # Check if weekend first
    day_of_week = bar_time.weekday()  # Monday=0, Sunday=6
    if day_of_week in [5, 6]:  # Saturday or Sunday
        return "WEEKEND"

    hour = bar_time.hour

    # Check overlapping sessions (priority order)
    if 13 <= hour < 16:
        return "LONDON_NY_OVERLAP"  # Highest priority - most volatile
    elif 13 <= hour < 22:
        return "NEW_YORK"
    elif 7 <= hour < 16:
        return "LONDON"
    elif 0 <= hour < 6:
        return "TOKYO"
    elif 0 <= hour < 8:
        return "ASIAN"
    else:
        return "NEW_YORK"  # Default fallback


def get_session_for_hour(hour: int) -> List[str]:
    """Get all sessions active during a specific hour."""
    sessions = []

    if 0 <= hour < 8:
        sessions.append("ASIAN")
    if 0 <= hour < 6:
        sessions.append("TOKYO")
    if 7 <= hour < 16:
        sessions.append("LONDON")
    if 13 <= hour < 22:
        sessions.append("NEW_YORK")
    if 13 <= hour < 16:
        sessions.append("LONDON_NY_OVERLAP")

    return sessions if sessions else ["OFF_HOURS"]


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
    avg_entry_price: float = 0.0
    total_cost: float = 0.0
    margin_used: float = 0.0
    is_boosted: bool = False
    half_closed: bool = False
    entry_session: str = ""  # Track which session the trade was entered in


class ParameterOptimizer:
    """
    Optimizes trading parameters for each symbol.
    Uses backtesting to find the best settings.
    """

    def __init__(self, symbols: List[str] = None, days: int = 60, quick_mode: bool = False):
        """
        Initialize optimizer.

        Args:
            symbols: Symbols to optimize (default: BTC, ETH, BNB)
            days: Days of historical data for backtesting
            quick_mode: Use reduced parameter space for faster testing
        """
        self.symbols = symbols or FUTURES_SYMBOLS_LIVE
        self.days = days
        self.quick_mode = quick_mode
        self.param_space = PARAMETER_SPACE_QUICK if quick_mode else PARAMETER_SPACE

        self.client = BinanceClient(testnet=True)
        self.signal_generator = MasterMomentumSignal()

        # Results storage
        self.results: Dict[str, List[OptimizationResult]] = {s: [] for s in self.symbols}
        self.best_params: Dict[str, Dict[str, Any]] = {}

        # Data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Calculate total combinations
        self.total_combinations = 1
        for values in self.param_space.values():
            self.total_combinations *= len(values)

        print(f"\n{'='*70}")
        print("PARAMETER OPTIMIZATION SYSTEM")
        print(f"{'='*70}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Days: {self.days}")
        print(f"Mode: {'Quick' if quick_mode else 'Full'}")
        print(f"Total combinations per symbol: {self.total_combinations}")
        print(f"{'='*70}\n")

    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol."""
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        print(f"Fetching {self.days} days of data for {symbol}...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")

        try:
            df = self.client.get_historical_klines(symbol, "1m", start_date, end_date)
            if df is not None and not df.empty:
                self.data_cache[symbol] = df
                print(f"  {symbol}: {len(df)} candles loaded")
                return df
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")

        return None

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test."""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def run_single_backtest(self, symbol: str, params: Dict[str, Any],
                            df: pd.DataFrame) -> OptimizationResult:
        """
        Run a single backtest with specific parameters.

        Args:
            symbol: Trading symbol
            params: Parameter combination to test
            df: Historical data

        Returns:
            OptimizationResult with performance metrics
        """
        # Extract parameters
        tp_roi = params["tp_roi"]
        leverage = params["leverage"]
        boost_trigger_roi = params["boost_trigger_roi"]
        boost_multiplier = params["boost_multiplier"]
        trailing_activation_roi = params["trailing_activation_roi"]
        trailing_distance_roi = params["trailing_distance_roi"]
        half_close_enabled = params["half_close_enabled"]

        # Initialize backtest state
        initial_balance = BACKTEST_CONFIG["initial_balance"]
        balance = initial_balance
        positions: Dict[str, BacktestPosition] = {}  # key = "LONG" or "SHORT"
        trades = []

        # Tracking
        peak_balance = balance
        max_drawdown = 0.0
        liquidations = 0
        equity_curve = [balance]

        # Hedge mode: Fixed margin per side
        margin_per_side = 5.0  # $5 per position (LONG and SHORT)

        # Commission
        commission_rate = BACKTEST_CONFIG["commission_per_trade"]
        slippage = BACKTEST_CONFIG["slippage_pct"]

        # Process each bar
        entry_bars = {}

        # Session tracking
        session_trades: Dict[str, List[Dict]] = {
            "ASIAN": [], "TOKYO": [], "LONDON": [],
            "NEW_YORK": [], "LONDON_NY_OVERLAP": [], "WEEKEND": []
        }
        hourly_pnl: Dict[int, float] = {h: 0.0 for h in range(24)}

        for bar_idx in range(100, len(df)):  # Start after warmup period
            current_bar = df.iloc[bar_idx]
            bar_time = df.index[bar_idx]

            # Convert to datetime if needed
            if hasattr(bar_time, 'to_pydatetime'):
                bar_time_dt = bar_time.to_pydatetime()
            else:
                bar_time_dt = bar_time

            current_session = get_trading_session(bar_time_dt)
            current_hour = bar_time_dt.hour

            current_price = current_bar["close"]
            high = current_bar["high"]
            low = current_bar["low"]

            # Check existing positions
            for side in list(positions.keys()):
                pos = positions[side]

                # Calculate current ROI
                if side == "LONG":
                    price_pnl_pct = (current_price - pos.avg_entry_price) / pos.avg_entry_price
                else:
                    price_pnl_pct = (pos.avg_entry_price - current_price) / pos.avg_entry_price

                current_roi = price_pnl_pct * leverage

                # Check liquidation (95% ROI loss)
                if current_roi <= -0.95:
                    # Liquidated
                    liquidations += 1
                    balance += 0  # Lose entire margin
                    trade_record = {
                        "side": side,
                        "pnl": -pos.margin_used,
                        "pnl_pct": -1.0,
                        "exit_reason": "LIQUIDATION",
                        "entry_session": pos.entry_session,
                        "exit_session": current_session,
                        "exit_hour": current_hour
                    }
                    trades.append(trade_record)
                    # Track by session
                    if pos.entry_session in session_trades:
                        session_trades[pos.entry_session].append(trade_record)
                    hourly_pnl[current_hour] += trade_record["pnl"]
                    del positions[side]
                    continue

                # Check boost trigger on opposite side
                if current_roi <= boost_trigger_roi and not pos.is_boosted:
                    opposite_side = "SHORT" if side == "LONG" else "LONG"
                    if opposite_side in positions:
                        opp_pos = positions[opposite_side]
                        if not opp_pos.is_boosted:
                            # Boost opposite position
                            boost_margin = margin_per_side * (boost_multiplier - 1)
                            if balance >= boost_margin:
                                balance -= boost_margin
                                opp_pos.margin_used += boost_margin
                                opp_pos.is_boosted = True
                                # Adjust TP
                                boosted_tp_roi = tp_roi * 1.5  # Increase TP by 50%
                                tp_price_pct = boosted_tp_roi / leverage
                                if opposite_side == "LONG":
                                    opp_pos.take_profit = opp_pos.avg_entry_price * (1 + tp_price_pct)
                                else:
                                    opp_pos.take_profit = opp_pos.avg_entry_price * (1 - tp_price_pct)

                # Check take profit
                exit_price = None
                exit_reason = None

                if side == "LONG":
                    if high >= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = "TAKE_PROFIT"
                else:
                    if low <= pos.take_profit:
                        exit_price = pos.take_profit
                        exit_reason = "TAKE_PROFIT"

                # Check trailing stop
                if pos.trailing_stop_active and pos.trailing_stop_price:
                    if side == "LONG" and low <= pos.trailing_stop_price:
                        exit_price = pos.trailing_stop_price
                        exit_reason = "TRAILING_STOP"
                    elif side == "SHORT" and high >= pos.trailing_stop_price:
                        exit_price = pos.trailing_stop_price
                        exit_reason = "TRAILING_STOP"

                # Update trailing stop
                if current_roi >= trailing_activation_roi:
                    pos.trailing_stop_active = True
                    if side == "LONG":
                        pos.highest_price = max(pos.highest_price, high)
                        trail_price = pos.highest_price * (1 - trailing_distance_roi / leverage)
                        pos.trailing_stop_price = max(pos.trailing_stop_price or 0, trail_price)
                    else:
                        pos.lowest_price = min(pos.lowest_price, low)
                        trail_price = pos.lowest_price * (1 + trailing_distance_roi / leverage)
                        pos.trailing_stop_price = min(pos.trailing_stop_price or float("inf"), trail_price)

                # Execute exit
                if exit_price and exit_reason:
                    # Half-close logic
                    if half_close_enabled and exit_reason == "TAKE_PROFIT" and not pos.half_closed:
                        # Close half
                        half_qty = pos.quantity / 2
                        if side == "LONG":
                            pnl = (exit_price - pos.avg_entry_price) * half_qty
                        else:
                            pnl = (pos.avg_entry_price - exit_price) * half_qty

                        # Deduct commission
                        pnl -= exit_price * half_qty * commission_rate

                        # Return half margin + PnL
                        half_margin = pos.margin_used / 2
                        balance += half_margin + pnl
                        pos.margin_used -= half_margin
                        pos.quantity = half_qty
                        pos.half_closed = True

                        # Widen TP for remaining half
                        new_tp_roi = tp_roi * 2
                        tp_price_pct = new_tp_roi / leverage
                        if side == "LONG":
                            pos.take_profit = pos.avg_entry_price * (1 + tp_price_pct)
                        else:
                            pos.take_profit = pos.avg_entry_price * (1 - tp_price_pct)

                        trade_record = {
                            "side": side,
                            "pnl": pnl,
                            "pnl_pct": pnl / half_margin if half_margin > 0 else 0,
                            "exit_reason": "HALF_CLOSE",
                            "entry_session": pos.entry_session,
                            "exit_session": current_session,
                            "exit_hour": current_hour
                        }
                        trades.append(trade_record)
                        if pos.entry_session in session_trades:
                            session_trades[pos.entry_session].append(trade_record)
                        hourly_pnl[current_hour] += pnl
                    else:
                        # Full close
                        if side == "LONG":
                            pnl = (exit_price - pos.avg_entry_price) * pos.quantity
                        else:
                            pnl = (pos.avg_entry_price - exit_price) * pos.quantity

                        # Deduct commission
                        pnl -= exit_price * pos.quantity * commission_rate

                        # Return margin + PnL
                        balance += pos.margin_used + pnl

                        trade_record = {
                            "side": side,
                            "pnl": pnl,
                            "pnl_pct": pnl / pos.margin_used if pos.margin_used > 0 else 0,
                            "exit_reason": exit_reason,
                            "entry_session": pos.entry_session,
                            "exit_session": current_session,
                            "exit_hour": current_hour
                        }
                        trades.append(trade_record)
                        if pos.entry_session in session_trades:
                            session_trades[pos.entry_session].append(trade_record)
                        hourly_pnl[current_hour] += pnl

                        del positions[side]

            # Enter new positions (hedge mode: both LONG and SHORT)
            for side in ["LONG", "SHORT"]:
                if side not in positions and balance >= margin_per_side:
                    # Check signal
                    lookback_df = df.iloc[bar_idx-100:bar_idx+1]
                    if len(lookback_df) >= 30:
                        # Generate signal
                        signal = self.signal_generator.generate_signal_no_cooldown(symbol, lookback_df)

                        # In hedge mode, we always enter both sides regardless of signal
                        # (signal is only used for entry timing optimization)
                        # For now, enter if we don't have position

                        # Calculate entry price with slippage
                        if side == "LONG":
                            entry_price = current_price * (1 + slippage)
                        else:
                            entry_price = current_price * (1 - slippage)

                        # Calculate position size
                        position_value = margin_per_side * leverage
                        quantity = position_value / entry_price

                        # Calculate TP/SL
                        tp_price_pct = tp_roi / leverage
                        sl_price_pct = 0.90 / leverage  # 90% ROI SL (near liquidation)

                        if side == "LONG":
                            tp = entry_price * (1 + tp_price_pct)
                            sl = entry_price * (1 - sl_price_pct)
                        else:
                            tp = entry_price * (1 - tp_price_pct)
                            sl = entry_price * (1 + sl_price_pct)

                        # Deduct margin and commission
                        commission = entry_price * quantity * commission_rate
                        balance -= margin_per_side + commission

                        # Create position
                        positions[side] = BacktestPosition(
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
                            margin_used=margin_per_side,
                            entry_session=current_session  # Track entry session
                        )

            # Update equity curve
            equity_curve.append(balance)

            # Update drawdown
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Close remaining positions at end
        for side, pos in positions.items():
            current_price = df.iloc[-1]["close"]
            if side == "LONG":
                pnl = (current_price - pos.avg_entry_price) * pos.quantity
            else:
                pnl = (pos.avg_entry_price - current_price) * pos.quantity

            pnl -= current_price * pos.quantity * commission_rate
            balance += pos.margin_used + pnl

            trade_record = {
                "side": side,
                "pnl": pnl,
                "pnl_pct": pnl / pos.margin_used if pos.margin_used > 0 else 0,
                "exit_reason": "END_OF_BACKTEST",
                "entry_session": pos.entry_session,
                "exit_session": "END",
                "exit_hour": 0
            }
            trades.append(trade_record)
            if pos.entry_session in session_trades:
                session_trades[pos.entry_session].append(trade_record)

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_return_pct = (balance - initial_balance) / initial_balance
        avg_trade_pnl = sum(t["pnl"] for t in trades) / total_trades if total_trades > 0 else 0

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Calculate session-specific results
        session_results = {}
        for session_name, session_trade_list in session_trades.items():
            if not session_trade_list:
                continue

            s_total = len(session_trade_list)
            s_winners = [t for t in session_trade_list if t["pnl"] > 0]
            s_losers = [t for t in session_trade_list if t["pnl"] <= 0]
            s_win_rate = len(s_winners) / s_total if s_total > 0 else 0
            s_total_pnl = sum(t["pnl"] for t in session_trade_list)
            s_avg_pnl = s_total_pnl / s_total if s_total > 0 else 0

            s_gross_profit = sum(t["pnl"] for t in s_winners)
            s_gross_loss = abs(sum(t["pnl"] for t in s_losers))
            s_pf = s_gross_profit / s_gross_loss if s_gross_loss > 0 else 10.0

            # Find best/worst hours for this session
            session_hourly = {}
            for t in session_trade_list:
                h = t.get("exit_hour", 0)
                session_hourly[h] = session_hourly.get(h, 0) + t["pnl"]

            best_hour = max(session_hourly.keys(), key=lambda h: session_hourly[h]) if session_hourly else 0
            worst_hour = min(session_hourly.keys(), key=lambda h: session_hourly[h]) if session_hourly else 0

            session_results[session_name] = SessionResult(
                session_name=session_name,
                total_trades=s_total,
                winning_trades=len(s_winners),
                losing_trades=len(s_losers),
                total_pnl=s_total_pnl,
                win_rate=s_win_rate,
                profit_factor=min(s_pf, 10),
                avg_trade_pnl=s_avg_pnl,
                best_hour=best_hour,
                worst_hour=worst_hour,
                hourly_pnl=session_hourly
            )

        return OptimizationResult(
            symbol=symbol,
            params=params,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=min(profit_factor, 10),  # Cap at 10
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe,
            liquidations=liquidations,
            avg_trade_pnl=avg_trade_pnl,
            final_balance=balance,
            session_results=session_results
        )

    def optimize_symbol(self, symbol: str) -> Optional[OptimizationResult]:
        """
        Optimize parameters for a single symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Best OptimizationResult for this symbol
        """
        print(f"\n{'='*50}")
        print(f"OPTIMIZING {symbol}")
        print(f"{'='*50}")

        # Fetch data
        df = self.fetch_data(symbol)
        if df is None or df.empty:
            print(f"  No data available for {symbol}")
            return None

        # Generate combinations
        combinations = self.generate_param_combinations()
        total = len(combinations)

        print(f"Testing {total} parameter combinations...")

        # Test each combination
        best_result = None
        best_score = -float("inf")

        for i, params in enumerate(combinations):
            # Progress update
            if (i + 1) % 10 == 0 or i == 0:
                pct = (i + 1) / total * 100
                print(f"  Progress: {i+1}/{total} ({pct:.1f}%)", end="")
                if best_result:
                    print(f" | Best: {best_result.total_return_pct*100:.1f}% return", end="")
                print()

            # Run backtest
            try:
                result = self.run_single_backtest(symbol, params, df)
                self.results[symbol].append(result)

                # Check if this is the best
                score = result.score()
                if score > best_score:
                    best_score = score
                    best_result = result
            except Exception as e:
                print(f"  ERROR testing params {params}: {e}")
                continue

        if best_result:
            self.best_params[symbol] = best_result.params
            print(f"\n  BEST RESULT for {symbol}:")
            print(f"    Return: {best_result.total_return_pct*100:.2f}%")
            print(f"    Win Rate: {best_result.win_rate*100:.1f}%")
            print(f"    Profit Factor: {best_result.profit_factor:.2f}")
            print(f"    Max Drawdown: {best_result.max_drawdown_pct*100:.1f}%")
            print(f"    Liquidations: {best_result.liquidations}")
            print(f"    Score: {best_score:.3f}")
            print(f"    Parameters: {best_result.params}")

        return best_result

    def optimize_all(self) -> Dict[str, OptimizationResult]:
        """
        Optimize all symbols.

        Returns:
            Dict of best results per symbol
        """
        best_results = {}

        for symbol in self.symbols:
            result = self.optimize_symbol(symbol)
            if result:
                best_results[symbol] = result

        return best_results

    def save_results(self, filename: str = None):
        """Save optimization results to file."""
        if filename is None:
            filename = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs",
            filename
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare data
        data = {
            "optimization_date": datetime.now().isoformat(),
            "days_tested": self.days,
            "mode": "quick" if self.quick_mode else "full",
            "symbols": {},
        }

        for symbol in self.symbols:
            if symbol in self.best_params:
                best = None
                for r in self.results[symbol]:
                    if r.params == self.best_params[symbol]:
                        best = r
                        break

                if best:
                    # Session results
                    session_data = {}
                    for session_name, sr in best.session_results.items():
                        session_data[session_name] = {
                            "total_trades": sr.total_trades,
                            "win_rate": sr.win_rate,
                            "total_pnl": sr.total_pnl,
                            "profit_factor": sr.profit_factor,
                            "is_profitable": sr.is_profitable(),
                            "best_hour": sr.best_hour,
                            "worst_hour": sr.worst_hour,
                        }

                    data["symbols"][symbol] = {
                        "best_params": best.params,
                        "performance": {
                            "total_return_pct": best.total_return_pct,
                            "win_rate": best.win_rate,
                            "total_trades": best.total_trades,
                            "profit_factor": best.profit_factor,
                            "max_drawdown_pct": best.max_drawdown_pct,
                            "sharpe_ratio": best.sharpe_ratio,
                            "liquidations": best.liquidations,
                            "score": best.score()
                        },
                        "sessions": session_data
                    }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def export_best_config(self, filename: str = "optimized_config.py"):
        """Export best parameters as Python config."""
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config",
            filename
        )

        lines = [
            '"""',
            'Optimized Trading Configuration',
            f'Generated: {datetime.now().isoformat()}',
            f'Period: {self.days} days backtest',
            '"""',
            '',
            '# Best parameters per symbol (from optimization)',
            'OPTIMIZED_SYMBOL_SETTINGS = {'
        ]

        for symbol in self.symbols:
            if symbol in self.best_params:
                params = self.best_params[symbol]
                best = None
                for r in self.results[symbol]:
                    if r.params == params:
                        best = r
                        break

                lines.append(f'    "{symbol}": {{')
                lines.append(f'        # Return: {best.total_return_pct*100:.1f}%, Win: {best.win_rate*100:.0f}%, PF: {best.profit_factor:.2f}')
                lines.append(f'        "tp_roi": {params["tp_roi"]},')
                lines.append(f'        "leverage": {params["leverage"]},')
                lines.append(f'        "boost_trigger_roi": {params["boost_trigger_roi"]},')
                lines.append(f'        "boost_multiplier": {params["boost_multiplier"]},')
                lines.append(f'        "trailing_activation_roi": {params["trailing_activation_roi"]},')
                lines.append(f'        "trailing_distance_roi": {params["trailing_distance_roi"]},')
                lines.append(f'        "half_close_enabled": {params["half_close_enabled"]},')

                # Add session recommendations
                if best.session_results:
                    profitable = [s for s in best.session_results.values() if s.is_profitable()]
                    losing = [s for s in best.session_results.values() if not s.is_profitable()]
                    if profitable:
                        lines.append(f'        # BEST SESSIONS: {", ".join([s.session_name for s in profitable])}')
                    if losing:
                        lines.append(f'        # AVOID SESSIONS: {", ".join([s.session_name for s in losing])}')
                    # Find best overall hour
                    all_hours = {}
                    for sr in best.session_results.values():
                        for h, pnl in sr.hourly_pnl.items():
                            all_hours[h] = all_hours.get(h, 0) + pnl
                    if all_hours:
                        best_hour = max(all_hours.keys(), key=lambda h: all_hours[h])
                        worst_hour = min(all_hours.keys(), key=lambda h: all_hours[h])
                        lines.append(f'        # BEST HOUR: {best_hour}:00 UTC | WORST: {worst_hour}:00 UTC')

                lines.append('    },')

        lines.append('}')
        lines.append('')

        with open(filepath, "w") as f:
            f.write('\n'.join(lines))

        print(f"Config exported to: {filepath}")
        return filepath

    def print_session_analysis(self, result: OptimizationResult):
        """Print detailed session analysis for a result."""
        if not result.session_results:
            return

        print(f"\n  SESSION ANALYSIS:")
        print(f"  {'-'*60}")

        # Sort sessions by profitability
        sorted_sessions = sorted(
            result.session_results.values(),
            key=lambda s: s.total_pnl,
            reverse=True
        )

        for sr in sorted_sessions:
            status = "PROFITABLE" if sr.is_profitable() else "LOSING"
            print(f"    {sr.session_name}:")
            print(f"      Status: {status}")
            print(f"      Trades: {sr.total_trades} | Win Rate: {sr.win_rate*100:.1f}%")
            print(f"      PnL: ${sr.total_pnl:+.2f} | PF: {sr.profit_factor:.2f}x")
            print(f"      Best Hour: {sr.best_hour}:00 UTC | Worst: {sr.worst_hour}:00 UTC")

        # Recommend which sessions to trade
        profitable_sessions = [s for s in sorted_sessions if s.is_profitable()]
        losing_sessions = [s for s in sorted_sessions if not s.is_profitable()]

        print(f"\n  RECOMMENDATIONS:")
        if profitable_sessions:
            print(f"    TRADE: {', '.join([s.session_name for s in profitable_sessions])}")
        if losing_sessions:
            print(f"    AVOID: {', '.join([s.session_name for s in losing_sessions])}")

    def print_summary(self, show_sessions: bool = True):
        """Print optimization summary."""
        print("\n" + "="*70)
        print("OPTIMIZATION SUMMARY")
        print("="*70)

        for symbol in self.symbols:
            if symbol not in self.best_params:
                continue

            params = self.best_params[symbol]
            best = None
            for r in self.results[symbol]:
                if r.params == params:
                    best = r
                    break

            if best:
                print(f"\n{symbol}:")
                print(f"  Return: {best.total_return_pct*100:+.2f}%")
                print(f"  Win Rate: {best.win_rate*100:.1f}%")
                print(f"  Profit Factor: {best.profit_factor:.2f}x")
                print(f"  Max Drawdown: {best.max_drawdown_pct*100:.1f}%")
                print(f"  Trades: {best.total_trades}")
                print(f"  Liquidations: {best.liquidations}")
                print(f"  Optimal Settings:")
                print(f"    TP ROI: {params['tp_roi']*100:.0f}%")
                print(f"    Leverage: {params['leverage']}x")
                print(f"    Boost Trigger: {params['boost_trigger_roi']*100:.0f}% ROI")
                print(f"    Boost Multiplier: {params['boost_multiplier']}x")
                print(f"    Trailing Activation: {params['trailing_activation_roi']*100:.0f}% ROI")
                print(f"    Trailing Distance: {params['trailing_distance_roi']*100:.0f}% ROI")
                print(f"    Half-Close: {params['half_close_enabled']}")

                # Show session analysis
                if show_sessions and best.session_results:
                    self.print_session_analysis(best)

        print("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parameter Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to optimize (default: all)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Days of historical data (default: 60)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer combinations)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export best config as Python file"
    )

    parser.add_argument(
        "--sessions",
        action="store_true",
        help="Show detailed session analysis (Asian, Tokyo, London, NY)"
    )

    args = parser.parse_args()

    # Determine symbols
    symbols = [args.symbol] if args.symbol else FUTURES_SYMBOLS_LIVE

    # Create optimizer
    optimizer = ParameterOptimizer(
        symbols=symbols,
        days=args.days,
        quick_mode=args.quick
    )

    # Run optimization
    start_time = time.time()
    best_results = optimizer.optimize_all()
    elapsed = time.time() - start_time

    # Print summary (always show sessions now - it's important!)
    optimizer.print_summary(show_sessions=True)

    print(f"\nOptimization completed in {elapsed/60:.1f} minutes")

    # Save results
    if args.save or True:  # Always save
        optimizer.save_results()

    # Export config
    if args.export or True:  # Always export
        optimizer.export_best_config()

    return best_results


if __name__ == "__main__":
    main()
