"""
Binance Futures Live Trading Engine
===================================
Real trading with actual orders on Binance Futures
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.trading_config import (
    FUTURES_SYMBOLS, FUTURES_SYMBOLS_LIVE, FUTURES_SYMBOLS_DEMO,
    STRATEGY_CONFIG, RISK_CONFIG, DCA_CONFIG,
    LOGGING_CONFIG, MOMENTUM_CONFIG, BINANCE_CONFIG, SYMBOL_SETTINGS,
    SMART_COMPOUNDING_CONFIG
)
from engine.binance_client import BinanceClient
from engine.momentum_signal import MasterMomentumSignal, MultiTimeframeMomentumSignal, TradingSignal, SignalType


@dataclass
class LivePosition:
    """Represents a live position on Binance"""
    symbol: str
    side: str                      # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss_order_id: Optional[int] = None
    take_profit_order_id: Optional[int] = None
    dca_count: int = 0
    avg_entry_price: float = 0.0
    margin_used: float = 0.0       # Total margin used for this symbol
    # Trailing TP tracking
    peak_roi: float = 0.0          # Highest ROI reached
    trailing_active: bool = False  # True when trailing TP is activated
    tp_cancelled: bool = False     # True when Binance TP order was cancelled for trailing
    # Hedge tracking
    is_hedged: bool = False        # True if this position has a hedge open
    hedge_side: Optional[str] = None  # Side of hedge position ("LONG" or "SHORT")
    waiting_for_breakeven: bool = False  # True if waiting to exit at breakeven
    hedge_start_time: Optional[datetime] = None  # When hedge was opened
    # ENHANCED BOOST MODE tracking
    is_boosted: bool = False       # True if this position is boosted (1.5x)
    boost_multiplier: float = 1.0  # Current boost multiplier (1.0 = normal, 1.5 = boosted)
    half_close_count: int = 0      # Number of half-close cycles performed


class BinanceLiveTradingEngine:
    """
    Live Trading Engine for Binance Futures

    WARNING: This engine places REAL orders with REAL money!
    Use with caution. Always test on testnet first.
    """

    def __init__(self, testnet: bool = True, use_mtf: bool = True):
        """
        Initialize live trading engine

        Args:
            testnet: If True, use Binance testnet (RECOMMENDED for testing)
            use_mtf: Use multi-timeframe signal confirmation
        """
        # Initialize client
        self.client = BinanceClient(testnet=testnet)
        self.testnet = testnet

        # Select symbols based on mode (LIVE = DOTUSDT only, DEMO = all symbols)
        if testnet:
            self.symbols = FUTURES_SYMBOLS_DEMO
        else:
            self.symbols = FUTURES_SYMBOLS_LIVE
            print(f"[LIVE MODE] Trading only: {', '.join(self.symbols)}")

        # Initialize signal generator
        if use_mtf:
            self.signal_generator = MultiTimeframeMomentumSignal()
        else:
            self.signal_generator = MasterMomentumSignal()
        self.use_mtf = use_mtf

        # Track our positions locally
        self.positions: Dict[str, LivePosition] = {}

        # Dynamic Fund Allocation
        self.num_symbols = len(self.symbols)
        self.symbol_budgets: Dict[str, float] = {}      # Max budget per symbol
        self.symbol_margin_used: Dict[str, float] = {}  # Current margin used per symbol
        self.leverage = STRATEGY_CONFIG["leverage"]      # Fixed 20x leverage

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        
        # Per-symbol trade statistics
        self.symbol_stats: Dict[str, Dict] = {}  # symbol -> {wins, losses, tp_count, sl_count, pnl}
        for symbol in self.symbols:
            self.symbol_stats[symbol] = {
                "wins": 0,
                "losses": 0,
                "tp_count": 0,
                "sl_count": 0,
                "pnl": 0.0
            }
        self.starting_balance = 0.0
        self.daily_start_balance = 0.0  # Balance at start of day (for daily loss limit)
        self.last_reset_date = datetime.now().date()

        # Symbol tracking
        self.last_check_time: Dict[str, datetime] = {}
        self.symbol_check_interval = 60  # seconds

        # Data buffer for SMART DCA (stores recent market data per symbol)
        self.data_buffer: Dict[str, any] = {}

        # Hybrid Hold + Trade System tracking
        self.current_trend: Dict[str, str] = {}   # Current detected trend per symbol ("BULLISH"/"BEARISH")
        self.last_trend_check: Dict[str, datetime] = {}  # Last trend check time per symbol
        self.trend_check_interval = 60  # Check trend every 60 seconds
        self.last_flip_time: Dict[str, datetime] = {}  # Last flip time per symbol (for cooldown)

        # Pending re-entry after TP (for ALWAYS HOLD strategy)
        self.pending_reentry: Dict[str, str] = {}  # symbol -> side to re-enter ("LONG"/"SHORT")

        # HEDGE MODE: Track positions by symbol_side key (e.g., "BTCUSDT_LONG", "BTCUSDT_SHORT")
        # When hedge_mode enabled, we track BOTH LONG and SHORT positions per symbol
        self.hedge_mode = DCA_CONFIG.get("hedge_mode", {}).get("enabled", False)
        self.hedge_budget_split = DCA_CONFIG.get("hedge_mode", {}).get("budget_split", 0.5)

        # ENHANCED BOOST MODE tracking (per symbol)
        # When one side hits DCA trigger level, boost the opposite side
        self.boost_mode_active: Dict[str, bool] = {}      # symbol -> True if boost mode active
        self.boosted_side: Dict[str, str] = {}            # symbol -> "LONG" or "SHORT" (which side is boosted)
        self.boost_trigger_side: Dict[str, str] = {}      # symbol -> side that triggered boost (DCA side)
        self.boost_locked_profit: Dict[str, float] = {}   # symbol -> profit locked from half-closes
        self.boost_cycle_count: Dict[str, int] = {}       # symbol -> number of half-close cycles
        self.boost_multiplier = 1.5                       # 1.5x boost
        # STRONG TREND MODE - ADX-based trend detection for DCA blocking
        # When ADX > 40, block DCA 2+ on loser side and 2x boost winner side
        # MUTUALLY EXCLUSIVE with Boost Mode (Boost Mode takes priority)
        self.strong_trend_mode: Dict[str, bool] = {}       # symbol -> True if strong trend active
        self.strong_trend_direction: Dict[str, str] = {}   # symbol -> "UP" or "DOWN"
        self.adx_threshold = 40                            # ADX > 40 = strong trend
        self.current_adx: Dict[str, float] = {}            # symbol -> current ADX value
        self.current_plus_di: Dict[str, float] = {}        # symbol -> current +DI value
        self.current_minus_di: Dict[str, float] = {}       # symbol -> current -DI value
        self.trend_boosted_side: Dict[str, str] = {}       # symbol -> side that got 2x trend boost

        # STOP FOR DAY - After SL hit, stop trading that symbol until next day
        self.stopped_for_day: Dict[str, bool] = {}        # symbol -> True if stopped for the day
        self.sl_hit_date: Dict[str, datetime] = {}        # symbol -> date when SL hit

        # SMART COMPOUNDING - Reserve Fund System
        # Instead of 100% compounding, split profits: 50% compound, 50% reserve
        self.smart_compounding_enabled = SMART_COMPOUNDING_CONFIG.get("enabled", True)
        self.compound_pct = SMART_COMPOUNDING_CONFIG.get("compound_pct", 0.50)
        self.reserve_pct = SMART_COMPOUNDING_CONFIG.get("reserve_pct", 0.50)
        self.initial_capital = SMART_COMPOUNDING_CONFIG.get("initial_capital", 125.0)
        self.reserve_fund = 0.0           # Protected profits (never traded)
        self.total_realized_profit = 0.0  # Total lifetime realized profit
        self.trading_capital = self.initial_capital  # Capital available for trading
        self.reserve_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Binance_Futures_Trading",
            SMART_COMPOUNDING_CONFIG.get("reserve_file", "reserve_fund.json")
        )
        self._load_reserve_fund()  # Load saved state

        # POSITION STATE PERSISTENCE - Save DCA levels and boost state
        # This allows the bot to properly restore position states after restart
        # Put it in Binance_Futures_Trading/ directory (parent of engine/)
        self.position_state_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "position_state.json"
        )

        # Logging
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Binance_Futures_Trading",
            LOGGING_CONFIG["log_dir"]
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Trade history
        self.trades = []

        self.running = False

        # Order cleanup tracking
        self.last_order_cleanup: datetime = datetime.now()
        self.order_cleanup_interval = 300  # Clean up orphaned orders every 5 minutes

    def log(self, message: str, level: str = "INFO"):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = "TESTNET" if self.testnet else "LIVE"
        print(f"[{timestamp}] [{mode}] [{level}] {message}")

    def log_trade(self, symbol: str, side: str, entry_price: float, exit_price: float,
                  quantity: float, pnl: float, exit_type: str, dca_level: int = 0):
        """
        Log a closed trade to JSON file for permanent record.
        This ensures we track all trades even if the bot restarts.
        """
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "pnl_pct": (pnl / (entry_price * quantity / self.leverage)) * 100 if entry_price > 0 else 0,
            "exit_type": exit_type,  # "TP", "SL", "AUTO_CLOSE", "MANUAL"
            "dca_level": dca_level,
            "leverage": self.leverage,
            "testnet": self.testnet
        }

        # Add to in-memory list
        self.trades.append(trade_record)

        # Save to JSON file
        trade_log_path = os.path.join(self.log_dir, LOGGING_CONFIG["trade_log_file"])
        try:
            # Load existing trades
            if os.path.exists(trade_log_path):
                with open(trade_log_path, 'r') as f:
                    all_trades = json.load(f)
            else:
                all_trades = []

            # Append new trade
            all_trades.append(trade_record)

            # Save back
            with open(trade_log_path, 'w') as f:
                json.dump(all_trades, f, indent=2)

            self.log(f"Trade logged: {symbol} {side} | P&L: ${pnl:+.2f} | Exit: {exit_type}")

            # SMART COMPOUNDING: Process the realized profit/loss
            # This splits profits 50/50 between compound and reserve
            self.process_realized_profit(pnl)

        except Exception as e:
            self.log(f"Error logging trade: {e}", level="WARN")

    # =========================================================================
    # SMART COMPOUNDING - RESERVE FUND SYSTEM
    # =========================================================================

    def _load_reserve_fund(self):
        """
        Load reserve fund state from file.
        IMPORTANT: On startup, sync trading_capital with ACTUAL Binance balance
        to prevent sizing based on outdated/wrong capital.
        """
        try:
            # FIRST: Get ACTUAL balance from Binance
            actual_balance = self.client.get_balance()

            if os.path.exists(self.reserve_file):
                with open(self.reserve_file, 'r') as f:
                    data = json.load(f)
                    self.reserve_fund = data.get("reserve_fund", 0.0)
                    self.total_realized_profit = data.get("total_realized_profit", 0.0)
                    saved_trading_capital = data.get("trading_capital", self.initial_capital)
                    saved_initial = data.get("initial_capital", self.initial_capital)

                    # CHECK: If actual balance is significantly different from saved capital,
                    # reset to actual balance (prevents sizing based on wrong capital)
                    capital_diff = abs(actual_balance - saved_trading_capital)

                    if capital_diff > 10:  # More than $10 difference = reset to actual
                        self.log(f"[SMART COMPOUND] WARNING: Saved capital ${saved_trading_capital:.2f} differs from actual ${actual_balance:.2f}")
                        self.log(f"[SMART COMPOUND] RESETTING to actual balance ${actual_balance:.2f}")
                        self.trading_capital = actual_balance
                        self.initial_capital = actual_balance
                        self.reserve_fund = 0.0
                        self.total_realized_profit = 0.0
                        self._save_reserve_fund()  # Save the reset state
                    else:
                        # Normal load - use saved values
                        self.trading_capital = saved_trading_capital
                        self.log(f"[SMART COMPOUND] Loaded: Trading=${self.trading_capital:.2f} | Reserve=${self.reserve_fund:.2f}")
            else:
                # No saved file - initialize from actual balance
                self.log(f"[SMART COMPOUND] No saved state - initializing from actual balance ${actual_balance:.2f}")
                self.trading_capital = actual_balance
                self.initial_capital = actual_balance
                self._save_reserve_fund()

        except Exception as e:
            self.log(f"[SMART COMPOUND] Error loading reserve fund: {e}", level="WARN")
            # Fallback to actual balance on error
            try:
                actual_balance = self.client.get_balance()
                self.trading_capital = actual_balance
                self.initial_capital = actual_balance
                self.log(f"[SMART COMPOUND] Fallback: Using actual balance ${actual_balance:.2f}")
            except:
                pass


    def _save_reserve_fund(self):
        """Save reserve fund state to file"""
        try:
            data = {
                "reserve_fund": self.reserve_fund,
                "total_realized_profit": self.total_realized_profit,
                "trading_capital": self.trading_capital,
                "initial_capital": self.initial_capital,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.reserve_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log(f"[SMART COMPOUND] Error saving reserve fund: {e}", level="WARN")

    def process_realized_profit(self, pnl: float):
        """
        Process realized profit/loss with smart compounding.
        - 50% of profit compounds into trading capital
        - 50% of profit goes to reserve fund (protected)
        - Losses reduce trading capital only (reserve is protected)
        """
        if not self.smart_compounding_enabled:
            return

        self.total_realized_profit += pnl

        if pnl > 0:
            # PROFIT: Split 50/50
            compound_amount = pnl * self.compound_pct
            reserve_amount = pnl * self.reserve_pct

            self.trading_capital += compound_amount
            self.reserve_fund += reserve_amount

            self.log(f"[SMART COMPOUND] Profit ${pnl:.2f} -> Compound +${compound_amount:.2f} | Reserve +${reserve_amount:.2f}")
            self.log(f"[SMART COMPOUND] Trading Capital: ${self.trading_capital:.2f} | Reserve Fund: ${self.reserve_fund:.2f}")
        else:
            # LOSS: Only affects trading capital (reserve protected)
            self.log(f"[SMART COMPOUND] Loss ${pnl:.2f} -> Reserve protected (${self.reserve_fund:.2f})")

        # Save state after each profit/loss
        self._save_reserve_fund()

    def get_trading_capital(self) -> float:
        """
        Get the capital available for trading.
        This is initial_capital + compounded profits (NOT total balance).
        Reserve fund is NEVER included in trading capital.
        """
        if self.smart_compounding_enabled:
            return self.trading_capital
        else:
            # Fall back to actual balance if smart compounding disabled
            return self.client.get_balance()

    def get_reserve_fund_status(self) -> dict:
        """Get reserve fund status for display"""
        return {
            "enabled": self.smart_compounding_enabled,
            "initial_capital": self.initial_capital,
            "trading_capital": self.trading_capital,
            "reserve_fund": self.reserve_fund,
            "total_realized_profit": self.total_realized_profit,
            "compound_pct": self.compound_pct * 100,
            "reserve_pct": self.reserve_pct * 100
        }

    # =========================================================================
    # POSITION STATE PERSISTENCE - Save/Load DCA levels and boost state
    # =========================================================================
    # This allows the bot to properly restore position states after restart
    # instead of guessing DCA levels from margin used

    def _save_position_state(self):
        """
        Save position state to file for restart recovery.
        Saves: DCA levels, boost mode state, position details
        """
        try:
            state = {
                "last_updated": datetime.now().isoformat(),
                "positions": {},
                "boost_state": {
                    "boost_mode_active": dict(self.boost_mode_active),
                    "boosted_side": dict(self.boosted_side),
                    "boost_trigger_side": dict(self.boost_trigger_side),
                    "boost_locked_profit": dict(self.boost_locked_profit),
                    "boost_cycle_count": dict(self.boost_cycle_count),
                }
            }

            # Save each position's state
            for pos_key, pos in self.positions.items():
                state["positions"][pos_key] = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "avg_entry_price": pos.avg_entry_price,
                    "quantity": pos.quantity,
                    "dca_count": pos.dca_count,
                    "margin_used": pos.margin_used,
                    "is_boosted": pos.is_boosted,
                    "boost_multiplier": pos.boost_multiplier,
                    "half_close_count": pos.half_close_count,
                    "peak_roi": pos.peak_roi,
                    "trailing_active": pos.trailing_active,
                }

            with open(self.position_state_file, 'w') as f:
                json.dump(state, f, indent=2)

            self.log(f"[STATE] Saved position state: {len(self.positions)} positions")

        except Exception as e:
            self.log(f"[STATE] Error saving position state: {e}", level="WARN")

    def _load_position_state(self) -> dict:
        """
        Load saved position state from file.
        Returns dict with positions and boost_state, or empty dict if not found.
        """
        try:
            if os.path.exists(self.position_state_file):
                with open(self.position_state_file, 'r') as f:
                    state = json.load(f)
                    self.log(f"[STATE] Loaded saved state from {state.get('last_updated', 'unknown')}")
                    return state
        except Exception as e:
            self.log(f"[STATE] Error loading position state: {e}", level="WARN")
        return {}

    def _clear_position_state(self, position_key: str = None):
        """
        Clear position state from file.
        If position_key is given, only clear that position.
        If None, clear all state.
        """
        try:
            if position_key:
                # Load, remove specific position, save
                state = self._load_position_state()
                if state and "positions" in state:
                    if position_key in state["positions"]:
                        del state["positions"][position_key]
                        self.log(f"[STATE] Cleared state for {position_key}")
                    # Check if boost state needs clearing for this symbol
                    symbol = position_key.split("_")[0] if "_" in position_key else position_key
                    if "boost_state" in state:
                        for key in ["boost_mode_active", "boosted_side", "boost_trigger_side",
                                    "boost_locked_profit", "boost_cycle_count"]:
                            if symbol in state["boost_state"].get(key, {}):
                                del state["boost_state"][key][symbol]
                    # Save updated state
                    with open(self.position_state_file, 'w') as f:
                        json.dump(state, f, indent=2)
            else:
                # Clear all state
                if os.path.exists(self.position_state_file):
                    os.remove(self.position_state_file)
                    self.log("[STATE] Cleared all position state")
        except Exception as e:
            self.log(f"[STATE] Error clearing position state: {e}", level="WARN")

    def setup_symbol(self, symbol: str):
        """Set up a symbol for trading (leverage, margin type)"""
        try:
            # Set leverage
            leverage = STRATEGY_CONFIG["leverage"]
            self.client.set_leverage(symbol, leverage)
            self.log(f"Set leverage for {symbol}: {leverage}x")

            # Set margin type (ISOLATED for safety)
            try:
                self.client.set_margin_type(symbol, "ISOLATED")
                self.log(f"Set margin type for {symbol}: ISOLATED")
            except Exception as e:
                # May fail if already set
                pass

        except Exception as e:
            self.log(f"Error setting up {symbol}: {e}", level="WARN")

    def initialize_dynamic_allocation(self):
        """
        Initialize dynamic fund allocation based on TRADING CAPITAL (not total balance).
        With smart compounding: uses initial_capital + compounded profits
        Reserve fund is NEVER included in trading allocation.
        """
        # If hedge mode is enabled, ensure Binance account is set to Hedge Mode
        if self.hedge_mode:
            self.log("HEDGE MODE: Checking Binance position mode...", level="INFO")
            if self.client.ensure_hedge_mode():
                self.log("HEDGE MODE: Binance account is in Hedge Mode (Dual Side Position)", level="INFO")
            else:
                self.log("HEDGE MODE: WARNING - Could not enable Hedge Mode on Binance!", level="WARN")
                self.log("  You may need to close all positions first, then enable manually", level="WARN")

        # Get actual balance from Binance
        actual_balance = self.client.get_balance()

        # SMART COMPOUNDING: Use trading_capital instead of total balance
        # This excludes the reserve fund from trading allocations
        if self.smart_compounding_enabled:
            # Use our calculated trading capital (initial + 50% of profits)
            trading_capital = self.get_trading_capital()
            self.log(f"[SMART COMPOUND] Actual Balance: ${actual_balance:.2f}")
            self.log(f"[SMART COMPOUND] Trading Capital: ${trading_capital:.2f} (excludes reserve)")
            self.log(f"[SMART COMPOUND] Reserve Fund: ${self.reserve_fund:.2f} (protected)")
            balance = trading_capital
        else:
            balance = actual_balance

        # Apply buffer for fees/safety
        buffer = RISK_CONFIG.get("allocation_buffer_pct", 0.05)
        available_balance = balance * (1 - buffer)

        self.log(f"Dynamic Allocation: ${balance:.2f} balance")
        self.log(f"  Buffer: {buffer*100:.1f}% reserved for fees")
        self.log(f"  Available: ${available_balance:.2f}")
        self.log(f"  Symbols: {self.num_symbols}")
        self.log(f"  Leverage: {self.leverage}x (ISOLATED)")

        # Simple equal distribution among symbols
        per_symbol = available_balance / self.num_symbols

        # Initialize budgets for each symbol
        for symbol in self.symbols:
            self.symbol_budgets[symbol] = per_symbol
            self.symbol_margin_used[symbol] = 0.0
            self.log(f"  {symbol}: ${per_symbol:.2f}")

        if self.hedge_mode:
            avg_budget = sum(self.symbol_budgets.values()) / self.num_symbols
            self.log(f"  HEDGE MODE: ~${avg_budget * self.hedge_budget_split:.2f} per side (LONG + SHORT)")

        return available_balance / self.num_symbols  # Return average for compatibility

    # =========================================================================
    # HEDGE MODE HELPER FUNCTIONS
    # =========================================================================

    def get_position_key(self, symbol: str, side: str = None) -> str:
        """
        Get position tracking key.
        In hedge mode: returns 'SYMBOL_SIDE' (e.g., 'BTCUSDT_LONG')
        In normal mode: returns 'SYMBOL' (e.g., 'BTCUSDT')
        """
        if self.hedge_mode and side:
            return f"{symbol}_{side}"
        return symbol

    def get_symbol_from_key(self, key: str) -> str:
        """Extract symbol from position key (handles both 'BTCUSDT' and 'BTCUSDT_LONG')"""
        if "_LONG" in key:
            return key.replace("_LONG", "")
        if "_SHORT" in key:
            return key.replace("_SHORT", "")
        return key

    def get_side_from_key(self, key: str) -> str:
        """Extract side from position key"""
        if "_LONG" in key:
            return "LONG"
        if "_SHORT" in key:
            return "SHORT"
        return None

    def has_position(self, symbol: str, side: str = None) -> bool:
        """
        Check if we have a position in this symbol (optionally for specific side).
        In hedge mode, checks for specific side.
        In normal mode, checks for any position.
        """
        if self.hedge_mode:
            if side:
                key = self.get_position_key(symbol, side)
                return key in self.positions
            else:
                # Check if any position exists for this symbol
                return f"{symbol}_LONG" in self.positions or f"{symbol}_SHORT" in self.positions
        else:
            return symbol in self.positions

    def get_position(self, symbol: str, side: str = None) -> Optional[LivePosition]:
        """Get position for symbol (and optionally side in hedge mode)"""
        key = self.get_position_key(symbol, side)
        return self.positions.get(key)

    def get_all_positions_for_symbol(self, symbol: str) -> List[LivePosition]:
        """Get all positions for a symbol (both LONG and SHORT in hedge mode)"""
        if self.hedge_mode:
            positions = []
            if f"{symbol}_LONG" in self.positions:
                positions.append(self.positions[f"{symbol}_LONG"])
            if f"{symbol}_SHORT" in self.positions:
                positions.append(self.positions[f"{symbol}_SHORT"])
            return positions
        else:
            if symbol in self.positions:
                return [self.positions[symbol]]
            return []

    def get_hedge_entry_margin(self, symbol: str) -> float:
        """Get margin for hedge mode entry (split between LONG and SHORT)"""
        if symbol not in self.symbol_budgets:
            return 0.0
        budget = self.symbol_budgets[symbol]
        entry_pct = RISK_CONFIG.get("initial_entry_pct", 0.20)
        # In hedge mode, split the budget between LONG and SHORT
        hedge_split = self.hedge_budget_split  # 0.5 = 50% each side
        return budget * entry_pct * hedge_split

    def get_entry_margin(self, symbol: str, side: str = None) -> float:
        """Get margin for initial entry (20% of symbol budget, or 10% per side in hedge mode)."""
        if symbol not in self.symbol_budgets:
            return 0.0
        budget = self.symbol_budgets[symbol]

        # In hedge mode, split the budget between LONG and SHORT
        if self.hedge_mode:
            budget = budget * self.hedge_budget_split
        entry_pct = RISK_CONFIG.get("initial_entry_pct", 0.20)
        return budget * entry_pct

    def get_dca_margin(self, symbol: str, dca_level: int, side: str = None) -> float:
        """Get margin for DCA level (1-4). In hedge mode, split by side."""
        if symbol not in self.symbol_budgets:
            return 0.0
        budget = self.symbol_budgets[symbol]

        # In hedge mode, split the budget between LONG and SHORT
        if self.hedge_mode:
            budget = budget * self.hedge_budget_split

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

    def detect_dca_level_from_margin(self, symbol: str, actual_margin: float) -> int:
        """
        Auto-detect DCA level based on margin used.
        This allows recovery of DCA state after bot restart.

        Margin distribution:
        - Initial entry: 10% of budget
        - DCA 1: +15% = 25% cumulative
        - DCA 2: +20% = 45% cumulative
        - DCA 3: +25% = 70% cumulative
        - DCA 4: +30% = 100% cumulative
        """
        if symbol not in self.symbol_budgets or actual_margin <= 0:
            return 0

        budget = self.symbol_budgets[symbol]

        # Calculate cumulative margin thresholds
        entry_pct = RISK_CONFIG.get("initial_entry_pct", 0.10)
        dca1_pct = RISK_CONFIG.get("dca1_pct", 0.15)
        dca2_pct = RISK_CONFIG.get("dca2_pct", 0.20)
        dca3_pct = RISK_CONFIG.get("dca3_pct", 0.25)
        dca4_pct = RISK_CONFIG.get("dca4_pct", 0.30)

        # Cumulative thresholds (with 10% tolerance)
        threshold_0 = budget * entry_pct * 1.1                          # ~11% = DCA 0
        threshold_1 = budget * (entry_pct + dca1_pct) * 1.1             # ~27.5% = DCA 1
        threshold_2 = budget * (entry_pct + dca1_pct + dca2_pct) * 1.1  # ~49.5% = DCA 2
        threshold_3 = budget * (entry_pct + dca1_pct + dca2_pct + dca3_pct) * 1.1  # ~77% = DCA 3

        # Determine DCA level
        if actual_margin > threshold_3:
            return 4
        elif actual_margin > threshold_2:
            return 3
        elif actual_margin > threshold_1:
            return 2
        elif actual_margin > threshold_0:
            return 1
        else:
            return 0

    # =========================================================================
    # ENHANCED BOOST MODE FUNCTIONS
    # =========================================================================
    # When one side hits DCA trigger level -> boost the OTHER side 1.5x
    # At TP on boosted side: Close HALF, lock profit, add back 0.5x
    # Trailing activates AFTER each half-close cycle
    # Continue until losing side recovers (TP) or hits SL


    # =========================================================================
    # STRONG TREND MODE - ADX-BASED TREND DETECTION
    # =========================================================================
    # When ADX > 40, we're in a strong trend:
    # - Winner side (LONG in UP, SHORT in DOWN): Gets 2x entry boost
    # - Loser side (SHORT in UP, LONG in DOWN): DCA 2+ is BLOCKED
    # This prevents adding to losing positions in strong trending markets

    def calculate_adx_for_symbol(self, symbol: str, df=None) -> tuple:
        """
        Calculate ADX (Average Directional Index) for a symbol.
        Returns: (adx, plus_di, minus_di)
        """
        try:
            # Get market data if not provided
            if df is None:
                if symbol in self.data_buffer and self.data_buffer[symbol] is not None:
                    if isinstance(self.data_buffer[symbol], dict):
                        df = self.data_buffer[symbol].get("1m")
                    else:
                        df = self.data_buffer[symbol]

                if df is None:
                    market_data = self.client.get_market_data(symbol)
                    if market_data and "1m" in market_data:
                        df = market_data["1m"]

            if df is None or len(df) < 28:
                return 0.0, 0.0, 0.0

            period = 14
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = -low.diff()

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # When +DM > -DM, -DM = 0 and vice versa
            plus_dm_copy = plus_dm.copy()
            minus_dm_copy = minus_dm.copy()
            plus_dm_copy[(plus_dm < minus_dm)] = 0
            minus_dm_copy[(minus_dm < plus_dm)] = 0
            plus_dm = plus_dm_copy
            minus_dm = minus_dm_copy

            # Smoothed TR, +DM, -DM using Wilder's smoothing
            atr = tr.ewm(alpha=1/period, min_periods=period).mean()
            plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

            # Calculate +DI and -DI
            plus_di = 100 * plus_dm_smooth / atr
            minus_di = 100 * minus_dm_smooth / atr

            # Calculate DX
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, 0.0001)  # Avoid division by zero
            dx = 100 * abs(plus_di - minus_di) / di_sum

            # Calculate ADX (smoothed DX)
            adx = dx.ewm(alpha=1/period, min_periods=period).mean()

            # Return latest values
            adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
            plus_di_val = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
            minus_di_val = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

            return adx_val, plus_di_val, minus_di_val

        except Exception as e:
            self.log(f"[STRONG TREND] ADX calculation error for {symbol}: {e}", level="WARN")
            return 0.0, 0.0, 0.0

    def check_strong_trend_mode(self, symbol: str) -> tuple:
        """
        Check if we're in a strong trend based on ADX.
        Returns: (is_strong_trend, trend_direction)
        """
        adx, plus_di, minus_di = self.calculate_adx_for_symbol(symbol)

        self.current_adx[symbol] = adx
        self.current_plus_di[symbol] = plus_di
        self.current_minus_di[symbol] = minus_di

        if adx < self.adx_threshold:
            return False, None

        if plus_di > minus_di:
            return True, "UP"
        else:
            return True, "DOWN"

    def activate_strong_trend_mode(self, symbol: str, direction: str):
        """
        Activate strong trend mode for a symbol.
        - Block ALL DCA on loser side
        - BOOST winner side by 2x (add 1x to current position)
        """
        if self.boost_mode_active.get(symbol, False):
            self.log(f"[STRONG TREND] {symbol}: NOT activating - Boost Mode already active", level="INFO")
            return False

        if self.strong_trend_mode.get(symbol, False):
            return False

        self.strong_trend_mode[symbol] = True
        self.strong_trend_direction[symbol] = direction

        adx = self.current_adx.get(symbol, 0)
        winner_side = "LONG" if direction == "UP" else "SHORT"
        loser_side = "SHORT" if direction == "UP" else "LONG"

        self.log(f"[STRONG TREND] >>> {symbol} ACTIVATED! Direction: {direction} | ADX: {adx:.1f}")
        self.log(f"[STRONG TREND]     Winner ({winner_side}): 2x boost | Loser ({loser_side}): ALL DCA BLOCKED")

        # ================================================================
        # ACTUALLY BOOST THE WINNER SIDE: Add 1x to make it 2x total
        # This is the key feature - don't just log, actually increase position!
        # ================================================================
        winner_key = self.get_position_key(symbol, winner_side)
        winner_pos = self.positions.get(winner_key)

        if winner_pos and winner_pos.quantity > 0:
            # Already boosted for this trend? (Prevent double-boost)
            if self.trend_boosted_side.get(symbol) == winner_side:
                self.log(f"[STRONG TREND]     {winner_side} already boosted, skipping")
                return True

            try:
                boost_add_qty = winner_pos.quantity * 1.0  # Add 100% to make 2x

                # Round quantity to symbol precision
                symbol_config = SYMBOL_SETTINGS.get(symbol, {})
                qty_precision = symbol_config.get("qty_precision", 1)
                boost_add_qty = round(boost_add_qty, qty_precision)

                if boost_add_qty > 0:
                    # Place market order to add to the winner position
                    order_side = "BUY" if winner_side == "LONG" else "SELL"
                    self.log(f"[STRONG TREND]     Adding {boost_add_qty} to {winner_side} position (2x trend boost)")

                    boost_order = self.client.place_market_order(
                        symbol,
                        order_side,
                        boost_add_qty,
                        position_side=winner_side
                    )

                    if "orderId" in boost_order:
                        # Update position quantity
                        winner_pos.quantity += boost_add_qty
                        self.trend_boosted_side[symbol] = winner_side
                        self.log(f"[STRONG TREND]     SUCCESS: {winner_side} now has {winner_pos.quantity} qty (2x trend boost)")
                        # SAVE STATE after trend boost
                        self._save_position_state()
                    else:
                        self.log(f"[STRONG TREND]     WARNING: Failed to add trend boost qty: {boost_order}", level="WARN")
            except Exception as e:
                self.log(f"[STRONG TREND]     ERROR adding trend boost position: {e}", level="ERROR")
        else:
            self.log(f"[STRONG TREND]     No {winner_side} position to boost (will boost on next entry)")

        return True

    def deactivate_strong_trend_mode(self, symbol: str, reason: str):
        """Deactivate strong trend mode for a symbol."""
        if not self.strong_trend_mode.get(symbol, False):
            return

        self.log(f"[STRONG TREND] >>> {symbol} ENDED - {reason}")
        self.strong_trend_mode[symbol] = False
        self.strong_trend_direction[symbol] = None
        self.trend_boosted_side[symbol] = None

    def is_dca_blocked_by_strong_trend(self, symbol: str, position_side: str, dca_level: int) -> tuple:
        """
        Check if DCA should be blocked due to Strong Trend Mode.
        ALL DCA is blocked on the LOSER side during strong trends (ADX > 40).
        Returns: (is_blocked, reason)
        """
        # First check if strong trend mode is active
        if not self.strong_trend_mode.get(symbol, False):
            return False, None

        direction = self.strong_trend_direction.get(symbol)
        if not direction:
            return False, None

        # Determine loser side based on trend direction
        loser_side = "SHORT" if direction == "UP" else "LONG"

        # Block ALL DCA on loser side (regardless of DCA level)
        if position_side == loser_side:
            adx = self.current_adx.get(symbol, 0)
            reason = f"Strong {direction} trend (ADX: {adx:.1f}) - ALL DCA blocked on {position_side} (loser side)"
            return True, reason

        return False, None

    def cleanup_orphaned_orders(self):
        """
        Clean up orphaned/stale orders that don't match current positions.
        Runs periodically to catch orders that weren't cancelled properly.
        """
        try:
            for symbol in self.symbols:
                open_orders = self.client.get_open_orders(symbol)
                if not open_orders:
                    continue

                long_key = f"{symbol}_LONG"
                short_key = f"{symbol}_SHORT"
                long_pos = self.positions.get(long_key)
                short_pos = self.positions.get(short_key)

                for order in open_orders:
                    order_id = order.get("orderId")
                    order_position_side = order.get("positionSide", "BOTH")
                    order_type = order.get("type", "")

                    is_orphaned = False

                    if self.hedge_mode:
                        if order_position_side == "LONG":
                            if not long_pos:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found orphaned LONG order for {symbol}: {order_type} #{order_id}")
                            elif long_pos.stop_loss_order_id != order_id and long_pos.take_profit_order_id != order_id:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found untracked LONG order for {symbol}: {order_type} #{order_id}")
                        elif order_position_side == "SHORT":
                            if not short_pos:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found orphaned SHORT order for {symbol}: {order_type} #{order_id}")
                            elif short_pos.stop_loss_order_id != order_id and short_pos.take_profit_order_id != order_id:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found untracked SHORT order for {symbol}: {order_type} #{order_id}")
                    else:
                        pos = self.positions.get(symbol)
                        if not pos:
                            is_orphaned = True
                            self.log(f"[CLEANUP] Found orphaned order for {symbol}: {order_type} #{order_id}")

                    if is_orphaned:
                        try:
                            is_algo = order_type in ["TAKE_PROFIT_MARKET", "STOP_MARKET", "TRAILING_STOP_MARKET"]
                            self.client.cancel_order(symbol, order_id, is_algo_order=is_algo)
                            self.log(f"[CLEANUP] Cancelled orphaned order #{order_id} for {symbol}")
                        except Exception as e:
                            self.log(f"[CLEANUP] Could not cancel order #{order_id}: {e}", level="WARN")

        except Exception as e:
            self.log(f"[CLEANUP] Error during order cleanup: {e}", level="WARN")


    def _get_boost_trigger_level(self, symbol: str) -> int:
        """Get the DCA level that triggers boost mode for this symbol."""
        symbol_config = SYMBOL_SETTINGS.get(symbol, {})
        return symbol_config.get("boost_trigger_dca", 3)  # Default: DCA 3 (legacy, now ROI-based)

    def _get_boost_trigger_roi(self, symbol: str) -> float:
        """Get the ROI threshold that triggers boost mode. Default: -20%"""
        symbol_config = SYMBOL_SETTINGS.get(symbol, {})
        return symbol_config.get("boost_trigger_roi", 0.20)  # Default: 20% loss = -20% ROI

    def _check_roi_boost_activation(self, symbol: str, current_price: float):
        """
        ROI-BASED BOOST ACTIVATION (replaces DCA level-based)
        When one side's ROI drops below -20%, boost the opposite side.
        This triggers EARLIER than waiting for DCA 3 (-45%).
        """
        if not self.hedge_mode:
            return  # Boost only works in hedge mode

        # Already in boost mode for this symbol?
        if self.boost_mode_active.get(symbol, False):
            return

        # Get both positions
        long_key = self.get_position_key(symbol, "LONG")
        short_key = self.get_position_key(symbol, "SHORT")
        long_pos = self.positions.get(long_key)
        short_pos = self.positions.get(short_key)

        # Need both positions for boost mode
        if not long_pos or not short_pos:
            return

        # Get ROI trigger threshold
        roi_trigger = self._get_boost_trigger_roi(symbol)  # Default 0.20 = -20%

        # Calculate ROI for both positions
        leverage = STRATEGY_CONFIG.get("leverage", 20)

        # LONG ROI
        long_roi = 0
        if long_pos.avg_entry_price > 0 and long_pos.margin_used > 0:
            long_pnl = (current_price - long_pos.avg_entry_price) * long_pos.quantity
            long_roi = long_pnl / long_pos.margin_used  # Positive = profit, Negative = loss

        # SHORT ROI
        short_roi = 0
        if short_pos.avg_entry_price > 0 and short_pos.margin_used > 0:
            short_pnl = (short_pos.avg_entry_price - current_price) * short_pos.quantity
            short_roi = short_pnl / short_pos.margin_used

        # Check if either side hit the trigger (ROI below -20%)
        trigger_side = None
        boost_side = None

        if long_roi <= -roi_trigger:  # LONG at -20% or worse
            trigger_side = "LONG"
            boost_side = "SHORT"
            self.log(f"[BOOST ROI] {symbol}: LONG ROI {long_roi*100:.1f}% <= -{roi_trigger*100:.0f}% TRIGGER")
        elif short_roi <= -roi_trigger:  # SHORT at -20% or worse
            trigger_side = "SHORT"
            boost_side = "LONG"
            self.log(f"[BOOST ROI] {symbol}: SHORT ROI {short_roi*100:.1f}% <= -{roi_trigger*100:.0f}% TRIGGER")

        if not trigger_side:
            return  # Neither side hit trigger

        # IMPORTANT: Deactivate Strong Trend Mode when activating Boost Mode
        if self.strong_trend_mode.get(symbol, False):
            self.deactivate_strong_trend_mode(symbol, "Boost Mode taking over (ROI trigger)")

        # Get the position to boost
        boost_pos = long_pos if boost_side == "LONG" else short_pos

        # ACTIVATE BOOST MODE
        self.boost_mode_active[symbol] = True
        self.boosted_side[symbol] = boost_side
        self.boost_trigger_side[symbol] = trigger_side
        self.boost_locked_profit[symbol] = 0.0
        self.boost_cycle_count[symbol] = 0

        # Mark the boosted position
        boost_pos.is_boosted = True
        boost_pos.boost_multiplier = self.boost_multiplier

        self.log(f">>> [BOOST] {symbol}: ACTIVATED! {trigger_side} hit -{roi_trigger*100:.0f}% ROI -> {boost_side} now BOOSTED 1.5x")
        self.log(f"    [BOOST] Logic: At TP -> Close HALF, lock profit, add 0.5x, trailing starts")

        # ACTUALLY BOOST THE POSITION: Add 0.5x more to make it 1.5x total
        try:
            boost_add_qty = boost_pos.quantity * 0.5  # Add 50% to make 1.5x

            # Round quantity to symbol precision
            symbol_config = SYMBOL_SETTINGS.get(symbol, {})
            qty_precision = symbol_config.get("qty_precision", 1)
            boost_add_qty = round(boost_add_qty, qty_precision)

            if boost_add_qty > 0:
                # Place market order to add to the boosted position
                order_side = "SELL" if boost_side == "SHORT" else "BUY"
                self.log(f"    [BOOST] Adding {boost_add_qty} to {boost_side} position (0.5x boost)")

                boost_order = self.client.place_market_order(
                    symbol,
                    order_side,
                    boost_add_qty,
                    position_side=boost_side
                )

                if "orderId" in boost_order:
                    # Update position quantity
                    boost_pos.quantity += boost_add_qty
                    self.log(f"    [BOOST] SUCCESS: {boost_side} now has {boost_pos.quantity} qty (1.5x)")
                    # SAVE STATE after boost activation
                    self._save_position_state()
                else:
                    self.log(f"    [BOOST] WARNING: Failed to add boost qty: {boost_order}", level="WARN")
        except Exception as e:
            self.log(f"    [BOOST] ERROR adding boost position: {e}", level="ERROR")

    def _check_boost_activation(self, symbol: str, position: LivePosition, dca_level: int):
        """
        Check if this DCA level should trigger boost mode.
        When triggered, boost the OPPOSITE side position.
        """
        if not self.hedge_mode:
            return  # Boost only works in hedge mode

        # Already in boost mode for this symbol?
        if self.boost_mode_active.get(symbol, False):
            return

        # IMPORTANT: Deactivate Strong Trend Mode when activating Boost Mode
        # These two should NEVER be active together - Boost Mode takes priority
        if self.strong_trend_mode.get(symbol, False):
            self.deactivate_strong_trend_mode(symbol, "Boost Mode taking over")

        # Check if this DCA level triggers boost
        boost_trigger = self._get_boost_trigger_level(symbol)
        if dca_level != boost_trigger:
            return

        # Get the opposite side position
        opposite_side = "SHORT" if position.side == "LONG" else "LONG"
        opposite_key = self.get_position_key(symbol, opposite_side)
        opposite_pos = self.positions.get(opposite_key)

        if not opposite_pos:
            self.log(f"[BOOST] {symbol}: {position.side} hit DCA {dca_level} but no {opposite_side} position to boost")
            return

        # ACTIVATE BOOST MODE
        self.boost_mode_active[symbol] = True
        self.boosted_side[symbol] = opposite_side
        self.boost_trigger_side[symbol] = position.side
        self.boost_locked_profit[symbol] = 0.0
        self.boost_cycle_count[symbol] = 0

        # Mark the opposite position as boosted
        opposite_pos.is_boosted = True
        opposite_pos.boost_multiplier = self.boost_multiplier

        self.log(f">>> [BOOST] {symbol}: ACTIVATED! {position.side} hit DCA {dca_level} -> {opposite_side} now BOOSTED 1.5x")
        self.log(f"    [BOOST] Logic: At TP -> Close HALF, lock profit, add 0.5x, trailing starts")

        # ACTUALLY BOOST THE OPPOSITE SIDE: Add 0.5x more to make it 1.5x total
        try:
            boost_add_qty = opposite_pos.quantity * 0.5  # Add 50% to make 1.5x

            # Round quantity to symbol precision
            symbol_config = SYMBOL_SETTINGS.get(symbol, {})
            qty_precision = symbol_config.get("qty_precision", 1)
            boost_add_qty = round(boost_add_qty, qty_precision)

            if boost_add_qty > 0:
                # Place market order to add to the boosted position
                boost_side = "SELL" if opposite_side == "SHORT" else "BUY"
                self.log(f"    [BOOST] Adding {boost_add_qty} to {opposite_side} position (0.5x boost)")

                boost_order = self.client.place_market_order(
                    symbol,
                    boost_side,
                    boost_add_qty,
                    position_side=opposite_side
                )

                if "orderId" in boost_order:
                    # Update position quantity
                    opposite_pos.quantity += boost_add_qty
                    self.log(f"    [BOOST] SUCCESS: {opposite_side} now has {opposite_pos.quantity} qty (1.5x)")
                    # SAVE STATE after boost activation
                    self._save_position_state()
                else:
                    self.log(f"    [BOOST] WARNING: Failed to add boost qty: {boost_order}", level="WARN")
        except Exception as e:
            self.log(f"    [BOOST] ERROR adding boost position: {e}", level="ERROR")

    def _deactivate_boost_mode(self, symbol: str, reason: str):
        """Deactivate boost mode for a symbol."""
        if not self.boost_mode_active.get(symbol, False):
            return

        locked = self.boost_locked_profit.get(symbol, 0)
        cycles = self.boost_cycle_count.get(symbol, 0)

        self.log(f">>> [BOOST] {symbol}: ENDED - {reason}")
        self.log(f"    [BOOST] Summary: {cycles} half-close cycles | Locked profit: ${locked:+.2f}")

        # Reset boost state
        self.boost_mode_active[symbol] = False
        self.boosted_side[symbol] = None
        self.boost_trigger_side[symbol] = None
        self.boost_locked_profit[symbol] = 0.0
        self.boost_cycle_count[symbol] = 0

        # Reset position boost flags
        for side in ["LONG", "SHORT"]:
            pos_key = self.get_position_key(symbol, side)
            pos = self.positions.get(pos_key)
            if pos:
                pos.is_boosted = False
                pos.boost_multiplier = 1.0
                pos.half_close_count = 0

        # SAVE STATE after boost deactivation
        self._save_position_state()

    def _check_and_fix_inconsistencies(self):
        """
        SELF-HEALING: Check for and fix inconsistencies during each loop iteration.
        Detects and auto-fixes:
        1. Boost state mismatch (global vs position)
        2. DCA level inconsistencies
        3. Missing positions in tracking
        4. Boost mode that should be active but isn't
        """
        fixes_applied = []

        for symbol in self.symbols:
            # Check 1: Boost state consistency
            boost_active = self.boost_mode_active.get(symbol, False)
            boosted_side = self.boosted_side.get(symbol, None)

            if boost_active and boosted_side:
                pos_key = self.get_position_key(symbol, boosted_side)
                pos = self.positions.get(pos_key)
                if pos and not pos.is_boosted:
                    # FIX: Position should be boosted but isn't flagged
                    pos.is_boosted = True
                    pos.boost_multiplier = self.boost_multiplier
                    fixes_applied.append(f"[HEAL] {pos_key}: Applied missing boost flag")

            # Check 2: Position exists in Binance but not tracked locally
            for side in ["LONG", "SHORT"]:
                pos_key = self.get_position_key(symbol, side)
                if pos_key not in self.positions:
                    # Check if Binance has this position
                    try:
                        binance_pos = self.client.get_position(symbol, position_side=side)
                        if binance_pos and float(binance_pos.get("quantity", 0)) > 0:
                            fixes_applied.append(f"[HEAL] {pos_key}: Found untracked position on Binance - needs sync")
                    except:
                        pass

            # Check 3: Boost mode should be active based on DCA level
            for side in ["LONG", "SHORT"]:
                pos_key = self.get_position_key(symbol, side)
                pos = self.positions.get(pos_key)
                if pos and pos.dca_count >= self._get_boost_trigger_level(symbol):
                    opposite_side = "SHORT" if side == "LONG" else "LONG"
                    opposite_key = self.get_position_key(symbol, opposite_side)
                    opposite_pos = self.positions.get(opposite_key)

                    # If this side hit boost trigger but boost not active, it might need activation
                    if opposite_pos and not boost_active:
                        # Check if the opposite side should be boosted
                        fixes_applied.append(f"[HEAL] {symbol}: DCA{pos.dca_count} on {side} but no boost active - may need manual check")

        # Log any fixes
        if fixes_applied:
            for fix in fixes_applied:
                self.log(fix, level="WARN")
            self._save_position_state()

    def _check_boost_deactivation(self, symbol: str, closed_side: str, exit_type: str):
        """
        Check if boost mode should be deactivated.
        Deactivates when:
        1. TRIGGER side recovers (TP) or hits SL
        2. BOOSTED side hits SL (means trigger side recovered - price moved in trigger's favor)
        """
        if not self.boost_mode_active.get(symbol, False):
            return

        trigger_side = self.boost_trigger_side.get(symbol)
        boosted_side = self.boosted_side.get(symbol)

        if closed_side == trigger_side:
            # The losing side that triggered boost has recovered or stopped out
            self._deactivate_boost_mode(symbol, f"{trigger_side} {exit_type}")
        elif closed_side == boosted_side and exit_type == "SL":
            # BOOSTED side hit SL - this means price moved in trigger side's favor
            # (e.g., SHORT boosted hit SL because price went UP, so LONG should be recovering)
            self._deactivate_boost_mode(symbol, f"{boosted_side} SL (trigger side recovering)")

    def _stop_for_day(self, symbol: str):
        """
        STOP TRADING THIS SYMBOL FOR THE DAY after SL hit.
        Close remaining position and don't re-enter until next day.
        """
        self.stopped_for_day[symbol] = True
        self.sl_hit_date[symbol] = datetime.now()

        self.log(f">>> [STOP FOR DAY] {symbol}: SL hit - STOPPING trading until tomorrow", level="TRADE")

        # Close the OTHER side position if exists
        opposite_positions_to_close = []
        for pos_key, position in list(self.positions.items()):
            if position.symbol == symbol:
                opposite_positions_to_close.append((pos_key, position))

        for pos_key, position in opposite_positions_to_close:
            try:
                # Get current price
                price_data = self.client.get_current_price(symbol)
                current_price = price_data["price"]

                # Close position
                close_side = "SELL" if position.side == "LONG" else "BUY"
                hedge_position_side = position.side if self.hedge_mode else None

                close_order = self.client.place_market_order(
                    symbol, close_side, position.quantity, position_side=hedge_position_side
                )

                if "orderId" in close_order:
                    fill_price = float(close_order.get("avgPrice", current_price))
                    if position.side == "LONG":
                        pnl = (fill_price - position.avg_entry_price) * position.quantity
                    else:
                        pnl = (position.avg_entry_price - fill_price) * position.quantity

                    self.log(f"    [STOP DAY] Closed {position.side} @ ${fill_price:,.4f} | P&L: ${pnl:+.2f}", level="TRADE")

                    # Update daily stats
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    if pnl >= 0:
                        self.daily_wins += 1
                    else:
                        self.daily_losses += 1

                    # Log trade
                    self.log_trade(
                        symbol=symbol,
                        side=position.side,
                        entry_price=position.avg_entry_price,
                        exit_price=fill_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        exit_type="STOP_DAY",
                        dca_level=position.dca_count
                    )

                # Cancel all orders for this symbol
                try:
                    self.client.cancel_all_orders(symbol)
                    self.log(f"    [STOP DAY] Cancelled all orders for {symbol}")
                except Exception as e:
                    self.log(f"    [STOP DAY] Warning: Could not cancel orders: {e}", level="WARN")

                # Release margin
                margin_released = position.margin_used
                current_margin = self.symbol_margin_used.get(symbol, 0)
                self.symbol_margin_used[symbol] = max(0, current_margin - margin_released)

                # Remove from tracking
                del self.positions[pos_key]

            except Exception as e:
                self.log(f"    [STOP DAY] Error closing {position.side}: {e}", level="ERROR")

        # Deactivate boost mode if active
        if self.boost_mode_active.get(symbol, False):
            self._deactivate_boost_mode(symbol, "Stopped for day")

        # Clear pending re-entries
        if symbol in self.pending_reentry:
            del self.pending_reentry[symbol]
        if f"{symbol}_LONG" in self.pending_reentry:
            del self.pending_reentry[f"{symbol}_LONG"]
        if f"{symbol}_SHORT" in self.pending_reentry:
            del self.pending_reentry[f"{symbol}_SHORT"]

        self.log(f">>> [STOP FOR DAY] {symbol}: All positions closed. Will restart tomorrow.", level="TRADE")

        # Save state
        self._save_position_state()

    def _check_new_day_restart(self, symbol: str) -> bool:
        """
        Check if it's a new day and we should restart trading for a stopped symbol.
        Returns True if trading should restart.
        """
        if not self.stopped_for_day.get(symbol, False):
            return False

        sl_date = self.sl_hit_date.get(symbol)
        if sl_date is None:
            return False

        current_date = datetime.now().date()
        sl_hit_date = sl_date.date() if hasattr(sl_date, 'date') else sl_date

        if current_date > sl_hit_date:
            # New day - restart trading
            self.stopped_for_day[symbol] = False
            self.sl_hit_date[symbol] = None
            self.log(f">>> [NEW DAY] {symbol}: Restarting trading after yesterday's SL stop", level="TRADE")
            return True

        return False

    def _handle_boosted_tp(self, symbol: str, position: LivePosition, current_price: float) -> bool:
        """
        Handle TP for BOOSTED positions with half-close cycle:
        1. Close 50% of position -> lock profit
        2. Keep 50% running with trailing SL
        3. Re-enter NEW position at 1.5x size

        Returns True if half-close was executed, False otherwise.
        """
        if not position.is_boosted:
            return False

        pos_key = self.get_position_key(symbol, position.side)

        # Calculate 50% quantity to close
        half_qty = position.quantity / 2

        # Round to symbol precision
        symbol_config = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = symbol_config.get("qty_precision", 1)
        half_qty = round(half_qty, qty_precision)
        remaining_qty = round(position.quantity - half_qty, qty_precision)

        if half_qty <= 0:
            self.log(f"[BOOST HALF-CLOSE] {symbol}: Quantity too small to split", level="WARN")
            return False

        self.log(f">>> [BOOST HALF-CLOSE] {symbol} {position.side}: TP HIT!", level="TRADE")
        self.log(f"    [BOOST] Closing 50%: {half_qty} qty (keeping {remaining_qty})", level="TRADE")

        try:
            # 1. CLOSE 50% - Lock profit
            close_side = "SELL" if position.side == "LONG" else "BUY"
            hedge_position_side = position.side if self.hedge_mode else None

            close_order = self.client.place_market_order(
                symbol, close_side, half_qty, position_side=hedge_position_side
            )

            if "orderId" not in close_order:
                self.log(f"    [BOOST] FAILED to close 50%: {close_order}", level="ERROR")
                return False

            fill_price = float(close_order.get("avgPrice", current_price))

            # Calculate P&L for the closed portion
            if position.side == "LONG":
                half_pnl = (fill_price - position.avg_entry_price) * half_qty
            else:
                half_pnl = (position.avg_entry_price - fill_price) * half_qty

            self.log(f"    [BOOST] Closed 50% @ ${fill_price:,.4f} | P&L: ${half_pnl:+.2f}", level="TRADE")

            # Lock profit
            self.boost_locked_profit[symbol] = self.boost_locked_profit.get(symbol, 0) + half_pnl
            position.half_close_count += 1
            self.boost_cycle_count[symbol] = self.boost_cycle_count.get(symbol, 0) + 1

            # Update position quantity
            position.quantity = remaining_qty

            # Log trade
            self.log_trade(
                symbol=symbol,
                side=position.side,
                entry_price=position.avg_entry_price,
                exit_price=fill_price,
                quantity=half_qty,
                pnl=half_pnl,
                exit_type="BOOST_HALF_CLOSE",
                dca_level=position.dca_count
            )

            # Update daily stats
            self.daily_pnl += half_pnl
            self.daily_trades += 1
            if half_pnl > 0:
                self.daily_wins += 1
            else:
                self.daily_losses += 1

            # 2. ACTIVATE TRAILING SL on remaining 50%
            position.trailing_active = True
            position.peak_roi = 0.0  # Reset peak for trailing
            self.log(f"    [BOOST] Remaining 50% now has TRAILING SL active", level="TRADE")

            # 3. RE-ENTER at 1.5x size
            # Calculate new position size (1.5x of original base)
            base_qty = remaining_qty  # The remaining 50% is the base
            new_entry_qty = round(base_qty * 1.5, qty_precision)  # 1.5x

            if new_entry_qty > 0:
                entry_side = "BUY" if position.side == "LONG" else "SELL"
                self.log(f"    [BOOST] Re-entering at 1.5x: {new_entry_qty} qty", level="TRADE")

                reentry_order = self.client.place_market_order(
                    symbol, entry_side, new_entry_qty, position_side=position.side
                )

                if "orderId" in reentry_order:
                    reentry_price = float(reentry_order.get("avgPrice", current_price))

                    # Update position with new average entry
                    total_qty = remaining_qty + new_entry_qty
                    # Weighted average entry price
                    new_avg_entry = (
                        (position.avg_entry_price * remaining_qty) +
                        (reentry_price * new_entry_qty)
                    ) / total_qty

                    position.quantity = total_qty
                    position.avg_entry_price = new_avg_entry

                    self.log(f"    [BOOST] SUCCESS: New total qty {total_qty} @ avg ${new_avg_entry:,.4f}", level="TRADE")
                    self.log(f"    [BOOST] Cycle #{position.half_close_count} complete | Total locked: ${self.boost_locked_profit.get(symbol, 0):+.2f}", level="TRADE")
                else:
                    self.log(f"    [BOOST] WARNING: Re-entry failed: {reentry_order}", level="WARN")

            # Update SL/TP orders for the new position
            pos_dict = {
                "symbol": symbol,
                "side": position.side,
                "entry_price": position.avg_entry_price,
                "quantity": position.quantity
            }

            # Cancel old orders and place new ones
            try:
                # Cancel only orders for this position side
                orders = self.client.get_open_orders(symbol)
                for order in orders:
                    if order.get("positionSide") == position.side:
                        try:
                            self.client.cancel_order(symbol, order["orderId"], is_algo_order=True)
                        except:
                            pass
            except:
                pass

            self._ensure_sl_tp_orders(symbol, pos_dict, position.dca_count, position_key=pos_key)

            # Save state
            self._save_position_state()

            return True

        except Exception as e:
            self.log(f"    [BOOST] ERROR in half-close cycle: {e}", level="ERROR")
            return False

    def check_daily_reset(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.log(f"Daily reset - Previous day P&L: ${self.daily_pnl:.2f} (W:{self.daily_wins}/L:{self.daily_losses})")
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0

            # Reset per-symbol stats
            if hasattr(self, "symbol_stats"):
                for symbol in self.symbols:
                    self.symbol_stats[symbol] = {
                        "wins": 0,
                        "losses": 0,
                        "tp_count": 0,
                        "sl_count": 0,
                        "pnl": 0.0
                    }
            current_balance = self.client.get_balance()
            self.daily_start_balance = current_balance  # Reset daily loss limit baseline
            self.last_reset_date = today

    def can_trade(self, symbol: str) -> tuple:
        """
        Check if trading is allowed.
        Returns (can_trade: bool, reason: str)
        """
        try:
            # Check daily trade limit
            if self.daily_trades >= STRATEGY_CONFIG["max_trades_per_day"]:
                return False, "DAILY_LIMIT"

            # Check daily loss limit (use daily_start_balance, not session start)
            current_balance = self.client.get_balance()
            if self.daily_start_balance > 0:
                daily_loss_pct = (self.daily_start_balance - current_balance) / self.daily_start_balance
                if daily_loss_pct >= RISK_CONFIG["max_daily_loss_pct"]:
                    return False, "DAILY_LOSS_LIMIT"

            # Check max positions
            positions = self.client.get_positions()
            if len(positions) >= RISK_CONFIG["max_total_positions"]:
                return False, "MAX_POSITIONS"

            # Check if already have position in this symbol
            # In hedge mode, we can have BOTH LONG and SHORT, so check differently
            if self.hedge_mode:
                # In hedge mode, we allow up to 2 positions per symbol (LONG + SHORT)
                has_long = f"{symbol}_LONG" in self.positions
                has_short = f"{symbol}_SHORT" in self.positions
                if has_long and has_short:
                    return False, "HAS_BOTH_POSITIONS"
            else:
                # Normal mode - only one position per symbol
                if symbol in self.positions:
                    return False, "HAS_POSITION"

            # Check symbol cooldown
            last_check = self.last_check_time.get(symbol, datetime.min)
            cooldown_remaining = self.symbol_check_interval - (datetime.now() - last_check).total_seconds()
            if cooldown_remaining > 0:
                return False, f"COOLDOWN_{int(cooldown_remaining)}s"

            return True, "OK"
        except Exception as e:
            # If we can't check, allow trading but log the error
            self.log(f"Warning: can_trade check failed for {symbol}: {e}", level="WARN")
            return True, "OK"

    # =========================================================================
    # HYBRID HOLD + TRADE SYSTEM - TREND DETECTION
    # =========================================================================

    def detect_trend(self, symbol: str, df: pd.DataFrame = None, strict: bool = True) -> str:
        """
        Detect current market trend for a symbol.

        Args:
            symbol: Trading pair
            df: Optional DataFrame with OHLCV data
            strict: If True, use strict detection (for trend flip detection)
                    If False, use simple detection (for auto-entry - ALWAYS returns a direction)

        STRICT MODE (for detecting trend changes):
        - EMA crossover with separation threshold
        - Price position relative to EMAs
        - RSI confirmation
        - ADX trend strength
        - Multiple candles confirming
        - Can return "NEUTRAL"

        SIMPLE MODE (for auto-entry):
        - Just looks at EMA direction
        - ALWAYS returns "BULLISH" or "BEARISH" (never NEUTRAL)

        Returns: "BULLISH", "BEARISH", or "NEUTRAL" (only in strict mode)
        """
        try:
            # Get market data if not provided
            if df is None:
                market_data = self.client.get_market_data(symbol)
                if not market_data or "1m" not in market_data:
                    return "BULLISH" if not strict else "NEUTRAL"  # Default to BULLISH for auto-entry
                df = market_data["1m"]

            if df is None or len(df) < 50:
                return "BULLISH" if not strict else "NEUTRAL"

            # Get trend detection config
            trend_config = DCA_CONFIG.get("trend_detection", {})
            ema_fast_period = trend_config.get("ema_fast", 12)
            ema_slow_period = trend_config.get("ema_slow", 26)

            # Calculate EMAs
            ema_fast = df["close"].ewm(span=ema_fast_period, adjust=False).mean()
            ema_slow = df["close"].ewm(span=ema_slow_period, adjust=False).mean()

            current_ema_fast = ema_fast.iloc[-1]
            current_ema_slow = ema_slow.iloc[-1]

            # Check for NaN
            if pd.isna(current_ema_fast) or pd.isna(current_ema_slow):
                return "BULLISH" if not strict else "NEUTRAL"

            # ================================================================
            # SIMPLE MODE: Just EMA direction (for auto-entry)
            # Always returns a direction - NEVER neutral
            # ================================================================
            if not strict:
                if current_ema_fast >= current_ema_slow:
                    return "BULLISH"
                else:
                    return "BEARISH"

            # ================================================================
            # STRICT MODE: Full confirmation (for trend change detection)
            # ================================================================
            bullish_rsi_min = trend_config.get("bullish_rsi_min", 50)
            bearish_rsi_max = trend_config.get("bearish_rsi_max", 50)
            confirmation_candles = trend_config.get("confirmation_candles", 5)
            ema_separation_pct = trend_config.get("ema_separation_pct", 0.002)
            use_adx = trend_config.get("use_adx_filter", True)
            adx_min = trend_config.get("adx_min_trend", 20)

            # Calculate RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            if pd.isna(current_rsi):
                return "NEUTRAL"

            # Calculate ADX for trend strength
            adx_value = 25  # Default if not calculated
            if use_adx:
                try:
                    high_low = df["high"] - df["low"]
                    high_close = (df["high"] - df["close"].shift()).abs()
                    low_close = (df["low"] - df["close"].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()

                    plus_dm = df["high"].diff()
                    minus_dm = df["low"].diff().abs() * -1
                    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
                    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

                    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
                    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
                    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
                    adx = dx.rolling(window=14).mean()
                    adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
                except:
                    adx_value = 25  # Default on error

            # Get latest values
            current_price = df["close"].iloc[-1]

            # Calculate EMA separation
            ema_separation = abs(current_ema_fast - current_ema_slow) / current_ema_slow

            # Check 1: EMA separation must be significant (avoid noise at crossover)
            if ema_separation < ema_separation_pct:
                return "NEUTRAL"  # EMAs too close, no clear trend

            # Check 2: ADX must show trending market
            if use_adx and adx_value < adx_min:
                return "NEUTRAL"  # Market not trending strongly enough

            # Check 3: Multiple candles must confirm (look back N candles)
            bullish_confirms = 0
            bearish_confirms = 0
            for i in range(-confirmation_candles, 0):
                if i >= -len(ema_fast):
                    if ema_fast.iloc[i] > ema_slow.iloc[i]:
                        bullish_confirms += 1
                    elif ema_fast.iloc[i] < ema_slow.iloc[i]:
                        bearish_confirms += 1

            # Need ALL confirmation candles to agree
            strong_bullish = bullish_confirms >= confirmation_candles
            strong_bearish = bearish_confirms >= confirmation_candles

            # BULLISH: All conditions met
            if (strong_bullish and
                current_ema_fast > current_ema_slow and
                current_price > current_ema_slow and
                current_rsi > bullish_rsi_min):
                return "BULLISH"

            # BEARISH: All conditions met
            if (strong_bearish and
                current_ema_fast < current_ema_slow and
                current_price < current_ema_slow and
                current_rsi < bearish_rsi_max):
                return "BEARISH"

            return "NEUTRAL"

        except Exception as e:
            self.log(f"Trend detection error for {symbol}: {e}", level="WARN")
            return "BULLISH" if not strict else "NEUTRAL"

    def check_trend_change(self, symbol: str, position: LivePosition) -> bool:
        """
        Check if trend has changed against our position (WITH COOLDOWN).
        Returns True if we should close and flip.

        LONG position + BEARISH trend = Close and flip to SHORT
        SHORT position + BULLISH trend = Close and flip to LONG
        """
        try:
            hybrid_config = DCA_CONFIG.get("hybrid_hold", {})
            trend_config = DCA_CONFIG.get("trend_detection", {})

            if not hybrid_config.get("flip_on_trend_change", True):
                return False

            # ================================================================
            # COOLDOWN CHECK: Don't flip too soon after last flip
            # ================================================================
            flip_cooldown_minutes = trend_config.get("flip_cooldown_minutes", 30)
            if symbol in self.last_flip_time:
                time_since_flip = (datetime.now() - self.last_flip_time[symbol]).total_seconds() / 60
                if time_since_flip < flip_cooldown_minutes:
                    # Still in cooldown, don't check for flip
                    return False

            # Get market data for trend detection
            market_data = self.client.get_market_data(symbol)
            if not market_data or "1m" not in market_data:
                return False

            df = market_data["1m"]
            current_trend = self.detect_trend(symbol, df)

            # Store current trend
            self.current_trend[symbol] = current_trend

            # If trend is NEUTRAL, don't flip (strict detection didn't confirm)
            if current_trend == "NEUTRAL":
                return False

            # Check if trend is opposite to our position
            if position.side == "LONG" and current_trend == "BEARISH":
                self.log(f"TREND CHANGE {symbol}: Strong BEARISH detected while LONG (confirmed)", level="TRADE")
                self.last_flip_time[symbol] = datetime.now()  # Record flip time
                return True

            if position.side == "SHORT" and current_trend == "BULLISH":
                self.log(f"TREND CHANGE {symbol}: Strong BULLISH detected while SHORT (confirmed)", level="TRADE")
                self.last_flip_time[symbol] = datetime.now()  # Record flip time
                return True

            return False

        except Exception as e:
            self.log(f"Trend change check error for {symbol}: {e}", level="WARN")
            return False

    def auto_enter_hedge_positions(self, symbol: str) -> bool:
        """
        HEDGE MODE: Auto-enter BOTH LONG and SHORT positions on the same symbol.
        Each side gets 50% of the symbol budget.

        Returns True if positions were entered.
        """
        try:
            # CHECK IF STOPPED FOR DAY - Don't enter if symbol is stopped
            if self.stopped_for_day.get(symbol, False):
                self.log(f"HEDGE AUTO-ENTER: {symbol} SKIPPED - stopped for the day", level="WARN")
                return False

            hedge_config = DCA_CONFIG.get("hedge_mode", {})
            if not hedge_config.get("enabled", True):
                return False

            # Check if already have positions on this symbol
            long_key = f"{symbol}_LONG"
            short_key = f"{symbol}_SHORT"

            entered_any = False

            # Enter LONG if not already present
            if long_key not in self.positions:
                can, reason = self.can_trade(symbol)
                if can or "HAS_POSITION" in reason:
                    self.log(f"HEDGE AUTO-ENTER: Opening LONG {symbol}", level="TRADE")
                    long_signal = TradingSignal(
                        signal="BUY",
                        signal_type=SignalType.MOMENTUM,
                        confidence=0.8,
                        reason="HEDGE_AUTO_ENTER_LONG",
                        momentum_value=0.5,
                        ema_trend="BULLISH",
                        rsi_value=50.0,
                        adx_value=25.0
                    )
                    self.enter_position(symbol, long_signal, is_hedge=True)
                    entered_any = True
                    time.sleep(0.5)  # Brief pause between orders

            # Enter SHORT if not already present
            if short_key not in self.positions:
                can, reason = self.can_trade(symbol)
                if can or "HAS_POSITION" in reason:
                    self.log(f"HEDGE AUTO-ENTER: Opening SHORT {symbol}", level="TRADE")
                    short_signal = TradingSignal(
                        signal="SELL",
                        signal_type=SignalType.MOMENTUM,
                        confidence=0.8,
                        reason="HEDGE_AUTO_ENTER_SHORT",
                        momentum_value=0.5,
                        ema_trend="BEARISH",
                        rsi_value=50.0,
                        adx_value=25.0
                    )
                    self.enter_position(symbol, short_signal, is_hedge=True)
                    entered_any = True

            if entered_any:
                self.log(f"HEDGE: Opened positions for {symbol} (LONG + SHORT)", level="TRADE")

            return entered_any

        except Exception as e:
            self.log(f"Hedge auto-enter error for {symbol}: {e}", level="ERROR")
            return False

    def auto_enter_on_trend(self, symbol: str) -> bool:
        """
        Auto-enter position based on detected trend.
        Called on startup or after trend flip to immediately enter market.

        Returns True if position was entered.
        """
        try:
            hybrid_config = DCA_CONFIG.get("hybrid_hold", {})
            if not hybrid_config.get("auto_enter_on_start", True):
                return False

            # Skip if already have position
            if symbol in self.positions:
                return False

            # Check if can trade
            can, reason = self.can_trade(symbol)
            if not can:
                self.log(f"Auto-enter {symbol} blocked: {reason}", level="INFO")
                return False

            # Get market data
            market_data = self.client.get_market_data(symbol)
            if not market_data or "1m" not in market_data:
                self.log(f"Auto-enter {symbol}: No market data available", level="WARN")
                return False

            df = market_data["1m"]
            # Use SIMPLE mode (strict=False) for auto-entry - ALWAYS get a direction
            trend = self.detect_trend(symbol, df, strict=False)

            # Store trend
            self.current_trend[symbol] = trend

            # Create signal based on trend
            signal_type = "BUY" if trend == "BULLISH" else "SELL"

            # Create a TradingSignal object for enter_position
            signal = TradingSignal(
                signal=signal_type,
                signal_type=SignalType.MOMENTUM,
                confidence=0.7,
                reason="HYBRID_AUTO_ENTER",
                momentum_value=0.5,
                ema_trend=trend,
                rsi_value=50.0,
                adx_value=25.0
            )

            self.log(f"AUTO-ENTER {symbol}: Detected {trend} trend, entering {signal_type}", level="TRADE")

            # Enter position
            self.enter_position(symbol, signal)

            return True

        except Exception as e:
            self.log(f"Auto-enter error for {symbol}: {e}", level="ERROR")
            return False

    def close_and_flip(self, symbol: str, position: LivePosition, new_trend: str):
        """
        Close current position and immediately enter opposite direction.
        Called when trend changes.
        """
        try:
            self.log(f"FLIP {symbol}: Closing {position.side}, entering {new_trend}", level="TRADE")

            # Cancel existing SL/TP orders
            if position.stop_loss_order_id:
                try:
                    self.client.cancel_order(symbol, position.stop_loss_order_id, is_algo_order=True)
                except:
                    pass

            if position.take_profit_order_id:
                try:
                    self.client.cancel_order(symbol, position.take_profit_order_id, is_algo_order=True)
                except:
                    pass

            # Close current position (in hedge mode, include positionSide)
            close_side = "SELL" if position.side == "LONG" else "BUY"
            position_side = position.side if self.hedge_mode else None

            order_result = self.client.place_market_order(symbol, close_side, position.quantity, position_side=position_side)

            if "orderId" in order_result:
                fill_price = float(order_result.get("avgPrice", 0)) or current_price
                if fill_price == 0:
                    price_data = self.client.get_current_price(symbol)
                    fill_price = price_data["price"]

                # Calculate P&L
                if position.side == "LONG":
                    pnl = (fill_price - position.avg_entry_price) * position.quantity
                else:
                    pnl = (position.avg_entry_price - fill_price) * position.quantity

                # Update daily stats
                self.daily_pnl += pnl
                if pnl > 0:
                    self.daily_wins += 1
                else:
                    self.daily_losses += 1
                self.daily_trades += 1

                # Log the close
                self.log(
                    f"FLIP CLOSE {symbol}: {position.side} @ ${fill_price:,.4f} | "
                    f"P&L: ${pnl:+,.2f} | Reason: TREND_CHANGE to {new_trend}",
                    level="TRADE"
                )

                # Log trade
                self.log_trade(
                    symbol=symbol,
                    side=position.side,
                    entry_price=position.avg_entry_price,
                    exit_price=fill_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    exit_type="TREND_FLIP",
                    dca_level=position.dca_count
                )

                # Release margin
                self.symbol_margin_used[symbol] = 0.0

                # Check if boost mode should be deactivated
                self._check_boost_deactivation(symbol, position.side, "TREND_FLIP")

                # Remove position
                del self.positions[symbol]

            # Wait a moment for exchange to process
            time.sleep(0.5)

            # Now enter new position in opposite direction
            new_signal_type = "BUY" if new_trend == "BULLISH" else "SELL"

            signal = TradingSignal(
                signal=new_signal_type,
                signal_type=SignalType.MOMENTUM,
                confidence=0.7,
                reason="HYBRID_TREND_FLIP",
                momentum_value=0.5,
                ema_trend=new_trend,
                rsi_value=50.0,
                adx_value=25.0
            )

            self.log(f"FLIP ENTER {symbol}: Entering {new_signal_type} following {new_trend} trend", level="TRADE")
            self.enter_position(symbol, signal)

        except Exception as e:
            self.log(f"Close and flip error for {symbol}: {e}", level="ERROR")

    def hedge_on_trend_flip(self, symbol: str, position: LivePosition, new_trend: str):
        """
        Hedge strategy for DCA positions when trend flips.
        Instead of closing at a loss, we:
        1. Tighten SL to minimize max loss
        2. Open hedge position in new trend direction (initial size)
        3. Set breakeven TP on losing position
        4. Wait for breakeven exit, then keep hedge running
        """
        try:
            hybrid_config = DCA_CONFIG.get("hybrid_hold", {})
            leverage = STRATEGY_CONFIG["leverage"]

            self.log(
                f"HEDGE STRATEGY {symbol}: DCA L{position.dca_count} {position.side} | "
                f"Opening {new_trend} hedge instead of closing at loss",
                level="TRADE"
            )

            # ================================================================
            # STEP 1: Cancel old SL/TP orders
            # ================================================================
            try:
                self.client.cancel_all_orders(symbol)
                self.log(f"HEDGE: Cancelled all existing orders for {symbol}")
            except Exception as e:
                self.log(f"HEDGE: Could not cancel orders: {e}", level="WARN")

            # ================================================================
            # STEP 2: Place TIGHT SL to minimize max loss
            # ================================================================
            tight_sl_roi = hybrid_config.get("hedge_tighten_sl_roi", 0.15)  # -15% ROI
            sl_price_pct = tight_sl_roi / leverage

            if position.side == "LONG":
                new_sl_price = position.avg_entry_price * (1 - sl_price_pct)
                sl_side = "SELL"
            else:
                new_sl_price = position.avg_entry_price * (1 + sl_price_pct)
                sl_side = "BUY"

            # In hedge mode, include positionSide
            hedge_position_side = position.side if self.hedge_mode else None
            sl_order = self.client.place_stop_loss(symbol, sl_side, position.quantity, new_sl_price, position_side=hedge_position_side)
            if "orderId" in sl_order:
                position.stop_loss_order_id = sl_order.get("orderId")
                self.log(f"HEDGE: Tight SL placed @ ${new_sl_price:,.4f} (-{tight_sl_roi*100:.0f}% ROI)")
            else:
                self.log(f"HEDGE: Failed to place tight SL: {sl_order}", level="ERROR")

            # ================================================================
            # STEP 3: Place BREAKEVEN TP on losing position
            # ================================================================
            breakeven_buffer = hybrid_config.get("hedge_breakeven_buffer", 0.005)  # 0.5%

            if position.side == "LONG":
                # For LONG, exit slightly above entry (breakeven + buffer)
                breakeven_tp_price = position.avg_entry_price * (1 + breakeven_buffer)
                tp_side = "SELL"
            else:
                # For SHORT, exit slightly below entry (breakeven + buffer)
                breakeven_tp_price = position.avg_entry_price * (1 - breakeven_buffer)
                tp_side = "BUY"

            tp_order = self.client.place_take_profit(symbol, tp_side, position.quantity, breakeven_tp_price, position_side=hedge_position_side)
            if "orderId" in tp_order:
                position.take_profit_order_id = tp_order.get("orderId")
                self.log(f"HEDGE: Breakeven TP placed @ ${breakeven_tp_price:,.4f} (entry was ${position.avg_entry_price:,.4f})")
            else:
                self.log(f"HEDGE: Failed to place breakeven TP: {tp_order}", level="ERROR")

            # ================================================================
            # STEP 4: Mark position as hedged and waiting for breakeven
            # ================================================================
            position.is_hedged = True
            position.hedge_side = "LONG" if new_trend == "BULLISH" else "SHORT"
            position.waiting_for_breakeven = True
            position.hedge_start_time = datetime.now()

            # ================================================================
            # STEP 5: Open HEDGE position in new trend direction
            # This uses the INITIAL entry size (small position)
            # ================================================================
            hedge_signal_type = "BUY" if new_trend == "BULLISH" else "SELL"

            signal = TradingSignal(
                signal=hedge_signal_type,
                signal_type=SignalType.MOMENTUM,
                confidence=0.7,
                reason="HYBRID_HEDGE_ENTRY",
                momentum_value=0.5,
                ema_trend=new_trend,
                rsi_value=50.0,
                adx_value=25.0
            )

            self.log(
                f"HEDGE ENTER {symbol}: Opening {hedge_signal_type} hedge following {new_trend} trend",
                level="TRADE"
            )

            # Note: enter_position will create a NEW position entry
            # The old position stays open with tight SL and breakeven TP
            # Binance hedge mode allows both LONG and SHORT simultaneously
            self.enter_position(symbol, signal, is_hedge=True)

            self.log(
                f"HEDGE ACTIVE {symbol}: {position.side} waiting for breakeven | "
                f"{position.hedge_side} hedge running",
                level="TRADE"
            )

        except Exception as e:
            self.log(f"Hedge on trend flip error for {symbol}: {e}", level="ERROR")
            # Fallback: just do normal close and flip
            self.log(f"HEDGE FALLBACK: Executing normal close and flip", level="WARN")
            self.close_and_flip(symbol, position, new_trend)

    def manage_hedged_position(self, symbol: str, position: LivePosition, current_price: float):
        """
        Manage a hedged position - check if breakeven was hit or max wait time exceeded.
        Called from manage_positions() for hedged positions.
        """
        if not position.is_hedged or not position.waiting_for_breakeven:
            return False  # Not a hedged position waiting for exit

        hybrid_config = DCA_CONFIG.get("hybrid_hold", {})
        max_wait_hours = hybrid_config.get("hedge_max_wait_hours", 4)

        # Check how long we've been waiting
        if position.hedge_start_time:
            wait_time = datetime.now() - position.hedge_start_time
            wait_hours = wait_time.total_seconds() / 3600

            if wait_hours >= max_wait_hours:
                # Max wait time exceeded - close at current price (cut loss)
                self.log(
                    f"HEDGE TIMEOUT {symbol}: Waited {wait_hours:.1f}h (max {max_wait_hours}h) | "
                    f"Closing {position.side} at market",
                    level="TRADE"
                )

                # Close the losing position at market (in hedge mode, include positionSide)
                close_side = "SELL" if position.side == "LONG" else "BUY"
                hedge_position_side = position.side if self.hedge_mode else None
                order_result = self.client.place_market_order(symbol, close_side, position.quantity, position_side=hedge_position_side)

                if "orderId" in order_result:
                    fill_price = float(order_result.get("avgPrice", current_price))

                    # Calculate P&L
                    if position.side == "LONG":
                        pnl = (fill_price - position.avg_entry_price) * position.quantity
                    else:
                        pnl = (position.avg_entry_price - fill_price) * position.quantity

                    # Update daily stats
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    if pnl > 0:
                        self.daily_wins += 1
                    else:
                        self.daily_losses += 1

                    self.log(
                        f"HEDGE CLOSED {symbol}: {position.side} @ ${fill_price:,.4f} | "
                        f"P&L: ${pnl:+,.2f} | Reason: HEDGE_TIMEOUT",
                        level="TRADE"
                    )

                    # Log trade
                    self.log_trade(
                        symbol=symbol,
                        side=position.side,
                        entry_price=position.avg_entry_price,
                        exit_price=fill_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        exit_type="HEDGE_TIMEOUT",
                        dca_level=position.dca_count
                    )

                    # Cancel any remaining orders for this side
                    try:
                        if position.stop_loss_order_id:
                            self.client.cancel_order(symbol, position.stop_loss_order_id, is_algo_order=True)
                        if position.take_profit_order_id:
                            self.client.cancel_order(symbol, position.take_profit_order_id, is_algo_order=True)
                    except:
                        pass

                    # Check if boost mode should be deactivated
                    self._check_boost_deactivation(symbol, position.side, "TP")

                    # Remove from tracking (hedge position continues)
                    self.symbol_margin_used[symbol] -= position.margin_used
                    del self.positions[symbol]

                    return True  # Position was closed

        return False  # Position still waiting

    def reenter_after_tp(self, symbol: str, closed_side: str):
        """
        Re-enter position in same direction after TP hit.
        Keeps us in the trend after taking profit.
        ALWAYS HOLD STRATEGY: Re-enter immediately, no momentum check needed!
        """
        try:
            # CHECK IF STOPPED FOR DAY - Don't re-enter if symbol is stopped
            if self.stopped_for_day.get(symbol, False):
                self.log(f"Re-enter {symbol}: BLOCKED - stopped for the day (SL hit)", level="WARN")
                return

            hybrid_config = DCA_CONFIG.get("hybrid_hold", {})
            if not hybrid_config.get("reenter_on_tp", True):
                self.log(f"Re-enter {symbol}: Disabled in config", level="INFO")
                return

            # Brief pause to let position clear on Binance side
            time.sleep(1.0)

            # Check if can trade (but ignore HAS_POSITION since we just cleared it)
            can, reason = self.can_trade(symbol)
            if not can and "HAS_POSITION" not in reason:
                self.log(f"Re-enter {symbol} blocked: {reason} - adding to pending queue", level="INFO")
                # Add to pending re-entry for next scan cycle
                self.pending_reentry[symbol] = closed_side
                return

            # Optional: Check trend for direction (but ALWAYS re-enter regardless)
            entry_side = closed_side  # Default: same direction
            try:
                market_data = self.client.get_market_data(symbol)
                if market_data and "1m" in market_data:
                    current_trend = self.detect_trend(symbol, market_data["1m"], strict=False)
                    expected_trend = "BULLISH" if closed_side == "LONG" else "BEARISH"

                    if current_trend != expected_trend:
                        # Trend changed - follow the NEW trend
                        entry_side = "LONG" if current_trend == "BULLISH" else "SHORT"
                        self.log(f"Re-enter {symbol}: Trend flipped to {current_trend}, entering {entry_side}", level="INFO")
            except Exception as e:
                self.log(f"Re-enter {symbol}: Trend check failed ({e}), using original direction {closed_side}", level="WARN")

            # Create signal for immediate re-entry
            signal_type = "BUY" if entry_side == "LONG" else "SELL"
            trend = "BULLISH" if entry_side == "LONG" else "BEARISH"

            signal = TradingSignal(
                signal=signal_type,
                signal_type=SignalType.MOMENTUM,
                confidence=0.8,
                reason="ALWAYS_HOLD_REENTRY",
                momentum_value=0.5,
                ema_trend=trend,
                rsi_value=50.0,
                adx_value=25.0
            )

            self.log(f">>> RE-ENTERING {symbol} {entry_side} <<< (ALWAYS HOLD strategy)", level="TRADE")
            self.enter_position(symbol, signal)

            # Clear from pending if it was there
            if symbol in self.pending_reentry:
                del self.pending_reentry[symbol]

        except Exception as e:
            self.log(f"Re-enter error for {symbol}: {e} - adding to pending queue", level="ERROR")
            # Add to pending so next scan cycle can try again
            self.pending_reentry[symbol] = closed_side

    def check_hybrid_dca_filter(self, symbol: str, position: LivePosition, dca_level: int, df: pd.DataFrame, market_data: dict = None) -> tuple:
        """
        Check DCA filter with STRICT rules + Higher Timeframe validation.

        Validates:
        1. 1m: RSI extreme, reversal candle, momentum weakening
        2. 1H: RSI confirmation (not extreme opposite direction)
        3. 4H: Trend alignment (EMA direction must match DCA side)

        Returns: (can_dca: bool, reason: str)
        """
        try:
            filter_config = DCA_CONFIG.get("hybrid_dca_filters", {})
            strict_levels = [1, 2, 3, 4]  # ALL LEVELS ARE STRICT

            reasons = []
            passed = True

            # ================================================================
            # 1M TIMEFRAME CHECKS (existing strict logic)
            # ================================================================
            # Calculate 1m RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi_1m = rsi.iloc[-1]

            # Calculate 1m momentum
            momentum = df["close"].pct_change(periods=5).iloc[-1] * 100
            prev_momentum = df["close"].pct_change(periods=5).iloc[-2] * 100 if len(df) > 6 else momentum
            momentum_weakening = abs(momentum) < abs(prev_momentum)

            # RSI thresholds
            rsi_oversold = 30  # RSI must be below 30 for LONG DCA
            rsi_overbought = 70  # RSI must be above 70 for SHORT DCA

            # Check 1m RSI extreme
            if position.side == "LONG" and current_rsi_1m > rsi_oversold:
                passed = False
                reasons.append(f"1m RSI not oversold ({current_rsi_1m:.0f}>{rsi_oversold})")
            elif position.side == "SHORT" and current_rsi_1m < rsi_overbought:
                passed = False
                reasons.append(f"1m RSI not overbought ({current_rsi_1m:.0f}<{rsi_overbought})")

            # Check reversal candle
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]

            if position.side == "LONG":
                is_reversal = (last_candle["close"] > last_candle["open"] and
                              prev_candle["close"] < prev_candle["open"])
            else:
                is_reversal = (last_candle["close"] < last_candle["open"] and
                              prev_candle["close"] > prev_candle["open"])

            if not is_reversal:
                passed = False
                reasons.append("No 1m reversal candle")

            # ================================================================
            # HIGHER TIMEFRAME VALIDATION (NEW)
            # ================================================================
            if market_data:
                # 1H RSI CONFIRMATION
                df_1h = market_data.get("1h")
                if df_1h is not None and len(df_1h) >= 14:
                    delta_1h = df_1h["close"].diff()
                    gain_1h = (delta_1h.where(delta_1h > 0, 0)).rolling(window=14).mean()
                    loss_1h = (-delta_1h.where(delta_1h < 0, 0)).rolling(window=14).mean()
                    rs_1h = gain_1h / loss_1h
                    rsi_1h = 100 - (100 / (1 + rs_1h))
                    current_rsi_1h = rsi_1h.iloc[-1]

                    if not pd.isna(current_rsi_1h):
                        # For LONG DCA: 1H RSI should NOT be overbought (< 75)
                        # For SHORT DCA: 1H RSI should NOT be oversold (> 25)
                        if position.side == "LONG" and current_rsi_1h > 75:
                            passed = False
                            reasons.append(f"1H RSI overbought ({current_rsi_1h:.0f}>75)")
                        elif position.side == "SHORT" and current_rsi_1h < 25:
                            passed = False
                            reasons.append(f"1H RSI oversold ({current_rsi_1h:.0f}<25)")

                # 4H TREND ALIGNMENT (EMA check)
                df_4h = market_data.get("4h")
                if df_4h is not None and len(df_4h) >= 26:
                    # Calculate EMAs on 4H
                    ema_fast = df_4h["close"].ewm(span=12, adjust=False).mean()
                    ema_slow = df_4h["close"].ewm(span=26, adjust=False).mean()

                    ema_fast_val = ema_fast.iloc[-1]
                    ema_slow_val = ema_slow.iloc[-1]
                    current_price = df_4h["close"].iloc[-1]

                    # Determine 4H trend
                    is_4h_bullish = ema_fast_val > ema_slow_val and current_price > ema_fast_val
                    is_4h_bearish = ema_fast_val < ema_slow_val and current_price < ema_fast_val

                    # For LONG DCA: 4H should NOT be strongly bearish
                    # For SHORT DCA: 4H should NOT be strongly bullish
                    if position.side == "LONG" and is_4h_bearish:
                        passed = False
                        reasons.append(f"4H trend bearish (EMA12<EMA26)")
                    elif position.side == "SHORT" and is_4h_bullish:
                        passed = False
                        reasons.append(f"4H trend bullish (EMA12>EMA26)")

            # ================================================================
            # RESULT
            # ================================================================
            if passed:
                htf_status = ""
                if market_data:
                    df_1h = market_data.get("1h")
                    df_4h = market_data.get("4h")
                    if df_1h is not None and df_4h is not None:
                        htf_status = " | HTF aligned"
                return True, f"L{dca_level} STRICT: All checks passed{htf_status}"
            else:
                return False, f"L{dca_level} STRICT BLOCKED: {', '.join(reasons)}"

        except Exception as e:
            self.log(f"Hybrid DCA filter error: {e}", level="WARN")
            return False, f"Filter error: {e}"

    def calculate_position_size(self, symbol: str, price: float, is_dca: bool = False, dca_level: int = 0) -> float:
        """
        Calculate position size based on dynamic allocation.
        Uses margin-based allocation with leverage.
        Ensures minimum quantity requirements are met.
        """
        # Get margin for this entry type
        if is_dca and dca_level > 0:
            margin = self.get_dca_margin(symbol, dca_level)
        else:
            margin = self.get_entry_margin(symbol)

        # Check if we have enough budget
        remaining = self.get_remaining_budget(symbol)
        if margin > remaining:
            margin = remaining

        if margin <= 0:
            return 0.0

        # Calculate position value with leverage
        position_value = margin * self.leverage

        # Convert to quantity
        quantity = position_value / price

        # Get symbol settings for min_qty and precision
        settings = SYMBOL_SETTINGS.get(symbol, {})
        min_qty = settings.get("min_qty", 0.001)
        qty_precision = settings.get("qty_precision", 3)
        min_notional = settings.get("min_notional", 100.0)  # Binance requires $100 min notional for BTC

        # Round to precision
        quantity = round(quantity, qty_precision)

        # Enforce minimum quantity - if below min, use min_qty
        if quantity < min_qty:
            quantity = min_qty

        # Enforce minimum notional value (Binance requires $100 for BTC)
        notional = quantity * price
        if notional < min_notional:
            # Calculate minimum quantity needed for min_notional
            min_qty_for_notional = min_notional / price
            # Round up to precision
            import math
            min_qty_for_notional = math.ceil(min_qty_for_notional * (10 ** qty_precision)) / (10 ** qty_precision)
            quantity = max(quantity, min_qty_for_notional)
            self.log(f"{symbol}: Adjusted qty to {quantity} for min notional ${min_notional}")

        # Final check if we can afford this quantity
        final_margin_needed = (quantity * price) / self.leverage
        if final_margin_needed > remaining:
            self.log(f"{symbol}: Cannot afford qty {quantity} (need ${final_margin_needed:.2f}, have ${remaining:.2f})", level="WARN")
            return 0.0

        return quantity

    def enter_position(self, symbol: str, signal: TradingSignal, is_hedge: bool = False):
        """
        Enter a new position with dynamic margin allocation.

        Args:
            symbol: Trading pair
            signal: Trading signal with direction
            is_hedge: If True, this is a hedge position (tracking is handled differently)
        """
        try:
            side = "BUY" if signal.signal == "BUY" else "SELL"
            position_side = "LONG" if side == "BUY" else "SHORT"

            # ================================================================
            # CANCEL EXISTING ORDERS for this side BEFORE entering new position
            # This prevents duplicate TP/SL orders
            # ================================================================
            if self.hedge_mode:
                # In hedge mode, only cancel orders for this specific side
                cancel_result = self.client.cancel_orders_for_side(symbol, position_side)
                if cancel_result.get("cancelled"):
                    self.log(f"Cancelled {len(cancel_result['cancelled'])} existing {position_side} orders before entry")
            else:
                # In normal mode, cancel all orders for this symbol
                self.client.cancel_all_orders(symbol)

            # Get margin for initial entry
            margin = self.get_entry_margin(symbol)
            remaining = self.get_remaining_budget(symbol)

            if margin > remaining or margin <= 0:
                self.log(f"Skip {symbol}: Insufficient budget (need ${margin:.2f}, have ${remaining:.2f})", level="WARN")
                return

            # Get current price
            price_data = self.client.get_current_price(symbol)
            current_price = price_data["price"]

            # Calculate position size based on margin + leverage
            quantity = self.calculate_position_size(symbol, current_price)

            if quantity <= 0:
                self.log(f"Skip {symbol}: Position size too small", level="WARN")
                return

            budget = self.symbol_budgets.get(symbol, 0)
            self.log(f"Entering {position_side} {symbol}: {quantity} @ ~${current_price:,.2f} | Margin: ${margin:.2f}/{budget:.2f}")

            # Place market order
            # In hedge mode, pass positionSide to enable dual positions
            hedge_position_side = position_side if (self.hedge_mode or is_hedge) else None
            order_result = self.client.place_market_order(symbol, side, quantity, position_side=hedge_position_side)

            if "orderId" not in order_result:
                self.log(f"Order failed: {order_result}", level="ERROR")
                return

            # Get actual fill price - market orders may return 0 initially
            fill_price = float(order_result.get("avgPrice", 0)) or current_price
            filled_qty = float(order_result.get("executedQty", 0))

            # If avgPrice is 0, retry multiple times to get actual entry price
            max_retries = 5
            retry_delay = 0.5

            for retry in range(max_retries):
                if fill_price > 0 and filled_qty > 0:
                    break

                import time
                time.sleep(retry_delay)

                # In hedge mode, filter by position side to get correct position
                hedge_side = position_side if (self.hedge_mode or is_hedge) else None
                position = self.client.get_position(symbol, position_side=hedge_side)
                if position and position.get("entry_price", 0) > 0:
                    fill_price = position["entry_price"]
                    filled_qty = position["quantity"]
                    self.log(f"Got fill price from position query (retry {retry + 1}): ${fill_price:,.2f}")
                    break

                if retry == max_retries - 1:
                    # Last resort: use current market price
                    fill_price = current_price
                    filled_qty = quantity
                    self.log(f"Could not get actual fill price after {max_retries} retries, using market price ${fill_price:,.2f}", level="WARN")

            # Track margin used
            self.symbol_margin_used[symbol] = margin

            # VALIDATE fill price before placing SL/TP
            sl_order_id = None
            tp_order_id = None

            if fill_price <= 0:
                self.log(f"CRITICAL: fill_price is {fill_price} - cannot place SL/TP orders!", level="ERROR")
            else:
                # Calculate SL/TP prices using ROI-BASED calculation for leveraged scalping
                # Formula: price_move = roi / leverage
                leverage = STRATEGY_CONFIG["leverage"]  # 20x

                if DCA_CONFIG["enabled"]:
                    # ROI-based: 15% ROI TP, 80% ROI SL (wide for DCA)
                    tp_roi = DCA_CONFIG["take_profit_roi"]    # 15% ROI
                    sl_roi = DCA_CONFIG["stop_loss_roi"]      # 80% ROI
                    tp_price_pct = tp_roi / leverage          # 15% / 20 = 0.75% price move
                    sl_price_pct = sl_roi / leverage          # 80% / 20 = 4% price move
                else:
                    # Price-based for non-DCA (legacy)
                    tp_price_pct = STRATEGY_CONFIG["take_profit_pct"]  # 2% price move
                    sl_price_pct = STRATEGY_CONFIG["stop_loss_pct"]    # 1% price move

                if position_side == "LONG":
                    stop_loss_price = fill_price * (1 - sl_price_pct)
                    take_profit_price = fill_price * (1 + tp_price_pct)
                    sl_side = "SELL"
                    tp_side = "SELL"
                else:
                    stop_loss_price = fill_price * (1 + sl_price_pct)
                    take_profit_price = fill_price * (1 - tp_price_pct)
                    sl_side = "BUY"
                    tp_side = "BUY"

                # ================================================================
                # LIQUIDATION PROTECTION: Ensure SL is ALWAYS before liquidation
                # ================================================================
                position_data = self.client.get_position(symbol, position_side=hedge_side)
                if position_data:
                    liq_price = float(position_data.get("liquidation_price", 0))
                    if liq_price > 0:
                        buffer_pct = DCA_CONFIG.get("liquidation_buffer_pct", 0.01)  # 1% buffer
                        if position_side == "LONG":
                            # For LONG: SL must be ABOVE liquidation price
                            min_sl_price = liq_price * (1 + buffer_pct)
                            if stop_loss_price < min_sl_price:
                                self.log(f"SL ${stop_loss_price:,.2f} too close to liquidation ${liq_price:,.2f}, adjusting to ${min_sl_price:,.2f}", level="WARN")
                                stop_loss_price = min_sl_price
                        else:
                            # For SHORT: SL must be BELOW liquidation price
                            max_sl_price = liq_price * (1 - buffer_pct)
                            if stop_loss_price > max_sl_price:
                                self.log(f"SL ${stop_loss_price:,.2f} too close to liquidation ${liq_price:,.2f}, adjusting to ${max_sl_price:,.2f}", level="WARN")
                                stop_loss_price = max_sl_price

                # Log with ROI context
                if DCA_CONFIG["enabled"]:
                    self.log(f"ROI-Based SL/TP for {symbol}: TP={tp_roi*100:.1f}% ROI (${take_profit_price:,.2f}), SL={sl_roi*100:.0f}% ROI (${stop_loss_price:,.2f})")

                # Place stop loss order (pass position_side for hedge mode)
                sl_order = self.client.place_stop_loss(symbol, sl_side, filled_qty, stop_loss_price,
                                                        position_side=hedge_position_side)
                if "orderId" in sl_order:
                    sl_order_id = sl_order.get("orderId")
                    self.log(f"SL order placed: ID={sl_order_id}")
                else:
                    self.log(f"SL order FAILED: {sl_order}", level="ERROR")

                # Place take profit order (pass position_side for hedge mode)
                tp_order = self.client.place_take_profit(symbol, tp_side, filled_qty, take_profit_price,
                                                          position_side=hedge_position_side)
                if "orderId" in tp_order:
                    tp_order_id = tp_order.get("orderId")
                    self.log(f"TP order placed: ID={tp_order_id}")
                else:
                    self.log(f"TP order FAILED: {tp_order}", level="ERROR")

            # Track position
            # For hedge mode, use symbol_SIDE as key (e.g., DOTUSDT_LONG, DOTUSDT_SHORT)
            # For normal mode, just use symbol
            if self.hedge_mode or is_hedge:
                position_key = f"{symbol}_{position_side}"
            else:
                position_key = symbol

            self.positions[position_key] = LivePosition(
                symbol=symbol,
                side=position_side,
                entry_price=fill_price if fill_price > 0 else current_price,
                quantity=filled_qty if filled_qty > 0 else quantity,
                entry_time=datetime.now(),
                stop_loss_order_id=sl_order_id,
                take_profit_order_id=tp_order_id,
                avg_entry_price=fill_price if fill_price > 0 else current_price,
                margin_used=margin
            )

            # SAVE STATE after new position entry
            self._save_position_state()

            # NOTE: daily_trades is counted on position CLOSE, not entry
            # This ensures trades/wins/losses are all counted consistently

            # Set signal cooldown AFTER successful trade entry
            # This prevents rapid-fire signals on the same symbol
            self.signal_generator.set_cooldown_time(symbol)

            # Build status message
            entry_price_display = fill_price if fill_price > 0 else current_price
            if sl_order_id:
                sl_status = f"SL: ${stop_loss_price:,.2f}"
            else:
                sl_status = "SL: FAILED"
            if tp_order_id:
                tp_status = f"TP: ${take_profit_price:,.2f}"
            else:
                tp_status = "TP: FAILED"

            hedge_label = " [HEDGE]" if is_hedge else ""
            self.log(
                f"ENTERED {position_side} {symbol}{hedge_label} @ ${entry_price_display:,.2f} | "
                f"Qty: {filled_qty:.6f} | Margin: ${margin:.2f}/{budget:.2f} | "
                f"Leverage: {self.leverage}x | {sl_status} | {tp_status}",
                level="TRADE"
            )

        except Exception as e:
            self.log(f"Error entering position {symbol}: {e}", level="ERROR")

    def sync_positions(self):
        """Sync local positions with Binance and release margin for closed positions"""
        try:
            binance_positions = self.client.get_positions()

            # Check for closed positions OR position side changes (LONG->SHORT or SHORT->LONG)
            for position_key in list(self.positions.keys()):
                local_pos = self.positions[position_key]

                # Extract actual symbol from key (handles both 'DOTUSDT' and 'DOTUSDT_LONG')
                actual_symbol = self.get_symbol_from_key(position_key)

                found = False
                side_changed = False
                binance_pos = None

                for pos in binance_positions:
                    if pos["symbol"] == actual_symbol and pos["side"] == local_pos.side:
                        # In hedge mode, match BOTH symbol AND side
                        found = True
                        binance_pos = pos
                        break
                    elif pos["symbol"] == actual_symbol and not self.hedge_mode:
                        # In non-hedge mode, just match symbol
                        found = True
                        binance_pos = pos
                        if pos["side"] != local_pos.side:
                            side_changed = True
                            self.log(f"Position {position_key} SIDE CHANGED: {local_pos.side} -> {pos['side']}")
                        break

                if not found or side_changed:
                    # Position was closed (SL/TP hit)
                    # local_pos is already set above from position_key
                    margin_released = local_pos.margin_used
                    closed_side = local_pos.side

                    # ============================================
                    # GET ACTUAL REALIZED PNL FROM BINANCE (not calculated)
                    # ============================================
                    realized_pnl = 0.0
                    exit_price = 0.0
                    exit_type = "SL/TP"
                    try:
                        # Get the ACTUAL realized PNL from Binance income history
                        income_records = self.client.get_income_history(actual_symbol, "REALIZED_PNL", limit=20)
                        if income_records:
                            # Get the most recent realized PNL (within last 60 seconds)
                            import time
                            now_ms = int(time.time() * 1000)
                            for record in income_records:
                                record_time = int(record.get("time", 0))
                                # Only consider PNL from last 60 seconds
                                if now_ms - record_time < 60000:
                                    realized_pnl = float(record.get("income", 0))
                                    self.log(f"  Binance reported PNL: ${realized_pnl:+.4f}")
                                    break

                        # Get exit price for logging
                        try:
                            recent_trades = self.client.get_recent_trades(actual_symbol, limit=5)
                            if recent_trades:
                                exit_price = float(recent_trades[0].get("price", 0))
                        except:
                            price_data = self.client.get_current_price(actual_symbol)
                            exit_price = price_data["price"]

                        # Determine win/loss based on BINANCE reported PNL
                        # SIMPLE RULE: Positive PNL = WIN (TP), Negative PNL = LOSS (SL)
                        if realized_pnl > 0:
                            exit_type = "TP"
                            self.daily_wins += 1
                        elif realized_pnl < 0:
                            exit_type = "SL"
                            self.daily_losses += 1
                        else:
                            # Breakeven or no PNL found - count as win
                            exit_type = "BE"
                            self.daily_wins += 1

                    except Exception as e:
                        # Fallback: use Binance unrealized PNL from position
                        self.log(f"  Could not get realized PNL: {e}", level="WARN")
                        try:
                            # Get the unrealized PNL that was showing before close
                            realized_pnl = local_pos.unrealized_pnl if hasattr(local_pos, 'unrealized_pnl') else 0
                            if realized_pnl >= 0:
                                exit_type = "TP"
                                self.daily_wins += 1
                            else:
                                exit_type = "SL"
                                self.daily_losses += 1
                        except:
                            # Last resort - count as win if balance growing
                            exit_type = "UNKNOWN"
                            self.daily_wins += 1  # Assume win if unsure

                    # Update daily stats
                    self.daily_pnl += realized_pnl
                    self.daily_trades += 1

                    # Update per-symbol stats
                    if hasattr(self, "symbol_stats") and actual_symbol in self.symbol_stats:
                        self.symbol_stats[actual_symbol]["pnl"] += realized_pnl
                        if exit_type == "TP":
                            self.symbol_stats[actual_symbol]["tp_count"] += 1
                            self.symbol_stats[actual_symbol]["wins"] += 1
                        elif exit_type == "SL":
                            self.symbol_stats[actual_symbol]["sl_count"] += 1
                            self.symbol_stats[actual_symbol]["losses"] += 1
                        elif realized_pnl >= 0:
                            self.symbol_stats[actual_symbol]["wins"] += 1
                        else:
                            self.symbol_stats[actual_symbol]["losses"] += 1

                    hedge_label = f" [{closed_side}]" if self.hedge_mode else ""
                    self.log(
                        f"Position {actual_symbol}{hedge_label} CLOSED ({exit_type}) | "
                        f"P&L: ${realized_pnl:+.2f} | Margin released: ${margin_released:.2f}",
                        level="TRADE"
                    )

                    # ============================================
                    # LOG TRADE TO FILE FOR PERMANENT RECORD
                    # ============================================
                    self.log_trade(
                        symbol=actual_symbol,
                        side=local_pos.side,
                        entry_price=local_pos.avg_entry_price,
                        exit_price=exit_price,
                        quantity=local_pos.quantity,
                        pnl=realized_pnl,
                        exit_type=exit_type,
                        dca_level=local_pos.dca_count
                    )

                    # ============================================
                    # CANCEL ALL REMAINING ORDERS FOR THIS SYMBOL
                    # When TP hits, SL remains open and vice versa
                    # Use cancel_all_orders to ensure all algo orders are cancelled too
                    # NOTE: In hedge mode, we should NOT cancel orders for the OTHER side!
                    # For now, we only cancel if NOT in hedge mode
                    # ============================================
                    if not self.hedge_mode:
                        try:
                            self.client.cancel_all_orders(actual_symbol)
                            self.log(f"  Cancelled all remaining orders for {actual_symbol}")
                        except Exception as e:
                            self.log(f"  Warning: Could not cancel orders for {actual_symbol}: {e}")

                    # closed_side already set above

                    # Release symbol margin (for hedge mode, track per-side)
                    if self.hedge_mode:
                        # Only reduce margin by this side's portion
                        current_margin = self.symbol_margin_used.get(actual_symbol, 0)
                        self.symbol_margin_used[actual_symbol] = max(0, current_margin - margin_released)
                    else:
                        self.symbol_margin_used[actual_symbol] = 0.0

                    # Check if boost mode should be deactivated
                    self._check_boost_deactivation(actual_symbol, closed_side, exit_type)

                    # Remove position from tracking
                    del self.positions[position_key]

                    # CLEAR STATE for closed position
                    self._save_position_state()

                    # ============================================
                    # HEDGE MODE or HYBRID HOLD: RE-ENTER LOGIC
                    # CHANGED: If SL hit -> STOP FOR THE DAY (no re-entry)
                    #          If TP hit -> Re-enter same direction
                    # ============================================
                    hedge_config = DCA_CONFIG.get("hedge_mode", {})
                    hybrid_config = DCA_CONFIG.get("hybrid_hold", {})

                    # CHECK IF SL HIT - STOP FOR THE DAY
                    if exit_type == "SL":
                        # SL hit - STOP trading this symbol for the day
                        self._stop_for_day(actual_symbol)
                        # Don't re-enter - skip to next position check
                        continue

                    # TP HIT - Re-enter in hedge mode
                    if self.hedge_mode and hedge_config.get("reenter_on_tp", True):
                        # HEDGE MODE: Re-enter the SAME SIDE that just closed (TP only)
                        exit_label = "WIN" if realized_pnl >= 0 else "LOSS"
                        self.log(f"HEDGE: {actual_symbol} [{closed_side}] closed ({exit_type}, {exit_label}), IMMEDIATE RE-ENTRY", level="TRADE")
                        self.pending_reentry[position_key] = closed_side
                        # Try immediate re-entry
                        self.reenter_after_tp(actual_symbol, closed_side)

                    elif (hybrid_config.get("enabled", False) and
                          hybrid_config.get("reenter_on_tp", True)):
                        # HYBRID MODE: Re-enter same direction (TP only)
                        exit_label = "WIN" if realized_pnl >= 0 else "LOSS"
                        self.log(f"HYBRID: {actual_symbol} closed ({exit_type}, {exit_label}), IMMEDIATE RE-ENTRY {closed_side}", level="TRADE")
                        self.pending_reentry[actual_symbol] = closed_side
                        self.reenter_after_tp(actual_symbol, closed_side)

            # Check for new positions (from DCA or manual) or positions with changed side
            for pos in binance_positions:
                binance_symbol = pos["symbol"]
                binance_side = pos["side"]

                # Determine the position key based on mode
                if self.hedge_mode:
                    pos_key = f"{binance_symbol}_{binance_side}"
                else:
                    pos_key = binance_symbol

                # Check if we need to sync this position
                need_sync = False
                if pos_key not in self.positions:
                    need_sync = True
                elif not self.hedge_mode and binance_side != self.positions[pos_key].side:
                    # Side changed in non-hedge mode
                    need_sync = True

                if need_sync:
                    # Get actual margin from Binance
                    margin_used = float(pos.get("isolated_wallet", 0)) or float(pos.get("isolatedWallet", 0))

                    # Auto-detect DCA level from margin used
                    dca_level = self.detect_dca_level_from_margin(binance_symbol, margin_used)

                    self.positions[pos_key] = LivePosition(
                        symbol=binance_symbol,
                        side=binance_side,
                        entry_price=pos["entry_price"],
                        quantity=pos["quantity"],
                        entry_time=datetime.now(),
                        avg_entry_price=pos["entry_price"],
                        margin_used=margin_used,
                        dca_count=dca_level
                    )

                    # Update margin tracking
                    if self.hedge_mode:
                        current_margin = self.symbol_margin_used.get(binance_symbol, 0)
                        self.symbol_margin_used[binance_symbol] = current_margin + margin_used
                    else:
                        self.symbol_margin_used[binance_symbol] = margin_used

                    hedge_label = f" [{binance_side}]" if self.hedge_mode else ""
                    self.log(f"Synced position: {binance_symbol}{hedge_label} | Margin: ${margin_used:.2f} | DCA Level: {dca_level}/4")

                    # Ensure SL/TP orders are placed for synced positions
                    self._ensure_sl_tp_orders(binance_symbol, pos, dca_level, position_key=pos_key)

        except Exception as e:
            self.log(f"Error syncing positions: {e}", level="ERROR")

    def check_dca(self, symbol: str, position: LivePosition, current_price: float):
        """Check and execute DCA with margin-based allocation and pair-specific volatility"""
        if not DCA_CONFIG["enabled"]:
            return

        # Skip invalid positions
        if position.avg_entry_price <= 0:
            return

        if position.dca_count >= len(DCA_CONFIG["levels"]):
            return

        # ENHANCED BOOST MODE: Skip DCA on boosted positions
        # When a position is boosted (1.5x), it should NOT DCA
        # It maintains its boosted size until TP (half-close) or losing side recovers
        if position.is_boosted:
            return  # Boosted positions don't DCA

        dca_level = position.dca_count + 1  # Next DCA level (1-4)

        # Check if we can afford this DCA level
        if not self.can_afford_dca(symbol, dca_level):
            return

        # Calculate current ROI (for leveraged positions)
        leverage = STRATEGY_CONFIG["leverage"]  # 20x
        if position.side == "LONG":
            price_drawdown = (position.avg_entry_price - current_price) / position.avg_entry_price
        else:
            price_drawdown = (current_price - position.avg_entry_price) / position.avg_entry_price

        # Convert to ROI: ROI = price_change * leverage
        current_roi_loss = price_drawdown * leverage  # e.g., 0.5% price loss * 20 = 10% ROI loss

        # Check if DCA level triggered (ROI-based) with SYMBOL-SPECIFIC DCA LEVELS
        symbol_settings = SYMBOL_SETTINGS.get(symbol, {})
        
        # Use symbol-specific DCA levels if defined, otherwise use default
        symbol_dca_levels = symbol_settings.get("dca_levels", None)
        if symbol_dca_levels and position.dca_count < len(symbol_dca_levels):
            level = symbol_dca_levels[position.dca_count]
            trigger_roi = abs(level["trigger_roi"])  # Already symbol-specific, no multiplier needed
            self.log(f"[DCA] Using symbol-specific DCA level for {symbol}: trigger={trigger_roi*100:.0f}%", level="DEBUG")
        else:
            # Fall back to default levels with volatility multiplier
            level = DCA_CONFIG["levels"][position.dca_count]
            base_trigger_roi = abs(level["trigger_roi"])  # e.g., 0.20 = 20% ROI loss
            volatility_mult = symbol_settings.get("dca_volatility_mult", 1.0)
            trigger_roi = base_trigger_roi * volatility_mult  # e.g., 0.20 * 1.8 = 0.36 (36% ROI for DOT)

        if current_roi_loss >= trigger_roi:
            # ================================================================
            # CHECK require_trend_filter FROM SYMBOL CONFIG
            # Symbol-specific DCA levels can require trend validation
            # ================================================================
            if symbol_dca_levels and position.dca_count < len(symbol_dca_levels):
                require_filter = symbol_dca_levels[position.dca_count].get("require_trend_filter", False)
                if require_filter:
                    # Must pass strict trend/reversal validation
                    if symbol in self.data_buffer and self.data_buffer[symbol] is not None:
                        df = self.data_buffer[symbol].get("1m") if isinstance(self.data_buffer[symbol], dict) else self.data_buffer[symbol]
                        if df is not None and len(df) > 14:
                            # Strict check: RSI extreme + reversal candle
                            delta = df["close"].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            current_rsi = rsi.iloc[-1]
                            
                            last_candle = df.iloc[-1]
                            prev_candle = df.iloc[-2]
                            
                            # Check for reversal pattern
                            if position.side == "LONG":
                                is_reversal = (last_candle["close"] > last_candle["open"] and 
                                              prev_candle["close"] < prev_candle["open"])
                                rsi_ok = current_rsi < 35  # Oversold
                            else:
                                is_reversal = (last_candle["close"] < last_candle["open"] and 
                                              prev_candle["close"] > prev_candle["open"])
                                rsi_ok = current_rsi > 65  # Overbought
                            
                            if not (is_reversal or rsi_ok):
                                self.log(f"[TREND FILTER] {symbol} {position.side} DCA {dca_level} BLOCKED - No reversal (RSI:{current_rsi:.0f})", level="DCA")
                                return

            # ================================================================
            # STRONG TREND MODE: Block DCA 2+ on loser side
            # In strong trends (ADX > 40), don't add to losing positions
            # ================================================================
            is_blocked, block_reason = self.is_dca_blocked_by_strong_trend(symbol, position.side, dca_level)
            if is_blocked:
                self.log(f"[STRONG TREND] {symbol} {position.side} DCA {dca_level} BLOCKED - {block_reason}", level="DCA")
                return

            # =================================================================
            # HYBRID DCA FILTER: STRICT + Higher Timeframe Validation
            # =================================================================
            if symbol in self.data_buffer and self.data_buffer[symbol] is not None:
                market_data = self.data_buffer[symbol] if isinstance(self.data_buffer[symbol], dict) else {"1m": self.data_buffer[symbol]}
                df = market_data.get("1m")
                if df is not None:
                    # Use STRICT filter with Higher Timeframe validation
                    can_dca, reason = self.check_hybrid_dca_filter(symbol, position, dca_level, df, market_data)

                    if not can_dca:
                        # Filter blocked DCA
                        self.log(
                            f"DCA {dca_level} {symbol}: BLOCKED - {reason}",
                            level="DCA"
                        )
                        self.log(
                            f"       >>> Waiting for conditions to improve before adding to position",
                            level="DCA"
                        )
                        return

                    self.log(f"DCA {dca_level} {symbol}: Signal validated - {reason}", level="DCA")

            try:
                # Get DCA margin for this level
                dca_margin = self.get_dca_margin(symbol, dca_level)

                # Calculate DCA quantity based on margin + leverage
                dca_qty = self.calculate_position_size(symbol, current_price, is_dca=True, dca_level=dca_level)

                if dca_qty <= 0:
                    self.log(f"DCA {dca_level} skipped for {symbol}: Quantity too small", level="WARN")
                    return

                # Place DCA order (in hedge mode, include positionSide)
                side = "BUY" if position.side == "LONG" else "SELL"
                position_side = position.side if self.hedge_mode else None
                order_result = self.client.place_market_order(symbol, side, dca_qty, position_side=position_side)

                if "orderId" in order_result:
                    fill_price = float(order_result.get("avgPrice", current_price))
                    filled_qty = float(order_result.get("executedQty", dca_qty))

                    # Update margin tracking
                    self.symbol_margin_used[symbol] += dca_margin
                    position.margin_used += dca_margin

                    # ================================================================
                    # CRITICAL: Get ACTUAL break-even price from Binance after DCA
                    # Don't calculate locally - use Binance's actual entry price!
                    # ================================================================
                    time.sleep(0.5)  # Wait for Binance to update position

                    # Retry to get accurate position data from Binance
                    max_retries = 3
                    actual_entry_price = 0.0
                    actual_quantity = 0.0

                    for retry in range(max_retries):
                        # In hedge mode, filter by position side to get correct position
                        position_data = self.client.get_position(symbol, position_side=position_side)
                        if position_data:
                            actual_entry_price = float(position_data.get("entry_price", 0))
                            actual_quantity = float(position_data.get("quantity", 0))
                            if actual_entry_price > 0 and actual_quantity > 0:
                                break
                        time.sleep(0.3)

                    # Use Binance's actual values if available, otherwise fallback to calculation
                    if actual_entry_price > 0 and actual_quantity > 0:
                        position.avg_entry_price = actual_entry_price
                        position.quantity = actual_quantity
                        self.log(f"DCA {dca_level}: Using Binance break-even price: ${actual_entry_price:,.4f}")
                    else:
                        # Fallback: calculate locally (less accurate)
                        dca_position_value = dca_margin * self.leverage
                        old_cost = position.avg_entry_price * position.quantity
                        position.quantity += filled_qty
                        position.avg_entry_price = (old_cost + dca_position_value) / position.quantity
                        self.log(f"DCA {dca_level}: Using calculated avg price: ${position.avg_entry_price:,.4f} (Binance not available)", level="WARN")

                    position.dca_count = dca_level

                    # ================================================================
                    # ENHANCED BOOST MODE: Check if this DCA level should trigger boost
                    # When one side hits boost trigger DCA level, boost the OTHER side
                    # ================================================================
                    self._check_boost_activation(symbol, position, dca_level)

                    # SAVE POSITION STATE after DCA (for restart recovery)
                    self._save_position_state()

                    # Cancel old SL/TP and place new ones
                    if position.stop_loss_order_id:
                        try:
                            cancel_result = self.client.cancel_order(symbol, position.stop_loss_order_id)
                            if cancel_result and "orderId" in cancel_result:
                                self.log(f"DCA {dca_level}: Cancelled old SL order {position.stop_loss_order_id}")
                            else:
                                self.log(f"DCA {dca_level}: Old SL order already gone (filled/expired)")
                        except Exception as e:
                            self.log(f"DCA {dca_level}: Old SL already gone: {e}")
                        position.stop_loss_order_id = None

                    if position.take_profit_order_id:
                        try:
                            cancel_result = self.client.cancel_order(symbol, position.take_profit_order_id)
                            if cancel_result and "orderId" in cancel_result:
                                self.log(f"DCA {dca_level}: Cancelled old TP order {position.take_profit_order_id}")
                            else:
                                self.log(f"DCA {dca_level}: Old TP order already gone (filled/expired)")
                        except Exception as e:
                            self.log(f"DCA {dca_level}: Old TP already gone: {e}")
                        position.take_profit_order_id = None

                    # ================================================================
                    # ROI-BASED SL/TP after DCA with LIQUIDATION PROTECTION
                    # REDUCED TP: Exit faster to reduce exposure on DCA positions
                    # ================================================================
                    leverage = STRATEGY_CONFIG["leverage"]

                    # Use tighter SL after DCA (cost basis improved)
                    sl_roi_after_dca = DCA_CONFIG["sl_after_dca_roi"]  # 20% ROI

                    # Get REDUCED TP from DCA level config (exit faster to reduce exposure)
                    dca_level_config = DCA_CONFIG["levels"][dca_level - 1]  # dca_level is 1-indexed
                    tp_roi = dca_level_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])

                    sl_price_pct = sl_roi_after_dca / leverage  # 20% / 20 = 1% price move
                    tp_price_pct = tp_roi / leverage            # e.g., 10% / 20 = 0.5% price move

                    if position.side == "LONG":
                        new_sl = position.avg_entry_price * (1 - sl_price_pct)
                        new_tp = position.avg_entry_price * (1 + tp_price_pct)
                        sl_side = "SELL"
                    else:
                        new_sl = position.avg_entry_price * (1 + sl_price_pct)
                        new_tp = position.avg_entry_price * (1 - tp_price_pct)
                        sl_side = "BUY"

                    # ================================================================
                    # GET NEW LIQUIDATION PRICE after DCA (position size changed!)
                    # ================================================================
                    if position_data:
                        new_liq_price = float(position_data.get("liquidation_price", 0))
                        if new_liq_price > 0:
                            buffer_pct = DCA_CONFIG.get("liquidation_buffer_pct", 0.01)
                            if position.side == "LONG":
                                min_sl_price = new_liq_price * (1 + buffer_pct)
                                if new_sl < min_sl_price:
                                    self.log(f"DCA {dca_level}: SL ${new_sl:,.2f} adjusted to ${min_sl_price:,.2f} (liq: ${new_liq_price:,.2f})", level="WARN")
                                    new_sl = min_sl_price
                            else:
                                max_sl_price = new_liq_price * (1 - buffer_pct)
                                if new_sl > max_sl_price:
                                    self.log(f"DCA {dca_level}: SL ${new_sl:,.2f} adjusted to ${max_sl_price:,.2f} (liq: ${new_liq_price:,.2f})", level="WARN")
                                    new_sl = max_sl_price

                            self.log(f"DCA {dca_level}: New liquidation price: ${new_liq_price:,.2f}")

                    # Place new SL/TP orders with retry
                    sl_order = None
                    tp_order = None

                    for retry in range(3):
                        if sl_order is None or "orderId" not in sl_order:
                            sl_order = self.client.place_stop_loss(symbol, sl_side, position.quantity, new_sl, position_side=position_side)
                            if "orderId" in sl_order:
                                position.stop_loss_order_id = sl_order.get("orderId")
                                self.log(f"DCA {dca_level}: New SL placed @ ${new_sl:,.4f} (ID: {sl_order.get('orderId')})")
                            else:
                                self.log(f"DCA {dca_level}: SL order failed (attempt {retry+1}): {sl_order}", level="WARN")
                                time.sleep(0.3)

                        if tp_order is None or "orderId" not in tp_order:
                            tp_order = self.client.place_take_profit(symbol, sl_side, position.quantity, new_tp, position_side=position_side)
                            if "orderId" in tp_order:
                                position.take_profit_order_id = tp_order.get("orderId")
                                self.log(f"DCA {dca_level}: New TP placed @ ${new_tp:,.4f} (ID: {tp_order.get('orderId')})")
                            else:
                                self.log(f"DCA {dca_level}: TP order failed (attempt {retry+1}): {tp_order}", level="WARN")
                                time.sleep(0.3)

                        if sl_order and "orderId" in sl_order and tp_order and "orderId" in tp_order:
                            break

                    self.log(f"DCA {dca_level}: SL @ ${new_sl:,.4f} ({sl_roi_after_dca*100:.0f}% ROI), TP @ ${new_tp:,.4f} ({tp_roi*100:.0f}% ROI)")

                    budget = self.symbol_budgets.get(symbol, 0)
                    # Use actual Binance entry price for display (fill_price may be 0 from API response)
                    display_price = position.avg_entry_price if position.avg_entry_price > 0 else fill_price
                    self.log(
                        f"DCA {dca_level} {symbol} @ ${display_price:,.4f} | "
                        f"Margin: +${dca_margin:.2f} (${position.margin_used:.2f}/{budget:.2f}) | "
                        f"Total Qty: {position.quantity:.6f} | Avg: ${position.avg_entry_price:,.4f}",
                        level="DCA"
                    )

            except Exception as e:
                self.log(f"DCA error for {symbol}: {e}", level="ERROR")

    def manage_positions(self):
        """Manage all open positions - check TP/SL, Trailing TP, Trend Change, and DCA"""
        # Sync with Binance
        self.sync_positions()

        # ================================================================
        # STRONG TREND MODE: Check and update for each symbol
        # When ADX > 40, activate Strong Trend Mode and block DCA 2+ on loser side
        # ================================================================
        processed_symbols = set()
        for pos_key in list(self.positions.keys()):
            sym = self.get_symbol_from_key(pos_key)
            if sym in processed_symbols:
                continue
            processed_symbols.add(sym)

            if self.boost_mode_active.get(sym, False):
                continue

            is_strong, direction = self.check_strong_trend_mode(sym)

            if is_strong and not self.strong_trend_mode.get(sym, False):
                self.activate_strong_trend_mode(sym, direction)
            elif not is_strong and self.strong_trend_mode.get(sym, False):
                self.deactivate_strong_trend_mode(sym, f"ADX dropped below {self.adx_threshold}")


        # Check each position
        roi_boost_checked = set()  # Track symbols we've already checked for ROI boost
        for position_key, position in list(self.positions.items()):
            try:
                # Extract actual symbol from key (handles both 'DOTUSDT' and 'DOTUSDT_LONG')
                symbol = self.get_symbol_from_key(position_key)
                price_data = self.client.get_current_price(symbol)
                current_price = price_data["price"]

                # ================================================================
                # ROI-BASED BOOST: Check if any side hit -20% ROI (once per symbol)
                # This triggers EARLIER than DCA 3 for faster hedge recovery
                # ================================================================
                if self.hedge_mode and symbol not in roi_boost_checked:
                    roi_boost_checked.add(symbol)
                    self._check_roi_boost_activation(symbol, current_price)

                # ================================================================
                # HEDGE MANAGEMENT: Check if position is waiting for breakeven
                # ================================================================
                hybrid_config = DCA_CONFIG.get("hybrid_hold", {})
                if position.is_hedged and position.waiting_for_breakeven:
                    if self.manage_hedged_position(symbol, position, current_price):
                        continue  # Position was closed due to timeout, skip to next

                # ================================================================
                # HYBRID HOLD: Check for TREND CHANGE first (NOT used in Hedge Mode)
                # If trend reversed, either close+flip OR hedge (for DCA positions)
                # NOTE: In HEDGE MODE we always hold BOTH sides, so trend flip is irrelevant
                # ================================================================
                if not self.hedge_mode and hybrid_config.get("enabled", False) and hybrid_config.get("flip_on_trend_change", True):
                    # Skip trend change check if already hedged
                    if not position.is_hedged and self.check_trend_change(symbol, position):
                        # Trend has changed
                        new_trend = self.current_trend.get(symbol, "NEUTRAL")
                        if new_trend != "NEUTRAL":
                            # Check if we should use HEDGE strategy (for DCA positions)
                            hedge_enabled = hybrid_config.get("hedge_on_dca_flip", True)
                            min_dca_for_hedge = hybrid_config.get("hedge_min_dca_level", 1)

                            if hedge_enabled and position.dca_count >= min_dca_for_hedge:
                                # Use HEDGE strategy: don't close, open opposite position
                                self.log(
                                    f"TREND FLIP {symbol}: DCA L{position.dca_count} detected | "
                                    f"Using HEDGE strategy instead of close",
                                    level="TRADE"
                                )
                                self.hedge_on_trend_flip(symbol, position, new_trend)
                            else:
                                # Normal close and flip (no DCA or hedge disabled)
                                self.close_and_flip(symbol, position, new_trend)
                            continue  # Position handled, skip to next

                # ================================================================
                # STALE EXIT: Close positions held too long at min profit
                # ================================================================
                if position.avg_entry_price > 0:
                    should_close = self._check_stale_exit(symbol, position, current_price)
                    if should_close:
                        continue  # Position was closed, skip to next

                # ================================================================
                # TRAILING TP: Check if we should trail and close on pullback
                # This runs BEFORE auto-close check to capture profits at peak
                # ================================================================
                if position.avg_entry_price > 0:
                    should_close = self._check_trailing_tp(symbol, position, current_price)
                    if should_close:
                        continue  # Position was closed, skip to next

                # ================================================================
                # AUTO-CLOSE: Check if price is past TP target
                # If TP order failed or is missing, close position with market order
                # ================================================================
                if position.avg_entry_price > 0:
                    should_close = self._check_auto_close_tp(symbol, position, current_price)
                    if should_close:
                        continue  # Position was closed, skip to next

                # Refresh market data for SMART DCA validation
                if symbol not in self.data_buffer or self.data_buffer[symbol] is None:
                    try:
                        market_data = self.client.get_market_data(symbol)
                        if market_data and "1m" in market_data:
                            self.data_buffer[symbol] = market_data
                    except:
                        pass  # Continue with DCA check even without fresh data

                self.check_dca(symbol, position, current_price)

            except Exception as e:
                self.log(f"Error managing {symbol}: {e}", level="ERROR")

    def _check_stale_exit(self, symbol: str, position: LivePosition, current_price: float) -> bool:
        """
        Close positions held too long once they reach min profit ROI.
        Returns True if position was closed, False otherwise.
        """
        stale_config = DCA_CONFIG.get("stale_exit", {})
        if not stale_config.get("enabled", False):
            return False

        max_hold_hours = stale_config.get("max_hold_hours", 4)
        min_exit_roi = stale_config.get("min_exit_roi", 0.10)

        # Check how long position has been held
        hold_time = datetime.now() - position.entry_time
        hold_hours = hold_time.total_seconds() / 3600

        if hold_hours < max_hold_hours:
            return False  # Not stale yet

        # Position is stale - check if profitable enough to exit
        margin = position.margin_used
        if margin <= 0:
            return False

        # Calculate current ROI
        if position.side == "LONG":
            pnl = (current_price - position.avg_entry_price) * position.quantity
        else:
            pnl = (position.avg_entry_price - current_price) * position.quantity

        current_roi = pnl / margin

        if current_roi < min_exit_roi:
            return False  # Not profitable enough yet

        # Position is stale AND profitable - close it!
        self.log(
            f"STALE EXIT {symbol}: Held {hold_hours:.1f}h (>{max_hold_hours}h) | "
            f"ROI: {current_roi*100:.1f}% >= {min_exit_roi*100:.0f}%",
            level="TRADE"
        )

        # Close position with market order (in hedge mode, include positionSide)
        close_side = "SELL" if position.side == "LONG" else "BUY"
        hedge_position_side = position.side if self.hedge_mode else None
        order_result = self.client.place_market_order(symbol, close_side, position.quantity, position_side=hedge_position_side)

        if "orderId" in order_result:
            fill_price = float(order_result.get("avgPrice", current_price))

            # Calculate final PNL
            if position.side == "LONG":
                final_pnl = (fill_price - position.avg_entry_price) * position.quantity
            else:
                final_pnl = (position.avg_entry_price - fill_price) * position.quantity

            final_roi = final_pnl / margin * 100

            self.log(
                f"STALE EXIT CLOSED {symbol} @ ${fill_price:,.4f} | "
                f"P&L: ${final_pnl:+.2f} ({final_roi:+.1f}% ROI) | "
                f"Held: {hold_hours:.1f} hours",
                level="TRADE"
            )

            # Cancel remaining orders
            try:
                self.client.cancel_all_orders(symbol)
            except:
                pass

            # Log trade
            self.log_trade(
                symbol=symbol,
                side=position.side,
                entry_price=position.avg_entry_price,
                exit_price=fill_price,
                quantity=position.quantity,
                pnl=final_pnl,
                exit_type="STALE_EXIT",
                dca_level=position.dca_count
            )

            # Update stats
            self.daily_pnl += final_pnl
            self.daily_trades += 1
            if final_pnl > 0:
                self.daily_wins += 1
            else:
                self.daily_losses += 1

            # Check if boost mode should be deactivated
            self._check_boost_deactivation(symbol, position.side, "STALE_EXIT")

            # Release margin and remove position
            self.symbol_margin_used[symbol] = 0.0
            del self.positions[symbol]

            return True
        else:
            self.log(f"STALE EXIT FAILED {symbol}: {order_result}", level="ERROR")
            return False

    def _check_trailing_tp(self, symbol: str, position: LivePosition, current_price: float) -> bool:
        """
        Check trailing take profit and close position if price pulls back from peak.
        Returns True if position was closed, False otherwise.

        Logic:
        1. Calculate current ROI
        2. Update peak ROI if current is higher
        3. If peak ROI >= activation threshold, activate trailing
        4. If trailing active and ROI drops by trail_distance from peak, close position
        """
        trailing_config = DCA_CONFIG.get("trailing_tp", {})
        if not trailing_config.get("enabled", False):
            return False

        leverage = STRATEGY_CONFIG["leverage"]
        margin = position.margin_used

        if margin <= 0:
            return False

        # Calculate current ROI
        if position.side == "LONG":
            pnl = (current_price - position.avg_entry_price) * position.quantity
        else:
            pnl = (position.avg_entry_price - current_price) * position.quantity

        current_roi = pnl / margin  # ROI as decimal (0.20 = 20%)

        # Update peak ROI (only track positive ROI)
        if current_roi > position.peak_roi:
            position.peak_roi = current_roi

        activation_roi = trailing_config.get("activation_roi", 0.20)
        trail_distance = trailing_config.get("trail_distance_roi", 0.15)
        min_profit = trailing_config.get("min_profit_roi", 0.10)

        # Check if we should activate trailing
        if not position.trailing_active and position.peak_roi >= activation_roi:
            position.trailing_active = True
            self.log(
                f"TRAILING TP ACTIVATED {symbol}: Peak ROI {position.peak_roi*100:.1f}% >= {activation_roi*100:.0f}%",
                level="TRAIL"
            )

            # Cancel Binance TP order - we'll manage exit manually
            if position.take_profit_order_id and not position.tp_cancelled:
                try:
                    self.client.cancel_order(symbol, position.take_profit_order_id)
                    position.tp_cancelled = True
                    self.log(f"  Cancelled Binance TP order {position.take_profit_order_id} - using trailing TP")
                except Exception as e:
                    self.log(f"  Could not cancel TP order: {e}", level="WARN")

        # Check if trailing triggered (pullback from peak)
        if position.trailing_active:
            roi_from_peak = position.peak_roi - current_roi

            # Close if pullback exceeds trail distance AND still profitable
            if roi_from_peak >= trail_distance and current_roi >= min_profit:
                self.log(
                    f"TRAILING TP TRIGGERED {symbol}: "
                    f"Peak {position.peak_roi*100:.1f}% -> Current {current_roi*100:.1f}% "
                    f"(Pullback: {roi_from_peak*100:.1f}% >= {trail_distance*100:.0f}%)",
                    level="TRADE"
                )

                # Close position with market order (in hedge mode, include positionSide)
                close_side = "SELL" if position.side == "LONG" else "BUY"
                hedge_position_side = position.side if self.hedge_mode else None
                order_result = self.client.place_market_order(symbol, close_side, position.quantity, position_side=hedge_position_side)

                if "orderId" in order_result:
                    fill_price = float(order_result.get("avgPrice", current_price))

                    # Calculate final PNL
                    if position.side == "LONG":
                        final_pnl = (fill_price - position.avg_entry_price) * position.quantity
                    else:
                        final_pnl = (position.avg_entry_price - fill_price) * position.quantity

                    final_roi = final_pnl / margin * 100

                    self.log(
                        f"TRAILING TP CLOSED {symbol} @ ${fill_price:,.4f} | "
                        f"P&L: ${final_pnl:+.2f} ({final_roi:+.1f}% ROI) | "
                        f"Peak was {position.peak_roi*100:.1f}%",
                        level="TRADE"
                    )

                    # Cancel remaining SL orders
                    try:
                        self.client.cancel_all_orders(symbol)
                        self.log(f"  Cancelled remaining orders for {symbol}")
                    except:
                        pass

                    # Log trade
                    self.log_trade(
                        symbol=symbol,
                        side=position.side,
                        entry_price=position.avg_entry_price,
                        exit_price=fill_price,
                        quantity=position.quantity,
                        pnl=final_pnl,
                        exit_type="TRAILING_TP",
                        dca_level=position.dca_count
                    )

                    # Update stats
                    self.daily_pnl += final_pnl
                    self.daily_trades += 1
                    if final_pnl > 0:
                        self.daily_wins += 1
                    else:
                        self.daily_losses += 1

                    # Check if boost mode should be deactivated
                    self._check_boost_deactivation(symbol, position.side, "TRAILING_TP")

                    # Release margin and remove position
                    self.symbol_margin_used[symbol] = 0.0
                    del self.positions[symbol]

                    return True
                else:
                    self.log(f"TRAILING TP CLOSE FAILED {symbol}: {order_result}", level="ERROR")
                    return False

        return False

    def _check_auto_close_tp(self, symbol: str, position: LivePosition, current_price: float) -> bool:
        """
        Check if position should be auto-closed because price is past TP target.
        For BOOSTED positions: executes half-close cycle instead of full close.
        Returns True if position was closed/handled, False otherwise.
        """
        leverage = STRATEGY_CONFIG["leverage"]

        # Calculate expected TP based on DCA level
        dca_level = position.dca_count
        if dca_level > 0 and dca_level <= len(DCA_CONFIG["levels"]):
            # Use reduced TP from DCA level config
            dca_level_config = DCA_CONFIG["levels"][dca_level - 1]
            tp_roi = dca_level_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])
        else:
            tp_roi = DCA_CONFIG["take_profit_roi"]

        tp_price_pct = tp_roi / leverage

        # Calculate expected TP price
        if position.side == "LONG":
            expected_tp = position.avg_entry_price * (1 + tp_price_pct)
            price_past_tp = current_price >= expected_tp
        else:
            expected_tp = position.avg_entry_price * (1 - tp_price_pct)
            price_past_tp = current_price <= expected_tp

        if not price_past_tp:
            return False

        # Price is past TP - check if we have a valid TP order
        try:
            orders = self.client.get_open_orders(symbol)
            has_valid_tp = False

            for order in orders:
                order_type = order.get("type", "")
                order_qty = float(order.get("origQty", 0))
                order_position_side = order.get("positionSide", "BOTH")

                # In hedge mode, match position side
                if self.hedge_mode and order_position_side != position.side:
                    continue

                if order_type == "TAKE_PROFIT_MARKET":
                    # Check if quantity matches (within 1% tolerance)
                    if abs(order_qty - position.quantity) <= position.quantity * 0.01:
                        has_valid_tp = True
                        break

            if has_valid_tp:
                # TP order exists with correct qty - Binance should execute it
                return False

            # *** BOOSTED POSITION: Execute half-close cycle ***
            if position.is_boosted:
                self.log(f"[BOOST TP] {symbol} {position.side}: Price ${current_price:,.4f} hit TP ${expected_tp:,.4f}", level="TRADE")
                return self._handle_boosted_tp(symbol, position, current_price)

            # NO valid TP order and price is past target - AUTO CLOSE!
            self.log(f"AUTO-CLOSE {symbol}: Price ${current_price:,.4f} past TP ${expected_tp:,.4f} (no valid TP order)", level="TRADE")

            # Calculate actual P&L
            if position.side == "LONG":
                pnl = (current_price - position.avg_entry_price) * position.quantity
            else:
                pnl = (position.avg_entry_price - current_price) * position.quantity

            # Close position with market order (in hedge mode, include positionSide)
            close_side = "SELL" if position.side == "LONG" else "BUY"
            hedge_position_side = position.side if self.hedge_mode else None
            order_result = self.client.place_market_order(symbol, close_side, position.quantity, position_side=hedge_position_side)

            if "orderId" in order_result:
                # Get actual fill price from order result
                fill_price = float(order_result.get("avgPrice", current_price))

                self.log(f"AUTO-CLOSE {symbol}: CLOSED @ ${fill_price:,.4f} | P&L: ${pnl:+,.2f}", level="TRADE")

                # Cancel any remaining SL orders
                try:
                    self.client.cancel_all_orders(symbol)
                    self.log(f"AUTO-CLOSE {symbol}: Cancelled remaining orders")
                except:
                    pass

                # LOG TRADE TO FILE
                self.log_trade(
                    symbol=symbol,
                    side=position.side,
                    entry_price=position.avg_entry_price,
                    exit_price=fill_price,
                    quantity=position.quantity,
                    pnl=pnl,
                    exit_type="AUTO_CLOSE",
                    dca_level=position.dca_count
                )

                # Update stats
                self.daily_pnl += pnl
                self.daily_trades += 1
                if pnl > 0:
                    self.daily_wins += 1
                else:
                    self.daily_losses += 1

                # Check if boost mode should be deactivated
                self._check_boost_deactivation(symbol, position.side, "AUTO_TP")

                # Release margin and remove position
                self.symbol_margin_used[symbol] = 0.0
                del self.positions[symbol]

                return True
            else:
                self.log(f"AUTO-CLOSE {symbol}: FAILED to close - {order_result}", level="ERROR")
                return False

        except Exception as e:
            self.log(f"AUTO-CLOSE {symbol}: Error checking TP - {e}", level="ERROR")
            return False

    def check_entry_signals(self):
        """Check for entry signals with verbose logging like OANDA/Alpaca system"""
        # Get balance info
        balance = self.client.get_balance()
        open_count = len(self.positions)
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Print scan header
        print("\n" + "="*70)
        print(f"[{timestamp}] SCANNING {len(self.symbols)} SYMBOLS | Balance: ${balance:,.2f} | Positions: {open_count}")
        print("="*70)

        total_unrealized_pnl = 0.0

        for symbol in self.symbols:
            try:
                # Get market data first (need it for logging even if can't trade)
                market_data = self.client.get_market_data(symbol)

                if not market_data or "1m" not in market_data:
                    print(f"  {symbol}: NO DATA")
                    continue

                # Store market data for SMART DCA validation
                self.data_buffer[symbol] = market_data

                # Get current price and calculate indicators
                df = market_data["1m"]
                current_price = df["close"].iloc[-1] if len(df) > 0 else 0

                # Calculate momentum
                momentum_threshold = MOMENTUM_CONFIG.get("momentum_threshold", 0.08)
                if len(df) >= 4:
                    momentum = ((df["close"].iloc[-1] - df["close"].iloc[-4]) / df["close"].iloc[-4]) * 100
                else:
                    momentum = 0.0

                # Calculate RSI and ADX for display
                rsi_val = 50.0
                adx_val = 20.0
                if hasattr(self.signal_generator, 'calculate_rsi') and len(df) >= 14:
                    rsi_series = self.signal_generator.calculate_rsi(df["close"], 14)
                    if len(rsi_series) > 0:
                        rsi_val = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50.0
                if hasattr(self.signal_generator, 'calculate_adx') and len(df) >= 14:
                    adx_series = self.signal_generator.calculate_adx(df["high"], df["low"], df["close"], 14)
                    if len(adx_series) > 0:
                        adx_val = adx_series.iloc[-1] if not pd.isna(adx_series.iloc[-1]) else 20.0

                mom_sign = "+" if momentum >= 0 else ""

                # Get positions for this symbol (handles both normal and hedge mode)
                symbol_positions = self.get_all_positions_for_symbol(symbol)

                if symbol_positions:
                    # In hedge mode, we might have BOTH LONG and SHORT positions
                    price_data = self.client.get_current_price(symbol)
                    current_price = price_data["price"]

                    for pos in symbol_positions:
                        # Skip invalid positions (entry price = 0)
                        if pos.avg_entry_price <= 0:
                            print(f"  {symbol}: {pos.side} (INVALID - syncing...)")
                            continue

                        # Calculate P&L
                        if pos.side == "LONG":
                            pnl_pct = ((current_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
                            pnl_dollar = (current_price - pos.avg_entry_price) * pos.quantity
                        else:
                            pnl_pct = ((pos.avg_entry_price - current_price) / pos.avg_entry_price) * 100
                            pnl_dollar = (pos.avg_entry_price - current_price) * pos.quantity

                        total_unrealized_pnl += pnl_dollar
                        pnl_sign = "+" if pnl_pct >= 0 else ""

                        # Calculate next DCA trigger (ROI-based)
                        dca_level = pos.dca_count
                        next_dca_roi = 0.0
                        if dca_level < len(DCA_CONFIG["levels"]):
                            next_dca_roi = abs(DCA_CONFIG["levels"][dca_level]["trigger_roi"]) * 100

                        # Calculate current ROI (pnl_pct is already ROI for leveraged positions)
                        leverage = STRATEGY_CONFIG["leverage"]
                        current_roi = pnl_pct  # This is already the ROI %

                        # Print position line with ROI (show side for hedge mode)
                        if self.hedge_mode:
                            print(f"  {symbol} [{pos.side}]: @ ${pos.avg_entry_price:,.2f} | P&L: ${pnl_dollar:+.2f} ({pnl_sign}{current_roi:.2f}% ROI) | DCA: {dca_level}/4")
                        else:
                            print(f"  {symbol}: {pos.side} @ ${pos.avg_entry_price:,.2f} | P&L: ${pnl_dollar:+.2f} ({pnl_sign}{current_roi:.2f}% ROI) | DCA: {dca_level}/4")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | Mom: {mom_sign}{momentum:.2f}% | Next DCA: -{next_dca_roi:.0f}% ROI")

                    # In hedge mode, don't skip to next symbol - we may still need to check for missing sides
                    if not self.hedge_mode:
                        continue

                    # Check if we have BOTH sides in hedge mode
                    has_long = f"{symbol}_LONG" in self.positions
                    has_short = f"{symbol}_SHORT" in self.positions
                    if has_long and has_short:
                        continue  # Both sides present, skip to next symbol

                # ============================================
                # CHECK FOR PENDING RE-ENTRY (ALWAYS HOLD / HEDGE STRATEGY)
                # If TP just hit, re-enter IMMEDIATELY - no momentum check needed!
                # ============================================
                # Check for pending re-entries (handles both 'SYMBOL' and 'SYMBOL_SIDE' keys)
                pending_keys = []
                if self.hedge_mode:
                    # Check for hedge mode keys
                    if f"{symbol}_LONG" in self.pending_reentry:
                        pending_keys.append(f"{symbol}_LONG")
                    if f"{symbol}_SHORT" in self.pending_reentry:
                        pending_keys.append(f"{symbol}_SHORT")
                elif symbol in self.pending_reentry:
                    pending_keys.append(symbol)

                for pending_key in pending_keys:
                    pending_side = self.pending_reentry[pending_key]
                    mode_label = "HEDGE" if self.hedge_mode else "ALWAYS HOLD"
                    print(f"  {symbol}: >>> RE-ENTERING {pending_side} <<< (TP hit, {mode_label})")

                    # Create signal for re-entry
                    signal_type = "BUY" if pending_side == "LONG" else "SELL"
                    trend = "BULLISH" if pending_side == "LONG" else "BEARISH"

                    signal = TradingSignal(
                        signal=signal_type,
                        signal_type=SignalType.MOMENTUM,
                        confidence=0.8,
                        reason=f"{mode_label}_REENTRY",
                        momentum_value=0.5,
                        ema_trend=trend,
                        rsi_value=50.0,
                        adx_value=25.0
                    )

                    # Enter position immediately (in hedge mode, pass is_hedge=True)
                    self.enter_position(symbol, signal, is_hedge=self.hedge_mode)

                    # Remove from pending
                    del self.pending_reentry[pending_key]

                if pending_keys:
                    continue

                # No position - check if we can trade
                can_trade_result, trade_reason = self.can_trade(symbol)

                if not can_trade_result:
                    if trade_reason == "DAILY_LIMIT":
                        print(f"  {symbol}: DAILY LIMIT REACHED")
                    elif trade_reason == "MAX_POSITIONS":
                        print(f"  {symbol}: MAX POSITIONS REACHED")
                    elif trade_reason.startswith("COOLDOWN"):
                        # Show as momentum check with cooldown timer
                        print(f"  {symbol}: MOM-CHECK @ ${current_price:,.2f} ({mom_sign}{momentum:.2f}% vs {momentum_threshold:.2f}%)")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | {trade_reason.replace('_', ' ')}")
                    else:
                        print(f"  {symbol}: {trade_reason} @ ${current_price:,.2f}")
                    continue

                self.last_check_time[symbol] = datetime.now()

                # Generate signal
                if self.use_mtf:
                    signal = self.signal_generator.generate_signal_mtf(symbol, market_data)
                else:
                    signal = self.signal_generator.generate_signal(symbol, market_data["1m"])

                if signal.signal is not None:
                    # SIGNAL FOUND!
                    print(f"  {symbol}: >>> {signal.signal} SIGNAL <<< Conf={signal.confidence:.0%} @ ${current_price:,.2f}")
                    print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | Mom: {mom_sign}{momentum:.2f}%")
                    print(f"    Reason: {signal.reason}")
                    self.enter_position(symbol, signal)
                else:
                    # No signal - show detailed reason
                    reason = signal.reason if signal.reason else "Filter failed"

                    if "Cooldown" in reason:
                        # Bar-based signal cooldown
                        print(f"  {symbol}: SIGNAL COOLDOWN @ ${current_price:,.2f} | Mom: {mom_sign}{momentum:.2f}%")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | {reason}")
                    elif abs(momentum) < momentum_threshold:
                        # No momentum spike
                        print(f"  {symbol}: MOM-CHECK: No spike ({mom_sign}{momentum:.2f}% < {momentum_threshold:.2f}%) @ ${current_price:,.2f}")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f}")
                    elif "Trend not aligned" in reason:
                        # Momentum but wrong trend
                        print(f"  {symbol}: WAIT @ ${current_price:,.2f} | Mom: {mom_sign}{momentum:.2f}% (Trend mismatch)")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | {reason}")
                    elif "RSI" in reason:
                        # RSI filter
                        print(f"  {symbol}: WAIT @ ${current_price:,.2f} | Mom: {mom_sign}{momentum:.2f}% (RSI filter)")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | {reason}")
                    elif "ADX" in reason or "Weak trend" in reason:
                        # ADX filter
                        print(f"  {symbol}: WAIT @ ${current_price:,.2f} | Mom: {mom_sign}{momentum:.2f}% (Weak trend)")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | {reason}")
                    else:
                        # Generic hold
                        print(f"  {symbol}: HOLD @ ${current_price:,.2f} | Mom: {mom_sign}{momentum:.2f}%")
                        print(f"    RSI: {rsi_val:.1f} | ADX: {adx_val:.1f} | {reason}")

            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")

        # Show detailed position info from BINANCE (fresh data)
        if self.positions:
            mode_label = "LIVE" if not self.testnet else "TESTNET"
            print(f"\n--- BINANCE {mode_label} POSITIONS ({len(self.positions)}) ---")
            leverage = STRATEGY_CONFIG["leverage"]
            total_pnl_from_binance = 0.0  # Track actual total PNL

            for position_key, pos in self.positions.items():
                try:
                    # Extract actual symbol from key (handles both 'DOTUSDT' and 'DOTUSDT_LONG')
                    symbol = self.get_symbol_from_key(position_key)
                    side = pos.side
                    dca_level = pos.dca_count

                    # Get FRESH data from Binance API (in hedge mode, specify position side)
                    if self.hedge_mode:
                        binance_pos = self.client.get_position(symbol, position_side=side)
                    else:
                        binance_pos = self.client.get_position(symbol)
                    if not binance_pos:
                        print(f"  {position_key}: {side} (Position not found on Binance)")
                        continue

                    # Use Binance's actual values
                    entry_price = float(binance_pos.get("entry_price", 0))
                    quantity = float(binance_pos.get("quantity", 0))
                    pnl_dollar = float(binance_pos.get("unrealized_pnl", 0))
                    liq_price = float(binance_pos.get("liquidation_price", 0))
                    margin_used = float(binance_pos.get("isolated_wallet", 0)) or pos.margin_used

                    # Skip if no entry price
                    if entry_price <= 0:
                        print(f"  {position_key}: {side} (SYNCING - waiting for entry price)")
                        continue

                    # Get current price
                    price_data = self.client.get_current_price(symbol)
                    mark_price = price_data["price"]

                    # Add to total PNL
                    total_pnl_from_binance += pnl_dollar

                    # Calculate ROI from actual PNL and margin
                    roi_pct = (pnl_dollar / margin_used * 100) if margin_used > 0 else 0

                    # Position value
                    position_value = mark_price * quantity

                    # Get actual SL/TP from Binance orders
                    tp_price = 0.0
                    sl_price = 0.0
                    try:
                        orders = self.client.get_open_orders(symbol)
                        for order in orders:
                            order_type = order.get("type", "")
                            stop_price = float(order.get("stopPrice", 0))
                            order_side = order.get("side", "")
                            if side == "LONG":
                                if order_type == "TAKE_PROFIT_MARKET" and order_side == "SELL" and stop_price > entry_price:
                                    tp_price = stop_price
                                elif order_type == "STOP_MARKET" and order_side == "SELL" and stop_price < entry_price:
                                    sl_price = stop_price
                            else:
                                if order_type == "TAKE_PROFIT_MARKET" and order_side == "BUY" and stop_price < entry_price:
                                    tp_price = stop_price
                                elif order_type == "STOP_MARKET" and order_side == "BUY" and stop_price > entry_price:
                                    sl_price = stop_price
                    except:
                        pass

                    # Calculate TP away percentage and ROI
                    if tp_price > 0:
                        if side == "LONG":
                            tp_away = ((tp_price - mark_price) / mark_price) * 100
                        else:
                            tp_away = ((mark_price - tp_price) / mark_price) * 100
                        tp_roi = abs(tp_price - entry_price) / entry_price * leverage * 100
                    else:
                        tp_away = 0
                        tp_roi = DCA_CONFIG["take_profit_roi"] * 100

                    # DCA level already set from pos.dca_count above

                    # Next DCA info (ROI-based with SYMBOL-SPECIFIC LEVELS)
                    symbol_settings = SYMBOL_SETTINGS.get(symbol, {})
                    symbol_dca_levels = symbol_settings.get("dca_levels", None)
                    
                    if symbol_dca_levels and dca_level < len(symbol_dca_levels):
                        # Use symbol-specific DCA levels (no volatility multiplier needed)
                        next_dca_roi = abs(symbol_dca_levels[dca_level]["trigger_roi"])
                        volatility_mult = 1.0  # Custom levels = no multiplier
                        
                        next_dca_price_pct = next_dca_roi / leverage
                        if side == "LONG":
                            next_dca_price = entry_price * (1 - next_dca_price_pct)
                        else:
                            next_dca_price = entry_price * (1 + next_dca_price_pct)
                    elif dca_level < len(DCA_CONFIG["levels"]):
                        # Fall back to default levels with volatility multiplier
                        base_next_dca_roi = abs(DCA_CONFIG["levels"][dca_level]["trigger_roi"])
                        volatility_mult = symbol_settings.get("dca_volatility_mult", 1.0)
                        next_dca_roi = base_next_dca_roi * volatility_mult
                        
                        next_dca_price_pct = next_dca_roi / leverage
                        if side == "LONG":
                            next_dca_price = entry_price * (1 - next_dca_price_pct)
                        else:
                            next_dca_price = entry_price * (1 + next_dca_price_pct)
                    else:
                        next_dca_price = 0
                        next_dca_roi = 0
                        volatility_mult = 1.0

                    roi_sign = "+" if roi_pct >= 0 else ""

                    # Show volatility multiplier if not default
                    vol_indicator = f" [x{volatility_mult:.1f}]" if volatility_mult != 1.0 else ""

                    # Display position key (includes side in hedge mode)
                    display_label = position_key if self.hedge_mode else symbol
                    print(f"  {display_label}: {quantity:.6f} {side} @ ${entry_price:,.4f}{vol_indicator}")
                    print(f"    Mark: ${mark_price:,.4f} | Value: ${position_value:,.2f} | P&L: ${pnl_dollar:+.2f} ({roi_sign}{roi_pct:.2f}%)")
                    if tp_price > 0 and sl_price > 0:
                        print(f"    TP: ${tp_price:,.4f} ({tp_roi:.1f}%) | SL: ${sl_price:,.4f} | Liq: ${liq_price:,.4f}")
                    elif tp_price > 0:
                        print(f"    TP: ${tp_price:,.4f} ({tp_roi:.1f}%) | SL: Not set | Liq: ${liq_price:,.4f}")
                    else:
                        print(f"    TP/SL: Orders not found | Liq: ${liq_price:,.4f}")
                    # Show trailing TP status if active
                    local_pos = self.positions.get(position_key)
                    is_boosted = local_pos.is_boosted if local_pos else False
                    half_closes = local_pos.half_close_count if local_pos else 0
                    locked_profit = self.boost_locked_profit.get(symbol, 0)

                    # Build boost tag with cycle info
                    if is_boosted and half_closes > 0:
                        boost_tag = f" [BOOSTED x{half_closes} | Locked: ${locked_profit:+.2f}]"
                    elif is_boosted:
                        boost_tag = " [BOOSTED]"
                    else:
                        boost_tag = ""

                    if local_pos and local_pos.trailing_active:
                        trailing_config = DCA_CONFIG.get("trailing_tp", {})
                        trail_distance = trailing_config.get("trail_distance_roi", 0.15) * 100
                        trigger_roi = (local_pos.peak_roi - trailing_config.get("trail_distance_roi", 0.15)) * 100
                        print(f"    DCA: {dca_level}/4{boost_tag} | Margin: ${margin_used:.2f} | TRAILING TP: Peak {local_pos.peak_roi*100:.1f}% (exit @ {trigger_roi:.1f}%)")
                    elif local_pos and local_pos.peak_roi > 0:
                        trailing_config = DCA_CONFIG.get("trailing_tp", {})
                        activation = trailing_config.get("activation_roi", 0.20) * 100
                        print(f"    DCA: {dca_level}/4{boost_tag} | Margin: ${margin_used:.2f} | Peak ROI: {local_pos.peak_roi*100:.1f}% (trail @ {activation:.0f}%)")
                    else:
                        print(f"    DCA: {dca_level}/4{boost_tag} | Margin: ${margin_used:.2f}")
                    if next_dca_price > 0:
                        # BOOSTED positions don't DCA - skip showing next DCA
                        if is_boosted:
                            print(f"    Next DCA: BLOCKED (position is BOOSTED - no further DCA)")
                        else:
                            # Check if DCA trigger is close and show reversal status
                            current_roi_loss = abs(roi_pct) / 100  # Convert to decimal
                            dca_trigger = next_dca_roi  # Already in decimal form (with volatility mult)
                            roi_to_dca = (dca_trigger - current_roi_loss) * 100  # Percentage points to DCA

                            if roi_to_dca <= 5:  # Within 5% ROI of DCA trigger
                                # Check Smart DCA status
                                dca_status = "PENDING"
                                if symbol in self.data_buffer and self.data_buffer[symbol] is not None:
                                    df = self.data_buffer[symbol].get("1m") if isinstance(self.data_buffer[symbol], dict) else self.data_buffer[symbol]
                                    if df is not None:
                                        can_dca, reason = self.signal_generator.can_dca(df, side, dca_level + 1)
                                        if can_dca:
                                            dca_status = f"READY ({reason})"
                                        else:
                                            dca_status = f"WAITING ({reason})"
                                print(f"    Next DCA @ ${next_dca_price:,.4f} (-{next_dca_roi*100:.0f}% ROI) | {roi_to_dca:.1f}% away | {dca_status}")
                            else:
                                print(f"    Next DCA @ ${next_dca_price:,.4f} (-{next_dca_roi*100:.0f}% ROI) | {roi_to_dca:.1f}% away")

                except Exception as e:
                    print(f"  {position_key}: Error displaying - {e}")

            print(f"\n  TOTAL Unrealized P&L: ${total_pnl_from_binance:+.2f}")
        else:
            print("\n--- No open positions ---")

        # Session stats
        if hasattr(self, 'session_start_time'):
            runtime = datetime.now() - self.session_start_time
        else:
            self.session_start_time = datetime.now()
            runtime = timedelta(seconds=0)

        print(f"\n--- SESSION STATS (Runtime: {runtime}) ---")
        print(f"  Trades Today: {self.daily_trades} (W:{self.daily_wins} / L:{self.daily_losses})")
        win_rate = (self.daily_wins / self.daily_trades * 100) if self.daily_trades > 0 else 0
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Daily P&L: ${self.daily_pnl:+.2f}")

        # Per-symbol statistics
        if hasattr(self, "symbol_stats") and self.symbol_stats:
            print(f"\n--- PER-SYMBOL STATS ---")
            for symbol in self.symbols:
                stats = self.symbol_stats.get(symbol, {"wins": 0, "losses": 0, "tp_count": 0, "sl_count": 0, "pnl": 0.0})
                wins = stats["wins"]
                losses = stats["losses"]
                tp_count = stats["tp_count"]
                sl_count = stats["sl_count"]
                pnl = stats["pnl"]
                total_trades = wins + losses
                if total_trades > 0:
                    sym_win_rate = wins / total_trades * 100
                    pnl_sign = "+" if pnl >= 0 else ""
                    print(f"  {symbol}: W:{wins}/L:{losses} ({sym_win_rate:.0f}%) | TP:{tp_count} SL:{sl_count} | P&L: ${pnl_sign}{pnl:.2f}")
                else:
                    print(f"  {symbol}: No trades yet")

        print(f"\nNext scan in 60 seconds... (Ctrl+C to stop)")

    def print_status(self):
        """Print current status with margin allocation info"""
        print("\n" + "="*80)
        mode = "TESTNET" if self.testnet else "LIVE"
        print(f"BINANCE FUTURES {mode} TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        try:
            balance = self.client.get_balance()
            available = self.client.get_available_balance()
            print(f"\nBalance: ${balance:,.2f} (Available: ${available:,.2f})")

            # SMART COMPOUNDING - Show reserve fund status
            if self.smart_compounding_enabled:
                status = self.get_reserve_fund_status()
                print(f"\n[SMART COMPOUNDING] 50/50 Split Active")
                print(f"  Trading Capital: ${status['trading_capital']:.2f} (used for sizing)")
                print(f"  Reserve Fund: ${status['reserve_fund']:.2f} (protected)")
                print(f"  Total Profit: ${status['total_realized_profit']:.2f}")

            print(f"Daily Trades: {self.daily_trades}/{STRATEGY_CONFIG['max_trades_per_day']}")
            print(f"Leverage: {self.leverage}x (ISOLATED)")

            # Show margin allocation summary
            total_margin_used = sum(self.symbol_margin_used.values())
            total_budget = sum(self.symbol_budgets.values())
            if total_budget > 0:
                print(f"\nMargin Allocation: ${total_margin_used:,.2f} / ${total_budget:,.2f} used")

            positions = self.client.get_positions()
            if positions:
                print(f"\nOpen Positions ({len(positions)}):")
                for pos in positions:
                    symbol = pos['symbol']
                    side = pos['side']
                    pnl = pos.get("unrealized_pnl", 0)

                    # In hedge mode, use symbol_side as key
                    if self.hedge_mode:
                        pos_key = f"{symbol}_{side}"
                    else:
                        pos_key = symbol

                    local_pos = self.positions.get(pos_key)
                    margin = local_pos.margin_used if local_pos else 0
                    dca_count = local_pos.dca_count if local_pos else 0
                    is_boosted = local_pos.is_boosted if local_pos else False
                    budget = self.symbol_budgets.get(symbol, 0)

                    # Build status string
                    boost_str = " [BOOSTED]" if is_boosted else ""
                    dca_str = f" DCA:{dca_count}/4" if dca_count > 0 else ""

                    print(f"  {symbol} {side}: "
                          f"Entry ${pos['entry_price']:,.2f} | "
                          f"Qty: {pos['quantity']:.6f} | "
                          f"Margin: ${margin:.2f}/{budget:.2f} | "
                          f"P&L: ${pnl:+.2f}{dca_str}{boost_str}")
            else:
                print("\nNo open positions")

        except Exception as e:
            print(f"\nError getting status: {e}")

        print("="*80 + "\n")


    def log_trading_config(self):
        """Log all trading configuration parameters at startup"""
        self.log("="*70)
        self.log("TRADING CONFIGURATION PARAMETERS")
        self.log("="*70)

        # General settings
        self.log(f"Leverage: {STRATEGY_CONFIG.get('leverage', 20)}x")
        self.log(f"ADX Threshold (Strong Trend): {self.adx_threshold}")
        self.log(f"Strong Trend DCA Block: ALL DCA on loser side")

        # Default DCA Levels
        self.log("DEFAULT DCA LEVELS:")
        self.log(f"  TP ROI: {DCA_CONFIG.get('take_profit_roi', 0.08)*100:.0f}%")
        self.log(f"  SL ROI: {DCA_CONFIG.get('stop_loss_roi', 0.90)*100:.0f}%")
        for i, lvl in enumerate(DCA_CONFIG.get('levels', []), 1):
            self.log(f"  DCA {i}: Trigger={lvl.get('trigger_roi', 0)*100:.0f}% | Mult={lvl.get('multiplier', 1.0)}x | TP={lvl.get('tp_roi', 0.08)*100:.0f}%")

        # Per-symbol settings
        self.log("PER-SYMBOL DCA SETTINGS:")
        for symbol in self.symbols:
            settings = SYMBOL_SETTINGS.get(symbol, {})
            symbol_dca = settings.get('dca_levels', None)
            vol_mult = settings.get('dca_volatility_mult', 1.0)
            tp_roi = settings.get('tp_roi', DCA_CONFIG.get('take_profit_roi', 0.08))

            self.log(f"  {symbol}:")
            self.log(f"    TP ROI: {tp_roi*100:.0f}%")

            if symbol_dca:
                self.log(f"    Custom DCA Levels (no volatility multiplier):")
                for i, lvl in enumerate(symbol_dca, 1):
                    trend_filter = "+ Trend Filter" if lvl.get('require_trend_filter', False) else ""
                    self.log(f"      DCA {i}: Trigger={lvl.get('trigger_roi', 0)*100:.0f}% | Mult={lvl.get('multiplier', 1.0)}x {trend_filter}")
            else:
                self.log(f"    Using default levels with volatility_mult={vol_mult}x")
                for i, lvl in enumerate(DCA_CONFIG.get('levels', []), 1):
                    effective_trigger = lvl.get('trigger_roi', 0) * vol_mult
                    self.log(f"      DCA {i}: Trigger={effective_trigger*100:.0f}% (base {lvl.get('trigger_roi', 0)*100:.0f}% x {vol_mult})")

        # Boost Mode settings
        self.log("BOOST MODE:")
        self.log("  Trigger at DCA: 3")
        self.log("  Boost multiplier: 1.5x")
        self.log("  Half-close at TP: Yes")
        self.log("  Trailing after half-close: Yes")

        self.log("="*70)

    def run(self, duration_hours: float = 0):
        """
        Run live trading

        Args:
            duration_hours: How long to run (0 = run continuously)
        """
        self.running = True
        start_time = datetime.now()
        continuous = duration_hours == 0
        end_time = None if continuous else start_time + timedelta(hours=duration_hours)

        mode = "TESTNET" if self.testnet else "LIVE"
        duration_str = "CONTINUOUSLY (24/7)" if continuous else f"for {duration_hours} hours"
        self.log(f"Starting {mode} trading {duration_str}")
        self.log(f"Symbols: {', '.join(self.symbols)}")

        # Test connection
        if not self.client.test_connection():
            self.log("Connection failed!", level="ERROR")
            return

        # Get initial balance
        self.starting_balance = self.client.get_balance()
        self.daily_start_balance = self.starting_balance  # Also set daily baseline
        self.log(f"Starting balance: ${self.starting_balance:,.2f}")

        # Log all trading configuration
        self.log_trading_config()

        # Initialize dynamic fund allocation
        self.initialize_dynamic_allocation()

        # Setup symbols (leverage, margin type)
        for symbol in self.symbols:
            self.setup_symbol(symbol)

        # Sync existing positions from previous sessions
        self.sync_existing_positions()

        check_interval = 60  # Scan every 60 seconds (like Forex system)
        status_interval = 300
        last_status = datetime.now()

        # Track session start for runtime display
        self.session_start_time = datetime.now()

        self.log(f"Scan interval: {check_interval}s | Status update: {status_interval}s")

        # ================================================================
        # HEDGE MODE or HYBRID HOLD: AUTO-ENTER ON STARTUP
        # ================================================================
        hedge_config = DCA_CONFIG.get("hedge_mode", {})
        hybrid_config = DCA_CONFIG.get("hybrid_hold", {})

        if hedge_config.get("enabled", False):
            # HEDGE MODE: Enter BOTH LONG and SHORT on each symbol
            self.log("=" * 60, level="TRADE")
            self.log("HEDGE MODE: Auto-entering LONG + SHORT positions...", level="TRADE")
            self.log("=" * 60, level="TRADE")
            for symbol in self.symbols:
                self.auto_enter_hedge_positions(symbol)
                time.sleep(1)  # Pause between symbols to avoid rate limiting

        elif hybrid_config.get("enabled", False) and hybrid_config.get("auto_enter_on_start", True):
            # HYBRID MODE: Enter based on trend direction
            self.log("HYBRID MODE: Auto-entering positions based on trend detection...", level="TRADE")
            for symbol in self.symbols:
                if symbol not in self.positions:
                    self.auto_enter_on_trend(symbol)
                else:
                    self.log(f"HYBRID: {symbol} already has position, skipping auto-enter", level="INFO")

        try:
            while self.running and (continuous or datetime.now() < end_time):
                self.check_daily_reset()

                # CHECK FOR NEW DAY - Restart trading for stopped symbols
                for symbol in self.symbols:
                    if self._check_new_day_restart(symbol):
                        # New day - auto-enter positions for this symbol
                        self.auto_enter_hedge_positions(symbol)

                self.check_entry_signals()
                self.manage_positions()

                # SELF-HEALING: Check and fix any state inconsistencies
                self._check_and_fix_inconsistencies()

                # PERIODIC ORDER CLEANUP - Cancel orphaned/stale orders
                if (datetime.now() - self.last_order_cleanup).total_seconds() >= self.order_cleanup_interval:
                    self.cleanup_orphaned_orders()
                    self.last_order_cleanup = datetime.now()

                if (datetime.now() - last_status).total_seconds() >= status_interval:
                    self.print_status()
                    last_status = datetime.now()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n[!] Stopping trading...")
        finally:
            self.running = False
            self.print_session_summary()
            self.log("Trading stopped")

    def stop(self):
        """Stop trading"""
        self.running = False

    def sync_existing_positions(self):
        """
        Sync existing positions from Binance on startup.
        This allows the bot to manage positions from previous sessions.

        IMPROVED: Now uses saved position_state.json to restore DCA levels
        and boost state instead of guessing from margin used.
        """
        self.log("Checking for existing positions...")

        # Load saved position state first
        saved_state = self._load_position_state()
        saved_positions = saved_state.get("positions", {})
        saved_boost = saved_state.get("boost_state", {})

        try:
            binance_positions = self.client.get_positions()

            if not binance_positions:
                self.log("No existing positions found")
                # Clear saved state if no positions on Binance
                if saved_positions:
                    self._clear_position_state()
                return

            self.log(f"Found {len(binance_positions)} existing position(s)")

            # Restore boost state from saved state
            if saved_boost:
                self.boost_mode_active = saved_boost.get("boost_mode_active", {})
                self.boosted_side = saved_boost.get("boosted_side", {})
                self.boost_trigger_side = saved_boost.get("boost_trigger_side", {})
                self.boost_locked_profit = saved_boost.get("boost_locked_profit", {})
                self.boost_cycle_count = saved_boost.get("boost_cycle_count", {})

                # Log restored boost state
                for symbol, active in self.boost_mode_active.items():
                    if active:
                        boosted = self.boosted_side.get(symbol, "?")
                        trigger = self.boost_trigger_side.get(symbol, "?")
                        self.log(f"  [BOOST] Restored: {symbol} {boosted} is BOOSTED (triggered by {trigger})")

            for pos in binance_positions:
                symbol = pos["symbol"]
                position_side = pos["side"]  # LONG or SHORT

                # Skip if not in our trading symbols
                if symbol not in self.symbols:
                    self.log(f"  Skipping {symbol} (not in trading list)")
                    continue

                # In hedge mode, use SYMBOL_SIDE as key (e.g., DOTUSDT_LONG)
                if self.hedge_mode:
                    position_key = f"{symbol}_{position_side}"
                else:
                    position_key = symbol

                # Skip if already tracking
                if position_key in self.positions:
                    continue

                # Estimate margin used based on position value
                position_value = pos["entry_price"] * pos["quantity"]
                estimated_margin = position_value / self.leverage

                # Get actual margin from Binance if available
                actual_margin = float(pos.get("isolated_wallet", 0)) or estimated_margin

                # ================================================================
                # IMPROVED: Use saved state if available, otherwise estimate
                # ================================================================
                saved_pos = saved_positions.get(position_key, {})

                if saved_pos:
                    # Use saved DCA level and boost state
                    dca_level = saved_pos.get("dca_count", 0)
                    is_boosted = saved_pos.get("is_boosted", False)
                    boost_multiplier = saved_pos.get("boost_multiplier", 1.0)
                    half_close_count = saved_pos.get("half_close_count", 0)
                    peak_roi = saved_pos.get("peak_roi", 0.0)
                    trailing_active = saved_pos.get("trailing_active", False)
                    self.log(f"  [STATE] {position_key}: Restored DCA={dca_level}/4, Boosted={is_boosted}")
                else:
                    # Fallback: estimate DCA level from margin used
                    dca_level = self.detect_dca_level_from_margin(symbol, actual_margin)
                    is_boosted = False
                    boost_multiplier = 1.0
                    half_close_count = 0
                    peak_roi = 0.0
                    trailing_active = False
                    self.log(f"  [STATE] {position_key}: No saved state, estimated DCA={dca_level}/4")

                # ================================================================
                # FIX: Cross-reference global boost_state with position
                # If boost_state says this symbol+side should be boosted, apply it
                # Also verify position size matches boosted amount (1.5x)
                # ================================================================
                boost_qty_added = False
                if self.boost_mode_active.get(symbol, False):
                    boosted_side_for_symbol = self.boosted_side.get(symbol, None)
                    if boosted_side_for_symbol == position_side:
                        if not is_boosted:
                            is_boosted = True
                            boost_multiplier = self.boost_multiplier
                            self.log(f"  [BOOST FIX] {position_key}: Applied boost flag from global state")

                        # Check if position size is correct for boosted (should be ~1.5x base)
                        # Base position margin should be around hedge_budget / 2 per side
                        # If current margin is close to base (not 1.5x), we need to add 0.5x
                        symbol_config = SYMBOL_SETTINGS.get(symbol, {})
                        qty_precision = symbol_config.get("qty_precision", 2)

                        # Calculate expected base qty (non-boosted)
                        # Base margin = ~$5-6 for a $75 balance split across 2 symbols, 2 sides
                        expected_base_margin = 5.0  # Approximate base margin
                        current_qty = pos["quantity"]

                        # Check saved state for original quantity before boost
                        saved_qty = saved_pos.get("quantity", 0) if saved_pos else 0

                        # If margin is close to base (not boosted size), add 0.5x
                        if actual_margin < expected_base_margin * 1.3:  # Less than 1.3x base = not boosted
                            try:
                                boost_add_qty = current_qty * 0.5
                                boost_add_qty = round(boost_add_qty, qty_precision)

                                if boost_add_qty > 0:
                                    boost_side = "SELL" if position_side == "SHORT" else "BUY"
                                    self.log(f"  [BOOST FIX] {position_key}: Position size not boosted, adding {boost_add_qty} ({boost_side})")

                                    boost_order = self.client.place_market_order(
                                        symbol,
                                        boost_side,
                                        boost_add_qty,
                                        position_side=position_side
                                    )

                                    if boost_order and "orderId" in boost_order:
                                        # Update position quantity
                                        pos["quantity"] = current_qty + boost_add_qty
                                        actual_margin = actual_margin * 1.5  # Approximate new margin
                                        boost_qty_added = True
                                        self.log(f"  [BOOST FIX] {position_key}: SUCCESS - Now has {pos['quantity']} qty (1.5x boosted)")
                                    else:
                                        self.log(f"  [BOOST FIX] {position_key}: FAILED to add boost qty: {boost_order}", level="WARN")
                            except Exception as e:
                                self.log(f"  [BOOST FIX] {position_key}: ERROR adding boost qty: {e}", level="ERROR")

                # Create local position object
                self.positions[position_key] = LivePosition(
                    symbol=symbol,
                    side=position_side,
                    entry_price=pos["entry_price"],
                    quantity=pos["quantity"],
                    entry_time=datetime.now(),  # Unknown actual entry time
                    avg_entry_price=pos["entry_price"],
                    margin_used=actual_margin,
                    dca_count=dca_level,
                    is_boosted=is_boosted,
                    boost_multiplier=boost_multiplier,
                    half_close_count=half_close_count,
                    peak_roi=peak_roi,
                    trailing_active=trailing_active
                )

                self.symbol_margin_used[position_key] = actual_margin

                # Calculate current P&L
                current_price = pos["entry_price"]
                try:
                    price_data = self.client.get_current_price(symbol)
                    current_price = price_data["price"]

                    if pos["side"] == "LONG":
                        pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
                    else:
                        pnl_pct = ((pos["entry_price"] - current_price) / pos["entry_price"]) * 100

                    boost_str = " [BOOSTED]" if is_boosted else ""
                    self.log(f"  Synced {symbol} {pos['side']}: Entry ${pos['entry_price']:,.2f} | Qty: {pos['quantity']:.6f} | P&L: {pnl_pct:+.2f}% | DCA: {dca_level}/4{boost_str}")
                except:
                    boost_str = " [BOOSTED]" if is_boosted else ""
                    self.log(f"  Synced {symbol} {pos['side']}: Entry ${pos['entry_price']:,.2f} | Qty: {pos['quantity']:.6f} | DCA: {dca_level}/4{boost_str}")

                # Check if position has SL/TP orders - if DCA level > 0, update TP to reduced level
                self._ensure_sl_tp_orders(symbol, pos, dca_level, position_key=position_key)

            self.log(f"Position sync complete - Managing {len(self.positions)} position(s)")

        except Exception as e:
            self.log(f"Error syncing positions: {e}", level="ERROR")

    def _ensure_sl_tp_orders(self, symbol: str, pos: dict, dca_level: int = 0, position_key: str = None):
        """
        Check if a position has SL/TP orders, and place/update them if needed.
        This is called for synced positions from previous sessions.
        If DCA level > 0, update TP to reduced level for faster exit.
        position_key is used for hedge mode (e.g., DOTUSDT_LONG)
        """
        # Use position_key for self.positions lookup, default to symbol for non-hedge mode
        if position_key is None:
            position_key = symbol

        try:
            # Get open orders for this symbol
            orders = self.client.get_open_orders(symbol)

            has_sl = False
            has_tp = False
            tp_order_id = None
            sl_order_id = None
            current_tp_price = 0.0
            current_sl_price = 0.0

            # Get the position side for filtering orders in hedge mode
            position_side = pos["side"]  # LONG or SHORT

            # Collect ALL SL and TP orders for this position side
            sl_orders = []
            tp_orders = []

            self.log(f"  {symbol} [{position_side}]: Found {len(orders)} open orders")
            for order in orders:
                order_type = order.get("type", "")
                order_side = order.get("side", "")
                order_position_side = order.get("positionSide", "BOTH")
                trigger_price = float(order.get("stopPrice", order.get("triggerPrice", 0)))
                order_id = order.get("orderId")

                # In hedge mode, only count orders that match our position side
                if self.hedge_mode and order_position_side != position_side:
                    continue

                self.log(f"    Order: type={order_type}, side={order_side}, positionSide={order_position_side}, trigger=${trigger_price:,.4f}")
                if order_type == "STOP_MARKET":
                    sl_orders.append({"id": order_id, "price": trigger_price})
                elif order_type == "TAKE_PROFIT_MARKET":
                    tp_orders.append({"id": order_id, "price": trigger_price})

            # CLEANUP: Cancel duplicate orders, keep only the newest (last) one
            if len(sl_orders) > 1:
                self.log(f"  [CLEANUP] Found {len(sl_orders)} duplicate SL orders, cancelling extras...")
                for sl in sl_orders[:-1]:  # Cancel all but the last one
                    try:
                        self.client.cancel_order(symbol, sl["id"], is_algo_order=True)
                        self.log(f"    Cancelled duplicate SL order {sl['id']} @ ${sl['price']:,.4f}")
                    except Exception as e:
                        self.log(f"    Failed to cancel SL {sl['id']}: {e}", level="WARN")
                sl_orders = [sl_orders[-1]]  # Keep only the last one

            if len(tp_orders) > 1:
                self.log(f"  [CLEANUP] Found {len(tp_orders)} duplicate TP orders, cancelling extras...")
                for tp in tp_orders[:-1]:  # Cancel all but the last one
                    try:
                        self.client.cancel_order(symbol, tp["id"], is_algo_order=True)
                        self.log(f"    Cancelled duplicate TP order {tp['id']} @ ${tp['price']:,.4f}")
                    except Exception as e:
                        self.log(f"    Failed to cancel TP {tp['id']}: {e}", level="WARN")
                tp_orders = [tp_orders[-1]]  # Keep only the last one

            # Set final values
            has_sl = len(sl_orders) > 0
            has_tp = len(tp_orders) > 0
            sl_order_id = sl_orders[0]["id"] if has_sl else None
            tp_order_id = tp_orders[0]["id"] if has_tp else None
            current_sl_price = sl_orders[0]["price"] if has_sl else 0.0
            current_tp_price = tp_orders[0]["price"] if has_tp else 0.0

            # Need to place missing orders
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]
            # position_side already set above for order filtering

            if entry_price <= 0:
                self.log(f"  Cannot place SL/TP for {symbol}: entry_price is {entry_price}", level="WARN")
                return

            # Calculate SL/TP prices using ROI-BASED calculation for leveraged scalping
            # Formula: price_move = roi / leverage
            leverage = STRATEGY_CONFIG["leverage"]  # 20x

            if DCA_CONFIG["enabled"]:
                # Get TP based on DCA level (reduced TP for faster exit on DCA positions)
                if dca_level > 0 and dca_level <= len(DCA_CONFIG["levels"]):
                    # Use reduced TP from DCA level config
                    dca_level_config = DCA_CONFIG["levels"][dca_level - 1]
                    tp_roi = dca_level_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])
                    self.log(f"  {symbol} DCA level {dca_level}: Using reduced TP {tp_roi*100:.0f}% ROI")
                else:
                    # Initial entry - use default TP
                    tp_roi = DCA_CONFIG["take_profit_roi"]    # 8% ROI

                # ALWAYS use 90% ROI SL regardless of DCA level
                sl_roi = DCA_CONFIG["stop_loss_roi"]  # 90% ROI - ALWAYS SAME

                tp_price_pct = tp_roi / leverage
                sl_price_pct = sl_roi / leverage
            else:
                # Price-based for non-DCA (legacy)
                tp_price_pct = STRATEGY_CONFIG["take_profit_pct"]  # 2% price move
                sl_price_pct = STRATEGY_CONFIG["stop_loss_pct"]    # 1% price move
                tp_roi = tp_price_pct
                sl_roi = sl_price_pct

            if position_side == "LONG":
                stop_loss_price = entry_price * (1 - sl_price_pct)
                take_profit_price = entry_price * (1 + tp_price_pct)
                sl_side = "SELL"
                tp_side = "SELL"
            else:
                stop_loss_price = entry_price * (1 + sl_price_pct)
                take_profit_price = entry_price * (1 - tp_price_pct)
                sl_side = "BUY"
                tp_side = "BUY"

            # ================================================================
            # LIQUIDATION PROTECTION: Ensure SL is ALWAYS before liquidation
            # ================================================================
            liq_price = float(pos.get("liquidation_price", 0))
            if liq_price > 0:
                buffer_pct = DCA_CONFIG.get("liquidation_buffer_pct", 0.01)  # 1% buffer
                if position_side == "LONG":
                    # For LONG: SL must be ABOVE liquidation price
                    min_sl_price = liq_price * (1 + buffer_pct)
                    if stop_loss_price < min_sl_price:
                        self.log(f"  SL ${stop_loss_price:,.2f} too close to liquidation ${liq_price:,.2f}, adjusting to ${min_sl_price:,.2f}", level="WARN")
                        stop_loss_price = min_sl_price
                else:
                    # For SHORT: SL must be BELOW liquidation price
                    max_sl_price = liq_price * (1 - buffer_pct)
                    if stop_loss_price > max_sl_price:
                        self.log(f"  SL ${stop_loss_price:,.2f} too close to liquidation ${liq_price:,.2f}, adjusting to ${max_sl_price:,.2f}", level="WARN")
                        stop_loss_price = max_sl_price

            # Place or UPDATE SL order - ALWAYS ensure SL is at correct 90% ROI level
            need_new_sl = not has_sl
            hedge_position_side = position_side if self.hedge_mode else None

            # Check if existing SL is at wrong price (needs update)
            if has_sl and current_sl_price > 0:
                # Calculate how far off the current SL is from correct price
                sl_tolerance = 0.001  # 0.1% tolerance
                if position_side == "LONG":
                    # For LONG, SL should be below entry. Check if current SL is wrong
                    if abs(current_sl_price - stop_loss_price) / stop_loss_price > sl_tolerance:
                        self.log(f"  {symbol}: SL at ${current_sl_price:,.4f} is wrong, should be ${stop_loss_price:,.4f} (90% ROI)")
                        # Cancel old SL and place new one
                        try:
                            self.client.cancel_order(symbol, sl_order_id, is_algo_order=True)
                            self.log(f"  Cancelled old SL order {sl_order_id}")
                        except Exception as e:
                            self.log(f"  Could not cancel old SL: {e}", level="WARN")
                        need_new_sl = True
                else:
                    # For SHORT, SL should be above entry. Check if current SL is wrong
                    if abs(current_sl_price - stop_loss_price) / stop_loss_price > sl_tolerance:
                        self.log(f"  {symbol}: SL at ${current_sl_price:,.4f} is wrong, should be ${stop_loss_price:,.4f} (90% ROI)")
                        # Cancel old SL and place new one
                        try:
                            self.client.cancel_order(symbol, sl_order_id, is_algo_order=True)
                            self.log(f"  Cancelled old SL order {sl_order_id}")
                        except Exception as e:
                            self.log(f"  Could not cancel old SL: {e}", level="WARN")
                        need_new_sl = True

            if need_new_sl:
                if DCA_CONFIG["enabled"]:
                    self.log(f"  Placing SL for {symbol} @ ${stop_loss_price:,.2f} ({sl_roi*100:.0f}% ROI loss)")
                else:
                    self.log(f"  Placing SL for {symbol} @ ${stop_loss_price:,.2f}")
                sl_order = self.client.place_stop_loss(symbol, sl_side, quantity, stop_loss_price, position_side=hedge_position_side)
                if "orderId" in sl_order:
                    if position_key in self.positions:
                        self.positions[position_key].stop_loss_order_id = sl_order["orderId"]
                    self.log(f"  SL order placed: ID={sl_order['orderId']}")
                else:
                    self.log(f"  SL order FAILED: {sl_order}", level="ERROR")

            # Place or UPDATE TP order
            # If DCA level > 0, check if existing TP needs to be reduced
            need_new_tp = not has_tp
            if has_tp and dca_level > 0 and tp_order_id:
                # Check if current TP is higher than reduced TP target
                if position_side == "LONG" and current_tp_price > take_profit_price * 1.001:
                    # Current TP is too high, need to reduce it
                    self.log(f"  {symbol}: Reducing TP from ${current_tp_price:,.4f} to ${take_profit_price:,.4f} (DCA level {dca_level})")
                    try:
                        self.client.cancel_order(symbol, tp_order_id)
                        need_new_tp = True
                    except Exception as e:
                        self.log(f"  Failed to cancel old TP: {e}", level="WARN")
                elif position_side == "SHORT" and current_tp_price < take_profit_price * 0.999:
                    # Current TP is too low (for short), need to increase it
                    self.log(f"  {symbol}: Adjusting TP from ${current_tp_price:,.4f} to ${take_profit_price:,.4f} (DCA level {dca_level})")
                    try:
                        self.client.cancel_order(symbol, tp_order_id)
                        need_new_tp = True
                    except Exception as e:
                        self.log(f"  Failed to cancel old TP: {e}", level="WARN")

            if need_new_tp:
                if DCA_CONFIG["enabled"]:
                    self.log(f"  Placing TP for {symbol} @ ${take_profit_price:,.2f} ({tp_roi*100:.1f}% ROI target)")
                else:
                    self.log(f"  Placing TP for {symbol} @ ${take_profit_price:,.2f}")
                hedge_position_side = position_side if self.hedge_mode else None
                tp_order = self.client.place_take_profit(symbol, tp_side, quantity, take_profit_price, position_side=hedge_position_side)
                if "orderId" in tp_order:
                    if position_key in self.positions:
                        self.positions[position_key].take_profit_order_id = tp_order["orderId"]
                    self.log(f"  TP order placed: ID={tp_order['orderId']}")
                else:
                    self.log(f"  TP order FAILED: {tp_order}", level="ERROR")
            elif has_tp and has_sl:
                self.log(f"  {symbol} already has SL/TP orders")

        except Exception as e:
            self.log(f"  Error ensuring SL/TP for {symbol}: {e}", level="ERROR")

    def close_all_positions(self):
        """Close all open positions with market orders"""
        if not self.positions:
            self.log("No positions to close")
            return

        self.log(f"Closing {len(self.positions)} position(s)...")

        closed_count = 0
        for position_key, pos in list(self.positions.items()):
            try:
                # Extract actual symbol from key (handles both 'DOTUSDT' and 'DOTUSDT_LONG')
                symbol = self.get_symbol_from_key(position_key)

                # Determine close side (opposite of position side)
                close_side = "SELL" if pos.side == "LONG" else "BUY"

                # Place market order to close (in hedge mode, include positionSide)
                self.log(f"  Closing {position_key} {pos.side} ({pos.quantity:.6f})...")
                position_side = pos.side if self.hedge_mode else None
                order_result = self.client.place_market_order(symbol, close_side, pos.quantity, position_side=position_side)

                if "orderId" in order_result:
                    # Cancel any pending SL/TP orders
                    try:
                        self.client.cancel_all_orders(symbol)
                    except:
                        pass

                    # Get fill price
                    fill_price = float(order_result.get("avgPrice", 0)) or current_price
                    if fill_price == 0:
                        time.sleep(0.3)
                        price_data = self.client.get_current_price(symbol)
                        fill_price = price_data["price"]

                    # Calculate P&L
                    if pos.side == "LONG":
                        pnl = (fill_price - pos.avg_entry_price) * pos.quantity
                    else:
                        pnl = (pos.avg_entry_price - fill_price) * pos.quantity

                    self.log(f"  CLOSED {position_key} @ ${fill_price:,.2f} | P&L: ${pnl:+.2f}")
                    self.daily_pnl += pnl

                    # Release margin
                    self.symbol_margin_used[symbol] = 0.0
                    del self.positions[position_key]
                    closed_count += 1
                else:
                    self.log(f"  Failed to close {position_key}: {order_result}", level="ERROR")

            except Exception as e:
                self.log(f"  Error closing {position_key}: {e}", level="ERROR")

        self.log(f"Closed {closed_count}/{len(self.positions) + closed_count} positions")

    def prompt_close_positions(self):
        """Ask user whether to close positions on stop"""
        if not self.positions:
            return

        print("\n" + "!"*60)
        print(f"! You have {len(self.positions)} OPEN POSITION(S) !")
        print("!"*60)

        # Show positions with P&L
        total_pnl = 0.0
        for position_key, pos in self.positions.items():
            try:
                # Extract actual symbol from key (handles both 'DOTUSDT' and 'DOTUSDT_LONG')
                symbol = self.get_symbol_from_key(position_key)
                price_data = self.client.get_current_price(symbol)
                current_price = price_data["price"]

                if pos.side == "LONG":
                    pnl = (current_price - pos.avg_entry_price) * pos.quantity
                    pnl_pct = ((current_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
                else:
                    pnl = (pos.avg_entry_price - current_price) * pos.quantity
                    pnl_pct = ((pos.avg_entry_price - current_price) / pos.avg_entry_price) * 100

                total_pnl += pnl
                print(f"  {position_key} {pos.side}: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            except:
                print(f"  {position_key} {pos.side}: Entry ${pos.avg_entry_price:,.2f}")

        print(f"\n  TOTAL Unrealized P&L: ${total_pnl:+.2f}")
        print("\nOptions:")
        print("  [1] CLOSE ALL - Close all positions now (market orders)")
        print("  [2] KEEP OPEN - Leave positions open (manage manually)")
        print("")

        try:
            choice = input("Enter choice (1 or 2): ").strip()

            if choice == "1":
                print("\nClosing all positions...")
                self.close_all_positions()
            else:
                print("\nKeeping positions open.")
                print("You can manage them manually at: https://demo.binance.com/en/futures")
        except:
            print("\nNo input - keeping positions open.")

    def print_session_summary(self):
        """Print session summary like the Forex system"""
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)

        # Get current balance
        try:
            current_balance = self.client.get_balance()
        except:
            current_balance = self.starting_balance

        # Calculate session P&L
        session_pnl = current_balance - self.starting_balance
        session_pnl_pct = (session_pnl / self.starting_balance * 100) if self.starting_balance > 0 else 0

        print(f"Starting Balance: ${self.starting_balance:,.2f}")
        print(f"Final Balance: ${current_balance:,.2f}")
        print(f"Session P&L: ${session_pnl:+,.2f} ({session_pnl_pct:+.2f}%)")
        print(f"Total Trades: {self.daily_trades}")

        # Show open positions
        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for position_key, pos in self.positions.items():
                try:
                    # Extract actual symbol from key (handles both 'DOTUSDT' and 'DOTUSDT_LONG')
                    symbol = self.get_symbol_from_key(position_key)
                    price_data = self.client.get_current_price(symbol)
                    current_price = price_data["price"]

                    if pos.avg_entry_price > 0:
                        if pos.side == "LONG":
                            pnl_pct = ((current_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
                        else:
                            pnl_pct = ((pos.avg_entry_price - current_price) / pos.avg_entry_price) * 100
                    else:
                        pnl_pct = 0

                    dca_str = f" (DCA{pos.dca_count})" if pos.dca_count > 0 else ""
                    print(f"  {position_key} {pos.side}{dca_str}: Entry ${pos.avg_entry_price:,.2f} | P&L: {pnl_pct:+.2f}%")
                except:
                    print(f"  {position_key} {pos.side}: Entry ${pos.avg_entry_price:,.2f}")
        else:
            print("\nNo open positions")

        print("="*70)

        # Ask about closing positions
        self.prompt_close_positions()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance Futures Live Trading")
    parser.add_argument("--hours", type=float, default=24, help="Duration")
    parser.add_argument("--live", action="store_true", help="Use mainnet (REAL MONEY!)")
    parser.add_argument("--no-mtf", action="store_true", help="Disable MTF")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    testnet = not args.live

    if not args.yes:
        print("\n" + "="*60)
        mode = "MAINNET (REAL MONEY!)" if args.live else "TESTNET"
        print(f"BINANCE FUTURES {mode} TRADING")
        print("="*60)

        if args.live:
            print("\n!!! WARNING: REAL MONEY MODE !!!")
            print("This will place REAL orders with REAL funds!")

        print(f"\nDuration: {args.hours} hours")
        symbols_to_use = FUTURES_SYMBOLS_LIVE if args.live else FUTURES_SYMBOLS_DEMO
        print(f"Symbols: {', '.join(symbols_to_use)}")
        print("="*60)

        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled")
            sys.exit(0)

    engine = BinanceLiveTradingEngine(testnet=testnet, use_mtf=not args.no_mtf)
    engine.run(duration_hours=args.hours)
