"""
HTF Confluence Live Trading Engine
==================================
Live trading engine for the HTF Trend + Confluence Strategy.

Strategy:
- HTF (4H): 21/50 EMA crossover trend filter
- LTF (15m): MACD + RSI + EMA (9/21) confluence entry
- Single direction trading (follow the trend)
- MODERATE Config: 20x leverage, 30% ROI TP, 10% ROI SL (3:1 R:R)

Backtest Results (60 days, DOT + BNB):
- Total Return: +105.5%
- Win Rate: 30.7%
- Max Drawdown: 30.0%
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.binance_client import BinanceClient
from strategies.htf_confluence_strategy import (
    HTFConfluenceStrategy,
    MODERATE_CONFIG,
    TrendDirection,
    SignalStrength
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HTFConfluenceLiveEngine:
    """
    Live Trading Engine for HTF Confluence Strategy

    Features:
    - Multi-symbol support (DOT, BNB, etc.)
    - HTF trend detection on 4H timeframe
    - LTF entry signals on 15m timeframe
    - Automatic TP/SL order placement
    - Position tracking and management
    - Cooldown between trades
    """

    def __init__(
        self,
        symbols: List[str] = None,
        config: dict = None,
        testnet: bool = True,
        total_capital: float = None,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize live trading engine.

        Args:
            symbols: List of symbols to trade (default: DOTUSDT, BNBUSDT)
            config: Strategy config (default: MODERATE_CONFIG)
            testnet: Use demo/testnet mode (default: True for safety)
            total_capital: Total capital to use (None = use account balance)
            risk_per_trade: Risk per trade as fraction (default: 2%)
        """
        self.symbols = symbols or ["DOTUSDT", "BNBUSDT"]
        self.config = config or MODERATE_CONFIG
        self.testnet = testnet
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade

        # Extract config values
        self.leverage = self.config["leverage"]
        self.tp_roi = self.config["tp_roi"]
        self.sl_roi = self.config["sl_roi"]

        # Initialize Binance client
        self.client = BinanceClient(testnet=testnet, use_demo=testnet)

        # Initialize strategy for each symbol
        self.strategies = {}
        for symbol in self.symbols:
            self.strategies[symbol] = HTFConfluenceStrategy(**self.config)

        # Position tracking
        self.positions = {}  # symbol -> position info
        self.pending_orders = {}  # symbol -> {tp_order_id, sl_order_id}

        # Cooldown tracking (prevent over-trading)
        self.last_trade_time = {}  # symbol -> datetime
        self.cooldown_minutes = 60  # 1 hour cooldown between trades per symbol

        # Statistics - load from file if exists (persist across restarts)
        self.stats_file = os.path.join(os.path.dirname(__file__), "session_stats.json")
        self._load_stats()

        # Tracked positions file (for detecting closed positions after restart)
        self.positions_file = os.path.join(os.path.dirname(__file__), "tracked_positions.json")

        # Running state
        self.running = False
        self.check_interval = 60  # Check every 60 seconds (on 15m TF, no need for faster)

        logger.info("=" * 60)
        logger.info("HTF CONFLUENCE LIVE TRADING ENGINE")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Mode: {'DEMO/TESTNET' if testnet else 'LIVE MAINNET'}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"TP ROI: {self.tp_roi * 100:.0f}% | SL ROI: {self.sl_roi * 100:.0f}%")
        logger.info(f"R:R Ratio: {self.tp_roi / self.sl_roi:.1f}:1")
        logger.info(f"Risk per trade: {self.risk_per_trade * 100:.0f}%")
        logger.info("=" * 60)

    def _load_stats(self):
        """Load session stats from file (persist across restarts)."""
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.symbol_stats = {}
        for symbol in self.symbols:
            self.symbol_stats[symbol] = {"wins": 0, "losses": 0, "pnl": 0.0}

        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    # Check if stats are from today
                    if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        self.trades_today = data.get("trades_today", 0)
                        self.wins_today = data.get("wins_today", 0)
                        self.losses_today = data.get("losses_today", 0)
                        self.pnl_today = data.get("pnl_today", 0.0)
                        saved_stats = data.get("symbol_stats", {})
                        for symbol in self.symbols:
                            if symbol in saved_stats:
                                self.symbol_stats[symbol] = saved_stats[symbol]
                        logger.info(f"Loaded session stats: {self.wins_today}W/{self.losses_today}L, PnL: ${self.pnl_today:+.2f}")
                    else:
                        logger.info("Stats file is from a previous day, starting fresh")
        except Exception as e:
            logger.warning(f"Could not load stats: {e}")

    def _save_stats(self):
        """Save session stats to file."""
        try:
            data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "trades_today": self.trades_today,
                "wins_today": self.wins_today,
                "losses_today": self.losses_today,
                "pnl_today": self.pnl_today,
                "symbol_stats": self.symbol_stats
            }
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save stats: {e}")

    def _save_tracked_positions(self):
        """Save tracked positions to file for resume after restart."""
        try:
            data = {}
            for symbol, pos in self.positions.items():
                data[symbol] = {
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "quantity": pos["quantity"],
                    "tp_price": pos["tp_price"],
                    "sl_price": pos["sl_price"],
                    "entry_time": pos["entry_time"].isoformat() if isinstance(pos["entry_time"], datetime) else str(pos["entry_time"])
                }
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save positions: {e}")

    def _load_tracked_positions(self) -> Dict:
        """Load tracked positions from file."""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load positions: {e}")
        return {}

    def _check_closed_positions_on_startup(self):
        """Check if any tracked positions were closed while engine was down."""
        saved_positions = self._load_tracked_positions()
        if not saved_positions:
            return

        for symbol, saved_pos in saved_positions.items():
            # Check if position still exists on Binance
            position = self.client.get_position(symbol, saved_pos["side"])

            if not position or position["quantity"] == 0:
                # Position was closed while engine was down
                entry_price = saved_pos["entry_price"]
                tp_price = saved_pos["tp_price"]
                sl_price = saved_pos["sl_price"]
                quantity = saved_pos["quantity"]

                # Get current price to estimate exit type
                try:
                    price_data = self.client.get_current_price(symbol)
                    current_price = price_data["price"]

                    # Determine exit type
                    if saved_pos["side"] == "LONG":
                        if current_price >= tp_price * 0.995:
                            exit_type = "TP"
                            pnl = quantity * (tp_price - entry_price)
                        else:
                            exit_type = "SL"
                            pnl = quantity * (sl_price - entry_price)
                    else:
                        if current_price <= tp_price * 1.005:
                            exit_type = "TP"
                            pnl = quantity * (entry_price - tp_price)
                        else:
                            exit_type = "SL"
                            pnl = quantity * (entry_price - sl_price)

                    roi = (pnl / (quantity * entry_price)) * self.leverage * 100

                    # Update stats
                    if exit_type == "TP":
                        self.wins_today += 1
                        self.symbol_stats[symbol]["wins"] += 1
                        logger.info(f"[{symbol}] Position closed while offline - TP HIT! ROI: +{roi:.1f}%")
                    else:
                        self.losses_today += 1
                        self.symbol_stats[symbol]["losses"] += 1
                        logger.info(f"[{symbol}] Position closed while offline - SL HIT. ROI: {roi:.1f}%")

                    self.trades_today += 1
                    self.pnl_today += pnl
                    self.symbol_stats[symbol]["pnl"] += pnl
                    self._save_stats()

                except Exception as e:
                    logger.error(f"[{symbol}] Failed to detect closed position: {e}")

    def initialize(self) -> bool:
        """
        Initialize the trading engine.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing trading engine...")

        # Check for positions that closed while offline
        self._check_closed_positions_on_startup()

        # Test API connection
        if not self.client.test_connection():
            logger.error("Failed to connect to Binance API")
            return False

        # Get account balance
        try:
            balance = self.client.get_balance()
            available = self.client.get_available_balance()
            logger.info(f"Account Balance: ${balance:,.2f} USDT")
            logger.info(f"Available Balance: ${available:,.2f} USDT")

            # Store initial capital for reference (if specified)
            if self.total_capital is None:
                self.total_capital = available

            # Dynamic allocation: margin is calculated per-trade based on available balance
            # Each symbol gets equal share of available margin when opening
            logger.info(f"Margin Mode: DYNAMIC (Available balance split equally among symbols)")
            logger.info(f"Initial per-symbol margin: ~${available / len(self.symbols):.2f} (will adjust based on availability)")

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return False

        # Set up each symbol
        for symbol in self.symbols:
            try:
                # Set leverage
                result = self.client.set_leverage(symbol, self.leverage)
                if "code" in result and result["code"] != 0:
                    logger.warning(f"[{symbol}] Leverage may already be set: {result}")
                else:
                    logger.info(f"[{symbol}] Leverage set to {self.leverage}x")

                # Set isolated margin
                try:
                    self.client.set_margin_type(symbol, "ISOLATED")
                    logger.info(f"[{symbol}] Margin type set to ISOLATED")
                except:
                    logger.info(f"[{symbol}] Margin type already ISOLATED")

                # Check for existing positions and resume management
                position = self.client.get_position(symbol)
                if position and position.get("quantity", 0) > 0:
                    logger.info(f"[{symbol}] RESUMING existing position: {position['side']} {position['quantity']} @ ${position['entry_price']:,.4f}")

                    # Calculate TP/SL for existing position
                    strategy = self.strategies[symbol]
                    sl_price, tp_price = strategy.calculate_exit_levels(
                        position["entry_price"],
                        position["side"]
                    )

                    # Track position with calculated levels
                    self.positions[symbol] = {
                        "side": position["side"],
                        "entry_price": position["entry_price"],
                        "quantity": position["quantity"],
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "entry_time": datetime.now(),
                        "signal_strength": "RESUMED",
                        "confluence_score": 0
                    }
                    logger.info(f"[{symbol}] TP: ${tp_price:,.4f} | SL: ${sl_price:,.4f}")

                    # Verify TP/SL orders exist for this position
                    self._verify_tp_sl_orders(symbol, position)

            except Exception as e:
                logger.error(f"[{symbol}] Setup failed: {e}")
                return False

        # Save tracked positions for restart detection
        self._save_tracked_positions()

        logger.info("Initialization complete!")
        return True

    def _verify_tp_sl_orders(self, symbol: str, position: dict):
        """
        Verify TP/SL orders exist for a position, create if missing.

        Args:
            symbol: Trading symbol
            position: Position dict with side, quantity, entry_price
        """
        try:
            position_side = position["side"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]

            # Get open orders for this symbol
            open_orders = self.client.get_open_orders(symbol=symbol)

            # Check for TP and SL orders
            has_tp = False
            has_sl = False

            for order in open_orders:
                order_position_side = order.get("positionSide", "")
                if order_position_side != position_side:
                    continue

                order_type = order.get("type", "")
                if order_type == "TAKE_PROFIT_MARKET":
                    has_tp = True
                    logger.info(f"[{symbol}] Found existing TP order: {order.get('orderId')}")
                elif order_type == "STOP_MARKET":
                    has_sl = True
                    logger.info(f"[{symbol}] Found existing SL order: {order.get('orderId')}")

            # Calculate expected TP/SL prices
            strategy = self.strategies[symbol]
            sl_price, tp_price = strategy.calculate_exit_levels(entry_price, position_side)

            close_side = "SELL" if position_side == "LONG" else "BUY"

            # Create missing TP order
            if not has_tp:
                logger.info(f"[{symbol}] Creating missing TP order at ${tp_price:,.4f}")
                tp_result = self.client.place_take_profit(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    take_profit_price=tp_price,
                    position_side=position_side
                )
                if "code" not in tp_result or tp_result.get("code") == 0:
                    logger.info(f"[{symbol}] TP order created: {tp_result.get('orderId')}")
                else:
                    logger.warning(f"[{symbol}] TP order failed: {tp_result}")

            # Create missing SL order
            if not has_sl:
                logger.info(f"[{symbol}] Creating missing SL order at ${sl_price:,.4f}")
                sl_result = self.client.place_stop_loss(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    stop_price=sl_price,
                    position_side=position_side
                )
                if "code" not in sl_result or sl_result.get("code") == 0:
                    logger.info(f"[{symbol}] SL order created: {sl_result.get('orderId')}")
                else:
                    logger.warning(f"[{symbol}] SL order failed: {sl_result}")

        except Exception as e:
            logger.error(f"[{symbol}] Failed to verify TP/SL orders: {e}")
            import traceback
            traceback.print_exc()

    def get_market_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch market data for strategy analysis.

        Args:
            symbol: Trading symbol

        Returns:
            (ltf_df, htf_df) - 15m and 4h DataFrames
        """
        try:
            # Get 15m data (LTF) - need ~100 candles for indicators
            ltf_df = self.client.get_klines(symbol, "15m", 150)

            # Get 4h data (HTF) - need ~100 candles for 50 EMA
            htf_df = self.client.get_klines(symbol, "4h", 150)

            if ltf_df.empty or htf_df.empty:
                logger.warning(f"[{symbol}] Empty data received")
                return None, None

            return ltf_df, htf_df

        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch market data: {e}")
            return None, None

    def check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period.

        Returns:
            True if can trade (not in cooldown)
        """
        if symbol not in self.last_trade_time:
            return True

        elapsed = datetime.now() - self.last_trade_time[symbol]
        if elapsed.total_seconds() < self.cooldown_minutes * 60:
            remaining = self.cooldown_minutes - (elapsed.total_seconds() / 60)
            logger.debug(f"[{symbol}] Cooldown: {remaining:.1f} minutes remaining")
            return False

        return True

    def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """
        Calculate position size using dynamic available margin split equally among symbols.

        Uses CURRENT available balance divided by number of symbols without positions.
        This ensures each symbol gets equal margin based on what's actually available.

        Args:
            symbol: Trading symbol
            entry_price: Entry price

        Returns:
            Position quantity
        """
        # Get CURRENT available balance (dynamic, not fixed)
        available_balance = self.client.get_available_balance()

        # Count how many symbols DON'T have positions (need margin)
        symbols_needing_margin = [s for s in self.symbols if s not in self.positions]
        num_symbols_needing_margin = len(symbols_needing_margin)

        if num_symbols_needing_margin == 0:
            logger.warning(f"[{symbol}] All symbols have positions, cannot calculate margin")
            return 0.0

        # Split available balance equally among symbols needing margin
        margin_per_symbol = available_balance / num_symbols_needing_margin

        # Apply a small buffer (95%) to avoid "insufficient margin" errors
        margin = margin_per_symbol * 0.95

        logger.info(f"[{symbol}] Dynamic margin: ${margin:.2f} (${available_balance:.2f} available / {num_symbols_needing_margin} symbols * 0.95)")

        # Position value = margin * leverage
        position_value = margin * self.leverage

        # Quantity = position_value / price
        quantity = position_value / entry_price

        # Get symbol precision
        from config.trading_config import SYMBOL_SETTINGS
        settings = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = settings.get("qty_precision", 3)

        quantity = round(quantity, qty_precision)

        logger.info(f"[{symbol}] Position size: {quantity} (margin ${margin:.2f} x {self.leverage}x leverage = ${position_value:.2f})")

        return quantity

    def open_position(self, symbol: str, signal) -> bool:
        """
        Open a new position based on signal.

        Args:
            symbol: Trading symbol
            signal: ConfluenceSignal from strategy

        Returns:
            True if position opened successfully
        """
        try:
            # Get current price
            price_data = self.client.get_current_price(symbol)
            current_price = price_data["price"]

            # Determine position side
            if signal.action == "BUY":
                side = "BUY"
                position_side = "LONG"
            else:
                side = "SELL"
                position_side = "SHORT"

            # Calculate position size
            quantity = self.calculate_position_size(symbol, current_price)

            if quantity <= 0:
                logger.warning(f"[{symbol}] Invalid position size: {quantity}")
                return False

            # Place market order
            logger.info(f"[{symbol}] Opening {position_side} position...")
            logger.info(f"[{symbol}] Entry: ${current_price:,.4f} | Qty: {quantity}")

            order_result = self.client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                position_side=position_side
            )

            if "code" in order_result:
                logger.error(f"[{symbol}] Order failed: {order_result}")
                return False

            logger.info(f"[{symbol}] Market order placed: {order_result.get('orderId')}")

            # Wait for fill
            time.sleep(1)

            # Get actual entry price from position
            position = self.client.get_position(symbol, position_side)
            if not position:
                logger.error(f"[{symbol}] Position not found after order")
                return False

            actual_entry = position["entry_price"]
            actual_qty = position["quantity"]

            logger.info(f"[{symbol}] Position opened: {position_side} {actual_qty} @ ${actual_entry:,.4f}")

            # Calculate TP and SL prices
            strategy = self.strategies[symbol]
            sl_price, tp_price = strategy.calculate_exit_levels(actual_entry, position_side)

            logger.info(f"[{symbol}] TP: ${tp_price:,.4f} ({self.tp_roi*100:.0f}% ROI)")
            logger.info(f"[{symbol}] SL: ${sl_price:,.4f} ({self.sl_roi*100:.0f}% ROI)")

            # Place TP order
            close_side = "SELL" if position_side == "LONG" else "BUY"

            tp_result = self.client.place_take_profit(
                symbol=symbol,
                side=close_side,
                quantity=actual_qty,
                take_profit_price=tp_price,
                position_side=position_side
            )

            if "code" in tp_result and tp_result.get("code") != 0:
                logger.warning(f"[{symbol}] TP order warning: {tp_result}")
            else:
                tp_order_id = tp_result.get("orderId") or tp_result.get("algoId")
                logger.info(f"[{symbol}] TP order placed: {tp_order_id}")

            # Place SL order
            sl_result = self.client.place_stop_loss(
                symbol=symbol,
                side=close_side,
                quantity=actual_qty,
                stop_price=sl_price,
                position_side=position_side
            )

            if "code" in sl_result and sl_result.get("code") != 0:
                logger.warning(f"[{symbol}] SL order warning: {sl_result}")
            else:
                sl_order_id = sl_result.get("orderId") or sl_result.get("algoId")
                logger.info(f"[{symbol}] SL order placed: {sl_order_id}")

            # Track position
            self.positions[symbol] = {
                "side": position_side,
                "entry_price": actual_entry,
                "quantity": actual_qty,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_time": datetime.now(),
                "signal_strength": signal.strength.value,
                "confluence_score": signal.confluence_score
            }

            # Save tracked positions for restart detection
            self._save_tracked_positions()

            # Set cooldown
            self.last_trade_time[symbol] = datetime.now()

            return True

        except Exception as e:
            logger.error(f"[{symbol}] Failed to open position: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_position(self, symbol: str) -> Optional[str]:
        """
        Check if position was closed (TP or SL hit).

        Args:
            symbol: Trading symbol

        Returns:
            "TP", "SL", or None if still open
        """
        if symbol not in self.positions:
            return None

        try:
            # Check if position still exists
            tracked = self.positions[symbol]
            position_side = tracked["side"]

            position = self.client.get_position(symbol, position_side)

            if not position or position["quantity"] == 0:
                # Position closed - determine how
                entry_price = tracked["entry_price"]
                tp_price = tracked["tp_price"]
                sl_price = tracked["sl_price"]

                # Get current price to estimate exit
                price_data = self.client.get_current_price(symbol)
                current_price = price_data["price"]

                # Determine exit type based on price proximity
                if position_side == "LONG":
                    if current_price >= tp_price * 0.999:  # Within 0.1% of TP
                        exit_type = "TP"
                        pnl = tracked["quantity"] * (tp_price - entry_price)
                    else:
                        exit_type = "SL"
                        pnl = tracked["quantity"] * (sl_price - entry_price)
                else:  # SHORT
                    if current_price <= tp_price * 1.001:  # Within 0.1% of TP
                        exit_type = "TP"
                        pnl = tracked["quantity"] * (entry_price - tp_price)
                    else:
                        exit_type = "SL"
                        pnl = tracked["quantity"] * (entry_price - sl_price)

                # Update stats
                roi = (pnl / (tracked["quantity"] * entry_price)) * self.leverage * 100

                if exit_type == "TP":
                    self.wins_today += 1
                    self.symbol_stats[symbol]["wins"] += 1
                    logger.info(f"[{symbol}] Position closed - TP HIT! ROI: +{roi:.1f}%")
                else:
                    self.losses_today += 1
                    self.symbol_stats[symbol]["losses"] += 1
                    logger.info(f"[{symbol}] Position closed - SL HIT. ROI: {roi:.1f}%")

                self.pnl_today += pnl
                self.symbol_stats[symbol]["pnl"] += pnl
                self.trades_today += 1

                # Save stats and update tracked positions
                self._save_stats()

                # Clean up
                del self.positions[symbol]
                self._save_tracked_positions()

                # Cancel any remaining orders
                self.client.cancel_orders_for_side(symbol, position_side)

                return exit_type

            return None

        except Exception as e:
            logger.error(f"[{symbol}] Failed to check position: {e}")
            return None

    def analyze_symbol(self, symbol: str) -> Optional[dict]:
        """
        Analyze a symbol for trading signals.

        Args:
            symbol: Trading symbol

        Returns:
            Signal dict or None
        """
        try:
            # Get market data
            ltf_df, htf_df = self.get_market_data(symbol)

            if ltf_df is None or htf_df is None:
                return None

            # Get strategy signal
            strategy = self.strategies[symbol]
            signal = strategy.should_enter(ltf_df, htf_df)

            if signal.action:
                logger.info(f"[{symbol}] Signal: {signal.action} | Strength: {signal.strength.value} | Score: {signal.confluence_score}/4")
                logger.info(f"[{symbol}] Reason: {signal.reason}")

            return signal

        except Exception as e:
            logger.error(f"[{symbol}] Analysis failed: {e}")
            return None

    def run_cycle(self):
        """Run one trading cycle."""
        logger.info("-" * 40)
        logger.info(f"Cycle: {datetime.now().strftime('%H:%M:%S')}")

        for symbol in self.symbols:
            try:
                # Check if we have an existing position
                if symbol in self.positions:
                    # Check if position was closed
                    exit_type = self.check_position(symbol)
                    if exit_type:
                        logger.info(f"[{symbol}] Position exit: {exit_type}")
                    else:
                        # Position still open - show detailed status
                        pos = self.positions[symbol]
                        try:
                            price_data = self.client.get_current_price(symbol)
                            current_price = price_data["price"]
                            entry = pos["entry_price"]
                            qty = pos["quantity"]

                            # Calculate price move %
                            if pos["side"] == "LONG":
                                price_move = (current_price - entry) / entry * 100
                                roi = price_move * self.leverage
                                to_tp_price = (pos["tp_price"] - current_price) / current_price * 100
                                to_sl_price = (current_price - pos["sl_price"]) / current_price * 100
                            else:
                                price_move = (entry - current_price) / entry * 100
                                roi = price_move * self.leverage
                                to_tp_price = (current_price - pos["tp_price"]) / current_price * 100
                                to_sl_price = (pos["sl_price"] - current_price) / current_price * 100

                            # Calculate margin and unrealized PnL
                            position_value = qty * entry
                            margin = position_value / self.leverage
                            unrealized_pnl = margin * (roi / 100)

                            # ROI to TP/SL
                            to_tp_roi = to_tp_price * self.leverage
                            to_sl_roi = to_sl_price * self.leverage

                            logger.info(f"┌─ {symbol} {pos['side']} ─────────────────────────────────")
                            logger.info(f"│ Entry: ${entry:,.4f} | Now: ${current_price:,.4f} | Qty: {qty}")
                            logger.info(f"│ Price Move: {price_move:+.3f}% | ROI: {roi:+.1f}% | PnL: ${unrealized_pnl:+.2f}")
                            logger.info(f"│ Margin: ${margin:.2f} | Position: ${position_value:.2f}")
                            logger.info(f"│ To TP: {to_tp_price:.3f}% ({to_tp_roi:+.1f}% ROI) | To SL: {to_sl_price:.3f}% ({to_sl_roi:.1f}% ROI)")
                            logger.info(f"└────────────────────────────────────────────────")
                        except Exception as e:
                            logger.info(f"[{symbol}] {pos['side']} @ ${pos['entry_price']:,.4f} (monitoring) - {e}")
                    continue

                # Check cooldown
                if not self.check_cooldown(symbol):
                    continue

                # Analyze for new signal
                signal = self.analyze_symbol(symbol)

                if signal and signal.action:
                    # We have a signal - check if strong enough
                    if signal.confluence_score >= 3:
                        logger.info(f"[{symbol}] SIGNAL: {signal.action} (Score: {signal.confluence_score}/4)")

                        # Open position
                        if self.open_position(symbol, signal):
                            logger.info(f"[{symbol}] Position opened successfully")
                        else:
                            logger.warning(f"[{symbol}] Failed to open position")
                    else:
                        logger.debug(f"[{symbol}] Signal too weak: {signal.confluence_score}/4")

            except Exception as e:
                logger.error(f"[{symbol}] Cycle error: {e}")
                import traceback
                traceback.print_exc()

        # Get account info for balance and realized PnL
        try:
            account = self.client.get_account_info()
            total_balance = float(account.get('totalWalletBalance', 0))
            available_balance = float(account.get('availableBalance', 0))

            # Calculate total unrealized PnL across all positions
            total_unrealized_pnl = 0.0
            positions_data = self.client.get_positions()
            for pos in positions_data:
                # get_positions() returns 'unrealized_pnl' (processed field name)
                unrealized = float(pos.get('unrealized_pnl', 0))
                total_unrealized_pnl += unrealized

            logger.info(f"┌─ ACCOUNT SUMMARY ──────────────────────────────")
            logger.info(f"│ Balance: ${total_balance:,.2f} | Available: ${available_balance:,.2f}")
            logger.info(f"│ Total Unrealized PnL: ${total_unrealized_pnl:+,.2f}")
            logger.info(f"│ Session Realized PnL: ${self.pnl_today:+,.2f}")
            logger.info(f"└────────────────────────────────────────────────")
        except Exception as e:
            logger.warning(f"Could not fetch account info: {e}")

        # Log daily stats with per-symbol breakdown
        win_rate = (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0
        logger.info(f"┌─ SESSION STATS ─────────────────────────────────")
        logger.info(f"│ Total: {self.trades_today} trades | {self.wins_today}W/{self.losses_today}L | WR: {win_rate:.0f}%")
        for symbol in self.symbols:
            stats = self.symbol_stats[symbol]
            sym_trades = stats["wins"] + stats["losses"]
            sym_wr = (stats["wins"] / sym_trades * 100) if sym_trades > 0 else 0
            logger.info(f"│ {symbol}: {stats['wins']}W/{stats['losses']}L | WR: {sym_wr:.0f}% | PnL: ${stats['pnl']:+,.2f}")
        logger.info(f"└────────────────────────────────────────────────")

    def run(self):
        """Main trading loop."""
        if not self.initialize():
            logger.error("Initialization failed. Exiting.")
            return

        self.running = True
        logger.info("=" * 60)
        logger.info("LIVE TRADING STARTED")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        try:
            while self.running:
                self.run_cycle()

                # Wait for next cycle
                logger.info(f"Next check in {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nStopping trading engine...")
            self.running = False

        self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        logger.info("=" * 60)
        logger.info("SHUTDOWN")
        logger.info("=" * 60)

        # Log final stats
        logger.info(f"Total trades today: {self.trades_today}")
        logger.info(f"Wins: {self.wins_today} | Losses: {self.losses_today}")
        if self.trades_today > 0:
            logger.info(f"Win rate: {self.wins_today/self.trades_today*100:.1f}%")
        logger.info(f"P&L today: ${self.pnl_today:+,.2f}")

        # List open positions
        if self.positions:
            logger.info("\nOpen positions (will remain open):")
            for symbol, pos in self.positions.items():
                logger.info(f"  {symbol}: {pos['side']} @ ${pos['entry_price']:,.4f}")

        logger.info("=" * 60)
        logger.info("Engine stopped.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HTF Confluence Live Trading Engine")
    parser.add_argument("--symbols", nargs="+", default=["DOTUSDT", "BNBUSDT"],
                        help="Symbols to trade")
    parser.add_argument("--live", action="store_true",
                        help="Use live mainnet (default: demo/testnet)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Total capital to use (default: account balance)")
    parser.add_argument("--risk", type=float, default=0.02,
                        help="Risk per trade (default: 0.02 = 2%%)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds (default: 60)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip confirmation prompt for live trading")

    args = parser.parse_args()

    # Create engine
    engine = HTFConfluenceLiveEngine(
        symbols=args.symbols,
        config=MODERATE_CONFIG,
        testnet=not args.live,
        total_capital=args.capital,
        risk_per_trade=args.risk
    )

    engine.check_interval = args.interval

    # Safety confirmation for live trading
    if args.live and not args.no_confirm:
        print("\n" + "=" * 60)
        print("WARNING: LIVE MAINNET TRADING MODE")
        print("=" * 60)
        print("You are about to trade with REAL MONEY!")
        print(f"Symbols: {', '.join(args.symbols)}")
        print(f"Leverage: {engine.leverage}x")
        print(f"TP: {engine.tp_roi*100:.0f}% ROI | SL: {engine.sl_roi*100:.0f}% ROI")
        print("=" * 60)

        confirm = input("\nType 'CONFIRM' to start live trading: ")
        if confirm != "CONFIRM":
            print("Live trading cancelled.")
            return

    # Run engine
    engine.run()


if __name__ == "__main__":
    main()
