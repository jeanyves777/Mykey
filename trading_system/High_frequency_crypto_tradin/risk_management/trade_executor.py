"""
Unified Trade Executor
=======================

Provides consistent order execution interface for:
- Backtesting (simulated execution)
- Paper Trading (Alpaca paper API)
- Live Trading (Alpaca live API)

All modes use the same interface and risk checks.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionMode(Enum):
    """Execution mode for the trade executor."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class Order:
    """Unified order representation."""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Metadata
    signal_confidence: float = 0.0
    model_agreement: int = 0
    reason: str = ""

    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())[:8]


@dataclass
class Fill:
    """Order fill event."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    slippage: float = 0.0


class TradeExecutor:
    """
    Unified Trade Executor for all trading modes.

    This class provides a consistent interface for order execution
    across backtesting, paper trading, and live trading.

    The same risk checks and order logic are applied regardless of mode.
    """

    def __init__(self,
                 mode: ExecutionMode = ExecutionMode.BACKTEST,
                 risk_manager=None,
                 commission_rate: float = 0.001,
                 slippage_pct: float = 0.0005):
        """
        Initialize Trade Executor.

        Args:
            mode: Execution mode (backtest, paper, live)
            risk_manager: RiskManager instance for risk checks
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_pct: Slippage percentage (0.0005 = 0.05%)
        """
        self.mode = mode
        self.risk_manager = risk_manager
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct

        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}

        # Callbacks
        self.on_fill: Optional[Callable[[Fill], None]] = None
        self.on_order_status: Optional[Callable[[Order], None]] = None

        # External API client (for paper/live)
        self.api_client = None

        # Statistics
        self.total_orders = 0
        self.total_fills = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0

    def set_api_client(self, client):
        """Set the API client for paper/live trading."""
        self.api_client = client

    # ==========================================
    # ORDER SUBMISSION
    # ==========================================

    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            Tuple of (success, message)
        """
        # Pre-submission risk check
        if self.risk_manager:
            allowed, reason = self.risk_manager.can_open_position(
                order.symbol,
                1 if order.side == OrderSide.BUY else -1,
                order.signal_confidence
            )
            if not allowed:
                order.status = OrderStatus.REJECTED
                order.reason = reason
                logger.warning(f"Order rejected: {reason}")
                return False, reason

        # Generate order ID if not set
        if not order.order_id:
            order.order_id = str(uuid.uuid4())[:8]

        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED

        self.pending_orders[order.order_id] = order
        self.total_orders += 1

        logger.info(f"Order submitted: {order.order_id} {order.side.value} "
                   f"{order.quantity} {order.symbol} @ {order.order_type.value}")

        # Execute based on mode
        if self.mode == ExecutionMode.BACKTEST:
            return self._execute_backtest(order)
        elif self.mode == ExecutionMode.PAPER:
            return self._execute_paper(order)
        else:
            return self._execute_live(order)

    def submit_bracket_order(self,
                             symbol: str,
                             side: OrderSide,
                             quantity: float,
                             stop_loss: float,
                             take_profit: float,
                             confidence: float = 0.0) -> Tuple[bool, str, Order]:
        """
        Submit a bracket order (entry + stop loss + take profit).

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Signal confidence

        Returns:
            Tuple of (success, message, order)
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_confidence=confidence
        )

        success, message = self.submit_order(order)
        return success, message, order

    # ==========================================
    # EXECUTION BY MODE
    # ==========================================

    def _execute_backtest(self, order: Order) -> Tuple[bool, str]:
        """Execute order in backtest mode (immediate fill with slippage)."""
        # Simulate slippage
        slippage = order.limit_price * self.slippage_pct if order.limit_price else 0

        if order.order_type == OrderType.MARKET:
            # Market orders fill immediately at current price + slippage
            fill_price = order.limit_price or 0  # Should be set by caller

            if order.side == OrderSide.BUY:
                fill_price += slippage
            else:
                fill_price -= slippage

            # Calculate commission
            commission = fill_price * order.quantity * self.commission_rate

            # Create fill
            self._process_fill(order, order.quantity, fill_price, commission, slippage)

            return True, "Filled"

        elif order.order_type == OrderType.LIMIT:
            # Limit orders go to pending (checked on price updates)
            return True, "Pending"

        return False, "Unknown order type"

    def _execute_paper(self, order: Order) -> Tuple[bool, str]:
        """Execute order via Alpaca Paper API."""
        if not self.api_client:
            logger.error("No API client configured for paper trading")
            return False, "No API client"

        try:
            # Submit to Alpaca
            api_order = self.api_client.submit_order(
                symbol=order.symbol.replace("/", ""),
                qty=order.quantity,
                side=order.side.value,
                order_type=order.order_type.value,
                time_in_force='gtc'
            )

            # Update order with API response
            order.order_id = str(api_order.id)
            order.status = OrderStatus.SUBMITTED

            return True, "Submitted to Alpaca"

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = str(e)
            logger.error(f"Paper order failed: {e}")
            return False, str(e)

    def _execute_live(self, order: Order) -> Tuple[bool, str]:
        """Execute order via Alpaca Live API."""
        # Same as paper but with additional safety checks
        if not self.api_client:
            logger.error("No API client configured for live trading")
            return False, "No API client"

        # Additional live trading safeguards
        if order.quantity * (order.limit_price or 0) > 10000:
            logger.warning("Large order detected in live mode")

        return self._execute_paper(order)

    # ==========================================
    # FILL PROCESSING
    # ==========================================

    def _process_fill(self,
                      order: Order,
                      quantity: float,
                      price: float,
                      commission: float,
                      slippage: float = 0.0):
        """Process an order fill."""
        order.filled_quantity = quantity
        order.filled_price = price
        order.commission = commission
        order.slippage = slippage
        order.filled_at = datetime.now()
        order.status = OrderStatus.FILLED

        # Move to filled orders
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        self.filled_orders[order.order_id] = order

        # Update statistics
        self.total_fills += 1
        self.total_commission += commission
        self.total_slippage += slippage

        # Create fill event
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=datetime.now(),
            slippage=slippage
        )

        # Update risk manager
        if self.risk_manager:
            side = 1 if order.side == OrderSide.BUY else -1
            self.risk_manager.on_trade_opened(
                order.symbol, side, price, quantity,
                order.stop_loss, order.take_profit
            )

        # Trigger callback
        if self.on_fill:
            self.on_fill(fill)

        logger.info(f"Fill: {order.order_id} {quantity} @ ${price:.2f} "
                   f"(commission: ${commission:.2f})")

    def on_price_update(self, symbol: str, price: float, high: float, low: float):
        """
        Process price update for pending orders.

        Called by backtest engine or live data feed to check
        if any pending orders should be filled.
        """
        orders_to_process = [
            o for o in self.pending_orders.values()
            if o.symbol == symbol and o.status == OrderStatus.SUBMITTED
        ]

        for order in orders_to_process:
            if order.order_type == OrderType.LIMIT:
                # Check if limit price is reached
                if order.side == OrderSide.BUY and low <= order.limit_price:
                    self._process_fill(
                        order, order.quantity, order.limit_price,
                        order.limit_price * order.quantity * self.commission_rate
                    )
                elif order.side == OrderSide.SELL and high >= order.limit_price:
                    self._process_fill(
                        order, order.quantity, order.limit_price,
                        order.limit_price * order.quantity * self.commission_rate
                    )

            elif order.order_type == OrderType.STOP:
                # Check if stop price is reached
                if order.side == OrderSide.BUY and high >= order.stop_price:
                    self._process_fill(
                        order, order.quantity, order.stop_price,
                        order.stop_price * order.quantity * self.commission_rate
                    )
                elif order.side == OrderSide.SELL and low <= order.stop_price:
                    self._process_fill(
                        order, order.quantity, order.stop_price,
                        order.stop_price * order.quantity * self.commission_rate
                    )

    # ==========================================
    # ORDER MANAGEMENT
    # ==========================================

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self.cancelled_orders[order_id] = order

            # Cancel on API if paper/live
            if self.mode in [ExecutionMode.PAPER, ExecutionMode.LIVE] and self.api_client:
                try:
                    self.api_client.cancel_order(order_id)
                except Exception as e:
                    logger.error(f"API cancel failed: {e}")

            logger.info(f"Order cancelled: {order_id}")
            return True

        return False

    def cancel_all_orders(self, symbol: str = None):
        """Cancel all pending orders, optionally for a specific symbol."""
        orders_to_cancel = list(self.pending_orders.keys())

        for order_id in orders_to_cancel:
            order = self.pending_orders.get(order_id)
            if order and (symbol is None or order.symbol == symbol):
                self.cancel_order(order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return (
            self.pending_orders.get(order_id) or
            self.filled_orders.get(order_id) or
            self.cancelled_orders.get(order_id)
        )

    def get_pending_orders(self, symbol: str = None) -> List[Order]:
        """Get all pending orders, optionally for a specific symbol."""
        orders = list(self.pending_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_filled_orders(self, symbol: str = None) -> List[Order]:
        """Get all filled orders, optionally for a specific symbol."""
        orders = list(self.filled_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # ==========================================
    # POSITION CLOSE
    # ==========================================

    def close_position(self,
                       symbol: str,
                       quantity: float,
                       current_price: float,
                       reason: str = "") -> Tuple[bool, str]:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            quantity: Quantity to close
            current_price: Current market price
            reason: Reason for closing

        Returns:
            Tuple of (success, message)
        """
        # Determine side (opposite of position)
        if self.risk_manager and symbol in self.risk_manager.open_positions:
            pos = self.risk_manager.open_positions[symbol]
            close_side = OrderSide.SELL if pos['side'] == 1 else OrderSide.BUY
        else:
            close_side = OrderSide.SELL  # Default to sell

        order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            limit_price=current_price,
            reason=reason
        )

        success, message = self.submit_order(order)

        if success and self.risk_manager:
            # Calculate P&L
            if symbol in self.risk_manager.open_positions:
                pos = self.risk_manager.open_positions[symbol]
                if pos['side'] == 1:  # Was long
                    pnl = (current_price - pos['entry_price']) * quantity
                else:  # Was short
                    pnl = (pos['entry_price'] - current_price) * quantity

                pnl -= order.commission

                self.risk_manager.on_trade_closed(symbol, pnl, current_price)

        return success, message

    # ==========================================
    # STATISTICS
    # ==========================================

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            'mode': self.mode.value,
            'total_orders': self.total_orders,
            'total_fills': self.total_fills,
            'pending_orders': len(self.pending_orders),
            'filled_orders': len(self.filled_orders),
            'cancelled_orders': len(self.cancelled_orders),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'fill_rate': self.total_fills / self.total_orders if self.total_orders > 0 else 0
        }

    def reset(self):
        """Reset the executor state."""
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()
        self.total_orders = 0
        self.total_fills = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
