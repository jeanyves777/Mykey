"""
Position Manager
=================

Centralized position tracking and management.
Used consistently across all trading modes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime = field(default_factory=datetime.now)

    # Price tracking
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0

    # P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0

    # Risk levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None

    # Metadata
    signal_confidence: float = 0.0
    model_agreement: int = 0

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.current_price = self.entry_price

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost of position."""
        return self.quantity * self.entry_price

    @property
    def holding_time(self) -> float:
        """Holding time in seconds."""
        return (datetime.now() - self.entry_time).total_seconds()

    def update_price(self, price: float, high: float = None, low: float = None):
        """Update position with new price data."""
        self.current_price = price
        self.highest_price = max(self.highest_price, high or price)
        self.lowest_price = min(self.lowest_price, low or price)

        # Calculate unrealized P&L
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

        self.unrealized_pnl_pct = self.unrealized_pnl / self.cost_basis if self.cost_basis > 0 else 0


class PositionManager:
    """
    Centralized Position Manager.

    Tracks all open positions and provides unified interface
    for position operations across all trading modes.
    """

    def __init__(self, risk_manager=None):
        """
        Initialize Position Manager.

        Args:
            risk_manager: RiskManager instance for risk checks
        """
        self.risk_manager = risk_manager
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.position_history: List[dict] = []

    # ==========================================
    # POSITION OPERATIONS
    # ==========================================

    def open_position(self,
                      symbol: str,
                      side: PositionSide,
                      quantity: float,
                      price: float,
                      stop_loss: float = None,
                      take_profit: float = None,
                      confidence: float = 0.0,
                      agreement: int = 0) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: Position side (long/short)
            quantity: Position size
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: ML signal confidence
            agreement: Number of models agreeing

        Returns:
            The created Position object
        """
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return self.positions[symbol]

        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_confidence=confidence,
            model_agreement=agreement
        )

        self.positions[symbol] = position

        logger.info(f"Position opened: {symbol} {side.value} {quantity} @ ${price:.2f}")

        return position

    def close_position(self,
                       symbol: str,
                       price: float,
                       reason: str = "") -> Tuple[Optional[Position], float]:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            price: Exit price
            reason: Reason for closing

        Returns:
            Tuple of (closed_position, pnl)
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None, 0.0

        position = self.positions.pop(symbol)
        position.update_price(price)

        # Calculate realized P&L
        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - price) * position.quantity

        position.realized_pnl = pnl

        # Record in history
        self.closed_positions.append(position)
        self.position_history.append({
            'symbol': symbol,
            'side': position.side.value,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': pnl / position.cost_basis if position.cost_basis > 0 else 0,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'holding_time': position.holding_time,
            'reason': reason,
            'confidence': position.signal_confidence
        })

        logger.info(f"Position closed: {symbol} @ ${price:.2f} P&L: ${pnl:.2f} ({reason})")

        return position, pnl

    def update_position(self, symbol: str, price: float, high: float = None, low: float = None):
        """Update a position with new price data."""
        if symbol in self.positions:
            self.positions[symbol].update_price(price, high, low)

    def update_all_positions(self, prices: Dict[str, float]):
        """Update all positions with current prices."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    # ==========================================
    # STOP LOSS / TAKE PROFIT
    # ==========================================

    def set_stop_loss(self, symbol: str, price: float):
        """Set stop loss for a position."""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = price
            logger.info(f"Stop loss set for {symbol}: ${price:.2f}")

    def set_take_profit(self, symbol: str, price: float):
        """Set take profit for a position."""
        if symbol in self.positions:
            self.positions[symbol].take_profit = price
            logger.info(f"Take profit set for {symbol}: ${price:.2f}")

    def update_trailing_stop(self, symbol: str, trailing_pct: float):
        """Update trailing stop for a position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        if pos.side == PositionSide.LONG:
            # For long, trail below highest price
            new_stop = pos.highest_price * (1 - trailing_pct)
            if pos.trailing_stop is None or new_stop > pos.trailing_stop:
                pos.trailing_stop = new_stop
        else:
            # For short, trail above lowest price
            new_stop = pos.lowest_price * (1 + trailing_pct)
            if pos.trailing_stop is None or new_stop < pos.trailing_stop:
                pos.trailing_stop = new_stop

    def check_exits(self, symbol: str, current_price: float, high: float, low: float) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            symbol: Symbol to check
            current_price: Current price
            high: High price for this bar
            low: Low price for this bar

        Returns:
            Tuple of (should_exit, reason)
        """
        if symbol not in self.positions:
            return False, ""

        pos = self.positions[symbol]
        pos.update_price(current_price, high, low)

        # Check stop loss
        if pos.stop_loss:
            if pos.side == PositionSide.LONG and low <= pos.stop_loss:
                return True, "Stop loss triggered"
            if pos.side == PositionSide.SHORT and high >= pos.stop_loss:
                return True, "Stop loss triggered"

        # Check take profit
        if pos.take_profit:
            if pos.side == PositionSide.LONG and high >= pos.take_profit:
                return True, "Take profit triggered"
            if pos.side == PositionSide.SHORT and low <= pos.take_profit:
                return True, "Take profit triggered"

        # Check trailing stop
        if pos.trailing_stop:
            if pos.side == PositionSide.LONG and low <= pos.trailing_stop:
                return True, "Trailing stop triggered"
            if pos.side == PositionSide.SHORT and high >= pos.trailing_stop:
                return True, "Trailing stop triggered"

        return False, ""

    # ==========================================
    # QUERIES
    # ==========================================

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return symbol in self.positions

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_total_exposure(self) -> float:
        """Get total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_symbols(self) -> List[str]:
        """Get list of symbols with open positions."""
        return list(self.positions.keys())

    # ==========================================
    # PORTFOLIO METRICS
    # ==========================================

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        positions = list(self.positions.values())

        if not positions:
            return {
                'position_count': 0,
                'total_exposure': 0,
                'total_unrealized_pnl': 0,
                'long_exposure': 0,
                'short_exposure': 0
            }

        long_positions = [p for p in positions if p.side == PositionSide.LONG]
        short_positions = [p for p in positions if p.side == PositionSide.SHORT]

        return {
            'position_count': len(positions),
            'long_count': len(long_positions),
            'short_count': len(short_positions),
            'total_exposure': sum(p.market_value for p in positions),
            'long_exposure': sum(p.market_value for p in long_positions),
            'short_exposure': sum(p.market_value for p in short_positions),
            'total_unrealized_pnl': sum(p.unrealized_pnl for p in positions),
            'best_position': max(positions, key=lambda p: p.unrealized_pnl).symbol if positions else None,
            'worst_position': min(positions, key=lambda p: p.unrealized_pnl).symbol if positions else None
        }

    def get_position_history(self) -> List[dict]:
        """Get history of all closed positions."""
        return self.position_history.copy()

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from closed positions."""
        if not self.position_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0
            }

        trades = self.position_history
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'max_win': max(wins) if wins else 0,
            'max_loss': min(losses) if losses else 0,
            'avg_holding_time': np.mean([t['holding_time'] for t in trades]) if trades else 0
        }

    # ==========================================
    # UTILITIES
    # ==========================================

    def close_all_positions(self, prices: Dict[str, float], reason: str = "Close all") -> List[Tuple[str, float]]:
        """
        Close all open positions.

        Args:
            prices: Dict of current prices by symbol
            reason: Reason for closing

        Returns:
            List of (symbol, pnl) tuples
        """
        results = []
        symbols = list(self.positions.keys())

        for symbol in symbols:
            price = prices.get(symbol, self.positions[symbol].current_price)
            _, pnl = self.close_position(symbol, price, reason)
            results.append((symbol, pnl))

        return results

    def reset(self):
        """Reset all positions and history."""
        self.positions.clear()
        self.closed_positions.clear()
        self.position_history.clear()
