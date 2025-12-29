"""
Position Manager for Forex ML System
====================================

Track open positions and P&L.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class PendingDCAOrder:
    """Represents a pending DCA limit order."""
    level: int           # DCA level (1-4)
    units: float         # Units to add
    trigger_price: float # Limit price for execution
    order_id: str        # OANDA order ID
    placed_time: datetime


@dataclass
class DCAEntry:
    """Represents a DCA entry."""
    level: int           # DCA level (1-4)
    units: float         # Units added
    price: float         # Entry price
    time: datetime       # Entry time
    trade_id: str = ""   # OANDA trade ID


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    units: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    trade_id: str = ""
    unrealized_pnl: float = 0.0
    current_price: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0   # For trailing stop

    # DCA tracking
    dca_entries: List['DCAEntry'] = field(default_factory=list)
    avg_entry_price: float = 0.0     # Weighted average entry after DCAs
    total_units: float = 0.0         # Total units including DCAs
    dca_level: int = 0               # Current DCA level (0 = no DCA yet)
    dca_active: bool = False         # Whether DCA SL/TP is active

    # Pending DCA limit orders
    pending_dca_orders: List['PendingDCAOrder'] = field(default_factory=list)
    use_pending_orders: bool = False  # Whether pending limit orders are used

    def __post_init__(self):
        """Initialize calculated fields."""
        if self.avg_entry_price == 0.0:
            self.avg_entry_price = self.entry_price
        if self.total_units == 0.0:
            self.total_units = self.units
        # Initialize high/low to entry price for proper trailing stop calculation
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price

    def add_dca_entry(self, level: int, units: float, price: float, trade_id: str = "") -> None:
        """Add a DCA entry and recalculate average price."""
        entry = DCAEntry(
            level=level,
            units=units,
            price=price,
            time=datetime.now(),
            trade_id=trade_id
        )
        self.dca_entries.append(entry)
        self.dca_level = level
        self.dca_active = True

        # Recalculate weighted average entry price
        total_cost = self.avg_entry_price * self.total_units + price * units
        self.total_units += units
        self.avg_entry_price = total_cost / self.total_units

        # Remove the corresponding pending order if it exists
        self.pending_dca_orders = [o for o in self.pending_dca_orders if o.level != level]

    def add_pending_dca_order(self, level: int, units: float, trigger_price: float, order_id: str) -> None:
        """Add a pending DCA limit order."""
        order = PendingDCAOrder(
            level=level,
            units=units,
            trigger_price=trigger_price,
            order_id=order_id,
            placed_time=datetime.now()
        )
        self.pending_dca_orders.append(order)
        self.use_pending_orders = True

    def get_pending_order_ids(self) -> List[str]:
        """Get all pending DCA order IDs for cancellation."""
        return [o.order_id for o in self.pending_dca_orders]

    def clear_pending_orders(self) -> None:
        """Clear all pending DCA orders (after cancellation or fill)."""
        self.pending_dca_orders = []

    def get_price_change_pct(self) -> float:
        """Get price change % from average entry."""
        if self.avg_entry_price == 0:
            return 0.0
        if self.direction == 'BUY':
            return (self.current_price - self.avg_entry_price) / self.avg_entry_price
        else:
            return (self.avg_entry_price - self.current_price) / self.avg_entry_price

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'units': self.units,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'trade_id': self.trade_id,
            'unrealized_pnl': self.unrealized_pnl,
            'current_price': self.current_price,
            'dca_level': self.dca_level,
            'avg_entry_price': self.avg_entry_price,
            'total_units': self.total_units,
            'dca_active': self.dca_active
        }


@dataclass
class ClosedTrade:
    """Represents a closed trade."""
    symbol: str
    direction: str
    units: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pips: float
    exit_reason: str  # 'TP', 'SL', 'MANUAL', 'SIGNAL'


class PositionManager:
    """Manages open positions and trade history."""

    def __init__(self, config=None):
        """Initialize position manager."""
        self.config = config
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trade_history: List[ClosedTrade] = []
        self.total_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0

    def open_position(self, symbol: str, direction: str, units: float,
                      entry_price: float, stop_loss: float, take_profit: float,
                      trade_id: str = "") -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
            units: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            trade_id: Optional trade ID

        Returns:
            Position object
        """
        position = Position(
            symbol=symbol,
            direction=direction,
            units=units,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            trade_id=trade_id,
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price
        )

        self.positions[symbol] = position
        return position

    def close_position(self, symbol: str, exit_price: float,
                       exit_reason: str = 'MANUAL') -> Optional[ClosedTrade]:
        """
        Close an existing position.

        Args:
            symbol: Trading pair
            exit_price: Exit price
            exit_reason: Reason for closing ('TP', 'SL', 'MANUAL', 'SIGNAL')

        Returns:
            ClosedTrade object or None
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Calculate P&L in pips
        pip_value = self._get_pip_value(symbol)
        if position.direction == 'BUY':
            pnl_pips = (exit_price - position.entry_price) / pip_value
        else:
            pnl_pips = (position.entry_price - exit_price) / pip_value

        # Calculate P&L in USD
        # For most pairs: 1 pip = $0.0001 * units (for USD quote currency)
        # For XXX/USD pairs: P&L = price_diff * units
        # For USD/XXX pairs: P&L = price_diff * units / exit_price (converted to USD)
        # For cross pairs: More complex conversion needed

        price_diff = abs(exit_price - position.entry_price)

        if symbol.endswith('_USD'):
            # Quote currency is USD (EUR_USD, GBP_USD, etc.)
            # P&L = price_diff * units
            pnl = price_diff * position.units
            if pnl_pips < 0:
                pnl = -pnl
        elif symbol.startswith('USD_'):
            # Base currency is USD (USD_JPY, USD_CHF, etc.)
            # P&L = price_diff * units / exit_price (convert to USD)
            pnl = (price_diff * position.units) / exit_price
            if pnl_pips < 0:
                pnl = -pnl
        else:
            # Cross pair - use approximate calculation
            # For simplicity, use pip value * pips * units / 10000
            pnl = pnl_pips * position.units * pip_value

        # Create closed trade record
        closed_trade = ClosedTrade(
            symbol=symbol,
            direction=position.direction,
            units=position.units,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pips=pnl_pips,
            exit_reason=exit_reason
        )

        # Update statistics
        self.trade_history.append(closed_trade)
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Remove position
        del self.positions[symbol]

        return closed_trade

    def update_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Update position with current price and check for SL/TP.

        Args:
            symbol: Trading pair
            current_price: Current market price

        Returns:
            'SL', 'TP', or None
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        position.current_price = current_price

        # Update high/low for trailing stop
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)

        # NOTE: unrealized_pnl is set from OANDA directly in paper_trading_engine
        # Do NOT calculate it here - use real OANDA data only

        # Check TP/SL
        if position.direction == 'BUY':
            if current_price >= position.take_profit:
                return 'TP'
            if current_price <= position.stop_loss:
                return 'SL'
        else:
            if current_price <= position.take_profit:
                return 'TP'
            if current_price >= position.stop_loss:
                return 'SL'

        return None

    def update_trailing_stop(self, symbol: str, trailing_pips: float,
                             activation_pips: float) -> bool:
        """
        Update trailing stop for a position (pip-based).

        Args:
            symbol: Trading pair
            trailing_pips: Trailing stop distance in pips
            activation_pips: Minimum profit in pips to activate

        Returns:
            True if stop was updated
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        pip_value = self._get_pip_value(symbol)

        if position.direction == 'BUY':
            profit_pips = (position.highest_price - position.entry_price) / pip_value

            if profit_pips >= activation_pips:
                new_stop = position.highest_price - (trailing_pips * pip_value)
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    return True
        else:
            profit_pips = (position.entry_price - position.lowest_price) / pip_value

            if profit_pips >= activation_pips:
                new_stop = position.lowest_price + (trailing_pips * pip_value)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    return True

        return False

    def update_trailing_stop_pct(self, symbol: str, trailing_pct: float) -> bool:
        """
        Update trailing stop for a position (percentage-based for DCA).

        Args:
            symbol: Trading pair
            trailing_pct: Trailing stop as % of price (e.g., 0.014 = 1.4%)

        Returns:
            True if stop was updated
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]

        if position.direction == 'BUY':
            # For long positions, trail below the highest price
            new_stop = position.highest_price * (1 - trailing_pct)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                return True
        else:
            # For short positions, trail above the lowest price
            new_stop = position.lowest_price * (1 + trailing_pct)
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                return True

        return False

    def check_dca_trigger(self, symbol: str, dca_config, pip_value: float = 0.0001) -> Optional[int]:
        """
        Check if a DCA level should be triggered.

        Uses PIP-BASED triggers when use_pip_based_dca is True:
        - DCA 1: 3 pips loss
        - DCA 2: 6 pips loss
        - DCA 3: 10 pips loss
        - DCA 4: 15 pips loss

        Args:
            symbol: Trading pair
            dca_config: DCAConfig object
            pip_value: Pip value for the symbol (0.0001 for most, 0.01 for JPY)

        Returns:
            DCA level to trigger (1-4) or None
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # Already maxed out DCA
        if position.dca_level >= dca_config.max_dca_levels:
            return None

        # Calculate price drop in pips
        if position.direction == 'BUY':
            price_diff = position.avg_entry_price - position.current_price
        else:
            price_diff = position.current_price - position.avg_entry_price

        # Only trigger DCA on price moving against us
        if price_diff <= 0:
            return None

        # Convert to pips
        pips_loss = price_diff / pip_value

        # Check the NEXT DCA level only (sequential triggering)
        next_level = position.dca_level + 1

        if next_level <= dca_config.max_dca_levels:
            # Use pip-based triggers
            if getattr(dca_config, 'use_pip_based_dca', False):
                trigger_pips = dca_config.get_dca_trigger_pips(next_level)
                if pips_loss >= trigger_pips:
                    return next_level
            else:
                # Fallback to percentage-based
                price_drop_pct = abs(position.get_price_change_pct())
                dca_levels = dca_config.get_dca_levels()
                level_config = dca_levels[next_level - 1]
                if price_drop_pct >= level_config.trigger_pct:
                    return next_level

        return None

    def get_dca_units(self, symbol: str, base_units: float, dca_level: int, dca_config) -> float:
        """
        Calculate units for a DCA entry.

        With risk-based position sizing, each DCA level adds units based on
        equal dollar risk per entry. The old $1/pip cap no longer applies
        since we use risk % of equity for sizing.

        Args:
            symbol: Trading pair
            base_units: Original position units
            dca_level: DCA level (1-4)
            dca_config: DCAConfig object

        Returns:
            Units to add for this DCA level
        """
        if dca_level < 1 or dca_level > 4:
            return 0.0

        # Use pip-based multipliers if enabled
        if getattr(dca_config, 'use_pip_based_dca', False):
            multiplier = dca_config.get_dca_multiplier(dca_level)
        else:
            dca_levels = dca_config.get_dca_levels()
            level_config = dca_levels[dca_level - 1]
            multiplier = level_config.multiplier

        dca_units = base_units * multiplier

        # With risk-based sizing, total units are controlled by:
        # 1. Risk % per trade (e.g., 2% of equity)
        # 2. DCA divisor (splits risk across entries)
        # 3. Multipliers per DCA level
        #
        # No hard unit cap needed - risk is managed via equity %
        # Maximum exposure is naturally limited by account equity

        return dca_units

    def apply_dca_sl_tp(self, symbol: str, dca_config, pip_value: float = 0.0001) -> None:
        """
        Apply DCA-specific SL/TP after a DCA entry.

        IMPORTANT: Stop Loss is based on ORIGINAL ENTRY PRICE, not average entry.
        All trades (original + DCAs) share the same SL from the original entry.

        DYNAMIC TP: Take Profit REDUCES as DCA level increases for faster exit:
        - DCA0 (initial): 12 pips
        - DCA1: 10 pips
        - DCA2: 8 pips
        - DCA3: 6 pips (get out faster)
        - DCA4: 5 pips (exit ASAP)

        Args:
            symbol: Trading pair
            dca_config: DCAConfig object
            pip_value: Pip value for the symbol (0.0001 for most, 0.01 for JPY)
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Use pip-based SL/TP if enabled
        if getattr(dca_config, 'use_pip_based_dca', False):
            sl_pips = dca_config.sl_after_dca_pips  # 55 pips from ORIGINAL

            # DYNAMIC TP based on DCA level - use get_tp_for_dca_level if available
            if hasattr(dca_config, 'get_tp_for_dca_level'):
                tp_pips = dca_config.get_tp_for_dca_level(position.dca_level)
            else:
                tp_pips = dca_config.dca_profit_target_pips  # Fallback

            # SL is from ORIGINAL ENTRY PRICE (not average)
            # TP is from AVERAGE ENTRY PRICE (to ensure profit on full position)
            if position.direction == 'BUY':
                position.stop_loss = position.entry_price - (sl_pips * pip_value)
                position.take_profit = position.avg_entry_price + (tp_pips * pip_value)
            else:
                position.stop_loss = position.entry_price + (sl_pips * pip_value)
                position.take_profit = position.avg_entry_price - (tp_pips * pip_value)

            # Reset breakeven lock when DCA level changes (allows re-evaluation)
            position._breakeven_locked = False
        else:
            # Fallback to percentage-based SL/TP
            if position.direction == 'BUY':
                position.stop_loss = position.entry_price * (1 - dca_config.sl_after_dca_pct)
                position.take_profit = position.avg_entry_price * (1 + dca_config.dca_profit_target_pct)
            else:
                position.stop_loss = position.entry_price * (1 + dca_config.sl_after_dca_pct)
                position.take_profit = position.avg_entry_price * (1 - dca_config.dca_profit_target_pct)

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_total_exposure(self) -> float:
        """Get total exposure (sum of position values)."""
        return sum(p.units * p.current_price for p in self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_statistics(self) -> Dict:
        """Get trading statistics."""
        win_rate = self.winning_trades / max(1, self.total_trades)

        # Calculate average win/loss
        wins = [t.pnl for t in self.trade_history if t.pnl > 0]
        losses = [t.pnl for t in self.trade_history if t.pnl <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf')

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'open_positions': len(self.positions),
            'unrealized_pnl': self.get_total_unrealized_pnl()
        }

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        if self.config and hasattr(self.config, 'get_pip_value'):
            return self.config.get_pip_value(symbol)
        return 0.01 if 'JPY' in symbol else 0.0001

    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades
        }

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"PositionManager(open={stats['open_positions']}, "
                f"trades={stats['total_trades']}, "
                f"win_rate={stats['win_rate']:.1%}, "
                f"pnl={stats['total_pnl']:.2f})")
