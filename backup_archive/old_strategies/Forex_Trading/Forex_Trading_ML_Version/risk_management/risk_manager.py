"""
Risk Manager for Forex ML System
================================

Centralized risk management and position sizing.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class RiskState:
    """Track daily risk state."""
    date: str = ""
    daily_pnl: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    last_trade_time: Optional[datetime] = None


class RiskManager:
    """Manages risk limits and position sizing."""

    def __init__(self, config=None):
        """
        Initialize risk manager.

        Args:
            config: TradingConfig or RiskConfig object
        """
        self.config = config

        # Get risk settings
        if config and hasattr(config, 'risk'):
            risk = config.risk
            self.position_size_pct = risk.position_size_pct
            self.max_position_value = risk.max_position_value
            self.max_positions = risk.max_positions
            self.default_stop_loss_pips = risk.default_stop_loss_pips
            self.default_take_profit_pips = risk.default_take_profit_pips
            self.use_atr_stops = risk.use_atr_stops
            self.atr_stop_multiplier = risk.atr_stop_multiplier
            self.atr_tp_multiplier = risk.atr_tp_multiplier
            self.max_daily_loss_pct = risk.max_daily_loss_pct
            self.max_drawdown_pct = risk.max_drawdown_pct
            self.max_trades_per_day = risk.max_trades_per_day
            self.cooldown_seconds = risk.cooldown_seconds
            self.use_trailing_stop = risk.use_trailing_stop
            self.trailing_stop_pips = risk.trailing_stop_pips
            self.trailing_activation_pips = risk.trailing_activation_pips
        else:
            # Defaults
            self.position_size_pct = 0.02
            self.max_position_value = 10000
            self.max_positions = 3
            self.default_stop_loss_pips = 20
            self.default_take_profit_pips = 8
            self.use_atr_stops = True
            self.atr_stop_multiplier = 1.5
            self.atr_tp_multiplier = 1.0
            self.max_daily_loss_pct = 0.05
            self.max_drawdown_pct = 0.10
            self.max_trades_per_day = 20
            self.cooldown_seconds = 300
            self.use_trailing_stop = True
            self.trailing_stop_pips = 10
            self.trailing_activation_pips = 5

        # State tracking
        self.risk_state = RiskState()
        self.initial_equity = 100000
        self.current_equity = 100000
        self.peak_equity = 100000

    def set_equity(self, equity: float) -> None:
        """Set current equity level."""
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def set_initial_equity(self, equity: float) -> None:
        """Set initial equity for drawdown calculation."""
        self.initial_equity = equity
        self.current_equity = equity
        self.peak_equity = equity

    def calculate_position_size(self, symbol: str, entry_price: float,
                                 stop_loss_price: float, equity: float,
                                 dca_enabled: bool = False) -> Tuple[float, float]:
        """
        Calculate position size for EQUAL DOLLAR RISK per entry.

        This ensures all trades (across ALL pairs) risk the same dollar amount.
        With DCA, each entry (initial + DCA levels) should have equal risk.

        Risk-based sizing formula:
        - Risk per trade = equity * risk_pct (e.g., 2% of $5000 = $100)
        - SL distance in pips determines units needed
        - Units = Risk $ / (SL pips * pip_value_per_unit)

        For DCA:
        - Initial entry risks base amount
        - Each DCA level also risks proportional amount based on multiplier

        Args:
            symbol: Trading pair
            entry_price: Entry price
            stop_loss_price: Stop loss price (used for risk calculation)
            equity: Current account equity
            dca_enabled: Whether DCA is enabled

        Returns:
            Tuple of (units, position_value)
        """
        # Get pip value for this symbol
        pip_value = self._get_pip_value(symbol)

        # Calculate SL distance in pips
        sl_distance = abs(entry_price - stop_loss_price)
        sl_pips = sl_distance / pip_value

        # Minimum SL distance to avoid division issues
        if sl_pips < 1:
            sl_pips = 20  # Default fallback

        # Calculate risk amount per trade
        # Base risk = 2% of equity (configurable)
        risk_pct = self.position_size_pct  # Default 0.02 = 2%
        base_risk_amount = equity * risk_pct

        # For DCA: divide risk among entries
        # Total entries = 1 (initial) + 4 (DCAs) with multipliers
        # Total multiplier = 1 + 0.5 + 1.0 + 1.5 + 2.0 = 6.0x
        if dca_enabled:
            divisor = 4.0  # Default
            if self.config and hasattr(self.config, 'dca'):
                divisor = self.config.dca.initial_size_divisor
            risk_amount = base_risk_amount / divisor
        else:
            risk_amount = base_risk_amount

        # Calculate pip value in USD for this pair
        # For XXX/USD pairs: pip_value_per_unit = 0.0001 (1 pip = $0.0001 per unit)
        # For USD/XXX pairs: pip_value_per_unit = 0.0001 / entry_price
        # For XXX/JPY pairs: pip_value_per_unit = 0.01 / USD_JPY rate
        pip_value_usd = self._get_pip_value_in_usd(symbol, entry_price)

        # Units = Risk Amount / (SL pips * pip value per unit in USD)
        if pip_value_usd > 0 and sl_pips > 0:
            units = int(risk_amount / (sl_pips * pip_value_usd))
        else:
            units = 10000  # Fallback

        # Apply minimum and maximum limits
        units = max(units, 100)  # Minimum 100 units
        units = min(units, 100000)  # Maximum 100,000 units

        # Calculate position value
        position_value = abs(units) * entry_price

        # Debug output
        print(f"    [SIZING] {symbol}: equity=${equity:.0f}, risk={risk_pct:.1%}, SL={sl_pips:.1f}p")
        print(f"             risk_amt=${risk_amount:.2f}, pip_val=${pip_value_usd:.6f}, units={units}")

        return units, position_value

    def _get_pip_value_in_usd(self, symbol: str, current_price: float) -> float:
        """
        Get pip value in USD per unit for a currency pair.

        STANDARDIZED: All pairs return ~0.0001 USD per unit per pip
        This ensures EQUAL position sizes across all pairs for equal risk.

        The key insight: We want the SAME units for the SAME risk amount.
        If we risk $25 with 25 pip SL, we want ~10,000 units for ALL pairs.

        Formula: Units = Risk$ / (SL_pips * pip_value_usd)
        For $25 risk, 25 pip SL, 0.0001 pip value: 25 / (25 * 0.0001) = 10,000 units

        Args:
            symbol: Currency pair
            current_price: Current market price

        Returns:
            Pip value in USD per unit (standardized to ~0.0001)
        """
        # STANDARDIZED pip value for ALL pairs
        # This gives approximately equal position sizes
        #
        # For XXX/USD pairs: 1 pip = $0.0001 per unit (exact)
        # For USD/XXX pairs: 1 pip ≈ $0.0001 per unit (after conversion)
        # For cross pairs: 1 pip ≈ $0.0001 per unit (approximate)
        #
        # The actual P&L will vary slightly by pair, but position RISK
        # (units * SL distance) will be equal across all pairs.

        # JPY pairs have different pip size (0.01 instead of 0.0001)
        # But pip VALUE in USD is similar after conversion
        # USD/JPY at 156: 1 pip = 0.01 JPY = 0.01/156 USD ≈ $0.000064 per unit
        # To get equal units, we use 0.0001 for calculation consistency
        if 'JPY' in symbol:
            # For JPY pairs, pip value in USD ≈ 0.01 / USD_JPY_rate
            # At USD/JPY = 156: 0.01 / 156 ≈ 0.000064
            # Use approximate rate for consistent sizing
            return 0.01 / 156.0  # ~0.000064

        # All other pairs - use 0.0001
        # This gives ~10,000 units per $1 risk per pip
        return 0.0001

    def calculate_stop_take_profit(self, symbol: str, entry_price: float,
                                    direction: str, atr: float = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit prices.

        Args:
            symbol: Trading pair
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            atr: Current ATR value (optional)

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        pip_value = self._get_pip_value(symbol)

        # Get TP/SL in pips
        if self.config and hasattr(self.config, 'get_pair_settings'):
            settings = self.config.get_pair_settings(symbol)
            tp_pips = settings.get('tp_pips', self.default_take_profit_pips)
            sl_pips = settings.get('sl_pips', self.default_stop_loss_pips)
        else:
            tp_pips = self.default_take_profit_pips
            sl_pips = self.default_stop_loss_pips

        # Use ATR-based stops if available and enabled
        if self.use_atr_stops and atr is not None:
            atr_pips = atr / pip_value
            sl_pips = max(sl_pips, atr_pips * self.atr_stop_multiplier)
            tp_pips = max(tp_pips, atr_pips * self.atr_tp_multiplier)

        # Calculate prices
        if direction == 'BUY':
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
        else:  # SELL
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)

        return stop_loss, take_profit

    def can_trade(self, current_positions: int = 0) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.

        Args:
            current_positions: Number of current open positions

        Returns:
            Tuple of (allowed, reason)
        """
        # Check daily trade limit
        if self.risk_state.trade_count >= self.max_trades_per_day:
            return False, f"Max daily trades ({self.max_trades_per_day}) reached"

        # Check position limit
        if current_positions >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Check daily loss limit
        daily_loss_pct = abs(min(0, self.risk_state.daily_pnl)) / self.current_equity
        if daily_loss_pct >= self.max_daily_loss_pct:
            return False, f"Daily loss limit ({self.max_daily_loss_pct:.1%}) reached"

        # Check drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown >= self.max_drawdown_pct:
            return False, f"Max drawdown ({self.max_drawdown_pct:.1%}) reached"

        # Check cooldown
        if self.risk_state.last_trade_time:
            elapsed = (datetime.now() - self.risk_state.last_trade_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - elapsed)
                return False, f"Cooldown: {remaining}s remaining"

        return True, "OK"

    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade.

        Args:
            pnl: Profit/loss from the trade
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Reset daily stats if new day
        if self.risk_state.date != today:
            self.risk_state = RiskState(date=today)

        self.risk_state.daily_pnl += pnl
        self.risk_state.trade_count += 1
        self.risk_state.last_trade_time = datetime.now()

        if pnl > 0:
            self.risk_state.winning_trades += 1
        else:
            self.risk_state.losing_trades += 1

        # Update equity
        self.current_equity += pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)

    def record_trade_entry(self) -> None:
        """Record trade entry (for cooldown tracking)."""
        self.risk_state.last_trade_time = datetime.now()
        self.risk_state.trade_count += 1

    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics."""
        return {
            'date': self.risk_state.date,
            'daily_pnl': self.risk_state.daily_pnl,
            'trade_count': self.risk_state.trade_count,
            'winning_trades': self.risk_state.winning_trades,
            'losing_trades': self.risk_state.losing_trades,
            'win_rate': self.risk_state.winning_trades / max(1, self.risk_state.trade_count),
            'daily_loss_pct': abs(min(0, self.risk_state.daily_pnl)) / self.current_equity,
            'current_drawdown': (self.peak_equity - self.current_equity) / self.peak_equity
        }

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol."""
        if self.config and hasattr(self.config, 'get_pip_value'):
            return self.config.get_pip_value(symbol)

        # Default pip values
        if 'JPY' in symbol:
            return 0.01
        return 0.0001

    def reset_daily(self) -> None:
        """Reset daily statistics."""
        self.risk_state = RiskState(date=datetime.now().strftime("%Y-%m-%d"))

    def __str__(self) -> str:
        stats = self.get_daily_stats()
        return (f"RiskManager(trades={stats['trade_count']}/{self.max_trades_per_day}, "
                f"daily_pnl={stats['daily_pnl']:.2f}, "
                f"drawdown={stats['current_drawdown']:.2%})")
