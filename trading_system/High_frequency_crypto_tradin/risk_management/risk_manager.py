"""
Centralized Risk Manager
=========================

The single source of truth for all risk management decisions.
Used by training, backtesting, simulation, and live trading.

Risk Controls:
1. Position Sizing - Kelly Criterion / Fixed Fractional
2. Stop Loss / Take Profit - ATR-based and percentage-based
3. Daily Loss Limits - Maximum drawdown per day
4. Exposure Limits - Maximum position value and count
5. Correlation Risk - Avoid correlated positions
6. Volatility Adjustment - Reduce size in high volatility
7. Drawdown Protection - Scale down after losses
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class RiskConfig:
    """
    Centralized risk configuration for SPOT TRADING (BUY-only with DCA).

    These parameters are used consistently across:
    - Feature engineering (for risk-adjusted targets)
    - Backtesting (for realistic simulation)
    - Paper trading (for validation)
    - Live trading (for real execution)

    Strategy: BUY-only spot trading with DCA (Dollar Cost Averaging)
    - Wide stop loss to allow averaging down
    - Exit in profit after averaging
    """

    # === POSITION SIZING ===
    base_position_size_pct: float = 0.02  # 2% of equity per trade
    max_position_size_pct: float = 0.05   # Maximum 5% in any single position
    max_position_value: float = 10000.0   # Maximum dollar value per position
    min_position_value: float = 100.0     # Minimum dollar value per position

    # === STOP LOSS / TAKE PROFIT (TIGHT for HF day trading) ===
    stop_loss_pct: float = 0.025          # 2.5% final SL (after all DCA exhausted)
    take_profit_pct: float = 0.01         # 1% take profit (quick exit)
    trailing_stop_pct: float = 0.006      # 0.6% trailing stop
    trailing_stop_activation: float = 0.004  # Activate trailing after 0.4% profit
    use_atr_stops: bool = False           # Disable ATR stops for consistent DCA
    atr_stop_multiplier: float = 2.0      # ATR multiplier for stops
    atr_tp_multiplier: float = 3.0        # ATR multiplier for take profit

    # === DCA (Dollar Cost Averaging) Settings - TIGHT for HF Day Trading ===
    # Typical daily BTC moves: 1-3%, so DCA within those bounds for quick averaging
    dca_enabled: bool = True              # Enable DCA averaging down
    dca_trigger_pct: float = 0.005        # 0.5% drop triggers first DCA
    dca_spacing_pct: float = 0.005        # 0.5% between each DCA level
    max_dca_entries: int = 3              # Max 3 DCA adds (4 total entries)
    dca_multiplier: float = 1.5           # Each DCA is 1.5x the previous size
    dca_profit_target_pct: float = 0.004  # 0.4% profit on avg price to exit (quick)

    # === DAILY LIMITS ===
    max_daily_loss_pct: float = 0.05      # 5% max daily loss
    max_daily_trades: int = 50            # Maximum trades per day
    max_daily_profit_pct: float = 0.10    # 10% daily profit target (optional stop)

    # === EXPOSURE LIMITS ===
    max_total_exposure_pct: float = 0.25  # 25% max total exposure (higher for DCA)
    max_positions: int = 3                # Fewer positions due to DCA
    max_correlated_positions: int = 2     # Max positions in correlated assets
    correlation_threshold: float = 0.7    # Correlation threshold for grouping

    # === VOLATILITY ADJUSTMENT ===
    volatility_scaling: bool = True       # Scale position size by volatility
    target_volatility: float = 0.02       # Target 2% daily volatility
    max_volatility_multiplier: float = 2.0  # Max volatility adjustment
    min_volatility_multiplier: float = 0.5  # Min volatility adjustment

    # === DRAWDOWN PROTECTION ===
    drawdown_scaling: bool = True         # Reduce size during drawdown
    drawdown_threshold_1: float = 0.05    # 5% drawdown = reduce to 75%
    drawdown_threshold_2: float = 0.10    # 10% drawdown = reduce to 50%
    drawdown_threshold_3: float = 0.15    # 15% drawdown = reduce to 25%
    drawdown_pause_threshold: float = 0.20  # 20% drawdown = pause trading

    # === TIME CONTROLS ===
    cooldown_seconds: int = 60            # Seconds between NEW trades (not DCA)
    # NOTE: No max_holding_seconds - DCA handles exits (profit target or SL)
    session_start_buffer: int = 300       # 5 min buffer at session start
    session_end_buffer: int = 300         # 5 min buffer at session end

    # === SIGNAL QUALITY ===
    min_confidence: float = 0.60          # Minimum ML confidence (V18 optimal)
    min_model_agreement: int = 5          # 5/5 unanimous (V18 optimal)

    @classmethod
    def from_risk_level(cls, level: RiskLevel) -> 'RiskConfig':
        """Create config from predefined risk level."""
        if level == RiskLevel.CONSERVATIVE:
            return cls(
                base_position_size_pct=0.01,
                max_position_size_pct=0.03,
                stop_loss_pct=0.01,
                take_profit_pct=0.02,
                max_daily_loss_pct=0.03,
                max_positions=3,
                min_confidence=0.70
            )
        elif level == RiskLevel.AGGRESSIVE:
            return cls(
                base_position_size_pct=0.03,
                max_position_size_pct=0.08,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                max_daily_loss_pct=0.08,
                max_positions=8,
                min_confidence=0.55
            )
        else:  # MODERATE (default)
            return cls()


class RiskManager:
    """
    Centralized Risk Manager - Single source of truth for all risk decisions.

    Usage:
        risk_manager = RiskManager(config)

        # Check if trade is allowed
        allowed, reason = risk_manager.can_open_position(symbol, signal, confidence)

        # Get position size
        size = risk_manager.calculate_position_size(symbol, price, equity, volatility)

        # Get stop/take profit levels
        sl, tp = risk_manager.calculate_exit_levels(entry_price, side, atr)

        # Update after trade
        risk_manager.on_trade_opened(symbol, side, price, size)
        risk_manager.on_trade_closed(symbol, pnl)
    """

    def __init__(self, config: RiskConfig = None):
        """Initialize Risk Manager with configuration."""
        self.config = config or RiskConfig()
        self.reset()

    def reset(self):
        """Reset all tracking state."""
        self.initial_equity = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0

        # Daily tracking
        self.daily_start_equity = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.current_date = None

        # Position tracking
        self.open_positions: Dict[str, dict] = {}
        self.position_count = 0
        self.total_exposure = 0.0

        # Trade history
        self.trade_history: List[dict] = []
        self.last_trade_time: Optional[datetime] = None

        # Drawdown tracking
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Performance metrics
        self.total_trades = 0
        self.total_wins = 0
        self.total_pnl = 0.0

        # Trading state
        self.is_trading_paused = False
        self.pause_reason = ""

    def initialize(self, equity: float):
        """Initialize with starting equity."""
        self.initial_equity = equity
        self.current_equity = equity
        self.peak_equity = equity
        self.daily_start_equity = equity
        self.current_date = datetime.now().date()

    def update_equity(self, equity: float):
        """Update current equity and recalculate metrics."""
        self.current_equity = equity

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Check drawdown pause
        if self.current_drawdown >= self.config.drawdown_pause_threshold:
            self.is_trading_paused = True
            self.pause_reason = f"Drawdown limit reached: {self.current_drawdown*100:.1f}%"

    def check_new_day(self):
        """Check if it's a new trading day and reset daily metrics."""
        today = datetime.now().date()
        if self.current_date != today:
            self.current_date = today
            self.daily_start_equity = self.current_equity
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0

            # Reset pause if it was due to daily limits
            if "daily" in self.pause_reason.lower():
                self.is_trading_paused = False
                self.pause_reason = ""

    # ==========================================
    # TRADE VALIDATION
    # ==========================================

    def can_open_position(self,
                          symbol: str,
                          signal: int,
                          confidence: float,
                          current_time: datetime = None) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading symbol
            signal: 1 (buy) or -1 (sell)
            confidence: ML model confidence (0-1)
            current_time: Current timestamp

        Returns:
            Tuple of (allowed, reason)
        """
        current_time = current_time or datetime.now()
        self.check_new_day()

        # Check if trading is paused
        if self.is_trading_paused:
            return False, f"Trading paused: {self.pause_reason}"

        # Check signal
        if signal == 0:
            return False, "No signal (hold)"

        # Check confidence threshold
        if confidence < self.config.min_confidence:
            return False, f"Confidence {confidence:.2f} below threshold {self.config.min_confidence}"

        # Check if already in position for this symbol
        if symbol in self.open_positions:
            return False, f"Already in position for {symbol}"

        # Check max positions
        if self.position_count >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"

        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check daily loss limit
        if self.current_equity > 0:
            daily_loss_pct = -self.daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 else 0
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self.is_trading_paused = True
                self.pause_reason = f"Daily loss limit: {daily_loss_pct*100:.1f}%"
                return False, self.pause_reason

        # Check cooldown
        if self.last_trade_time:
            seconds_since_last = (current_time - self.last_trade_time).total_seconds()
            if seconds_since_last < self.config.cooldown_seconds:
                return False, f"Cooldown: {self.config.cooldown_seconds - seconds_since_last:.0f}s remaining"

        # Check total exposure
        if self.total_exposure >= self.current_equity * self.config.max_total_exposure_pct:
            return False, f"Max exposure reached ({self.config.max_total_exposure_pct*100:.0f}%)"

        return True, "OK"

    # ==========================================
    # POSITION SIZING
    # ==========================================

    def calculate_position_size(self,
                                symbol: str,
                                price: float,
                                equity: float = None,
                                volatility: float = None,
                                confidence: float = 1.0) -> float:
        """
        Calculate position size with all risk adjustments.

        Args:
            symbol: Trading symbol
            price: Current price
            equity: Account equity (uses current if None)
            volatility: Current volatility (for vol scaling)
            confidence: ML confidence for size adjustment

        Returns:
            Position size in units
        """
        equity = equity or self.current_equity
        if equity <= 0:
            return 0.0

        # Base position size (percentage of equity)
        base_value = equity * self.config.base_position_size_pct

        # 1. Confidence adjustment (0.5 to 1.5x based on confidence)
        confidence_multiplier = 0.5 + confidence
        base_value *= confidence_multiplier

        # 2. Volatility adjustment
        if self.config.volatility_scaling and volatility and volatility > 0:
            vol_ratio = self.config.target_volatility / volatility
            vol_multiplier = np.clip(
                vol_ratio,
                self.config.min_volatility_multiplier,
                self.config.max_volatility_multiplier
            )
            base_value *= vol_multiplier

        # 3. Drawdown adjustment
        if self.config.drawdown_scaling and self.current_drawdown > 0:
            if self.current_drawdown >= self.config.drawdown_threshold_3:
                base_value *= 0.25
            elif self.current_drawdown >= self.config.drawdown_threshold_2:
                base_value *= 0.50
            elif self.current_drawdown >= self.config.drawdown_threshold_1:
                base_value *= 0.75

        # 4. Apply limits
        max_value = min(
            equity * self.config.max_position_size_pct,
            self.config.max_position_value
        )
        position_value = max(
            min(base_value, max_value),
            self.config.min_position_value
        )

        # 5. Check remaining exposure allowance
        remaining_exposure = (equity * self.config.max_total_exposure_pct) - self.total_exposure
        position_value = min(position_value, remaining_exposure)

        # Convert to units
        if price > 0:
            units = position_value / price
            return round(units, 8)  # Crypto precision

        return 0.0

    # ==========================================
    # EXIT LEVELS
    # ==========================================

    def calculate_exit_levels(self,
                              entry_price: float,
                              side: int,
                              atr: float = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price
            side: 1 (long) or -1 (short)
            atr: Average True Range for ATR-based stops

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Calculate stop distance
        if self.config.use_atr_stops and atr and atr > 0:
            stop_distance = atr * self.config.atr_stop_multiplier
            tp_distance = atr * self.config.atr_tp_multiplier
        else:
            stop_distance = entry_price * self.config.stop_loss_pct
            tp_distance = entry_price * self.config.take_profit_pct

        # Apply to price based on side
        if side == 1:  # Long
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit

    def calculate_trailing_stop(self,
                                entry_price: float,
                                current_price: float,
                                highest_price: float,
                                lowest_price: float,
                                side: int) -> Optional[float]:
        """
        Calculate trailing stop price.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry
            side: 1 (long) or -1 (short)

        Returns:
            Trailing stop price or None if not activated
        """
        if side == 1:  # Long
            profit_pct = (highest_price - entry_price) / entry_price
            if profit_pct >= self.config.trailing_stop_activation:
                return highest_price * (1 - self.config.trailing_stop_pct)
        else:  # Short
            profit_pct = (entry_price - lowest_price) / entry_price
            if profit_pct >= self.config.trailing_stop_activation:
                return lowest_price * (1 + self.config.trailing_stop_pct)

        return None

    # ==========================================
    # POSITION TRACKING
    # ==========================================

    def on_trade_opened(self,
                        symbol: str,
                        side: int,
                        price: float,
                        size: float,
                        stop_loss: float = None,
                        take_profit: float = None):
        """Record a new position opening (BUY-only for spot trading)."""
        position_value = price * size

        self.open_positions[symbol] = {
            'symbol': symbol,
            'side': side,  # Should always be 1 (LONG) for spot
            'entry_price': price,  # Average entry price
            'size': size,
            'value': position_value,
            'total_cost': position_value,  # Total cost basis
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'highest_price': price,
            'lowest_price': price,
            'unrealized_pnl': 0.0,
            # DCA tracking
            'dca_count': 0,
            'dca_levels': [price],
            'original_entry': price
        }

        self.position_count = len(self.open_positions)
        self.total_exposure += position_value
        self.daily_trades += 1
        self.total_trades += 1
        self.last_trade_time = datetime.now()

        logger.info(f"Position opened: {symbol} LONG "
                   f"{size:.6f} @ ${price:.2f}")

    def on_trade_closed(self, symbol: str, pnl: float, exit_price: float = None):
        """Record a position closing."""
        if symbol in self.open_positions:
            position = self.open_positions.pop(symbol)

            # Update metrics
            self.position_count = len(self.open_positions)
            self.total_exposure -= position['value']
            self.daily_pnl += pnl
            self.total_pnl += pnl

            if pnl > 0:
                self.daily_wins += 1
                self.total_wins += 1
            else:
                self.daily_losses += 1

            # Record in history
            self.trade_history.append({
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price or position['entry_price'],
                'size': position['size'],
                'pnl': pnl,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'holding_time': (datetime.now() - position['entry_time']).total_seconds()
            })

            # Update equity
            self.current_equity += pnl
            self.update_equity(self.current_equity)

            logger.info(f"Position closed: {symbol} P&L: ${pnl:.2f}")

    def update_position(self, symbol: str, current_price: float):
        """Update position with current price."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]
        pos['highest_price'] = max(pos['highest_price'], current_price)
        pos['lowest_price'] = min(pos['lowest_price'], current_price)

        # Calculate unrealized P&L
        if pos['side'] == 1:  # Long
            pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['size']
        else:  # Short
            pos['unrealized_pnl'] = (pos['entry_price'] - current_price) * pos['size']

    def check_position_exits(self, symbol: str, current_price: float, high: float, low: float) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Returns:
            Tuple of (should_exit, reason)
        """
        if symbol not in self.open_positions:
            return False, ""

        pos = self.open_positions[symbol]
        side = pos['side']
        entry_price = pos['entry_price']
        entry_time = pos['entry_time']

        # Update price tracking
        pos['highest_price'] = max(pos['highest_price'], high)
        pos['lowest_price'] = min(pos['lowest_price'], low)

        # Check stop loss
        if pos['stop_loss']:
            if side == 1 and low <= pos['stop_loss']:
                return True, "Stop loss triggered"
            if side == -1 and high >= pos['stop_loss']:
                return True, "Stop loss triggered"

        # Check take profit
        if pos['take_profit']:
            if side == 1 and high >= pos['take_profit']:
                return True, "Take profit triggered"
            if side == -1 and low <= pos['take_profit']:
                return True, "Take profit triggered"

        # Check trailing stop
        trailing_stop = self.calculate_trailing_stop(
            entry_price, current_price,
            pos['highest_price'], pos['lowest_price'],
            side
        )
        if trailing_stop:
            if side == 1 and low <= trailing_stop:
                return True, "Trailing stop triggered"
            if side == -1 and high >= trailing_stop:
                return True, "Trailing stop triggered"

        # NOTE: No max holding time - DCA handles exits (profit target or SL)

        return False, ""

    # ==========================================
    # DCA (Dollar Cost Averaging) MANAGEMENT
    # ==========================================

    def check_dca_opportunity(self, symbol: str, current_price: float) -> Tuple[bool, float]:
        """
        Check if we should add to position via DCA.

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Tuple of (should_dca, dca_size)
        """
        if not self.config.dca_enabled:
            return False, 0.0

        if symbol not in self.open_positions:
            return False, 0.0

        pos = self.open_positions[symbol]

        # Only DCA for LONG positions (spot trading)
        if pos['side'] != 1:
            return False, 0.0

        # Check if max DCA reached
        if pos['dca_count'] >= self.config.max_dca_entries:
            return False, 0.0

        # Calculate next DCA trigger price
        original_entry = pos['original_entry']

        if pos['dca_count'] == 0:
            # First DCA trigger
            dca_trigger_price = original_entry * (1 - self.config.dca_trigger_pct)
        else:
            # Subsequent DCA triggers
            last_dca_price = pos['dca_levels'][-1]
            dca_trigger_price = last_dca_price * (1 - self.config.dca_spacing_pct)

        # Check if price has dropped to trigger
        if current_price > dca_trigger_price:
            return False, 0.0

        # Calculate DCA size
        base_size = self.current_equity * self.config.base_position_size_pct
        dca_size = base_size * (self.config.dca_multiplier ** (pos['dca_count'] + 1))
        dca_size = min(dca_size, self.config.max_position_value)

        # Check exposure limits
        new_exposure = self.total_exposure + dca_size
        if new_exposure > self.current_equity * self.config.max_total_exposure_pct:
            return False, 0.0

        return True, dca_size

    def execute_dca(self,
                    symbol: str,
                    price: float,
                    dca_value: float) -> bool:
        """
        Execute a DCA (Dollar Cost Averaging) buy.

        Args:
            symbol: Trading symbol
            price: DCA entry price
            dca_value: Value to add to position

        Returns:
            True if DCA executed successfully
        """
        if symbol not in self.open_positions:
            return False

        pos = self.open_positions[symbol]

        # Calculate new quantity
        dca_quantity = dca_value / price

        # Update position
        new_total_quantity = pos['size'] + dca_quantity
        new_total_cost = pos['total_cost'] + dca_value
        new_avg_price = new_total_cost / new_total_quantity

        pos['size'] = new_total_quantity
        pos['entry_price'] = new_avg_price
        pos['value'] = new_total_cost
        pos['total_cost'] = new_total_cost
        pos['dca_count'] += 1
        pos['dca_levels'].append(price)

        # Update take profit based on new average price
        pos['take_profit'] = new_avg_price * (1 + self.config.dca_profit_target_pct)

        # Update exposure
        self.total_exposure += dca_value

        logger.info(f"DCA executed: {symbol} #{pos['dca_count']} "
                   f"+{dca_quantity:.6f} @ ${price:.2f} "
                   f"(new avg: ${new_avg_price:.2f})")

        return True

    def get_dca_status(self, symbol: str) -> Dict:
        """Get DCA status for a position."""
        if symbol not in self.open_positions:
            return {}

        pos = self.open_positions[symbol]

        return {
            'dca_count': pos.get('dca_count', 0),
            'max_dca': self.config.max_dca_entries,
            'dca_remaining': self.config.max_dca_entries - pos.get('dca_count', 0),
            'original_entry': pos.get('original_entry', pos['entry_price']),
            'current_avg': pos['entry_price'],
            'dca_levels': pos.get('dca_levels', []),
            'total_cost': pos.get('total_cost', pos['value'])
        }

    # ==========================================
    # METRICS & REPORTING
    # ==========================================

    def get_daily_metrics(self) -> Dict:
        """Get current daily performance metrics."""
        win_rate = self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0

        return {
            'date': self.current_date,
            'trades': self.daily_trades,
            'wins': self.daily_wins,
            'losses': self.daily_losses,
            'win_rate': win_rate,
            'pnl': self.daily_pnl,
            'pnl_pct': self.daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 else 0,
            'positions': self.position_count,
            'exposure': self.total_exposure,
            'exposure_pct': self.total_exposure / self.current_equity if self.current_equity > 0 else 0
        }

    def get_overall_metrics(self) -> Dict:
        """Get overall performance metrics."""
        win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        total_return = (self.current_equity - self.initial_equity) / self.initial_equity if self.initial_equity > 0 else 0

        return {
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'total_pnl': self.total_pnl,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'win_rate': win_rate,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'is_trading_paused': self.is_trading_paused,
            'pause_reason': self.pause_reason
        }

    def get_risk_status(self) -> Dict:
        """Get current risk status."""
        return {
            'can_trade': not self.is_trading_paused,
            'pause_reason': self.pause_reason,
            'positions': self.position_count,
            'max_positions': self.config.max_positions,
            'exposure_pct': self.total_exposure / self.current_equity if self.current_equity > 0 else 0,
            'max_exposure_pct': self.config.max_total_exposure_pct,
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.config.max_daily_trades,
            'daily_loss_pct': -self.daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 and self.daily_pnl < 0 else 0,
            'max_daily_loss_pct': self.config.max_daily_loss_pct,
            'drawdown_pct': self.current_drawdown,
            'drawdown_pause_pct': self.config.drawdown_pause_threshold
        }


# Singleton instance for global access
_default_risk_manager: Optional[RiskManager] = None


def get_risk_manager(config: RiskConfig = None) -> RiskManager:
    """Get or create the default risk manager instance."""
    global _default_risk_manager
    if _default_risk_manager is None:
        _default_risk_manager = RiskManager(config)
    return _default_risk_manager


def reset_risk_manager():
    """Reset the default risk manager."""
    global _default_risk_manager
    if _default_risk_manager:
        _default_risk_manager.reset()
