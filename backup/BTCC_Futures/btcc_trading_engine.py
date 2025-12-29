"""
BTCC Futures Trading Engine
===========================
Main trading engine for BTCC Futures platform.

Features:
- Paper and live trading modes
- Multiple strategy support
- Position management with TP/SL
- Risk management and drawdown protection
- Real-time monitoring
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .btcc_api_client import BTCCAPIClient
from .btcc_config import BTCCConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position"""
    id: int
    symbol: str
    direction: int  # 1=Buy, 2=Sell
    volume: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    leverage: int = 1
    pnl: float = 0.0
    open_time: datetime = field(default_factory=datetime.now)

    def update_pnl(self, current_price: float):
        """Update position P/L based on current price."""
        self.current_price = current_price
        if self.direction == 1:  # Long
            self.pnl = (current_price - self.entry_price) * self.volume * self.leverage
        else:  # Short
            self.pnl = (self.entry_price - current_price) * self.volume * self.leverage


@dataclass
class TradeSignal:
    """Represents a trading signal from strategy"""
    symbol: str
    direction: int  # 1=Buy, 2=Sell
    strength: float  # 0.0 to 1.0
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime = field(default_factory=datetime.now)


class BTCCTradingEngine:
    """Main trading engine for BTCC Futures"""

    def __init__(self, config: BTCCConfig, strategy_evaluator: Callable = None):
        """
        Initialize trading engine.

        Args:
            config: BTCCConfig instance
            strategy_evaluator: Optional function to evaluate strategies
        """
        self.config = config
        self.strategy_evaluator = strategy_evaluator

        # Initialize API client
        self.client = BTCCAPIClient(
            api_key=config.API_KEY,
            secret_key=config.SECRET_KEY,
            user_name=config.USER_NAME,
            password=config.PASSWORD,
            company_id=config.COMPANY_ID
        )

        # Trading state
        self.is_running = False
        self.positions: Dict[int, Position] = {}
        self.pending_signals: List[TradeSignal] = []

        # Paper trading state
        self.paper_balance = config.PAPER_INITIAL_BALANCE
        self.paper_positions: Dict[int, Position] = {}
        self.paper_position_id_counter = 1000

        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.last_reset_date = datetime.now().date()

        # Symbol data cache
        self.symbols_cache: Dict[str, Dict] = {}
        self.prices_cache: Dict[str, float] = {}

        # Threads
        self._main_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Connect to BTCC API and login."""
        if self.config.is_paper_mode():
            logger.info("Running in PAPER TRADING mode")
            self._load_symbols()
            return True

        if not self.config.USER_NAME or not self.config.PASSWORD:
            logger.error("Username and password required for live trading")
            return False

        result = self.client.login()
        if result.get('code') == 0:
            logger.info(f"Connected to BTCC. Account: {self.client.account_no}")
            self._load_symbols()
            return True

        logger.error(f"Failed to connect: {result.get('msg')}")
        return False

    def _load_symbols(self):
        """Load available symbols and cache them."""
        if self.config.is_paper_mode():
            # Use config symbols for paper trading
            self.symbols_cache = self.config.SYMBOLS.copy()
        else:
            result = self.client.get_symbols()
            if result.get('code') == 0:
                for symbol in result.get('symbols', []):
                    name = symbol.get('name', '').split('/')[-1]  # Extract symbol name
                    self.symbols_cache[name] = {
                        'id': symbol.get('id'),
                        'digits': symbol.get('digits', 2),
                        'min_volume': symbol.get('volumes_min', 0.01),
                        'max_volume': symbol.get('volumes_max', 100),
                        'volume_step': symbol.get('volumes_step', 0.01),
                        'contract_size': symbol.get('contract_size', 1),
                        'full_name': symbol.get('name'),
                    }

    def start(self):
        """Start the trading engine."""
        if self.is_running:
            logger.warning("Engine already running")
            return

        if not self.connect():
            logger.error("Failed to connect, cannot start engine")
            return

        self.is_running = True

        # Start main trading loop
        self._main_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._main_thread.start()

        # Start position monitor
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("Trading engine started")

    def stop(self):
        """Stop the trading engine."""
        self.is_running = False

        if self._main_thread:
            self._main_thread.join(timeout=5)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        if not self.config.is_paper_mode():
            self.client.disconnect()

        logger.info("Trading engine stopped")

    def _trading_loop(self):
        """Main trading loop."""
        while self.is_running:
            try:
                # Reset daily stats if new day
                self._check_daily_reset()

                # Check if within trading hours
                if not self._is_trading_time():
                    time.sleep(60)
                    continue

                # Check daily loss limit
                if self._check_daily_loss_limit():
                    logger.warning("Daily loss limit reached, pausing trading")
                    time.sleep(300)  # Wait 5 minutes
                    continue

                # Evaluate strategies for enabled symbols
                for symbol in self.config.get_enabled_symbols():
                    if not self.is_running:
                        break

                    signal = self._evaluate_symbol(symbol)
                    if signal:
                        self._process_signal(signal)

                time.sleep(self.config.STRATEGY_EVAL_INTERVAL)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)

    def _monitor_loop(self):
        """Position monitoring loop."""
        while self.is_running:
            try:
                self._update_positions()
                self._check_stops()

                time.sleep(self.config.POSITION_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)

    def _evaluate_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Evaluate trading strategy for a symbol."""
        if self.strategy_evaluator:
            return self.strategy_evaluator(symbol, self.config)

        # Default: No signal (implement your strategy here)
        return None

    def _process_signal(self, signal: TradeSignal):
        """Process a trading signal."""
        # Check if we can open more positions
        current_positions = len(self.paper_positions if self.config.is_paper_mode()
                                else self.positions)

        if current_positions >= self.config.MAX_POSITIONS:
            logger.debug(f"Max positions reached, ignoring signal for {signal.symbol}")
            return

        # Check if already have position in this symbol
        positions = self.paper_positions if self.config.is_paper_mode() else self.positions
        for pos in positions.values():
            if pos.symbol == signal.symbol:
                logger.debug(f"Already have position in {signal.symbol}")
                return

        # Calculate position size
        volume = self._calculate_position_size(signal)

        if volume <= 0:
            logger.debug(f"Position size too small for {signal.symbol}")
            return

        # Open position
        self._open_position(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=volume,
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """Calculate position size based on risk management."""
        symbol_config = self.config.get_symbol_config(signal.symbol)
        if not symbol_config:
            return 0

        # Get account balance
        if self.config.is_paper_mode():
            balance = self.paper_balance
        else:
            balance = self.client.get_balance()

        # Calculate position value
        position_value = balance * (self.config.POSITION_SIZE_PCT / 100)
        position_value = min(position_value, self.config.MAX_POSITION_VALUE)

        # Calculate volume
        leverage = symbol_config.get('leverage', self.config.DEFAULT_LEVERAGE)
        volume = (position_value * leverage) / signal.entry_price

        # Round to volume step
        volume_step = symbol_config.get('volume_step', 0.01)
        volume = round(volume / volume_step) * volume_step

        # Clamp to min/max
        volume = max(volume, symbol_config.get('min_volume', 0.01))
        volume = min(volume, symbol_config.get('max_volume', 100))

        return volume

    def _open_position(self, symbol: str, direction: int, volume: float,
                       price: float, stop_loss: float = 0, take_profit: float = 0):
        """Open a new position."""
        symbol_config = self.config.get_symbol_config(symbol)
        leverage = symbol_config.get('leverage', self.config.DEFAULT_LEVERAGE) if symbol_config else self.config.DEFAULT_LEVERAGE

        if self.config.is_paper_mode():
            # Paper trading
            position_id = self.paper_position_id_counter
            self.paper_position_id_counter += 1

            position = Position(
                id=position_id,
                symbol=symbol,
                direction=direction,
                volume=volume,
                entry_price=price,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
            )

            self.paper_positions[position_id] = position

            # Deduct commission
            commission = price * volume * (self.config.PAPER_COMMISSION_PCT / 100)
            self.paper_balance -= commission

            logger.info(f"[PAPER] Opened {'LONG' if direction == 1 else 'SHORT'} "
                       f"{symbol} x{volume} @ {price:.2f} (SL: {stop_loss:.2f}, TP: {take_profit:.2f})")

        else:
            # Live trading
            # Find full symbol name from cache
            full_symbol = symbol
            if symbol in self.symbols_cache:
                full_symbol = self.symbols_cache[symbol].get('full_name', symbol)

            result = self.client.open_position(
                symbol=full_symbol,
                direction=direction,
                volume=volume,
                price=price,
                leverage=leverage,
                stop_loss=stop_loss if stop_loss > 0 else None,
                take_profit=take_profit if take_profit > 0 else None,
            )

            if result.get('code') == 0:
                pos_data = result.get('position', {})
                position = Position(
                    id=pos_data.get('id'),
                    symbol=symbol,
                    direction=direction,
                    volume=volume,
                    entry_price=pos_data.get('open_price', price),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage,
                )
                self.positions[position.id] = position

                logger.info(f"[LIVE] Opened {'LONG' if direction == 1 else 'SHORT'} "
                           f"{symbol} x{volume} @ {price:.2f}")
            else:
                logger.error(f"Failed to open position: {result.get('msg')}")

        self.trades_today += 1

    def _close_position(self, position: Position, price: float, reason: str = ""):
        """Close an existing position."""
        position.update_pnl(price)

        if self.config.is_paper_mode():
            # Paper trading
            # Deduct commission
            commission = price * position.volume * (self.config.PAPER_COMMISSION_PCT / 100)
            self.paper_balance += position.pnl - commission

            # Track performance
            self.daily_pnl += position.pnl
            self.total_pnl += position.pnl

            if position.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            del self.paper_positions[position.id]

            logger.info(f"[PAPER] Closed {position.symbol} @ {price:.2f} | "
                       f"P/L: ${position.pnl:.2f} | Reason: {reason}")

        else:
            # Live trading
            full_symbol = position.symbol
            if position.symbol in self.symbols_cache:
                full_symbol = self.symbols_cache[position.symbol].get('full_name', position.symbol)

            result = self.client.close_position(
                position_id=position.id,
                symbol=full_symbol,
                direction=position.direction,
                volume=position.volume,
                price=price,
            )

            if result.get('code') == 0:
                pos_data = result.get('position', {})
                realized_pnl = pos_data.get('exec_profit', position.pnl)

                self.daily_pnl += realized_pnl
                self.total_pnl += realized_pnl

                if realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                del self.positions[position.id]

                logger.info(f"[LIVE] Closed {position.symbol} @ {price:.2f} | "
                           f"P/L: ${realized_pnl:.2f} | Reason: {reason}")
            else:
                logger.error(f"Failed to close position: {result.get('msg')}")

    def _update_positions(self):
        """Update all position prices and P/L."""
        positions = self.paper_positions if self.config.is_paper_mode() else self.positions

        for position in positions.values():
            # Get current price (implement price fetching)
            current_price = self._get_current_price(position.symbol)
            if current_price:
                position.update_pnl(current_price)

    def _check_stops(self):
        """Check stop loss and take profit levels."""
        positions = list(self.paper_positions.values() if self.config.is_paper_mode()
                        else self.positions.values())

        for position in positions:
            if position.current_price <= 0:
                continue

            # Check stop loss
            if position.stop_loss > 0:
                if position.direction == 1 and position.current_price <= position.stop_loss:
                    self._close_position(position, position.current_price, "Stop Loss")
                    continue
                elif position.direction == 2 and position.current_price >= position.stop_loss:
                    self._close_position(position, position.current_price, "Stop Loss")
                    continue

            # Check take profit
            if position.take_profit > 0:
                if position.direction == 1 and position.current_price >= position.take_profit:
                    self._close_position(position, position.current_price, "Take Profit")
                    continue
                elif position.direction == 2 and position.current_price <= position.take_profit:
                    self._close_position(position, position.current_price, "Take Profit")
                    continue

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        # This should be implemented with actual price fetching
        # For now, return cached price or 0
        return self.prices_cache.get(symbol, 0)

    def update_price(self, symbol: str, price: float):
        """Update cached price for symbol (called externally)."""
        self.prices_cache[symbol] = price

    def _is_trading_time(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.utcnow()
        hour = now.hour

        # Crypto is 24/7, but we can restrict if needed
        return self.config.TRADING_START_HOUR <= hour < self.config.TRADING_END_HOUR

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached."""
        if self.config.is_paper_mode():
            balance = self.paper_balance
        else:
            balance = self.client.get_balance()

        max_loss = balance * (self.config.MAX_DAILY_LOSS_PCT / 100)
        return self.daily_pnl < -max_loss

    def _check_daily_reset(self):
        """Reset daily stats at midnight."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_reset_date = today
            logger.info("Daily stats reset")

    # ==================== Public Methods ====================

    def get_status(self) -> Dict:
        """Get current engine status."""
        positions = self.paper_positions if self.config.is_paper_mode() else self.positions

        return {
            'mode': 'paper' if self.config.is_paper_mode() else 'live',
            'running': self.is_running,
            'balance': self.paper_balance if self.config.is_paper_mode() else self.client.get_balance(),
            'open_positions': len(positions),
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'trades_today': self.trades_today,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / max(1, self.winning_trades + self.losing_trades)) * 100,
        }

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if self.config.is_paper_mode():
            return list(self.paper_positions.values())
        return list(self.positions.values())

    def close_all_positions(self, reason: str = "Manual close"):
        """Close all open positions."""
        positions = list(self.paper_positions.values() if self.config.is_paper_mode()
                        else self.positions.values())

        for position in positions:
            price = position.current_price or position.entry_price
            self._close_position(position, price, reason)

        logger.info(f"Closed {len(positions)} positions")

    def manual_trade(self, symbol: str, direction: int, volume: float,
                     price: float, stop_loss: float = 0, take_profit: float = 0):
        """Execute a manual trade."""
        self._open_position(
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
