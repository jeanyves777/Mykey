"""
Match-Trader Trading Engine for FundedNext
==========================================
Integrates our proven scalping strategy with Match-Trader platform.

FundedNext Stellar 2-Step $6k Account Rules:
- Phase 1 Profit Target: 8% ($480)
- Phase 2 Profit Target: 5% ($300)
- Daily Loss Limit: 5% ($300)
- Max Loss Limit: 10% ($600)
- Minimum Trading Days: 5
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import threading

from match_trader_client import MatchTraderClient, Position, AccountBalance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trading signal from strategy"""
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""


@dataclass
class RiskLimits:
    """FundedNext risk management limits"""
    starting_balance: float = 6000.0
    daily_loss_limit_pct: float = 5.0  # 5% = $300
    max_loss_limit_pct: float = 10.0   # 10% = $600
    phase1_target_pct: float = 8.0     # 8% = $480
    phase2_target_pct: float = 5.0     # 5% = $300
    max_position_size: float = 0.5     # Max 0.5 lots per position
    max_open_positions: int = 3        # Max concurrent positions


class MatchTraderEngine:
    """
    Trading engine for Match-Trader/FundedNext

    Features:
    - Risk management aligned with FundedNext rules
    - Daily loss limit protection
    - Max drawdown protection
    - Position sizing based on risk
    - Real-time P&L tracking
    """

    def __init__(self, client: MatchTraderClient, config: Dict = None):
        """
        Initialize trading engine

        Args:
            client: Authenticated MatchTraderClient instance
            config: Optional configuration overrides
        """
        self.client = client
        self.config = config or {}

        # Risk limits (FundedNext Stellar 2-Step $6k)
        self.risk_limits = RiskLimits(
            starting_balance=self.config.get('starting_balance', 6000.0),
            daily_loss_limit_pct=self.config.get('daily_loss_limit_pct', 5.0),
            max_loss_limit_pct=self.config.get('max_loss_limit_pct', 10.0),
            max_position_size=self.config.get('max_position_size', 0.5),
            max_open_positions=self.config.get('max_open_positions', 3)
        )

        # Trading state
        self.is_running = False
        self.is_trading_allowed = True
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0

        # Daily tracking
        self.day_start_balance = 0.0
        self.current_day = datetime.now().date()

        # Symbol settings (pip values for forex)
        self.symbol_settings = {
            'EURUSD': {'pip_value': 0.0001, 'pip_cost': 1.0, 'min_sl_pips': 10, 'default_sl_pips': 15},
            'GBPUSD': {'pip_value': 0.0001, 'pip_cost': 1.0, 'min_sl_pips': 12, 'default_sl_pips': 18},
            'USDJPY': {'pip_value': 0.01, 'pip_cost': 0.67, 'min_sl_pips': 10, 'default_sl_pips': 15},
            'USDCHF': {'pip_value': 0.0001, 'pip_cost': 1.12, 'min_sl_pips': 10, 'default_sl_pips': 15},
            'USDCAD': {'pip_value': 0.0001, 'pip_cost': 0.74, 'min_sl_pips': 10, 'default_sl_pips': 15},
            'AUDUSD': {'pip_value': 0.0001, 'pip_cost': 1.0, 'min_sl_pips': 10, 'default_sl_pips': 15},
            'NZDUSD': {'pip_value': 0.0001, 'pip_cost': 1.0, 'min_sl_pips': 10, 'default_sl_pips': 15},
            'XAUUSD': {'pip_value': 0.01, 'pip_cost': 1.0, 'min_sl_pips': 50, 'default_sl_pips': 100},
        }

        # Strategy parameters
        self.strategy_params = {
            'risk_per_trade_pct': 1.0,  # Risk 1% per trade
            'risk_reward_ratio': 1.5,   # 1:1.5 R:R
            'trailing_stop_trigger_pips': 10,
            'trailing_stop_distance_pips': 8,
            'breakeven_trigger_pips': 8,
        }

    # ==================== Risk Management ====================

    def check_daily_reset(self):
        """Reset daily P&L tracking at start of new day"""
        today = datetime.now().date()
        if today != self.current_day:
            logger.info(f"New trading day: {today}")
            self.current_day = today
            self.daily_pnl = 0.0

            # Get fresh balance for new day
            balance = self.client.get_balance()
            if balance:
                self.day_start_balance = balance.balance
                logger.info(f"Day start balance: ${self.day_start_balance:,.2f}")

    def calculate_daily_loss_limit(self) -> float:
        """Calculate daily loss limit in dollars"""
        return self.day_start_balance * (self.risk_limits.daily_loss_limit_pct / 100)

    def calculate_max_loss_limit(self) -> float:
        """Calculate max loss limit from starting balance"""
        return self.risk_limits.starting_balance * (self.risk_limits.max_loss_limit_pct / 100)

    def check_risk_limits(self, balance: AccountBalance) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check daily loss limit
        daily_loss_limit = self.calculate_daily_loss_limit()
        if self.daily_pnl <= -daily_loss_limit:
            return False, f"Daily loss limit reached: ${abs(self.daily_pnl):.2f} >= ${daily_loss_limit:.2f}"

        # Check max drawdown from starting balance
        total_loss = self.risk_limits.starting_balance - balance.equity
        max_loss_limit = self.calculate_max_loss_limit()
        if total_loss >= max_loss_limit:
            return False, f"Max drawdown reached: ${total_loss:.2f} >= ${max_loss_limit:.2f}"

        # Check equity vs balance (unrealized loss)
        unrealized_loss = balance.balance - balance.equity
        if unrealized_loss > daily_loss_limit * 0.8:
            return False, f"Approaching daily loss limit with unrealized loss: ${unrealized_loss:.2f}"

        return True, "OK"

    def calculate_position_size(self, symbol: str, stop_loss_pips: float,
                                balance: float) -> float:
        """
        Calculate position size based on risk percentage

        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss distance in pips
            balance: Current account balance

        Returns:
            Lot size
        """
        settings = self.symbol_settings.get(symbol, {'pip_cost': 1.0})
        pip_cost = settings['pip_cost']

        # Risk amount in dollars
        risk_amount = balance * (self.strategy_params['risk_per_trade_pct'] / 100)

        # Calculate lots: risk_amount / (stop_loss_pips * pip_cost * 10)
        # Standard lot = $10 per pip for most pairs
        lot_size = risk_amount / (stop_loss_pips * pip_cost * 10)

        # Round to 2 decimals and enforce limits
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, min(lot_size, self.risk_limits.max_position_size))

        return lot_size

    # ==================== Trading Operations ====================

    def execute_signal(self, signal: TradeSignal) -> Optional[str]:
        """
        Execute a trading signal

        Args:
            signal: TradeSignal object

        Returns:
            Order ID if successful, None otherwise
        """
        # Check if trading is allowed
        balance = self.client.get_balance()
        if not balance:
            logger.error("Cannot get balance - skipping signal")
            return None

        is_allowed, reason = self.check_risk_limits(balance)
        if not is_allowed:
            logger.warning(f"Trading not allowed: {reason}")
            self.is_trading_allowed = False
            return None

        # Check max open positions
        open_positions = self.client.get_open_positions()
        if len(open_positions) >= self.risk_limits.max_open_positions:
            logger.warning(f"Max open positions reached: {len(open_positions)}")
            return None

        # Check if already in position for this symbol
        for pos in open_positions:
            if pos.instrument == signal.symbol:
                logger.info(f"Already in position for {signal.symbol}")
                return None

        # Execute the trade
        order_id = self.client.open_position(
            instrument=signal.symbol,
            side=signal.direction,
            volume=signal.volume,
            sl_price=signal.stop_loss,
            tp_price=signal.take_profit
        )

        if order_id:
            self.trade_count += 1
            logger.info(f"Trade executed: {signal.direction} {signal.volume} {signal.symbol}")
            logger.info(f"  Entry: {signal.entry_price}, SL: {signal.stop_loss}, TP: {signal.take_profit}")
            logger.info(f"  Reason: {signal.reason}")

        return order_id

    def manage_positions(self):
        """
        Manage open positions (trailing stops, breakeven, etc.)
        """
        positions = self.client.get_open_positions()

        for pos in positions:
            settings = self.symbol_settings.get(pos.instrument, {})
            pip_value = settings.get('pip_value', 0.0001)

            # Calculate profit in pips
            if pos.side == "BUY":
                profit_pips = (pos.current_price - pos.entry_price) / pip_value
            else:
                profit_pips = (pos.entry_price - pos.current_price) / pip_value

            # Move to breakeven
            breakeven_trigger = self.strategy_params['breakeven_trigger_pips']
            if profit_pips >= breakeven_trigger and pos.sl_price != pos.entry_price:
                # Set SL to entry price (breakeven)
                self.client.edit_position(
                    position_id=pos.position_id,
                    instrument=pos.instrument,
                    sl_price=pos.entry_price
                )
                logger.info(f"Moved {pos.instrument} to breakeven")

            # Trailing stop
            trailing_trigger = self.strategy_params['trailing_stop_trigger_pips']
            trailing_distance = self.strategy_params['trailing_stop_distance_pips']

            if profit_pips >= trailing_trigger:
                if pos.side == "BUY":
                    new_sl = pos.current_price - (trailing_distance * pip_value)
                    if new_sl > pos.sl_price:
                        self.client.edit_position(
                            position_id=pos.position_id,
                            instrument=pos.instrument,
                            sl_price=new_sl
                        )
                        logger.info(f"Trailing stop updated for {pos.instrument}: {new_sl:.5f}")
                else:
                    new_sl = pos.current_price + (trailing_distance * pip_value)
                    if new_sl < pos.sl_price or pos.sl_price == 0:
                        self.client.edit_position(
                            position_id=pos.position_id,
                            instrument=pos.instrument,
                            sl_price=new_sl
                        )
                        logger.info(f"Trailing stop updated for {pos.instrument}: {new_sl:.5f}")

    def update_pnl(self):
        """Update P&L tracking"""
        balance = self.client.get_balance()
        if balance:
            self.total_pnl = balance.equity - self.risk_limits.starting_balance
            self.daily_pnl = balance.equity - self.day_start_balance

    # ==================== Main Trading Loop ====================

    def print_status(self):
        """Print current trading status"""
        balance = self.client.get_balance()
        positions = self.client.get_open_positions()

        print("\n" + "=" * 60)
        print(f"MATCH-TRADER ENGINE STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)

        if balance:
            print(f"Balance: ${balance.balance:,.2f} | Equity: ${balance.equity:,.2f}")
            print(f"Daily P&L: ${self.daily_pnl:+,.2f} | Total P&L: ${self.total_pnl:+,.2f}")
            print(f"Daily Limit: ${self.calculate_daily_loss_limit():,.2f} | "
                  f"Max DD Limit: ${self.calculate_max_loss_limit():,.2f}")

        print(f"\nOpen Positions: {len(positions)}/{self.risk_limits.max_open_positions}")
        for pos in positions:
            print(f"  {pos.side} {pos.volume} {pos.instrument} @ {pos.entry_price:.5f} "
                  f"-> P/L: ${pos.profit:+.2f}")

        print(f"\nTrades: {self.trade_count} | Wins: {self.win_count} | Losses: {self.loss_count}")
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Trading Allowed: {'YES' if self.is_trading_allowed else 'NO'}")
        print("=" * 60)

    def run(self, check_interval: float = 5.0):
        """
        Main trading loop

        Args:
            check_interval: Seconds between checks
        """
        self.is_running = True

        # Initialize day start balance
        balance = self.client.get_balance()
        if balance:
            self.day_start_balance = balance.balance
            logger.info(f"Starting balance: ${self.day_start_balance:,.2f}")

        logger.info("Match-Trader Engine started")
        logger.info(f"Risk limits: Daily {self.risk_limits.daily_loss_limit_pct}% | "
                    f"Max {self.risk_limits.max_loss_limit_pct}%")

        try:
            while self.is_running:
                # Check for new day
                self.check_daily_reset()

                # Update P&L
                self.update_pnl()

                # Manage existing positions
                self.manage_positions()

                # Check risk limits
                balance = self.client.get_balance()
                if balance:
                    is_allowed, reason = self.check_risk_limits(balance)
                    if not is_allowed and self.is_trading_allowed:
                        logger.warning(f"Trading disabled: {reason}")
                        self.is_trading_allowed = False

                # Print status
                self.print_status()

                # Wait for next cycle
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("Stopping engine...")
            self.is_running = False

    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Engine stopped")


# ==================== Configuration ====================

def create_fundednext_config() -> Dict:
    """
    Create configuration for FundedNext Stellar 2-Step $6k

    Returns:
        Configuration dictionary
    """
    return {
        # Account settings
        'starting_balance': 6000.0,

        # FundedNext rules
        'daily_loss_limit_pct': 5.0,   # 5% = $300
        'max_loss_limit_pct': 10.0,    # 10% = $600

        # Conservative position sizing
        'max_position_size': 0.3,      # Max 0.3 lots
        'max_open_positions': 2,       # Max 2 positions

        # Strategy parameters
        'risk_per_trade_pct': 1.0,     # 1% risk per trade
        'risk_reward_ratio': 1.5,      # 1:1.5 R:R
    }


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║      MATCH-TRADER ENGINE FOR FUNDEDNEXT                      ║
    ║      Stellar 2-Step $6,000 Account                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Phase 1 Target: 8% ($480)  |  Phase 2 Target: 5% ($300)     ║
    ║  Daily Loss Limit: 5% ($300)  |  Max Loss: 10% ($600)        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Configuration - UPDATE THESE WITH YOUR ACTUAL CREDENTIALS
    MT_CONFIG = {
        "base_url": "https://your-fundednext-url.match-trader.com",  # Get from FundedNext
        "broker_id": "fundednext",  # Get from FundedNext
        "email": "your_email@example.com",
        "password": "your_password"
    }

    # Initialize client
    client = MatchTraderClient(
        base_url=MT_CONFIG["base_url"],
        broker_id=MT_CONFIG["broker_id"]
    )

    # Login
    print("Connecting to Match-Trader...")
    if client.login(MT_CONFIG["email"], MT_CONFIG["password"]):
        print("Connected successfully!")

        # Create engine with FundedNext config
        engine = MatchTraderEngine(
            client=client,
            config=create_fundednext_config()
        )

        # Start trading loop
        engine.run(check_interval=5.0)
    else:
        print("Failed to connect. Check your credentials.")
