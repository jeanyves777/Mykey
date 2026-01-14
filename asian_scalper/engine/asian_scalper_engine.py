"""
Asian Session Portfolio Scalper Engine
======================================
Trades 15 currency pairs simultaneously during Asian session
Each pair gets its own direction based on 5m EMA (smarter direction)
Closes ALL when portfolio target ($30) is met
DCA when portfolio P/L hits -$15 for recovery
No session-end close - positions run until target or recovery
"""

import logging
import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SCALP_PAIRS,
    PAIR_SETTINGS,
    SCALP_CONFIG,
    SESSION_CONFIG,
    LOGGING_CONFIG,
    get_pip_location,
    get_pair_pip_value,
)
from engine.oanda_client import AsianScalperOANDAClient

# Create directories before setting up logging
os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
os.makedirs(LOGGING_CONFIG["data_dir"], exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGGING_CONFIG["log_dir"], "asian_scalper.log"))
    ]
)
logger = logging.getLogger("AsianScalper")


class AsianScalperEngine:
    """
    Portfolio Scalper Engine

    Strategy:
    1. During Asian session (22:00-08:00 UTC), open all 15 pairs
    2. Each pair gets its OWN direction based on EMA (smarter direction)
    3. Each pair has individual TP (3 pips) and SL (30 pips)
    4. Monitor portfolio unrealized P&L
    5. When portfolio P&L >= $30, close ALL positions (WIN)
    6. When portfolio P&L hits -$15, trigger DCA on all positions
    7. After DCA, exit when recovered to -$15 (50% recovery) or +$30 target
    8. NO session-end close - let positions run
    """

    def __init__(self):
        """Initialize the Asian Scalper Engine."""
        # Create directories
        os.makedirs(LOGGING_CONFIG["log_dir"], exist_ok=True)
        os.makedirs(LOGGING_CONFIG["data_dir"], exist_ok=True)

        # Initialize OANDA client
        self.client = AsianScalperOANDAClient(practice=True)

        # Configuration
        self.pairs = SCALP_PAIRS
        self.portfolio_target = SCALP_CONFIG["portfolio_target"]
        self.tp_pips = SCALP_CONFIG["individual_tp_pips"]
        self.sl_pips = SCALP_CONFIG["individual_sl_pips"]
        self.units_per_pair = SCALP_CONFIG["units_per_pair"]
        self.ema_fast = SCALP_CONFIG["ema_fast"]
        self.ema_slow = SCALP_CONFIG["ema_slow"]
        self.timeframe = SCALP_CONFIG["timeframe"]
        self.force_entry = SCALP_CONFIG["force_entry"]
        self.fallback_direction = SCALP_CONFIG["fallback_direction"]
        self.use_per_pair_direction = SCALP_CONFIG.get("use_per_pair_direction", True)

        # DCA settings
        self.dca_enabled = SCALP_CONFIG.get("dca_enabled", True)
        self.dca_trigger_pl = SCALP_CONFIG.get("dca_trigger_pl", -15.0)
        self.dca_units = SCALP_CONFIG.get("dca_units", 5000)
        self.dca_max_per_pair = SCALP_CONFIG.get("dca_max_per_pair", 1)

        # Recovery exit settings
        self.recovery_exit_enabled = SCALP_CONFIG.get("recovery_exit_enabled", True)
        self.recovery_exit_pl = SCALP_CONFIG.get("recovery_exit_pl", -15.0)

        # State tracking
        self.session_active = False
        self.positions_opened = False
        self.session_start_balance = 0.0
        self.session_trades = []
        self.current_direction = None
        self.dca_triggered = False  # Track if DCA has been done
        self.dca_positions = {}     # Track DCA count per pair: {pair: count}
        self.worst_pl_after_dca = 0.0  # Track worst P/L after DCA for recovery calc

        # Data file paths
        self.trades_file = os.path.join(
            LOGGING_CONFIG["data_dir"],
            LOGGING_CONFIG["trade_log_file"]
        )
        self.sessions_file = os.path.join(
            LOGGING_CONFIG["data_dir"],
            LOGGING_CONFIG["session_log_file"]
        )

        logger.info("=" * 60)
        logger.info("ASIAN SESSION PORTFOLIO SCALPER INITIALIZED")
        logger.info(f"Account: {self.client.account_id}")
        logger.info(f"Pairs: {len(self.pairs)}")
        logger.info(f"Portfolio Target: ${self.portfolio_target}")
        logger.info(f"Individual TP: {self.tp_pips} pips | SL: {self.sl_pips} pips")
        logger.info(f"Units per pair: {self.units_per_pair}")
        logger.info(f"Per-pair direction: {self.use_per_pair_direction}")
        logger.info(f"DCA enabled: {self.dca_enabled} | Trigger: ${self.dca_trigger_pl}")
        logger.info(f"Recovery exit: ${self.recovery_exit_pl}")
        logger.info("=" * 60)

    def is_trading_window(self) -> bool:
        """Check if we're in the trading window (Asian session)."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        start = SESSION_CONFIG["trade_window_start"]  # 22
        end = SESSION_CONFIG["trade_window_end"]      # 6

        # Handle wrap-around midnight (22:00 to 06:00)
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    def is_asian_session(self) -> bool:
        """Check if we're in Asian session."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        start = SESSION_CONFIG["asian_start_hour"]  # 22
        end = SESSION_CONFIG["asian_end_hour"]      # 8

        # Handle wrap-around midnight (22:00 to 08:00)
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate EMA for a DataFrame."""
        return df["close"].ewm(span=period, adjust=False).mean()

    def get_direction_for_pair(self, pair: str) -> Optional[str]:
        """
        Determine trade direction for a pair using 5m EMA crossover.

        Returns:
            "LONG", "SHORT", or None if no clear signal
        """
        try:
            # Get 5m candles
            df = self.client.get_candles(pair, self.timeframe, count=50)

            if df.empty or len(df) < self.ema_slow + 5:
                logger.warning(f"{pair}: Insufficient data for EMA calculation")
                return None

            # Calculate EMAs
            ema_fast = self.calculate_ema(df, self.ema_fast)
            ema_slow = self.calculate_ema(df, self.ema_slow)

            # Get latest values
            fast_current = ema_fast.iloc[-1]
            slow_current = ema_slow.iloc[-1]

            # Determine direction
            if fast_current > slow_current:
                return "LONG"
            elif fast_current < slow_current:
                return "SHORT"
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting direction for {pair}: {e}")
            return None

    def get_majority_direction(self) -> str:
        """
        Get the majority direction across all pairs.

        Returns:
            "LONG" or "SHORT" based on majority of EMA signals
        """
        directions = {"LONG": 0, "SHORT": 0, "NONE": 0}

        logger.info("Analyzing direction for all pairs...")

        for pair in self.pairs:
            direction = self.get_direction_for_pair(pair)

            if direction == "LONG":
                directions["LONG"] += 1
            elif direction == "SHORT":
                directions["SHORT"] += 1
            else:
                directions["NONE"] += 1

            logger.info(f"  {pair}: {direction or 'NONE'}")

        logger.info(f"Direction Summary - LONG: {directions['LONG']}, SHORT: {directions['SHORT']}, NONE: {directions['NONE']}")

        # Majority wins
        if directions["LONG"] > directions["SHORT"]:
            return "LONG"
        elif directions["SHORT"] > directions["LONG"]:
            return "SHORT"
        else:
            # Tie - use fallback
            if self.force_entry:
                logger.info(f"No clear majority, using fallback: {self.fallback_direction}")
                return self.fallback_direction
            return None

    def calculate_tp_sl(self, pair: str, direction: str, entry_price: float) -> Tuple[float, float]:
        """
        Calculate TP and SL prices for a pair.

        Returns:
            (take_profit_price, stop_loss_price)
        """
        pip_value = get_pair_pip_value(pair)
        pip_location = get_pip_location(pair)

        # Calculate pip distance in price terms
        tp_distance = self.tp_pips * pip_value
        sl_distance = self.sl_pips * pip_value

        if direction == "LONG":
            take_profit = entry_price + tp_distance
            stop_loss = entry_price - sl_distance
        else:  # SHORT
            take_profit = entry_price - tp_distance
            stop_loss = entry_price + sl_distance

        # Round based on pip location
        precision = abs(pip_location) + 1
        take_profit = round(take_profit, precision)
        stop_loss = round(stop_loss, precision)

        return take_profit, stop_loss

    def open_all_positions(self, direction: str = None) -> List[Dict]:
        """
        Open positions on all 15 pairs.
        If use_per_pair_direction is True, each pair gets its own direction.
        Otherwise, use the provided direction for all.

        Args:
            direction: "LONG" or "SHORT" (used only if not per-pair direction)

        Returns:
            List of order results
        """
        if self.use_per_pair_direction:
            logger.info(f"Opening ALL {len(self.pairs)} positions - PER-PAIR DIRECTION")
        else:
            logger.info(f"Opening ALL {len(self.pairs)} positions - Direction: {direction}")

        results = []
        successful = 0
        failed = 0
        directions_used = {"LONG": 0, "SHORT": 0}

        # Get all prices at once for faster execution
        prices = self.client.get_multiple_prices(self.pairs)

        for pair in self.pairs:
            try:
                # Determine direction for this pair
                if self.use_per_pair_direction:
                    pair_direction = self.get_direction_for_pair(pair)
                    if not pair_direction:
                        if self.force_entry:
                            pair_direction = self.fallback_direction
                            logger.info(f"  {pair}: No clear direction, using fallback: {pair_direction}")
                        else:
                            logger.info(f"  {pair}: No clear direction, skipping")
                            continue
                else:
                    pair_direction = direction

                # Get current price
                price_data = prices.get(pair)
                if not price_data:
                    price_data = self.client.get_current_price(pair)

                if not price_data:
                    logger.error(f"{pair}: Could not get price")
                    failed += 1
                    continue

                # Determine entry price (use ask for buy, bid for sell)
                if pair_direction == "LONG":
                    entry_price = price_data["ask"]
                    units = self.units_per_pair  # Positive for buy
                    directions_used["LONG"] += 1
                else:
                    entry_price = price_data["bid"]
                    units = -self.units_per_pair  # Negative for sell
                    directions_used["SHORT"] += 1

                # Calculate TP and SL
                take_profit, stop_loss = self.calculate_tp_sl(pair, pair_direction, entry_price)

                # Place order
                result = self.client.place_market_order(
                    instrument=pair,
                    units=units,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                if "error" not in result:
                    successful += 1
                    trade_info = {
                        "pair": pair,
                        "direction": pair_direction,
                        "units": units,
                        "entry_price": entry_price,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "result": result
                    }
                    results.append(trade_info)
                    self.session_trades.append(trade_info)
                    logger.info(f"  {pair}: OPENED {pair_direction} @ {entry_price:.5f} | TP: {take_profit:.5f} | SL: {stop_loss:.5f}")
                else:
                    failed += 1
                    logger.error(f"  {pair}: FAILED - {result.get('error', 'Unknown error')}")

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"{pair}: Error opening position - {e}")
                failed += 1

        logger.info(f"Positions opened: {successful} successful, {failed} failed")
        logger.info(f"Direction mix: {directions_used['LONG']} LONG, {directions_used['SHORT']} SHORT")
        self.positions_opened = True

        return results

    def check_portfolio_target(self) -> bool:
        """
        Check if portfolio unrealized P&L has reached target.

        Returns:
            True if target reached, False otherwise
        """
        unrealized_pl = self.client.get_portfolio_unrealized_pl()

        if unrealized_pl >= self.portfolio_target:
            logger.info(f"PORTFOLIO TARGET REACHED! Unrealized P/L: ${unrealized_pl:.2f}")
            return True

        return False

    def check_dca_trigger(self) -> bool:
        """
        Check if portfolio P/L has hit DCA trigger level.

        Returns:
            True if DCA should be triggered, False otherwise
        """
        if not self.dca_enabled or self.dca_triggered:
            return False

        unrealized_pl = self.client.get_portfolio_unrealized_pl()

        if unrealized_pl <= self.dca_trigger_pl:
            logger.info(f"DCA TRIGGER HIT! Unrealized P/L: ${unrealized_pl:.2f} <= ${self.dca_trigger_pl}")
            return True

        return False

    def execute_dca(self) -> List[Dict]:
        """
        Execute DCA - add to all open positions to lower average entry.

        Returns:
            List of DCA order results
        """
        if self.dca_triggered:
            logger.info("DCA already triggered this session, skipping")
            return []

        logger.info("=" * 60)
        logger.info("EXECUTING DCA - ADDING TO ALL POSITIONS")
        logger.info("=" * 60)

        open_trades = self.client.get_open_trades()
        results = []
        successful = 0
        failed = 0

        # Get current prices
        pairs_in_trade = [t["instrument"] for t in open_trades]
        prices = self.client.get_multiple_prices(pairs_in_trade)

        for trade in open_trades:
            pair = trade["instrument"]

            # Check if we've already done max DCA for this pair
            if self.dca_positions.get(pair, 0) >= self.dca_max_per_pair:
                logger.info(f"  {pair}: Max DCA ({self.dca_max_per_pair}) already reached, skipping")
                continue

            try:
                # Get current direction from existing position
                existing_units = trade["units"]
                direction = "LONG" if existing_units > 0 else "SHORT"

                # Get current price
                price_data = prices.get(pair)
                if not price_data:
                    price_data = self.client.get_current_price(pair)

                if not price_data:
                    logger.error(f"{pair}: Could not get price for DCA")
                    failed += 1
                    continue

                # DCA in same direction
                if direction == "LONG":
                    entry_price = price_data["ask"]
                    units = self.dca_units
                else:
                    entry_price = price_data["bid"]
                    units = -self.dca_units

                # Place DCA order (no new TP/SL - use existing position's)
                result = self.client.place_market_order(
                    instrument=pair,
                    units=units,
                    stop_loss=None,  # Let existing SL/TP handle it
                    take_profit=None
                )

                if "error" not in result:
                    successful += 1
                    self.dca_positions[pair] = self.dca_positions.get(pair, 0) + 1

                    dca_info = {
                        "pair": pair,
                        "direction": direction,
                        "units": units,
                        "entry_price": entry_price,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "type": "DCA",
                        "result": result
                    }
                    results.append(dca_info)
                    logger.info(f"  {pair}: DCA {direction} +{abs(units)} units @ {entry_price:.5f}")
                else:
                    failed += 1
                    logger.error(f"  {pair}: DCA FAILED - {result.get('error', 'Unknown error')}")

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"{pair}: Error executing DCA - {e}")
                failed += 1

        logger.info(f"DCA complete: {successful} successful, {failed} failed")

        # Mark DCA as triggered and record worst P/L point
        self.dca_triggered = True
        self.worst_pl_after_dca = self.client.get_portfolio_unrealized_pl()
        logger.info(f"DCA point P/L: ${self.worst_pl_after_dca:.2f}")

        return results

    def check_recovery_exit(self) -> bool:
        """
        Check if we should exit after DCA based on recovery.
        Exit when recovered to recovery_exit_pl (e.g., -$15, which is 50% recovery from -$30).

        Returns:
            True if recovery exit should happen, False otherwise
        """
        if not self.dca_triggered or not self.recovery_exit_enabled:
            return False

        unrealized_pl = self.client.get_portfolio_unrealized_pl()

        # Exit if recovered to recovery level
        if unrealized_pl >= self.recovery_exit_pl:
            logger.info(f"RECOVERY EXIT! P/L: ${unrealized_pl:.2f} >= ${self.recovery_exit_pl}")
            return True

        return False

    def close_all_positions(self, reason: str = "portfolio_target") -> Dict:
        """
        Close ALL open positions.

        Args:
            reason: Why positions are being closed

        Returns:
            Summary of closed positions
        """
        logger.info(f"CLOSING ALL POSITIONS - Reason: {reason}")

        # Get current state before closing
        unrealized_pl = self.client.get_portfolio_unrealized_pl()
        open_trades = self.client.get_open_trades()

        # Close all trades
        results = self.client.close_all_trades()

        # Log session result
        session_result = {
            "reason": reason,
            "unrealized_pl_at_close": unrealized_pl,
            "trades_closed": len(results),
            "close_time": datetime.now(timezone.utc).isoformat(),
            "results": results
        }

        logger.info(f"Closed {len(results)} positions with P/L: ${unrealized_pl:.2f}")

        self.positions_opened = False

        return session_result

    def check_individual_closures(self) -> List[Dict]:
        """
        Check which individual positions have been closed by TP/SL.

        Returns:
            List of closed trade info
        """
        # This is handled automatically by OANDA's TP/SL orders
        # We just need to track how many positions remain open
        open_trades = self.client.get_open_trades()
        open_count = len(open_trades)

        if open_count < len(self.pairs) and self.positions_opened:
            closed_count = len(self.pairs) - open_count
            logger.info(f"Individual closures detected: {closed_count} positions hit TP/SL")

        return open_trades

    def log_portfolio_status(self):
        """Log current portfolio status."""
        unrealized_pl = self.client.get_portfolio_unrealized_pl()
        open_trades = self.client.get_open_trades()
        balance = self.client.get_balance()

        logger.info(f"Portfolio Status | Balance: ${balance:.2f} | Unrealized P/L: ${unrealized_pl:.2f} | Open Trades: {len(open_trades)} | Target: ${self.portfolio_target}")

        # Log individual positions
        if open_trades:
            logger.info("  Open Positions:")
            for trade in open_trades:
                pl = trade["unrealized_pl"]
                sign = "+" if pl >= 0 else ""
                direction = "LONG" if trade["units"] > 0 else "SHORT"
                logger.info(f"    {trade['instrument']:10} | {direction:5} | {sign}${pl:.2f}")

    def save_session_data(self, session_result: Dict):
        """Save session data to file."""
        try:
            # Load existing sessions
            sessions = []
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)

            # Add this session
            session_data = {
                "session_start": self.session_start_time.isoformat() if hasattr(self, 'session_start_time') else None,
                "session_end": datetime.now(timezone.utc).isoformat(),
                "start_balance": self.session_start_balance,
                "direction": self.current_direction,
                "pairs_traded": len(self.pairs),
                "trades": self.session_trades,
                "result": session_result
            }

            sessions.append(session_data)

            # Save
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2, default=str)

            logger.info(f"Session data saved to {self.sessions_file}")

        except Exception as e:
            logger.error(f"Error saving session data: {e}")

    def run_session(self):
        """
        Run a complete trading session.

        1. Each pair gets its own direction based on EMA
        2. Open all positions
        3. Monitor until portfolio target ($30) OR recovery exit
        4. DCA when P/L hits -$15
        5. NO session-end close - let positions run
        """
        logger.info("=" * 60)
        logger.info("STARTING NEW TRADING SESSION")
        logger.info("=" * 60)

        # Test connection
        if not self.client.test_connection():
            logger.error("Failed to connect to OANDA. Aborting session.")
            return

        # Record session start
        self.session_start_time = datetime.now(timezone.utc)
        self.session_start_balance = self.client.get_balance()
        self.session_trades = []
        self.dca_triggered = False
        self.dca_positions = {}

        logger.info(f"Session Start Balance: ${self.session_start_balance:.2f}")

        # For per-pair direction, we don't need majority direction
        if self.use_per_pair_direction:
            logger.info("Using PER-PAIR direction (each pair analyzed individually)")
            self.current_direction = "PER_PAIR"
        else:
            # Determine trading direction (legacy mode)
            self.current_direction = self.get_majority_direction()

            if not self.current_direction:
                if self.force_entry:
                    self.current_direction = self.fallback_direction
                    logger.info(f"No clear direction - Using fallback: {self.current_direction}")
                else:
                    logger.info("No clear direction and force_entry is False. Skipping session.")
                    return

            logger.info(f"Session Direction: {self.current_direction}")

        # Open all positions (direction per-pair or fixed)
        self.open_all_positions(self.current_direction if self.current_direction != "PER_PAIR" else None)

        # Monitor loop
        check_interval = SESSION_CONFIG["check_interval_seconds"]
        logger.info(f"Monitoring portfolio (checking every {check_interval}s)...")
        logger.info("NOTE: No session-end close - positions will run until target or recovery")

        session_result = None

        while True:
            try:
                # Check portfolio target (+$30)
                if self.check_portfolio_target():
                    session_result = self.close_all_positions("portfolio_target_reached")
                    break

                # Check DCA trigger (-$15)
                if self.check_dca_trigger():
                    self.execute_dca()

                # Check recovery exit (after DCA, when recovered to -$15)
                if self.check_recovery_exit():
                    session_result = self.close_all_positions("recovery_exit")
                    break

                # Check remaining positions
                open_trades = self.check_individual_closures()

                if len(open_trades) == 0:
                    logger.info("All positions closed by individual TP/SL")
                    session_result = {
                        "reason": "all_individual_closures",
                        "close_time": datetime.now(timezone.utc).isoformat()
                    }
                    self.positions_opened = False
                    break

                # Log status
                self.log_portfolio_status()

                # NO SESSION END CLOSE - removed the is_asian_session check

                # Wait before next check
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                if self.positions_opened:
                    session_result = self.close_all_positions("user_interrupted")
                break

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)

        # Save session data
        if session_result:
            self.save_session_data(session_result)

        # Final summary
        end_balance = self.client.get_balance()
        session_pl = end_balance - self.session_start_balance

        logger.info("=" * 60)
        logger.info("SESSION COMPLETE")
        logger.info(f"Start Balance: ${self.session_start_balance:.2f}")
        logger.info(f"End Balance: ${end_balance:.2f}")
        logger.info(f"Session P/L: ${session_pl:.2f}")
        logger.info(f"DCA triggered: {self.dca_triggered}")
        logger.info("=" * 60)

    def run_continuous(self):
        """
        Run continuously, starting new sessions during trading windows.
        Includes DCA and recovery exit logic.
        """
        logger.info("Starting Asian Scalper in continuous mode...")
        logger.info(f"DCA enabled: {self.dca_enabled} | Trigger: ${self.dca_trigger_pl}")
        logger.info(f"Recovery exit: ${self.recovery_exit_pl}")

        # Check for existing positions on startup (in case of restart)
        existing_trades = self.client.get_open_trades()
        if len(existing_trades) > 0:
            logger.info(f"Found {len(existing_trades)} existing positions - resuming monitoring")
            self.positions_opened = True

        while True:
            try:
                # Check if we're in trading window and no positions open
                if self.is_trading_window() and not self.positions_opened:
                    # Check if we already traded this session
                    open_trades = self.client.get_open_trades()

                    if len(open_trades) == 0:
                        logger.info("Trading window active - Starting session")
                        self.run_session()

                        # Wait before looking for next session
                        logger.info("Waiting for next trading window...")
                        time.sleep(3600)  # Wait 1 hour
                    else:
                        # Positions exist, resume monitoring
                        logger.info(f"Found {len(open_trades)} existing positions - resuming monitoring")
                        self.positions_opened = True

                elif self.positions_opened:
                    # Monitor existing positions

                    # Check portfolio target (+$30)
                    if self.check_portfolio_target():
                        self.close_all_positions("portfolio_target_reached")
                        self.positions_opened = False
                        self.dca_triggered = False
                        self.dca_positions = {}
                        continue

                    # Check DCA trigger (-$15)
                    if self.check_dca_trigger():
                        self.execute_dca()

                    # Check recovery exit (after DCA)
                    if self.check_recovery_exit():
                        self.close_all_positions("recovery_exit")
                        self.positions_opened = False
                        self.dca_triggered = False
                        self.dca_positions = {}
                        continue

                    # Check remaining positions
                    open_trades = self.client.get_open_trades()
                    if len(open_trades) == 0:
                        logger.info("All positions closed by individual TP/SL")
                        self.positions_opened = False
                        self.dca_triggered = False
                        self.dca_positions = {}

                    self.log_portfolio_status()
                    time.sleep(SESSION_CONFIG["check_interval_seconds"])

                else:
                    # Outside trading window but still check for open positions
                    open_trades = self.client.get_open_trades()
                    if len(open_trades) > 0:
                        logger.info(f"Found {len(open_trades)} positions outside trading window - monitoring")
                        self.positions_opened = True
                        continue

                    now = datetime.now(timezone.utc)
                    logger.info(f"Outside trading window (current hour: {now.hour} UTC). Waiting...")
                    time.sleep(300)  # Check every 5 minutes

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                if self.positions_opened:
                    response = input("Close all positions before exit? (y/n): ")
                    if response.lower() == 'y':
                        self.close_all_positions("shutdown")
                break

            except Exception as e:
                logger.error(f"Error in continuous loop: {e}")
                time.sleep(60)


def main():
    """Main entry point."""
    engine = AsianScalperEngine()

    # Parse command line args
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run single session
        engine.run_session()
    else:
        # Run continuously
        engine.run_continuous()


if __name__ == "__main__":
    main()
