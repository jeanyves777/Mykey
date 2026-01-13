"""
Asian Session Portfolio Scalper Engine
======================================
Trades 15 currency pairs simultaneously during Asian session
Opens all positions at once based on 5m EMA direction
Closes ALL when portfolio target ($30) is met
Individual pairs close at 3 pip TP or 30 pip SL
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
    1. During Asian session (00:00-08:00 UTC), determine direction using 5m EMA
    2. Open ALL 15 pairs at once in the same direction
    3. Each pair has individual TP (3 pips) and SL (30 pips)
    4. Monitor portfolio unrealized P&L
    5. When portfolio P&L >= $30, close ALL positions
    6. Individual pairs close when they hit their TP/SL
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

        # State tracking
        self.session_active = False
        self.positions_opened = False
        self.session_start_balance = 0.0
        self.session_trades = []
        self.current_direction = None

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

    def open_all_positions(self, direction: str) -> List[Dict]:
        """
        Open positions on all 15 pairs in the given direction.

        Args:
            direction: "LONG" or "SHORT"

        Returns:
            List of order results
        """
        logger.info(f"Opening ALL {len(self.pairs)} positions - Direction: {direction}")

        results = []
        successful = 0
        failed = 0

        # Get all prices at once for faster execution
        prices = self.client.get_multiple_prices(self.pairs)

        for pair in self.pairs:
            try:
                # Get current price
                price_data = prices.get(pair)
                if not price_data:
                    price_data = self.client.get_current_price(pair)

                if not price_data:
                    logger.error(f"{pair}: Could not get price")
                    failed += 1
                    continue

                # Determine entry price (use ask for buy, bid for sell)
                if direction == "LONG":
                    entry_price = price_data["ask"]
                    units = self.units_per_pair  # Positive for buy
                else:
                    entry_price = price_data["bid"]
                    units = -self.units_per_pair  # Negative for sell

                # Calculate TP and SL
                take_profit, stop_loss = self.calculate_tp_sl(pair, direction, entry_price)

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
                        "direction": direction,
                        "units": units,
                        "entry_price": entry_price,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "result": result
                    }
                    results.append(trade_info)
                    self.session_trades.append(trade_info)
                    logger.info(f"  {pair}: OPENED {direction} @ {entry_price:.5f} | TP: {take_profit:.5f} | SL: {stop_loss:.5f}")
                else:
                    failed += 1
                    logger.error(f"  {pair}: FAILED - {result.get('error', 'Unknown error')}")

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"{pair}: Error opening position - {e}")
                failed += 1

        logger.info(f"Positions opened: {successful} successful, {failed} failed")
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

        1. Determine direction
        2. Open all positions
        3. Monitor until portfolio target or session end
        4. Close all remaining positions
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

        logger.info(f"Session Start Balance: ${self.session_start_balance:.2f}")

        # Determine trading direction
        self.current_direction = self.get_majority_direction()

        if not self.current_direction:
            if self.force_entry:
                self.current_direction = self.fallback_direction
                logger.info(f"No clear direction - Using fallback: {self.current_direction}")
            else:
                logger.info("No clear direction and force_entry is False. Skipping session.")
                return

        logger.info(f"Session Direction: {self.current_direction}")

        # Open all positions
        self.open_all_positions(self.current_direction)

        # Monitor loop
        check_interval = SESSION_CONFIG["check_interval_seconds"]
        logger.info(f"Monitoring portfolio (checking every {check_interval}s)...")

        session_result = None

        while True:
            try:
                # Check portfolio target
                if self.check_portfolio_target():
                    session_result = self.close_all_positions("portfolio_target_reached")
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

                # Check if still in trading session
                if not self.is_asian_session():
                    logger.info("Asian session ended - closing remaining positions")
                    session_result = self.close_all_positions("session_ended")
                    break

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
        logger.info("=" * 60)

    def run_continuous(self):
        """
        Run continuously, starting new sessions during trading windows.
        """
        logger.info("Starting Asian Scalper in continuous mode...")

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
                    if self.check_portfolio_target():
                        self.close_all_positions("portfolio_target_reached")

                    # Check remaining positions
                    open_trades = self.client.get_open_trades()
                    if len(open_trades) == 0:
                        self.positions_opened = False

                    self.log_portfolio_status()
                    time.sleep(SESSION_CONFIG["check_interval_seconds"])

                else:
                    # Outside trading window
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
