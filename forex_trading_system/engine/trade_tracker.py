"""
Trade Tracker - Comprehensive Trade and Analysis Logging
=========================================================
Tracks all trades, signals, analysis, and market data for deep analysis.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class TradeTracker:
    """
    Comprehensive trade and analysis tracker.

    Tracks:
    - All signal analysis (even non-triggering)
    - Trade entries and exits
    - Market conditions at entry/exit
    - Performance metrics
    - OANDA sync for closed trades
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize trade tracker.

        Args:
            data_dir: Directory to store tracking data
        """
        # Use absolute path based on script location
        if not os.path.isabs(data_dir):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(script_dir, data_dir)
        else:
            self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Trade Tracker data directory: {self.data_dir}")

        # File paths
        self.trades_file = os.path.join(data_dir, "trades_history.json")
        self.signals_file = os.path.join(data_dir, "signals_log.json")
        self.analysis_file = os.path.join(data_dir, "analysis_log.json")
        self.market_data_file = os.path.join(data_dir, "market_snapshots.json")
        self.sync_state_file = os.path.join(data_dir, "sync_state.json")

        # Load existing data
        self.trades = self._load_json(self.trades_file, {"active": {}, "closed": []})
        self.signals = self._load_json(self.signals_file, [])
        self.analysis_log = self._load_json(self.analysis_file, [])
        self.market_snapshots = self._load_json(self.market_data_file, [])
        self.sync_state = self._load_json(self.sync_state_file, {"last_trade_id": None, "last_sync": None})

        # In-memory tracking
        self.known_trade_ids = set()  # Track OANDA trade IDs we know about
        self._load_known_trades()

        logger.info(f"Trade Tracker initialized | Active: {len(self.trades['active'])} | Closed: {len(self.trades['closed'])}")

    def _load_json(self, filepath: str, default: Any) -> Any:
        """Load JSON file or return default."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
        return default

    def _convert_numpy(self, obj):
        """Recursively convert numpy types to Python types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _save_json(self, filepath: str, data: Any):
        """Save data to JSON file."""
        try:
            # Convert numpy types before saving
            clean_data = self._convert_numpy(data)
            with open(filepath, 'w') as f:
                json.dump(clean_data, f, indent=2, default=str)
            logger.debug(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_known_trades(self):
        """Load known trade IDs from history."""
        for trade in self.trades.get("closed", []):
            if "oanda_trade_id" in trade:
                self.known_trade_ids.add(trade["oanda_trade_id"])
        for symbol, trade in self.trades.get("active", {}).items():
            if "oanda_trade_id" in trade:
                self.known_trade_ids.add(trade["oanda_trade_id"])

    def log_signal_analysis(
        self,
        symbol: str,
        signal_data: Dict,
        indicators: Dict,
        market_state: Dict
    ):
        """
        Log detailed signal analysis (every check, not just triggers).

        Args:
            symbol: Trading pair
            signal_data: Signal information
            indicators: All indicator values
            market_state: Current market conditions
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": signal_data.get("action"),
            "triggered": signal_data.get("action") is not None,
            "confluence_score": signal_data.get("confluence_score", 0),
            "confidence": signal_data.get("confidence", 0),
            "reason": signal_data.get("reason", ""),
            "indicators": indicators,
            "market_state": market_state,
            "entry_price": signal_data.get("entry_price"),
            "tp_pips": signal_data.get("tp_pips"),
            "sl_pips": signal_data.get("sl_pips")
        }

        self.analysis_log.append(entry)

        # Keep only last 10000 entries
        if len(self.analysis_log) > 10000:
            self.analysis_log = self.analysis_log[-10000:]

        self._save_json(self.analysis_file, self.analysis_log)

        # Also log triggered signals separately
        if entry["triggered"]:
            self.signals.append(entry)
            if len(self.signals) > 1000:
                self.signals = self.signals[-1000:]
            self._save_json(self.signals_file, self.signals)

    def record_trade_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        units: int,
        tp_price: float,
        sl_price: float,
        tp_pips: float,
        sl_pips: float,
        confluence_score: int,
        signal_reason: str,
        indicators: Dict,
        oanda_trade_id: str = None
    ):
        """
        Record a new trade entry.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            entry_price: Entry price
            units: Position size
            tp_price: Take profit price
            sl_price: Stop loss price
            tp_pips: TP distance in pips
            sl_pips: SL distance in pips
            confluence_score: Signal strength
            signal_reason: Why the trade was taken
            indicators: Indicator values at entry
            oanda_trade_id: OANDA trade ID
        """
        trade = {
            "symbol": symbol,
            "side": side,
            "entry_time": datetime.now().isoformat(),
            "entry_price": entry_price,
            "units": units,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "confluence_score": confluence_score,
            "signal_reason": signal_reason,
            "indicators_at_entry": indicators,
            "oanda_trade_id": oanda_trade_id,
            "status": "OPEN",
            "peak_pips": 0,
            "min_pips": 0,
            "updates": []
        }

        self.trades["active"][symbol] = trade
        if oanda_trade_id:
            self.known_trade_ids.add(oanda_trade_id)

        self._save_json(self.trades_file, self.trades)

        logger.info(f"[TRACKER] Trade recorded: {symbol} {side} @ {entry_price:.5f}")

    def update_trade_status(
        self,
        symbol: str,
        current_price: float,
        current_pips: float,
        unrealized_pl: float
    ):
        """
        Update active trade status.

        Args:
            symbol: Trading pair
            current_price: Current market price
            current_pips: Current P&L in pips
            unrealized_pl: Unrealized P&L in USD
        """
        if symbol not in self.trades["active"]:
            return

        trade = self.trades["active"][symbol]

        # Update peak/min tracking
        trade["peak_pips"] = max(trade.get("peak_pips", 0), current_pips)
        trade["min_pips"] = min(trade.get("min_pips", 0), current_pips)

        # Add update entry
        update = {
            "time": datetime.now().isoformat(),
            "price": current_price,
            "pips": current_pips,
            "unrealized_pl": unrealized_pl
        }

        # Keep last 100 updates per trade
        if "updates" not in trade:
            trade["updates"] = []
        trade["updates"].append(update)
        if len(trade["updates"]) > 100:
            trade["updates"] = trade["updates"][-100:]

        self._save_json(self.trades_file, self.trades)

    def record_trade_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        realized_pips: float,
        realized_pl: float,
        indicators_at_exit: Dict = None
    ):
        """
        Record a trade exit.

        Args:
            symbol: Trading pair
            exit_price: Exit price
            exit_reason: Why the trade was closed
            realized_pips: Final P&L in pips
            realized_pl: Final P&L in USD
            indicators_at_exit: Indicator values at exit
        """
        if symbol not in self.trades["active"]:
            logger.warning(f"[TRACKER] No active trade for {symbol} to close")
            return

        trade = self.trades["active"][symbol]

        # Update with exit info
        trade["exit_time"] = datetime.now().isoformat()
        trade["exit_price"] = exit_price
        trade["exit_reason"] = exit_reason
        trade["realized_pips"] = realized_pips
        trade["realized_pl"] = realized_pl
        trade["indicators_at_exit"] = indicators_at_exit or {}
        trade["status"] = "CLOSED"

        # Calculate duration
        entry_time = datetime.fromisoformat(trade["entry_time"])
        duration = datetime.now() - entry_time
        trade["duration_seconds"] = duration.total_seconds()
        trade["duration_str"] = str(duration).split('.')[0]

        # Determine outcome
        trade["outcome"] = "WIN" if realized_pips > 0 else "LOSS" if realized_pips < 0 else "BREAKEVEN"

        # Move to closed trades
        self.trades["closed"].append(trade)
        del self.trades["active"][symbol]

        self._save_json(self.trades_file, self.trades)

        logger.info(f"[TRACKER] Trade closed: {symbol} | {trade['outcome']} | {realized_pips:+.1f} pips (${realized_pl:+.2f})")

    def sync_with_oanda(self, oanda_client, tracked_symbols: List[str]) -> List[Dict]:
        """
        Sync with OANDA to detect trades closed by SL/TP.

        Args:
            oanda_client: OANDA API client
            tracked_symbols: List of symbols we're tracking

        Returns:
            List of newly detected closed trades
        """
        closed_trades = []

        try:
            # Get current open trades from OANDA
            oanda_open_trades = oanda_client.get_open_trades()
            oanda_open_ids = {t["id"] for t in oanda_open_trades}

            # Check our active trades
            symbols_to_remove = []

            for symbol, trade in self.trades["active"].items():
                oanda_id = trade.get("oanda_trade_id")

                if oanda_id and oanda_id not in oanda_open_ids:
                    # Trade was closed by OANDA (SL/TP hit)
                    logger.info(f"[TRACKER] Detected closed trade: {symbol} (OANDA ID: {oanda_id})")

                    # Get trade details from OANDA
                    trade_details = oanda_client.get_trade_details(oanda_id)

                    if trade_details:
                        exit_price = trade_details.get("average_close_price", trade["entry_price"])
                        realized_pl = trade_details.get("realized_pl", 0)

                        # Calculate pips
                        pip_multiplier = 100 if "JPY" in symbol else 10000
                        if trade["side"] == "BUY":
                            realized_pips = (exit_price - trade["entry_price"]) * pip_multiplier
                        else:
                            realized_pips = (trade["entry_price"] - exit_price) * pip_multiplier

                        # Determine exit reason
                        if realized_pips >= trade.get("tp_pips", 0) * 0.9:
                            exit_reason = "TAKE_PROFIT"
                        elif realized_pips <= -trade.get("sl_pips", 0) * 0.9:
                            exit_reason = "STOP_LOSS"
                        else:
                            exit_reason = "CLOSED_BY_OANDA"

                        # Record the exit
                        trade["exit_time"] = trade_details.get("close_time", datetime.now().isoformat())
                        trade["exit_price"] = exit_price
                        trade["exit_reason"] = exit_reason
                        trade["realized_pips"] = realized_pips
                        trade["realized_pl"] = realized_pl
                        trade["status"] = "CLOSED"
                        trade["outcome"] = "WIN" if realized_pips > 0 else "LOSS" if realized_pips < 0 else "BREAKEVEN"

                        # Calculate duration
                        try:
                            entry_time = datetime.fromisoformat(trade["entry_time"].replace('Z', '+00:00'))
                            exit_time = datetime.fromisoformat(trade["exit_time"].replace('Z', '+00:00'))
                            duration = exit_time - entry_time
                            trade["duration_seconds"] = duration.total_seconds()
                            trade["duration_str"] = str(duration).split('.')[0]
                        except:
                            trade["duration_seconds"] = 0
                            trade["duration_str"] = "unknown"

                        closed_trades.append(trade)
                        symbols_to_remove.append(symbol)

                        logger.info(f"[TRACKER] {symbol} closed by {exit_reason}: {realized_pips:+.1f} pips (${realized_pl:+.2f})")

            # Move closed trades
            for symbol in symbols_to_remove:
                trade = self.trades["active"][symbol]
                self.trades["closed"].append(trade)
                del self.trades["active"][symbol]

            if closed_trades:
                self._save_json(self.trades_file, self.trades)

            # Update sync state
            self.sync_state["last_sync"] = datetime.now().isoformat()
            self._save_json(self.sync_state_file, self.sync_state)

        except Exception as e:
            logger.error(f"[TRACKER] Error syncing with OANDA: {e}")

        return closed_trades

    def record_market_snapshot(
        self,
        symbol: str,
        price_data: Dict,
        indicators: Dict,
        htf_trend: str,
        ltf_trend: str
    ):
        """
        Record market snapshot for data collection.

        Args:
            symbol: Trading pair
            price_data: OHLCV data
            indicators: Current indicator values
            htf_trend: Higher timeframe trend
            ltf_trend: Lower timeframe trend
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "price": price_data,
            "indicators": indicators,
            "htf_trend": htf_trend,
            "ltf_trend": ltf_trend
        }

        self.market_snapshots.append(snapshot)

        # Keep last 50000 snapshots
        if len(self.market_snapshots) > 50000:
            self.market_snapshots = self.market_snapshots[-50000:]

        # Save periodically (every 100 snapshots)
        if len(self.market_snapshots) % 100 == 0:
            self._save_json(self.market_data_file, self.market_snapshots)

    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        closed = self.trades.get("closed", [])

        if not closed:
            return {"total_trades": 0}

        wins = [t for t in closed if t.get("outcome") == "WIN"]
        losses = [t for t in closed if t.get("outcome") == "LOSS"]

        total_pips = sum(t.get("realized_pips", 0) for t in closed)
        total_pl = sum(t.get("realized_pl", 0) for t in closed)

        # Per-symbol stats
        by_symbol = defaultdict(lambda: {"wins": 0, "losses": 0, "pips": 0, "pl": 0})
        for t in closed:
            sym = t.get("symbol", "UNKNOWN")
            by_symbol[sym]["pips"] += t.get("realized_pips", 0)
            by_symbol[sym]["pl"] += t.get("realized_pl", 0)
            if t.get("outcome") == "WIN":
                by_symbol[sym]["wins"] += 1
            elif t.get("outcome") == "LOSS":
                by_symbol[sym]["losses"] += 1

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "total_pips": total_pips,
            "total_pl": total_pl,
            "avg_win_pips": sum(t.get("realized_pips", 0) for t in wins) / len(wins) if wins else 0,
            "avg_loss_pips": sum(t.get("realized_pips", 0) for t in losses) / len(losses) if losses else 0,
            "by_symbol": dict(by_symbol),
            "active_trades": len(self.trades.get("active", {}))
        }

    def get_recent_trades(self, count: int = 10) -> List[Dict]:
        """Get most recent closed trades."""
        return self.trades.get("closed", [])[-count:]

    def get_active_trades(self) -> Dict:
        """Get all active trades."""
        return self.trades.get("active", {})
