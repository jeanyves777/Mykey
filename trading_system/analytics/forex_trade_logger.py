"""
Forex Trade Logger for NinjaTrader Bridge

Records every forex futures trade with full details for strategy analysis:
- Entry/exit timestamps, prices, quantities
- TP/SL levels
- Exit reason (TP, SL, Manual)
- Actual P&L (ticks, dollars)
- Symbol details (tick size, tick value)
- FundedNext rule compliance tracking
"""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForexTradeRecord:
    """Complete record of a single forex futures trade"""
    # Trade identifiers
    trade_id: str
    symbol: str  # M6E, M6B, MJY, MCD, MSF
    oanda_symbol: str  # EUR_USD, GBP_USD, etc.

    # Entry details
    entry_time: str
    entry_price: float
    quantity: int  # Always 1 for FundedNext
    side: str  # BUY or SELL

    # Target levels
    stop_loss: float
    take_profit: float

    # Exit details (filled when closed)
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""  # TP, SL, MANUAL, RULE_VIOLATION

    # P&L details
    gross_pnl_ticks: float = 0.0  # P&L in ticks
    gross_pnl_usd: float = 0.0  # P&L in dollars
    fees_usd: float = 0.0
    net_pnl_usd: float = 0.0
    pnl_pct: float = 0.0

    # Hold duration
    hold_minutes: float = 0.0

    # Symbol details
    tick_size: float = 0.0
    tick_value: float = 0.0

    # FundedNext tracking
    account_balance_before: float = 0.0
    account_balance_after: float = 0.0
    daily_profit_before: float = 0.0
    daily_profit_after: float = 0.0
    trades_today_before: int = 0
    trades_today_after: int = 0

    # Market conditions at entry
    entry_volatility: float = 0.0

    # Strategy signal details
    signal_reason: str = ""
    signal_confidence: float = 0.0

    # Additional metadata
    session: str = ""  # e.g., "2025-01-15_morning"
    notes: str = ""

    # Status
    status: str = "OPEN"  # OPEN, CLOSED


class ForexTradeLogger:
    """
    Logger for forex futures trades via NinjaTrader bridge

    Saves to:
    - JSON file (one record per trade, easy to read/update)
    - CSV file (all trades, easy for Excel/analysis)
    """

    def __init__(self, log_dir: str = "trading_system/NinjaTrader_Bridge/trade_logs"):
        """Initialize trade logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.json_file = self.log_dir / "trades.json"
        self.csv_file = self.log_dir / "trades.csv"

        # Load existing trades
        self.trades: Dict[str, ForexTradeRecord] = {}
        self._load_trades()

        logger.info(f"ForexTradeLogger initialized. {len(self.trades)} trades loaded.")

    def _load_trades(self):
        """Load existing trades from JSON file"""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                    self.trades = {
                        tid: ForexTradeRecord(**record)
                        for tid, record in data.items()
                    }
                logger.info(f"Loaded {len(self.trades)} trades from {self.json_file}")
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
                self.trades = {}

    def _save_trades(self):
        """Save all trades to JSON and CSV"""
        try:
            # Save to JSON
            with open(self.json_file, 'w') as f:
                json.dump(
                    {tid: asdict(trade) for tid, trade in self.trades.items()},
                    f,
                    indent=2
                )

            # Save to CSV
            if self.trades:
                df = pd.DataFrame([asdict(t) for t in self.trades.values()])
                df.to_csv(self.csv_file, index=False)

            logger.debug(f"Saved {len(self.trades)} trades")

        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def log_entry(
        self,
        trade_id: str,
        symbol: str,
        oanda_symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        tick_size: float,
        tick_value: float,
        account_balance: float,
        daily_profit: float,
        trades_today: int,
        signal_reason: str = "",
        signal_confidence: float = 0.0,
        session: str = "",
        notes: str = ""
    ):
        """
        Log a new trade entry

        Args:
            trade_id: Unique identifier (e.g., "M6E_20250115_140532")
            symbol: NinjaTrader symbol (M6E, M6B, etc.)
            oanda_symbol: OANDA symbol (EUR_USD, GBP_USD, etc.)
            side: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            tick_size: Tick size for symbol
            tick_value: Dollar value per tick
            account_balance: Account balance before trade
            daily_profit: Daily profit before trade
            trades_today: Number of trades today before this one
            signal_reason: Why the trade was taken
            signal_confidence: Confidence score (0-1)
            session: Trading session identifier
            notes: Additional notes
        """
        trade = ForexTradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            oanda_symbol=oanda_symbol,
            entry_time=datetime.now().isoformat(),
            entry_price=entry_price,
            quantity=1,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tick_size=tick_size,
            tick_value=tick_value,
            account_balance_before=account_balance,
            daily_profit_before=daily_profit,
            trades_today_before=trades_today,
            signal_reason=signal_reason,
            signal_confidence=signal_confidence,
            session=session,
            notes=notes,
            status="OPEN"
        )

        self.trades[trade_id] = trade
        self._save_trades()

        logger.info(f"Logged entry: {trade_id} {symbol} {side} @ {entry_price}")

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        account_balance: float,
        daily_profit: float,
        trades_today: int,
        fees_usd: float = 0.0,
        notes: str = ""
    ):
        """
        Log trade exit

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: Why trade was closed (TP, SL, MANUAL, RULE_VIOLATION)
            account_balance: Account balance after trade
            daily_profit: Daily profit after trade
            trades_today: Number of trades today after this one
            fees_usd: Trading fees in USD
            notes: Additional exit notes
        """
        if trade_id not in self.trades:
            logger.error(f"Trade {trade_id} not found")
            return

        trade = self.trades[trade_id]

        # Update exit details
        trade.exit_time = datetime.now().isoformat()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.fees_usd = fees_usd
        trade.account_balance_after = account_balance
        trade.daily_profit_after = daily_profit
        trade.trades_today_after = trades_today
        trade.status = "CLOSED"

        if notes:
            trade.notes = f"{trade.notes}; {notes}" if trade.notes else notes

        # Calculate P&L
        if trade.side == "BUY":
            price_diff = exit_price - trade.entry_price
        else:  # SELL
            price_diff = trade.entry_price - exit_price

        # P&L in ticks
        trade.gross_pnl_ticks = price_diff / trade.tick_size

        # P&L in USD
        trade.gross_pnl_usd = trade.gross_pnl_ticks * trade.tick_value
        trade.net_pnl_usd = trade.gross_pnl_usd - fees_usd

        # P&L percentage (based on notional value)
        notional_value = trade.entry_price / trade.tick_size * trade.tick_value
        if notional_value > 0:
            trade.pnl_pct = (trade.net_pnl_usd / notional_value) * 100

        # Hold duration
        if trade.entry_time and trade.exit_time:
            entry_dt = datetime.fromisoformat(trade.entry_time)
            exit_dt = datetime.fromisoformat(trade.exit_time)
            trade.hold_minutes = (exit_dt - entry_dt).total_seconds() / 60

        self._save_trades()

        logger.info(
            f"Logged exit: {trade_id} {exit_reason} @ {exit_price} "
            f"P&L: ${trade.net_pnl_usd:+.2f} ({trade.gross_pnl_ticks:+.1f} ticks)"
        )

    def get_trade(self, trade_id: str) -> Optional[ForexTradeRecord]:
        """Get a specific trade record"""
        return self.trades.get(trade_id)

    def get_open_trades(self) -> List[ForexTradeRecord]:
        """Get all open trades"""
        return [t for t in self.trades.values() if t.status == "OPEN"]

    def get_closed_trades(self) -> List[ForexTradeRecord]:
        """Get all closed trades"""
        return [t for t in self.trades.values() if t.status == "CLOSED"]

    def get_trades_by_symbol(self, symbol: str) -> List[ForexTradeRecord]:
        """Get all trades for a specific symbol"""
        return [t for t in self.trades.values() if t.symbol == symbol]

    def get_trades_by_session(self, session: str) -> List[ForexTradeRecord]:
        """Get all trades for a specific session"""
        return [t for t in self.trades.values() if t.session == session]

    def get_daily_stats(self, date_str: str = None) -> Dict[str, Any]:
        """
        Get statistics for a specific day

        Args:
            date_str: Date in YYYY-MM-DD format (default: today)

        Returns:
            Dict with daily statistics
        """
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        daily_trades = [
            t for t in self.trades.values()
            if t.entry_time.startswith(date_str)
        ]

        if not daily_trades:
            return {
                'date': date_str,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }

        closed_trades = [t for t in daily_trades if t.status == "CLOSED"]
        winning = [t for t in closed_trades if t.net_pnl_usd > 0]
        losing = [t for t in closed_trades if t.net_pnl_usd < 0]

        total_pnl = sum(t.net_pnl_usd for t in closed_trades)

        return {
            'date': date_str,
            'total_trades': len(daily_trades),
            'closed_trades': len(closed_trades),
            'open_trades': len(daily_trades) - len(closed_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(closed_trades) * 100 if closed_trades else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0.0,
            'best_trade': max((t.net_pnl_usd for t in closed_trades), default=0.0),
            'worst_trade': min((t.net_pnl_usd for t in closed_trades), default=0.0),
            'avg_hold_minutes': sum(t.hold_minutes for t in closed_trades) / len(closed_trades) if closed_trades else 0.0
        }

    def export_to_csv(self, filepath: str = None):
        """Export all trades to CSV file"""
        if filepath is None:
            filepath = self.csv_file

        if self.trades:
            df = pd.DataFrame([asdict(t) for t in self.trades.values()])
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(self.trades)} trades to {filepath}")
            return filepath
        else:
            logger.warning("No trades to export")
            return None

    def export_to_excel(self, filepath: str = None):
        """Export all trades to Excel file with multiple sheets"""
        if filepath is None:
            filepath = self.log_dir / "trades.xlsx"

        if not self.trades:
            logger.warning("No trades to export")
            return None

        # Create DataFrames
        all_trades_df = pd.DataFrame([asdict(t) for t in self.trades.values()])
        open_trades_df = pd.DataFrame([asdict(t) for t in self.get_open_trades()])
        closed_trades_df = pd.DataFrame([asdict(t) for t in self.get_closed_trades()])

        # Write to Excel with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            all_trades_df.to_excel(writer, sheet_name='All Trades', index=False)
            if not open_trades_df.empty:
                open_trades_df.to_excel(writer, sheet_name='Open Trades', index=False)
            if not closed_trades_df.empty:
                closed_trades_df.to_excel(writer, sheet_name='Closed Trades', index=False)

        logger.info(f"Exported {len(self.trades)} trades to {filepath}")
        return filepath
