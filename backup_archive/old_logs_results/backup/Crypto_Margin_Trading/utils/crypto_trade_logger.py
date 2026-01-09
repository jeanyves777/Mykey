"""
Crypto Trade Logger
===================
Comprehensive logging system for crypto margin trades.
Mirrors the Forex trade logger with crypto-specific adaptations.

Features:
- Trade entry/exit logging with full details
- Session statistics tracking
- Daily/monthly summaries
- CSV export for analysis
- JSON and JSONL formats
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytz


class CryptoTradeLogger:
    """Trade logging system for crypto margin trading"""

    def __init__(self, log_dir: str = 'crypto_logs', account_type: str = 'paper'):
        """
        Initialize the trade logger.

        Args:
            log_dir: Directory for log files
            account_type: 'paper' or 'live'
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.account_type = account_type

        # Session start time
        self.session_start = datetime.utcnow()
        self.session_id = self.session_start.strftime('%Y%m%d_%H%M%S')

        # Log file paths
        self.trades_file = self.log_dir / f"trades_{self.session_id}.jsonl"
        self.market_file = self.log_dir / f"market_data_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"session_summary_{self.session_id}.json"

        # In-memory tracking
        self.trades: List[Dict] = []
        self.open_trades: Dict[str, Dict] = {}  # trade_id -> trade_data
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'by_pair': {},
            'by_strategy': {},
        }

        print(f"[Logger] Initialized: {self.trades_file}")

    def log_trade_entry(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        entry_price: float,
        volume: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        signal_reason: str,
        **kwargs
    ) -> Dict:
        """
        Log a trade entry.

        Args:
            trade_id: Unique trade identifier
            pair: Trading pair (e.g., 'BTCUSDT')
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            volume: Position volume (in base currency)
            leverage: Leverage used
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy name
            signal_reason: Reason for entry signal
            **kwargs: Additional data

        Returns:
            Trade entry record
        """
        now = datetime.utcnow()

        # Calculate risk/reward
        if direction == 'BUY':
            sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
            tp_distance_pct = ((take_profit - entry_price) / entry_price) * 100
        else:
            sl_distance_pct = ((stop_loss - entry_price) / entry_price) * 100
            tp_distance_pct = ((entry_price - take_profit) / entry_price) * 100

        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0

        trade_entry = {
            'trade_id': trade_id,
            'type': 'ENTRY',
            'timestamp': now.isoformat(),
            'timestamp_unix': int(now.timestamp()),
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'volume': volume,
            'leverage': leverage,
            'position_value': entry_price * volume,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_distance_pct': round(sl_distance_pct, 3),
            'tp_distance_pct': round(tp_distance_pct, 3),
            'rr_ratio': round(rr_ratio, 2),
            'strategy': strategy,
            'signal_reason': signal_reason,
            'account_type': self.account_type,
            **kwargs
        }

        # Store open trade
        self.open_trades[trade_id] = trade_entry

        # Write to file
        self._append_to_jsonl(self.trades_file, trade_entry)

        print(f"[Logger] ENTRY: {pair} {direction} @ {entry_price:.2f} | Vol: {volume:.6f} | {leverage}x | SL: {sl_distance_pct:.2f}% | TP: {tp_distance_pct:.2f}%")

        return trade_entry

    def log_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Log a trade exit.

        Args:
            trade_id: Trade identifier from entry
            exit_price: Exit price
            exit_reason: 'TP', 'SL', 'MANUAL', 'LIQUIDATION'
            pnl: Realized P&L (calculated if not provided)
            **kwargs: Additional data

        Returns:
            Trade exit record or None if trade not found
        """
        now = datetime.utcnow()

        # Get entry data
        entry = self.open_trades.pop(trade_id, None)
        if not entry:
            print(f"[Logger] WARNING: Trade {trade_id} not found in open trades")
            return None

        entry_price = entry['entry_price']
        volume = entry['volume']
        leverage = entry['leverage']
        direction = entry['direction']
        pair = entry['pair']
        strategy = entry['strategy']

        # Calculate P&L if not provided
        if pnl is None:
            if direction == 'BUY':
                pnl = (exit_price - entry_price) * volume * leverage
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
            else:
                pnl = (entry_price - exit_price) * volume * leverage
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
        else:
            position_value = entry_price * volume
            pnl_pct = (pnl / position_value) * 100 * leverage if position_value > 0 else 0

        # Calculate duration
        entry_time = datetime.fromisoformat(entry['timestamp'])
        duration_seconds = (now - entry_time).total_seconds()
        duration_minutes = duration_seconds / 60

        trade_exit = {
            'trade_id': trade_id,
            'type': 'EXIT',
            'timestamp': now.isoformat(),
            'timestamp_unix': int(now.timestamp()),
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'volume': volume,
            'leverage': leverage,
            'exit_reason': exit_reason,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 3),
            'duration_minutes': round(duration_minutes, 1),
            'strategy': strategy,
            'account_type': self.account_type,
            **kwargs
        }

        # Add to trades list
        self.trades.append(trade_exit)

        # Update session stats
        self._update_session_stats(pair, strategy, pnl, pnl_pct, exit_reason)

        # Write to file
        self._append_to_jsonl(self.trades_file, trade_exit)

        # Print result
        result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
        print(f"[Logger] EXIT: {pair} {direction} | {exit_reason} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Duration: {duration_minutes:.1f}m | {result}")

        return trade_exit

    def log_signal_skipped(
        self,
        pair: str,
        direction: str,
        reason: str,
        price: float = None,
        **kwargs
    ):
        """Log a skipped trading signal."""
        now = datetime.utcnow()

        record = {
            'type': 'SIGNAL_SKIPPED',
            'timestamp': now.isoformat(),
            'pair': pair,
            'direction': direction,
            'reason': reason,
            'price': price,
            **kwargs
        }

        self._append_to_jsonl(self.market_file, record)

    def log_market_data(self, pair: str, price: float, **kwargs):
        """Log market data snapshot."""
        now = datetime.utcnow()

        record = {
            'type': 'MARKET_DATA',
            'timestamp': now.isoformat(),
            'pair': pair,
            'price': price,
            **kwargs
        }

        self._append_to_jsonl(self.market_file, record)

    def _update_session_stats(
        self,
        pair: str,
        strategy: str,
        pnl: float,
        pnl_pct: float,
        exit_reason: str
    ):
        """Update session statistics."""
        self.session_stats['total_trades'] += 1
        self.session_stats['total_pnl'] += pnl
        self.session_stats['total_pnl_pct'] += pnl_pct

        if pnl > 0:
            self.session_stats['wins'] += 1
        elif pnl < 0:
            self.session_stats['losses'] += 1
        else:
            self.session_stats['breakeven'] += 1

        # By pair
        if pair not in self.session_stats['by_pair']:
            self.session_stats['by_pair'][pair] = {
                'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0
            }
        self.session_stats['by_pair'][pair]['trades'] += 1
        self.session_stats['by_pair'][pair]['pnl'] += pnl
        if pnl > 0:
            self.session_stats['by_pair'][pair]['wins'] += 1
        elif pnl < 0:
            self.session_stats['by_pair'][pair]['losses'] += 1

        # By strategy
        if strategy not in self.session_stats['by_strategy']:
            self.session_stats['by_strategy'][strategy] = {
                'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0
            }
        self.session_stats['by_strategy'][strategy]['trades'] += 1
        self.session_stats['by_strategy'][strategy]['pnl'] += pnl
        if pnl > 0:
            self.session_stats['by_strategy'][strategy]['wins'] += 1
        elif pnl < 0:
            self.session_stats['by_strategy'][strategy]['losses'] += 1

    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        stats = self.session_stats.copy()

        # Calculate win rate
        total = stats['total_trades']
        if total > 0:
            stats['win_rate'] = (stats['wins'] / total) * 100
        else:
            stats['win_rate'] = 0.0

        return stats

    def print_session_stats(self):
        """Print session statistics to console."""
        stats = self.get_session_stats()

        wins = stats['wins']
        losses = stats['losses']
        total = stats['total_trades']
        win_rate = stats['win_rate']
        total_pnl = stats['total_pnl']

        print("\n" + "=" * 60)
        print(f"SESSION STATS: {wins}W/{losses}L ({win_rate:.1f}%) | Total P&L: ${total_pnl:+.2f}")
        print("-" * 60)
        print(f"{'PAIR':<12} {'TRADES':<8} {'WINS':<6} {'LOSSES':<8} {'WIN%':<8} {'P&L':<10}")
        print("-" * 60)

        for pair, data in stats['by_pair'].items():
            pair_wr = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            print(f"{pair:<12} {data['trades']:<8} {data['wins']:<6} {data['losses']:<8} {pair_wr:<8.1f} ${data['pnl']:+.2f}")

        print("-" * 60)
        print(f"{'TOTAL':<12} {total:<8} {wins:<6} {losses:<8} {win_rate:<8.1f} ${total_pnl:+.2f}")
        print("=" * 60)

    def generate_session_summary(self) -> Dict:
        """Generate and save session summary."""
        now = datetime.utcnow()
        duration = (now - self.session_start).total_seconds() / 3600  # hours

        summary = {
            'session_id': self.session_id,
            'account_type': self.account_type,
            'start_time': self.session_start.isoformat(),
            'end_time': now.isoformat(),
            'duration_hours': round(duration, 2),
            'stats': self.get_session_stats(),
            'trades': self.trades,
        }

        # Save to file
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[Logger] Session summary saved: {self.summary_file}")

        return summary

    def export_to_csv(self) -> Path:
        """Export trades to CSV format."""
        import csv

        csv_file = self.log_dir / f"trades_{self.session_id}.csv"

        if not self.trades:
            print("[Logger] No trades to export")
            return csv_file

        # Get all unique keys from trades
        all_keys = set()
        for trade in self.trades:
            all_keys.update(trade.keys())

        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(self.trades)

        print(f"[Logger] Exported {len(self.trades)} trades to {csv_file}")

        return csv_file

    def _append_to_jsonl(self, filepath: Path, record: Dict):
        """Append a record to a JSONL file."""
        with open(filepath, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def get_monthly_stats(self, year: int = None, month: int = None) -> Dict:
        """
        Get monthly statistics from log files.

        Args:
            year: Year (defaults to current)
            month: Month (defaults to current)

        Returns:
            Monthly statistics dictionary
        """
        if year is None:
            year = datetime.utcnow().year
        if month is None:
            month = datetime.utcnow().month

        # Find all trade files for this month
        pattern = f"trades_{year}{month:02d}*.jsonl"
        trade_files = list(self.log_dir.glob(pattern))

        all_trades = []
        for file in trade_files:
            with open(file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('type') == 'EXIT':
                            all_trades.append(record)
                    except json.JSONDecodeError:
                        continue

        # Calculate stats
        total_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in all_trades if t.get('pnl', 0) < 0)
        total_pnl = sum(t.get('pnl', 0) for t in all_trades)

        return {
            'year': year,
            'month': month,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
        }


if __name__ == "__main__":
    # Test the logger
    logger = CryptoTradeLogger(log_dir='crypto_logs_test', account_type='paper')

    # Simulate some trades
    logger.log_trade_entry(
        trade_id='test_001',
        pair='BTCUSDT',
        direction='BUY',
        entry_price=50000.0,
        volume=0.1,
        leverage=3,
        stop_loss=49500.0,
        take_profit=50400.0,
        strategy='RSI_REVERSAL',
        signal_reason='RSI crossed UP through 35'
    )

    logger.log_trade_exit(
        trade_id='test_001',
        exit_price=50350.0,
        exit_reason='TP'
    )

    logger.log_trade_entry(
        trade_id='test_002',
        pair='ETHUSDT',
        direction='SELL',
        entry_price=3000.0,
        volume=1.0,
        leverage=2,
        stop_loss=3036.0,
        take_profit=2985.0,
        strategy='MACD_CROSS',
        signal_reason='MACD crossed BELOW signal'
    )

    logger.log_trade_exit(
        trade_id='test_002',
        exit_price=3030.0,
        exit_reason='SL'
    )

    # Print stats
    logger.print_session_stats()

    # Generate summary
    logger.generate_session_summary()
    logger.export_to_csv()
