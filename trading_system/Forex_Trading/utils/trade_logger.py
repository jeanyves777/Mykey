"""
Comprehensive Trade and Market Data Logger for Forex Trading
Logs every trade with full signal analysis and market conditions
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


def make_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


class ForexTradeLogger:
    """Logs all trades and market data for post-analysis"""

    def __init__(self, log_dir: str = "forex_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create session-based log files (date + time for unique session ID)
        self.session_start = datetime.now()
        self.date_str = self.session_start.strftime("%Y%m%d")
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        # Session-specific log files
        self.trade_log_file = self.log_dir / f"trades_{self.session_id}.jsonl"
        self.market_log_file = self.log_dir / f"market_data_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"session_summary_{self.session_id}.json"

        # In-memory tracking for session summary
        self.daily_trades = []
        self.daily_signals_skipped = []

        # Log session start marker
        self._log_session_start()

        print(f"[LOGGER] ========================================")
        print(f"[LOGGER] NEW SESSION STARTED: {self.session_id}")
        print(f"[LOGGER] ========================================")
        print(f"[LOGGER] Trade log: {self.trade_log_file}")
        print(f"[LOGGER] Market log: {self.market_log_file}")

    def _log_session_start(self):
        """Log session start marker for easy identification"""
        session_marker = {
            "type": "SESSION_START",
            "session_id": self.session_id,
            "timestamp": self.session_start.isoformat(),
            "date": self.date_str
        }

        with open(self.trade_log_file, 'w') as f:  # 'w' to start fresh
            f.write(json.dumps(session_marker) + '\n')

        with open(self.market_log_file, 'w') as f:  # 'w' to start fresh
            f.write(json.dumps(session_marker) + '\n')

    def log_trade_entry(self,
                       instrument: str,
                       direction: str,
                       entry_price: float,
                       entry_time: datetime,
                       units: int,
                       stop_loss: float,
                       take_profit: float,
                       trailing_distance: float,
                       signal_analysis: Dict,
                       account_balance: float,
                       trade_id: str = None):
        """Log trade entry with full signal analysis"""

        # Make signal_analysis JSON serializable
        safe_analysis = make_json_serializable(signal_analysis) if signal_analysis else {}

        entry_data = {
            "type": "ENTRY",
            "timestamp": entry_time.isoformat(),
            "instrument": instrument,
            "direction": direction,
            "entry_price": float(entry_price),
            "units": int(units),
            "position_value": float(abs(units * entry_price)),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "trailing_distance": float(trailing_distance),
            "trade_id": trade_id,
            "account_balance": float(account_balance),

            # Signal analysis
            "signal_analysis": {
                "momentum": safe_analysis.get('momentum', {}),
                "htf_trend": safe_analysis.get('htf', {}),
                "rsi": float(safe_analysis.get('rsi', 0)) if safe_analysis.get('rsi') else 0,
                "pullback": safe_analysis.get('pullback', {}),
                "confidence": safe_analysis.get('confidence', 0),
                "session": safe_analysis.get('session', '')
            },

            # Risk metrics (JPY pairs use 100, others use 10000)
            "risk_metrics": {
                "sl_pips": float(abs(stop_loss - entry_price) * (100 if 'JPY' in instrument else 10000)),
                "tp_pips": float(abs(take_profit - entry_price) * (100 if 'JPY' in instrument else 10000)),
                "risk_reward_ratio": float(abs(take_profit - entry_price) / abs(stop_loss - entry_price)) if abs(stop_loss - entry_price) > 0 else 0,
                "position_size_pct": float((abs(units * entry_price) / account_balance) * 100)
            }
        }

        # Write to JSONL (one line per trade)
        with open(self.trade_log_file, 'a') as f:
            f.write(json.dumps(entry_data) + '\n')

        self.daily_trades.append(entry_data)

        print(f"[LOGGER] Logged ENTRY: {instrument} {direction} @ {entry_price:.5f}")

    def log_trade_exit(self,
                      instrument: str,
                      direction: str,
                      entry_price: float,
                      entry_time: datetime,
                      exit_price: float,
                      exit_time: datetime,
                      exit_reason: str,
                      pnl: float,
                      pnl_pct: float,
                      account_balance: float,
                      trade_id: str = None):
        """Log trade exit with P&L"""

        # Calculate pips correctly (JPY pairs use 100, others use 10000)
        pip_multiplier = 100 if 'JPY' in instrument else 10000

        # Calculate pips based on direction
        if direction == "BUY":
            pips = (exit_price - entry_price) * pip_multiplier
        else:  # SELL
            pips = (entry_price - exit_price) * pip_multiplier

        exit_data = {
            "type": "EXIT",
            "timestamp": exit_time.isoformat(),
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time.isoformat(),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "trade_id": trade_id,

            # P&L metrics
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "pips": pips,  # Now direction-aware and JPY-correct
            "duration_minutes": (exit_time - entry_time).total_seconds() / 60,
            "account_balance": account_balance,

            # Classification - won is based on ACTUAL P&L, not exit reason
            "won": pnl > 0,
            "exit_type": exit_reason
        }

        # Write to JSONL
        with open(self.trade_log_file, 'a') as f:
            f.write(json.dumps(exit_data) + '\n')

        print(f"[LOGGER] Logged EXIT: {instrument} {exit_reason} P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

    def log_signal_skipped(self,
                          instrument: str,
                          reason: str,
                          timestamp: datetime,
                          signal_analysis: Dict = None):
        """Log when a signal is skipped and why"""

        # Make analysis JSON serializable (handle numpy bools, etc.)
        safe_analysis = make_json_serializable(signal_analysis) if signal_analysis else {}

        skip_data = {
            "type": "SKIP",
            "timestamp": timestamp.isoformat(),
            "instrument": instrument,
            "reason": reason,
            "analysis": safe_analysis
        }

        self.daily_signals_skipped.append(skip_data)

        # Write to market log
        with open(self.market_log_file, 'a') as f:
            f.write(json.dumps(skip_data) + '\n')

    def log_market_snapshot(self,
                           instrument: str,
                           timestamp: datetime,
                           price: float,
                           indicators: Dict):
        """Log market conditions at a point in time"""

        snapshot = {
            "type": "MARKET",
            "timestamp": timestamp.isoformat(),
            "instrument": instrument,
            "price": price,
            "indicators": indicators
        }

        with open(self.market_log_file, 'a') as f:
            f.write(json.dumps(snapshot) + '\n')

    def generate_daily_summary(self):
        """Generate session summary"""

        entries = [t for t in self.daily_trades if t['type'] == 'ENTRY']

        if not entries:
            print("[LOGGER] No trades this session - no summary generated")
            return

        # Read all trade exits
        exits = []
        if self.trade_log_file.exists():
            with open(self.trade_log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('type') == 'EXIT':
                        exits.append(data)

        # Calculate statistics
        total_trades = len(exits)
        winners = [t for t in exits if t['won']]
        losers = [t for t in exits if not t['won']]

        # Calculate session duration
        session_end = datetime.now()
        session_duration = session_end - self.session_start

        summary = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "session_duration_minutes": session_duration.total_seconds() / 60,
            "date": self.date_str,
            "total_entries": len(entries),
            "total_exits": total_trades,
            "open_trades": len(entries) - total_trades,

            "performance": {
                "total_trades": total_trades,
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": len(winners) / total_trades * 100 if total_trades > 0 else 0,

                "total_pnl": sum(t['pnl'] for t in exits),
                "avg_win": sum(t['pnl'] for t in winners) / len(winners) if winners else 0,
                "avg_loss": sum(t['pnl'] for t in losers) / len(losers) if losers else 0,

                "avg_win_pips": sum(t['pips'] for t in winners) / len(winners) if winners else 0,
                "avg_loss_pips": sum(t['pips'] for t in losers) / len(losers) if losers else 0,

                "profit_factor": abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in losers)) if losers and sum(t['pnl'] for t in losers) != 0 else float('inf')
            },

            "by_instrument": {},
            "by_direction": {
                "LONG": {"count": 0, "won": 0, "pnl": 0},
                "SHORT": {"count": 0, "won": 0, "pnl": 0}
            },

            "exit_reasons": {
                "TP": len([t for t in exits if t['exit_reason'] == 'TP']),
                "SL": len([t for t in exits if t['exit_reason'] == 'SL']),
                "TRAIL": len([t for t in exits if t['exit_reason'] == 'TRAIL'])
            },

            "signals_skipped": len(self.daily_signals_skipped)
        }

        # Per-instrument breakdown
        for t in exits:
            inst = t['instrument']
            if inst not in summary['by_instrument']:
                summary['by_instrument'][inst] = {
                    "trades": 0, "won": 0, "lost": 0, "pnl": 0
                }

            summary['by_instrument'][inst]['trades'] += 1
            if t['won']:
                summary['by_instrument'][inst]['won'] += 1
            else:
                summary['by_instrument'][inst]['lost'] += 1
            summary['by_instrument'][inst]['pnl'] += t['pnl']

        # Per-direction breakdown
        for t in exits:
            direction = "LONG" if t['direction'] == 'BUY' else "SHORT"
            summary['by_direction'][direction]['count'] += 1
            if t['won']:
                summary['by_direction'][direction]['won'] += 1
            summary['by_direction'][direction]['pnl'] += t['pnl']

        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[LOGGER] Session summary saved: {self.summary_file}")
        print(f"[LOGGER] Session: {self.session_id} | Duration: {session_duration}")
        print(f"[LOGGER] Trades: {total_trades} | Win Rate: {summary['performance']['win_rate']:.1f}% | P&L: ${summary['performance']['total_pnl']:+.2f}")
        return summary

    def export_to_csv(self):
        """Export trades to CSV for Excel analysis"""

        if not self.trade_log_file.exists():
            return

        # Read all trades
        trades = []
        with open(self.trade_log_file, 'r') as f:
            for line in f:
                trades.append(json.loads(line))

        # Match entries with exits
        matched_trades = []

        entries = {t['timestamp']: t for t in trades if t['type'] == 'ENTRY'}
        exits = [t for t in trades if t['type'] == 'EXIT']

        for exit_trade in exits:
            entry_time = exit_trade.get('entry_time')
            entry = entries.get(entry_time)

            if entry:
                matched_trades.append({
                    'Date': exit_trade['timestamp'][:10],
                    'Entry_Time': entry['timestamp'],
                    'Exit_Time': exit_trade['timestamp'],
                    'Instrument': exit_trade['instrument'],
                    'Direction': exit_trade['direction'],
                    'Entry_Price': entry['entry_price'],
                    'Exit_Price': exit_trade['exit_price'],
                    'Exit_Reason': exit_trade['exit_reason'],
                    'PnL': exit_trade['pnl'],
                    'PnL_Pct': exit_trade['pnl_pct'],
                    'Pips': exit_trade['pips'],
                    'Duration_Min': exit_trade['duration_minutes'],
                    'Won': exit_trade['won'],

                    # Signal data
                    'Momentum_Score': entry['signal_analysis']['momentum'].get('score', 0),
                    'HTF_Trend': entry['signal_analysis']['htf_trend'].get('trend', ''),
                    'RSI': entry['signal_analysis']['rsi'],
                    'Confidence': entry['signal_analysis']['confidence'],

                    # Risk metrics
                    'SL_Pips': entry['risk_metrics']['sl_pips'],
                    'TP_Pips': entry['risk_metrics']['tp_pips'],
                    'RR_Ratio': entry['risk_metrics']['risk_reward_ratio'],
                    'Position_Size_Pct': entry['risk_metrics']['position_size_pct']
                })

        if matched_trades:
            csv_file = self.log_dir / f"trades_{self.session_id}.csv"
            df = pd.DataFrame(matched_trades)
            df.to_csv(csv_file, index=False)
            print(f"[LOGGER] Exported {len(matched_trades)} trades to CSV: {csv_file}")
            return csv_file

        return None

    def log_session_end(self, final_stats: dict = None):
        """Log session end marker with final statistics"""
        session_end = datetime.now()
        session_duration = session_end - self.session_start

        end_marker = {
            "type": "SESSION_END",
            "session_id": self.session_id,
            "timestamp": session_end.isoformat(),
            "duration_minutes": session_duration.total_seconds() / 60,
            "final_stats": final_stats or {}
        }

        with open(self.trade_log_file, 'a') as f:
            f.write(json.dumps(end_marker) + '\n')

        print(f"\n[LOGGER] ========================================")
        print(f"[LOGGER] SESSION ENDED: {self.session_id}")
        print(f"[LOGGER] Duration: {session_duration}")
        print(f"[LOGGER] ========================================")

    def get_monthly_stats(self):
        """Aggregate all logs for the month"""

        current_month = datetime.now().strftime("%Y%m")

        all_trades = []

        # Read all trade logs for this month
        for log_file in self.log_dir.glob(f"trades_{current_month}*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data['type'] == 'EXIT':
                        all_trades.append(data)

        if not all_trades:
            print("[LOGGER] No monthly data yet")
            return None

        winners = [t for t in all_trades if t['won']]
        losers = [t for t in all_trades if not t['won']]

        monthly_stats = {
            "month": current_month,
            "total_trades": len(all_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(all_trades) * 100,
            "total_pnl": sum(t['pnl'] for t in all_trades),
            "avg_win": sum(t['pnl'] for t in winners) / len(winners) if winners else 0,
            "avg_loss": sum(t['pnl'] for t in losers) / len(losers) if losers else 0,
            "profit_factor": abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in losers)) if losers and sum(t['pnl'] for t in losers) != 0 else float('inf'),
            "trading_days": len(set(t['timestamp'][:10] for t in all_trades))
        }

        print(f"\n[LOGGER] Monthly Stats ({current_month}):")
        print(f"  Total Trades: {monthly_stats['total_trades']}")
        print(f"  Win Rate: {monthly_stats['win_rate']:.1f}%")
        print(f"  Total P&L: ${monthly_stats['total_pnl']:+,.2f}")
        print(f"  Profit Factor: {monthly_stats['profit_factor']:.2f}")
        print(f"  Trading Days: {monthly_stats['trading_days']}")

        # Save monthly summary
        monthly_file = self.log_dir / f"monthly_summary_{current_month}.json"
        with open(monthly_file, 'w') as f:
            json.dump(monthly_stats, f, indent=2)

        return monthly_stats
