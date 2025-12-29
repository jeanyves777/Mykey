#!/usr/bin/env python3
"""
Analyze Trade History - DCA vs Non-DCA Performance
===================================================
Analyzes trade log to show performance by DCA level
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.trading_config import LOGGING_CONFIG


def analyze_trades():
    """Analyze trade history from JSON log"""

    # Get log file path
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        LOGGING_CONFIG["log_dir"]
    )
    trade_log_path = os.path.join(log_dir, LOGGING_CONFIG["trade_log_file"])

    if not os.path.exists(trade_log_path):
        print(f"No trade log found at: {trade_log_path}")
        return

    with open(trade_log_path, 'r') as f:
        trades = json.load(f)

    print("=" * 80)
    print("TRADE HISTORY ANALYSIS")
    print("=" * 80)
    print(f"Total trades in log: {len(trades)}")
    print()

    # Group by DCA level
    dca_stats = {
        0: {"trades": [], "pnl": 0, "wins": 0, "losses": 0},  # No DCA
        1: {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
        2: {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
        3: {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
        4: {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
    }

    # Group by symbol
    symbol_stats = {}

    # Exit type stats
    exit_stats = {}

    for trade in trades:
        dca_level = trade.get("dca_level", 0)
        pnl = trade.get("pnl", 0)
        symbol = trade.get("symbol", "UNKNOWN")
        exit_type = trade.get("exit_type", "UNKNOWN")

        # DCA stats
        if dca_level not in dca_stats:
            dca_stats[dca_level] = {"trades": [], "pnl": 0, "wins": 0, "losses": 0}

        dca_stats[dca_level]["trades"].append(trade)
        dca_stats[dca_level]["pnl"] += pnl
        if pnl > 0:
            dca_stats[dca_level]["wins"] += 1
        else:
            dca_stats[dca_level]["losses"] += 1

        # Symbol stats
        if symbol not in symbol_stats:
            symbol_stats[symbol] = {
                "trades": 0, "pnl": 0, "wins": 0, "losses": 0,
                "dca_counts": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            }
        symbol_stats[symbol]["trades"] += 1
        symbol_stats[symbol]["pnl"] += pnl
        if pnl > 0:
            symbol_stats[symbol]["wins"] += 1
        else:
            symbol_stats[symbol]["losses"] += 1
        symbol_stats[symbol]["dca_counts"][dca_level] = symbol_stats[symbol]["dca_counts"].get(dca_level, 0) + 1

        # Exit type stats
        if exit_type not in exit_stats:
            exit_stats[exit_type] = {"trades": 0, "pnl": 0}
        exit_stats[exit_type]["trades"] += 1
        exit_stats[exit_type]["pnl"] += pnl

    # Print DCA Analysis
    print("=" * 80)
    print("ANALYSIS BY DCA LEVEL")
    print("=" * 80)
    print(f"{'DCA Level':<12} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'Win Rate':>10} {'Total PNL':>14} {'Avg PNL':>12}")
    print("-" * 80)

    total_trades = 0
    total_pnl = 0
    total_wins = 0
    total_losses = 0

    for level in sorted(dca_stats.keys()):
        stats = dca_stats[level]
        num_trades = len(stats["trades"])
        if num_trades == 0:
            continue

        total_trades += num_trades
        total_pnl += stats["pnl"]
        total_wins += stats["wins"]
        total_losses += stats["losses"]

        win_rate = (stats["wins"] / num_trades * 100) if num_trades > 0 else 0
        avg_pnl = stats["pnl"] / num_trades if num_trades > 0 else 0

        level_name = "No DCA" if level == 0 else f"DCA L{level}"
        print(f"{level_name:<12} {num_trades:>8} {stats['wins']:>6} {stats['losses']:>8} {win_rate:>9.1f}% ${stats['pnl']:>12.2f} ${avg_pnl:>10.2f}")

    print("-" * 80)
    total_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_avg = total_pnl / total_trades if total_trades > 0 else 0
    print(f"{'TOTAL':<12} {total_trades:>8} {total_wins:>6} {total_losses:>8} {total_wr:>9.1f}% ${total_pnl:>12.2f} ${total_avg:>10.2f}")

    # Print Symbol Analysis with DCA breakdown
    print("\n" + "=" * 80)
    print("ANALYSIS BY SYMBOL (with DCA breakdown)")
    print("=" * 80)

    for symbol in sorted(symbol_stats.keys(), key=lambda x: symbol_stats[x]["pnl"], reverse=True):
        stats = symbol_stats[symbol]
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0

        print(f"\n{symbol}:")
        print(f"  Trades: {stats['trades']} | Wins: {stats['wins']} | Losses: {stats['losses']} | Win Rate: {wr:.1f}%")
        print(f"  Total PNL: ${stats['pnl']:+.2f}")

        # DCA breakdown
        dca_str = []
        for level in range(5):
            count = stats["dca_counts"].get(level, 0)
            if count > 0:
                level_name = "No DCA" if level == 0 else f"L{level}"
                dca_str.append(f"{level_name}: {count}")
        print(f"  DCA Levels: {' | '.join(dca_str)}")

    # Print Exit Type Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS BY EXIT TYPE")
    print("=" * 80)
    print(f"{'Exit Type':<15} {'Trades':>8} {'Total PNL':>14}")
    print("-" * 40)

    for exit_type in sorted(exit_stats.keys()):
        stats = exit_stats[exit_type]
        print(f"{exit_type:<15} {stats['trades']:>8} ${stats['pnl']:>12.2f}")

    # Print detailed trade list
    print("\n" + "=" * 80)
    print("DETAILED TRADE LIST")
    print("=" * 80)
    print(f"{'Timestamp':<20} {'Symbol':<10} {'Side':<6} {'DCA':>4} {'Exit':>12} {'PNL':>12}")
    print("-" * 80)

    for trade in sorted(trades, key=lambda x: x.get("timestamp", "")):
        ts = trade.get("timestamp", "")[:19]
        symbol = trade.get("symbol", "")
        side = trade.get("side", "")
        dca = trade.get("dca_level", 0)
        exit_type = trade.get("exit_type", "")
        pnl = trade.get("pnl", 0)

        dca_str = "None" if dca == 0 else f"L{dca}"
        print(f"{ts:<20} {symbol:<10} {side:<6} {dca_str:>4} {exit_type:>12} ${pnl:>10.2f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate DCA vs Non-DCA comparison
    non_dca_trades = len(dca_stats[0]["trades"])
    dca_trades = sum(len(dca_stats[l]["trades"]) for l in range(1, 5))
    non_dca_pnl = dca_stats[0]["pnl"]
    dca_pnl = sum(dca_stats[l]["pnl"] for l in range(1, 5))

    print(f"\nNon-DCA Trades: {non_dca_trades} | PNL: ${non_dca_pnl:+.2f}")
    print(f"DCA Trades: {dca_trades} | PNL: ${dca_pnl:+.2f}")

    if non_dca_trades > 0 and dca_trades > 0:
        non_dca_avg = non_dca_pnl / non_dca_trades
        dca_avg = dca_pnl / dca_trades
        print(f"\nAvg PNL per Non-DCA trade: ${non_dca_avg:+.2f}")
        print(f"Avg PNL per DCA trade: ${dca_avg:+.2f}")

    print(f"\nOVERALL: {total_trades} trades | ${total_pnl:+.2f} | {total_wr:.1f}% win rate")
    print("=" * 80)


if __name__ == "__main__":
    analyze_trades()
