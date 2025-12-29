"""
Verify and sync local trade logs with OANDA actual trade history
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.utils.trade_logger import ForexTradeLogger
import json
from datetime import datetime
import pytz

print("=" * 100)
print("TRADE LOG VERIFICATION & SYNC")
print("=" * 100)

# Initialize OANDA client
client = OandaClient()
print(f"\nConnected to OANDA Account: {client.account_id}")

# Get account balance
balance = client.get_balance()
print(f"Current Balance: ${balance:,.2f}")

# Get trade history from OANDA (last 100 trades)
print("\n[1/4] Fetching trade history from OANDA...")
oanda_history = client.get_trade_history(count=100)
print(f"      Found {len(oanda_history)} trades in OANDA history")

# Read local log files
print("\n[2/4] Reading local trade logs...")
log_files = [
    "forex_logs/trades_20251211.jsonl",
    "forex_logs/trades_20251212.jsonl"
]

local_entries = []
local_exits = []

for log_file in log_files:
    if not os.path.exists(log_file):
        print(f"      {log_file}: NOT FOUND (skipping)")
        continue

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'ENTRY':
                    local_entries.append(data)
                elif data.get('type') == 'EXIT':
                    local_exits.append(data)
            except:
                pass

    print(f"      {log_file}: OK")

print(f"      Total: {len(local_entries)} entries, {len(local_exits)} exits")

# Compare local logs with OANDA history
print("\n[3/4] Comparing local logs with OANDA history...")

# Get trade IDs from local logs
local_trade_ids = set()
for entry in local_entries:
    if entry.get('trade_id'):
        local_trade_ids.add(entry['trade_id'])

# Get trade IDs from OANDA
oanda_trade_ids = set()
for trade in oanda_history:
    trade_id = trade.get('id')
    if trade_id:
        oanda_trade_ids.add(trade_id)

# Find missing trades
missing_in_logs = oanda_trade_ids - local_trade_ids
extra_in_logs = local_trade_ids - oanda_trade_ids

print(f"\n      Trade IDs in OANDA:     {len(oanda_trade_ids)}")
print(f"      Trade IDs in local logs: {len(local_trade_ids)}")
print(f"      Missing from logs:       {len(missing_in_logs)}")
print(f"      Extra in logs:           {len(extra_in_logs)}")

if missing_in_logs:
    print(f"\n      WARNING: {len(missing_in_logs)} trades in OANDA but not in local logs!")
    print(f"      Missing trade IDs: {sorted(list(missing_in_logs))[:10]}...")

# Analyze OANDA trades
print("\n[4/4] Analyzing OANDA trade history...")

total_pl = 0
winners = 0
losers = 0
by_pair = {}

for trade in oanda_history:
    pl = float(trade.get('realized_pl', 0))
    instrument = trade.get('instrument', 'UNKNOWN')

    total_pl += pl

    if pl > 0:
        winners += 1
    elif pl < 0:
        losers += 1

    # Per-pair tracking
    if instrument not in by_pair:
        by_pair[instrument] = {'count': 0, 'pl': 0, 'wins': 0}
    by_pair[instrument]['count'] += 1
    by_pair[instrument]['pl'] += pl
    if pl > 0:
        by_pair[instrument]['wins'] += 1

total_trades = len(oanda_history)
win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

print("\n" + "=" * 100)
print("OANDA TRADE HISTORY SUMMARY")
print("=" * 100)

print(f"\nTotal Trades: {total_trades}")
print(f"Winners: {winners} ({win_rate:.1f}%)")
print(f"Losers: {losers}")
print(f"Break-even: {total_trades - winners - losers}")

print(f"\nTotal P&L: ${total_pl:+,.2f}")

if winners > 0 and losers > 0:
    total_wins = sum(float(t.get('realized_pl', 0)) for t in oanda_history if float(t.get('realized_pl', 0)) > 0)
    total_losses = abs(sum(float(t.get('realized_pl', 0)) for t in oanda_history if float(t.get('realized_pl', 0)) < 0))

    if total_losses > 0:
        pf = total_wins / total_losses
        print(f"Profit Factor: {pf:.2f}")

    avg_win = total_wins / winners
    avg_loss = total_losses / losers

    print(f"\nAvg Win: ${avg_win:+.2f}")
    print(f"Avg Loss: ${-avg_loss:.2f}")

# Per-pair breakdown
print("\n" + "=" * 100)
print("PER-PAIR PERFORMANCE (OANDA)")
print("=" * 100)

print(f"\n{'Pair':<12} {'Trades':<8} {'P&L':<12} {'Win Rate':<10}")
print("-" * 50)

for pair in sorted(by_pair.keys()):
    data = by_pair[pair]
    wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
    print(f"{pair:<12} {data['count']:<8} ${data['pl']:+10.2f} {wr:>6.1f}%")

# Status check
print("\n" + "=" * 100)
print("LOG STATUS")
print("=" * 100)

if len(missing_in_logs) > 0:
    print(f"\nSTATUS: INCOMPLETE LOGS")
    print(f"  {len(missing_in_logs)} trades are missing from local logs")
    print(f"  This means some trades were not logged during live trading")
    print(f"\nRECOMMENDATION:")
    print(f"  - Check if the live trading script is properly calling logger.log_trade_entry()")
    print(f"  - Check if the live trading script is properly calling logger.log_trade_exit()")
    print(f"  - Verify the logger is initialized before any trades are placed")
else:
    print(f"\nSTATUS: LOGS COMPLETE")
    print(f"  All OANDA trades are present in local logs")
    print(f"  Trade logging system is working correctly")

print("\n" + "=" * 100)
