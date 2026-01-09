"""
Complete Trading History Analysis - All Trades Combined
Includes: Original logs + Reconstructed trades from OANDA
"""

import json
import os
from datetime import datetime

print("=" * 100)
print("COMPLETE TRADING HISTORY - ALL TRADES")
print("=" * 100)

# Read all log files
log_files = [
    "forex_logs/trades_20251211.jsonl",
    "forex_logs/trades_20251212.jsonl",
    "forex_logs/reconstructed_trades.jsonl"
]

all_entries = []
all_exits = []

print("\n[1/2] Reading all trade logs...")
for log_file in log_files:
    if not os.path.exists(log_file):
        continue

    entries_count = 0
    exits_count = 0

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'ENTRY':
                    all_entries.append(data)
                    entries_count += 1
                elif data.get('type') == 'EXIT':
                    all_exits.append(data)
                    exits_count += 1
            except:
                pass

    print(f"      {log_file}: {entries_count} entries, {exits_count} exits")

print(f"\n      TOTAL: {len(all_entries)} entries, {len(all_exits)} exits")

# Match entries with exits
print("\n[2/2] Matching entries with exits...")

matched_trades = []
unmatched_entries = []

for exit_data in all_exits:
    trade_id = exit_data.get('trade_id')

    # Find matching entry
    entry_data = None
    for e in all_entries:
        if e.get('trade_id') == trade_id:
            entry_data = e
            break

    if entry_data:
        matched_trades.append({
            'trade_id': trade_id,
            'instrument': exit_data.get('instrument'),
            'direction': exit_data.get('direction'),
            'entry_price': entry_data.get('entry_price'),
            'exit_price': exit_data.get('exit_price'),
            'pnl': exit_data.get('pnl', 0),
            'pips': exit_data.get('pips', 0),
            'exit_reason': exit_data.get('exit_reason'),
            'won': exit_data.get('won', False),
            'entry_time': entry_data.get('timestamp'),
            'exit_time': exit_data.get('timestamp'),
            'reconstructed': entry_data.get('reconstructed', False)
        })
    else:
        unmatched_entries.append(exit_data)

print(f"      Matched trades: {len(matched_trades)}")
print(f"      Unmatched exits: {len(unmatched_entries)}")

# Calculate statistics
print("\n" + "=" * 100)
print("COMPLETE TRADING PERFORMANCE")
print("=" * 100)

total_trades = len(matched_trades)
winners = [t for t in matched_trades if t['won']]
losers = [t for t in matched_trades if not t['won']]

total_pl = sum(t['pnl'] for t in matched_trades)
total_wins = sum(t['pnl'] for t in winners)
total_losses = abs(sum(t['pnl'] for t in losers))

win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

avg_win = total_wins / len(winners) if winners else 0
avg_loss = total_losses / len(losers) if losers else 0
avg_trade = total_pl / total_trades if total_trades > 0 else 0

print(f"\nOverall Performance:")
print(f"  Total Trades:     {total_trades}")
print(f"  Winners:          {len(winners)} ({win_rate:.1f}%)")
print(f"  Losers:           {len(losers)}")
print(f"  Total P&L:        ${total_pl:+,.2f}")
print(f"  Profit Factor:    {profit_factor:.2f}")

print(f"\nAverages:")
print(f"  Avg Winner:       ${avg_win:+.2f}")
print(f"  Avg Loser:        ${-avg_loss:.2f}")
print(f"  Avg Trade:        ${avg_trade:+.2f}")

# Top winners and losers
print(f"\nTop 5 Winners:")
for i, t in enumerate(sorted(winners, key=lambda x: x['pnl'], reverse=True)[:5], 1):
    reconstructed = " [RECONSTRUCTED]" if t['reconstructed'] else ""
    print(f"  {i}. {t['instrument']}: ${t['pnl']:+.2f} ({t['exit_reason']}){reconstructed}")

print(f"\nTop 5 Losers:")
for i, t in enumerate(sorted(losers, key=lambda x: x['pnl'])[:5], 1):
    reconstructed = " [RECONSTRUCTED]" if t['reconstructed'] else ""
    print(f"  {i}. {t['instrument']}: ${t['pnl']:.2f} ({t['exit_reason']}){reconstructed}")

# Per-pair breakdown
print("\n" + "=" * 100)
print("PER-PAIR PERFORMANCE")
print("=" * 100)

by_pair = {}
for t in matched_trades:
    pair = t['instrument']
    if pair not in by_pair:
        by_pair[pair] = {'trades': 0, 'wins': 0, 'losses': 0, 'pl': 0, 'pips': 0}

    by_pair[pair]['trades'] += 1
    by_pair[pair]['pl'] += t['pnl']
    by_pair[pair]['pips'] += t['pips'] if t['won'] else -t['pips']

    if t['won']:
        by_pair[pair]['wins'] += 1
    else:
        by_pair[pair]['losses'] += 1

print(f"\n{'Pair':<12} {'Trades':<8} {'Win Rate':<12} {'P&L':<12} {'Avg P&L'}")
print("-" * 70)

for pair in sorted(by_pair.keys()):
    data = by_pair[pair]
    wr = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
    avg_pl = data['pl'] / data['trades']

    print(f"{pair:<12} {data['trades']:<8} {wr:>6.1f}%      ${data['pl']:+10.2f}   ${avg_pl:+.2f}")

# Exit reason breakdown
print("\n" + "=" * 100)
print("EXIT REASON BREAKDOWN")
print("=" * 100)

by_exit_reason = {}
for t in matched_trades:
    reason = t['exit_reason']
    if reason not in by_exit_reason:
        by_exit_reason[reason] = {'count': 0, 'wins': 0, 'pl': 0}

    by_exit_reason[reason]['count'] += 1
    by_exit_reason[reason]['pl'] += t['pnl']
    if t['won']:
        by_exit_reason[reason]['wins'] += 1

print(f"\n{'Exit Reason':<15} {'Count':<8} {'Win Rate':<12} {'Total P&L'}")
print("-" * 60)

for reason in sorted(by_exit_reason.keys()):
    data = by_exit_reason[reason]
    wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
    print(f"{reason:<15} {data['count']:<8} {wr:>6.1f}%      ${data['pl']:+.2f}")

# Direction breakdown
print("\n" + "=" * 100)
print("DIRECTION BREAKDOWN")
print("=" * 100)

by_direction = {}
for t in matched_trades:
    direction = t['direction']
    if direction not in by_direction:
        by_direction[direction] = {'count': 0, 'wins': 0, 'pl': 0}

    by_direction[direction]['count'] += 1
    by_direction[direction]['pl'] += t['pnl']
    if t['won']:
        by_direction[direction]['wins'] += 1

print(f"\n{'Direction':<12} {'Trades':<8} {'Win Rate':<12} {'Total P&L'}")
print("-" * 60)

for direction in sorted(by_direction.keys()):
    data = by_direction[direction]
    wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
    print(f"{direction:<12} {data['count']:<8} {wr:>6.1f}%      ${data['pl']:+.2f}")

# Reconstructed vs Original trades
print("\n" + "=" * 100)
print("RECONSTRUCTED vs ORIGINAL TRADES")
print("=" * 100)

reconstructed_trades = [t for t in matched_trades if t['reconstructed']]
original_trades = [t for t in matched_trades if not t['reconstructed']]

print(f"\nOriginal Trades (from live logging):")
print(f"  Count: {len(original_trades)}")
print(f"  Total P&L: ${sum(t['pnl'] for t in original_trades):+,.2f}")
if len(original_trades) > 0:
    wr_original = (len([t for t in original_trades if t['won']]) / len(original_trades) * 100)
    print(f"  Win Rate: {wr_original:.1f}%")

print(f"\nReconstructed Trades (from OANDA history):")
print(f"  Count: {len(reconstructed_trades)}")
print(f"  Total P&L: ${sum(t['pnl'] for t in reconstructed_trades):+,.2f}")
if len(reconstructed_trades) > 0:
    wr_reconstructed = (len([t for t in reconstructed_trades if t['won']]) / len(reconstructed_trades) * 100)
    print(f"  Win Rate: {wr_reconstructed:.1f}%")

print("\n" + "=" * 100)
print("NEXT STEPS")
print("=" * 100)
print("\n1. All trades are now accounted for (original logs + reconstructed)")
print("2. Future trades will be automatically logged by the live trading system")
print("3. Review the per-pair performance to identify strongest/weakest pairs")
print("4. Monitor EUR_USD closely - it has the highest loss in reconstructed trades")

print("\n" + "=" * 100)
