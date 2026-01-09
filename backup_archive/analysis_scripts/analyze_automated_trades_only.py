"""
Analyze ONLY automated trades (exclude manual exits during testing phase)
Focus on trades where TP/SL worked automatically
"""

import json
import os

print("=" * 100)
print("AUTOMATED TRADES ANALYSIS - STRATEGY PERFORMANCE")
print("Excludes manual exits from testing/debugging phase")
print("=" * 100)

# Read all log files
log_files = [
    "forex_logs/trades_20251211.jsonl",
    "forex_logs/trades_20251212.jsonl",
]

all_entries = []
all_exits = []

print("\n[1/3] Reading trade logs...")
for log_file in log_files:
    if not os.path.exists(log_file):
        continue

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'ENTRY':
                    all_entries.append(data)
                elif data.get('type') == 'EXIT':
                    all_exits.append(data)
            except:
                pass

print(f"      Total: {len(all_entries)} entries, {len(all_exits)} exits")

# Filter for AUTOMATED exits only (TP, SL, TRAIL)
print("\n[2/3] Filtering for automated exits only...")

automated_exits = [e for e in all_exits if e.get('exit_reason') in ['TP', 'SL', 'TRAIL']]
manual_exits = [e for e in all_exits if e.get('exit_reason') not in ['TP', 'SL', 'TRAIL']]

print(f"      Automated exits (TP/SL/TRAIL): {len(automated_exits)}")
print(f"      Manual exits (testing phase): {len(manual_exits)}")

# Match automated exits with entries
print("\n[3/3] Matching automated trades...")

automated_trades = []

for exit_data in automated_exits:
    trade_id = exit_data.get('trade_id')

    # Find matching entry
    entry_data = None
    for e in all_entries:
        if e.get('trade_id') == trade_id:
            entry_data = e
            break

    if entry_data:
        automated_trades.append({
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
            'exit_time': exit_data.get('timestamp')
        })

print(f"      Matched automated trades: {len(automated_trades)}")

if len(automated_trades) == 0:
    print("\n[WARNING] No automated trades found yet!")
    print("This means all recent trades are still open or were manually closed during testing.")
    print("Wait for more trades to hit TP/SL automatically.")
    exit(0)

# Calculate statistics
print("\n" + "=" * 100)
print("AUTOMATED STRATEGY PERFORMANCE (TP/SL/TRAIL ONLY)")
print("=" * 100)

total_trades = len(automated_trades)
winners = [t for t in automated_trades if t['won']]
losers = [t for t in automated_trades if not t['won']]

total_pl = sum(t['pnl'] for t in automated_trades)
total_wins = sum(t['pnl'] for t in winners)
total_losses = abs(sum(t['pnl'] for t in losers))

win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

avg_win = total_wins / len(winners) if winners else 0
avg_loss = total_losses / len(losers) if losers else 0
avg_trade = total_pl / total_trades if total_trades > 0 else 0

print(f"\nOverall Performance:")
print(f"  Total Automated Trades: {total_trades}")
print(f"  Winners:                {len(winners)} ({win_rate:.1f}%)")
print(f"  Losers:                 {len(losers)}")
print(f"  Total P&L:              ${total_pl:+,.2f}")
print(f"  Profit Factor:          {profit_factor:.2f}")

print(f"\nAverages:")
print(f"  Avg Winner:             ${avg_win:+.2f}")
print(f"  Avg Loser:              ${-avg_loss:.2f}")
print(f"  Avg Trade:              ${avg_trade:+.2f}")

# Top winners and losers
if winners:
    print(f"\nTop Winners:")
    for i, t in enumerate(sorted(winners, key=lambda x: x['pnl'], reverse=True)[:5], 1):
        print(f"  {i}. {t['instrument']}: ${t['pnl']:+.2f} ({t['pips']:.1f} pips) [{t['exit_reason']}]")

if losers:
    print(f"\nTop Losers:")
    for i, t in enumerate(sorted(losers, key=lambda x: x['pnl'])[:5], 1):
        print(f"  {i}. {t['instrument']}: ${t['pnl']:.2f} ({t['pips']:.1f} pips) [{t['exit_reason']}]")

# Per-pair breakdown
print("\n" + "=" * 100)
print("PER-PAIR PERFORMANCE (AUTOMATED ONLY)")
print("=" * 100)

by_pair = {}
for t in automated_trades:
    pair = t['instrument']
    if pair not in by_pair:
        by_pair[pair] = {'trades': 0, 'wins': 0, 'pl': 0}

    by_pair[pair]['trades'] += 1
    by_pair[pair]['pl'] += t['pnl']

    if t['won']:
        by_pair[pair]['wins'] += 1

print(f"\n{'Pair':<12} {'Trades':<8} {'Win Rate':<12} {'Total P&L':<12} {'Avg P&L'}")
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

by_exit = {}
for t in automated_trades:
    reason = t['exit_reason']
    if reason not in by_exit:
        by_exit[reason] = {'count': 0, 'wins': 0, 'pl': 0}

    by_exit[reason]['count'] += 1
    by_exit[reason]['pl'] += t['pnl']
    if t['won']:
        by_exit[reason]['wins'] += 1

print(f"\n{'Exit Reason':<15} {'Count':<8} {'Win Rate':<12} {'Total P&L'}")
print("-" * 60)

for reason in ['TP', 'SL', 'TRAIL']:
    if reason in by_exit:
        data = by_exit[reason]
        wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f"{reason:<15} {data['count']:<8} {wr:>6.1f}%      ${data['pl']:+.2f}")

# Direction breakdown
print("\n" + "=" * 100)
print("DIRECTION BREAKDOWN")
print("=" * 100)

by_direction = {'BUY': {'count': 0, 'wins': 0, 'pl': 0}, 'SELL': {'count': 0, 'wins': 0, 'pl': 0}}

for t in automated_trades:
    direction = t['direction']
    by_direction[direction]['count'] += 1
    by_direction[direction]['pl'] += t['pnl']
    if t['won']:
        by_direction[direction]['wins'] += 1

print(f"\n{'Direction':<12} {'Trades':<8} {'Win Rate':<12} {'Total P&L'}")
print("-" * 60)

for direction in ['BUY', 'SELL']:
    data = by_direction[direction]
    if data['count'] > 0:
        wr = (data['wins'] / data['count'] * 100)
        print(f"{direction:<12} {data['count']:<8} {wr:>6.1f}%      ${data['pl']:+.2f}")

# Compare to backtest
print("\n" + "=" * 100)
print("STRATEGY VALIDATION")
print("=" * 100)

print(f"\nCombined V2 Strategy:")
print(f"  Backtest (1 month):        +25.82% | 51.9% WR | 1.34 PF")
print(f"  Live Automated (Recent):   N/A     | {win_rate:.1f}% WR | {profit_factor:.2f} PF")

if win_rate >= 45 and profit_factor >= 1.2:
    print(f"\n  Status: VALIDATED - Strategy performing within expected range")
elif total_trades < 20:
    print(f"\n  Status: NEED MORE DATA - Only {total_trades} automated trades")
else:
    print(f"\n  Status: UNDERPERFORMING - Review entry criteria")

print("\n" + "=" * 100)
print("EXCLUDED FROM ANALYSIS")
print("=" * 100)

print(f"\nManual exits during testing: {len(manual_exits)} trades")
print(f"These were excluded because they represent testing/debugging, not strategy performance")

print("\n" + "=" * 100)
