import json
import os
from datetime import datetime

print("=" * 100)
print("REAL TRADING SESSION ANALYSIS")
print("=" * 100)

starting_balance = 4906.17
current_balance = 4937.58
total_pl = current_balance - starting_balance

print(f"\nAccount Performance:")
print(f"  Starting Balance: ${starting_balance:,.2f}")
print(f"  Current Balance:  ${current_balance:,.2f}")
print(f"  Total P&L:        ${total_pl:+,.2f}")
print(f"  Return:           {total_pl/starting_balance*100:+.2f}%")

log_file = "forex_logs/trades_20251211.jsonl"
if not os.path.exists(log_file):
    print(f"\nERROR: Log file not found")
    exit(1)

entries = []
exits = []

with open(log_file, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'ENTRY':
                entries.append(data)
            elif data.get('type') == 'EXIT':
                exits.append(data)
        except:
            pass

print(f"\nTotal Entries: {len(entries)}")
print(f"Total Exits: {len(exits)}")

trades = []
for exit_log in exits:
    trade_id = exit_log.get('trade_id')
    entry_log = None
    for e in entries:
        if e.get('trade_id') == trade_id:
            entry_log = e
            break
    if entry_log:
        trades.append({
            'id': trade_id,
            'instrument': exit_log.get('instrument'),
            'pl': exit_log.get('pl', 0),
            'pips': exit_log.get('pips', 0),
            'exit_reason': exit_log.get('exit_reason', 'UNKNOWN')
        })

print(f"\nMatched Trades: {len(trades)}")

if len(trades) > 0:
    winners = [t for t in trades if t['pl'] > 0]
    losers = [t for t in trades if t['pl'] < 0]
    win_rate = len(winners) / len(trades) * 100
    
    print(f"\nWinners: {len(winners)}")
    print(f"Losers: {len(losers)}")
    print(f"Win Rate: {win_rate:.1f}%")
    
    total_wins = sum(t['pl'] for t in winners)
    total_losses = sum(abs(t['pl']) for t in losers)
    
    print(f"\nTotal Wins: ${total_wins:+.2f}")
    print(f"Total Losses: ${-total_losses:.2f}")
    print(f"Net P&L: ${total_wins - total_losses:+.2f}")
    
    if total_losses > 0:
        pf = total_wins / total_losses
        print(f"Profit Factor: {pf:.2f}")

print("\n" + "=" * 100)
