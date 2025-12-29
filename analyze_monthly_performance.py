"""
Analyze Monthly Trading Performance from Logs
Run this after 1 month of trading to evaluate strategy
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

print("=" * 80)
print("MONTHLY FOREX TRADING PERFORMANCE ANALYSIS")
print("=" * 80)

# Check for logs
log_dir = Path("forex_logs")

if not log_dir.exists():
    print(f"\nERROR: Log directory '{log_dir}' not found!")
    print("Make sure you have run the live trading system first.")
    sys.exit(1)

# Find all trade logs
trade_files = list(log_dir.glob("trades_*.jsonl"))

if not trade_files:
    print(f"\nERROR: No trade logs found in '{log_dir}'!")
    print("Run the live trading system to generate logs.")
    sys.exit(1)

print(f"\nFound {len(trade_files)} trading log file(s):")
for f in sorted(trade_files):
    print(f"  - {f.name}")

# Load all trades
all_entries = []
all_exits = []

for trade_file in trade_files:
    with open(trade_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['type'] == 'ENTRY':
                all_entries.append(data)
            elif data['type'] == 'EXIT':
                all_exits.append(data)

print(f"\nTotal Entries Logged: {len(all_entries)}")
print(f"Total Exits Logged: {len(all_exits)}")

if len(all_exits) == 0:
    print("\nNo completed trades yet. Keep trading and run this analysis later!")
    sys.exit(0)

# Convert to DataFrame
df_exits = pd.DataFrame(all_exits)
df_exits['timestamp'] = pd.to_datetime(df_exits['timestamp'])
df_exits['entry_time'] = pd.to_datetime(df_exits['entry_time'])

# Calculate statistics
total_trades = len(df_exits)
winners = df_exits[df_exits['won'] == True]
losers = df_exits[df_exits['won'] == False]

win_rate = len(winners) / total_trades * 100
total_pnl = df_exits['pnl'].sum()
total_pips = df_exits['pips'].sum()

avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0

avg_win_pips = winners['pips'].mean() if len(winners) > 0 else 0
avg_loss_pips = losers['pips'].mean() if len(losers) > 0 else 0

total_win_pnl = winners['pnl'].sum()
total_loss_pnl = abs(losers['pnl'].sum())
profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')

# Time analysis
df_exits['date'] = df_exits['timestamp'].dt.date
trading_days = df_exits['date'].nunique()
trades_per_day = total_trades / trading_days

# Print results
print("\n" + "=" * 80)
print("OVERALL PERFORMANCE")
print("=" * 80)

print(f"\nTrading Period:")
print(f"  Start: {df_exits['entry_time'].min()}")
print(f"  End: {df_exits['timestamp'].max()}")
print(f"  Trading Days: {trading_days}")
print(f"  Trades/Day: {trades_per_day:.1f}")

print(f"\nTrade Statistics:")
print(f"  Total Trades: {total_trades}")
print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
print(f"  Losers: {len(losers)}")

print(f"\nProfit & Loss:")
print(f"  Total P&L: ${total_pnl:+,.2f}")
print(f"  Total Pips: {total_pips:+,.0f}")
print(f"  Avg Win: ${avg_win:,.2f} ({avg_win_pips:+.1f} pips)")
print(f"  Avg Loss: ${avg_loss:,.2f} ({avg_loss_pips:+.1f} pips)")
print(f"  Profit Factor: {profit_factor:.2f}")

# Risk/Reward
avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
print(f"  Avg Risk:Reward: 1:{avg_rr:.2f}")

# Per-instrument analysis
print("\n" + "=" * 80)
print("PER-INSTRUMENT BREAKDOWN")
print("=" * 80)

print(f"\n{'Instrument':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<12} {'Pips':<10} {'Avg Win':<10} {'Avg Loss'}")
print("-" * 80)

for instrument in sorted(df_exits['instrument'].unique()):
    inst_trades = df_exits[df_exits['instrument'] == instrument]
    inst_winners = inst_trades[inst_trades['won'] == True]
    inst_losers = inst_trades[inst_trades['won'] == False]

    inst_wr = len(inst_winners) / len(inst_trades) * 100
    inst_pnl = inst_trades['pnl'].sum()
    inst_pips = inst_trades['pips'].sum()
    inst_avg_win = inst_winners['pnl'].mean() if len(inst_winners) > 0 else 0
    inst_avg_loss = inst_losers['pnl'].mean() if len(inst_losers) > 0 else 0

    print(f"{instrument:<12} {len(inst_trades):<8} {inst_wr:<9.0f}% "
          f"${inst_pnl:<10,.2f} {inst_pips:<9.0f} "
          f"${inst_avg_win:<9.2f} ${inst_avg_loss:<.2f}")

# Directional bias
print("\n" + "=" * 80)
print("DIRECTIONAL ANALYSIS")
print("=" * 80)

for instrument in sorted(df_exits['instrument'].unique()):
    inst_trades = df_exits[df_exits['instrument'] == instrument]

    longs = inst_trades[inst_trades['direction'] == 'BUY']
    shorts = inst_trades[inst_trades['direction'] == 'SELL']

    long_winners = longs[longs['won'] == True]
    short_winners = shorts[shorts['won'] == True]

    long_wr = len(long_winners) / len(longs) * 100 if len(longs) > 0 else 0
    short_wr = len(short_winners) / len(shorts) * 100 if len(shorts) > 0 else 0

    long_pnl = longs['pnl'].sum() if len(longs) > 0 else 0
    short_pnl = shorts['pnl'].sum() if len(shorts) > 0 else 0

    print(f"\n{instrument}:")
    print(f"  LONG:  {len(longs)} trades | {long_wr:.0f}% WR | ${long_pnl:+.2f}")
    print(f"  SHORT: {len(shorts)} trades | {short_wr:.0f}% WR | ${short_pnl:+.2f}")

    if long_wr > 55 and long_wr > short_wr:
        print(f"  --> LONG BIAS ({long_wr:.0f}% WR)")
    elif short_wr > 55 and short_wr > long_wr:
        print(f"  --> SHORT BIAS ({short_wr:.0f}% WR)")

# Exit reasons
print("\n" + "=" * 80)
print("EXIT ANALYSIS")
print("=" * 80)

exit_counts = df_exits['exit_reason'].value_counts()
print(f"\nExit Reasons:")
for reason, count in exit_counts.items():
    pct = count / total_trades * 100
    print(f"  {reason}: {count} ({pct:.1f}%)")

# Duration analysis
df_exits['duration_hours'] = df_exits['duration_minutes'] / 60

print(f"\nTrade Duration:")
print(f"  Average: {df_exits['duration_minutes'].mean():.0f} minutes ({df_exits['duration_hours'].mean():.1f} hours)")
print(f"  Median: {df_exits['duration_minutes'].median():.0f} minutes")
print(f"  Min: {df_exits['duration_minutes'].min():.0f} minutes")
print(f"  Max: {df_exits['duration_minutes'].max():.0f} minutes")

# Best and worst trades
print("\n" + "=" * 80)
print("BEST & WORST TRADES")
print("=" * 80)

best_trade = df_exits.loc[df_exits['pnl'].idxmax()]
worst_trade = df_exits.loc[df_exits['pnl'].idxmin()]

print(f"\nBest Trade:")
print(f"  {best_trade['instrument']} {best_trade['direction']} on {best_trade['timestamp'].strftime('%Y-%m-%d')}")
print(f"  P&L: ${best_trade['pnl']:+.2f} ({best_trade['pnl_pct']:+.1f}%)")
print(f"  Pips: {best_trade['pips']:+.0f}")
print(f"  Exit: {best_trade['exit_reason']}")

print(f"\nWorst Trade:")
print(f"  {worst_trade['instrument']} {worst_trade['direction']} on {worst_trade['timestamp'].strftime('%Y-%m-%d')}")
print(f"  P&L: ${worst_trade['pnl']:+.2f} ({worst_trade['pnl_pct']:+.1f}%)")
print(f"  Pips: {worst_trade['pips']:+.0f}")
print(f"  Exit: {worst_trade['exit_reason']}")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# 1. Remove underperforming pairs
print("\n1. PAIR SELECTION:")
for instrument in sorted(df_exits['instrument'].unique()):
    inst_trades = df_exits[df_exits['instrument'] == instrument]
    inst_winners = inst_trades[inst_trades['won'] == True]
    inst_wr = len(inst_winners) / len(inst_trades) * 100
    inst_pnl = inst_trades['pnl'].sum()

    if inst_wr >= 45 and inst_pnl > 0:
        status = "[KEEP]"
    else:
        status = "[REVIEW]"

    print(f"  {status} {instrument}: {inst_wr:.0f}% WR | ${inst_pnl:+.2f} | {len(inst_trades)} trades")

# 2. Directional filters
print("\n2. CONSIDER DIRECTIONAL FILTERS:")
for instrument in sorted(df_exits['instrument'].unique()):
    inst_trades = df_exits[df_exits['instrument'] == instrument]
    longs = inst_trades[inst_trades['direction'] == 'BUY']
    shorts = inst_trades[inst_trades['direction'] == 'SELL']

    if len(longs) > 0 and len(shorts) > 0:
        long_winners = longs[longs['won'] == True]
        short_winners = shorts[shorts['won'] == True]

        long_wr = len(long_winners) / len(longs) * 100
        short_wr = len(short_winners) / len(shorts) * 100

        if long_wr > 60 and long_wr > short_wr + 15:
            print(f"  {instrument}: Trade LONG only ({long_wr:.0f}% vs {short_wr:.0f}%)")
        elif short_wr > 60 and short_wr > long_wr + 15:
            print(f"  {instrument}: Trade SHORT only ({short_wr:.0f}% vs {long_wr:.0f}%)")

# 3. Overall verdict
print("\n3. OVERALL VERDICT:")
if win_rate >= 50 and profit_factor >= 1.5:
    print("  [EXCELLENT] Strategy is profitable. Keep trading!")
elif win_rate >= 45 and profit_factor >= 1.2:
    print("  [GOOD] Strategy is profitable but could be improved.")
elif total_pnl > 0:
    print("  [MARGINAL] Strategy is barely profitable. Consider optimizations.")
else:
    print("  [NEEDS WORK] Strategy needs significant improvements.")

print("\n" + "=" * 80)
print(f"Analysis complete! Keep trading to gather more data.")
print("=" * 80)
