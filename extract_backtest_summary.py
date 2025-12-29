import json
import sys

# Load the backtest results
with open('backtest_multi_trade_test.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("BACKTEST SUMMARY - COIN Multi-Trade Strategy (Nov 2024)")
print("=" * 60)

trades = data.get('trades', [])
stats = data.get('statistics', {})

print(f"\nTotal Trades: {len(trades)}")
print(f"Final Capital: ${stats.get('final_capital', 0):,.2f}")
print(f"Total P&L: ${stats.get('total_pnl', 0):,.2f}")
print(f"Total Return: {stats.get('total_return_pct', 0):.2f}%")
print(f"\nWin Rate: {stats.get('win_rate', 0):.1f}%")
print(f"Wins: {stats.get('total_wins', 0)}")
print(f"Losses: {stats.get('total_losses', 0)}")
print(f"\nAvg Win: ${stats.get('avg_win', 0):,.2f}")
print(f"Avg Loss: ${stats.get('avg_loss', 0):,.2f}")
print(f"Max Drawdown: {stats.get('max_drawdown_pct', 0):.2f}%")
print(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")

# Daily breakdown
print("\n" + "=" * 60)
print("DAILY PERFORMANCE SUMMARY")
print("=" * 60)

# Group trades by date
from collections import defaultdict
daily_pnl = defaultdict(float)
daily_trades = defaultdict(int)

for trade in trades:
    entry_time = trade.get('entry_time', '')
    if entry_time:
        date = entry_time.split('T')[0]
        pnl = trade.get('pnl', 0)
        daily_pnl[date] += pnl
        daily_trades[date] += 1

print(f"\n{'Date':<12} {'Trades':>8} {'P&L':>12}")
print("-" * 34)
total_winning_days = 0
total_losing_days = 0
for date in sorted(daily_pnl.keys()):
    pnl = daily_pnl[date]
    trades_count = daily_trades[date]
    status = '+' if pnl >= 0 else '-'
    print(f"{date:<12} {trades_count:>8} ${pnl:>10,.2f} {status}")
    if pnl >= 0:
        total_winning_days += 1
    else:
        total_losing_days += 1

print("-" * 34)
print(f"\nTrading Days: {len(daily_pnl)}")
print(f"Winning Days: {total_winning_days}")
print(f"Losing Days: {total_losing_days}")
print(f"Win Day Rate: {total_winning_days / len(daily_pnl) * 100:.1f}%")
