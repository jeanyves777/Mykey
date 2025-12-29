"""
Analyze the most recent trades from OANDA
Focus on last night + this morning (automated TP/SL)
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))
from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from datetime import datetime, timedelta
import pytz

client = OandaClient()

print("=" * 100)
print("RECENT AUTOMATED TRADES - LAST 24 HOURS")
print("=" * 100)

# Get recent trade history
history = client.get_trade_history(count=50)

# Filter for last 24 hours
now = datetime.now(pytz.UTC)
yesterday = now - timedelta(hours=24)

recent_trades = []
for trade in history:
    close_time_str = trade.get('close_time', '')
    if close_time_str:
        try:
            # Parse the close time
            if isinstance(close_time_str, str):
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
            else:
                close_time = close_time_str

            if close_time >= yesterday:
                recent_trades.append(trade)
        except:
            pass

print(f"\nFound {len(recent_trades)} trades in last 24 hours")
print(f"Balance: ${client.get_balance():,.2f}")

if len(recent_trades) == 0:
    print("\nNo trades found in last 24 hours")
    exit(0)

# Analyze these trades
total_pl = 0
winners = 0
losers = 0
by_pair = {}

print("\n" + "=" * 100)
print("LAST 24 HOURS - TRADE BY TRADE")
print("=" * 100)

for i, trade in enumerate(recent_trades, 1):
    trade_id = trade.get('id')
    instrument = trade.get('instrument')
    pl = float(trade.get('realized_pl', 0))
    units = trade.get('initial_units', 0)
    direction = "BUY" if units > 0 else "SELL"

    total_pl += pl

    if pl > 0:
        winners += 1
        status = "WIN"
    elif pl < 0:
        losers += 1
        status = "LOSS"
    else:
        status = "BE"

    # Per-pair tracking
    if instrument not in by_pair:
        by_pair[instrument] = {'count': 0, 'pl': 0, 'wins': 0}
    by_pair[instrument]['count'] += 1
    by_pair[instrument]['pl'] += pl
    if pl > 0:
        by_pair[instrument]['wins'] += 1

    print(f"{i}. Trade #{trade_id}: {instrument} {direction:<5} ${pl:+8.2f} [{status}]")

# Summary
print("\n" + "=" * 100)
print("24-HOUR PERFORMANCE SUMMARY")
print("=" * 100)

total_trades = len(recent_trades)
win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

print(f"\nTotal Trades: {total_trades}")
print(f"Winners: {winners} ({win_rate:.1f}%)")
print(f"Losers: {losers}")
print(f"Total P&L: ${total_pl:+,.2f}")

if winners > 0 and losers > 0:
    total_wins = sum(float(t.get('realized_pl', 0)) for t in recent_trades if float(t.get('realized_pl', 0)) > 0)
    total_losses = abs(sum(float(t.get('realized_pl', 0)) for t in recent_trades if float(t.get('realized_pl', 0)) < 0))

    if total_losses > 0:
        pf = total_wins / total_losses
        print(f"Profit Factor: {pf:.2f}")

    avg_win = total_wins / winners
    avg_loss = total_losses / losers

    print(f"\nAvg Win: ${avg_win:+.2f}")
    print(f"Avg Loss: ${-avg_loss:.2f}")
    print(f"Avg Trade: ${total_pl/total_trades:+.2f}")

# Per-pair breakdown
print("\n" + "=" * 100)
print("PER-PAIR PERFORMANCE (24 HOURS)")
print("=" * 100)

print(f"\n{'Pair':<12} {'Trades':<8} {'Win Rate':<12} {'Total P&L'}")
print("-" * 60)

for pair in sorted(by_pair.keys()):
    data = by_pair[pair]
    wr = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
    print(f"{pair:<12} {data['count']:<8} {wr:>6.1f}%      ${data['pl']:+.2f}")

# Compare to backtest
print("\n" + "=" * 100)
print("STRATEGY VALIDATION")
print("=" * 100)

print(f"\nCombined V2 Strategy:")
print(f"  Backtest (1 month):  +25.82% | 51.9% WR | 1.34 PF")

if total_trades > 0 and winners > 0 and losers > 0:
    total_wins = sum(float(t.get('realized_pl', 0)) for t in recent_trades if float(t.get('realized_pl', 0)) > 0)
    total_losses = abs(sum(float(t.get('realized_pl', 0)) for t in recent_trades if float(t.get('realized_pl', 0)) < 0))
    pf = total_wins / total_losses if total_losses > 0 else 0

    print(f"  Live (24h):          {total_pl/client.get_balance()*100:+.2f}% | {win_rate:.1f}% WR | {pf:.2f} PF")

    if win_rate >= 45 and pf >= 1.2:
        print(f"\n  Status: STRATEGY VALIDATED!")
        print(f"  Live performance matches backtest expectations")
    elif total_trades < 15:
        print(f"\n  Status: NEED MORE DATA (only {total_trades} trades)")
    else:
        print(f"\n  Status: UNDERPERFORMING - Review settings")

print("\n" + "=" * 100)
