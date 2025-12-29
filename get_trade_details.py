import sys, os
sys.path.insert(0, os.path.abspath('.'))
from trading_system.Forex_Trading.engine.oanda_client import OandaClient

client = OandaClient()
history = client.get_trade_history(count=15)

print("=" * 100)
print("LAST 15 TRADES (Most Recent Session)")
print("=" * 100)

total_pl = 0
winners = 0
losers = 0

for i, trade in enumerate(history[:15], 1):
    pl = float(trade.get('realized_pl', 0))
    instrument = trade.get('instrument', 'UNKNOWN')
    
    total_pl += pl
    if pl > 0:
        winners += 1
    elif pl < 0:
        losers += 1
    
    status = "WIN" if pl > 0 else "LOSS" if pl < 0 else "BE"
    print(f"{i}. {instrument}: ${pl:+.2f} [{status}]")

print("=" * 100)
print(f"\nTotal: {len(history[:15])} trades")
print(f"Winners: {winners} | Losers: {losers}")
print(f"Win Rate: {winners/15*100:.1f}%")
print(f"Total P&L: ${total_pl:+.2f}")
print("=" * 100)
