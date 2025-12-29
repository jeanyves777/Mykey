import sys, os
sys.path.insert(0, os.path.abspath('.'))
from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from datetime import datetime

client = OandaClient()
print("Getting trade history from OANDA...")

history = client.get_trade_history(count=50)
print(f"\nFound {len(history)} trades\n")
print("=" * 100)

total_pl = 0
count = 0
for trade in history:
    pl = float(trade.get('realized_pl', 0))
    instrument = trade.get('instrument', 'UNKNOWN')
    units = trade.get('initial_units', 0)
    
    total_pl += pl
    count += 1
    
    print(f"{count}. {instrument}: ${pl:+.2f}")

print("=" * 100)
print(f"Total: {count} trades | P&L: ${total_pl:+.2f}")
print("=" * 100)
