"""Close all test trades and show actual position values"""
import sys, os
sys.path.insert(0, '.')
from trading_system.Forex_Trading.engine.oanda_client import OandaClient

os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

balance = client.get_balance()
print(f"\nAccount Balance: ${balance:,.2f}")

trades = client.get_open_trades()
print(f"\nOpen Test Trades: {len(trades)}\n")

print(f"{'Pair':<12} {'Units':<10} {'Entry Price':<12} {'Position $':<14} {'% of Acct':<12}")
print("-" * 70)

for trade in trades:
    inst = trade['instrument']
    units = abs(trade['units'])
    price = trade['price']

    # Calculate position value
    pos_value = units * price
    pct = (pos_value / balance) * 100

    print(f"{inst:<12} {units:<10.0f} {price:<12.5f} ${pos_value:<13,.2f} {pct:<11.2f}%")

print("\nClosing all test trades...")
for trade in trades:
    result = client.close_trade(trade['id'])
    if result.get('success'):
        pnl = result.get('pl', 0)
        print(f"  Closed {trade['instrument']}: P&L ${pnl:+.2f}")
    else:
        print(f"  Failed to close {trade['instrument']}: {result.get('error')}")

print("\nAll test trades closed!")
