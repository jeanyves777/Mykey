"""
Check current OANDA positions
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient

print("=" * 80)
print("CHECKING OANDA OPEN POSITIONS")
print("=" * 80)

# Initialize client
client = OandaClient()
print(f"\nAccount: {client.account_id}")

# Get account info
account_info = client.get_account_info()
balance = float(account_info.get('balance', 0))
print(f"Balance: ${balance:,.2f}")

# Get open trades
open_trades = client.get_open_trades()

print(f"\n{'='*80}")
print(f"OPEN POSITIONS: {len(open_trades)}")
print(f"{'='*80}\n")

if len(open_trades) == 0:
    print("✓ No open positions - ready to start fresh!\n")
else:
    print(f"Found {len(open_trades)} open position(s):\n")
    for trade in open_trades:
        trade_id = trade.get('id')
        instrument = trade.get('instrument')
        units = trade.get('units')
        entry_price = trade.get('price')
        unrealized_pl = trade.get('unrealized_pl', 0)
        direction = "LONG" if units > 0 else "SHORT"

        print(f"  Trade #{trade_id}: {instrument}")
        print(f"    Direction: {direction}")
        print(f"    Units: {abs(units):,}")
        print(f"    Entry: {entry_price:.5f}")
        print(f"    Unrealized P&L: ${unrealized_pl:+.2f}")
        print()

print("=" * 80)
