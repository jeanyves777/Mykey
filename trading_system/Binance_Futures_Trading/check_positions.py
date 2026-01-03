"""Check actual Binance positions"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient

client = BinanceClient(testnet=False)

positions = client.get_positions()

print(f"\n{'='*80}")
print(f"ACTUAL BINANCE POSITIONS")
print(f"{'='*80}")

for pos in positions:
    symbol = pos.get('symbol', 'N/A')
    side = pos.get('side', 'N/A')
    qty = pos.get('quantity', 0)
    entry = pos.get('entry_price', 0)
    leverage = pos.get('leverage', 0)
    margin = pos.get('isolated_wallet', 0)
    
    position_value = qty * entry
    
    print(f"\n{symbol} {side}:")
    print(f"  Qty: {qty}")
    print(f"  Entry: ${entry:,.2f}")
    print(f"  Position Value: ${position_value:,.2f}")
    print(f"  Leverage: {leverage}x")
    print(f"  Isolated Margin: ${margin:.2f}")

print(f"\n{'='*80}\n")
