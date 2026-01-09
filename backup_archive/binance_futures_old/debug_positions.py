#!/usr/bin/env python3
"""Debug: Check raw Binance position data for hedge mode."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient

# Connect to LIVE (not testnet)
client = BinanceClient(testnet=False, use_demo=False)

print("=" * 60)
print("RAW BINANCE POSITION DATA (HEDGE MODE DEBUG)")
print("=" * 60)

account = client.get_account_info()

print("\n--- All positions with non-zero data ---")
for pos in account.get("positions", []):
    position_amt = float(pos.get("positionAmt", 0))
    entry_price = float(pos.get("entryPrice", 0))
    unrealized_pnl = float(pos.get("unrealizedProfit", 0))
    position_side = pos.get("positionSide", "BOTH")
    isolated_wallet = float(pos.get("isolatedWallet", 0))

    # Show positions that have any activity
    if position_amt != 0 or entry_price > 0 or isolated_wallet != 0:
        print(f"\nSymbol: {pos.get('symbol')}")
        print(f"  positionSide: {position_side}")  # KEY FIELD FOR HEDGE MODE
        print(f"  positionAmt: {position_amt}")
        print(f"  entryPrice: {entry_price}")
        print(f"  unrealizedProfit: {unrealized_pnl}")
        print(f"  isolatedWallet: {isolated_wallet}")
        print(f"  leverage: {pos.get('leverage')}")

print("\n--- Processed positions ---")
positions = client.get_positions()
print(f"Total positions found: {len(positions)}")
for p in positions:
    print(f"  {p['symbol']} {p['side']}: qty={p['quantity']}, pnl={p['unrealized_pnl']}")

print("\n--- Test get_position with side filter ---")
for symbol in ["DOTUSDT", "AVAXUSDT"]:
    for side in ["LONG", "SHORT"]:
        pos = client.get_position(symbol, position_side=side)
        if pos:
            print(f"  {symbol} {side}: entry={pos['entry_price']}, pnl={pos['unrealized_pnl']}, qty={pos['quantity']}")
        else:
            print(f"  {symbol} {side}: NOT FOUND")

print("\n--- Total PNL calculation ---")
total_pnl = sum(p['unrealized_pnl'] for p in positions)
print(f"Total Unrealized PNL: ${total_pnl:.4f}")
