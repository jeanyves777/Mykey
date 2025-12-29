#!/usr/bin/env python3
"""Close DOT position now - it's already past TP target!"""
import sys
sys.path.insert(0, '.')
from engine.binance_client import BinanceClient

client = BinanceClient(testnet=True)

# Get DOT position
pos = client.get_position("DOTUSDT")
if not pos:
    print("No DOTUSDT position found!")
    sys.exit(1)

price_data = client.get_current_price("DOTUSDT")
entry = pos["entry_price"]
current = price_data["price"]
qty = pos["quantity"]
pnl = pos["unrealized_pnl"]
side = pos["side"]

pnl_pct = ((current - entry) / entry) * 100 if side == "LONG" else ((entry - current) / entry) * 100
roi = pnl_pct * 20

print("=" * 60)
print("DOTUSDT POSITION STATUS")
print("=" * 60)
print(f"Side: {side}")
print(f"Entry Price: ${entry:,.4f}")
print(f"Current Price: ${current:,.4f}")
print(f"Quantity: {qty}")
print(f"Unrealized PNL: ${pnl:+,.2f}")
print(f"Price Change: {pnl_pct:+.2f}%")
print(f"ROI (20x leverage): {roi:+.1f}%")
print()
print(f"Expected TP was at 5% ROI = ${entry * 1.0025:,.4f}")
print(f"Current ROI is {roi:.1f}% - ALREADY WAY PAST TP!")
print("=" * 60)

# Ask to close
response = input("\nClose position now to take profit? (y/n): ")
if response.lower() == 'y':
    # Close position
    close_side = "SELL" if side == "LONG" else "BUY"
    print(f"\nClosing {qty} {side} position...")
    result = client.place_market_order("DOTUSDT", close_side, qty)

    if "orderId" in result:
        print(f"SUCCESS! Order ID: {result['orderId']}")
        print(f"Realized PNL: ~${pnl:+,.2f}")

        # Cancel any remaining SL orders
        try:
            client.cancel_all_orders("DOTUSDT")
            print("Cancelled remaining SL/TP orders")
        except:
            pass
    else:
        print(f"FAILED: {result}")
else:
    print("Cancelled - position kept open")
