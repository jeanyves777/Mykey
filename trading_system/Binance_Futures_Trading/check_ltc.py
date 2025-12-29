#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from engine.binance_client import BinanceClient

client = BinanceClient(testnet=True)

# Check LTC
pos = client.get_position("LTCUSDT")
if not pos:
    print("No LTCUSDT position")
    sys.exit(0)

price = client.get_current_price("LTCUSDT")
entry = pos["entry_price"]
current = price["price"]
qty = pos["quantity"]
pnl = pos["unrealized_pnl"]
side = pos["side"]

# Expected TP at 15% ROI (no DCA)
tp_price = entry * 1.0075 if side == "LONG" else entry * 0.9925
roi = ((current - entry) / entry) * 20 * 100 if side == "LONG" else ((entry - current) / entry) * 20 * 100

print(f"LTCUSDT {side}")
print(f"Entry: ${entry:.2f}, Current: ${current:.2f}")
print(f"Expected TP: ${tp_price:.2f}")
print(f"ROI: {roi:.1f}%")
print(f"PNL: ${pnl:.2f}")

if (side == "LONG" and current >= tp_price) or (side == "SHORT" and current <= tp_price):
    print("\nPAST TP - should close!")
    response = input("Close now? (y/n): ")
    if response.lower() == 'y':
        close_side = "SELL" if side == "LONG" else "BUY"
        result = client.place_market_order("LTCUSDT", close_side, qty)
        if "orderId" in result:
            print(f"CLOSED! Order: {result['orderId']}")
            client.cancel_all_orders("LTCUSDT")
        else:
            print(f"Failed: {result}")
else:
    print("\nNot past TP yet")
