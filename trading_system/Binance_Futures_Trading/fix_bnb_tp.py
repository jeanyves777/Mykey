#!/usr/bin/env python3
"""Fix wrong TP order for BNBUSDT"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient

client = BinanceClient(testnet=False)

# Get all open orders for BNBUSDT
orders = client.get_open_orders("BNBUSDT")
print(f"Found {len(orders)} orders for BNBUSDT:")
for o in orders:
    oid = o.get("orderId")
    otype = o.get("type")
    oside = o.get("side")
    pside = o.get("positionSide")
    trigger = o.get("stopPrice")
    print(f"  ID: {oid} | Type: {otype} | Side: {oside} | PosSide: {pside} | Trigger: ${trigger}")

# Cancel the wrong TP at $342
for o in orders:
    if float(o.get("stopPrice", 0)) < 500:
        print(f"\nCancelling wrong TP order {o.get('orderId')} at ${o.get('stopPrice')}")
        try:
            client.cancel_order("BNBUSDT", o["orderId"], is_algo_order=True)
            print("  Cancelled successfully!")
        except Exception as e:
            print(f"  Error: {e}")

# Place correct TP
# For SHORT at $875.92, TP at 8% profit = $875.92 * (1 - 0.08/20) = $872.40
correct_tp = round(875.92 * (1 - 0.08/20), 2)
print(f"\nPlacing correct TP at ${correct_tp} for BNBUSDT SHORT")
try:
    result = client.place_take_profit_order("BNBUSDT", "BUY", 0.23, correct_tp, position_side="SHORT")
    print(f"  TP placed: {result}")
except Exception as e:
    print(f"  Error: {e}")
