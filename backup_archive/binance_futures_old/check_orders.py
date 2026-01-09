#!/usr/bin/env python3
"""Check all pending orders on Binance LIVE."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient

# Connect to LIVE
client = BinanceClient(testnet=False, use_demo=False)

print("=" * 70)
print("BINANCE LIVE - PENDING ORDERS")
print("=" * 70)

# Get all open orders for each symbol
for symbol in ["DOTUSDT", "AVAXUSDT"]:
    orders = client.get_open_orders(symbol)
    print(f"\n{symbol}: {len(orders)} orders")

    for order in orders:
        order_id = order.get("orderId")
        order_type = order.get("type")
        side = order.get("side")
        position_side = order.get("positionSide", "BOTH")
        stop_price = float(order.get("stopPrice", 0))
        qty = float(order.get("origQty", 0))
        status = order.get("status")
        time_str = order.get("time", "")

        # Determine if TP or SL
        label = ""
        if order_type == "TAKE_PROFIT_MARKET":
            label = "TP"
        elif order_type == "STOP_MARKET":
            label = "SL"
        else:
            label = order_type

        print(f"  [{position_side:5}] {label:3} {side:4} qty={qty:.4f} @ ${stop_price:,.4f} | ID: {order_id}")

# Count totals
total_orders = 0
for symbol in ["DOTUSDT", "AVAXUSDT"]:
    orders = client.get_open_orders(symbol)
    total_orders += len(orders)

print(f"\n{'='*70}")
print(f"TOTAL PENDING ORDERS: {total_orders}")
print(f"EXPECTED: 8 (4 positions x 2 orders each)")
print(f"EXCESS: {total_orders - 8}")
print(f"{'='*70}")
