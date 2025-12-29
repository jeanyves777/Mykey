#!/usr/bin/env python3
"""Cancel all pending orders on Binance LIVE."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient

# Connect to LIVE
client = BinanceClient(testnet=False, use_demo=False)

print("=" * 70)
print("BINANCE LIVE - CANCEL ALL PENDING ORDERS")
print("=" * 70)

symbols = ["DOTUSDT", "AVAXUSDT"]

# First show what we have
total_orders = 0
for symbol in symbols:
    orders = client.get_open_orders(symbol)
    total_orders += len(orders)
    print(f"\n{symbol}: {len(orders)} orders")

    for order in orders:
        order_id = order.get("orderId")
        order_type = order.get("type")
        side = order.get("side")
        position_side = order.get("positionSide", "BOTH")
        stop_price = float(order.get("stopPrice", 0))
        qty = float(order.get("origQty", 0))

        label = ""
        if order_type == "TAKE_PROFIT_MARKET":
            label = "TP"
        elif order_type == "STOP_MARKET":
            label = "SL"
        else:
            label = order_type

        print(f"  [{position_side:5}] {label:3} {side:4} qty={qty:.4f} @ ${stop_price:,.4f} | ID: {order_id}")

print(f"\n{'='*70}")
print(f"TOTAL PENDING ORDERS: {total_orders}")
print(f"{'='*70}")

if total_orders == 0:
    print("\nNo orders to cancel!")
    sys.exit(0)

# Confirm cancellation
confirm = input(f"\nCancel all {total_orders} orders? (y/n): ")
if confirm.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

# Cancel all orders
print("\nCancelling orders...")
for symbol in symbols:
    try:
        result = client.cancel_all_orders(symbol)
        print(f"  {symbol}: Cancelled all orders - {result}")
    except Exception as e:
        print(f"  {symbol}: Error - {e}")

# Verify
print("\n" + "=" * 70)
print("VERIFICATION - Remaining orders:")
print("=" * 70)

remaining = 0
for symbol in symbols:
    orders = client.get_open_orders(symbol)
    remaining += len(orders)
    print(f"  {symbol}: {len(orders)} orders")

print(f"\n{'='*70}")
print(f"REMAINING ORDERS: {remaining}")
print(f"{'='*70}")

if remaining == 0:
    print("\nAll orders cancelled successfully!")
    print("Now restart live_trading.py to place correct SL/TP orders with proper positionSide.")
