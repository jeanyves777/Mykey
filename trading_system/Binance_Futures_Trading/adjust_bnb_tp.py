#!/usr/bin/env python3
"""
Adjust BNB position TP from 40% ROI to 30% ROI
Run on VPS: python adjust_bnb_tp.py
"""
import sys
sys.path.insert(0, '.')
from engine.binance_client import BinanceClient

def main():
    client = BinanceClient(testnet=False)

    # Get current BNB position
    print("Fetching BNB position...")
    positions = client.get_account_info().get('positions', [])
    bnb_pos = next((p for p in positions if p['symbol'] == 'BNBUSDT' and float(p['positionAmt']) != 0), None)

    if not bnb_pos:
        print("No BNB position found")
        return

    pos_amt = float(bnb_pos['positionAmt'])
    entry_price = float(bnb_pos['entryPrice'])
    print(f"BNB Position: {pos_amt} @ entry ${entry_price:.2f}")

    # Check current orders
    print("\nCurrent orders:")
    algo_orders = client.get_algo_orders('BNBUSDT')
    regular_orders = client.get_open_orders('BNBUSDT')

    tp_exists = False
    for order in algo_orders:
        trigger = order.get('triggerPrice', order.get('stopPrice', 'N/A'))
        print(f"  ALGO: {order.get('type')} @ {trigger}")
        if 'TAKE_PROFIT' in order.get('type', ''):
            tp_exists = True

    for order in regular_orders:
        print(f"  Regular: {order.get('type')} @ {order.get('stopPrice', order.get('price', 'N/A'))}")
        if 'TAKE_PROFIT' in order.get('type', ''):
            tp_exists = True

    # New TP at 30% ROI = 1.5% price move (with 20x leverage)
    # LONG: TP = entry * (1 + 0.30/20) = entry * 1.015
    tp_price = entry_price * 1.015
    print(f"\nNew TP price (30% ROI): ${tp_price:.2f}")

    if tp_exists:
        print("WARNING: TP order already exists! Please cancel it first if you want to change.")
        return

    # Place TP order - SELL to close LONG position
    print("\nPlacing new TP order...")
    result = client.place_take_profit(
        symbol='BNBUSDT',
        side='SELL',
        quantity=abs(pos_amt),
        take_profit_price=tp_price,
        position_side='LONG'
    )
    print(f"Result: {result}")

    if result and (result.get('algoId') or result.get('orderId')):
        print("\n✓ TP order placed successfully!")
    else:
        print("\n✗ Failed to place TP order")

if __name__ == "__main__":
    main()
