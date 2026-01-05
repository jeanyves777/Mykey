#!/usr/bin/env python3
"""
Close existing positions and restart with correct sizing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient
import time

def main():
    print("=" * 60)
    print("CLOSING POSITIONS AND VERIFYING ACCOUNT")
    print("=" * 60)

    # Initialize client for LIVE trading
    client = BinanceClient(testnet=False)

    # 1. Get all open positions
    print("\n[1] Checking ALL open positions...")
    positions = client.get_positions()
    open_positions = [p for p in positions if abs(float(p.get('positionAmt', 0))) > 0]

    if open_positions:
        print(f"Found {len(open_positions)} open position(s):")
        for pos in open_positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry = float(pos.get('entryPrice', 0))
            side = pos.get('positionSide', 'BOTH')
            margin = float(pos.get('isolatedMargin', 0))
            print(f"  - {symbol} {side}: {amt} @ ${entry:.4f} (margin: ${margin:.2f})")
    else:
        print("No open positions found.")

    # 2. Cancel ALL open orders
    print("\n[2] Cancelling ALL open orders...")
    try:
        # Get all open orders
        all_orders = client.client.futures_get_open_orders()
        if all_orders:
            print(f"Found {len(all_orders)} open order(s)")
            for order in all_orders:
                symbol = order['symbol']
                order_id = order['orderId']
                try:
                    client.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    print(f"  Cancelled order {order_id} on {symbol}")
                except Exception as e:
                    print(f"  Failed to cancel {order_id}: {e}")
        else:
            print("No open orders found.")
    except Exception as e:
        print(f"Error getting orders: {e}")

    # 3. Close ALL open positions
    print("\n[3] Closing ALL open positions...")
    for pos in open_positions:
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        position_side = pos.get('positionSide', 'BOTH')

        if amt == 0:
            continue

        # Determine close side
        if amt > 0:
            close_side = 'SELL'
            close_qty = abs(amt)
        else:
            close_side = 'BUY'
            close_qty = abs(amt)

        print(f"  Closing {symbol} {position_side}: {close_side} {close_qty}")

        try:
            # Close with market order
            order_params = {
                'symbol': symbol,
                'side': close_side,
                'type': 'MARKET',
                'quantity': close_qty,
                'reduceOnly': True
            }

            # Add positionSide for hedge mode
            if position_side in ['LONG', 'SHORT']:
                order_params['positionSide'] = position_side

            result = client.client.futures_create_order(**order_params)
            print(f"  ✓ Closed {symbol}: Order ID {result['orderId']}")
        except Exception as e:
            print(f"  ✗ Failed to close {symbol}: {e}")

    # 4. Wait and verify
    print("\n[4] Waiting 3 seconds for settlement...")
    time.sleep(3)

    # 5. Final verification
    print("\n[5] Final verification...")
    positions = client.get_positions()
    open_positions = [p for p in positions if abs(float(p.get('positionAmt', 0))) > 0]

    if open_positions:
        print(f"⚠ WARNING: Still have {len(open_positions)} open position(s):")
        for pos in open_positions:
            print(f"  - {pos['symbol']}: {pos['positionAmt']}")
    else:
        print("✓ All positions closed successfully!")

    all_orders = client.client.futures_get_open_orders()
    if all_orders:
        print(f"⚠ WARNING: Still have {len(all_orders)} open order(s)")
    else:
        print("✓ All orders cancelled successfully!")

    # 6. Show account balance
    print("\n[6] Account Balance:")
    try:
        account = client.client.futures_account()
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        print(f"  Total Balance: ${balance:.2f}")
        print(f"  Available: ${available:.2f}")
    except Exception as e:
        print(f"  Error getting balance: {e}")

    print("\n" + "=" * 60)
    print("READY TO RESTART WITH CORRECT SIZING")
    print("=" * 60)
    print("\nRun the following command to start trading:")
    print("python3 engine/htf_confluence_live_engine.py --symbols DOTUSDT BNBUSDT --capital 40 --live --no-confirm")

if __name__ == "__main__":
    main()
