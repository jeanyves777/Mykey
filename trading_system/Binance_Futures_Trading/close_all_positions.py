#!/usr/bin/env python3
"""
Close ALL Binance Futures positions to enable Hedge Mode
=========================================================
Hedge Mode (Dual Side Position) can only be enabled when there are NO open positions.
Run this script first, then run the demo trading bot.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient


def main():
    parser = argparse.ArgumentParser(description="Close all Binance Futures positions")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("CLOSE ALL BINANCE FUTURES POSITIONS")
    print("=" * 60)
    print("This will close ALL open positions on your DEMO account.")
    print("Required to enable Hedge Mode (Dual Side Position).")
    print("=" * 60)

    if not args.yes:
        confirm = input("\nType 'YES' to close all positions: ")
        if confirm != "YES":
            print("Cancelled.")
            return

    # Connect to Binance Demo
    client = BinanceClient(testnet=True, use_demo=True)

    # Get all positions
    positions = client.get_positions()

    if not positions:
        print("\nNo open positions found!")
    else:
        print(f"\nFound {len(positions)} open position(s):")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['side']} | Qty: {pos['quantity']:.6f} | Entry: ${pos['entry_price']:.4f}")

        print("\nClosing positions...")

        for pos in positions:
            symbol = pos['symbol']
            side = pos['side']
            quantity = pos['quantity']

            # To close: SELL if LONG, BUY if SHORT
            close_side = "SELL" if side == "LONG" else "BUY"

            print(f"  Closing {symbol} {side}...")

            # First cancel all orders for this symbol
            try:
                client.cancel_all_orders(symbol)
                print(f"    Cancelled all orders for {symbol}")
            except Exception as e:
                print(f"    Warning: Could not cancel orders: {e}")

            # Close the position
            try:
                result = client.place_market_order(symbol, close_side, quantity)
                if "orderId" in result:
                    print(f"    [OK] Closed {symbol} {side} position")
                else:
                    print(f"    [ERROR] Failed to close: {result}")
            except Exception as e:
                print(f"    [ERROR] {e}")

    # Always try to enable Hedge Mode
    print("\n" + "=" * 60)
    print("Enabling Hedge Mode...")
    print("=" * 60)

    # Try to enable Hedge Mode
    try:
        result = client.set_position_mode(True)
        if "code" in result:
            if result.get("code") == -4059:
                print("[OK] Hedge Mode already enabled!")
            else:
                print(f"[ERROR] Failed to enable Hedge Mode: {result}")
        else:
            print("[OK] Hedge Mode (Dual Side Position) enabled!")
    except Exception as e:
        error_str = str(e)
        if "-4059" in error_str or "No need to change" in error_str:
            print("[OK] Hedge Mode already enabled!")
        else:
            print(f"[ERROR] {e}")

    # Verify
    try:
        mode = client.get_position_mode()
        is_hedge = mode.get("dualSidePosition", False)
        print(f"\nCurrent Position Mode: {'HEDGE (Dual Side)' if is_hedge else 'ONE-WAY'}")
        if is_hedge:
            print("\nYou can now run: py run_demo_trading.py -y")
    except Exception as e:
        print(f"Could not verify mode: {e}")


if __name__ == "__main__":
    main()
