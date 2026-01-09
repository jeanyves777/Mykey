#!/usr/bin/env python3
"""
Clear Negative Isolated Margin
==============================
Finds and clears any negative isolated margin balances.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient


def main():
    print("=" * 60)
    print("CLEAR NEGATIVE ISOLATED MARGIN")
    print("=" * 60)

    # Connect to Binance Demo
    client = BinanceClient(testnet=True, use_demo=True)

    # Get account info
    account = client.get_account_info()
    positions = account.get('positions', [])

    negative_found = []
    for pos in positions:
        isolated_wallet = float(pos.get('isolatedWallet', 0))
        if isolated_wallet < 0:
            negative_found.append({
                'symbol': pos['symbol'],
                'isolated_wallet': isolated_wallet,
                'position_side': pos.get('positionSide', 'BOTH')
            })

    if not negative_found:
        print("\nNo negative isolated margin found!")
        print("Trying to enable Hedge Mode...")

        try:
            result = client.set_position_mode(True)
            if "code" in result:
                print(f"[ERROR] {result}")
            else:
                print("[OK] Hedge Mode enabled!")
        except Exception as e:
            if "-4059" in str(e):
                print("[OK] Hedge Mode already enabled!")
            else:
                print(f"[ERROR] {e}")

        # Verify
        mode = client.get_position_mode()
        is_hedge = mode.get("dualSidePosition", False)
        print(f"\nCurrent Position Mode: {'HEDGE (Dual Side)' if is_hedge else 'ONE-WAY'}")
        return

    print(f"\nFound {len(negative_found)} position(s) with negative isolated margin:")
    for item in negative_found:
        print(f"  {item['symbol']}: {item['isolated_wallet']:.8f} USDT (side: {item['position_side']})")

    print("\nAttempting to add margin to clear negative balances...")

    for item in negative_found:
        symbol = item['symbol']
        amount = abs(item['isolated_wallet']) + 0.01  # Add a little extra
        position_side = item['position_side']

        print(f"\n  Adding {amount:.4f} USDT to {symbol} ({position_side})...")

        try:
            result = client.modify_isolated_position_margin(
                symbol=symbol,
                amount=amount,
                add=True,
                position_side=position_side
            )
            if "code" in result and result.get("code") != 200:
                print(f"    [ERROR] {result}")

                # Try switching margin type to CROSS first
                print(f"    Trying to switch {symbol} to CROSS margin...")
                try:
                    margin_result = client.set_margin_type(symbol, "CROSSED")
                    print(f"    {margin_result}")
                except Exception as e:
                    if "No need to change" in str(e):
                        print(f"    Already in CROSS mode")
                    else:
                        print(f"    {e}")
            else:
                print(f"    [OK] Added margin: {result}")
        except Exception as e:
            print(f"    [ERROR] {e}")

            # Try switching margin type
            print(f"    Trying to switch {symbol} to CROSS margin...")
            try:
                margin_result = client.set_margin_type(symbol, "CROSSED")
                print(f"    {margin_result}")
            except Exception as e2:
                if "No need to change" in str(e2):
                    print(f"    Already in CROSS mode")
                else:
                    print(f"    {e2}")

    # Check again
    print("\n" + "=" * 60)
    print("Checking balance again...")
    print("=" * 60)

    account = client.get_account_info()
    positions = account.get('positions', [])

    still_negative = False
    for pos in positions:
        isolated_wallet = float(pos.get('isolatedWallet', 0))
        if isolated_wallet < 0:
            still_negative = True
            print(f"  {pos['symbol']}: {isolated_wallet:.8f} USDT - STILL NEGATIVE")

    if not still_negative:
        print("  All negative balances cleared!")

    # Try to enable Hedge Mode
    print("\n" + "=" * 60)
    print("Attempting to enable Hedge Mode...")
    print("=" * 60)

    try:
        result = client.set_position_mode(True)
        if "code" in result:
            if result.get("code") == 200:
                print("[OK] Hedge Mode enabled!")
            elif result.get("code") == -4059:
                print("[OK] Hedge Mode already enabled!")
            else:
                print(f"[ERROR] {result}")
        else:
            print("[OK] Hedge Mode enabled!")
    except Exception as e:
        if "-4059" in str(e):
            print("[OK] Hedge Mode already enabled!")
        else:
            print(f"[ERROR] {e}")

    # Re-sync and verify
    import time
    time.sleep(1)
    mode = client.get_position_mode()
    is_hedge = mode.get("dualSidePosition", False)
    print(f"\nCurrent Position Mode: {'HEDGE (Dual Side)' if is_hedge else 'ONE-WAY'}")

    if is_hedge:
        print("\nYou can now run: py run_demo_trading.py -y")


if __name__ == "__main__":
    main()
