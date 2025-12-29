#!/usr/bin/env python3
"""
Check Binance Futures Account Balance
=====================================
Find any isolated margin balances that might be blocking Hedge Mode switch.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient


def main():
    print("=" * 60)
    print("BINANCE FUTURES ACCOUNT BALANCE CHECK")
    print("=" * 60)

    # Connect to Binance Demo
    client = BinanceClient(testnet=True, use_demo=True)

    # Get account info
    print("\n--- Account Info ---")
    try:
        account = client.get_account_info()
        print(f"Total Wallet Balance: ${float(account.get('totalWalletBalance', 0)):,.2f}")
        print(f"Available Balance: ${float(account.get('availableBalance', 0)):,.2f}")
        print(f"Total Unrealized PnL: ${float(account.get('totalUnrealizedProfit', 0)):,.2f}")

        # Check all assets
        assets = account.get('assets', [])
        print(f"\n--- Assets with Balance ---")
        for asset in assets:
            wallet_balance = float(asset.get('walletBalance', 0))
            margin_balance = float(asset.get('marginBalance', 0))
            available_balance = float(asset.get('availableBalance', 0))

            if wallet_balance != 0 or margin_balance != 0:
                print(f"\n{asset['asset']}:")
                print(f"  Wallet Balance: {wallet_balance}")
                print(f"  Margin Balance: {margin_balance}")
                print(f"  Available Balance: {available_balance}")
                print(f"  Cross Wallet Balance: {asset.get('crossWalletBalance', 0)}")
                print(f"  Cross UnPnL: {asset.get('crossUnPnl', 0)}")

        # Check positions for isolated margin
        print(f"\n--- Positions with Isolated Margin ---")
        positions = account.get('positions', [])
        found_isolated = False
        for pos in positions:
            isolated = float(pos.get('isolatedWallet', 0))
            if isolated != 0:
                found_isolated = True
                print(f"\n{pos['symbol']}:")
                print(f"  Position Amt: {pos.get('positionAmt', 0)}")
                print(f"  Isolated Wallet: {isolated}")
                print(f"  Isolated Margin: {pos.get('isolatedMargin', 0)}")
                print(f"  Entry Price: {pos.get('entryPrice', 0)}")
                print(f"  Unrealized PnL: {pos.get('unrealizedProfit', 0)}")
                print(f"  Position Side: {pos.get('positionSide', 'BOTH')}")

        if not found_isolated:
            print("  No isolated margin positions found.")

    except Exception as e:
        print(f"[ERROR] {e}")

    # Check position mode
    print(f"\n--- Position Mode ---")
    try:
        mode = client.get_position_mode()
        is_hedge = mode.get("dualSidePosition", False)
        print(f"Current Mode: {'HEDGE (Dual Side)' if is_hedge else 'ONE-WAY'}")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Get positions directly
    print(f"\n--- Open Positions ---")
    positions = client.get_positions()
    if not positions:
        print("  No open positions.")
    else:
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['side']} | Qty: {pos['quantity']:.6f}")


if __name__ == "__main__":
    main()
