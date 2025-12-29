#!/usr/bin/env python3
"""
BTCC API Connection Test
========================
Test connection to BTCC Futures API.

Usage:
    python test_btcc_connection.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.btcc_api_client import BTCCAPIClient


def test_connection():
    """Test BTCC API connection without login."""
    print("\n" + "=" * 60)
    print("BTCC API CONNECTION TEST")
    print("=" * 60)

    # API credentials
    api_key = "2cc78c86-cf78-47a4-ad38-ca1c0e9b8517"
    secret_key = "6697e175-4ec6-4748-99cf-0ce4ac92790b"

    print(f"\nAPI Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Secret Key: {secret_key[:8]}...{secret_key[-4:]}")

    # Create client
    client = BTCCAPIClient(
        api_key=api_key,
        secret_key=secret_key,
    )

    # Test signature generation
    print("\n1. Testing signature generation...")
    test_params = {
        'accountid': 142,
        'token': '00000011391572680255209',
    }
    signature = client._generate_signature(test_params)
    print(f"   Generated signature: {signature}")
    print("   [OK] Signature generation working")

    # Note: Full login requires username/password
    print("\n2. Login test (requires credentials)...")
    print("   [SKIP] Username and password not configured")
    print("   To test login, edit this file and add your credentials")

    print("\n" + "=" * 60)
    print("CONNECTION TEST COMPLETE")
    print("=" * 60)
    print("\nTo fully test the API:")
    print("1. Set USER_NAME and PASSWORD in btcc_config.py")
    print("2. Run: python run_btcc_paper_trading.py (for paper trading)")
    print("3. Run: python run_btcc_live_trading.py (for live trading)")
    print("=" * 60)


def test_full_connection(user_name: str, password: str):
    """Test full BTCC connection with login."""
    print("\n" + "=" * 60)
    print("BTCC FULL CONNECTION TEST")
    print("=" * 60)

    api_key = "2cc78c86-cf78-47a4-ad38-ca1c0e9b8517"
    secret_key = "6697e175-4ec6-4748-99cf-0ce4ac92790b"

    client = BTCCAPIClient(
        api_key=api_key,
        secret_key=secret_key,
        user_name=user_name,
        password=password,
    )

    # Test login
    print("\n1. Testing login...")
    result = client.login()

    if result.get('code') == 0:
        print(f"   [OK] Login successful!")
        print(f"   Account ID: {client.account_id}")
        print(f"   Account No: {client.account_no}")

        # Get account info
        print("\n2. Getting account info...")
        account = client.get_account()
        if account.get('code') == 0:
            acc = account.get('account', {})
            print(f"   Balance: ${acc.get('balance', 0):,.2f}")
            print(f"   Equity: ${acc.get('equity', 0):,.2f}")
            print(f"   Free Margin: ${acc.get('free_margin', 0):,.2f}")
            print(f"   Positions: {acc.get('positions', 0)}")
            print("   [OK] Account info retrieved")
        else:
            print(f"   [ERROR] {account.get('msg')}")

        # Get symbols
        print("\n3. Getting available symbols...")
        symbols = client.get_symbols()
        if symbols.get('code') == 0:
            symbol_list = symbols.get('symbols', [])
            print(f"   Found {len(symbol_list)} symbols")
            for s in symbol_list[:5]:  # Show first 5
                print(f"   - {s.get('name')}")
            if len(symbol_list) > 5:
                print(f"   ... and {len(symbol_list) - 5} more")
            print("   [OK] Symbols retrieved")
        else:
            print(f"   [ERROR] {symbols.get('msg')}")

        # Get positions
        print("\n4. Getting positions...")
        positions = client.get_positions()
        if positions.get('code') == 0:
            pos_list = positions.get('positions', [])
            open_pos = [p for p in pos_list if p.get('status') == 1]
            print(f"   Open positions: {len(open_pos)}")
            for p in open_pos[:3]:
                direction = "LONG" if p.get('direction') == 1 else "SHORT"
                print(f"   - {p.get('symbol')}: {direction} x{p.get('volume')}")
            print("   [OK] Positions retrieved")
        else:
            print(f"   [ERROR] {positions.get('msg')}")

        # Disconnect
        print("\n5. Disconnecting...")
        client.disconnect()
        print("   [OK] Disconnected")

    else:
        print(f"   [ERROR] Login failed: {result.get('msg')}")

    print("\n" + "=" * 60)
    print("FULL CONNECTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Full test with credentials
        test_full_connection(sys.argv[1], sys.argv[2])
    else:
        # Basic test without login
        test_connection()

        print("\nTo run full test with login:")
        print("  python test_btcc_connection.py YOUR_EMAIL YOUR_PASSWORD")
