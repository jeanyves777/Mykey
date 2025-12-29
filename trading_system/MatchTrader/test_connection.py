"""
Test Match-Trader Connection
============================
Quick test to verify API connection to your demo account.

Usage:
    python test_connection.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from match_trader_client import MatchTraderClient


def main():
    print("\n" + "=" * 60)
    print("  MATCH-TRADER CONNECTION TEST")
    print("=" * 60)

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    mt_config = config['match_trader']

    print(f"\nServer: {mt_config['base_url']}")
    print(f"Broker ID: {mt_config['broker_id']}")

    # Check if credentials are set
    if 'YOUR_' in mt_config['email'] or 'YOUR_' in mt_config['password']:
        print("\n" + "!" * 60)
        print("  PLEASE ENTER YOUR CREDENTIALS")
        print("!" * 60)

        email = input("\nEnter your Match-Trader email: ").strip()
        password = input("Enter your Match-Trader password: ").strip()

        if not email or not password:
            print("Email and password required!")
            return
    else:
        email = mt_config['email']
        password = mt_config['password']

    # Initialize client
    print("\n[1/5] Initializing client...")
    client = MatchTraderClient(
        base_url=mt_config['base_url'],
        broker_id=mt_config['broker_id']
    )

    # Login
    print("[2/5] Logging in...")
    if not client.login(email, password):
        print("\n ERROR: Login failed!")
        print("  - Check your email and password")
        print("  - Make sure you're using the correct broker ID")
        return

    print("  LOGIN SUCCESSFUL!")
    print(f"  System UUID: {client.system_uuid}")

    # Get balance
    print("\n[3/5] Getting account balance...")
    balance = client.get_balance()
    if balance:
        print(f"\n  +{'─' * 40}+")
        print(f"  │ {'ACCOUNT BALANCE':^38} │")
        print(f"  +{'─' * 40}+")
        print(f"  │ Balance:     ${balance.balance:>20,.2f}  │")
        print(f"  │ Equity:      ${balance.equity:>20,.2f}  │")
        print(f"  │ Free Margin: ${balance.margin_free:>20,.2f}  │")
        print(f"  │ P/L:         ${balance.profit_loss:>20,.2f}  │")
        print(f"  +{'─' * 40}+")
    else:
        print("  WARNING: Could not get balance")

    # Get quote
    print("\n[4/5] Getting market quote for EURUSD...")
    quote = client.get_quote("EURUSD")
    if quote:
        spread = (quote['ask'] - quote['bid']) * 10000  # pips
        print(f"\n  EURUSD Quote:")
        print(f"    Bid: {quote['bid']:.5f}")
        print(f"    Ask: {quote['ask']:.5f}")
        print(f"    Spread: {spread:.1f} pips")
    else:
        print("  WARNING: Could not get quote")

    # Get open positions
    print("\n[5/5] Checking open positions...")
    positions = client.get_open_positions()
    print(f"  Open positions: {len(positions)}")
    for pos in positions:
        print(f"    - {pos.side} {pos.volume} {pos.instrument} @ {pos.entry_price}")

    print("\n" + "=" * 60)
    print("  CONNECTION TEST COMPLETE!")
    print("=" * 60)

    # Save credentials to config if entered manually
    if 'YOUR_' in mt_config['email']:
        save = input("\nSave credentials to config.json? (y/n): ").strip().lower()
        if save == 'y':
            config['match_trader']['email'] = email
            config['match_trader']['password'] = password
            config['match_trader']['system_uuid'] = client.system_uuid

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print("  Credentials saved!")

    print("\nYou can now run: python run_fundednext.py")


if __name__ == "__main__":
    main()
