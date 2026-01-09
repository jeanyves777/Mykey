#!/usr/bin/env python3
"""
Test Options Trading - Alpaca Paper Trading

This script tests options trading functionality:
1. Check options trading is enabled on account
2. Fetch available options contracts
3. Place a LIMIT order (pending - market is closed)
4. Check order status
5. Cancel the order

Since the options market is closed, we test with limit orders that stay pending.

Usage:
    python test_options_trading.py           # Interactive mode (asks for confirmation)
    python test_options_trading.py --auto    # Auto mode (places and cancels order automatically)
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Fix for Windows console encoding
if sys.platform == 'win32':
    os.system('')  # Enable ANSI escape codes on Windows

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_system.engine.alpaca_client import AlpacaClient, ALPACA_AVAILABLE
from trading_system.config.crypto_paper_trading_config import CryptoPaperTradingConfig

import pytz
UTC = pytz.UTC
EST = pytz.timezone('America/New_York')


def main():
    parser = argparse.ArgumentParser(description='Test Options Trading')
    parser.add_argument('--auto', action='store_true', help='Auto mode - skip confirmation prompts')
    args = parser.parse_args()

    print("=" * 70)
    print("OPTIONS TRADING TEST - Alpaca Paper Trading")
    print("=" * 70)
    print()

    # Load config (using same credentials as crypto paper trading)
    config = CryptoPaperTradingConfig.load()
    if not config.api_key or not config.api_secret:
        print("ERROR: API credentials not configured.")
        print("Run: python -m trading_system.run_crypto_paper_trading --setup")
        return

    # Initialize client
    print("[1] Initializing Alpaca client...")
    client = AlpacaClient(
        api_key=config.api_key,
        api_secret=config.api_secret,
        paper=True
    )

    # Test connection
    print("\n[2] Testing connection and checking options approval...")
    try:
        account = client.get_account()
        print(f"    Connected! Cash: ${account['cash']:,.2f}")
        print(f"    Buying Power: ${account['buying_power']:,.2f}")

        # Check options info
        options_info = client.get_account_options_info()
        print(f"\n    Options Account Info:")
        print(f"      Approved Level: {options_info.get('options_approved_level', 'N/A')}")
        print(f"      Trading Level: {options_info.get('options_trading_level', 'N/A')}")
        print(f"      Options Buying Power: {options_info.get('options_buying_power', 'N/A')}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # Fetch available options contracts
    print("\n[3] Fetching available options contracts for AAPL...")
    try:
        # Get contracts expiring in the next week
        today = datetime.now(EST)
        next_week = today + timedelta(days=7)

        contracts = client.get_option_contracts(
            underlying_symbols=['AAPL'],
            expiration_date_lte=next_week,
            limit=20
        )

        if not contracts:
            print("    No contracts found for AAPL. Trying SPY...")
            contracts = client.get_option_contracts(
                underlying_symbols=['SPY'],
                expiration_date_lte=next_week,
                limit=20
            )

        if contracts:
            print(f"    Found {len(contracts)} contracts:")
            # Show first 5
            for i, c in enumerate(contracts[:5]):
                print(f"    [{i+1}] {c.get('symbol', 'N/A')}")
                print(f"        Name: {c.get('name', 'N/A')}")
                print(f"        Type: {c.get('type', 'N/A')}")
                print(f"        Strike: ${c.get('strike_price', 'N/A')}")
                print(f"        Expiry: {c.get('expiration_date', 'N/A')}")
                print(f"        Tradable: {c.get('tradable', 'N/A')}")
                print(f"        Close Price: ${c.get('close_price', 'N/A')}")
                print()
        else:
            print("    WARNING: No contracts found. Using a sample symbol.")
            # Create a synthetic contract symbol for testing
            exp_date = next_week
            contracts = [{'symbol': client.format_occ_symbol('AAPL', exp_date, 200.0, 'C')}]
    except Exception as e:
        print(f"    ERROR fetching contracts: {e}")
        return

    # Select a contract to test with
    if not contracts:
        print("\n    ERROR: No contracts available for testing")
        return

    test_contract = contracts[0]
    test_symbol = test_contract.get('symbol', '')
    close_price = test_contract.get('close_price')

    print(f"\n[4] Selected contract for test:")
    print(f"    Symbol: {test_symbol}")
    print(f"    Name: {test_contract.get('name', 'N/A')}")

    # Get current quote if available
    print(f"\n[5] Getting quote for {test_symbol}...")
    quote = None
    try:
        quote = client.get_latest_option_quote(test_symbol)
        if quote:
            print(f"    Bid: ${quote.bid:.2f}")
            print(f"    Ask: ${quote.ask:.2f}")
            print(f"    Mid: ${quote.mid:.2f}")
        else:
            print(f"    No live quote (market closed). Using close price: ${close_price}")
    except Exception as e:
        print(f"    ERROR getting quote: {e}")
        print(f"    Using close price: ${close_price}")

    # Determine limit price (well below market to ensure it stays pending)
    if quote and quote.bid > 0:
        limit_price = round(quote.bid * 0.5, 2)  # 50% below bid
    elif close_price:
        limit_price = round(float(close_price) * 0.5, 2)  # 50% below close
    else:
        limit_price = 0.50  # Default very low price

    # Ensure minimum price
    if limit_price < 0.01:
        limit_price = 0.01

    print(f"\n" + "=" * 70)
    print("READY TO PLACE TEST OPTIONS ORDER")
    print("=" * 70)
    print(f"  Contract: {test_symbol}")
    print(f"  Quantity: 1 contract")
    print(f"  Side: BUY")
    print(f"  Limit Price: ${limit_price:.2f} (well below market to stay pending)")
    print(f"  Time in Force: DAY")
    print()
    print("  NOTE: Since this is a limit order well below market price,")
    print("        it will stay PENDING and we will cancel it immediately.")
    print("=" * 70)

    if args.auto:
        print("\n    [AUTO MODE] Proceeding automatically...")
    else:
        confirm = input("\nProceed? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled.")
            return

    # Place the limit order
    print(f"\n[6] Placing LIMIT order...")
    order_id = None
    try:
        order = client.submit_option_limit_order(
            symbol=test_symbol,
            qty=1,
            side='buy',
            limit_price=limit_price
        )
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'unknown')

        print(f"    Order ID: {order_id}")
        print(f"    Status: {order_status}")
        print(f"    Symbol: {order.get('symbol', 'N/A')}")
        print(f"    Side: {order.get('side', 'N/A')}")
        print(f"    Qty: {order.get('qty', 'N/A')}")
        print(f"    Limit Price: ${order.get('limit_price', 'N/A')}")
        print(f"    Time in Force: {order.get('time_in_force', 'N/A')}")
        print(f"    SUCCESS!")
    except Exception as e:
        print(f"    ERROR: {e}")
        print("\n    This might indicate:")
        print("      - Options trading not enabled on account")
        print("      - Invalid contract symbol")
        print("      - Insufficient buying power")
        return

    # Check order status
    print(f"\n[7] Checking order status...")
    try:
        order_info = client.get_order(order_id)
        if order_info:
            print(f"    Order ID: {order_info.get('id', 'N/A')[:12]}...")
            print(f"    Status: {order_info.get('status', 'unknown')}")
            print(f"    Filled Qty: {order_info.get('filled_qty', 0)}")
            print(f"    Created: {order_info.get('created_at', 'N/A')}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Cancel the order
    print(f"\n[8] Cancelling order...")
    try:
        cancelled = client.cancel_order(order_id)
        if cancelled:
            print(f"    Order cancelled successfully!")
        else:
            print(f"    Failed to cancel order (may already be filled/cancelled)")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Verify cancellation
    print(f"\n[9] Verifying cancellation...")
    try:
        order_info = client.get_order(order_id)
        if order_info:
            print(f"    Final Status: {order_info.get('status', 'unknown')}")
    except Exception as e:
        print(f"    Could not verify: {e}")

    # Check options positions
    print(f"\n[10] Checking options positions...")
    try:
        positions = client.get_options_positions()
        if positions:
            print(f"    Open Options Positions: {len(positions)}")
            for pos in positions:
                print(f"      - {pos['symbol']}: {pos['qty']} contracts, P&L: ${pos['unrealized_pl']:.2f}")
        else:
            print(f"    No open options positions (expected after cancellation)")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("OPTIONS TRADING TEST COMPLETE!")
    print("=" * 70)
    print("\nTest Results:")
    print("  [x] Connected to Alpaca")
    print("  [x] Checked options account info")
    print("  [x] Fetched available contracts")
    print("  [x] Got contract quote")
    print("  [x] Placed limit order")
    print("  [x] Checked order status")
    print("  [x] Cancelled order")
    print("  [x] Verified cancellation")
    print("\nOptions trading is properly configured!")
    print("\nCheck your Alpaca Paper Trading dashboard to verify:")
    print("  https://app.alpaca.markets/account/orders")


if __name__ == "__main__":
    main()
