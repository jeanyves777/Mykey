#!/usr/bin/env python3
"""
Asian Session Portfolio Scalper - Run Script
=============================================
Trades 15 currency pairs simultaneously during Asian session
Uses OANDA Account: 101-001-8364309-002

Usage:
    python run_scalper.py           # Run continuously
    python run_scalper.py --once    # Run single session
    python run_scalper.py --test    # Test connection only
    python run_scalper.py --status  # Show account status
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.asian_scalper_engine import AsianScalperEngine
from engine.oanda_client import AsianScalperOANDAClient
from config import SCALP_PAIRS, SCALP_CONFIG


def test_connection():
    """Test OANDA connection and show account info."""
    print("=" * 60)
    print("ASIAN SCALPER - CONNECTION TEST")
    print("=" * 60)

    client = AsianScalperOANDAClient(practice=True)

    if client.test_connection():
        print("\n[SUCCESS] Connected to OANDA")

        summary = client.get_account_summary()
        print(f"\nAccount ID: {client.account_id}")
        print(f"Balance: ${float(summary.get('balance', 0)):,.2f}")
        print(f"NAV: ${float(summary.get('NAV', 0)):,.2f}")
        print(f"Unrealized P/L: ${float(summary.get('unrealizedPL', 0)):,.2f}")
        print(f"Margin Available: ${float(summary.get('marginAvailable', 0)):,.2f}")

        # Check open positions
        trades = client.get_open_trades()
        print(f"\nOpen Trades: {len(trades)}")

        if trades:
            print("\nOpen Positions:")
            for trade in trades:
                pl = trade["unrealized_pl"]
                symbol = "+" if pl >= 0 else ""
                print(f"  {trade['instrument']}: {trade['units']} units @ {trade['price']:.5f} | P/L: {symbol}${pl:.2f}")

        return True
    else:
        print("\n[ERROR] Failed to connect to OANDA")
        return False


def show_status():
    """Show current account and position status."""
    print("=" * 60)
    print("ASIAN SCALPER - ACCOUNT STATUS")
    print("=" * 60)

    client = AsianScalperOANDAClient(practice=True)

    if not client.test_connection():
        print("[ERROR] Cannot connect to OANDA")
        return

    summary = client.get_account_summary()
    trades = client.get_open_trades()

    print(f"\nAccount: {client.account_id}")
    print(f"Balance: ${float(summary.get('balance', 0)):,.2f}")
    print(f"Unrealized P/L: ${float(summary.get('unrealizedPL', 0)):,.2f}")
    print(f"Portfolio Target: ${SCALP_CONFIG['portfolio_target']}")

    print(f"\n{'=' * 40}")
    print(f"OPEN POSITIONS ({len(trades)})")
    print(f"{'=' * 40}")

    if trades:
        total_pl = 0
        for trade in trades:
            pl = trade["unrealized_pl"]
            total_pl += pl
            symbol = "+" if pl >= 0 else ""
            direction = "LONG" if trade["units"] > 0 else "SHORT"
            print(f"{trade['instrument']:10} | {direction:5} | {abs(trade['units']):>8.0f} units | P/L: {symbol}${pl:>7.2f}")

        print(f"{'=' * 40}")
        symbol = "+" if total_pl >= 0 else ""
        print(f"{'TOTAL':10} | {'':5} | {'':>14} | P/L: {symbol}${total_pl:>7.2f}")

        # Progress to target
        target = SCALP_CONFIG['portfolio_target']
        progress = (total_pl / target) * 100 if target > 0 else 0
        print(f"\nProgress to ${target} target: {progress:.1f}%")
    else:
        print("No open positions")

    print(f"\n{'=' * 40}")
    print("CONFIGURED PAIRS")
    print(f"{'=' * 40}")
    for i, pair in enumerate(SCALP_PAIRS, 1):
        print(f"  {i:2}. {pair}")


def main():
    """Main entry point with CLI argument handling."""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == "--test":
            test_connection()
            return

        elif arg == "--status":
            show_status()
            return

        elif arg == "--once":
            print("Running single session...")
            engine = AsianScalperEngine()
            engine.run_session()
            return

        elif arg == "--help" or arg == "-h":
            print(__doc__)
            return

        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            return

    # Default: run continuously
    print("Starting Asian Scalper in continuous mode...")
    print("Use Ctrl+C to stop")
    print()

    engine = AsianScalperEngine()
    engine.run_continuous()


if __name__ == "__main__":
    main()
