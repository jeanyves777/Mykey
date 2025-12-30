#!/usr/bin/env python3
"""
Run Binance Futures Live Trading
================================
REAL trading with REAL money on Binance Futures

WARNING: This will place actual orders on your Binance account!
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.live_trading_engine import BinanceLiveTradingEngine
from config.trading_config import FUTURES_SYMBOLS_LIVE, FUTURES_SYMBOLS_DEMO, BINANCE_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Binance Futures Live Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_live_trading.py                     # Run CONTINUOUSLY on TESTNET
    python run_live_trading.py --hours 8           # Run for 8 hours only
    python run_live_trading.py --live              # REAL MONEY MODE (DANGEROUS!)
    python run_live_trading.py --live -y           # Skip confirmation (NOT RECOMMENDED)
        """
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=0,
        help="Duration in hours (0 = run continuously, default: 0)"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Use mainnet (REAL MONEY - USE WITH CAUTION!)"
    )

    parser.add_argument(
        "--no-mtf",
        action="store_true",
        help="Disable multi-timeframe confirmation"
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt (NOT RECOMMENDED for live trading)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip ALL confirmations (for VPS/automated deployment)"
    )

    args = parser.parse_args()

    # Safety check for live mode (skip if --force is used)
    if args.live and args.yes and not args.force:
        print("\n" + "!"*60)
        print("! DANGER: You are about to start LIVE trading with -y flag !")
        print("! This will place REAL orders without confirmation!         !")
        print("!"*60)
        final_confirm = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
        if final_confirm != "I UNDERSTAND THE RISKS":
            print("Cancelled for safety")
            return

    # Display configuration
    mode = "MAINNET (REAL MONEY!)" if args.live else "TESTNET"

    print("\n" + "="*60)
    print(f"BINANCE FUTURES {mode} TRADING")
    print("="*60)

    if args.live:
        print("\n" + "!"*40)
        print("! WARNING: REAL MONEY MODE ACTIVATED !")
        print("! This will place REAL orders!       !")
        print("!"*40 + "\n")

    # Select symbols based on mode
    symbols = FUTURES_SYMBOLS_LIVE if args.live else FUTURES_SYMBOLS_DEMO

    print(f"Duration: {'CONTINUOUS (24/7)' if args.hours == 0 else f'{args.hours} hours'}")
    print(f"Mode: {'Single Timeframe' if args.no_mtf else 'Multi-Timeframe'}")
    print(f"Symbols: {', '.join(symbols)}")
    print("="*60)

    # Confirm
    if not args.yes:
        if args.live:
            confirm = input("\n!!! Are you SURE you want to trade with REAL MONEY? (type 'YES' in caps): ")
            if confirm != "YES":
                print("Cancelled for safety")
                return
        else:
            confirm = input("\nStart testnet trading? (y/n): ")
            if confirm.lower() not in ("y", "yes"):
                print("Cancelled")
                return

    # Run live trading
    engine = BinanceLiveTradingEngine(
        testnet=not args.live,
        use_mtf=not args.no_mtf
    )

    try:
        engine.run(duration_hours=args.hours)
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        engine.stop()


if __name__ == "__main__":
    main()
