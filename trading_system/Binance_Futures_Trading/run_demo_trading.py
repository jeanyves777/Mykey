#!/usr/bin/env python3
"""
Run Binance Futures DEMO Trading
================================
Live trading on Binance DEMO (demo.binance.com)
Places REAL orders on the DEMO account - no real money at risk.
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.live_trading_engine import BinanceLiveTradingEngine
from config.trading_config import FUTURES_SYMBOLS_LIVE, BINANCE_CONFIG

# Use LIVE symbols (BTC, ETH, BNB) - the best performers
FUTURES_SYMBOLS = FUTURES_SYMBOLS_LIVE


def main():
    parser = argparse.ArgumentParser(
        description="Binance Futures DEMO Trading (demo.binance.com)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_demo_trading.py                     # Run CONTINUOUSLY on DEMO
    python run_demo_trading.py --hours 8           # Run for 8 hours only
    python run_demo_trading.py --no-mtf            # Single timeframe mode
    python run_demo_trading.py -y                  # Skip confirmation
        """
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=0,
        help="Duration in hours (0 = run continuously, default: 0)"
    )

    parser.add_argument(
        "--no-mtf",
        action="store_true",
        help="Disable multi-timeframe confirmation"
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Display configuration
    print("\n" + "="*60)
    print("BINANCE FUTURES DEMO TRADING")
    print("="*60)
    print("Mode: DEMO (demo.binance.com) - NO REAL MONEY")
    print(f"Duration: {'CONTINUOUS (24/7)' if args.hours == 0 else f'{args.hours} hours'}")
    print(f"Signals: {'Single Timeframe' if args.no_mtf else 'Multi-Timeframe'}")
    print(f"Symbols: {', '.join(FUTURES_SYMBOLS)}")
    print(f"View positions: https://demo.binance.com/en/futures")
    print("="*60)

    # Confirm
    if not args.yes:
        confirm = input("\nStart DEMO trading? (y/n): ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled")
            return

    # Run demo trading (testnet=True means demo mode)
    engine = BinanceLiveTradingEngine(
        testnet=True,  # DEMO mode
        use_mtf=not args.no_mtf
    )

    # Engine handles KeyboardInterrupt internally with session summary
    engine.run(duration_hours=args.hours)


if __name__ == "__main__":
    main()
