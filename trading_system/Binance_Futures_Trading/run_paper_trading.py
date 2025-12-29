#!/usr/bin/env python3
"""
Run Binance Futures Paper Trading
=================================
Paper trading with real-time market data (no real orders)
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.paper_trading_engine import BinancePaperTradingEngine
from config.trading_config import FUTURES_SYMBOLS, PAPER_TRADING_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Binance Futures Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_paper_trading.py                    # Run CONTINUOUSLY
    python run_paper_trading.py --hours 8          # Run for 8 hours only
    python run_paper_trading.py --no-mtf           # Single timeframe mode
    python run_paper_trading.py -y                 # Skip confirmation
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
    print("BINANCE FUTURES PAPER TRADING")
    print("="*60)
    print(f"Duration: {'CONTINUOUS (24/7)' if args.hours == 0 else f'{args.hours} hours'}")
    print(f"Mode: {'Single Timeframe' if args.no_mtf else 'Multi-Timeframe'}")
    print(f"Symbols: {', '.join(FUTURES_SYMBOLS)}")
    print(f"Initial Balance: ${PAPER_TRADING_CONFIG['initial_balance']:,.2f}")
    print("="*60)

    # Confirm
    if not args.yes:
        confirm = input("\nStart paper trading? (y/n): ")
        if confirm.lower() not in ("y", "yes"):
            print("Cancelled")
            return

    # Run paper trading
    engine = BinancePaperTradingEngine(use_mtf=not args.no_mtf)

    try:
        engine.run(duration_hours=args.hours)
    except KeyboardInterrupt:
        print("\nStopping paper trading...")
        engine.stop()


if __name__ == "__main__":
    main()
