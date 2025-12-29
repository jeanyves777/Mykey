#!/usr/bin/env python3
"""
Run Binance Futures Backtest
============================
Historical backtesting with real Binance data
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.backtest_engine import BinanceBacktestEngine
from config.trading_config import FUTURES_SYMBOLS, BACKTEST_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Binance Futures Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backtest.py                         # Backtest last 30 days
    python run_backtest.py --days 60               # Backtest last 60 days
    python run_backtest.py --symbol BTCUSDT        # Single symbol backtest
    python run_backtest.py --start 2024-01-01 --end 2024-02-01
    python run_backtest.py --days 30 --save        # Save results to file
        """
    )

    parser.add_argument(
        "--days",
        type=int,
        default=BACKTEST_CONFIG["default_days"],
        help=f"Days to backtest (default: {BACKTEST_CONFIG['default_days']})"
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to backtest (e.g., BTCUSDT)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Calculate dates
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Determine symbols
    symbols = [args.symbol] if args.symbol else FUTURES_SYMBOLS

    # Display configuration
    print("\n" + "="*60)
    print("BINANCE FUTURES BACKTEST")
    print("="*60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Balance: ${BACKTEST_CONFIG['initial_balance']:,.2f}")
    print("="*60)

    # Run backtest
    engine = BinanceBacktestEngine(symbols=symbols)
    result = engine.run_backtest(start_date, end_date, symbol=args.symbol)

    # Save results if requested
    if args.save and result.total_trades > 0:
        engine.save_results(result)
        print("Results saved!")

    return result


if __name__ == "__main__":
    result = main()
