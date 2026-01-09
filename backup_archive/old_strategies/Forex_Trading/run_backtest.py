"""
Run Forex Backtest
Backtest multi-timeframe momentum strategy on historical OANDA data
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
from datetime import datetime, timedelta
from trading_system.Forex_Trading.engine.forex_backtest_engine import ForexBacktestEngine
from trading_system.Forex_Trading.config.forex_trading_config import FOREX_INSTRUMENTS


def main():
    parser = argparse.ArgumentParser(description="Run Forex Backtest")
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to backtest (default: 30, used if --start not provided)"
    )
    parser.add_argument(
        "--instruments",
        type=str,
        nargs='+',
        default=["EUR_USD", "GBP_USD"],
        help="Forex pairs to trade (default: EUR_USD GBP_USD)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Determine date range
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=args.days)
        start_date = start_dt.strftime("%Y-%m-%d")

    print("="*80)
    print("FOREX BACKTEST - OANDA Multi-Timeframe Momentum Strategy")
    print("="*80)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Instruments: {', '.join(args.instruments)}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print("\nMake sure you have set up your .env file with:")
    print("  OANDA_PRACTICE_API_KEY=your_practice_api_key")
    print("  OANDA_PRACTICE_ACCOUNT_ID=your_practice_account_id")
    print()

    if not args.yes:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    try:
        engine = ForexBacktestEngine(
            instruments=args.instruments,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital
        )

        engine.run_backtest()

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
