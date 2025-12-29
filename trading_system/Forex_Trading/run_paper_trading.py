"""
Run Forex Paper Trading
Execute multi-timeframe momentum strategy with OANDA practice account
"""

import sys
import argparse
from trading_system.Forex_Trading.engine.forex_paper_trading_engine import ForexPaperTradingEngine


def main():
    parser = argparse.ArgumentParser(description="Run Forex Paper Trading")
    parser.add_argument(
        "--hours",
        type=float,
        default=8,
        help="Number of hours to run (default: 8)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    print("="*80)
    print("FOREX PAPER TRADING - OANDA Multi-Timeframe Momentum Strategy")
    print("="*80)
    print("\nThis will run paper trading with real OANDA data but simulated trades.")
    print(f"Duration: {args.hours} hours")
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
        engine = ForexPaperTradingEngine()
        engine.run(duration_hours=args.hours)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
