"""
Run Optimized Forex Trading
Execute pair-specific 70%+ win rate strategies with OANDA
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_system.Forex_Trading.engine.optimized_forex_engine import OptimizedForexEngine


def main():
    parser = argparse.ArgumentParser(description="Run Optimized Forex Trading")
    parser.add_argument(
        "--account",
        type=str,
        default="practice",
        choices=["practice", "live"],
        help="Account type: practice (demo) or live"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OPTIMIZED FOREX TRADING - 70%+ Win Rate Strategies")
    print("=" * 70)
    print("\nPair-Specific Strategy Settings:")
    print("-" * 50)
    print(f"{'PAIR':<10} {'STRATEGY':<15} {'TP':<5} {'SL':<5} {'Expected WR'}")
    print("-" * 50)
    print(f"{'EUR_USD':<10} {'RSI_30_70':<15} {'5p':<5} {'20p':<5} 87.1%")
    print(f"{'GBP_USD':<10} {'STRONG_TREND':<15} {'5p':<5} {'20p':<5} 84.0%")
    print(f"{'USD_JPY':<10} {'RSI_30_70':<15} {'5p':<5} {'20p':<5} 82.1%")
    print(f"{'USD_CHF':<10} {'STRONG_TREND':<15} {'8p':<5} {'24p':<5} 88.0%")
    print(f"{'USD_CAD':<10} {'RSI_30_70':<15} {'5p':<5} {'20p':<5} 86.3%")
    print("-" * 50)
    print(f"\nAccount: {args.account.upper()}")
    print()
    print("Strategy Types:")
    print("  RSI_30_70: Buy when RSI crosses UP through 30 + green candle")
    print("             Sell when RSI crosses DOWN through 70 + red candle")
    print("  STRONG_TREND: Trade pullbacks in EMA9 > EMA21 > EMA50 trends")
    print()
    print("Position Sizing: $1 per pip")
    print()

    if args.account == "live":
        print("WARNING: LIVE TRADING with real money!")
        print("Make sure you understand the risks.")
        print()

    if not args.yes:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    try:
        engine = OptimizedForexEngine(account_type=args.account)
        engine.run()

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
