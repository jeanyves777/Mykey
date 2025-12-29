#!/usr/bin/env python3
"""
THE VOLUME AI - COIN CALLS-ONLY Pullback Trading Runner

This strategy ONLY buys CALLs - never PUTs.
It waits for pullbacks (dips) then enters on recovery signals.

Philosophy:
- Buying calls on pullbacks tends to be more profitable
- Even in bearish markets, there are bounces to catch
- Defined risk (premium paid) vs undefined profit potential

Usage:
    python -m trading_system.run_coin_calls_only [-y]

Options:
    -y          Skip confirmation prompt (auto-start)
    --test      Test API connection only
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.engine.alpaca_client import test_connection, ALPACA_AVAILABLE
from trading_system.strategies.coin_calls_only_pullback import COINCallsOnlyPullbackConfig


def print_banner():
    """Print application banner."""
    print("""
================================================================================
                           THE VOLUME AI
              COIN CALLS-ONLY Pullback Strategy (Paper Trading)
================================================================================

  Strategy: Buy CALLs on pullbacks/dips, ride the recovery

  NEVER buys PUTs - only CALLs!

  Entry Logic:
    1. Detect pullback (RSI oversold, red candles, below VWAP)
    2. Wait for recovery signal (green candle, momentum shift)
    3. Enter CALL position on the bounce

  Exit Logic:
    - Take Profit: +7.5% (LIMIT order on exchange)
    - Stop Loss: -25% (monitored internally)
    - Max Hold: 30 minutes
    - Force Exit: 3:50 PM EST

================================================================================
    """)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    if not ALPACA_AVAILABLE:
        missing.append("alpaca-py")

    try:
        import pytz
    except ImportError:
        missing.append("pytz")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def display_config(config: COINCallsOnlyPullbackConfig):
    """Display current configuration."""
    print("\n--- Configuration ---")
    print(f"  Symbol:               {config.underlying_symbol}")
    print(f"  Position Size:        ${config.fixed_position_value:,.2f}")
    print(f"  Take Profit:          {config.target_profit_pct}%")
    print(f"  Stop Loss:            {config.stop_loss_pct}%")
    print(f"  Max Hold Time:        {config.max_hold_minutes} minutes")
    print(f"  Entry Window:         {config.entry_time_start} - {config.entry_time_end} EST")
    print(f"  Force Exit:           {config.force_exit_time} EST")
    print(f"  Max Trades/Day:       {config.max_trades_per_day}")
    print()
    print("--- Pullback Detection ---")
    print(f"  Min Pullback:         {config.min_pullback_pct}% from recent high")
    print(f"  RSI Oversold:         < {config.rsi_oversold_threshold}")
    print(f"  Min Red Candles:      {config.min_red_candles_for_pullback}")
    print()
    print("--- Recovery Detection ---")
    print(f"  Require Green Candle: {config.require_green_candle}")
    print(f"  Min Bounce:           {config.min_bounce_pct}% from low")
    print(f"  RSI Recovery:         > {config.rsi_recovery_threshold}")
    print()
    print("--- Safety Limits ---")
    print(f"  Max RSI for Entry:    {config.max_rsi_for_entry} (avoid overbought)")
    print(f"  Max Spread:           {config.max_spread_pct}%")
    print()


def load_api_credentials():
    """Load API credentials from environment or config."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Try COIN-specific keys first, then fall back to general paper keys
    api_key = os.getenv('ALPACA_COIN_KEY') or os.getenv('ALPACA_API_KEY_PAPER') or os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_COIN_SECRET') or os.getenv('ALPACA_SECRET_KEY_PAPER') or os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not api_secret:
        print("ERROR: Alpaca API credentials not found.")
        print("Set ALPACA_COIN_KEY and ALPACA_COIN_SECRET in .env file")
        print("Or set ALPACA_API_KEY_PAPER and ALPACA_SECRET_KEY_PAPER")
        return None, None

    return api_key, api_secret


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI - COIN CALLS-ONLY Pullback Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_coin_calls_only
    python -m trading_system.run_coin_calls_only -y
    python -m trading_system.run_coin_calls_only --test
        """
    )

    parser.add_argument(
        '-y',
        action='store_true',
        help='Skip confirmation prompt (auto-start)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test API connection only'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load API credentials
    api_key, api_secret = load_api_credentials()
    if not api_key:
        sys.exit(1)

    # Create configuration
    config = COINCallsOnlyPullbackConfig()

    # Display configuration
    display_config(config)

    # Test connection mode
    if args.test:
        print("Testing API connection...")
        if test_connection(api_key, api_secret, paper=True):
            print("\nConnection successful!")
        else:
            print("\nConnection failed. Please check your API credentials.")
            sys.exit(1)
        sys.exit(0)

    # Confirm before starting (unless -y flag)
    if not args.y:
        print("Ready to start CALLS-ONLY paper trading.")
        print("\nThis strategy will ONLY buy CALL options on pullbacks.")
        print("Press Enter to continue, or Ctrl+C to cancel...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Import and run the engine
    from trading_system.engine.coin_calls_only_engine import COINCallsOnlyEngine

    try:
        engine = COINCallsOnlyEngine(
            api_key=api_key,
            api_secret=api_secret,
            config=config
        )
        engine.run()
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
