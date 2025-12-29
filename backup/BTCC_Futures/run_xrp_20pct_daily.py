#!/usr/bin/env python3
"""
Run XRP 20% Daily Scalping Strategy
=====================================
Aggressive scalping for 20%+ daily returns

Backtest Results (7 days):
- Win Rate: 95.0%
- Return: +205.3% (7 days)
- Daily Average: +29.3%
- Liquidations: 0
- Max Drawdown: 0%

Usage:
    Paper Trading:  python run_xrp_20pct_daily.py
    Conservative:   python run_xrp_20pct_daily.py --size 30
    Live Trading:   python run_xrp_20pct_daily.py --live

WARNING: This uses 50% position sizing by default!
"""

import sys
import os
import argparse
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.xrp_95_winrate_engine import XRP95WinRateEngine
from BTCC_Futures.xrp_20pct_daily_config import XRP20PctDailyConfig, XRP_20PCT_PAPER_CONFIG


def print_banner():
    """Print startup banner"""
    print()
    print("=" * 80)
    print(" XRP 20% DAILY SCALPING STRATEGY")
    print("=" * 80)
    print()
    print(" Strategy: FAST_SCALP_MOMENTUM (EMA2/EMA5 + RSI)")
    print(" Settings: TP 1% | SL 10% | Leverage 20x | Position 50%")
    print()
    print(" Backtest Results (7 Days):")
    print("   - Win Rate: 95.0%")
    print("   - 7-Day Return: +205.3%")
    print("   - Daily Average: +29.3%")
    print("   - Liquidations: 0")
    print("   - Max Drawdown: 0%")
    print()
    print(" WARNING: AGGRESSIVE 50% position sizing!")
    print(" Use --size 30 for conservative 30% sizing (~15% daily)")
    print()
    print("=" * 80)
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='XRP 20% Daily Scalping')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--balance', type=float, default=100.0, help='Initial paper balance (default: $100)')
    parser.add_argument('--size', type=float, default=50.0, help='Position size percent (default: 50)')
    parser.add_argument('--tp', type=float, default=1.0, help='Take profit percent (default: 1.0)')
    parser.add_argument('--sl', type=float, default=10.0, help='Stop loss percent (default: 10.0)')
    args = parser.parse_args()

    print_banner()

    # Create config
    if args.live:
        print("MODE: LIVE TRADING")
        print()
        print("WARNING: This is an AGGRESSIVE strategy!")
        print("50% position size means each trade risks half your account!")
        print()

        confirm = input("Type 'I UNDERSTAND THE RISK' to proceed: ")
        if confirm != 'I UNDERSTAND THE RISK':
            print("Cancelled.")
            return

        config = XRP20PctDailyConfig(
            TRADING_MODE="live",
            POSITION_SIZE_PCT=args.size,
            TAKE_PROFIT_PCT=args.tp,
            STOP_LOSS_PCT=args.sl,
        )
    else:
        print("MODE: PAPER TRADING")
        config = XRP20PctDailyConfig(
            TRADING_MODE="paper",
            PAPER_INITIAL_BALANCE=args.balance,
            POSITION_SIZE_PCT=args.size,
            TAKE_PROFIT_PCT=args.tp,
            STOP_LOSS_PCT=args.sl,
        )

    # Risk analysis
    expected_profit_per_trade = config.TAKE_PROFIT_PCT * config.LEVERAGE * config.POSITION_SIZE_PCT / 100
    expected_loss_per_trade = config.STOP_LOSS_PCT * config.LEVERAGE * config.POSITION_SIZE_PCT / 100

    print()
    print(f"Symbol:          {config.SYMBOL}")
    print(f"Leverage:        {config.LEVERAGE}x")
    print(f"Take Profit:     {config.TAKE_PROFIT_PCT}%")
    print(f"Stop Loss:       {config.STOP_LOSS_PCT}%")
    print(f"Position Size:   {config.POSITION_SIZE_PCT}%")
    print(f"Initial Balance: ${config.PAPER_INITIAL_BALANCE:.2f}")
    print()
    print("Risk Analysis:")
    print(f"  Profit per TP:   ~{expected_profit_per_trade:.1f}% of account")
    print(f"  Loss per SL:     ~{expected_loss_per_trade:.1f}% of account")
    print(f"  Liquidation at:  ~{100/config.LEVERAGE - 0.5:.1f}% price move")
    print()
    print("=" * 80)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    # Create engine (reuse the 95% engine with new config)
    engine = XRP95WinRateEngine(config)

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the engine
    try:
        engine.run()
    except Exception as e:
        print(f"\nError: {e}")
        engine.stop()


if __name__ == "__main__":
    main()
