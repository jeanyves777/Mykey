#!/usr/bin/env python3
"""
Run Crypto Margin Paper Trading
===============================
Paper trading runner for crypto margin trading system.
Uses Binance for real-time prices, simulates trades locally.

Usage:
    python run_crypto_margin_paper.py
"""

import sys
import os

# Add trading system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Crypto_Margin_Trading'))

from config import crypto_paper_config as config
from engine.crypto_paper_trading_engine import CryptoPaperTradingEngine


def main():
    """Main entry point for paper trading"""
    print("\n" + "=" * 80)
    print("CRYPTO MARGIN TRADING SYSTEM - PAPER MODE")
    print("=" * 80)

    # Print configuration
    config.print_config_info()

    # Confirm start
    print("\nThis will start PAPER trading (simulated, no real money).")
    print("Press Enter to start or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    # Initialize and run engine
    try:
        engine = CryptoPaperTradingEngine(config)
        engine.run(
            display_interval=60,      # Display status every 60 seconds
            signal_check_interval=60  # Check for signals every 60 seconds
        )
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
