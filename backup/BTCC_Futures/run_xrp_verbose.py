#!/usr/bin/env python3
"""
Run XRP Verbose Trading Engine
===============================
Comprehensive logging of all trading activity:
- Real-time indicator values
- Signal generation logic
- Position P&L tracking
- Distance to TP/SL/Liquidation

Usage:
    python run_xrp_verbose.py
    python run_xrp_verbose.py --size 50
"""

import sys
import os
import argparse
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.xrp_verbose_engine import XRPVerboseEngine
from BTCC_Futures.xrp_95_winrate_config import XRP95WinRateConfig


def main():
    parser = argparse.ArgumentParser(description='XRP Verbose Trading')
    parser.add_argument('--balance', type=float, default=100.0, help='Initial balance (default: $100)')
    parser.add_argument('--size', type=float, default=10.0, help='Position size percent (default: 10)')
    parser.add_argument('--tp', type=float, default=1.0, help='Take profit percent (default: 1.0)')
    parser.add_argument('--sl', type=float, default=10.0, help='Stop loss percent (default: 10.0)')
    parser.add_argument('--ema-fast', type=int, default=5, help='Fast EMA period (default: 5)')
    parser.add_argument('--ema-slow', type=int, default=13, help='Slow EMA period (default: 13)')
    args = parser.parse_args()

    config = XRP95WinRateConfig(
        TRADING_MODE="paper",
        PAPER_INITIAL_BALANCE=args.balance,
        POSITION_SIZE_PCT=args.size,
        TAKE_PROFIT_PCT=args.tp,
        STOP_LOSS_PCT=args.sl,
        EMA_FAST=args.ema_fast,
        EMA_SLOW=args.ema_slow,
        CHECK_INTERVAL=3.0,  # Faster updates for verbose mode
    )

    engine = XRPVerboseEngine(config)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        engine.run()
    except Exception as e:
        print(f"Error: {e}")
        engine.stop()


if __name__ == "__main__":
    main()
