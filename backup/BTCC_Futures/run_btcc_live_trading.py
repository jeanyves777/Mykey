#!/usr/bin/env python3
"""
BTCC Futures Live Trading
=========================
Run live trading on BTCC Futures platform.

WARNING: This trades with real money! Use at your own risk.

Usage:
    python run_btcc_live_trading.py
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.btcc_trading_engine import BTCCTradingEngine, TradeSignal
from BTCC_Futures.btcc_config import BTCCConfig
from BTCC_Futures.btcc_strategies import get_strategy, OHLCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BTCCLiveTrader:
    """Live trading runner for BTCC Futures"""

    def __init__(self):
        # Configure for live trading
        self.config = BTCCConfig(
            # API Credentials
            API_KEY="2cc78c86-cf78-47a4-ad38-ca1c0e9b8517",
            SECRET_KEY="6697e175-4ec6-4748-99cf-0ce4ac92790b",

            # IMPORTANT: Set your login credentials
            USER_NAME="",  # Your BTCC email or phone
            PASSWORD="",   # Your BTCC password

            TRADING_MODE="live",
            MAX_POSITIONS=3,        # Conservative for live
            POSITION_SIZE_PCT=1.0,  # 1% per trade
            MAX_POSITION_VALUE=100, # Max $100 per position
            DEFAULT_LEVERAGE=10,    # Conservative leverage

            # Risk management
            DEFAULT_STOP_LOSS_PCT=1.5,
            DEFAULT_TAKE_PROFIT_PCT=3.0,
            MAX_DAILY_LOSS_PCT=3.0,  # Stop if 3% daily loss
        )

        # Enable symbols (conservative selection)
        self.config.SYMBOLS = {
            'BTCUSDT150x': {
                'enabled': True,
                'leverage': 10,
                'min_volume': 0.001,
                'max_volume': 1.0,
                'volume_step': 0.001,
                'digits': 2,
                'strategy': 'SCALPING_MOMENTUM',
                'tp_pct': 2.0,
                'sl_pct': 1.0,
            },
            'ETHUSDT100x': {
                'enabled': True,
                'leverage': 10,
                'min_volume': 0.01,
                'max_volume': 10.0,
                'volume_step': 0.01,
                'digits': 2,
                'strategy': 'RSI_REVERSAL',
                'tp_pct': 2.5,
                'sl_pct': 1.5,
            },
        }

        self.engine = None
        self.running = False

    def _verify_credentials(self) -> bool:
        """Verify that credentials are set."""
        if not self.config.USER_NAME or not self.config.PASSWORD:
            print("\n" + "!" * 70)
            print("ERROR: Login credentials not set!")
            print("!" * 70)
            print("\nPlease edit this file and set:")
            print("  USER_NAME = 'your_email_or_phone'")
            print("  PASSWORD = 'your_password'")
            print("\nThese are your BTCC account login credentials.")
            print("!" * 70)
            return False
        return True

    def _confirm_live_trading(self) -> bool:
        """Get user confirmation for live trading."""
        print("\n" + "!" * 70)
        print("WARNING: LIVE TRADING MODE")
        print("!" * 70)
        print("\nThis will trade with REAL MONEY on your BTCC account!")
        print(f"Max Position Value: ${self.config.MAX_POSITION_VALUE}")
        print(f"Max Daily Loss: {self.config.MAX_DAILY_LOSS_PCT}%")
        print(f"Symbols: {', '.join(self.config.get_enabled_symbols())}")
        print("!" * 70)

        response = input("\nType 'YES' to confirm live trading: ")
        return response.upper() == 'YES'

    def run(self):
        """Run the live trading session."""
        # Verify credentials
        if not self._verify_credentials():
            return

        # Confirm live trading
        if not self._confirm_live_trading():
            print("\nLive trading cancelled.")
            return

        # Create engine
        self.engine = BTCCTradingEngine(self.config)

        print("\n" + "=" * 70)
        print("BTCC FUTURES LIVE TRADING")
        print("=" * 70)
        print(f"Max Positions: {self.config.MAX_POSITIONS}")
        print(f"Position Size: {self.config.POSITION_SIZE_PCT}%")
        print(f"Default Leverage: {self.config.DEFAULT_LEVERAGE}x")
        print(f"Daily Loss Limit: {self.config.MAX_DAILY_LOSS_PCT}%")
        print(f"\nEnabled Symbols: {', '.join(self.config.get_enabled_symbols())}")
        print("=" * 70)

        # Connect to BTCC
        print("\nConnecting to BTCC...")

        if not self.engine.connect():
            print("\nFailed to connect to BTCC. Check your credentials.")
            return

        print("Connected successfully!")
        print("-" * 70)
        print("\nStarting live trading... (Press Ctrl+C to stop)")
        print("-" * 70)

        # Start engine
        self.engine.start()
        self.running = True

        try:
            while self.running:
                # Print status
                status = self.engine.get_status()
                positions = self.engine.get_positions()

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Balance: ${status['balance']:,.2f} | "
                      f"Positions: {status['open_positions']} | "
                      f"Daily P/L: ${status['daily_pnl']:+,.2f} | "
                      f"Trades: {status['trades_today']} | "
                      f"Win Rate: {status['win_rate']:.1f}%",
                      end='', flush=True)

                # Show positions
                if positions:
                    print("\n  Open Positions:")
                    for pos in positions:
                        direction = "LONG" if pos.direction == 1 else "SHORT"
                        print(f"    {pos.symbol}: {direction} x{pos.volume} @ {pos.entry_price:.2f} "
                              f"| P/L: ${pos.pnl:+.2f}")

                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\nStopping live trading...")

        finally:
            # Close all positions before stopping
            print("\nClosing all positions...")
            self.engine.close_all_positions("Session end")

            self.engine.stop()
            self._print_summary()

    def _print_summary(self):
        """Print trading session summary."""
        if not self.engine:
            return

        status = self.engine.get_status()

        print("\n" + "=" * 70)
        print("LIVE TRADING SESSION SUMMARY")
        print("=" * 70)
        print(f"Final Balance:     ${status['balance']:,.2f}")
        print(f"Total P/L:         ${status['total_pnl']:+,.2f}")
        print(f"Daily P/L:         ${status['daily_pnl']:+,.2f}")
        print(f"Total Trades:      {status['winning_trades'] + status['losing_trades']}")
        print(f"Winning Trades:    {status['winning_trades']}")
        print(f"Losing Trades:     {status['losing_trades']}")
        print(f"Win Rate:          {status['win_rate']:.1f}%")
        print("=" * 70)


def main():
    """Main entry point."""
    trader = BTCCLiveTrader()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        trader.running = False

    signal.signal(signal.SIGINT, signal_handler)

    trader.run()


if __name__ == "__main__":
    main()
