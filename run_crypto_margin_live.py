#!/usr/bin/env python3
"""
Run Crypto Margin LIVE Trading
==============================
LIVE trading runner for crypto margin trading on Kraken.

!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Requirements:
- KRAKEN_API_KEY in .env
- KRAKEN_API_SECRET in .env
- Kraken account with margin trading enabled
- Sufficient USD balance

Usage:
    python run_crypto_margin_live.py
"""

import sys
import os
import time
from datetime import datetime

# Add trading system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Crypto_Margin_Trading'))

from config import crypto_live_config as config
from engine.kraken_client import KrakenClient
from engine.binance_data_client import BinanceDataClient
from strategies.crypto_margin_strategy import calculate_indicators, get_signal
from utils.crypto_trade_logger import CryptoTradeLogger


class CryptoLiveTradingEngine:
    """Live trading engine using Kraken for execution"""

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.kraken = KrakenClient()
        self.binance = BinanceDataClient()  # For price data
        self.logger = CryptoTradeLogger(
            log_dir=config.LOG_DIR,
            account_type='live'
        )

        # Tracking
        self.trade_counter = 0
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.session_date = datetime.utcnow().date()

        print("[Live Engine] Initialized")

    def get_account_info(self):
        """Get Kraken account info"""
        return self.kraken.get_account_info()

    def check_daily_loss_limit(self):
        """Check if daily loss limit exceeded"""
        if self.daily_start_balance <= 0:
            return False

        loss_pct = (self.daily_pnl / self.daily_start_balance) * 100
        if loss_pct <= -config.MAX_DAILY_LOSS_PCT:
            print(f"\n[RISK] Daily loss limit reached: {loss_pct:.1f}%")
            return True
        return False

    def run(self):
        """Main live trading loop"""
        # Get initial account info
        account = self.get_account_info()
        self.daily_start_balance = account.get('equity', 0)

        print(f"\n[Live] Starting balance: ${self.daily_start_balance:,.2f}")
        print(f"[Live] Max daily loss: {config.MAX_DAILY_LOSS_PCT}%")

        iteration = 0
        while True:
            try:
                iteration += 1

                # Reset daily tracking at midnight
                if datetime.utcnow().date() != self.session_date:
                    self.session_date = datetime.utcnow().date()
                    account = self.get_account_info()
                    self.daily_start_balance = account.get('equity', 0)
                    self.daily_pnl = 0.0
                    print(f"\n[Live] New day - Balance: ${self.daily_start_balance:,.2f}")

                # Check daily loss limit
                if self.check_daily_loss_limit():
                    print("[Live] Daily loss limit reached - stopping for today")
                    time.sleep(3600)  # Sleep for an hour
                    continue

                # Get open positions
                positions = self.kraken.get_open_positions()

                # Check for signals on enabled pairs
                for pair in config.TRADING_PAIRS:
                    binance_pair = pair  # Already in Binance format in config
                    kraken_pair = config.get_kraken_symbol(pair)

                    # Skip if already have position
                    has_position = any(p['pair'] == kraken_pair for p in positions)
                    if has_position:
                        continue

                    # Skip if at max positions
                    if len(positions) >= config.MAX_CONCURRENT_POSITIONS:
                        continue

                    # Get candle data from Binance
                    candles = self.binance.get_klines(binance_pair, interval='1m', limit=100)
                    if not candles:
                        continue

                    # Create DataFrame and calculate indicators
                    import pandas as pd
                    df = pd.DataFrame(candles)
                    df = calculate_indicators(df)

                    # Get signal
                    signal, reason = get_signal(binance_pair, df, config)

                    if signal:
                        print(f"\n[Signal] {pair} {signal}: {reason}")

                        # Get current price
                        price_data = self.kraken.get_current_price(kraken_pair)
                        if not price_data:
                            continue

                        current_price = price_data['ask'] if signal == 'BUY' else price_data['bid']

                        # Calculate position size
                        settings = config.get_pair_settings(pair)
                        account = self.get_account_info()
                        balance = account.get('equity', 0)

                        volume = config.calculate_position_size(
                            account_balance=balance,
                            entry_price=current_price,
                            sl_pct=settings['sl_pct'],
                            leverage=settings['leverage']
                        )

                        # Calculate SL
                        _, stop_loss = config.calculate_tp_sl(pair, current_price, signal)

                        # Place order
                        print(f"[Live] Placing {signal} order: {volume:.6f} {kraken_pair} @ ${current_price:,.2f}")

                        result = self.kraken.place_margin_order(
                            pair=kraken_pair,
                            side=signal.lower(),
                            volume=volume,
                            leverage=settings['leverage'],
                            stop_loss=stop_loss
                        )

                        if result.get('success'):
                            print(f"[Live] Order placed: {result.get('order_ids')}")
                            self.trade_counter += 1

                            # Log entry
                            self.logger.log_trade_entry(
                                trade_id=result.get('order_ids', ['unknown'])[0],
                                pair=pair,
                                direction=signal,
                                entry_price=current_price,
                                volume=volume,
                                leverage=settings['leverage'],
                                stop_loss=stop_loss,
                                take_profit=0,  # Manual TP management
                                strategy=settings['strategy'],
                                signal_reason=reason,
                            )
                        else:
                            print(f"[Live] Order failed: {result.get('error')}")

                # Display status every few iterations
                if iteration % 12 == 0:  # Every minute
                    account = self.get_account_info()
                    print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] Equity: ${account.get('equity', 0):,.2f} | Positions: {len(positions)}")

                time.sleep(5)

            except KeyboardInterrupt:
                print("\n\nShutdown requested...")
                break
            except Exception as e:
                print(f"[Error] {e}")
                time.sleep(10)

        # Shutdown
        self.logger.print_session_stats()
        self.logger.generate_session_summary()


def main():
    """Main entry point for live trading"""
    print("\n" + "=" * 80)
    print("CRYPTO MARGIN TRADING SYSTEM - LIVE MODE")
    print("=" * 80)
    print("\n" + "!" * 80)
    print("!!!  WARNING: THIS WILL TRADE WITH REAL MONEY  !!!")
    print("!" * 80)

    # Print configuration
    config.print_config_info()

    # Multiple confirmations
    print("\n" + "-" * 80)
    print("RISK ACKNOWLEDGMENT")
    print("-" * 80)
    print("You are about to start LIVE trading with REAL MONEY.")
    print("Losses may occur. Only trade with money you can afford to lose.")
    print("-" * 80)

    # First confirmation
    print("\nType 'I ACCEPT THE RISK' to continue:")
    confirm1 = input().strip()
    if confirm1 != 'I ACCEPT THE RISK':
        print("Confirmation failed. Exiting.")
        return

    # Second confirmation
    print("\nType 'START' to begin live trading:")
    confirm2 = input().strip()
    if confirm2 != 'START':
        print("Confirmation failed. Exiting.")
        return

    # Verify Kraken connection
    print("\nVerifying Kraken connection...")
    try:
        client = KrakenClient()
        account = client.get_account_info()

        if not account.get('balances'):
            print("[Error] Could not connect to Kraken. Check API credentials.")
            return

        print(f"[OK] Connected to Kraken")
        print(f"     USD Balance: ${account.get('usd_balance', 0):,.2f}")
        print(f"     Equity: ${account.get('equity', 0):,.2f}")

    except Exception as e:
        print(f"[Error] Kraken connection failed: {e}")
        return

    # Final countdown
    print("\nStarting in 5 seconds... Press Ctrl+C to cancel.")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # Start live trading
    print("\n" + "=" * 80)
    print("LIVE TRADING STARTED")
    print("=" * 80)

    try:
        engine = CryptoLiveTradingEngine(config)
        engine.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
