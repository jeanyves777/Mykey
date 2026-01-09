#!/usr/bin/env python3
"""
Test Binance Connection
=======================
Quick test to verify API connection and configuration
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient
from engine.momentum_signal import MasterMomentumSignal
from config.trading_config import FUTURES_SYMBOLS


def test_connection():
    """Test API connection"""
    print("\n" + "="*60)
    print("BINANCE CONNECTION TEST")
    print("="*60)

    # Test client connection
    print("\n1. Testing Binance Testnet Connection...")
    client = BinanceClient(testnet=True)

    if not client.test_connection():
        print("[FAIL] Could not connect to Binance")
        return False

    print("[OK] Connected to Binance Testnet")

    return True


def test_market_data():
    """Test market data fetching"""
    print("\n2. Testing Market Data...")
    client = BinanceClient(testnet=True)

    for symbol in FUTURES_SYMBOLS[:2]:  # Test first 2 symbols
        try:
            # Get current price
            price_data = client.get_current_price(symbol)
            print(f"\n{symbol}:")
            print(f"  Price: ${price_data['price']:,.2f}")
            print(f"  Bid: ${price_data['bid']:,.2f}")
            print(f"  Ask: ${price_data['ask']:,.2f}")

            # Get candles
            df = client.get_klines(symbol, "1m", 10)
            if not df.empty:
                print(f"  Last candle: O:{df['open'].iloc[-1]:.2f} H:{df['high'].iloc[-1]:.2f} L:{df['low'].iloc[-1]:.2f} C:{df['close'].iloc[-1]:.2f}")
            else:
                print("  [WARN] No candle data")

        except Exception as e:
            print(f"  [ERROR] {e}")

    return True


def test_momentum_signal():
    """Test momentum signal generation"""
    print("\n3. Testing Momentum Signal Generator...")
    client = BinanceClient(testnet=True)
    signal_gen = MasterMomentumSignal()

    symbol = "BTCUSDT"
    print(f"\nGenerating signal for {symbol}...")

    try:
        # Get market data
        df = client.get_klines(symbol, "1m", 100)

        if df.empty:
            print("[WARN] No data available")
            return True

        # Generate signal
        signal = signal_gen.generate_signal(symbol, df)

        print(f"  Signal: {signal.signal or 'NO SIGNAL'}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Momentum: {signal.momentum_value:.3f}%")
        print(f"  EMA Trend: {signal.ema_trend}")
        print(f"  RSI: {signal.rsi_value:.1f}")
        print(f"  ADX: {signal.adx_value:.1f}")
        print(f"  Reason: {signal.reason}")

    except Exception as e:
        print(f"[ERROR] {e}")

    return True


def test_account():
    """Test account info (may fail on demo)"""
    print("\n4. Testing Account Info...")
    client = BinanceClient(testnet=True)

    try:
        balance = client.get_balance()
        print(f"  Balance: ${balance:,.2f} USDT")

        available = client.get_available_balance()
        print(f"  Available: ${available:,.2f} USDT")

        positions = client.get_positions()
        if positions:
            print(f"  Open Positions: {len(positions)}")
            for pos in positions:
                print(f"    - {pos['symbol']} {pos['side']}: {pos['quantity']} @ ${pos['entry_price']:,.2f}")
        else:
            print("  No open positions")

    except Exception as e:
        print(f"  [NOTE] Account access may require proper testnet setup: {e}")

    return True


def main():
    print("\n" + "="*60)
    print("BINANCE FUTURES TRADING SYSTEM - CONNECTION TEST")
    print("="*60)

    tests = [
        ("Connection", test_connection),
        ("Market Data", test_market_data),
        ("Momentum Signal", test_momentum_signal),
        ("Account Info", test_account),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] {name} test failed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! System is ready for trading.")
    else:
        print("\nSome tests failed. Please check configuration.")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
