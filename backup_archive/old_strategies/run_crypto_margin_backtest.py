#!/usr/bin/env python3
"""
Run Crypto Margin Backtest & Optimization
=========================================
Download historical data and optimize strategy parameters.

Usage:
    # Download data first
    python run_crypto_margin_backtest.py --download

    # Run optimization
    python run_crypto_margin_backtest.py --optimize

    # Run single backtest
    python run_crypto_margin_backtest.py --backtest BTCUSDT
"""

import sys
import os
import argparse

# Add trading system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Crypto_Margin_Trading'))

from engine.binance_data_client import BinanceDataClient
from engine.crypto_backtest_engine_fast import FastCryptoBacktestEngine as CryptoBacktestEngine


def download_data(pairs: list = None, candle_count: int = 100000):
    """Download historical data from Binance"""
    print("\n" + "=" * 70)
    print("DOWNLOADING HISTORICAL DATA FROM BINANCE")
    print("=" * 70)

    client = BinanceDataClient()

    if pairs:
        client.download_all_pairs(pairs=pairs, candle_count=candle_count)
    else:
        client.download_all_pairs(candle_count=candle_count)


def run_optimization(pairs: list = None):
    """Run strategy optimization on all pairs"""
    print("\n" + "=" * 70)
    print("RUNNING STRATEGY OPTIMIZATION")
    print("=" * 70)

    engine = CryptoBacktestEngine()

    if pairs is None:
        pairs = [
            'BTCUSDT',
            'ETHUSDT',
            'SOLUSDT',
            'XRPUSDT',
            'DOGEUSDT',
            'LTCUSDT',
            'ADAUSDT',
            'LINKUSDT',
        ]

    engine.optimize_all_pairs(pairs)


def run_single_backtest(pair: str, strategy: str = 'RSI_REVERSAL', tp: float = 0.5, sl: float = 1.2):
    """Run a single backtest"""
    print("\n" + "=" * 70)
    print(f"BACKTESTING {pair}")
    print("=" * 70)

    engine = CryptoBacktestEngine()

    # Load data
    df = engine.load_data(pair)
    if df is None:
        print(f"No data found for {pair}. Run --download first.")
        return

    # Run backtest
    result = engine.run_backtest(
        df=df,
        strategy=strategy,
        tp_pct=tp,
        sl_pct=sl,
    )

    # Print results
    print(f"\n{'BACKTEST RESULTS':^70}")
    print("-" * 70)
    print(f"Pair: {pair}")
    print(f"Strategy: {strategy}")
    print(f"TP: {tp}% | SL: {sl}%")
    print("-" * 70)
    print(f"Total Trades: {result['total_trades']}")
    print(f"Wins: {result['wins']} | Losses: {result['losses']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Profit Factor: {result['profit_factor']:.2f}")
    print(f"Total P&L: ${result['total_pnl']:,.2f}")
    print(f"Return: {result['return_pct']:.2f}%")
    print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description='Crypto Margin Trading Backtest & Optimization')
    parser.add_argument('--download', action='store_true', help='Download historical data from Binance')
    parser.add_argument('--optimize', action='store_true', help='Run strategy optimization')
    parser.add_argument('--backtest', type=str, help='Run single backtest on pair (e.g., BTCUSDT)')
    parser.add_argument('--pairs', type=str, nargs='+', help='Specific pairs to process')
    parser.add_argument('--candles', type=int, default=100000, help='Number of candles to download')
    parser.add_argument('--strategy', type=str, default='RSI_REVERSAL', help='Strategy for single backtest')
    parser.add_argument('--tp', type=float, default=0.5, help='Take profit % for single backtest')
    parser.add_argument('--sl', type=float, default=1.2, help='Stop loss % for single backtest')

    args = parser.parse_args()

    if args.download:
        download_data(pairs=args.pairs, candle_count=args.candles)

    elif args.optimize:
        run_optimization(pairs=args.pairs)

    elif args.backtest:
        run_single_backtest(
            pair=args.backtest,
            strategy=args.strategy,
            tp=args.tp,
            sl=args.sl
        )

    else:
        # Interactive menu
        print("\n" + "=" * 70)
        print("CRYPTO MARGIN TRADING - BACKTEST & OPTIMIZATION")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Download historical data from Binance")
        print("  2. Run strategy optimization on all pairs")
        print("  3. Run single backtest on a pair")
        print("  4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            download_data()
        elif choice == '2':
            run_optimization()
        elif choice == '3':
            pair = input("Enter pair (e.g., BTCUSDT): ").strip().upper()
            run_single_backtest(pair)
        else:
            print("Exiting.")


if __name__ == "__main__":
    main()
