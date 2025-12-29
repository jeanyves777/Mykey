"""
Run DCA Backtest
================

Run backtest for the DCA momentum strategy on crypto data.

Usage:
    python -m trading_system.High_frequency_crypto_tradin.run_dca_backtest
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from trading_system.High_frequency_crypto_tradin.dca_config import DCAConfig, load_dca_config
from trading_system.High_frequency_crypto_tradin.dca_engine import DCAEngine


def main():
    print("=" * 70)
    print("DCA MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load configuration
    config = load_dca_config()

    # Print configuration
    print("\nConfiguration:")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Risk per trade: {config.risk_per_trade_pct*100:.1f}%")
    print(f"  TP: {config.take_profit_pct*100:.2f}%")
    print(f"  SL: {config.stop_loss_pct*100:.2f}%")
    print(f"  DCA spacing: {config.dca_spacing_pct*100:.2f}%")
    print(f"  Momentum threshold: {config.momentum_threshold}%")
    print(f"  Symbols: {config.symbols}")

    # Show position sizing
    print("\nPosition Sizing:")
    sizes = config.calculate_position_sizes(config.initial_capital)
    print(f"  {'Stage':<10} {'Risk $':<10} {'Cumulative':<12} {'% of Base'}")
    print(f"  {'-'*45}")
    for s in sizes:
        print(f"  {s['stage']:<10} ${s['risk_amount']:<8.2f} ${s['cumulative_risk']:<10.2f} {s['pct_of_base']:.1f}%")

    # Run backtest on each symbol
    all_results = {}

    for symbol in config.symbols:
        print(f"\n{'='*70}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*70}")

        data_file = Path(config.data_dir) / f"{symbol}_1m.csv"

        if not data_file.exists():
            print(f"  Data file not found: {data_file}")
            continue

        # Load data
        df = pd.read_csv(data_file)

        # Use recent data for testing
        df = df.tail(50000).reset_index(drop=True)
        print(f"  Data: {len(df):,} bars")

        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            except:
                pass

        # Create engine and run backtest
        engine = DCAEngine(config)
        results = engine.run_backtest(df, symbol)

        all_results[symbol] = {
            'results': results,
            'trades': engine.trade_history.copy()
        }

        # Print results
        print(f"\n  Results for {symbol}:")
        print(f"    Trades: {results['total_trades']}")
        print(f"    Win Rate: {results['win_rate']*100:.1f}%")
        print(f"    Total P&L: ${results['total_pnl']:,.2f}")
        print(f"    Return: {results['return_pct']:+.1f}%")
        print(f"    Final Capital: ${results['final_capital']:,.2f}")

        if results['total_trades'] > 0:
            print(f"    Profit Factor: {results['profit_factor']:.2f}")
            print(f"    Avg DCA Stages: {results['avg_dca_stages']:.1f}")
            print(f"    Outcomes: {results['outcomes']}")

        # Save trades
        engine.save_trades()

    # Combined summary
    print(f"\n{'='*70}")
    print("COMBINED SUMMARY")
    print(f"{'='*70}")

    total_trades = sum(r['results']['total_trades'] for r in all_results.values())
    total_pnl = sum(r['results']['total_pnl'] for r in all_results.values())
    total_wins = sum(r['results']['wins'] for r in all_results.values())

    if total_trades > 0:
        print(f"  Total Trades: {total_trades}")
        print(f"  Overall Win Rate: {100*total_wins/total_trades:.1f}%")
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Return: {100*total_pnl/config.initial_capital:+.1f}%")

    # Test different configurations
    print(f"\n{'='*70}")
    print("PARAMETER SENSITIVITY TEST")
    print(f"{'='*70}")

    # Test different TP/SL combinations
    test_configs = [
        {'tp': 0.004, 'sl': 0.004, 'dca': 0.0015, 'name': '0.4% TP/SL'},
        {'tp': 0.005, 'sl': 0.005, 'dca': 0.002, 'name': '0.5% TP/SL (Default)'},
        {'tp': 0.006, 'sl': 0.004, 'dca': 0.002, 'name': '0.6% TP, 0.4% SL'},
        {'tp': 0.008, 'sl': 0.005, 'dca': 0.003, 'name': '0.8% TP, 0.5% SL'},
    ]

    # Use BTCUSD for sensitivity test
    symbol = 'BTCUSD'
    data_file = Path(config.data_dir) / f"{symbol}_1m.csv"

    if data_file.exists():
        df = pd.read_csv(data_file)
        df = df.tail(50000).reset_index(drop=True)

        print(f"\nTesting on {symbol}:")
        print(f"  {'Config':<25} {'Trades':<8} {'WR%':<8} {'Return':<10} {'PF'}")
        print(f"  {'-'*60}")

        for test in test_configs:
            # Create config with test params
            test_config = DCAConfig(
                take_profit_pct=test['tp'],
                stop_loss_pct=test['sl'],
                dca_spacing_pct=test['dca']
            )

            engine = DCAEngine(test_config)
            results = engine.run_backtest(df, symbol)

            marker = " ***" if results['return_pct'] > 5 else ""
            pf = results.get('profit_factor', 0)
            print(f"  {test['name']:<25} {results['total_trades']:<8} "
                  f"{results['win_rate']*100:<7.1f}% {results['return_pct']:>+8.1f}% "
                  f"{pf:.2f}{marker}")

    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
