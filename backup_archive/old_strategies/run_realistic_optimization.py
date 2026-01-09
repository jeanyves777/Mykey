#!/usr/bin/env python3
"""
REALISTIC Crypto Strategy Optimization
=======================================
Optimizes all 17 pairs WITH realistic trading costs:
- Kraken taker fees: 0.26% per side (0.52% round-trip)
- Slippage: 0.02% per side
- Total cost: ~0.56% per trade

IMPORTANT: TP must be >= 1.5% to be profitable after fees!
"""

import sys
import os

# Add trading system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Crypto_Margin_Trading'))

from engine.crypto_backtest_engine_fast import FastCryptoBacktestEngine

# All 17 trading pairs
TRADING_PAIRS = [
    # 10x leverage pairs
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'LTCUSDT', 'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'SUIUSDT',
    # 3x leverage pairs
    'DOTUSDT', 'ZECUSDT', 'BCHUSDT', 'PEPEUSDT', 'UNIUSDT',
    # 2x leverage pairs
    'XLMUSDT', 'XMRUSDT',
]


def main():
    print("\n" + "=" * 80)
    print("REALISTIC CRYPTO STRATEGY OPTIMIZATION")
    print("=" * 80)
    print("\nThis optimization includes REAL trading costs:")
    print("  - Kraken taker fee: 0.26% per side (0.52% round-trip)")
    print("  - Slippage: 0.02% per side (0.04% round-trip)")
    print("  - Total cost: ~0.56% per trade")
    print("\nMinimum viable TP: 1.5% (to exceed fees)")
    print("TP range: [1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%, 5.0%]")
    print("SL range: [1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 4.0%]")
    print("=" * 80)

    # Initialize engine with fees ENABLED (default)
    engine = FastCryptoBacktestEngine(include_fees=True)

    # Run optimization with realistic filters
    # Start with 60% WR, 1.2 PF minimum
    results = engine.optimize_all_pairs(
        pairs=TRADING_PAIRS,
        output_file='realistic_optimization_results.json',
        min_win_rate=60.0,      # Minimum 60% win rate
        min_profit_factor=1.2,  # Minimum 1.2 profit factor AFTER fees
    )

    # Summary statistics
    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS")
    print("=" * 80)

    profitable = [p for p, r in results.items() if r and r[0].get('net_pnl', 0) > 0]
    unprofitable = [p for p, r in results.items() if not r or r[0].get('net_pnl', 0) <= 0]

    print(f"\nPROFITABLE PAIRS ({len(profitable)}):")
    for pair in profitable:
        best = results[pair][0]
        print(f"  + {pair}: {best['strategy']} TP={best['tp_pct']}% SL={best['sl_pct']}% "
              f"WR={best['win_rate']}% PF={best['profit_factor']}")

    if unprofitable:
        print(f"\nUNPROFITABLE PAIRS ({len(unprofitable)}) - DISABLE THESE:")
        for pair in unprofitable:
            print(f"  - {pair}: No strategy profitable after fees")

    print("\n" + "=" * 80)
    print("Next step: Update crypto_paper_config.py with these settings")
    print("=" * 80)


if __name__ == "__main__":
    main()
