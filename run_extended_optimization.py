#!/usr/bin/env python3
"""
Extended Optimization - Test NEW Strategies on Unprofitable Pairs
=================================================================
Tests 8 NEW strategies on the 11 pairs that weren't profitable
with the original strategies.

New Strategies:
- MOMENTUM_BREAKOUT: Strong price move with trend confirmation
- RSI_DIVERGENCE: RSI divergence from price
- BOLLINGER_SQUEEZE: Volatility breakout
- EMA_CROSSOVER: Classic EMA 9/21 crossover
- RSI_TREND_FOLLOW: RSI pullback in trend
- MACD_HISTOGRAM: MACD histogram reversal
- SWING_REVERSAL: Swing high/low reversal
- VOLUME_BREAKOUT: Volume-confirmed breakout
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Crypto_Margin_Trading'))

from engine.crypto_backtest_engine_fast import FastCryptoBacktestEngine

# Pairs that were unprofitable with original strategies
UNPROFITABLE_PAIRS = [
    'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT',
    'LINKUSDT', 'AVAXUSDT', 'SUIUSDT', 'DOTUSDT', 'ZECUSDT', 'PEPEUSDT',
]

# New strategies to test
NEW_STRATEGIES = [
    'MOMENTUM_BREAKOUT',
    'RSI_DIVERGENCE',
    'BOLLINGER_SQUEEZE',
    'EMA_CROSSOVER',
    'RSI_TREND_FOLLOW',
    'MACD_HISTOGRAM',
    'SWING_REVERSAL',
    'VOLUME_BREAKOUT',
]


def main():
    print("\n" + "=" * 80)
    print("EXTENDED OPTIMIZATION - NEW STRATEGIES ON UNPROFITABLE PAIRS")
    print("=" * 80)
    print(f"\nTesting {len(NEW_STRATEGIES)} NEW strategies on {len(UNPROFITABLE_PAIRS)} pairs")
    print(f"Strategies: {', '.join(NEW_STRATEGIES)}")
    print(f"\nPairs: {', '.join(UNPROFITABLE_PAIRS)}")
    print("=" * 80)

    engine = FastCryptoBacktestEngine(include_fees=True)

    # Test with wider TP/SL ranges to find larger moves
    tp_range = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    sl_range = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    all_results = {}

    for pair in UNPROFITABLE_PAIRS:
        print(f"\n{'='*70}")
        print(f"Testing NEW strategies on {pair}")
        print(f"{'='*70}")

        results = engine.optimize_pair(
            pair=pair,
            strategies=NEW_STRATEGIES,
            tp_range=tp_range,
            sl_range=sl_range,
            min_win_rate=55.0,  # Slightly lower WR threshold
            min_profit_factor=1.15,  # Slightly lower PF threshold
            top_n=5,
        )
        all_results[pair] = results

    # Summary
    print("\n" + "=" * 90)
    print("EXTENDED OPTIMIZATION SUMMARY")
    print("=" * 90)

    newly_profitable = []
    still_unprofitable = []

    for pair, results in all_results.items():
        if results and results[0].get('net_pnl', 0) > 0 and results[0].get('validated', False):
            best = results[0]
            newly_profitable.append({
                'pair': pair,
                'strategy': best['strategy'],
                'tp_pct': best['tp_pct'],
                'sl_pct': best['sl_pct'],
                'win_rate': best['win_rate'],
                'profit_factor': best['profit_factor'],
                'net_pnl': best['net_pnl'],
            })
        else:
            still_unprofitable.append(pair)

    if newly_profitable:
        print(f"\nNEWLY PROFITABLE PAIRS ({len(newly_profitable)}):")
        print("-" * 90)
        print(f"{'PAIR':<12} {'STRATEGY':<18} {'TP%':<5} {'SL%':<5} {'WIN%':<6} {'PF':<6} {'NET':<10}")
        print("-" * 90)
        for r in newly_profitable:
            print(f"{r['pair']:<12} {r['strategy']:<18} {r['tp_pct']:<5.1f} {r['sl_pct']:<5.1f} "
                  f"{r['win_rate']:<6.1f} {r['profit_factor']:<6.2f} ${r['net_pnl']:<9.0f}")
    else:
        print("\nNo new profitable configurations found.")

    if still_unprofitable:
        print(f"\nSTILL UNPROFITABLE ({len(still_unprofitable)}):")
        print(f"  {', '.join(still_unprofitable)}")
        print("  These pairs may not be suitable for short-term trading with current fees.")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
