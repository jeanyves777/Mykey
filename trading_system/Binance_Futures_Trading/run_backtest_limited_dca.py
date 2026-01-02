#!/usr/bin/env python3
"""
Backtest with Limited DCA (0-2 only)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_enhanced_boost import EnhancedBoostBacktester

def run_limited_dca_test():
    """Test with only DCA 0-2 (no DCA 3-4)"""
    BACKTEST_DAYS = 90
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT"]

    print("="*80)
    print(f"BACKTEST WITH LIMITED DCA (0-2 ONLY) - {BACKTEST_DAYS} DAYS")
    print("="*80)
    print("DCA Levels: 0, 1, 2 only (removed 3 and 4)")
    print("="*80)

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        backtester = EnhancedBoostBacktester(symbol, start_balance=100.0)
        df = backtester.get_historical_data(days=BACKTEST_DAYS, interval="1h")

        if df is not None and len(df) > 0:
            result = backtester.run_backtest(df)
            result["symbol"] = symbol
            all_results.append(result)
        else:
            print(f"ERROR: No data for {symbol}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - LIMITED DCA (0-2 ONLY)")
    print("="*80)

    header = f"{'Symbol':<12} {'Start':>10} {'End':>10} {'P&L':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>10} {'Liqs':>6}"
    print(header)
    print("-"*86)

    for r in all_results:
        pnl = r['balance'] - 100
        liqs = r.get('liquidations_total', 0)
        row = f"{r['symbol']:<12} ${100:>9.2f} ${r['balance']:>9.2f} ${pnl:>+9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>9.1f}% {liqs:>5}"
        print(row)

    print("-"*86)
    total_start = len(all_results) * 100
    total_end = sum(r['balance'] for r in all_results)
    total_pnl = total_end - total_start
    total_return = (total_end / total_start - 1) * 100 if total_start > 0 else 0
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_winrate = sum(r['win_rate'] for r in all_results) / len(all_results) if all_results else 0
    max_dd = max(r['max_drawdown'] for r in all_results) if all_results else 0
    total_liqs = sum(r.get('liquidations_total', 0) for r in all_results)

    total_row = f"{'TOTAL':<12} ${total_start:>9.2f} ${total_end:>9.2f} ${total_pnl:>+9.2f} {total_return:>+9.1f}% {total_trades:>7} {avg_winrate:>7.1f}% {max_dd:>9.1f}% {total_liqs:>5}"
    print(total_row)

    return all_results


if __name__ == "__main__":
    run_limited_dca_test()
