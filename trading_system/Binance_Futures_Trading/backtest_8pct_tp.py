#!/usr/bin/env python3
"""
Backtest with 8% TP for BNB, ETH, DOT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_enhanced_boost import EnhancedBoostBacktester

def run_8pct_tp_test():
    """Test with 8% TP for selected pairs"""
    BACKTEST_DAYS = 90
    symbols = ["BNBUSDT", "ETHUSDT", "DOTUSDT"]

    print("="*80)
    print(f"BACKTEST WITH 8% TP - {BACKTEST_DAYS} DAYS")
    print("="*80)
    print("TP ROI: 8% (current setting)")
    print("="*80)

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol} with 8% TP")
        print(f"{'='*80}")

        backtester = EnhancedBoostBacktester(symbol, start_balance=100.0)

        # Override TP to 8%
        backtester.tp_roi = 0.08  # 8% TP
        print(f"[{symbol}] TP Override: {backtester.tp_roi*100:.0f}%")

        df = backtester.get_historical_data(days=BACKTEST_DAYS, interval="1h")

        if df is not None and len(df) > 0:
            result = backtester.run_backtest(df)
            result["symbol"] = symbol
            result["price_change"] = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            all_results.append(result)
        else:
            print(f"ERROR: No data for {symbol}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - 8% TP BACKTEST")
    print("="*80)
    print(f"{'Symbol':<12} {'Start':>10} {'End':>10} {'P&L':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>10}")
    print("-"*80)

    for r in all_results:
        pnl = r['balance'] - 100
        print(f"{r['symbol']:<12} ${100:>9.2f} ${r['balance']:>9.2f} ${pnl:>+9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>9.1f}%")

    print("-"*80)
    total_start = len(all_results) * 100
    total_end = sum(r['balance'] for r in all_results)
    total_pnl = total_end - total_start
    total_return = (total_end / total_start - 1) * 100
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_winrate = sum(r['win_rate'] for r in all_results) / len(all_results)
    max_dd = max(r['max_drawdown'] for r in all_results)

    print(f"{'TOTAL':<12} ${total_start:>9.2f} ${total_end:>9.2f} ${total_pnl:>+9.2f} {total_return:>+9.1f}% {total_trades:>7} {avg_winrate:>7.1f}% {max_dd:>9.1f}%")

    return all_results


if __name__ == "__main__":
    run_8pct_tp_test()
