"""
Quick Test: AUD_CHF, EUR_GBP optimization + USD_JPY analysis
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import strategy
from trading_system.Forex_Trading.strategies.optimized_strategy_V2 import (
    calculate_indicators,
    check_rsi_reversal_signal,
    check_rsi_extreme_signal,
    check_macd_cross_signal,
    check_macd_zero_signal,
    check_psar_adx_rsi_signal,
)

# Map strategy names to functions
STRATEGY_FUNCTIONS = {
    'RSI_REVERSAL': check_rsi_reversal_signal,
    'RSI_EXTREME': check_rsi_extreme_signal,
    'MACD_CROSS': check_macd_cross_signal,
    'MACD_ZERO': check_macd_zero_signal,
    'PSAR_ADX_RSI_MTF': check_psar_adx_rsi_signal,
}

DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data")

# Test pairs
TEST_PAIRS = ['AUD_CHF', 'EUR_GBP', 'USD_JPY']

CSV_PREFIXES = {
    'AUD_CHF': 'AUDCHF',
    'EUR_GBP': 'EURGBP',
    'USD_JPY': 'USDJPY',
}

PIP_MULTIPLIERS = {
    'AUD_CHF': 10000,
    'EUR_GBP': 10000,
    'USD_JPY': 100,
}

SPREAD_PIPS = {
    'AUD_CHF': 1.5,
    'EUR_GBP': 1.0,
    'USD_JPY': 0.9,
}

SESSIONS = {
    'ASIAN': list(range(0, 8)),
    'LONDON': list(range(8, 16)),
    'NEW_YORK': list(range(13, 21)),
    'ALL': list(range(0, 24)),
}

# Strategies to test (top performers from main optimization)
STRATEGIES_TO_TEST = ['RSI_REVERSAL', 'RSI_EXTREME', 'MACD_CROSS', 'MACD_ZERO', 'PSAR_ADX_RSI_MTF']

# TP/SL combinations
TP_VALUES = [4, 5, 6, 8]
SL_VALUES = [10, 15, 20]


def load_data(pair: str) -> pd.DataFrame:
    """Load and prepare data for a pair."""
    prefix = CSV_PREFIXES[pair]
    csv_path = DATA_DIR / f"{prefix}1.csv"

    if not csv_path.exists():
        print(f"  {pair}: CSV not found at {csv_path}")
        return None

    # Try loading with header first
    df = pd.read_csv(csv_path, nrows=100000)

    # Check if first column looks like a date (no header case)
    first_col = df.columns[0]
    if '-' in str(first_col) and ':' in str(first_col):
        # No header - reload with column names
        df = pd.read_csv(csv_path, nrows=100000, header=None,
                         names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                         sep='\t')
    else:
        # Standardize columns
        df.columns = df.columns.str.lower()
        if 'time' in df.columns:
            df.rename(columns={'time': 'datetime'}, inplace=True)
        elif 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['hour'] = df['datetime'].dt.hour

    # Calculate indicators
    df = calculate_indicators(df)

    return df


def run_backtest(pair: str, df: pd.DataFrame, strategy: str, tp_pips: int, sl_pips: int,
                 session: str = 'ALL', include_spread: bool = True) -> dict:
    """Run backtest for a specific configuration."""
    pip_mult = PIP_MULTIPLIERS[pair]
    spread = SPREAD_PIPS[pair] if include_spread else 0
    allowed_hours = set(SESSIONS.get(session, list(range(0, 24))))

    trades = []
    i = 100  # Start after warmup

    while i < len(df) - 50:
        row = df.iloc[i]

        # Session filter
        if row['hour'] not in allowed_hours:
            i += 1
            continue

        # Get signal using strategy function directly
        strategy_func = STRATEGY_FUNCTIONS.get(strategy)
        if not strategy_func:
            i += 1
            continue
        signal, _ = strategy_func(df.iloc[:i+1])

        if signal is None:
            i += 1
            continue

        # Entry
        entry_price = row['close']
        entry_time = row['datetime']

        # Calculate TP/SL
        pip_value = 0.01 if 'JPY' in pair else 0.0001

        if signal == 'BUY':
            tp_price = entry_price + (tp_pips * pip_value)
            sl_price = entry_price - (sl_pips * pip_value)
        else:
            tp_price = entry_price - (tp_pips * pip_value)
            sl_price = entry_price + (sl_pips * pip_value)

        # Simulate trade
        for j in range(i + 1, min(i + 500, len(df))):
            candle = df.iloc[j]

            if signal == 'BUY':
                if candle['low'] <= sl_price:
                    pnl = -sl_pips - spread
                    trades.append({'result': 'LOSS', 'pnl': pnl, 'time': entry_time, 'hour': row['hour']})
                    i = j + 5
                    break
                elif candle['high'] >= tp_price:
                    pnl = tp_pips - spread
                    trades.append({'result': 'WIN', 'pnl': pnl, 'time': entry_time, 'hour': row['hour']})
                    i = j + 5
                    break
            else:  # SELL
                if candle['high'] >= sl_price:
                    pnl = -sl_pips - spread
                    trades.append({'result': 'LOSS', 'pnl': pnl, 'time': entry_time, 'hour': row['hour']})
                    i = j + 5
                    break
                elif candle['low'] <= tp_price:
                    pnl = tp_pips - spread
                    trades.append({'result': 'WIN', 'pnl': pnl, 'time': entry_time, 'hour': row['hour']})
                    i = j + 5
                    break
        else:
            i += 1
            continue

        i += 1

    # Calculate stats
    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0, 'pf': 0}

    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = sum(1 for t in trades if t['result'] == 'LOSS')
    total_pips = sum(t['pnl'] for t in trades)
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'total_pips': total_pips,
        'pf': pf,
        'trades_list': trades
    }


def optimize_pair(pair: str, df: pd.DataFrame) -> dict:
    """Find best configuration for a pair."""
    print(f"\n{'='*60}")
    print(f"  OPTIMIZING {pair}")
    print(f"{'='*60}")

    best_result = None
    best_score = -999
    all_results = []

    total_combos = len(STRATEGIES_TO_TEST) * len(TP_VALUES) * len(SL_VALUES) * 4
    combo_count = 0

    for strategy in STRATEGIES_TO_TEST:
        print(f"\n  [{strategy}]")
        for tp in TP_VALUES:
            for sl in SL_VALUES:
                print(f"    TP:{tp} SL:{sl} -> ", end="", flush=True)
                session_results = []
                for session in ['ALL', 'ASIAN', 'LONDON', 'NEW_YORK']:
                    combo_count += 1
                    result = run_backtest(pair, df, strategy, tp, sl, session)

                    if result['trades'] >= 10:
                        # Score = expectancy * sqrt(trades) for statistical significance
                        expectancy = result['total_pips'] / result['trades'] if result['trades'] > 0 else 0
                        score = expectancy * (result['trades'] ** 0.5) * result['pf']

                        result['strategy'] = strategy
                        result['tp'] = tp
                        result['sl'] = sl
                        result['session'] = session
                        result['score'] = score

                        all_results.append(result)
                        session_results.append(f"{session[:3]}:{result['trades']}t/{result['win_rate']:.0f}%/{result['total_pips']:+.0f}p")

                        if score > best_score and result['win_rate'] >= 75 and result['total_pips'] > 0:
                            best_score = score
                            best_result = result
                    else:
                        session_results.append(f"{session[:3]}:{result['trades']}t")

                print(" | ".join(session_results))

    # Sort by score
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Print top 5
    print(f"\nTop 5 configurations for {pair}:")
    print(f"{'STRATEGY':<20} {'TP/SL':<8} {'SESSION':<10} {'TRADES':<8} {'WR%':<8} {'PF':<8} {'P&L':<10}")
    print("-" * 80)

    for r in all_results[:5]:
        print(f"{r['strategy']:<20} {r['tp']}/{r['sl']:<5} {r['session']:<10} {r['trades']:<8} {r['win_rate']:.1f}%   {r['pf']:.2f}     {r['total_pips']:+.0f}p")

    if best_result:
        print(f"\n*** BEST: {best_result['strategy']} TP:{best_result['tp']} SL:{best_result['sl']} {best_result['session']}")
        print(f"    {best_result['trades']} trades | {best_result['win_rate']:.1f}% WR | PF {best_result['pf']:.2f} | {best_result['total_pips']:+.0f} pips")

    return best_result


def analyze_usd_jpy_sessions(df: pd.DataFrame):
    """Deep analysis of USD_JPY by session."""
    print(f"\n{'='*60}")
    print(f"  USD_JPY DEEP ANALYSIS")
    print(f"{'='*60}")

    # Test current config: RSI_REVERSAL, NEW_YORK, 5/15
    print("\n--- Current Config (RSI_REVERSAL, NEW_YORK, TP:5/SL:15) ---")
    current = run_backtest('USD_JPY', df, 'RSI_REVERSAL', 5, 15, 'NEW_YORK')
    print(f"  Trades: {current['trades']} | WR: {current['win_rate']:.1f}% | P&L: {current['total_pips']:+.0f} pips")

    # Analyze by hour
    if current.get('trades_list'):
        print("\n  Breakdown by hour (NEW_YORK session):")
        hour_stats = {}
        for t in current['trades_list']:
            h = t['hour']
            if h not in hour_stats:
                hour_stats[h] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t['result'] == 'WIN':
                hour_stats[h]['wins'] += 1
            else:
                hour_stats[h]['losses'] += 1
            hour_stats[h]['pnl'] += t['pnl']

        print(f"  {'HOUR':<6} {'WINS':<6} {'LOSSES':<8} {'WR%':<8} {'P&L':<10}")
        for h in sorted(hour_stats.keys()):
            s = hour_stats[h]
            total = s['wins'] + s['losses']
            wr = s['wins'] / total * 100 if total > 0 else 0
            print(f"  {h:02d}:00  {s['wins']:<6} {s['losses']:<8} {wr:.1f}%    {s['pnl']:+.0f}p")

    # Test new optimized config: RSI_REVERSAL, NEW_YORK, 8/20
    print("\n--- New Optimized Config (RSI_REVERSAL, NEW_YORK, TP:8/SL:20) ---")
    new_config = run_backtest('USD_JPY', df, 'RSI_REVERSAL', 8, 20, 'NEW_YORK')
    print(f"  Trades: {new_config['trades']} | WR: {new_config['win_rate']:.1f}% | P&L: {new_config['total_pips']:+.0f} pips")

    # Test all sessions with 8/20
    print("\n--- Session Comparison (RSI_REVERSAL, TP:8/SL:20) ---")
    for session in ['ALL', 'ASIAN', 'LONDON', 'NEW_YORK']:
        result = run_backtest('USD_JPY', df, 'RSI_REVERSAL', 8, 20, session)
        print(f"  {session:<10}: {result['trades']:>4} trades | {result['win_rate']:.1f}% WR | PF {result['pf']:.2f} | {result['total_pips']:+.0f}p")

    # Test alternative strategies
    print("\n--- Alternative Strategies (TP:8/SL:20, NEW_YORK) ---")
    for strategy in STRATEGIES_TO_TEST:
        result = run_backtest('USD_JPY', df, strategy, 8, 20, 'NEW_YORK')
        if result['trades'] >= 10:
            print(f"  {strategy:<20}: {result['trades']:>4} trades | {result['win_rate']:.1f}% WR | {result['total_pips']:+.0f}p")


def main():
    print("="*60)
    print("  TESTING AUD_CHF, EUR_GBP + USD_JPY ANALYSIS")
    print("="*60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    for pair in TEST_PAIRS:
        print(f"\nLoading {pair}...")
        df = load_data(pair)

        if df is None:
            print(f"  FAILED to load {pair}")
            continue

        print(f"  Loaded {len(df):,} candles")

        if pair == 'USD_JPY':
            analyze_usd_jpy_sessions(df)

        best = optimize_pair(pair, df)
        if best:
            results[pair] = best

    # Final summary
    print("\n" + "="*60)
    print("  FINAL RECOMMENDATIONS")
    print("="*60)

    for pair, r in results.items():
        print(f"\n{pair}:")
        print(f"  Strategy: {r['strategy']}")
        print(f"  TP/SL: {r['tp']}/{r['sl']} pips")
        print(f"  Session: {r['session']}")
        print(f"  Performance: {r['trades']} trades | {r['win_rate']:.1f}% WR | {r['total_pips']:+.0f} pips")


if __name__ == '__main__':
    main()
