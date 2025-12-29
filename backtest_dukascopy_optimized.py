"""
Backtest the Optimized Strategy with Dukascopy 1-minute data
Uses the same strategy logic from optimized_strategy.py
Configuration loaded from optimized_paper_config.py
Data Period: 16 months (Aug 2023 - Nov 2024)
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from pathlib import Path

# Import config (shared with paper trading)
from trading_system.Forex_Trading.config import optimized_paper_config as config

# Import strategy logic
from trading_system.Forex_Trading.strategies.optimized_strategy import (
    calculate_indicators,
    get_signal,
    print_strategy_info
)

# Data directory for Dukascopy data
DUKASCOPY_DATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_Dukascopy')

# Print config info
print_strategy_info(config)
print("\nBACKTEST MODE - Using Dukascopy 1-minute data (16 months)")
print("=" * 70)


def load_dukascopy_data(pair):
    """Load 1-minute CSV data from Dukascopy download"""
    csv_path = DUKASCOPY_DATA_DIR / f"{pair}_M1.csv"

    if not csv_path.exists():
        print(f"  WARNING: Data file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Rename columns to match our format
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})

    # Calculate indicators using the same function as live trading
    df = calculate_indicators(df)

    return df


def backtest_pair(pair):
    """Backtest a single pair using strategy from optimized_strategy.py"""
    df = load_dukascopy_data(pair)

    if df is None or len(df) < 100:
        return []

    settings = config.get_pair_settings(pair)

    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    pip_mult = config.PIP_MULTIPLIERS[pair]

    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult

    trades = []
    cooldown = 0

    print(f"\nBacktesting {pair} ({settings['strategy']})...")
    print(f"  Data: {len(df):,} candles")
    print(f"  TP: {tp_pips} pips | SL: {sl_pips} pips")

    for i in range(50, len(df) - 100):
        if i < cooldown:
            continue

        # Create a slice of dataframe up to current bar (simulating live)
        df_slice = df.iloc[:i+1].copy()

        # Get signal using the same function as live trading
        signal, reason = get_signal(pair, df_slice, config)

        if signal is None:
            continue

        entry = df.iloc[i]['close']
        entry_time = df.iloc[i].get('timestamp', df.iloc[i].get('time', i))

        # Simulate trade outcome - look ahead up to 200 bars (200 minutes max)
        for j in range(i + 1, min(i + 200, len(df))):
            h = df.iloc[j]['high']
            l = df.iloc[j]['low']

            if signal == 'BUY':
                if h >= entry + tp_dist:
                    trades.append({
                        'pair': pair,
                        'direction': 'BUY',
                        'entry_time': entry_time,
                        'entry': entry,
                        'exit': entry + tp_dist,
                        'pips': tp_pips,
                        'result': 'WIN'
                    })
                    cooldown = j + 30  # 30 min cooldown
                    break
                if l <= entry - sl_dist:
                    trades.append({
                        'pair': pair,
                        'direction': 'BUY',
                        'entry_time': entry_time,
                        'entry': entry,
                        'exit': entry - sl_dist,
                        'pips': -sl_pips,
                        'result': 'LOSS'
                    })
                    cooldown = j + 30
                    break
            else:  # SELL
                if l <= entry - tp_dist:
                    trades.append({
                        'pair': pair,
                        'direction': 'SELL',
                        'entry_time': entry_time,
                        'entry': entry,
                        'exit': entry - tp_dist,
                        'pips': tp_pips,
                        'result': 'WIN'
                    })
                    cooldown = j + 30
                    break
                if h >= entry + sl_dist:
                    trades.append({
                        'pair': pair,
                        'direction': 'SELL',
                        'entry_time': entry_time,
                        'entry': entry,
                        'exit': entry + sl_dist,
                        'pips': -sl_pips,
                        'result': 'LOSS'
                    })
                    cooldown = j + 30
                    break

    return trades


# Check if data exists
print("\nChecking for Dukascopy data...")
data_files = list(DUKASCOPY_DATA_DIR.glob("*_M1.csv"))
if not data_files:
    print(f"\nNo data files found in {DUKASCOPY_DATA_DIR}")
    print("\nPlease run download_dukascopy_data.py first to download the data.")
    print("  Usage: python download_dukascopy_data.py")
    sys.exit(1)

print(f"Found {len(data_files)} data files:")
for f in data_files:
    print(f"  {f.name}")

# Run backtest for all pairs
all_trades = []
pair_results = {}

for pair in config.OPTIMIZED_PAIRS:
    trades = backtest_pair(pair)
    all_trades.extend(trades)

    if trades:
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        total_pips = sum(t['pips'] for t in trades)
        win_rate = wins / len(trades) * 100

        pair_results[pair] = {
            'trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate': win_rate,
            'total_pips': total_pips
        }

        print(f"  Trades: {len(trades)} | Wins: {wins} ({win_rate:.1f}%) | P&L: {total_pips:+.0f} pips")
    else:
        print(f"  No trades (data may be missing)")
        pair_results[pair] = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0}

# Summary
print("\n" + "=" * 70)
print("BACKTEST SUMMARY - OPTIMIZED STRATEGY (DUKASCOPY 1-MIN DATA)")
print("=" * 70)
print(f"\n{'PAIR':<10} {'STRATEGY':<15} {'TP':<4} {'SL':<4} {'TRADES':<8} {'WIN%':<8} {'P&L'}")
print("-" * 70)

total_trades = 0
total_wins = 0
total_pips = 0

for pair in config.OPTIMIZED_PAIRS:
    settings = config.get_pair_settings(pair)
    r = pair_results[pair]
    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']

    print(f"{pair:<10} {settings['strategy']:<15} {settings['tp_pips']:<4} {settings['sl_pips']:<4} "
          f"{r['trades']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:+.0f}p")

print("-" * 70)
overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {'':<15} {'':<4} {'':<4} {total_trades:<8} {overall_wr:<7.1f}% {total_pips:+.0f}p")
print("=" * 70)

# Profit factor and additional stats
if all_trades:
    wins_pips = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    losses_pips = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    profit_factor = wins_pips / losses_pips if losses_pips > 0 else 0

    print(f"\nTotal Winning Pips: +{wins_pips:.0f}")
    print(f"Total Losing Pips: -{losses_pips:.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Net P&L: {total_pips:+.0f} pips (${total_pips:+.2f} at $1/pip)")

    # Monthly breakdown if we have timestamps
    print("\n" + "=" * 70)
    print("TRADE DISTRIBUTION")
    print("=" * 70)

    buy_count = sum(1 for t in all_trades if t['direction'] == 'BUY')
    sell_count = sum(1 for t in all_trades if t['direction'] == 'SELL')
    print(f"BUY trades: {buy_count} ({buy_count/len(all_trades)*100:.1f}%)")
    print(f"SELL trades: {sell_count} ({sell_count/len(all_trades)*100:.1f}%)")

    # By pair
    print(f"\nBy Pair:")
    for pair in config.OPTIMIZED_PAIRS:
        pair_trades = [t for t in all_trades if t['pair'] == pair]
        if pair_trades:
            pair_wins = sum(1 for t in pair_trades if t['result'] == 'WIN')
            print(f"  {pair}: {len(pair_trades)} trades, {pair_wins/len(pair_trades)*100:.1f}% win rate")
