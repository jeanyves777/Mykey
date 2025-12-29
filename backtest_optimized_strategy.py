"""
Backtest the Optimized Strategy with local CSV data (1-minute)
Uses the same strategy logic from optimized_strategy.py
Configuration loaded from optimized_paper_config.py

Includes Session + Volume + Trend Filters
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np

# Import config (shared with paper trading)
from trading_system.Forex_Trading.config import optimized_paper_config as config

# Import strategy logic
from trading_system.Forex_Trading.strategies.optimized_strategy import (
    calculate_indicators,
    get_signal,
    print_strategy_info
)

# Print config info
print_strategy_info(config)

# Get typical SL from first pair for display
first_pair = config.OPTIMIZED_PAIRS[0]
typical_sl = config.get_pair_settings(first_pair)['sl_pips']

print(f"\nBACKTEST MODE - Using 1-minute CSV data")
print(f"  - Risk Example: {typical_sl} pip SL = ${typical_sl} risk per trade")
print(f"  - Cooldown: {config.COOLDOWN_MINUTES} min between trades per symbol")
print("=" * 80)


def load_csv_data(pair):
    """Load 1-minute CSV data for a pair"""
    prefix = config.CSV_PREFIXES.get(pair)
    if not prefix:
        print(f"  WARNING: No CSV prefix for {pair}, skipping")
        return None

    csv_path = config.DATA_DIR / f"{prefix}1.csv"

    if not csv_path.exists():
        print(f"  WARNING: CSV file not found for {pair}: {csv_path}")
        return None

    df = pd.read_csv(csv_path, sep='\t',
                     names=['time', 'open', 'high', 'low', 'close', 'volume'])

    # Parse time to get hour for session filter
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour

    # Use the same calculate_indicators from strategy file
    df = calculate_indicators(df)

    return df


def backtest_pair(pair):
    """Backtest a single pair using strategy from optimized_strategy.py"""
    df = load_csv_data(pair)
    if df is None:
        return []

    settings = config.get_pair_settings(pair)

    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    pip_mult = config.PIP_MULTIPLIERS.get(pair, 10000)
    cooldown_bars = config.COOLDOWN_MINUTES  # 1 bar = 1 minute

    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult

    trades = []
    cooldown = 0

    # Get filter settings
    volume_filter = settings.get('volume_filter', False)
    trend_filter = settings.get('trend_filter', False)
    session = settings.get('session', 'ALL')

    filter_info = []
    if volume_filter:
        filter_info.append("Vol")
    if trend_filter:
        filter_info.append("Trend")
    if session != 'ALL':
        filter_info.append(session)
    filter_str = f" [{'+'.join(filter_info)}]" if filter_info else ""

    print(f"\nBacktesting {pair} ({settings['strategy']}{filter_str})...")
    print(f"  Data: {len(df):,} candles | TP: {tp_pips}p | SL: {sl_pips}p")

    for i in range(50, len(df) - 100):
        if i < cooldown:
            continue

        # Session filter check
        if hasattr(config, 'is_allowed_hour'):
            hour = df.iloc[i]['hour']
            if not config.is_allowed_hour(pair, hour):
                continue

        # Create a slice of dataframe up to current bar (simulating live)
        df_slice = df.iloc[:i+1].copy()

        # Get signal using the same function as live trading (includes volume/trend filters)
        signal, reason = get_signal(pair, df_slice, config)

        if signal is None:
            continue

        entry = df.iloc[i]['close']
        entry_time = df.iloc[i]['time']

        # Simulate trade outcome
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
                    cooldown = j + cooldown_bars
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
                    cooldown = j + cooldown_bars
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
                    cooldown = j + cooldown_bars
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
                    cooldown = j + cooldown_bars
                    break

    return trades


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
        print(f"  No trades")
        pair_results[pair] = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0}

# Summary
print("\n" + "=" * 80)
print("BACKTEST SUMMARY - OPTIMIZED STRATEGY (Session + Volume + Trend Filters)")
print("=" * 80)
print(f"\n{'PAIR':<10} {'STRATEGY':<14} {'TP':<4} {'SL':<4} {'SESSION':<10} {'VOL':<4} {'TRD':<4} {'TRADES':<7} {'WIN%':<7} {'P&L'}")
print("-" * 85)

total_trades = 0
total_wins = 0
total_pips = 0

for pair in config.OPTIMIZED_PAIRS:
    settings = config.get_pair_settings(pair)
    r = pair_results[pair]
    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']

    vol = 'ON' if settings.get('volume_filter', False) else 'OFF'
    trd = 'ON' if settings.get('trend_filter', False) else 'OFF'
    session = settings.get('session', 'ALL')

    print(f"{pair:<10} {settings['strategy']:<14} {settings['tp_pips']:<4} {settings['sl_pips']:<4} "
          f"{session:<10} {vol:<4} {trd:<4} {r['trades']:<7} {r['win_rate']:<6.1f}% {r['total_pips']:+.0f}p")

print("-" * 85)
overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {'':<14} {'':<4} {'':<4} {'':<10} {'':<4} {'':<4} {total_trades:<7} {overall_wr:<6.1f}% {total_pips:+.0f}p")
print("=" * 80)

# Profit factor
if all_trades:
    wins_pips = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    losses_pips = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    profit_factor = wins_pips / losses_pips if losses_pips > 0 else 0

    print(f"\nTotal Winning Pips: +{wins_pips:.0f}")
    print(f"Total Losing Pips: -{losses_pips:.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Net P&L: {total_pips:+.0f} pips (${total_pips:+.2f} at $1/pip)")
