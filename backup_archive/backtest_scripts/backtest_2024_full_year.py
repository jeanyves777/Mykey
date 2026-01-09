"""
BACKTEST: FULL YEAR 2024 - EXACT MATCH TO LIVE STRATEGY
========================================================
Uses the EXACT same logic as run_optimized_forex_paper.py
with historical data from HistData.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.config import optimized_paper_config as config
from trading_system.Forex_Trading.strategies.optimized_strategy import calculate_indicators, get_signal

import pandas as pd
from datetime import datetime
from pathlib import Path

# Data directory
DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024")

# Map instrument names to file prefixes
FILE_MAP = {
    'EUR_USD': 'DAT_MT_EURUSD_M1_2024.csv',
    'GBP_USD': 'DAT_MT_GBPUSD_M1_2024.csv',
    'USD_CHF': 'DAT_MT_USDCHF_M1_2024.csv',
    'USD_CAD': 'DAT_MT_USDCAD_M1_2024.csv',
    'NZD_USD': 'DAT_MT_NZDUSD_M1_2024.csv',
    'AUD_JPY': 'DAT_MT_AUDJPY_M1_2024.csv',
    'USD_JPY': 'DAT_MT_USDJPY_M1_2024.csv',
    'EUR_GBP': 'DAT_MT_EURGBP_M1_2024.csv',
    'AUD_CHF': 'DAT_MT_AUDCHF_M1_2024.csv',
    'EUR_CAD': 'DAT_MT_EURCAD_M1_2024.csv',
}

# Spread simulation (typical spreads)
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}


def load_histdata_csv(pair: str) -> pd.DataFrame:
    """Load HistData CSV file for a pair."""
    filename = FILE_MAP.get(pair)
    if not filename:
        print(f"  No data file for {pair}")
        return pd.DataFrame()

    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return pd.DataFrame()

    # HistData format: date,time,open,high,low,close,volume
    # No header in file
    df = pd.read_csv(filepath, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])

    # Combine date and time into timestamp
    df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df = df.drop(columns=['date', 'volume'])
    df = df.sort_values('time').reset_index(drop=True)

    return df


def backtest_exact_match(df: pd.DataFrame, pair: str) -> dict:
    """
    Backtest that EXACTLY matches live trading behavior:
    - ONE position at a time
    - Per-pair cooldown (from config)
    - Per-pair session filter (from config)
    """
    if len(df) < 100:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0, 'tp': 0, 'sl': 0}

    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    cooldown_minutes = settings.get('cooldown_minutes', config.COOLDOWN_MINUTES)

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    # Calculate all indicators
    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    tp_price = 0
    sl_price = 0
    entry_time = None
    last_exit_time = None

    # Track monthly stats
    monthly_stats = {}

    for i in range(50, len(df)):
        row = df.iloc[i]
        current_time = row['time']
        hour_utc = current_time.hour
        month_key = current_time.strftime('%Y-%m')

        if in_position:
            # CHECK EXIT CONDITIONS
            high = row['high']
            low = row['low']

            exit_reason = None
            exit_pnl = 0

            if direction == 'BUY':
                if low <= sl_price:
                    exit_reason = 'SL'
                    exit_pnl = -sl_pips
                elif high >= tp_price:
                    exit_reason = 'TP'
                    exit_pnl = tp_pips
            else:  # SELL
                if high >= sl_price:
                    exit_reason = 'SL'
                    exit_pnl = -sl_pips
                elif low <= tp_price:
                    exit_reason = 'TP'
                    exit_pnl = tp_pips

            if exit_reason:
                won = exit_pnl > 0
                trades.append({
                    'direction': direction,
                    'pnl': exit_pnl,
                    'won': won,
                    'exit_reason': exit_reason,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'month': entry_time.strftime('%Y-%m')
                })
                in_position = False
                last_exit_time = current_time

                # Update monthly stats
                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {'trades': 0, 'wins': 0, 'pips': 0}
                monthly_stats[month_key]['trades'] += 1
                monthly_stats[month_key]['pips'] += exit_pnl
                if won:
                    monthly_stats[month_key]['wins'] += 1

        else:
            # NOT IN POSITION - CHECK FOR ENTRY

            # SESSION FILTER (per-pair)
            if hasattr(config, 'is_allowed_hour'):
                if not config.is_allowed_hour(pair, hour_utc):
                    continue

            # COOLDOWN CHECK (per-pair, after last EXIT)
            if cooldown_minutes > 0 and last_exit_time is not None:
                time_since = (current_time - last_exit_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            # GET SIGNAL using EXACT same function as live trading
            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal is not None:
                in_position = True
                direction = signal
                entry_time = current_time

                if signal == 'BUY':
                    entry_price = row['close'] + spread
                    tp_price = entry_price + tp_price_diff
                    sl_price = entry_price - sl_price_diff
                else:
                    entry_price = row['close'] - spread
                    tp_price = entry_price - tp_price_diff
                    sl_price = entry_price + sl_price_diff

    # Calculate results
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pips': 0, 'tp': 0, 'sl': 0, 'monthly': {}
        }

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    tp_exits = sum(1 for t in trades if t['exit_reason'] == 'TP')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'SL')

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp': tp_exits,
        'sl': sl_exits,
        'monthly': monthly_stats
    }


# =============================================================================
# MAIN BACKTEST
# =============================================================================
print("=" * 80)
print("BACKTEST: FULL YEAR 2024 - EXACT MATCH TO LIVE STRATEGY")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nConfiguration (2024-12-16 OPTIMIZED):")
print(f"  - Strategy: optimized_strategy.py (get_signal)")
print(f"  - Config: optimized_paper_config.py")
print(f"  - ONE position per symbol at a time")
print(f"  - Per-pair session filters")
print(f"  - Per-pair cooldown (0 or 30 minutes)")
print(f"  - Data: HistData 2024 M1 candles (full year)")
print("=" * 80)

# Print settings table
print(f"\n{'PAIR':<10} {'STRATEGY':<14} {'TP':<4} {'SL':<4} {'CD':<5} {'SF':<4}")
print("-" * 50)
for pair in config.OPTIMIZED_PAIRS:
    s = config.get_pair_settings(pair)
    session = config.TRADING_SESSIONS.get(pair, {})
    sf = 'ON' if session.get('session_filter', False) else 'OFF'
    cd = s.get('cooldown_minutes', 0)
    print(f"{pair:<10} {s['strategy']:<14} {s['tp_pips']:<4} {s['sl_pips']:<4} {cd:<5} {sf:<4}")
print("-" * 50)

results = {}
all_monthly = {}

for pair in config.OPTIMIZED_PAIRS:
    settings = config.get_pair_settings(pair)
    session = config.TRADING_SESSIONS.get(pair, {})
    sf_status = 'ON' if session.get('session_filter', False) else 'OFF'
    cd = settings.get('cooldown_minutes', 0)

    print(f"\n{'=' * 60}")
    print(f"[{pair}] {settings['strategy']} | TP:{settings['tp_pips']} SL:{settings['sl_pips']} | CD:{cd}m SF:{sf_status}")
    print(f"{'=' * 60}")

    df = load_histdata_csv(pair)

    if len(df) == 0:
        print(f"  No data available")
        continue

    print(f"  Loaded {len(df):,} candles")
    print(f"  Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Run EXACT match backtest
    result = backtest_exact_match(df, pair)
    results[pair] = result

    print(f"\n  RESULTS:")
    print(f"    Trades: {result['trades']}")
    print(f"    Win Rate: {result['win_rate']:.1f}% ({result['wins']}W / {result['losses']}L)")
    print(f"    Total Pips: {result['total_pips']:+,.0f}")
    print(f"    Exits: {result['tp']} TP, {result['sl']} SL")

    # Merge monthly stats
    for month, stats in result.get('monthly', {}).items():
        if month not in all_monthly:
            all_monthly[month] = {'trades': 0, 'wins': 0, 'pips': 0}
        all_monthly[month]['trades'] += stats['trades']
        all_monthly[month]['wins'] += stats['wins']
        all_monthly[month]['pips'] += stats['pips']

# =============================================================================
# SUMMARY BY PAIR
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY PAIR - FULL YEAR 2024")
print("=" * 80)
print(f"\n{'PAIR':<10} {'TRADES':<10} {'WINS':<8} {'LOSSES':<8} {'WIN%':<8} {'PIPS':<12} {'TP':<6} {'SL'}")
print("-" * 75)

total_trades = 0
total_wins = 0
total_pips = 0
total_tp = 0
total_sl = 0

for pair in config.OPTIMIZED_PAIRS:
    if pair not in results:
        continue
    r = results[pair]
    print(f"{pair:<10} {r['trades']:<10} {r['wins']:<8} {r['losses']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:>+10,.0f}   {r['tp']:<6} {r['sl']}")

    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']
    total_tp += r['tp']
    total_sl += r['sl']

print("-" * 75)
total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {total_trades:<10} {total_wins:<8} {total_trades - total_wins:<8} {total_wr:<7.1f}% {total_pips:>+10,.0f}   {total_tp:<6} {total_sl}")

# =============================================================================
# EXPECTED vs ACTUAL
# =============================================================================
print("\n" + "=" * 80)
print("EXPECTED vs ACTUAL WIN RATE")
print("=" * 80)

print(f"\n{'PAIR':<10} {'EXP WR%':<10} {'ACT WR%':<10} {'DIFF':<10} {'STATUS'}")
print("-" * 50)

for pair in config.OPTIMIZED_PAIRS:
    if pair not in results:
        continue
    r = results[pair]
    settings = config.get_pair_settings(pair)
    exp_wr = settings.get('expected_wr', 70)
    diff = r['win_rate'] - exp_wr
    status = "OK" if diff >= -15 else "BELOW"
    print(f"{pair:<10} {exp_wr:<10.1f} {r['win_rate']:<10.1f} {diff:+.1f}%    {status}")

# =============================================================================
# MONTHLY BREAKDOWN
# =============================================================================
print("\n" + "=" * 80)
print("MONTHLY BREAKDOWN - 2024")
print("=" * 80)
print(f"\n{'MONTH':<10} {'TRADES':<10} {'WINS':<8} {'WIN%':<10} {'PIPS':<12} {'CUMULATIVE'}")
print("-" * 65)

cumulative = 0
sorted_months = sorted(all_monthly.keys())

for month in sorted_months:
    stats = all_monthly[month]
    wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
    cumulative += stats['pips']
    print(f"{month:<10} {stats['trades']:<10} {stats['wins']:<8} {wr:<9.1f}% {stats['pips']:>+10,.0f}   {cumulative:>+10,.0f}")

print("-" * 65)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY - FULL YEAR 2024")
print("=" * 80)

# Calculate trading days (approximate - M1 data)
if results:
    first_pair = list(results.keys())[0]
    df_sample = load_histdata_csv(first_pair)
    if len(df_sample) > 0:
        trading_days = (df_sample['time'].iloc[-1] - df_sample['time'].iloc[0]).days

print(f"\n  Total Trades: {total_trades:,}")
print(f"  Win Rate: {total_wr:.1f}%")
print(f"  Total Pips: {total_pips:+,.0f}")
print(f"  TP Exits: {total_tp:,}")
print(f"  SL Exits: {total_sl:,}")

if total_trades > 0:
    avg_win = sum(r['total_pips'] for r in results.values() if r['total_pips'] > 0) / max(1, sum(1 for r in results.values() if r['total_pips'] > 0))
    print(f"\n  Avg Pips per Trade: {total_pips / total_trades:.2f}")

    # Approximate monthly/daily stats
    months_traded = len(all_monthly)
    if months_traded > 0:
        print(f"\n  Monthly Average:")
        print(f"    Trades: {total_trades / months_traded:.0f}")
        print(f"    Pips: {total_pips / months_traded:+,.0f}")

    # Daily estimate (approx 252 trading days)
    trading_days_est = 252
    print(f"\n  Daily Average (est. {trading_days_est} trading days):")
    print(f"    Trades: {total_trades / trading_days_est:.1f}")
    print(f"    Pips: {total_pips / trading_days_est:+.1f}")

# Profitable pairs
profitable_pairs = [p for p, r in results.items() if r['total_pips'] > 0]
losing_pairs = [p for p, r in results.items() if r['total_pips'] <= 0]

print(f"\n  Profitable Pairs ({len(profitable_pairs)}):")
for pair in sorted(profitable_pairs, key=lambda p: results[p]['total_pips'], reverse=True):
    r = results[pair]
    print(f"    {pair}: {r['total_pips']:+,.0f} pips ({r['win_rate']:.1f}% WR)")

print(f"\n  Losing Pairs ({len(losing_pairs)}):")
for pair in sorted(losing_pairs, key=lambda p: results[p]['total_pips']):
    r = results[pair]
    print(f"    {pair}: {r['total_pips']:+,.0f} pips ({r['win_rate']:.1f}% WR)")

print("\n" + "=" * 80)
print("STRATEGY VERDICT")
print("=" * 80)

if total_pips > 0:
    print(f"\n  >>> PROFITABLE: +{total_pips:,.0f} pips in 2024 <<<")
else:
    print(f"\n  >>> LOSING: {total_pips:,.0f} pips in 2024 <<<")

print(f"\n  With $1/pip position sizing:")
print(f"    Gross P&L: ${total_pips:+,.0f}")
print(f"    Per Month: ${total_pips / max(1, len(all_monthly)):+,.0f}")

print("=" * 80)
