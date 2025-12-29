"""
STRATEGY OPTIMIZATION: FULL YEAR 2024
======================================
Find the BEST settings for each pair using EXACT live trading logic.
Tests:
- All 3 strategies (RSI_30_70, MACD_CROSS, STRONG_TREND)
- Multiple TP/SL combinations
- Different RSI thresholds
- Session filters ON/OFF
- Cooldown variations
"""

import sys
import os

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
print("=" * 80, flush=True)
print("STARTING OPTIMIZATION SCRIPT...", flush=True)
print("=" * 80, flush=True)

sys.path.insert(0, os.path.abspath('.'))

print("[LOG] Importing modules...", flush=True)

try:
    from trading_system.Forex_Trading.strategies.optimized_strategy import calculate_indicators, check_rsi_30_70_signal, check_macd_cross_signal, check_strong_trend_signal
    print("[LOG] Successfully imported strategy modules", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to import strategy modules: {e}", flush=True)
    sys.exit(1)

import pandas as pd
from datetime import datetime
from pathlib import Path
from itertools import product

print("[LOG] All imports complete", flush=True)

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

PAIRS_TO_TEST = list(FILE_MAP.keys())

# Spread simulation
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================
STRATEGIES = ['RSI_30_70', 'MACD_CROSS', 'STRONG_TREND']

# TP/SL combinations to test (in pips)
TP_OPTIONS = [5, 8, 10, 12, 15, 20]
SL_OPTIONS = [10, 15, 20, 25, 30]

# RSI thresholds (only for RSI_30_70)
RSI_OVERSOLD_OPTIONS = [25, 30, 35]
RSI_OVERBOUGHT_OPTIONS = [65, 70, 75]

# Cooldown options (minutes)
COOLDOWN_OPTIONS = [0, 15, 30, 60]

# Session filter - test with and without
SESSION_FILTER_OPTIONS = [True, False]


def load_histdata_csv(pair: str) -> pd.DataFrame:
    """Load HistData CSV file for a pair."""
    print(f"[LOG] Loading data for {pair}...", flush=True)
    filename = FILE_MAP.get(pair)
    if not filename:
        print(f"[ERROR] No filename mapping for {pair}", flush=True)
        return pd.DataFrame()

    filepath = DATA_DIR / filename
    print(f"[LOG] File path: {filepath}", flush=True)

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}", flush=True)
        return pd.DataFrame()

    try:
        print(f"[LOG] Reading CSV...", flush=True)
        df = pd.read_csv(filepath, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        print(f"[LOG] CSV loaded, {len(df)} rows", flush=True)

        df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
        df = df.drop(columns=['date', 'volume'])
        df = df.sort_values('time').reset_index(drop=True)
        print(f"[LOG] Data processed successfully for {pair}", flush=True)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {pair}: {e}", flush=True)
        return pd.DataFrame()


def get_signal_for_strategy(strategy: str, df: pd.DataFrame, rsi_oversold: int = 30, rsi_overbought: int = 70):
    """Get signal based on strategy type."""
    if strategy == 'RSI_30_70':
        return check_rsi_30_70_signal(df, oversold=rsi_oversold, overbought=rsi_overbought)
    elif strategy == 'MACD_CROSS':
        return check_macd_cross_signal(df)
    elif strategy == 'STRONG_TREND':
        return check_strong_trend_signal(df)
    return None, "Unknown strategy"


def backtest_settings(df: pd.DataFrame, pair: str, strategy: str, tp_pips: int, sl_pips: int,
                      cooldown_minutes: int, use_session_filter: bool,
                      rsi_oversold: int = 30, rsi_overbought: int = 70,
                      allowed_hours: list = None) -> dict:
    """
    Backtest with specific settings.
    """
    if len(df) < 100:
        return {'trades': 0, 'wins': 0, 'total_pips': 0, 'win_rate': 0}

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    # Default allowed hours (US/London session overlap)
    if allowed_hours is None:
        allowed_hours = list(range(12, 21))  # 12:00-20:00 UTC

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    tp_price = 0
    sl_price = 0
    last_exit_time = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        current_time = row['time']
        hour_utc = current_time.hour

        if in_position:
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
            else:
                if high >= sl_price:
                    exit_reason = 'SL'
                    exit_pnl = -sl_pips
                elif low <= tp_price:
                    exit_reason = 'TP'
                    exit_pnl = tp_pips

            if exit_reason:
                trades.append({'pnl': exit_pnl, 'won': exit_pnl > 0})
                in_position = False
                last_exit_time = current_time

        else:
            # Session filter
            if use_session_filter and hour_utc not in allowed_hours:
                continue

            # Cooldown
            if cooldown_minutes > 0 and last_exit_time is not None:
                time_since = (current_time - last_exit_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            # Get signal
            signal, _ = get_signal_for_strategy(strategy, df.iloc[:i+1], rsi_oversold, rsi_overbought)

            if signal is not None:
                in_position = True
                direction = signal

                if signal == 'BUY':
                    entry_price = row['close'] + spread
                    tp_price = entry_price + tp_price_diff
                    sl_price = entry_price - sl_price_diff
                else:
                    entry_price = row['close'] - spread
                    tp_price = entry_price - tp_price_diff
                    sl_price = entry_price + sl_price_diff

    if not trades:
        return {'trades': 0, 'wins': 0, 'total_pips': 0, 'win_rate': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'total_pips': total_pips
    }


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================
print("=" * 80, flush=True)
print("STRATEGY OPTIMIZATION: FULL YEAR 2024", flush=True)
print("=" * 80, flush=True)
print(f"Time: {datetime.now()}", flush=True)
print(f"\nTesting:", flush=True)
print(f"  - Strategies: {STRATEGIES}", flush=True)
print(f"  - TP options: {TP_OPTIONS}", flush=True)
print(f"  - SL options: {SL_OPTIONS}", flush=True)
print(f"  - RSI thresholds: OS {RSI_OVERSOLD_OPTIONS}, OB {RSI_OVERBOUGHT_OPTIONS}", flush=True)
print(f"  - Cooldowns: {COOLDOWN_OPTIONS}", flush=True)
print(f"  - Session filter: ON/OFF", flush=True)
print("=" * 80, flush=True)

print(f"\n[LOG] Data directory: {DATA_DIR}", flush=True)
print(f"[LOG] Directory exists: {DATA_DIR.exists()}", flush=True)
if DATA_DIR.exists():
    files = list(DATA_DIR.glob("*.csv"))
    print(f"[LOG] Found {len(files)} CSV files", flush=True)

# Store best results per pair
best_results = {}

for pair_idx, pair in enumerate(PAIRS_TO_TEST):
    print(f"\n{'=' * 70}", flush=True)
    print(f"OPTIMIZING: {pair} ({pair_idx + 1}/{len(PAIRS_TO_TEST)})", flush=True)
    print(f"{'=' * 70}", flush=True)

    df = load_histdata_csv(pair)
    if len(df) == 0:
        print(f"  [SKIP] No data available for {pair}", flush=True)
        continue

    # Pre-calculate indicators once
    print(f"[LOG] Calculating indicators for {pair}...", flush=True)
    try:
        df = calculate_indicators(df)
        print(f"[LOG] Indicators calculated successfully", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to calculate indicators: {e}", flush=True)
        continue

    print(f"  Loaded {len(df):,} candles", flush=True)

    best_pips = float('-inf')
    best_config = None
    all_results = []

    # Count total combinations
    total_combos = 0
    for strategy in STRATEGIES:
        if strategy == 'RSI_30_70':
            total_combos += len(TP_OPTIONS) * len(SL_OPTIONS) * len(RSI_OVERSOLD_OPTIONS) * len(RSI_OVERBOUGHT_OPTIONS) * len(COOLDOWN_OPTIONS) * len(SESSION_FILTER_OPTIONS)
        else:
            total_combos += len(TP_OPTIONS) * len(SL_OPTIONS) * len(COOLDOWN_OPTIONS) * len(SESSION_FILTER_OPTIONS)

    print(f"  Testing {total_combos} combinations...", flush=True)
    tested = 0
    last_progress = 0

    for strategy in STRATEGIES:
        print(f"  [LOG] Testing strategy: {strategy}", flush=True)
        for tp in TP_OPTIONS:
            for sl in SL_OPTIONS:
                for cooldown in COOLDOWN_OPTIONS:
                    for use_session in SESSION_FILTER_OPTIONS:
                        if strategy == 'RSI_30_70':
                            # Test RSI threshold combinations
                            for rsi_os in RSI_OVERSOLD_OPTIONS:
                                for rsi_ob in RSI_OVERBOUGHT_OPTIONS:
                                    try:
                                        result = backtest_settings(
                                            df, pair, strategy, tp, sl, cooldown, use_session,
                                            rsi_oversold=rsi_os, rsi_overbought=rsi_ob
                                        )

                                        config = {
                                            'strategy': strategy,
                                            'tp_pips': tp,
                                            'sl_pips': sl,
                                            'cooldown': cooldown,
                                            'session_filter': use_session,
                                            'rsi_oversold': rsi_os,
                                            'rsi_overbought': rsi_ob
                                        }

                                        all_results.append({**config, **result})

                                        if result['total_pips'] > best_pips and result['trades'] >= 50:
                                            best_pips = result['total_pips']
                                            best_config = {**config, **result}

                                        tested += 1
                                    except Exception as e:
                                        print(f"[ERROR] Backtest failed: {e}", flush=True)
                                        tested += 1
                        else:
                            try:
                                result = backtest_settings(
                                    df, pair, strategy, tp, sl, cooldown, use_session
                                )

                                config = {
                                    'strategy': strategy,
                                    'tp_pips': tp,
                                    'sl_pips': sl,
                                    'cooldown': cooldown,
                                    'session_filter': use_session,
                                    'rsi_oversold': None,
                                    'rsi_overbought': None
                                }

                                all_results.append({**config, **result})

                                if result['total_pips'] > best_pips and result['trades'] >= 50:
                                    best_pips = result['total_pips']
                                    best_config = {**config, **result}

                                tested += 1
                            except Exception as e:
                                print(f"[ERROR] Backtest failed: {e}", flush=True)
                                tested += 1

                        # Progress update every 10%
                        progress = int(tested / total_combos * 100)
                        if progress >= last_progress + 10:
                            print(f"    [PROGRESS] {tested}/{total_combos} ({progress}%) - Best so far: {best_pips:+,.0f} pips", flush=True)
                            last_progress = progress

        # Progress update after each strategy
        print(f"    {strategy}: COMPLETE ({tested}/{total_combos})", flush=True)

    best_results[pair] = best_config

    if best_config:
        print(f"\n  BEST SETTINGS FOR {pair}:")
        print(f"    Strategy: {best_config['strategy']}")
        print(f"    TP: {best_config['tp_pips']} pips | SL: {best_config['sl_pips']} pips")
        if best_config['strategy'] == 'RSI_30_70':
            print(f"    RSI: {best_config['rsi_oversold']}/{best_config['rsi_overbought']}")
        print(f"    Cooldown: {best_config['cooldown']} min")
        print(f"    Session Filter: {'ON' if best_config['session_filter'] else 'OFF'}")
        print(f"    Results: {best_config['trades']} trades, {best_config['win_rate']:.1f}% WR, {best_config['total_pips']:+,} pips")

    # Show top 5 configurations
    sorted_results = sorted(all_results, key=lambda x: x['total_pips'], reverse=True)
    print(f"\n  TOP 5 CONFIGURATIONS:")
    for i, r in enumerate(sorted_results[:5]):
        if r['trades'] >= 50:
            rsi_str = f" RSI:{r['rsi_oversold']}/{r['rsi_overbought']}" if r['strategy'] == 'RSI_30_70' else ""
            sess_str = "SF:ON" if r['session_filter'] else "SF:OFF"
            print(f"    {i+1}. {r['strategy']} TP:{r['tp_pips']} SL:{r['sl_pips']}{rsi_str} CD:{r['cooldown']}m {sess_str}")
            print(f"       {r['trades']}t {r['win_rate']:.1f}% {r['total_pips']:+,}p")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE - BEST SETTINGS FOR EACH PAIR")
print("=" * 80)

print(f"\n{'PAIR':<10} {'STRATEGY':<14} {'TP':<4} {'SL':<4} {'RSI':<8} {'CD':<4} {'SF':<4} {'TRADES':<8} {'WIN%':<7} {'PIPS'}")
print("-" * 90)

total_pips_optimized = 0
total_trades_optimized = 0

for pair in PAIRS_TO_TEST:
    if pair not in best_results or best_results[pair] is None:
        print(f"{pair:<10} NO PROFITABLE SETTINGS FOUND")
        continue

    r = best_results[pair]
    rsi_str = f"{r['rsi_oversold']}/{r['rsi_overbought']}" if r['strategy'] == 'RSI_30_70' else "-"
    sf_str = "ON" if r['session_filter'] else "OFF"

    print(f"{pair:<10} {r['strategy']:<14} {r['tp_pips']:<4} {r['sl_pips']:<4} {rsi_str:<8} {r['cooldown']:<4} {sf_str:<4} {r['trades']:<8} {r['win_rate']:<6.1f}% {r['total_pips']:>+,}")

    total_pips_optimized += r['total_pips']
    total_trades_optimized += r['trades']

print("-" * 90)
print(f"{'TOTAL':<10} {'':<14} {'':<4} {'':<4} {'':<8} {'':<4} {'':<4} {total_trades_optimized:<8} {'':<7} {total_pips_optimized:>+,}")

# =============================================================================
# GENERATE NEW CONFIG
# =============================================================================
print("\n" + "=" * 80)
print("RECOMMENDED CONFIG UPDATE (copy to optimized_paper_config.py)")
print("=" * 80)

print("\nPAIR_SETTINGS = {")
for pair in PAIRS_TO_TEST:
    if pair not in best_results or best_results[pair] is None:
        continue

    r = best_results[pair]
    print(f"    '{pair}': {{")
    print(f"        'strategy': '{r['strategy']}',")
    print(f"        'tp_pips': {r['tp_pips']},")
    print(f"        'sl_pips': {r['sl_pips']},")
    if r['strategy'] == 'RSI_30_70':
        print(f"        'rsi_oversold': {r['rsi_oversold']},")
        print(f"        'rsi_overbought': {r['rsi_overbought']},")
    print(f"        'expected_wr': {r['win_rate']:.1f},")
    print(f"    }},")
print("}")

# Session filters
print("\nTRADING_SESSIONS = {")
for pair in PAIRS_TO_TEST:
    if pair not in best_results or best_results[pair] is None:
        continue
    r = best_results[pair]
    if r['session_filter']:
        print(f"    '{pair}': {{'allowed_hours': list(range(12, 21))}},  # Session filter ON")
    else:
        print(f"    '{pair}': {{'allowed_hours': list(range(0, 24))}},  # Session filter OFF (all hours)")
print("}")

# Cooldown recommendation
cooldowns = [best_results[p]['cooldown'] for p in PAIRS_TO_TEST if p in best_results and best_results[p] is not None]
if cooldowns:
    avg_cooldown = sum(cooldowns) / len(cooldowns)
    most_common = max(set(cooldowns), key=cooldowns.count)
    print(f"\nCOOLDOWN_MINUTES = {most_common}  # Most common optimal cooldown")

print("\n" + "=" * 80)
print("COMPARISON: CURRENT vs OPTIMIZED")
print("=" * 80)
print(f"\n  CURRENT CONFIG (2024 backtest): -7,250 pips")
print(f"  OPTIMIZED CONFIG (2024 backtest): {total_pips_optimized:+,} pips")
print(f"  IMPROVEMENT: {total_pips_optimized - (-7250):+,} pips")
print("=" * 80)
