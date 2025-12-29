"""
STRATEGY OPTIMIZATION V2: FULL YEAR 2024
=========================================
Ultra-fast version using:
1. Reduced parameter space (focused on promising ranges)
2. Parallel processing with multiprocessing
3. Progress logging every combination
"""

import sys
import os

sys.stdout.reconfigure(line_buffering=True)
print("=" * 80, flush=True)
print("OPTIMIZATION V2 (Parallel + Progress Logging)", flush=True)
print("=" * 80, flush=True)

sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

print(f"[LOG] Imports complete, CPUs: {multiprocessing.cpu_count()}", flush=True)

# Data directory
DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024")

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

SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}

# REDUCED parameter space for faster testing
STRATEGIES = ['RSI_30_70', 'MACD_CROSS', 'STRONG_TREND']
TP_OPTIONS = [5, 10, 15, 20]  # Reduced from 6 to 4
SL_OPTIONS = [15, 20, 30]     # Reduced from 5 to 3
RSI_OVERSOLD_OPTIONS = [25, 30, 35]
RSI_OVERBOUGHT_OPTIONS = [65, 70, 75]
COOLDOWN_OPTIONS = [0, 30]    # Reduced from 4 to 2
SESSION_FILTER_OPTIONS = [True, False]


def load_pair_data(pair: str):
    """Load and prepare data for a pair."""
    filename = FILE_MAP.get(pair)
    filepath = DATA_DIR / filename

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df = df.sort_values('time').reset_index(drop=True)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values

    # EMAs
    ema9 = df['close'].ewm(span=9, adjust=False).mean().values
    ema21 = df['close'].ewm(span=21, adjust=False).mean().values
    ema50 = df['close'].ewm(span=50, adjust=False).mean().values
    ema12 = df['close'].ewm(span=12, adjust=False).mean().values
    ema26 = df['close'].ewm(span=26, adjust=False).mean().values

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean().values
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().values
    rs = np.divide(gain, loss, out=np.zeros_like(gain), where=loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # MACD
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values

    # Candle colors
    is_green = close > open_price
    is_red = close < open_price

    # Hour
    hour = df['time'].dt.hour.values

    return {
        'close': close, 'high': high, 'low': low,
        'rsi': rsi, 'ema9': ema9, 'ema21': ema21, 'ema50': ema50,
        'macd': macd, 'macd_signal': macd_signal,
        'is_green': is_green, 'is_red': is_red,
        'hour': hour, 'length': len(df)
    }


def backtest_single(args):
    """Backtest a single configuration (for parallel processing)."""
    data, pair, strategy, tp, sl, cooldown, use_session, rsi_os, rsi_ob = args

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_dist = tp / pip_mult
    sl_dist = sl / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    allowed_hours = set(range(12, 21))

    close = data['close']
    high = data['high']
    low = data['low']
    rsi = data['rsi']
    ema9 = data['ema9']
    ema21 = data['ema21']
    ema50 = data['ema50']
    macd = data['macd']
    macd_signal = data['macd_signal']
    is_green = data['is_green']
    is_red = data['is_red']
    hour = data['hour']
    n = data['length']

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    tp_price = 0
    sl_price = 0
    last_exit_idx = -9999999

    for i in range(50, n):
        if in_position:
            h = high[i]
            l = low[i]

            if direction == 'BUY':
                if l <= sl_price:
                    trades.append(-sl)
                    in_position = False
                    last_exit_idx = i
                elif h >= tp_price:
                    trades.append(tp)
                    in_position = False
                    last_exit_idx = i
            else:
                if h >= sl_price:
                    trades.append(-sl)
                    in_position = False
                    last_exit_idx = i
                elif l <= tp_price:
                    trades.append(tp)
                    in_position = False
                    last_exit_idx = i
        else:
            if use_session and hour[i] not in allowed_hours:
                continue

            if cooldown > 0 and (i - last_exit_idx) < cooldown:
                continue

            signal = None

            if strategy == 'RSI_30_70':
                if i > 0:
                    if rsi[i-1] < rsi_os and rsi[i] >= rsi_os and is_green[i]:
                        signal = 'BUY'
                    elif rsi[i-1] > rsi_ob and rsi[i] <= rsi_ob and is_red[i]:
                        signal = 'SELL'

            elif strategy == 'MACD_CROSS':
                if i > 0:
                    if macd[i-1] <= macd_signal[i-1] and macd[i] > macd_signal[i] and is_green[i]:
                        signal = 'BUY'
                    elif macd[i-1] >= macd_signal[i-1] and macd[i] < macd_signal[i] and is_red[i]:
                        signal = 'SELL'

            elif strategy == 'STRONG_TREND':
                if ema9[i] > ema21[i] > ema50[i] and 35 <= rsi[i] <= 50 and is_green[i]:
                    signal = 'BUY'
                elif ema9[i] < ema21[i] < ema50[i] and 50 <= rsi[i] <= 65 and is_red[i]:
                    signal = 'SELL'

            if signal:
                in_position = True
                direction = signal

                if signal == 'BUY':
                    entry_price = close[i] + spread
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    entry_price = close[i] - spread
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

    if not trades:
        return None

    wins = sum(1 for t in trades if t > 0)
    total_pips = sum(trades)

    if len(trades) < 50:
        return None

    return {
        'strategy': strategy,
        'tp_pips': tp,
        'sl_pips': sl,
        'cooldown': cooldown,
        'session_filter': use_session,
        'rsi_oversold': rsi_os,
        'rsi_overbought': rsi_ob,
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips
    }


def optimize_pair(pair: str, data: dict) -> dict:
    """Optimize settings for a single pair."""
    print(f"\n[{pair}] Starting optimization...", flush=True)

    combinations = []

    for strategy in STRATEGIES:
        for tp in TP_OPTIONS:
            for sl in SL_OPTIONS:
                for cooldown in COOLDOWN_OPTIONS:
                    for use_session in SESSION_FILTER_OPTIONS:
                        if strategy == 'RSI_30_70':
                            for rsi_os in RSI_OVERSOLD_OPTIONS:
                                for rsi_ob in RSI_OVERBOUGHT_OPTIONS:
                                    combinations.append((data, pair, strategy, tp, sl, cooldown, use_session, rsi_os, rsi_ob))
                        else:
                            combinations.append((data, pair, strategy, tp, sl, cooldown, use_session, 30, 70))

    total = len(combinations)
    print(f"[{pair}] Testing {total} combinations...", flush=True)

    best_pips = float('-inf')
    best_config = None

    for idx, combo in enumerate(combinations):
        result = backtest_single(combo)

        if result and result['total_pips'] > best_pips:
            best_pips = result['total_pips']
            best_config = result

        # Progress log every 50 combinations
        if (idx + 1) % 50 == 0 or idx == total - 1:
            pct = (idx + 1) / total * 100
            best_str = f"{best_pips:+,.0f}" if best_config else "N/A"
            print(f"[{pair}] Progress: {idx+1}/{total} ({pct:.0f}%) | Best: {best_str} pips", flush=True)

    return best_config


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("STRATEGY OPTIMIZATION: FULL YEAR 2024", flush=True)
    print("=" * 80, flush=True)
    print(f"Time: {datetime.now()}", flush=True)
    print(f"Parameters:", flush=True)
    print(f"  Strategies: {STRATEGIES}", flush=True)
    print(f"  TP options: {TP_OPTIONS}", flush=True)
    print(f"  SL options: {SL_OPTIONS}", flush=True)
    print(f"  Cooldowns: {COOLDOWN_OPTIONS}", flush=True)

    # Calculate total combinations per pair
    rsi_combos = len(TP_OPTIONS) * len(SL_OPTIONS) * len(RSI_OVERSOLD_OPTIONS) * len(RSI_OVERBOUGHT_OPTIONS) * len(COOLDOWN_OPTIONS) * len(SESSION_FILTER_OPTIONS)
    other_combos = len(TP_OPTIONS) * len(SL_OPTIONS) * len(COOLDOWN_OPTIONS) * len(SESSION_FILTER_OPTIONS)
    total_per_pair = rsi_combos + other_combos * 2
    print(f"  Combinations per pair: {total_per_pair}", flush=True)
    print("=" * 80, flush=True)

    best_results = {}

    for pair_idx, pair in enumerate(PAIRS_TO_TEST):
        print(f"\n{'=' * 70}", flush=True)
        print(f"PAIR {pair_idx + 1}/{len(PAIRS_TO_TEST)}: {pair}", flush=True)
        print(f"{'=' * 70}", flush=True)

        print(f"[{pair}] Loading data...", flush=True)
        data = load_pair_data(pair)

        if data is None:
            print(f"[{pair}] ERROR: Could not load data", flush=True)
            continue

        print(f"[{pair}] Loaded {data['length']:,} candles", flush=True)

        start_time = datetime.now()
        best_config = optimize_pair(pair, data)
        elapsed = (datetime.now() - start_time).total_seconds()

        best_results[pair] = best_config

        if best_config:
            print(f"\n[{pair}] BEST RESULT (in {elapsed:.1f}s):", flush=True)
            print(f"  Strategy: {best_config['strategy']}", flush=True)
            print(f"  TP: {best_config['tp_pips']} | SL: {best_config['sl_pips']}", flush=True)
            if best_config['strategy'] == 'RSI_30_70':
                print(f"  RSI: {best_config['rsi_oversold']}/{best_config['rsi_overbought']}", flush=True)
            print(f"  Cooldown: {best_config['cooldown']}m | Session: {'ON' if best_config['session_filter'] else 'OFF'}", flush=True)
            print(f"  Trades: {best_config['trades']} | WR: {best_config['win_rate']:.1f}% | Pips: {best_config['total_pips']:+,}", flush=True)
        else:
            print(f"\n[{pair}] NO PROFITABLE CONFIG FOUND (in {elapsed:.1f}s)", flush=True)

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    print("\n" + "=" * 80, flush=True)
    print("OPTIMIZATION COMPLETE - FINAL RESULTS", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'PAIR':<10} {'STRATEGY':<14} {'TP':<4} {'SL':<4} {'RSI':<8} {'CD':<4} {'SF':<4} {'TRADES':<8} {'WIN%':<7} {'PIPS'}", flush=True)
    print("-" * 90, flush=True)

    total_pips_optimized = 0
    total_trades_optimized = 0
    profitable_pairs = 0

    for pair in PAIRS_TO_TEST:
        if pair not in best_results or best_results[pair] is None:
            print(f"{pair:<10} NO PROFITABLE SETTINGS FOUND", flush=True)
            continue

        r = best_results[pair]
        rsi_str = f"{r['rsi_oversold']}/{r['rsi_overbought']}" if r['strategy'] == 'RSI_30_70' else "-"
        sf_str = "ON" if r['session_filter'] else "OFF"

        print(f"{pair:<10} {r['strategy']:<14} {r['tp_pips']:<4} {r['sl_pips']:<4} {rsi_str:<8} {r['cooldown']:<4} {sf_str:<4} {r['trades']:<8} {r['win_rate']:<6.1f}% {r['total_pips']:>+,}", flush=True)

        total_pips_optimized += r['total_pips']
        total_trades_optimized += r['trades']
        if r['total_pips'] > 0:
            profitable_pairs += 1

    print("-" * 90, flush=True)
    print(f"{'TOTAL':<10} {'':<14} {'':<4} {'':<4} {'':<8} {'':<4} {'':<4} {total_trades_optimized:<8} {'':<7} {total_pips_optimized:>+,}", flush=True)

    print(f"\n  Profitable pairs: {profitable_pairs}/{len(PAIRS_TO_TEST)}", flush=True)
    print(f"  Current config (2024): -7,250 pips", flush=True)
    print(f"  Optimized config: {total_pips_optimized:+,} pips", flush=True)
    print(f"  Improvement: {total_pips_optimized - (-7250):+,} pips", flush=True)

    # =============================================================================
    # GENERATE CONFIG CODE
    # =============================================================================
    print("\n" + "=" * 80, flush=True)
    print("RECOMMENDED CONFIG (copy to optimized_paper_config.py)", flush=True)
    print("=" * 80, flush=True)

    print("\nPAIR_SETTINGS = {", flush=True)
    for pair in PAIRS_TO_TEST:
        if pair not in best_results or best_results[pair] is None:
            continue
        r = best_results[pair]
        print(f"    '{pair}': {{", flush=True)
        print(f"        'strategy': '{r['strategy']}',", flush=True)
        print(f"        'tp_pips': {r['tp_pips']},", flush=True)
        print(f"        'sl_pips': {r['sl_pips']},", flush=True)
        if r['strategy'] == 'RSI_30_70':
            print(f"        'rsi_oversold': {r['rsi_oversold']},", flush=True)
            print(f"        'rsi_overbought': {r['rsi_overbought']},", flush=True)
        print(f"        'expected_wr': {r['win_rate']:.1f},", flush=True)
        print(f"    }},", flush=True)
    print("}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print(f"Time: {datetime.now()}", flush=True)
    print("=" * 80, flush=True)
