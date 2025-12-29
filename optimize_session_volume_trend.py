"""
OPTIMIZATION: Session + Volume + Trend Filters
===============================================
Testing combinations of:
- 7 Strategy variations (RSI, MACD, Trend combinations)
- 3 Sessions (Asian, London, New York)
- Volume filter (ON/OFF)
- Trend filter (ON/OFF)
- 5 pip TP with 10, 12, 15 pip SL

Goal: Find quick profit setups with high win rate
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.strategies.optimized_strategy import calculate_indicators
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# Try to import OANDA client
try:
    sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading' / 'engine'))
    from oanda_client import OandaClient
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] OANDA client not available")

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

# Sessions (UTC hours)
SESSIONS = {
    'ASIAN': list(range(0, 8)),      # 00:00-08:00 UTC (Tokyo/Sydney)
    'LONDON': list(range(8, 16)),    # 08:00-16:00 UTC (London)
    'NEW_YORK': list(range(13, 21)), # 13:00-21:00 UTC (New York)
    'ALL': list(range(0, 24)),       # All hours
}

# TP/SL combinations (5 pip TP fixed, vary SL)
TP_PIPS = 5
SL_OPTIONS = [10, 12, 15]  # Need 67%, 71%, 75% WR to breakeven

# Volume filter options
VOLUME_FILTER = [True, False]

# Trend filter options
TREND_FILTER = [True, False]

# 7 Strategy variations
STRATEGIES = [
    'RSI_REVERSAL',        # RSI crosses back from oversold/overbought
    'RSI_EXTREME',         # RSI at extreme levels (25/75)
    'MACD_CROSS',          # MACD crosses signal line
    'MACD_ZERO',           # MACD crosses zero line
    'EMA_PULLBACK',        # Price pulls back to EMA in trend
    'RSI_MACD_COMBO',      # RSI + MACD both confirm
    'TRIPLE_CONFIRM',      # RSI + MACD + Trend all align
]

# Pairs to test
PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_CHF', 'EUR_GBP', 'NZD_USD']

# Spread simulation
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'USD_JPY': 1.0, 'EUR_GBP': 1.2,
    'AUD_CHF': 2.0, 'EUR_CAD': 2.0,
}


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    if not OANDA_AVAILABLE:
        return pd.DataFrame()

    client = OandaClient()
    candles = client.get_candles(pair, 'M1', count=count)

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df


def get_signal(df: pd.DataFrame, i: int, strategy: str,
               volume_filter: bool, trend_filter: bool) -> str:
    """
    Get trading signal based on strategy and filters.
    Returns 'BUY', 'SELL', or None
    """
    if i < 50:
        return None

    row = df.iloc[i]
    prev = df.iloc[i-1]

    rsi = row['rsi']
    prev_rsi = prev['rsi']
    macd = row['macd']
    macd_signal = row['macd_signal']
    prev_macd = prev['macd']
    prev_macd_signal = prev['macd_signal']
    ema9 = row['ema9']
    ema21 = row['ema21']
    ema50 = row['ema50']
    is_green = row['close'] > row['open']
    is_red = row['close'] < row['open']

    # Volume filter - check if volume is above average
    if volume_filter and 'volume' in df.columns:
        vol_avg = df['volume'].iloc[max(0, i-20):i].mean()
        if row['volume'] < vol_avg * 1.2:  # Need 20% above average
            return None

    # Trend filter - EMAs must be aligned
    if trend_filter:
        uptrend = ema9 > ema21 > ema50
        downtrend = ema9 < ema21 < ema50
        if not (uptrend or downtrend):
            return None

    signal = None

    # =================================================================
    # STRATEGY 1: RSI_REVERSAL (crosses back from oversold/overbought)
    # =================================================================
    if strategy == 'RSI_REVERSAL':
        if prev_rsi < 30 and rsi >= 30 and is_green:
            signal = 'BUY'
        elif prev_rsi > 70 and rsi <= 70 and is_red:
            signal = 'SELL'

    # =================================================================
    # STRATEGY 2: RSI_EXTREME (at extreme levels 25/75)
    # =================================================================
    elif strategy == 'RSI_EXTREME':
        if prev_rsi < 25 and rsi >= 25 and is_green:
            signal = 'BUY'
        elif prev_rsi > 75 and rsi <= 75 and is_red:
            signal = 'SELL'

    # =================================================================
    # STRATEGY 3: MACD_CROSS (MACD crosses signal line)
    # =================================================================
    elif strategy == 'MACD_CROSS':
        if prev_macd <= prev_macd_signal and macd > macd_signal and is_green:
            signal = 'BUY'
        elif prev_macd >= prev_macd_signal and macd < macd_signal and is_red:
            signal = 'SELL'

    # =================================================================
    # STRATEGY 4: MACD_ZERO (MACD crosses zero line)
    # =================================================================
    elif strategy == 'MACD_ZERO':
        if prev_macd <= 0 and macd > 0 and is_green:
            signal = 'BUY'
        elif prev_macd >= 0 and macd < 0 and is_red:
            signal = 'SELL'

    # =================================================================
    # STRATEGY 5: EMA_PULLBACK (price pulls back to EMA in trend)
    # =================================================================
    elif strategy == 'EMA_PULLBACK':
        # Uptrend pullback to EMA21
        if ema9 > ema21 > ema50:
            if prev['low'] <= ema21 and row['close'] > ema21 and is_green:
                if 40 <= rsi <= 60:  # RSI not extreme
                    signal = 'BUY'
        # Downtrend pullback to EMA21
        elif ema9 < ema21 < ema50:
            if prev['high'] >= ema21 and row['close'] < ema21 and is_red:
                if 40 <= rsi <= 60:
                    signal = 'SELL'

    # =================================================================
    # STRATEGY 6: RSI_MACD_COMBO (both RSI and MACD confirm)
    # =================================================================
    elif strategy == 'RSI_MACD_COMBO':
        # BUY: RSI crossing up from oversold + MACD bullish
        if prev_rsi < 35 and rsi >= 35 and macd > macd_signal and is_green:
            signal = 'BUY'
        # SELL: RSI crossing down from overbought + MACD bearish
        elif prev_rsi > 65 and rsi <= 65 and macd < macd_signal and is_red:
            signal = 'SELL'

    # =================================================================
    # STRATEGY 7: TRIPLE_CONFIRM (RSI + MACD + Trend all align)
    # =================================================================
    elif strategy == 'TRIPLE_CONFIRM':
        uptrend = ema9 > ema21 > ema50
        downtrend = ema9 < ema21 < ema50

        # BUY: Uptrend + RSI recovering + MACD bullish
        if uptrend and 30 <= rsi <= 50 and macd > macd_signal and is_green:
            if prev_rsi < rsi:  # RSI rising
                signal = 'BUY'
        # SELL: Downtrend + RSI declining + MACD bearish
        elif downtrend and 50 <= rsi <= 70 and macd < macd_signal and is_red:
            if prev_rsi > rsi:  # RSI falling
                signal = 'SELL'

    return signal


def backtest_combination(df: pd.DataFrame, pair: str, strategy: str,
                         session_hours: list, sl_pips: int,
                         volume_filter: bool, trend_filter: bool) -> dict:
    """
    Backtest a specific combination of parameters.
    Returns results dict.
    NOTE: df must already have indicators calculated!
    """
    if len(df) < 100:
        return {'trades': 0, 'wins': 0, 'pips': 0, 'win_rate': 0}

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_dist = TP_PIPS / pip_mult
    sl_dist = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    # Indicators should already be calculated before calling this function

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
            # Check exit
            high = row['high']
            low = row['low']
            exit_pnl = None

            if direction == 'BUY':
                if low <= sl_price:
                    exit_pnl = -sl_pips
                elif high >= tp_price:
                    exit_pnl = TP_PIPS
            else:
                if high >= sl_price:
                    exit_pnl = -sl_pips
                elif low <= tp_price:
                    exit_pnl = TP_PIPS

            if exit_pnl is not None:
                trades.append({'pnl': exit_pnl, 'won': exit_pnl > 0})
                in_position = False
                last_exit_time = current_time

        else:
            # Check for entry

            # Session filter
            if hour_utc not in session_hours:
                continue

            # Get signal
            signal = get_signal(df, i, strategy, volume_filter, trend_filter)

            if signal is not None:
                in_position = True
                direction = signal

                if signal == 'BUY':
                    entry_price = row['close'] + spread
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    entry_price = row['close'] - spread
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

    # Calculate results
    if not trades:
        return {'trades': 0, 'wins': 0, 'pips': 0, 'win_rate': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades) * 100

    return {
        'trades': len(trades),
        'wins': wins,
        'pips': total_pips,
        'win_rate': win_rate
    }


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================
print("=" * 80)
print("OPTIMIZATION: Session + Volume + Trend Filters")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nParameters:")
print(f"  TP: {TP_PIPS} pips (fixed)")
print(f"  SL options: {SL_OPTIONS}")
print(f"  Strategies: {len(STRATEGIES)}")
print(f"  Sessions: {list(SESSIONS.keys())}")
print(f"  Volume filter: ON/OFF")
print(f"  Trend filter: ON/OFF")

# Calculate total combinations
total_combos = len(STRATEGIES) * len(SESSIONS) * len(SL_OPTIONS) * len(VOLUME_FILTER) * len(TREND_FILTER)
print(f"\n  Total combinations per pair: {total_combos}")
print(f"  Pairs to test: {PAIRS}")
print("=" * 80)

all_results = {}

for pair in PAIRS:
    print(f"\n{'=' * 60}")
    print(f"[{pair}] Testing {total_combos} combinations...")
    print(f"{'=' * 60}")

    # Fetch data
    df = fetch_oanda_data(pair, count=5000)

    if len(df) == 0:
        print(f"  No data available")
        continue

    print(f"  Loaded {len(df)} candles")

    # Calculate indicators ONCE per pair (not per combination!)
    df = calculate_indicators(df)
    print(f"  Indicators calculated")

    best_result = None
    best_params = None
    results_list = []

    combo_count = 0

    for strategy in STRATEGIES:
        for session_name, session_hours in SESSIONS.items():
            for sl_pips in SL_OPTIONS:
                for vol_filter in VOLUME_FILTER:
                    for trend_filter in TREND_FILTER:
                        combo_count += 1

                        result = backtest_combination(
                            df, pair, strategy, session_hours, sl_pips,
                            vol_filter, trend_filter
                        )

                        if result['trades'] >= 5:  # Minimum trades
                            results_list.append({
                                'strategy': strategy,
                                'session': session_name,
                                'sl_pips': sl_pips,
                                'volume_filter': vol_filter,
                                'trend_filter': trend_filter,
                                **result
                            })

                            if best_result is None or result['pips'] > best_result['pips']:
                                best_result = result
                                best_params = {
                                    'strategy': strategy,
                                    'session': session_name,
                                    'sl_pips': sl_pips,
                                    'volume_filter': vol_filter,
                                    'trend_filter': trend_filter,
                                }

                        if combo_count % 50 == 0:
                            best_str = f"{best_result['pips']:+.0f}" if best_result else "N/A"
                            print(f"  [{pair}] {combo_count}/{total_combos} | Best: {best_str} pips", flush=True)

    # Store best for this pair
    if best_result and best_params:
        all_results[pair] = {
            'params': best_params,
            'result': best_result,
            'all_results': sorted(results_list, key=lambda x: x['pips'], reverse=True)[:10]
        }

        print(f"\n  BEST for {pair}:")
        print(f"    Strategy: {best_params['strategy']}")
        print(f"    Session: {best_params['session']}")
        print(f"    SL: {best_params['sl_pips']} pips")
        print(f"    Volume Filter: {'ON' if best_params['volume_filter'] else 'OFF'}")
        print(f"    Trend Filter: {'ON' if best_params['trend_filter'] else 'OFF'}")
        print(f"    Trades: {best_result['trades']} | WR: {best_result['win_rate']:.1f}% | Pips: {best_result['pips']:+.0f}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE - BEST SETTINGS PER PAIR")
print("=" * 80)

print(f"\n{'PAIR':<10} {'STRATEGY':<16} {'SESSION':<10} {'SL':<4} {'VOL':<5} {'TRD':<5} {'TRADES':<8} {'WR%':<8} {'PIPS'}")
print("-" * 85)

total_pips = 0
for pair in PAIRS:
    if pair in all_results:
        p = all_results[pair]['params']
        r = all_results[pair]['result']
        vol = 'ON' if p['volume_filter'] else 'OFF'
        trd = 'ON' if p['trend_filter'] else 'OFF'
        print(f"{pair:<10} {p['strategy']:<16} {p['session']:<10} {p['sl_pips']:<4} {vol:<5} {trd:<5} {r['trades']:<8} {r['win_rate']:<7.1f}% {r['pips']:+.0f}")
        total_pips += r['pips']

print("-" * 85)
print(f"{'TOTAL':<10} {'':<16} {'':<10} {'':<4} {'':<5} {'':<5} {'':<8} {'':<8} {total_pips:+.0f}")

# =============================================================================
# TOP 10 OVERALL COMBINATIONS
# =============================================================================
print("\n" + "=" * 80)
print("TOP 10 PROFITABLE COMBINATIONS (ACROSS ALL PAIRS)")
print("=" * 80)

all_combos = []
for pair in PAIRS:
    if pair in all_results:
        for combo in all_results[pair]['all_results']:
            combo['pair'] = pair
            all_combos.append(combo)

top_10 = sorted(all_combos, key=lambda x: x['pips'], reverse=True)[:10]

print(f"\n{'#':<3} {'PAIR':<10} {'STRATEGY':<16} {'SESSION':<10} {'SL':<4} {'VOL':<5} {'TRD':<5} {'TR':<6} {'WR%':<7} {'PIPS'}")
print("-" * 85)

for i, combo in enumerate(top_10, 1):
    vol = 'ON' if combo['volume_filter'] else 'OFF'
    trd = 'ON' if combo['trend_filter'] else 'OFF'
    print(f"{i:<3} {combo['pair']:<10} {combo['strategy']:<16} {combo['session']:<10} {combo['sl_pips']:<4} {vol:<5} {trd:<5} {combo['trades']:<6} {combo['win_rate']:<6.1f}% {combo['pips']:+.0f}")

# =============================================================================
# RECOMMENDED CONFIG
# =============================================================================
print("\n" + "=" * 80)
print("RECOMMENDED CONFIG (copy to optimized_paper_config.py)")
print("=" * 80)

print("\nPAIR_SETTINGS = {")
for pair in PAIRS:
    if pair in all_results:
        p = all_results[pair]['params']
        r = all_results[pair]['result']
        print(f"    '{pair}': {{")
        print(f"        'strategy': '{p['strategy']}',")
        print(f"        'tp_pips': {TP_PIPS},")
        print(f"        'sl_pips': {p['sl_pips']},")
        print(f"        'session': '{p['session']}',")
        print(f"        'volume_filter': {p['volume_filter']},")
        print(f"        'trend_filter': {p['trend_filter']},")
        print(f"        'expected_wr': {r['win_rate']:.1f},")
        print(f"    }},")
print("}")

print("\n" + "=" * 80)
print(f"Optimization complete at {datetime.now()}")
print("=" * 80)
