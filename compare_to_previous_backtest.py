"""
COMPARE: Combined Strategy vs Previous 3-Day Backtest
=====================================================
Previous results: 810 trades, 86.2% WR, +3,956 pips

Let's run the SAME backtest methodology with COMBINED settings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.config import optimized_paper_config as config
from trading_system.Forex_Trading.strategies.optimized_strategy import calculate_indicators, get_signal

import pandas as pd
from datetime import datetime

client = OandaClient('practice')

# Test ALL 10 pairs (including USD_JPY which was added)
TEST_PAIRS = list(config.OPTIMIZED_PAIRS)

# Spread simulation (same as backtest_oanda_3days.py)
SPREAD_PIPS = {
    'EUR_USD': 1.0,
    'GBP_USD': 1.2,
    'USD_CHF': 1.5,
    'USD_CAD': 1.2,
    'NZD_USD': 1.5,
    'AUD_JPY': 1.8,
    'EUR_GBP': 1.2,
    'AUD_CHF': 2.0,
    'EUR_CAD': 2.0,
    'USD_JPY': 1.0,
}

def fetch_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    return df


def backtest_original(df: pd.DataFrame, pair: str) -> dict:
    """
    Original backtest methodology (no cooldown between trades in backtest).
    This matches the previous backtest_oanda_3days.py results.
    """
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult

    # Apply spread
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    entry_price = 0
    direction = None

    for i in range(50, len(df)):
        row = df.iloc[i]

        if in_position:
            if direction == 'BUY':
                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP'})
                    in_position = False
                elif row['low'] <= entry_price - sl_price_diff:
                    trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL'})
                    in_position = False
            else:  # SELL
                if row['low'] <= entry_price - tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP'})
                    in_position = False
                elif row['high'] >= entry_price + sl_price_diff:
                    trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL'})
                    in_position = False
        else:
            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                # Apply spread to entry
                if direction == 'BUY':
                    entry_price = row['close'] + spread
                else:
                    entry_price = row['close'] - spread

    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips
    }


def backtest_combined(df: pd.DataFrame, pair: str) -> dict:
    """
    COMBINED strategy: 15-min cooldown + Breakeven @ 5 pips.
    """
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    cooldown_minutes = 15
    be_trigger_pips = 5

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    be_trigger_diff = be_trigger_pips / pip_mult

    # Apply spread
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    last_trade_time = None
    entry_price = 0
    direction = None
    current_sl = 0
    be_triggered = False

    for i in range(50, len(df)):
        row = df.iloc[i]
        current_time = row['time']

        if in_position:
            if direction == 'BUY':
                # Check breakeven trigger
                if not be_triggered and row['high'] >= entry_price + be_trigger_diff:
                    current_sl = entry_price + (1 / pip_mult)  # +1 pip on BE
                    be_triggered = True

                # Check TP
                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP'})
                    in_position = False
                    last_trade_time = current_time
                # Check SL/BE
                elif row['low'] <= current_sl:
                    if be_triggered:
                        trades.append({'pnl': 1, 'won': True, 'exit': 'BE'})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL'})
                    in_position = False
                    last_trade_time = current_time

            else:  # SELL
                if not be_triggered and row['low'] <= entry_price - be_trigger_diff:
                    current_sl = entry_price - (1 / pip_mult)
                    be_triggered = True

                if row['low'] <= entry_price - tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP'})
                    in_position = False
                    last_trade_time = current_time
                elif row['high'] >= current_sl:
                    if be_triggered:
                        trades.append({'pnl': 1, 'won': True, 'exit': 'BE'})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL'})
                    in_position = False
                    last_trade_time = current_time

        else:
            # Check cooldown
            if last_trade_time:
                time_since = (current_time - last_trade_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                be_triggered = False
                # Apply spread to entry
                if direction == 'BUY':
                    entry_price = row['close'] + spread
                    current_sl = entry_price - sl_price_diff
                else:
                    entry_price = row['close'] - spread
                    current_sl = entry_price + sl_price_diff

    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0, 'tp': 0, 'be': 0, 'sl': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    tp_exits = sum(1 for t in trades if t['exit'] == 'TP')
    be_exits = sum(1 for t in trades if t['exit'] == 'BE')
    sl_exits = sum(1 for t in trades if t['exit'] == 'SL')

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp': tp_exits,
        'be': be_exits,
        'sl': sl_exits
    }


print("=" * 80)
print("COMPARISON: Combined Strategy vs Previous 3-Day Results")
print("=" * 80)
print("\nPREVIOUS RESULTS (baseline to beat):")
print("  810 trades, 86.2% WR, +3,956 pips")
print("=" * 80)

# Store results
results_original = {}
results_combined = {}

for pair in TEST_PAIRS:
    print(f"\n{pair}:")
    df = fetch_data(pair, 5000)
    print(f"  Loaded {len(df)} candles")

    results_original[pair] = backtest_original(df, pair)
    results_combined[pair] = backtest_combined(df, pair)

    orig = results_original[pair]
    comb = results_combined[pair]

    print(f"  ORIGINAL (no CD):     {orig['trades']:>4}t  {orig['win_rate']:>5.1f}% WR  {orig['total_pips']:>+6.0f}p")
    print(f"  COMBINED (15m+BE@5):  {comb['trades']:>4}t  {comb['win_rate']:>5.1f}% WR  {comb['total_pips']:>+6.0f}p  (TP:{comb.get('tp',0)} BE:{comb.get('be',0)} SL:{comb.get('sl',0)})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print(f"{'PAIR':<10} {'ORIGINAL':<20} {'COMBINED':<20} {'DIFF':<10}")
print("-" * 60)

for pair in TEST_PAIRS:
    orig = results_original[pair]
    comb = results_combined[pair]
    diff = comb['total_pips'] - orig['total_pips']

    print(f"{pair:<10} "
          f"{orig['trades']:>3}t {orig['win_rate']:>5.1f}% {orig['total_pips']:>+5.0f}p   "
          f"{comb['trades']:>3}t {comb['win_rate']:>5.1f}% {comb['total_pips']:>+5.0f}p   "
          f"{diff:+.0f}p")

print("-" * 60)

# Totals
total_orig = sum(r['total_pips'] for r in results_original.values())
total_comb = sum(r['total_pips'] for r in results_combined.values())
trades_orig = sum(r['trades'] for r in results_original.values())
trades_comb = sum(r['trades'] for r in results_combined.values())
wins_orig = sum(r['wins'] for r in results_original.values())
wins_comb = sum(r['wins'] for r in results_combined.values())
wr_orig = wins_orig / trades_orig * 100 if trades_orig > 0 else 0
wr_comb = wins_comb / trades_comb * 100 if trades_comb > 0 else 0

total_tp = sum(r.get('tp', 0) for r in results_combined.values())
total_be = sum(r.get('be', 0) for r in results_combined.values())
total_sl = sum(r.get('sl', 0) for r in results_combined.values())

print(f"{'TOTAL':<10} "
      f"{trades_orig:>3}t {wr_orig:>5.1f}% {total_orig:>+5.0f}p   "
      f"{trades_comb:>3}t {wr_comb:>5.1f}% {total_comb:>+5.0f}p   "
      f"{total_comb - total_orig:+.0f}p")

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(f"\n  PREVIOUS REPORTED RESULTS:")
print(f"    810 trades, 86.2% WR, +3,956 pips")

print(f"\n  CURRENT ORIGINAL (same methodology):")
print(f"    {trades_orig} trades, {wr_orig:.1f}% WR, {total_orig:+,.0f} pips")

print(f"\n  COMBINED (15-min CD + BE @ 5 pips):")
print(f"    {trades_comb} trades, {wr_comb:.1f}% WR, {total_comb:+,.0f} pips")
print(f"    Exit breakdown: {total_tp} TP, {total_be} BE, {total_sl} SL")

print(f"\n  DIFFERENCE vs ORIGINAL:")
print(f"    Trades: {trades_comb - trades_orig:+} ({(trades_comb/trades_orig - 1)*100:+.1f}%)")
print(f"    Win Rate: {wr_comb - wr_orig:+.1f}%")
print(f"    Pips: {total_comb - total_orig:+,.0f}")

if total_comb > total_orig:
    print(f"\n  >>> COMBINED STRATEGY IS BETTER! <<<")
else:
    print(f"\n  >>> ORIGINAL STRATEGY IS BETTER <<<")
    print(f"  Combined loses {total_orig - total_comb:,.0f} pips vs original")

print("=" * 80)

# Per-day analysis
print("\n" + "=" * 80)
print("PER-DAY PROJECTION (based on ~5.5 days of data)")
print("=" * 80)
days = 5.5
print(f"\n  ORIGINAL: {total_orig/days:+.0f} pips/day, {trades_orig/days:.0f} trades/day")
print(f"  COMBINED: {total_comb/days:+.0f} pips/day, {trades_comb/days:.0f} trades/day")
print("=" * 80)
