"""
COMBINED ANALYSIS: 15-min Cooldown + Breakeven @ 5 pips
=======================================================
Testing on OANDA 4-DAY data (max available)
Compare to previous 3-day performance.
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

# Test ALL 10 pairs
TEST_PAIRS = list(config.OPTIMIZED_PAIRS)

# 4 days = ~5760 M1 candles (24 * 60 * 4)
CANDLE_COUNT = 5760

def fetch_data(pair: str, count: int = CANDLE_COUNT) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    return df


def backtest_baseline(df: pd.DataFrame, pair: str) -> dict:
    """Current settings: 30-min cooldown, fixed TP/SL."""
    return backtest_combined(df, pair, cooldown_minutes=30, use_breakeven=False, be_trigger=0)


def backtest_combined(df: pd.DataFrame, pair: str, cooldown_minutes: int, use_breakeven: bool, be_trigger: int) -> dict:
    """Backtest with configurable cooldown and breakeven."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    be_trigger_diff = be_trigger / pip_mult if use_breakeven else 0

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
                if use_breakeven and not be_triggered and row['high'] >= entry_price + be_trigger_diff:
                    current_sl = entry_price + (1 / pip_mult)  # +1 pip profit on BE
                    be_triggered = True

                # Check TP
                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP', 'be_used': be_triggered})
                    in_position = False
                    last_trade_time = current_time
                # Check SL (or BE)
                elif row['low'] <= current_sl:
                    if be_triggered:
                        pnl = 1  # Breakeven exit with 1 pip profit
                        trades.append({'pnl': pnl, 'won': True, 'exit_reason': 'BREAKEVEN', 'be_used': True})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit_reason': 'SL', 'be_used': False})
                    in_position = False
                    last_trade_time = current_time

            else:  # SELL
                if use_breakeven and not be_triggered and row['low'] <= entry_price - be_trigger_diff:
                    current_sl = entry_price - (1 / pip_mult)
                    be_triggered = True

                if row['low'] <= entry_price - tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP', 'be_used': be_triggered})
                    in_position = False
                    last_trade_time = current_time
                elif row['high'] >= current_sl:
                    if be_triggered:
                        pnl = 1
                        trades.append({'pnl': pnl, 'won': True, 'exit_reason': 'BREAKEVEN', 'be_used': True})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit_reason': 'SL', 'be_used': False})
                    in_position = False
                    last_trade_time = current_time

        else:
            # Check cooldown
            if last_trade_time:
                time_since = (current_time - last_trade_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            # Check for signal
            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                entry_price = row['close']
                be_triggered = False
                if direction == 'BUY':
                    current_sl = entry_price - sl_price_diff
                else:
                    current_sl = entry_price + sl_price_diff

    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0,
                'tp_exits': 0, 'be_exits': 0, 'sl_exits': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    tp_exits = sum(1 for t in trades if t['exit_reason'] == 'TP')
    be_exits = sum(1 for t in trades if t['exit_reason'] == 'BREAKEVEN')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'SL')

    # Calculate time span for trades per day
    time_span = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 86400
    trades_per_day = len(trades) / time_span if time_span > 0 else 0

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp_exits': tp_exits,
        'be_exits': be_exits,
        'sl_exits': sl_exits,
        'trades_per_day': trades_per_day
    }


print("=" * 80)
print("COMBINED OPTIMIZATION ANALYSIS - 4 DAY OANDA DATA")
print("=" * 80)
print(f"Testing ALL {len(TEST_PAIRS)} pairs")
print(f"Data: ~{CANDLE_COUNT} M1 candles (4 days)")
print("=" * 80)

# Store results
results = {
    'baseline': {},      # Current: 30-min CD, no BE
    'combined': {}       # 15-min CD + BE @ 5 pips
}

for pair in TEST_PAIRS:
    print(f"\n{pair}:")
    df = fetch_data(pair, CANDLE_COUNT)

    # Calculate actual days of data
    time_span = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 86400
    print(f"  Loaded {len(df)} candles ({time_span:.1f} days)")
    print(f"  Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Test both configurations
    results['baseline'][pair] = backtest_baseline(df, pair)
    results['combined'][pair] = backtest_combined(df, pair, cooldown_minutes=15, use_breakeven=True, be_trigger=5)

    b = results['baseline'][pair]
    comb = results['combined'][pair]

    print(f"  BASELINE (30m CD, no BE):  {b['trades']:>3}t  {b['win_rate']:>5.1f}% WR  {b['total_pips']:>+6.0f}p  ({b.get('trades_per_day', 0):.1f}/day)")
    print(f"  COMBINED (15m + BE@5):     {comb['trades']:>3}t  {comb['win_rate']:>5.1f}% WR  {comb['total_pips']:>+6.0f}p  ({comb.get('trades_per_day', 0):.1f}/day)  [{comb['total_pips']-b['total_pips']:+.0f}p]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("4-DAY SUMMARY TABLE")
print("=" * 80)
print(f"{'PAIR':<10} {'BASELINE (30m CD)':<25} {'COMBINED (15m+BE@5)':<25} {'DIFF'}")
print("-" * 75)

for pair in TEST_PAIRS:
    b = results['baseline'][pair]
    comb = results['combined'][pair]
    diff = comb['total_pips'] - b['total_pips']
    diff_str = f"{diff:+.0f}p" if diff != 0 else "0p"
    status = "BETTER" if diff > 0 else ("SAME" if diff == 0 else "WORSE")

    print(f"{pair:<10} "
          f"{b['trades']:>3}t {b['win_rate']:>5.1f}% {b['total_pips']:>+6.0f}p      "
          f"{comb['trades']:>3}t {comb['win_rate']:>5.1f}% {comb['total_pips']:>+6.0f}p      "
          f"{diff_str:>6} {status}")

print("-" * 75)

# Totals
total_baseline = sum(r['total_pips'] for r in results['baseline'].values())
total_combined = sum(r['total_pips'] for r in results['combined'].values())

trades_baseline = sum(r['trades'] for r in results['baseline'].values())
trades_combined = sum(r['trades'] for r in results['combined'].values())

wins_baseline = sum(r['wins'] for r in results['baseline'].values())
wins_combined = sum(r['wins'] for r in results['combined'].values())

wr_baseline = wins_baseline / trades_baseline * 100 if trades_baseline > 0 else 0
wr_combined = wins_combined / trades_combined * 100 if trades_combined > 0 else 0

diff_total = total_combined - total_baseline

print(f"{'TOTAL':<10} "
      f"{trades_baseline:>3}t {wr_baseline:>5.1f}% {total_baseline:>+6.0f}p      "
      f"{trades_combined:>3}t {wr_combined:>5.1f}% {total_combined:>+6.0f}p      "
      f"{diff_total:+.0f}p")

# Exit breakdown for combined
print("\n" + "=" * 80)
print("COMBINED STRATEGY - EXIT BREAKDOWN (4 DAYS)")
print("=" * 80)
print(f"{'PAIR':<10} {'TP':<6} {'BE':<6} {'SL':<6} {'WIN%':<8} {'PIPS':<8} {'TRADES/DAY'}")
print("-" * 60)

total_tp = 0
total_be = 0
total_sl = 0

for pair in TEST_PAIRS:
    r = results['combined'][pair]
    if r['trades'] > 0:
        print(f"{pair:<10} {r['tp_exits']:<6} {r['be_exits']:<6} {r['sl_exits']:<6} {r['win_rate']:<7.1f}% {r['total_pips']:>+6.0f}p   {r.get('trades_per_day', 0):.1f}")
        total_tp += r['tp_exits']
        total_be += r['be_exits']
        total_sl += r['sl_exits']

print("-" * 60)
avg_trades_per_day = trades_combined / 4  # Approximate
print(f"{'TOTAL':<10} {total_tp:<6} {total_be:<6} {total_sl:<6} {wr_combined:<7.1f}% {total_combined:>+6.0f}p   {avg_trades_per_day:.1f}")

# ============================================================================
# COMPARISON WITH 3-DAY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: 3-DAY vs 4-DAY RESULTS")
print("=" * 80)

# Previous 3-day results (from earlier run)
print("\n  3-DAY RESULTS (previous run):")
print("    BASELINE: 136 trades, 69.9% WR, -138 pips")
print("    COMBINED: 240 trades, 87.1% WR, +243 pips")
print("    Improvement: +381 pips")

print(f"\n  4-DAY RESULTS (this run):")
print(f"    BASELINE: {trades_baseline} trades, {wr_baseline:.1f}% WR, {total_baseline:+.0f} pips")
print(f"    COMBINED: {trades_combined} trades, {wr_combined:.1f}% WR, {total_combined:+.0f} pips")
print(f"    Improvement: {diff_total:+.0f} pips")

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if total_combined > total_baseline:
    print(f"\n  >>> COMBINED STRATEGY WINS! <<<")
    print(f"  +{diff_total:.0f} pips improvement over baseline")
    print(f"  Win rate: {wr_baseline:.1f}% -> {wr_combined:.1f}% (+{wr_combined - wr_baseline:.1f}%)")
    print(f"  Trades: {trades_baseline} -> {trades_combined} (+{trades_combined - trades_baseline})")
else:
    print(f"\n  BASELINE performs better on 4-day data")
    print(f"  Difference: {diff_total:.0f} pips")

print("\n  RECOMMENDATION:")
if total_combined > 0 and total_combined > total_baseline:
    print("    IMPLEMENT: 15-min cooldown + Breakeven @ 5 pips")
    print(f"    Expected daily P&L: ~{total_combined/4:.0f} pips/day")
else:
    print("    KEEP current settings or investigate further")

print("=" * 80)
