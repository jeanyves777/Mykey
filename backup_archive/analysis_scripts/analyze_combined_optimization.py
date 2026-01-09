"""
COMBINED ANALYSIS: 15-min Cooldown + Breakeven @ 5 pips
=======================================================
Testing the BEST settings found together.
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

def fetch_data(pair: str, count: int = 5000) -> pd.DataFrame:
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

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp_exits': tp_exits,
        'be_exits': be_exits,
        'sl_exits': sl_exits
    }


print("=" * 80)
print("COMBINED OPTIMIZATION ANALYSIS")
print("=" * 80)
print(f"Testing ALL {len(TEST_PAIRS)} pairs on OANDA 3-day data")
print("=" * 80)

# Store results
results = {
    'baseline': {},      # Current: 30-min CD, no BE
    'cd15_only': {},     # 15-min CD, no BE
    'be5_only': {},      # 30-min CD, BE @ 5 pips
    'combined': {}       # 15-min CD + BE @ 5 pips
}

for pair in TEST_PAIRS:
    print(f"\n{pair}:")
    df = fetch_data(pair, 5000)
    print(f"  Loaded {len(df)} candles")

    # Test all configurations
    results['baseline'][pair] = backtest_baseline(df, pair)
    results['cd15_only'][pair] = backtest_combined(df, pair, cooldown_minutes=15, use_breakeven=False, be_trigger=0)
    results['be5_only'][pair] = backtest_combined(df, pair, cooldown_minutes=30, use_breakeven=True, be_trigger=5)
    results['combined'][pair] = backtest_combined(df, pair, cooldown_minutes=15, use_breakeven=True, be_trigger=5)

    b = results['baseline'][pair]
    c15 = results['cd15_only'][pair]
    be5 = results['be5_only'][pair]
    comb = results['combined'][pair]

    print(f"  Baseline (30m CD, no BE):  {b['trades']:>3}t  {b['win_rate']:>5.1f}% WR  {b['total_pips']:>+6.0f}p")
    print(f"  15-min CD only:            {c15['trades']:>3}t  {c15['win_rate']:>5.1f}% WR  {c15['total_pips']:>+6.0f}p  [{c15['total_pips']-b['total_pips']:+.0f}p]")
    print(f"  BE @ 5 pips only:          {be5['trades']:>3}t  {be5['win_rate']:>5.1f}% WR  {be5['total_pips']:>+6.0f}p  [{be5['total_pips']-b['total_pips']:+.0f}p]")
    print(f"  COMBINED (15m + BE@5):     {comb['trades']:>3}t  {comb['win_rate']:>5.1f}% WR  {comb['total_pips']:>+6.0f}p  [{comb['total_pips']-b['total_pips']:+.0f}p]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE - ALL PAIRS")
print("=" * 80)
print(f"{'PAIR':<10} {'BASELINE':<20} {'15m CD':<20} {'BE@5':<20} {'COMBINED':<20}")
print("-" * 90)

for pair in TEST_PAIRS:
    b = results['baseline'][pair]
    c15 = results['cd15_only'][pair]
    be5 = results['be5_only'][pair]
    comb = results['combined'][pair]

    print(f"{pair:<10} "
          f"{b['trades']:>2}t {b['win_rate']:>5.1f}% {b['total_pips']:>+5.0f}p  "
          f"{c15['trades']:>2}t {c15['win_rate']:>5.1f}% {c15['total_pips']:>+5.0f}p  "
          f"{be5['trades']:>2}t {be5['win_rate']:>5.1f}% {be5['total_pips']:>+5.0f}p  "
          f"{comb['trades']:>2}t {comb['win_rate']:>5.1f}% {comb['total_pips']:>+5.0f}p")

print("-" * 90)

# Totals
total_baseline = sum(r['total_pips'] for r in results['baseline'].values())
total_cd15 = sum(r['total_pips'] for r in results['cd15_only'].values())
total_be5 = sum(r['total_pips'] for r in results['be5_only'].values())
total_combined = sum(r['total_pips'] for r in results['combined'].values())

trades_baseline = sum(r['trades'] for r in results['baseline'].values())
trades_cd15 = sum(r['trades'] for r in results['cd15_only'].values())
trades_be5 = sum(r['trades'] for r in results['be5_only'].values())
trades_combined = sum(r['trades'] for r in results['combined'].values())

wins_baseline = sum(r['wins'] for r in results['baseline'].values())
wins_cd15 = sum(r['wins'] for r in results['cd15_only'].values())
wins_be5 = sum(r['wins'] for r in results['be5_only'].values())
wins_combined = sum(r['wins'] for r in results['combined'].values())

wr_baseline = wins_baseline / trades_baseline * 100 if trades_baseline > 0 else 0
wr_cd15 = wins_cd15 / trades_cd15 * 100 if trades_cd15 > 0 else 0
wr_be5 = wins_be5 / trades_be5 * 100 if trades_be5 > 0 else 0
wr_combined = wins_combined / trades_combined * 100 if trades_combined > 0 else 0

print(f"{'TOTAL':<10} "
      f"{trades_baseline:>2}t {wr_baseline:>5.1f}% {total_baseline:>+5.0f}p  "
      f"{trades_cd15:>2}t {wr_cd15:>5.1f}% {total_cd15:>+5.0f}p  "
      f"{trades_be5:>2}t {wr_be5:>5.1f}% {total_be5:>+5.0f}p  "
      f"{trades_combined:>2}t {wr_combined:>5.1f}% {total_combined:>+5.0f}p")

# Exit breakdown for combined
print("\n" + "=" * 80)
print("COMBINED STRATEGY EXIT BREAKDOWN")
print("=" * 80)
print(f"{'PAIR':<10} {'TP':<6} {'BE':<6} {'SL':<6} {'WIN%':<8} {'PIPS'}")
print("-" * 50)

total_tp = 0
total_be = 0
total_sl = 0

for pair in TEST_PAIRS:
    r = results['combined'][pair]
    if r['trades'] > 0:
        print(f"{pair:<10} {r['tp_exits']:<6} {r['be_exits']:<6} {r['sl_exits']:<6} {r['win_rate']:<7.1f}% {r['total_pips']:>+.0f}p")
        total_tp += r['tp_exits']
        total_be += r['be_exits']
        total_sl += r['sl_exits']

print("-" * 50)
print(f"{'TOTAL':<10} {total_tp:<6} {total_be:<6} {total_sl:<6} {wr_combined:<7.1f}% {total_combined:>+.0f}p")

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print(f"\n  BASELINE (Current Settings):")
print(f"    - 30-min cooldown, Fixed TP/SL")
print(f"    - {trades_baseline} trades, {wr_baseline:.1f}% WR, {total_baseline:+.0f} pips")

print(f"\n  15-MINUTE COOLDOWN ONLY:")
print(f"    - More frequent trading")
print(f"    - {trades_cd15} trades, {wr_cd15:.1f}% WR, {total_cd15:+.0f} pips")
print(f"    - Improvement: {total_cd15 - total_baseline:+.0f} pips")

print(f"\n  BREAKEVEN @ 5 PIPS ONLY:")
print(f"    - Move SL to entry after 5 pips profit")
print(f"    - {trades_be5} trades, {wr_be5:.1f}% WR, {total_be5:+.0f} pips")
print(f"    - Improvement: {total_be5 - total_baseline:+.0f} pips")

print(f"\n  >>> COMBINED (15-min CD + BE @ 5 pips) <<<")
print(f"    - Best of both strategies")
print(f"    - {trades_combined} trades, {wr_combined:.1f}% WR, {total_combined:+.0f} pips")
print(f"    - Improvement: {total_combined - total_baseline:+.0f} pips vs baseline")
print(f"    - Exit breakdown: {total_tp} TP, {total_be} BE, {total_sl} SL")

best = max([
    ('BASELINE', total_baseline),
    ('15m CD Only', total_cd15),
    ('BE@5 Only', total_be5),
    ('COMBINED', total_combined)
], key=lambda x: x[1])

print(f"\n  WINNER: {best[0]} with {best[1]:+.0f} pips")
print("=" * 80)
