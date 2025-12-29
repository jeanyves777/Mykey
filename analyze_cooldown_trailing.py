"""
ANALYSIS: Cooldown and Trailing Stop Optimization
================================================
Tests:
1. Cooldown: 15, 30, 60 minutes between trades per symbol
2. Trailing Stop: 5 pips (OANDA minimum)
3. Breakeven Stop: Move SL to entry after X pips profit

This is ANALYSIS ONLY - no implementation changes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.config import optimized_paper_config as config
from trading_system.Forex_Trading.strategies.optimized_strategy import calculate_indicators, get_signal

import pandas as pd
from datetime import datetime, timedelta

client = OandaClient('practice')

# Test pairs
TEST_PAIRS = ['NZD_USD', 'EUR_GBP', 'AUD_CHF', 'GBP_USD', 'EUR_USD']

def fetch_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    return df

def backtest_with_cooldown(df: pd.DataFrame, pair: str, cooldown_minutes: int) -> dict:
    """Backtest with specific cooldown between trades."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult

    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    last_trade_time = None
    entry_price = 0
    direction = None
    entry_time = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        current_time = row['time']

        if in_position:
            # Check exit
            if direction == 'BUY':
                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP'})
                    in_position = False
                    last_trade_time = current_time
                elif row['low'] <= entry_price - sl_price_diff:
                    trades.append({'pnl': -sl_pips, 'won': False, 'exit_reason': 'SL'})
                    in_position = False
                    last_trade_time = current_time
            else:  # SELL
                if row['low'] <= entry_price - tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP'})
                    in_position = False
                    last_trade_time = current_time
                elif row['high'] >= entry_price + sl_price_diff:
                    trades.append({'pnl': -sl_pips, 'won': False, 'exit_reason': 'SL'})
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
                entry_time = current_time

    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_pips': 0, 'avg_trades_per_day': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)

    # Calculate trades per day
    time_span = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / 86400
    trades_per_day = len(trades) / time_span if time_span > 0 else 0

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'avg_trades_per_day': trades_per_day
    }


def backtest_with_trailing(df: pd.DataFrame, pair: str, trailing_pips: int) -> dict:
    """Backtest with trailing stop."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    trailing_price_diff = trailing_pips / pip_mult

    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    trailing_sl = 0
    max_favorable = 0

    for i in range(50, len(df)):
        row = df.iloc[i]

        if in_position:
            if direction == 'BUY':
                # Update max favorable excursion
                if row['high'] > entry_price:
                    max_favorable = max(max_favorable, row['high'] - entry_price)
                    # Update trailing stop
                    new_trailing_sl = row['high'] - trailing_price_diff
                    if new_trailing_sl > trailing_sl:
                        trailing_sl = new_trailing_sl

                # Check exits
                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP'})
                    in_position = False
                elif row['low'] <= trailing_sl:
                    # Trailing stop hit
                    pnl = (trailing_sl - entry_price) * pip_mult
                    trades.append({'pnl': pnl, 'won': pnl > 0, 'exit_reason': 'TRAILING'})
                    in_position = False

            else:  # SELL
                if row['low'] < entry_price:
                    max_favorable = max(max_favorable, entry_price - row['low'])
                    new_trailing_sl = row['low'] + trailing_price_diff
                    if new_trailing_sl < trailing_sl:
                        trailing_sl = new_trailing_sl

                if row['low'] <= entry_price - tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP'})
                    in_position = False
                elif row['high'] >= trailing_sl:
                    pnl = (entry_price - trailing_sl) * pip_mult
                    trades.append({'pnl': pnl, 'won': pnl > 0, 'exit_reason': 'TRAILING'})
                    in_position = False
        else:
            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                entry_price = row['close']
                max_favorable = 0
                # Initial trailing stop = regular stop loss
                if direction == 'BUY':
                    trailing_sl = entry_price - sl_price_diff
                else:
                    trailing_sl = entry_price + sl_price_diff

    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_pips': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)

    # Count exit reasons
    tp_exits = sum(1 for t in trades if t['exit_reason'] == 'TP')
    trailing_exits = sum(1 for t in trades if t['exit_reason'] == 'TRAILING')

    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp_exits': tp_exits,
        'trailing_exits': trailing_exits
    }


def backtest_with_breakeven(df: pd.DataFrame, pair: str, breakeven_trigger_pips: int) -> dict:
    """Backtest with breakeven stop (move SL to entry after X pips profit)."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    be_trigger = breakeven_trigger_pips / pip_mult

    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    current_sl = 0
    be_triggered = False

    for i in range(50, len(df)):
        row = df.iloc[i]

        if in_position:
            if direction == 'BUY':
                # Check if we should move to breakeven
                if not be_triggered and row['high'] >= entry_price + be_trigger:
                    current_sl = entry_price + 0.00001  # Slightly above entry for 1 pip profit
                    be_triggered = True

                # Check exits
                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP', 'be_used': be_triggered})
                    in_position = False
                elif row['low'] <= current_sl:
                    if be_triggered:
                        pnl = 1  # Breakeven with small profit
                        trades.append({'pnl': pnl, 'won': True, 'exit_reason': 'BREAKEVEN', 'be_used': True})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit_reason': 'SL', 'be_used': False})
                    in_position = False

            else:  # SELL
                if not be_triggered and row['low'] <= entry_price - be_trigger:
                    current_sl = entry_price - 0.00001
                    be_triggered = True

                if row['low'] <= entry_price - tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit_reason': 'TP', 'be_used': be_triggered})
                    in_position = False
                elif row['high'] >= current_sl:
                    if be_triggered:
                        pnl = 1
                        trades.append({'pnl': pnl, 'won': True, 'exit_reason': 'BREAKEVEN', 'be_used': True})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit_reason': 'SL', 'be_used': False})
                    in_position = False
        else:
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
        return {'trades': 0, 'win_rate': 0, 'total_pips': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)

    tp_exits = sum(1 for t in trades if t['exit_reason'] == 'TP')
    be_exits = sum(1 for t in trades if t['exit_reason'] == 'BREAKEVEN')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'SL')

    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp_exits': tp_exits,
        'be_exits': be_exits,
        'sl_exits': sl_exits
    }


print("=" * 80)
print("COOLDOWN & TRAILING STOP ANALYSIS")
print("=" * 80)
print(f"Testing on: {TEST_PAIRS}")
print(f"Data: OANDA 3-day (5000 candles)")
print("=" * 80)

# ============================================================================
# TEST 1: COOLDOWN ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: COOLDOWN ANALYSIS (15 vs 30 vs 60 minutes)")
print("=" * 80)

cooldowns = [15, 30, 60]
cooldown_results = {}

for pair in TEST_PAIRS:
    print(f"\nFetching {pair}...")
    df = fetch_data(pair, 5000)
    print(f"  Got {len(df)} candles")

    cooldown_results[pair] = {}
    for cd in cooldowns:
        result = backtest_with_cooldown(df, pair, cd)
        cooldown_results[pair][cd] = result
        print(f"  Cooldown {cd}m: {result['trades']} trades, {result['win_rate']:.1f}% WR, {result['total_pips']:+.0f} pips, {result['avg_trades_per_day']:.1f} trades/day")

print("\n" + "-" * 80)
print("COOLDOWN SUMMARY:")
print("-" * 80)
print(f"{'PAIR':<10} {'15min':<25} {'30min':<25} {'60min':<25}")
print("-" * 80)

for pair in TEST_PAIRS:
    cd15 = cooldown_results[pair][15]
    cd30 = cooldown_results[pair][30]
    cd60 = cooldown_results[pair][60]
    print(f"{pair:<10} {cd15['trades']:>3}t {cd15['win_rate']:>5.1f}% {cd15['total_pips']:>+6.0f}p   "
          f"{cd30['trades']:>3}t {cd30['win_rate']:>5.1f}% {cd30['total_pips']:>+6.0f}p   "
          f"{cd60['trades']:>3}t {cd60['win_rate']:>5.1f}% {cd60['total_pips']:>+6.0f}p")

# ============================================================================
# TEST 2: TRAILING STOP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: TRAILING STOP ANALYSIS (5 pips - OANDA minimum)")
print("=" * 80)
print("Comparing: Fixed TP/SL vs 5-pip Trailing Stop")
print("-" * 80)

trailing_results = {}

for pair in TEST_PAIRS:
    print(f"\n{pair}:")
    df = fetch_data(pair, 5000)

    # Baseline (no trailing)
    baseline = backtest_with_cooldown(df, pair, 30)

    # With trailing
    trailing = backtest_with_trailing(df, pair, 5)

    trailing_results[pair] = {
        'baseline': baseline,
        'trailing': trailing
    }

    print(f"  Baseline (Fixed):  {baseline['trades']:>3} trades, {baseline['win_rate']:>5.1f}% WR, {baseline['total_pips']:>+6.0f} pips")
    print(f"  5-pip Trailing:    {trailing['trades']:>3} trades, {trailing['win_rate']:>5.1f}% WR, {trailing['total_pips']:>+6.0f} pips")
    print(f"                     TP exits: {trailing.get('tp_exits', 0)}, Trailing exits: {trailing.get('trailing_exits', 0)}")

    diff = trailing['total_pips'] - baseline['total_pips']
    print(f"  Difference: {diff:+.0f} pips ({'BETTER' if diff > 0 else 'WORSE'})")

# ============================================================================
# TEST 3: BREAKEVEN STOP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: BREAKEVEN STOP ANALYSIS")
print("=" * 80)
print("Move SL to entry after X pips profit")
print("Testing triggers: 3, 5, 7 pips")
print("-" * 80)

be_triggers = [3, 5, 7]
breakeven_results = {}

for pair in TEST_PAIRS:
    print(f"\n{pair}:")
    df = fetch_data(pair, 5000)

    # Baseline
    baseline = backtest_with_cooldown(df, pair, 30)
    print(f"  Baseline (No BE):  {baseline['trades']:>3} trades, {baseline['win_rate']:>5.1f}% WR, {baseline['total_pips']:>+6.0f} pips")

    breakeven_results[pair] = {'baseline': baseline}

    for trigger in be_triggers:
        be_result = backtest_with_breakeven(df, pair, trigger)
        breakeven_results[pair][trigger] = be_result

        diff = be_result['total_pips'] - baseline['total_pips']
        print(f"  BE @ {trigger} pips:       {be_result['trades']:>3} trades, {be_result['win_rate']:>5.1f}% WR, {be_result['total_pips']:>+6.0f} pips "
              f"(TP:{be_result.get('tp_exits',0)} BE:{be_result.get('be_exits',0)} SL:{be_result.get('sl_exits',0)}) [{diff:+.0f}p]")

# ============================================================================
# FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL ANALYSIS SUMMARY")
print("=" * 80)

print("\n1. COOLDOWN ANALYSIS:")
print("-" * 40)
total_15 = sum(cooldown_results[p][15]['total_pips'] for p in TEST_PAIRS)
total_30 = sum(cooldown_results[p][30]['total_pips'] for p in TEST_PAIRS)
total_60 = sum(cooldown_results[p][60]['total_pips'] for p in TEST_PAIRS)
trades_15 = sum(cooldown_results[p][15]['trades'] for p in TEST_PAIRS)
trades_30 = sum(cooldown_results[p][30]['trades'] for p in TEST_PAIRS)
trades_60 = sum(cooldown_results[p][60]['trades'] for p in TEST_PAIRS)

print(f"   15-min cooldown: {trades_15} trades, {total_15:+.0f} pips total")
print(f"   30-min cooldown: {trades_30} trades, {total_30:+.0f} pips total (CURRENT)")
print(f"   60-min cooldown: {trades_60} trades, {total_60:+.0f} pips total")

best_cd = max([(15, total_15), (30, total_30), (60, total_60)], key=lambda x: x[1])
print(f"   BEST: {best_cd[0]}-minute cooldown ({best_cd[1]:+.0f} pips)")

print("\n2. TRAILING STOP (5 pips):")
print("-" * 40)
total_baseline = sum(trailing_results[p]['baseline']['total_pips'] for p in TEST_PAIRS)
total_trailing = sum(trailing_results[p]['trailing']['total_pips'] for p in TEST_PAIRS)
print(f"   Fixed TP/SL:     {total_baseline:+.0f} pips")
print(f"   5-pip Trailing:  {total_trailing:+.0f} pips")
print(f"   Difference:      {total_trailing - total_baseline:+.0f} pips")
print(f"   VERDICT: {'Trailing BETTER' if total_trailing > total_baseline else 'Fixed BETTER'}")

print("\n3. BREAKEVEN STOP:")
print("-" * 40)
for trigger in be_triggers:
    total_be = sum(breakeven_results[p][trigger]['total_pips'] for p in TEST_PAIRS)
    diff = total_be - total_baseline
    print(f"   BE @ {trigger} pips: {total_be:+.0f} pips ({diff:+.0f} vs baseline)")

best_be = max(be_triggers, key=lambda t: sum(breakeven_results[p][t]['total_pips'] for p in TEST_PAIRS))
best_be_pips = sum(breakeven_results[p][best_be]['total_pips'] for p in TEST_PAIRS)
print(f"   BEST BE trigger: {best_be} pips ({best_be_pips:+.0f} total)")

print("\n" + "=" * 80)
print("RECOMMENDATIONS (Analysis Only - No changes made)")
print("=" * 80)
