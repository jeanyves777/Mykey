"""
TEST: Selective Breakeven - Only on historically weaker pairs
=============================================================
Use breakeven ONLY on pairs that benefit from it.
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

TEST_PAIRS = list(config.OPTIMIZED_PAIRS)

SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}

# Pairs that BENEFIT from breakeven (based on analysis)
# These had negative results without BE
BREAKEVEN_PAIRS = ['USD_CHF', 'EUR_USD', 'EUR_GBP', 'NZD_USD']  # Winners with BE

# Pairs to keep ORIGINAL strategy (no BE) - already profitable
NO_BREAKEVEN_PAIRS = ['AUD_CHF', 'GBP_USD', 'USD_JPY', 'USD_CAD', 'AUD_JPY', 'EUR_CAD']


def fetch_data(pair: str, count: int = 5000) -> pd.DataFrame:
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    return df


def backtest_selective(df: pd.DataFrame, pair: str, use_breakeven: bool, cooldown_minutes: int = 15) -> dict:
    """Backtest with selective breakeven."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    be_trigger_pips = 5

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    be_trigger_diff = be_trigger_pips / pip_mult if use_breakeven else 0
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
                if use_breakeven and not be_triggered and row['high'] >= entry_price + be_trigger_diff:
                    current_sl = entry_price + (1 / pip_mult)
                    be_triggered = True

                if row['high'] >= entry_price + tp_price_diff:
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP'})
                    in_position = False
                    last_trade_time = current_time
                elif row['low'] <= current_sl:
                    if be_triggered:
                        trades.append({'pnl': 1, 'won': True, 'exit': 'BE'})
                    else:
                        trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL'})
                    in_position = False
                    last_trade_time = current_time

            else:  # SELL
                if use_breakeven and not be_triggered and row['low'] <= entry_price - be_trigger_diff:
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
            if last_trade_time:
                time_since = (current_time - last_trade_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                be_triggered = False
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

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp': sum(1 for t in trades if t['exit'] == 'TP'),
        'be': sum(1 for t in trades if t['exit'] == 'BE'),
        'sl': sum(1 for t in trades if t['exit'] == 'SL')
    }


print("=" * 80)
print("TEST: SELECTIVE BREAKEVEN STRATEGY")
print("=" * 80)
print(f"BE enabled for: {BREAKEVEN_PAIRS}")
print(f"NO BE for: {NO_BREAKEVEN_PAIRS}")
print("=" * 80)

results = {}

for pair in TEST_PAIRS:
    print(f"\n{pair}:")
    df = fetch_data(pair, 5000)

    # Decide if this pair uses breakeven
    use_be = pair in BREAKEVEN_PAIRS

    # Test all 3 strategies for comparison
    original = backtest_selective(df, pair, use_breakeven=False, cooldown_minutes=30)
    with_be = backtest_selective(df, pair, use_breakeven=True, cooldown_minutes=15)
    selective = backtest_selective(df, pair, use_breakeven=use_be, cooldown_minutes=15)

    results[pair] = {
        'original': original,
        'with_be': with_be,
        'selective': selective,
        'use_be': use_be
    }

    print(f"  Original (30m, no BE):  {original['trades']:>3}t  {original['win_rate']:>5.1f}%  {original['total_pips']:>+5.0f}p")
    print(f"  Full BE (15m + BE@5):   {with_be['trades']:>3}t  {with_be['win_rate']:>5.1f}%  {with_be['total_pips']:>+5.0f}p")
    print(f"  Selective ({'BE' if use_be else 'NO BE'}):        {selective['trades']:>3}t  {selective['win_rate']:>5.1f}%  {selective['total_pips']:>+5.0f}p  {'<-- USE BE' if use_be else '<-- NO BE'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STRATEGY COMPARISON SUMMARY")
print("=" * 80)
print(f"{'PAIR':<10} {'ORIGINAL':<15} {'FULL BE':<15} {'SELECTIVE':<15} {'BEST'}")
print("-" * 70)

for pair in TEST_PAIRS:
    r = results[pair]
    orig = r['original']['total_pips']
    full = r['with_be']['total_pips']
    sel = r['selective']['total_pips']

    best = max(orig, full, sel)
    best_label = 'ORIG' if best == orig else ('FULL BE' if best == full else 'SELECT')

    print(f"{pair:<10} {orig:>+6.0f}p        {full:>+6.0f}p        {sel:>+6.0f}p        {best_label}")

print("-" * 70)

total_orig = sum(r['original']['total_pips'] for r in results.values())
total_full = sum(r['with_be']['total_pips'] for r in results.values())
total_sel = sum(r['selective']['total_pips'] for r in results.values())

trades_orig = sum(r['original']['trades'] for r in results.values())
trades_full = sum(r['with_be']['trades'] for r in results.values())
trades_sel = sum(r['selective']['trades'] for r in results.values())

print(f"{'TOTAL':<10} {total_orig:>+6.0f}p        {total_full:>+6.0f}p        {total_sel:>+6.0f}p")

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"\n  ORIGINAL (30m CD, no BE):        {trades_orig} trades, {total_orig:+.0f} pips")
print(f"  FULL BE (15m CD + BE@5 all):     {trades_full} trades, {total_full:+.0f} pips")
print(f"  SELECTIVE (BE only on weak):     {trades_sel} trades, {total_sel:+.0f} pips")

best_strategy = max([
    ('ORIGINAL', total_orig),
    ('FULL BE', total_full),
    ('SELECTIVE', total_sel)
], key=lambda x: x[1])

print(f"\n  >>> WINNER: {best_strategy[0]} with {best_strategy[1]:+.0f} pips <<<")
print("=" * 80)
