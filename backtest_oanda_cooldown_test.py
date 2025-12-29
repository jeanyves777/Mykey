"""
BACKTEST: OANDA 3-DAY with COOLDOWN TEST
========================================
Test 15-min and 1-hour cooldown between trades per symbol.
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

SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    return df


def backtest_with_cooldown(df: pd.DataFrame, pair: str, cooldown_minutes: int) -> dict:
    """Backtest with cooldown between trades."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_diff = tp_pips / pip_mult
    sl_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    session_hours = config.TRADING_SESSIONS.get(pair, {}).get('allowed_hours', list(range(24)))

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
        hour_utc = current_time.hour

        if in_position:
            if direction == 'BUY':
                if row['high'] >= entry_price + tp_diff:
                    duration = (current_time - entry_time).total_seconds() / 60
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP', 'duration': duration})
                    in_position = False
                    last_trade_time = current_time
                elif row['low'] <= entry_price - sl_diff:
                    duration = (current_time - entry_time).total_seconds() / 60
                    trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL', 'duration': duration})
                    in_position = False
                    last_trade_time = current_time
            else:  # SELL
                if row['low'] <= entry_price - tp_diff:
                    duration = (current_time - entry_time).total_seconds() / 60
                    trades.append({'pnl': tp_pips, 'won': True, 'exit': 'TP', 'duration': duration})
                    in_position = False
                    last_trade_time = current_time
                elif row['high'] >= entry_price + sl_diff:
                    duration = (current_time - entry_time).total_seconds() / 60
                    trades.append({'pnl': -sl_pips, 'won': False, 'exit': 'SL', 'duration': duration})
                    in_position = False
                    last_trade_time = current_time
        else:
            # Check session filter
            if hour_utc not in session_hours:
                continue

            # Check cooldown
            if cooldown_minutes > 0 and last_trade_time:
                time_since = (current_time - last_trade_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                entry_time = current_time
                if direction == 'BUY':
                    entry_price = row['close'] + spread
                else:
                    entry_price = row['close'] - spread

    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0, 'avg_duration': 0}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    avg_duration = sum(t['duration'] for t in trades) / len(trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'avg_duration': avg_duration
    }


print("=" * 80)
print("BACKTEST: COOLDOWN COMPARISON (0 vs 15min vs 60min)")
print("=" * 80)
print(f"Time: {datetime.now()}")
print("=" * 80)

# Store results for each cooldown
results = {0: {}, 15: {}, 60: {}}

for pair in config.OPTIMIZED_PAIRS:
    print(f"\n{pair}:")
    df = fetch_oanda_data(pair, 5000)
    print(f"  Loaded {len(df)} candles")

    for cd in [0, 15, 60]:
        result = backtest_with_cooldown(df, pair, cd)
        results[cd][pair] = result

    r0 = results[0][pair]
    r15 = results[15][pair]
    r60 = results[60][pair]

    print(f"  NO COOLDOWN:  {r0['trades']:>4}t  {r0['win_rate']:>5.1f}% WR  {r0['total_pips']:>+7.1f}p")
    print(f"  15-MIN CD:    {r15['trades']:>4}t  {r15['win_rate']:>5.1f}% WR  {r15['total_pips']:>+7.1f}p  [{r15['total_pips']-r0['total_pips']:+.0f}p]")
    print(f"  60-MIN CD:    {r60['trades']:>4}t  {r60['win_rate']:>5.1f}% WR  {r60['total_pips']:>+7.1f}p  [{r60['total_pips']-r0['total_pips']:+.0f}p]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'PAIR':<10} {'NO COOLDOWN':<20} {'15-MIN CD':<20} {'60-MIN CD':<20}")
print("-" * 75)

for pair in config.OPTIMIZED_PAIRS:
    r0 = results[0][pair]
    r15 = results[15][pair]
    r60 = results[60][pair]

    print(f"{pair:<10} "
          f"{r0['trades']:>3}t {r0['win_rate']:>5.1f}% {r0['total_pips']:>+6.0f}p   "
          f"{r15['trades']:>3}t {r15['win_rate']:>5.1f}% {r15['total_pips']:>+6.0f}p   "
          f"{r60['trades']:>3}t {r60['win_rate']:>5.1f}% {r60['total_pips']:>+6.0f}p")

print("-" * 75)

# Totals
for cd in [0, 15, 60]:
    total_trades = sum(r['trades'] for r in results[cd].values())
    total_wins = sum(r['wins'] for r in results[cd].values())
    total_pips = sum(r['total_pips'] for r in results[cd].values())
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    results[cd]['_total'] = {'trades': total_trades, 'wins': total_wins, 'win_rate': total_wr, 'total_pips': total_pips}

r0 = results[0]['_total']
r15 = results[15]['_total']
r60 = results[60]['_total']

print(f"{'TOTAL':<10} "
      f"{r0['trades']:>3}t {r0['win_rate']:>5.1f}% {r0['total_pips']:>+6.0f}p   "
      f"{r15['trades']:>3}t {r15['win_rate']:>5.1f}% {r15['total_pips']:>+6.0f}p   "
      f"{r60['trades']:>3}t {r60['win_rate']:>5.1f}% {r60['total_pips']:>+6.0f}p")

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print(f"\n  NO COOLDOWN (current):")
print(f"    {r0['trades']} trades, {r0['win_rate']:.1f}% WR, {r0['total_pips']:+,.0f} pips")

print(f"\n  15-MINUTE COOLDOWN:")
print(f"    {r15['trades']} trades, {r15['win_rate']:.1f}% WR, {r15['total_pips']:+,.0f} pips")
print(f"    Difference: {r15['total_pips'] - r0['total_pips']:+,.0f} pips")

print(f"\n  60-MINUTE (1 HOUR) COOLDOWN:")
print(f"    {r60['trades']} trades, {r60['win_rate']:.1f}% WR, {r60['total_pips']:+,.0f} pips")
print(f"    Difference: {r60['total_pips'] - r0['total_pips']:+,.0f} pips")

# Find best
best = max([
    ('NO COOLDOWN', r0['total_pips']),
    ('15-MIN CD', r15['total_pips']),
    ('60-MIN CD', r60['total_pips'])
], key=lambda x: x[1])

print(f"\n  >>> BEST: {best[0]} with {best[1]:+,.0f} pips <<<")
print("=" * 80)
