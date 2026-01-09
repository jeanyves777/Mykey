"""
BACKTEST: OANDA 3-DAY with BREAKEVEN @ 5 pips
=============================================
Same as backtest_oanda_3days.py but with breakeven added.
Move SL to entry + 1 pip after price moves 5 pips in our favor.
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

# Spread simulation (OANDA typical)
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}

# BREAKEVEN SETTINGS
BE_TRIGGER_PIPS = 5  # Move to breakeven after 5 pips profit
BE_PROFIT_PIPS = 1   # Lock in 1 pip profit on breakeven


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    print(f"  Fetching {count} candles from OANDA...")
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    return df


def backtest_pair_with_breakeven(df: pd.DataFrame, pair: str) -> dict:
    """Backtest a single pair with breakeven."""
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    strategy = settings['strategy']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_diff = tp_pips / pip_mult
    sl_diff = sl_pips / pip_mult
    be_trigger = BE_TRIGGER_PIPS / pip_mult
    be_profit = BE_PROFIT_PIPS / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    # Session filter
    session_hours = config.TRADING_SESSIONS.get(pair, {}).get('allowed_hours', list(range(24)))

    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    current_sl = 0
    be_triggered = False
    entry_time = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        current_time = row['time']
        hour_utc = current_time.hour

        if in_position:
            if direction == 'BUY':
                # Check breakeven trigger
                if not be_triggered and row['high'] >= entry_price + be_trigger:
                    current_sl = entry_price + be_profit
                    be_triggered = True

                # Check TP
                if row['high'] >= entry_price + tp_diff:
                    duration = (current_time - entry_time).total_seconds() / 60
                    trades.append({
                        'pnl': tp_pips, 'won': True, 'exit': 'TP',
                        'direction': direction, 'time': entry_time, 'duration': duration
                    })
                    in_position = False
                # Check SL/BE
                elif row['low'] <= current_sl:
                    duration = (current_time - entry_time).total_seconds() / 60
                    if be_triggered:
                        trades.append({
                            'pnl': BE_PROFIT_PIPS, 'won': True, 'exit': 'BE',
                            'direction': direction, 'time': entry_time, 'duration': duration
                        })
                    else:
                        trades.append({
                            'pnl': -sl_pips, 'won': False, 'exit': 'SL',
                            'direction': direction, 'time': entry_time, 'duration': duration
                        })
                    in_position = False

            else:  # SELL
                if not be_triggered and row['low'] <= entry_price - be_trigger:
                    current_sl = entry_price - be_profit
                    be_triggered = True

                if row['low'] <= entry_price - tp_diff:
                    duration = (current_time - entry_time).total_seconds() / 60
                    trades.append({
                        'pnl': tp_pips, 'won': True, 'exit': 'TP',
                        'direction': direction, 'time': entry_time, 'duration': duration
                    })
                    in_position = False
                elif row['high'] >= current_sl:
                    duration = (current_time - entry_time).total_seconds() / 60
                    if be_triggered:
                        trades.append({
                            'pnl': BE_PROFIT_PIPS, 'won': True, 'exit': 'BE',
                            'direction': direction, 'time': entry_time, 'duration': duration
                        })
                    else:
                        trades.append({
                            'pnl': -sl_pips, 'won': False, 'exit': 'SL',
                            'direction': direction, 'time': entry_time, 'duration': duration
                        })
                    in_position = False

        else:
            # Check session filter
            if hour_utc not in session_hours:
                continue

            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal:
                in_position = True
                direction = signal
                be_triggered = False
                entry_time = current_time

                if direction == 'BUY':
                    entry_price = row['close'] + spread
                    current_sl = entry_price - sl_diff
                else:
                    entry_price = row['close'] - spread
                    current_sl = entry_price + sl_diff

    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0,
                'tp': 0, 'be': 0, 'sl': 0, 'avg_duration': 0, 'recent': []}

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    tp_exits = sum(1 for t in trades if t['exit'] == 'TP')
    be_exits = sum(1 for t in trades if t['exit'] == 'BE')
    sl_exits = sum(1 for t in trades if t['exit'] == 'SL')
    avg_duration = sum(t['duration'] for t in trades) / len(trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp': tp_exits,
        'be': be_exits,
        'sl': sl_exits,
        'avg_duration': avg_duration,
        'recent': trades[-5:]
    }


print("=" * 80)
print("BACKTEST: OANDA 3-DAY WITH BREAKEVEN @ 5 PIPS")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nBreakeven Settings:")
print(f"  - Trigger: After {BE_TRIGGER_PIPS} pips profit")
print(f"  - Lock in: {BE_PROFIT_PIPS} pip profit on BE exit")
print("=" * 80)

all_results = {}

for pair in config.OPTIMIZED_PAIRS:
    print(f"\n{'=' * 60}")
    print(f"[{pair}]")
    print(f"{'=' * 60}")

    settings = config.get_pair_settings(pair)
    print(f"Strategy: {settings['strategy']}")
    print(f"TP: {settings['tp_pips']} pips | SL: {settings['sl_pips']} pips")

    df = fetch_oanda_data(pair, 5000)
    print(f"  Got {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    result = backtest_pair_with_breakeven(df, pair)
    all_results[pair] = result

    print(f"\nRESULTS (with breakeven):")
    print(f"  Trades: {result['trades']}")
    print(f"  Win Rate: {result['win_rate']:.1f}% ({result['wins']}W/{result['losses']}L)")
    print(f"  Total Pips: {result['total_pips']:+.1f}")
    print(f"  Exit Breakdown: TP={result['tp']}, BE={result['be']}, SL={result['sl']}")
    print(f"  Avg Duration: {result['avg_duration']:.1f} min")

    if result['recent']:
        print(f"\n  Recent trades:")
        for t in result['recent']:
            exit_type = t['exit']
            pnl = t['pnl']
            dur = t['duration']
            time_str = t['time'].strftime('%m/%d %H:%M')
            status = '[WIN]' if t['won'] else '[LOSS]'
            print(f"    {status} {time_str} {t['direction']} ({exit_type}: {pnl:+.1f}p, {dur:.0f}m)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY - WITH BREAKEVEN @ 5 PIPS")
print("=" * 80)
print(f"\n{'PAIR':<10} {'TRADES':<8} {'WIN%':<8} {'PIPS':<10} {'TP':<6} {'BE':<6} {'SL':<6}")
print("-" * 65)

total_trades = 0
total_wins = 0
total_pips = 0
total_tp = 0
total_be = 0
total_sl = 0

for pair in config.OPTIMIZED_PAIRS:
    r = all_results[pair]
    print(f"{pair:<10} {r['trades']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:>+8.1f}  {r['tp']:<6} {r['be']:<6} {r['sl']:<6}")

    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']
    total_tp += r['tp']
    total_be += r['be']
    total_sl += r['sl']

print("-" * 65)
total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {total_trades:<8} {total_wr:<7.1f}% {total_pips:>+8.1f}  {total_tp:<6} {total_be:<6} {total_sl:<6}")

print("\n" + "=" * 80)
print("COMPARISON: WITH vs WITHOUT BREAKEVEN")
print("=" * 80)
print("\n  WITHOUT BREAKEVEN (previous run):")
print("    1073 trades, 84.2% WR, +4,298 pips")
print(f"\n  WITH BREAKEVEN @ 5 PIPS (this run):")
print(f"    {total_trades} trades, {total_wr:.1f}% WR, {total_pips:+,.1f} pips")
print(f"    Exit breakdown: {total_tp} TP, {total_be} BE, {total_sl} SL")

diff = total_pips - 4298
print(f"\n  DIFFERENCE: {diff:+,.1f} pips")

if diff > 0:
    print(f"\n  >>> BREAKEVEN IMPROVES RESULTS BY {diff:+,.1f} PIPS! <<<")
else:
    print(f"\n  >>> BREAKEVEN REDUCES RESULTS BY {abs(diff):,.1f} PIPS <<<")

print("=" * 80)
