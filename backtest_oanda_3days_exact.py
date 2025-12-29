"""
BACKTEST: OANDA 3-DAY DATA - EXACT MATCH TO LIVE STRATEGY
==========================================================
This backtest EXACTLY simulates the live trading behavior:
- ONE position per symbol at a time
- Per-pair cooldown (0 or 30 minutes)
- Per-pair session filter (ON or OFF)
- Spread simulation

Uses the OPTIMIZED 2024-12-16 config settings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.config import optimized_paper_config as config
from trading_system.Forex_Trading.strategies.optimized_strategy import calculate_indicators, get_signal

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Try to import OANDA client
try:
    sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading' / 'engine'))
    from oanda_client import OandaClient
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("[WARNING] OANDA client not available")

# Spread simulation (OANDA typical)
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'USD_JPY': 1.0, 'EUR_GBP': 1.2,
    'AUD_CHF': 2.0, 'EUR_CAD': 2.0,
}


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    if not OANDA_AVAILABLE:
        print(f"  [ERROR] OANDA client not available")
        return pd.DataFrame()

    client = OandaClient()
    print(f"  Fetching {count} candles from OANDA...")

    candles = client.get_candles(pair, 'M1', count=count)

    if not candles:
        print(f"  ERROR: No data returned for {pair}")
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    print(f"  Got {len(df)} candles from {df['time'].min()} to {df['time'].max()}")
    return df


def backtest_exact_match(df: pd.DataFrame, pair: str) -> dict:
    """
    Backtest that EXACTLY matches live trading behavior:
    - ONE position at a time
    - Per-pair cooldown (from config)
    - Per-pair session filter (from config)
    """
    if len(df) < 100:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0, 'tp': 0, 'sl': 0}

    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    cooldown_minutes = settings.get('cooldown_minutes', config.COOLDOWN_MINUTES)

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    # Calculate all indicators
    df = calculate_indicators(df.copy())

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    tp_price = 0
    sl_price = 0
    entry_time = None
    last_exit_time = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        current_time = row['time']
        hour_utc = current_time.hour

        if in_position:
            # CHECK EXIT CONDITIONS
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
            else:  # SELL
                if high >= sl_price:
                    exit_reason = 'SL'
                    exit_pnl = -sl_pips
                elif low <= tp_price:
                    exit_reason = 'TP'
                    exit_pnl = tp_pips

            if exit_reason:
                won = exit_pnl > 0
                trades.append({
                    'direction': direction,
                    'pnl': exit_pnl,
                    'won': won,
                    'exit_reason': exit_reason,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                })
                in_position = False
                last_exit_time = current_time

        else:
            # NOT IN POSITION - CHECK FOR ENTRY

            # SESSION FILTER (per-pair)
            if hasattr(config, 'is_allowed_hour'):
                if not config.is_allowed_hour(pair, hour_utc):
                    continue

            # COOLDOWN CHECK (per-pair, after last EXIT)
            if cooldown_minutes > 0 and last_exit_time is not None:
                time_since = (current_time - last_exit_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            # GET SIGNAL using EXACT same function as live trading
            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal is not None:
                in_position = True
                direction = signal
                entry_time = current_time

                if signal == 'BUY':
                    entry_price = row['close'] + spread
                    tp_price = entry_price + tp_price_diff
                    sl_price = entry_price - sl_price_diff
                else:
                    entry_price = row['close'] - spread
                    tp_price = entry_price - tp_price_diff
                    sl_price = entry_price + sl_price_diff

    # Calculate results
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pips': 0, 'tp': 0, 'sl': 0, 'trade_list': []
        }

    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    tp_exits = sum(1 for t in trades if t['exit_reason'] == 'TP')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'SL')

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'tp': tp_exits,
        'sl': sl_exits,
        'trade_list': trades
    }


# =============================================================================
# MAIN BACKTEST
# =============================================================================
print("=" * 80)
print("BACKTEST: OANDA 3-DAY DATA - EXACT MATCH TO LIVE STRATEGY")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nConfiguration (2024-12-16 OPTIMIZED):")
print(f"  - Strategy: optimized_strategy.py (get_signal)")
print(f"  - Config: optimized_paper_config.py")
print(f"  - ONE position per symbol at a time")
print(f"  - Per-pair session filters")
print(f"  - Per-pair cooldown (0 or 30 minutes)")
print(f"  - Data: OANDA 3-day M1 candles (~5000)")
print("=" * 80)

# Print settings table
print(f"\n{'PAIR':<10} {'STRATEGY':<14} {'TP':<4} {'SL':<4} {'CD':<5} {'SF':<4}")
print("-" * 50)
for pair in config.OPTIMIZED_PAIRS:
    s = config.get_pair_settings(pair)
    session = config.TRADING_SESSIONS.get(pair, {})
    sf = 'ON' if session.get('session_filter', False) else 'OFF'
    cd = s.get('cooldown_minutes', 0)
    print(f"{pair:<10} {s['strategy']:<14} {s['tp_pips']:<4} {s['sl_pips']:<4} {cd:<5} {sf:<4}")
print("-" * 50)

results = {}

for pair in config.OPTIMIZED_PAIRS:
    settings = config.get_pair_settings(pair)
    session = config.TRADING_SESSIONS.get(pair, {})
    sf_status = 'ON' if session.get('session_filter', False) else 'OFF'
    cd = settings.get('cooldown_minutes', 0)

    print(f"\n{'=' * 60}")
    print(f"[{pair}] {settings['strategy']} | TP:{settings['tp_pips']} SL:{settings['sl_pips']} | CD:{cd}m SF:{sf_status}")
    print(f"{'=' * 60}")

    df = fetch_oanda_data(pair, count=5000)

    if len(df) == 0:
        print(f"  No data available")
        continue

    # Run EXACT match backtest
    result = backtest_exact_match(df, pair)
    results[pair] = result

    print(f"\n  RESULTS:")
    print(f"    Trades: {result['trades']}")
    print(f"    Win Rate: {result['win_rate']:.1f}% ({result['wins']}W / {result['losses']}L)")
    print(f"    Total Pips: {result['total_pips']:+,.0f}")
    print(f"    Exits: {result['tp']} TP, {result['sl']} SL")

    # Show recent trades
    if result['trade_list']:
        print(f"\n  Recent trades:")
        for t in result['trade_list'][-5:]:
            emoji = "WIN" if t['won'] else "LOSS"
            print(f"    [{emoji}] {t['entry_time'].strftime('%m/%d %H:%M')} {t['direction']} {t['pnl']:+.0f}p ({t['exit_reason']})")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - OANDA 3-DAY BACKTEST (EXACT LIVE STRATEGY MATCH)")
print("=" * 80)
print(f"\n{'PAIR':<10} {'TRADES':<10} {'WINS':<8} {'LOSSES':<8} {'WIN%':<8} {'PIPS':<12} {'TP':<6} {'SL'}")
print("-" * 75)

total_trades = 0
total_wins = 0
total_pips = 0
total_tp = 0
total_sl = 0

for pair in config.OPTIMIZED_PAIRS:
    if pair not in results:
        continue
    r = results[pair]
    print(f"{pair:<10} {r['trades']:<10} {r['wins']:<8} {r['losses']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:>+10,.0f}   {r['tp']:<6} {r['sl']}")

    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']
    total_tp += r['tp']
    total_sl += r['sl']

print("-" * 75)
total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {total_trades:<10} {total_wins:<8} {total_trades - total_wins:<8} {total_wr:<7.1f}% {total_pips:>+10,.0f}   {total_tp:<6} {total_sl}")

# =============================================================================
# EXPECTED vs ACTUAL
# =============================================================================
print("\n" + "=" * 80)
print("EXPECTED vs ACTUAL WIN RATE")
print("=" * 80)

print(f"\n{'PAIR':<10} {'EXP WR%':<10} {'ACT WR%':<10} {'DIFF':<10} {'STATUS'}")
print("-" * 50)

for pair in config.OPTIMIZED_PAIRS:
    if pair not in results:
        continue
    r = results[pair]
    settings = config.get_pair_settings(pair)
    exp_wr = settings.get('expected_wr', 70)
    diff = r['win_rate'] - exp_wr
    status = "OK" if diff >= -15 else "BELOW"
    print(f"{pair:<10} {exp_wr:<10.1f} {r['win_rate']:<10.1f} {diff:+.1f}%    {status}")

# Profitable pairs
profitable_pairs = [p for p, r in results.items() if r['total_pips'] > 0]
losing_pairs = [p for p, r in results.items() if r['total_pips'] <= 0]

print(f"\n  Profitable Pairs ({len(profitable_pairs)}/{len(results)}):")
for pair in sorted(profitable_pairs, key=lambda p: results[p]['total_pips'], reverse=True):
    r = results[pair]
    print(f"    {pair}: {r['total_pips']:+,.0f} pips ({r['win_rate']:.1f}% WR)")

if losing_pairs:
    print(f"\n  Losing Pairs ({len(losing_pairs)}/{len(results)}):")
    for pair in sorted(losing_pairs, key=lambda p: results[p]['total_pips']):
        r = results[pair]
        print(f"    {pair}: {r['total_pips']:+,.0f} pips ({r['win_rate']:.1f}% WR)")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
print(f"\n  Total Result: {total_pips:+,.0f} pips over 3 days")
if total_pips > 0:
    print(f"  >>> PROFITABLE <<<")
else:
    print(f"  >>> LOSING <<<")

print(f"\n  Daily Average: {total_pips / 3:+,.0f} pips/day")
print(f"  Monthly Estimate: {total_pips / 3 * 20:+,.0f} pips/month (20 trading days)")
print("=" * 80)
