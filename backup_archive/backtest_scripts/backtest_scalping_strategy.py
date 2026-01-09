"""
BACKTEST: SCALPING STRATEGY - MULTI-INDICATOR VALIDATION
=========================================================

Test the 5-indicator scalping strategy on OANDA 3-day data
Requirements (ALL must pass):
1. Trend alignment (EMA9>EMA21>EMA50)
2. Momentum confirmation (RSI 40-60)
3. MACD trigger (crossover + histogram)
4. Candle confirmation (strong body >40%)
5. Volatility filter (ATR in range)

Target: 75%+ win rate with 5-15 pip scalps
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.config import scalping_config as config
from trading_system.Forex_Trading.strategies.enhanced_scalping_strategy import get_signal, calculate_indicators

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Import OANDA client
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
    'NZD_USD': 1.5, 'AUD_USD': 1.3, 'USD_JPY': 1.0, 'EUR_GBP': 1.2,
    'EUR_JPY': 1.5, 'GBP_JPY': 2.0,
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


def backtest_scalping(df: pd.DataFrame, pair: str) -> dict:
    """
    Backtest scalping strategy with multi-indicator validation.
    """
    if len(df) < 200:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0, 'tp': 0, 'sl': 0}

    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    cooldown_minutes = settings.get('cooldown_minutes', config.COOLDOWN_MINUTES)

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    trades = []
    in_position = False
    entry_price = 0
    direction = None
    tp_price = 0
    sl_price = 0
    entry_time = None
    last_exit_time = None

    for i in range(200, len(df)):  # Need more data for indicators
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

            # GET SIGNAL using multi-indicator validation
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
print("BACKTEST: SCALPING STRATEGY - MULTI-INDICATOR VALIDATION")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nConfiguration:")
print(f"  - Strategy: scalping_strategy.py (multi-indicator)")
print(f"  - Config: scalping_config.py")
print(f"  - ONE position per symbol at a time")
print(f"  - Per-pair session filters (12:00-20:00 UTC)")
print(f"  - Cooldown: 30 minutes")
print(f"  - Data: OANDA 3-day M1 candles (~5000)")
print("=" * 80)

print("\nStrategy Requirements (ALL must pass):")
print("  1. Trend Alignment: EMA9 > EMA21 > EMA50")
print("  2. Momentum: RSI in 40-60 range")
print("  3. MACD Trigger: Crossover + histogram momentum")
print("  4. Candle: Strong body (>40% of range)")
print("  5. Volatility: ATR in acceptable range")
print()

# Print settings table
print(f"\n{'PAIR':<10} {'TP':<6} {'SL':<6} {'R:R':<8} {'CD':<5} {'SF':<4}")
print("-" * 50)
for pair in config.SCALPING_PAIRS:
    s = config.get_pair_settings(pair)
    session = config.TRADING_SESSIONS.get(pair, {})
    sf = 'ON' if session.get('session_filter', False) else 'OFF'
    cd = s.get('cooldown_minutes', 0)
    rr = f"1:{s['sl_pips']/s['tp_pips']:.1f}"
    print(f"{pair:<10} {s['tp_pips']:<6} {s['sl_pips']:<6} {rr:<8} {cd:<5} {sf:<4}")
print("-" * 50)

results = {}

for pair in config.SCALPING_PAIRS:
    settings = config.get_pair_settings(pair)
    session = config.TRADING_SESSIONS.get(pair, {})
    sf_status = 'ON' if session.get('session_filter', False) else 'OFF'
    cd = settings.get('cooldown_minutes', 0)

    print(f"\n{'=' * 60}")
    print(f"[{pair}] TP:{settings['tp_pips']}p SL:{settings['sl_pips']}p | CD:{cd}m SF:{sf_status}")
    print(f"{'=' * 60}")

    df = fetch_oanda_data(pair, count=5000)

    if len(df) == 0:
        print(f"  No data available")
        continue

    # Run backtest
    result = backtest_scalping(df, pair)
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
print("SUMMARY - SCALPING STRATEGY BACKTEST")
print("=" * 80)
print(f"\n{'PAIR':<10} {'TRADES':<10} {'WINS':<8} {'LOSSES':<8} {'WIN%':<8} {'PIPS':<12} {'TP':<6} {'SL'}")
print("-" * 75)

total_trades = 0
total_wins = 0
total_pips = 0
total_tp = 0
total_sl = 0

for pair in config.SCALPING_PAIRS:
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

# Profitable pairs
profitable_pairs = [p for p, r in results.items() if r['total_pips'] > 0]
losing_pairs = [p for p, r in results.items() if r['total_pips'] <= 0]

print(f"\n  Profitable Pairs ({len(profitable_pairs)}/{len(results)}):")
for pair in sorted(profitable_pairs, key=lambda p: results[p]['total_pips'], reverse=True):
    r = results[pair]
    print(f"    {pair}: {r['total_pips']:+,.0f} pips ({r['win_rate']:.1f}% WR, {r['trades']} trades)")

if losing_pairs:
    print(f"\n  Losing Pairs ({len(losing_pairs)}/{len(results)}):")
    for pair in sorted(losing_pairs, key=lambda p: results[p]['total_pips']):
        r = results[pair]
        print(f"    {pair}: {r['total_pips']:+,.0f} pips ({r['win_rate']:.1f}% WR, {r['trades']} trades)")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
print(f"\n  Total Result: {total_pips:+,.0f} pips over 3 days")
if total_pips > 0:
    print(f"  >>> PROFITABLE <<<")
else:
    print(f"  >>> LOSING <<<")

if total_trades > 0:
    print(f"\n  Win Rate: {total_wr:.1f}%")
    print(f"  Avg Pips/Trade: {total_pips / total_trades:+.2f}")
    print(f"  Daily Average: {total_pips / 3:+,.0f} pips/day")
    print(f"  Monthly Estimate: {total_pips / 3 * 20:+,.0f} pips/month (20 trading days)")
    print(f"\n  Trade Frequency: {total_trades / 3:.1f} trades/day")
else:
    print("\n  No trades generated - strategy too strict!")
    print("  Consider:")
    print("    - Adjusting indicator thresholds")
    print("    - Widening RSI zones")
    print("    - Relaxing trend requirements")

print("=" * 80)
