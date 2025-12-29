"""
BACKTEST: EXACT MATCH TO LIVE TRADING STRATEGY
===============================================
This backtest EXACTLY matches how run_optimized_forex_paper.py trades:
- Uses optimized_strategy.py for signals (get_signal function)
- Uses optimized_paper_config.py for all settings
- ONE position per symbol at a time
- Session filters (is_allowed_hour)
- Cooldown AFTER trade closes (configurable)
- Spread simulation
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

# Spread simulation (OANDA typical spreads)
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    print(f"  Fetching {count} candles from OANDA...")
    candles = client.get_candles(pair, 'M1', count=count)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    print(f"  Got {len(df)} candles ({df['time'].iloc[0]} to {df['time'].iloc[-1]})")
    return df


def backtest_exact_match(df: pd.DataFrame, pair: str, cooldown_minutes: int) -> dict:
    """
    Backtest that EXACTLY matches live trading behavior.

    - Uses get_signal() from optimized_strategy.py
    - ONE position at a time per symbol
    - Session filter via config.is_allowed_hour()
    - Cooldown applied AFTER trade closes
    """
    settings = config.get_pair_settings(pair)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult

    # Calculate all indicators ONCE
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
            # CHECK EXIT CONDITIONS (TP or SL)
            high = row['high']
            low = row['low']

            if direction == 'BUY':
                # Check SL first (worst case)
                if low <= sl_price:
                    trades.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pnl': -sl_pips,
                        'won': False,
                        'exit_reason': 'SL',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    in_position = False
                    last_exit_time = current_time
                # Check TP
                elif high >= tp_price:
                    trades.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pnl': tp_pips,
                        'won': True,
                        'exit_reason': 'TP',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    in_position = False
                    last_exit_time = current_time

            else:  # SELL
                # Check SL first
                if high >= sl_price:
                    trades.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pnl': -sl_pips,
                        'won': False,
                        'exit_reason': 'SL',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    in_position = False
                    last_exit_time = current_time
                # Check TP
                elif low <= tp_price:
                    trades.append({
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pnl': tp_pips,
                        'won': True,
                        'exit_reason': 'TP',
                        'entry_time': entry_time,
                        'exit_time': current_time
                    })
                    in_position = False
                    last_exit_time = current_time

        else:
            # NOT IN POSITION - CHECK FOR ENTRY

            # SESSION FILTER (exactly like live trading)
            if hasattr(config, 'is_allowed_hour'):
                if not config.is_allowed_hour(pair, hour_utc):
                    continue

            # COOLDOWN CHECK (after last EXIT, not entry)
            if cooldown_minutes > 0 and last_exit_time is not None:
                time_since = (current_time - last_exit_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue

            # GET SIGNAL using the EXACT same function as live trading
            # Pass all data up to current candle (like live trading does)
            signal, reason = get_signal(pair, df.iloc[:i+1], config)

            if signal is not None:
                # ENTER POSITION
                in_position = True
                direction = signal
                entry_time = current_time

                # Apply spread (like live trading)
                if signal == 'BUY':
                    entry_price = row['close'] + spread  # Buy at ask
                    tp_price = entry_price + tp_price_diff
                    sl_price = entry_price - sl_price_diff
                else:  # SELL
                    entry_price = row['close'] - spread  # Sell at bid
                    tp_price = entry_price - tp_price_diff
                    sl_price = entry_price + sl_price_diff

    # Calculate results
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pips': 0, 'tp': 0, 'sl': 0
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
        'recent_trades': trades[-5:] if trades else []
    }


# =============================================================================
# MAIN BACKTEST
# =============================================================================
print("=" * 80)
print("BACKTEST: EXACT MATCH TO LIVE TRADING STRATEGY")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nThis backtest EXACTLY matches run_optimized_forex_paper.py:")
print(f"  - Strategy: optimized_strategy.py (get_signal)")
print(f"  - Config: optimized_paper_config.py")
print(f"  - ONE position per symbol at a time")
print(f"  - Session filters enabled")
print(f"  - Cooldown: {config.COOLDOWN_MINUTES} minutes after trade closes")
print("=" * 80)

# Test with different cooldowns
cooldowns_to_test = [0, 15, 30, 60]
results = {cd: {} for cd in cooldowns_to_test}

for pair in config.OPTIMIZED_PAIRS:
    settings = config.get_pair_settings(pair)
    print(f"\n{'=' * 60}")
    print(f"[{pair}] Strategy: {settings['strategy']} | TP: {settings['tp_pips']}p | SL: {settings['sl_pips']}p")
    print(f"{'=' * 60}")

    df = fetch_oanda_data(pair, 5000)

    if len(df) < 100:
        print(f"  Insufficient data, skipping")
        continue

    for cd in cooldowns_to_test:
        result = backtest_exact_match(df, pair, cd)
        results[cd][pair] = result

    # Show comparison
    r0 = results[0][pair]
    r30 = results[30][pair]

    print(f"\n  NO COOLDOWN:   {r0['trades']:>3}t  {r0['win_rate']:>5.1f}% WR  {r0['total_pips']:>+6.0f}p  (TP:{r0['tp']} SL:{r0['sl']})")
    print(f"  30-MIN CD:     {r30['trades']:>3}t  {r30['win_rate']:>5.1f}% WR  {r30['total_pips']:>+6.0f}p  (TP:{r30['tp']} SL:{r30['sl']})")

    # Show recent trades
    if r30['recent_trades']:
        print(f"\n  Recent trades (30m CD):")
        for t in r30['recent_trades']:
            result_str = 'WIN' if t['won'] else 'LOSS'
            print(f"    {t['entry_time'].strftime('%m/%d %H:%M')} {t['direction']} -> {t['exit_reason']} {t['pnl']:+.0f}p [{result_str}]")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("COOLDOWN COMPARISON SUMMARY")
print("=" * 80)
print(f"\n{'PAIR':<10} {'0-MIN CD':<18} {'15-MIN CD':<18} {'30-MIN CD':<18} {'60-MIN CD':<18}")
print("-" * 85)

for pair in config.OPTIMIZED_PAIRS:
    row = f"{pair:<10} "
    for cd in cooldowns_to_test:
        r = results[cd].get(pair, {'trades': 0, 'win_rate': 0, 'total_pips': 0})
        row += f"{r['trades']:>3}t {r['win_rate']:>4.0f}% {r['total_pips']:>+5.0f}p  "
    print(row)

print("-" * 85)

# Totals per cooldown
print(f"\n{'TOTALS':<10}", end=" ")
for cd in cooldowns_to_test:
    total_trades = sum(r['trades'] for r in results[cd].values())
    total_wins = sum(r['wins'] for r in results[cd].values())
    total_pips = sum(r['total_pips'] for r in results[cd].values())
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"{total_trades:>3}t {wr:>4.0f}% {total_pips:>+5.0f}p  ", end="")
print()

# =============================================================================
# FINAL RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS - EXACT MATCH TO LIVE STRATEGY")
print("=" * 80)

for cd in cooldowns_to_test:
    total_trades = sum(r['trades'] for r in results[cd].values())
    total_wins = sum(r['wins'] for r in results[cd].values())
    total_pips = sum(r['total_pips'] for r in results[cd].values())
    total_tp = sum(r['tp'] for r in results[cd].values())
    total_sl = sum(r['sl'] for r in results[cd].values())
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    label = "NO COOLDOWN" if cd == 0 else f"{cd}-MIN COOLDOWN"
    print(f"\n  {label}:")
    print(f"    {total_trades} trades, {wr:.1f}% WR, {total_pips:+,.0f} pips")
    print(f"    Exit breakdown: {total_tp} TP, {total_sl} SL")

# Find best
best_cd = max(cooldowns_to_test, key=lambda cd: sum(r['total_pips'] for r in results[cd].values()))
best_pips = sum(r['total_pips'] for r in results[best_cd].values())

print(f"\n  >>> BEST: {'NO COOLDOWN' if best_cd == 0 else f'{best_cd}-MIN COOLDOWN'} with {best_pips:+,.0f} pips <<<")

# Current config
print(f"\n  CURRENT CONFIG: {config.COOLDOWN_MINUTES}-MIN COOLDOWN")
current_pips = sum(r['total_pips'] for r in results[config.COOLDOWN_MINUTES].values())
print(f"    Result: {current_pips:+,.0f} pips")

if best_cd != config.COOLDOWN_MINUTES:
    diff = best_pips - current_pips
    print(f"    Could gain {diff:+,.0f} pips by switching to {'NO COOLDOWN' if best_cd == 0 else f'{best_cd}-MIN COOLDOWN'}")

print("=" * 80)
