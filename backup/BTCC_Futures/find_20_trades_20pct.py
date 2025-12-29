#!/usr/bin/env python3
"""
Find Scalping Parameters for 20 Trades/Day @ 20% Daily Returns
================================================================
Target: 20 trades per day with 20% total daily profit

Math:
- 20% daily / 20 trades = 1% per trade average
- With 80% win rate: wins need to offset losses
- With 20x leverage: 0.05% price move = 1% profit

Testing various combinations to find optimal settings.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

TAKER_FEE = 0.00045

print("=" * 100)
print("SCALPING OPTIMIZER: 20 TRADES/DAY @ 20% DAILY RETURNS")
print("=" * 100)
print()


def ema(arr, period):
    e = np.zeros_like(arr)
    alpha = 2 / (period + 1)
    e[0] = arr[0]
    for i in range(1, len(arr)):
        e[i] = alpha * arr[i] + (1 - alpha) * e[i-1]
    return e


def rsi(arr, period):
    r = np.zeros_like(arr)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0
    avg_loss = np.mean(losses[:period]) if len(losses) >= period else 0
    for i in range(period, len(arr)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        if avg_loss == 0:
            r[i] = 100
        else:
            rs = avg_gain / avg_loss
            r[i] = 100 - (100 / (1 + rs))
    return r


def run_scalping_test(symbol, leverage, tp_pct, sl_pct, pos_size, ema_fast, ema_slow,
                      rsi_period, rsi_long_min, rsi_long_max, rsi_short_min, rsi_short_max,
                      duration=1440):  # 1 day = 1440 minutes
    """Run scalping backtest for 1 day"""

    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema_f = ema(close, ema_fast)
    ema_s = ema(close, ema_slow)
    rsi_val = rsi(close, rsi_period)

    TP_PCT = tp_pct / 100
    SL_PCT = sl_pct / 100
    MAINT_MARGIN = 0.005

    balance = 100.0
    initial = balance
    max_balance = balance
    max_dd = 0

    position = None
    trades = []

    for i in range(50, min(duration, len(df))):
        if balance > max_balance:
            max_balance = balance
        dd = (max_balance - balance) / max_balance * 100
        if dd > max_dd:
            max_dd = dd

        if balance <= 0:
            break

        if position:
            direction, entry, margin, tp_price, sl_price, liq_price = position

            # Liquidation check
            if direction == 1 and low[i] <= liq_price:
                balance -= margin
                trades.append({'pnl': -margin, 'result': 'LIQ'})
                position = None
                continue
            elif direction == -1 and high[i] >= liq_price:
                balance -= margin
                trades.append({'pnl': -margin, 'result': 'LIQ'})
                position = None
                continue

            # TP/SL check
            if direction == 1:
                if high[i] >= tp_price:
                    pnl = ((tp_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP'})
                    position = None
                elif low[i] <= sl_price:
                    pnl = ((sl_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL'})
                    position = None
            else:
                if low[i] <= tp_price:
                    pnl = ((entry - tp_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP'})
                    position = None
                elif high[i] >= sl_price:
                    pnl = ((entry - sl_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL'})
                    position = None

        # Open new position
        if position is None:
            signal = 0
            if ema_f[i] > ema_s[i] and rsi_long_min < rsi_val[i] < rsi_long_max:
                signal = 1
            elif ema_f[i] < ema_s[i] and rsi_short_min < rsi_val[i] < rsi_short_max:
                signal = -1

            if signal != 0:
                margin = balance * pos_size
                entry = close[i]

                if signal == 1:
                    tp_price = entry * (1 + TP_PCT)
                    sl_price = entry * (1 - SL_PCT)
                    liq_price = entry * (1 - (1/leverage) + MAINT_MARGIN)
                else:
                    tp_price = entry * (1 - TP_PCT)
                    sl_price = entry * (1 + SL_PCT)
                    liq_price = entry * (1 + (1/leverage) - MAINT_MARGIN)

                position = (signal, entry, margin, tp_price, sl_price, liq_price)

    # Close remaining position
    if position:
        direction, entry, margin, tp_price, sl_price, liq_price = position
        exit_price = close[-1]
        if direction == 1:
            pnl = ((exit_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
        else:
            pnl = ((entry - exit_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
        balance += pnl
        trades.append({'pnl': pnl, 'result': 'END'})

    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len([t for t in trades if t['pnl'] <= 0])
    liqs = len([t for t in trades if t['result'] == 'LIQ'])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    return_pct = (balance - initial) / initial * 100

    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'liqs': liqs,
        'win_rate': win_rate,
        'return_pct': return_pct,
        'max_dd': max_dd,
        'final_balance': balance,
    }


# Test configurations targeting 20 trades/day
print("Testing scalping configurations for XRPUSDT (1 day)...")
print()

configs = [
    # (leverage, tp%, sl%, pos_size, ema_fast, ema_slow, rsi_period, rsi_long_min, rsi_long_max, rsi_short_min, rsi_short_max)
    # Aggressive scalping - small TP, tight SL
    (50, 0.3, 0.5, 0.15, 3, 8, 5, 45, 75, 25, 55),   # Very fast EMA, relaxed RSI
    (50, 0.4, 0.6, 0.15, 3, 8, 5, 45, 75, 25, 55),
    (50, 0.5, 0.8, 0.15, 3, 8, 5, 45, 75, 25, 55),
    (40, 0.4, 0.6, 0.15, 3, 8, 5, 45, 75, 25, 55),
    (40, 0.5, 0.8, 0.15, 3, 8, 5, 45, 75, 25, 55),
    (40, 0.5, 1.0, 0.15, 3, 8, 5, 45, 75, 25, 55),
    # Medium scalping
    (30, 0.5, 1.0, 0.15, 3, 8, 5, 45, 75, 25, 55),
    (30, 0.6, 1.2, 0.15, 3, 8, 5, 45, 75, 25, 55),
    (30, 0.7, 1.5, 0.15, 3, 8, 5, 45, 75, 25, 55),
    # Wider RSI range for more signals
    (30, 0.5, 1.0, 0.15, 3, 8, 5, 40, 80, 20, 60),
    (30, 0.5, 1.0, 0.15, 3, 8, 5, 35, 85, 15, 65),
    (30, 0.5, 1.0, 0.20, 3, 8, 5, 35, 85, 15, 65),
    # Higher leverage for more profit per trade
    (50, 0.5, 1.5, 0.10, 3, 8, 5, 40, 80, 20, 60),
    (50, 0.5, 2.0, 0.10, 3, 8, 5, 40, 80, 20, 60),
    (50, 0.5, 2.5, 0.10, 3, 8, 5, 40, 80, 20, 60),
    # Very fast indicators
    (40, 0.4, 0.8, 0.15, 2, 5, 3, 40, 80, 20, 60),
    (40, 0.5, 1.0, 0.15, 2, 5, 3, 40, 80, 20, 60),
    (50, 0.4, 1.0, 0.15, 2, 5, 3, 40, 80, 20, 60),
]

results = []
for cfg in configs:
    lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax = cfg
    r = run_scalping_test('XRPUSDT', lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax)
    if r:
        r['config'] = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}% EMA:{ef}/{es} RSI:{rp}"
        r['leverage'] = lev
        r['tp'] = tp
        r['sl'] = sl
        r['pos_size'] = ps
        results.append(r)

# Sort by return with at least 15 trades
results_filtered = [r for r in results if r['trades'] >= 15 and r['liqs'] == 0]
results_filtered.sort(key=lambda x: x['return_pct'], reverse=True)

print("=" * 120)
print("TOP SCALPING CONFIGURATIONS (Min 15 trades, 0 liquidations)")
print("=" * 120)
print(f"{'Config':<50} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Return':>10} {'MaxDD':>8} {'Liqs':>6}")
print("-" * 120)

for r in results_filtered[:10]:
    print(f"{r['config']:<50} {r['trades']:>8} {r['wins']:>6} {r['win_rate']:>7.1f}% {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}% {r['liqs']:>6}")

# Now test for 7 days with best configs
print()
print("=" * 120)
print("7-DAY BACKTEST OF TOP CONFIGS")
print("=" * 120)

if results_filtered:
    best = results_filtered[0]
    print(f"\nBest 1-day config: {best['config']}")
    print(f"1-day: {best['trades']} trades, {best['win_rate']:.1f}% WR, {best['return_pct']:+.1f}% return")

    # Test for 7 days
    r7 = run_scalping_test('XRPUSDT', best['leverage'], best['tp'], best['sl'], best['pos_size'],
                           3, 8, 5, 40, 80, 20, 60, duration=10080)
    if r7:
        print(f"\n7-day results:")
        print(f"  Trades: {r7['trades']} ({r7['trades']/7:.1f}/day)")
        print(f"  Win Rate: {r7['win_rate']:.1f}%")
        print(f"  Return: {r7['return_pct']:+.1f}%")
        print(f"  Daily Avg: {r7['return_pct']/7:+.1f}%")
        print(f"  Max DD: {r7['max_dd']:.1f}%")
        print(f"  Liquidations: {r7['liqs']}")

# Test more aggressive for 20% daily
print()
print("=" * 120)
print("TARGETING 20% DAILY - AGGRESSIVE CONFIGS")
print("=" * 120)

aggressive_configs = [
    # Higher position size + leverage
    (50, 0.5, 2.0, 0.20, 2, 5, 3, 35, 85, 15, 65),
    (50, 0.5, 2.5, 0.25, 2, 5, 3, 35, 85, 15, 65),
    (50, 0.6, 3.0, 0.20, 2, 5, 3, 35, 85, 15, 65),
    (75, 0.4, 2.0, 0.15, 2, 5, 3, 35, 85, 15, 65),
    (75, 0.5, 2.5, 0.15, 2, 5, 3, 35, 85, 15, 65),
    (100, 0.3, 1.5, 0.10, 2, 5, 3, 35, 85, 15, 65),
    (100, 0.4, 2.0, 0.10, 2, 5, 3, 35, 85, 15, 65),
]

agg_results = []
for cfg in aggressive_configs:
    lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax = cfg
    r = run_scalping_test('XRPUSDT', lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax)
    if r:
        r['config'] = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}%"
        r['leverage'] = lev
        r['tp'] = tp
        r['sl'] = sl
        r['pos_size'] = ps
        agg_results.append(r)

agg_results.sort(key=lambda x: x['return_pct'], reverse=True)

print(f"{'Config':<35} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'Return':>10} {'MaxDD':>8} {'Liqs':>6}")
print("-" * 100)

for r in agg_results:
    print(f"{r['config']:<35} {r['trades']:>8} {r['wins']:>6} {r['win_rate']:>7.1f}% {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}% {r['liqs']:>6}")

print()
print("=" * 120)
