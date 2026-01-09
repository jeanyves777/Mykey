#!/usr/bin/env python3
"""
Realistic 20% Daily Strategy Finder
=====================================
Finding the maximum SAFE daily return with proper risk management
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

TAKER_FEE = 0.00045

print("=" * 120)
print("REALISTIC 20% DAILY STRATEGY FINDER")
print("=" * 120)


def ema(arr, period):
    e = np.zeros_like(arr)
    alpha = 2 / (period + 1)
    e[0] = arr[0]
    for i in range(1, len(arr)):
        e[i] = alpha * arr[i] + (1 - alpha) * e[i-1]
    return e


def rsi(arr, period):
    r = np.zeros_like(arr)
    if len(arr) < period + 1:
        return r
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


def run_backtest(symbol, leverage, tp_pct, sl_pct, pos_size,
                 ema_fast, ema_slow, rsi_period,
                 rsi_long_min, rsi_long_max, rsi_short_min, rsi_short_max,
                 duration=10080):  # 7 days

    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)
    if len(df) < 200:
        return None

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

            # Liquidation
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

            # TP/SL
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

    if position:
        direction, entry, margin, tp_price, sl_price, liq_price = position
        exit_price = close[-1]
        if direction == 1:
            pnl = ((exit_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
        else:
            pnl = ((entry - exit_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
        balance += pnl
        trades.append({'pnl': pnl, 'result': 'END'})

    total = len(trades)
    if total == 0:
        return None

    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len([t for t in trades if t['pnl'] <= 0])
    liqs = len([t for t in trades if t['result'] == 'LIQ'])
    win_rate = wins / total * 100 if total > 0 else 0
    return_pct = (balance - initial) / initial * 100
    daily_avg = return_pct / 7

    return {
        'trades': total,
        'trades_day': total / 7,
        'wins': wins,
        'losses': losses,
        'liqs': liqs,
        'win_rate': win_rate,
        'return_pct': return_pct,
        'daily_avg': daily_avg,
        'max_dd': max_dd,
        'final': balance,
    }


# THE WORKING 95% CONFIG as baseline
baseline = run_backtest('XRPUSDT', 20, 1.0, 10.0, 0.10, 5, 13, 7, 50, 70, 30, 50)
if baseline:
    print()
    print("BASELINE (95% WR Strategy):")
    print(f"  20x | TP:1% | SL:10% | PS:10%")
    print(f"  Trades: {baseline['trades']} ({baseline['trades_day']:.1f}/day)")
    print(f"  Win Rate: {baseline['win_rate']:.1f}%")
    print(f"  Return: {baseline['return_pct']:+.1f}%")
    print(f"  Daily Avg: {baseline['daily_avg']:+.1f}%")
    print(f"  Liquidations: {baseline['liqs']}")
    print()

# To get 20% daily we need to scale up
# Current: 4.8% daily with 10% position
# Need: 20% daily = 4.2x more
# Option 1: 40% position size (but 1 loss = 40% account hit)
# Option 2: More trades per day

print("=" * 120)
print("SCALING UP THE 95% STRATEGY")
print("=" * 120)

# Test with higher position sizes but keeping the same safe SL
scale_configs = [
    # (leverage, tp%, sl%, pos_size%)
    (20, 1.0, 10.0, 0.15),  # 15%
    (20, 1.0, 10.0, 0.20),  # 20%
    (20, 1.0, 10.0, 0.25),  # 25%
    (20, 1.0, 10.0, 0.30),  # 30%
    (20, 1.0, 10.0, 0.35),  # 35%
    (20, 1.0, 10.0, 0.40),  # 40%
    (20, 1.0, 10.0, 0.50),  # 50%
]

print()
print(f"{'Config':<35} {'Trades':>7} {'T/Day':>6} {'WR%':>7} {'Return':>10} {'Daily':>8} {'MaxDD':>7} {'Liqs':>5}")
print("-" * 100)

for lev, tp, sl, ps in scale_configs:
    r = run_backtest('XRPUSDT', lev, tp, sl, ps, 5, 13, 7, 50, 70, 30, 50)
    if r:
        config = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}%"
        print(f"{config:<35} {r['trades']:>7} {r['trades_day']:>6.1f} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}% {r['liqs']:>5}")


# Now test with faster indicators for more trades
print()
print("=" * 120)
print("FASTER INDICATORS FOR MORE TRADES")
print("=" * 120)

fast_configs = [
    # Fast EMA with wide RSI for more signals
    (20, 1.0, 10.0, 0.20, 3, 8, 5, 45, 75, 25, 55),
    (20, 1.0, 10.0, 0.25, 3, 8, 5, 45, 75, 25, 55),
    (20, 1.0, 10.0, 0.30, 3, 8, 5, 45, 75, 25, 55),
    (20, 1.0, 10.0, 0.20, 2, 5, 3, 45, 75, 25, 55),
    (20, 1.0, 10.0, 0.25, 2, 5, 3, 45, 75, 25, 55),
    (20, 1.0, 10.0, 0.30, 2, 5, 3, 45, 75, 25, 55),
    # Wider RSI range
    (20, 1.0, 10.0, 0.25, 3, 8, 5, 40, 80, 20, 60),
    (20, 1.0, 10.0, 0.30, 3, 8, 5, 40, 80, 20, 60),
    (20, 1.0, 10.0, 0.30, 2, 5, 3, 40, 80, 20, 60),
]

print()
print(f"{'Config':<35} {'EMA':>6} {'Trades':>7} {'T/Day':>6} {'WR%':>7} {'Return':>10} {'Daily':>8} {'MaxDD':>7} {'Liqs':>5}")
print("-" * 120)

for lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax in fast_configs:
    r = run_backtest('XRPUSDT', lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax)
    if r:
        config = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}%"
        ema_str = f"{ef}/{es}"
        print(f"{config:<35} {ema_str:>6} {r['trades']:>7} {r['trades_day']:>6.1f} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}% {r['liqs']:>5}")


# Find THE BEST combination
print()
print("=" * 120)
print("COMPREHENSIVE SEARCH FOR 20%+ DAILY")
print("=" * 120)

all_results = []

leverages = [20]  # Safe leverage only
tp_pcts = [1.0, 1.5, 2.0]
sl_pcts = [10.0, 8.0, 6.0]
pos_sizes = [0.25, 0.30, 0.35, 0.40, 0.50]
ema_pairs = [(3, 8), (2, 5), (5, 13)]
rsi_ranges = [(45, 75, 25, 55), (40, 80, 20, 60), (50, 70, 30, 50)]

for lev in leverages:
    for tp in tp_pcts:
        for sl in sl_pcts:
            for ps in pos_sizes:
                for ef, es in ema_pairs:
                    for rlmin, rlmax, rsmin, rsmax in rsi_ranges:
                        r = run_backtest('XRPUSDT', lev, tp, sl, ps, ef, es, 5, rlmin, rlmax, rsmin, rsmax)
                        if r and r['liqs'] == 0 and r['return_pct'] > 0:
                            r['config'] = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}%"
                            r['ema'] = f"{ef}/{es}"
                            r['rsi'] = f"{rlmin}-{rlmax}/{rsmin}-{rsmax}"
                            all_results.append(r)

# Sort by daily avg return
all_results.sort(key=lambda x: x['daily_avg'], reverse=True)

print()
print(f"Found {len(all_results)} profitable configs with 0 liquidations")
print()
print(f"{'Config':<30} {'EMA':>6} {'Trades':>7} {'T/Day':>6} {'WR%':>7} {'Return':>10} {'Daily':>8} {'MaxDD':>7}")
print("-" * 110)

for r in all_results[:20]:
    print(f"{r['config']:<30} {r['ema']:>6} {r['trades']:>7} {r['trades_day']:>6.1f} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}%")

if all_results:
    best = all_results[0]
    print()
    print("=" * 120)
    print("RECOMMENDED STRATEGY FOR MAXIMUM SAFE RETURNS")
    print("=" * 120)
    print(f"Config: {best['config']}")
    print(f"EMA: {best['ema']}")
    print(f"RSI: {best['rsi']}")
    print()
    print(f"Trades: {best['trades']} ({best['trades_day']:.1f}/day)")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"7-Day Return: {best['return_pct']:+.1f}%")
    print(f"Daily Average: {best['daily_avg']:+.1f}%")
    print(f"Max Drawdown: {best['max_dd']:.1f}%")
    print(f"Liquidations: {best['liqs']}")
    print("=" * 120)
