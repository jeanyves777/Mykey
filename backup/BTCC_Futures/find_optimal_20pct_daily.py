#!/usr/bin/env python3
"""
Find OPTIMAL Settings for 20 Trades/Day @ 20% Daily
=====================================================
Key insight: We need SAFE leverage that won't liquidate

Math:
- 20x leverage = liquidation at 5% move
- Need SL < 5% to avoid liquidation
- TP small enough to hit often
- More signals = more opportunities
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
print("FINDING OPTIMAL 20 TRADES/DAY @ 20% DAILY - ZERO LIQUIDATIONS")
print("=" * 120)
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


def run_backtest(symbol, leverage, tp_pct, sl_pct, pos_size,
                 ema_fast, ema_slow, rsi_period,
                 rsi_long_min, rsi_long_max, rsi_short_min, rsi_short_max,
                 duration=10080):  # 7 days
    """Run 7-day backtest"""

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


# Key insight:
# - 20x = 5% move to liquidate, so SL must be < 4.5% to be safe
# - Higher leverage = more profit per trade but tighter liquidation
# - 10x = 10% move to liquidate (very safe)

# Configs: (leverage, tp%, sl%, pos_size%, ema_fast, ema_slow, rsi_period, rsi_long_min, rsi_long_max, rsi_short_min, rsi_short_max)
configs = [
    # Safe 20x with wide SL (proven 95% WR)
    (20, 1.0, 10.0, 0.10, 5, 13, 7, 50, 70, 30, 50),  # Original 95% WR

    # More aggressive but safe - wider RSI for more signals
    (20, 1.0, 8.0, 0.15, 3, 8, 5, 40, 75, 25, 60),
    (20, 1.0, 8.0, 0.20, 3, 8, 5, 35, 80, 20, 65),
    (20, 0.8, 6.0, 0.15, 3, 8, 5, 40, 75, 25, 60),
    (20, 0.8, 6.0, 0.20, 3, 8, 5, 35, 80, 20, 65),

    # Higher position for more profit
    (20, 1.0, 10.0, 0.25, 3, 8, 5, 40, 75, 25, 60),
    (20, 1.0, 10.0, 0.30, 3, 8, 5, 40, 75, 25, 60),

    # Fast EMAs for more signals
    (20, 0.8, 8.0, 0.20, 2, 5, 3, 40, 80, 20, 60),
    (20, 1.0, 8.0, 0.20, 2, 5, 3, 40, 80, 20, 60),
    (20, 1.0, 10.0, 0.20, 2, 5, 3, 40, 80, 20, 60),

    # Higher TP for more profit per trade
    (20, 1.5, 10.0, 0.15, 3, 8, 5, 45, 70, 30, 55),
    (20, 2.0, 10.0, 0.15, 3, 8, 5, 45, 70, 30, 55),

    # 25x leverage (4% liq) with 3.5% SL
    (25, 1.0, 3.5, 0.15, 3, 8, 5, 40, 75, 25, 60),
    (25, 0.8, 3.0, 0.15, 3, 8, 5, 40, 75, 25, 60),
    (25, 1.0, 3.5, 0.20, 2, 5, 3, 40, 80, 20, 60),

    # 30x leverage (3.3% liq) with 3% SL
    (30, 0.8, 2.8, 0.15, 3, 8, 5, 40, 75, 25, 60),
    (30, 1.0, 3.0, 0.15, 3, 8, 5, 40, 75, 25, 60),
    (30, 1.0, 3.0, 0.20, 2, 5, 3, 40, 80, 20, 60),

    # Very wide RSI for max signals
    (20, 1.0, 10.0, 0.20, 2, 5, 3, 30, 90, 10, 70),
    (25, 1.0, 3.5, 0.20, 2, 5, 3, 30, 90, 10, 70),
]

print("Testing configurations on XRPUSDT (7 days)...")
print()

results = []
for cfg in configs:
    lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax = cfg
    r = run_backtest('XRPUSDT', lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax)
    if r:
        r['config'] = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}%"
        r['leverage'] = lev
        r['tp'] = tp
        r['sl'] = sl
        r['pos_size'] = ps
        r['ema'] = f"{ef}/{es}"
        results.append(r)

# Filter: 0 liquidations, min 10 trades/day, positive return
good_results = [r for r in results if r['liqs'] == 0 and r['trades_day'] >= 10 and r['return_pct'] > 0]
good_results.sort(key=lambda x: x['daily_avg'], reverse=True)

print("=" * 130)
print("TOP RESULTS: 0 Liquidations, 10+ Trades/Day, Positive Return")
print("=" * 130)
print(f"{'Config':<30} {'EMA':>6} {'Trades':>7} {'T/Day':>6} {'Wins':>5} {'WR%':>7} {'Return':>10} {'Daily':>8} {'MaxDD':>7}")
print("-" * 130)

for r in good_results[:15]:
    print(f"{r['config']:<30} {r['ema']:>6} {r['trades']:>7} {r['trades_day']:>6.1f} {r['wins']:>5} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}%")

# Now filter for 20%+ daily average
daily_20_results = [r for r in results if r['liqs'] == 0 and r['daily_avg'] >= 20]
daily_20_results.sort(key=lambda x: x['daily_avg'], reverse=True)

print()
print("=" * 130)
print("RESULTS WITH 20%+ DAILY AVG (0 Liquidations)")
print("=" * 130)

if daily_20_results:
    for r in daily_20_results:
        print(f"{r['config']:<30} {r['ema']:>6} {r['trades']:>7} {r['trades_day']:>6.1f} {r['wins']:>5} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}%")
else:
    print("No configs achieved 20%+ daily with 0 liquidations.")
    print("\nBest performers:")
    best = sorted([r for r in results if r['liqs'] == 0], key=lambda x: x['daily_avg'], reverse=True)[:5]
    for r in best:
        print(f"{r['config']:<30} {r['ema']:>6} {r['trades']:>7} {r['trades_day']:>6.1f} {r['wins']:>5} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}%")

# Show the math required for 20% daily
print()
print("=" * 130)
print("MATH FOR 20% DAILY")
print("=" * 130)
print("""
To achieve 20% daily return:
- Option A: 20 trades x 1% profit each = 20%
- Option B: 10 trades x 2% profit each = 20%
- Option C: 5 trades x 4% profit each = 20%

With 20x leverage:
- 1% price move = 20% account move (with 100% position)
- 0.5% price move = 10% account move
- For 1% profit per trade with 10% position: need 0.5% TP hit

Current best (95% WR strategy):
- 1% TP at 20x = 20% profit on position
- 10% position = 2% profit per trade
- ~3 trades/day = 6% daily

To scale to 20% daily without liquidations:
- Need 10 trades/day at 2% each, OR
- Need to increase position size (more risk)
""")

print()
print("=" * 130)
print("RECOMMENDED CONFIG FOR AGGRESSIVE SCALPING")
print("=" * 130)

# Find best balance
best_for_20 = max([r for r in results if r['liqs'] == 0], key=lambda x: x['daily_avg'])
print(f"\nBest safe config: {best_for_20['config']}")
print(f"  Trades/Day: {best_for_20['trades_day']:.1f}")
print(f"  Win Rate: {best_for_20['win_rate']:.1f}%")
print(f"  Daily Avg: {best_for_20['daily_avg']:+.1f}%")
print(f"  7-Day Return: {best_for_20['return_pct']:+.1f}%")
print(f"  Max DD: {best_for_20['max_dd']:.1f}%")
print(f"  Liquidations: {best_for_20['liqs']}")

# To hit 20% we need higher position size
print()
print("To reach 20% daily, we need to INCREASE position size:")
print("Testing higher position sizes...")

high_pos_configs = [
    (20, 1.0, 10.0, 0.30, 2, 5, 3, 40, 80, 20, 60),
    (20, 1.0, 10.0, 0.40, 2, 5, 3, 40, 80, 20, 60),
    (20, 1.0, 10.0, 0.50, 2, 5, 3, 40, 80, 20, 60),
    (20, 1.0, 8.0, 0.40, 2, 5, 3, 40, 80, 20, 60),
    (20, 1.0, 8.0, 0.50, 2, 5, 3, 40, 80, 20, 60),
    (25, 1.0, 3.5, 0.40, 2, 5, 3, 40, 80, 20, 60),
    (25, 1.0, 3.5, 0.50, 2, 5, 3, 40, 80, 20, 60),
]

high_pos_results = []
for cfg in high_pos_configs:
    lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax = cfg
    r = run_backtest('XRPUSDT', lev, tp, sl, ps, ef, es, rp, rlmin, rlmax, rsmin, rsmax)
    if r:
        r['config'] = f"{lev}x TP:{tp}% SL:{sl}% PS:{ps*100:.0f}%"
        high_pos_results.append(r)

print()
print(f"{'Config':<30} {'Trades':>7} {'T/Day':>6} {'WR%':>7} {'Return':>10} {'Daily':>8} {'MaxDD':>7} {'Liqs':>5}")
print("-" * 100)
for r in sorted(high_pos_results, key=lambda x: x['daily_avg'], reverse=True):
    print(f"{r['config']:<30} {r['trades']:>7} {r['trades_day']:>6.1f} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['daily_avg']:>+7.1f}% {r['max_dd']:>6.1f}% {r['liqs']:>5}")

print("=" * 130)
