#!/usr/bin/env python3
"""
Quick 95%+ Win Rate Finder - Faster version
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
print("QUICK 95%+ WIN RATE FINDER")
print("=" * 100)
print(flush=True)


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
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(arr)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        if avg_loss == 0:
            r[i] = 100
        else:
            rs = avg_gain / avg_loss
            r[i] = 100 - (100 / (1 + rs))
    return r


def test(df, tp_pct, sl_pct, leverage):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema5 = ema(close, 5)
    ema13 = ema(close, 13)
    rsi7 = rsi(close, 7)

    balance = 10000
    position = None
    wins = 0
    losses = 0
    trades_list = []

    tp = tp_pct / 100
    sl = sl_pct / 100
    pos_size = 0.05
    maint = 0.005

    for i in range(50, len(df)):
        if balance <= 0:
            break

        if position:
            direction, entry, margin, tp_price, sl_price, liq, _ = position

            # LONG
            if direction == 1:
                if low[i] <= liq:
                    balance -= margin
                    losses += 1
                    trades_list.append(-margin)
                    position = None
                elif high[i] >= tp_price:
                    pnl = ((tp_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    wins += 1
                    trades_list.append(pnl)
                    position = None
                elif low[i] <= sl_price:
                    pnl = ((sl_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    losses += 1
                    trades_list.append(pnl)
                    position = None
            # SHORT
            else:
                if high[i] >= liq:
                    balance -= margin
                    losses += 1
                    trades_list.append(-margin)
                    position = None
                elif low[i] <= tp_price:
                    pnl = ((entry - tp_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    wins += 1
                    trades_list.append(pnl)
                    position = None
                elif high[i] >= sl_price:
                    pnl = ((entry - sl_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    losses += 1
                    trades_list.append(pnl)
                    position = None

        if position is None:
            signal = 0
            if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
                signal = 1
            elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
                signal = -1

            if signal != 0:
                margin = balance * pos_size
                entry = close[i]

                if signal == 1:
                    tp_price = entry * (1 + tp)
                    sl_price = entry * (1 - sl)
                    liq = entry * (1 - (1/leverage) + maint)
                else:
                    tp_price = entry * (1 - tp)
                    sl_price = entry * (1 + sl)
                    liq = entry * (1 + (1/leverage) - maint)

                position = (signal, entry, margin, tp_price, sl_price, liq, i)

    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0
    ret_pct = (balance - 10000) / 10000 * 100

    return {
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'return_pct': ret_pct,
        'balance': balance,
    }


# Test on ETHUSDT (best performer)
symbol = 'ETHUSDT'
file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
df = pd.read_csv(file_path)

# 7 days and 30 days
test_periods = [
    (10080, '7 DAYS'),
    (43200, '30 DAYS'),
]

# Configs to test - focus on high win rate combos
configs = [
    # Current
    (1.0, 3.0, 20),
    # Wider SL
    (1.0, 5.0, 20),
    (1.0, 7.0, 20),
    (1.0, 10.0, 20),
    (1.0, 15.0, 20),
    # Smaller TP + Wide SL
    (0.5, 5.0, 20),
    (0.5, 7.0, 20),
    (0.5, 10.0, 20),
    (0.5, 15.0, 20),
    # Very small TP
    (0.3, 5.0, 20),
    (0.3, 7.0, 20),
    (0.3, 10.0, 20),
    (0.3, 15.0, 20),
    # Lower leverage (safer)
    (1.0, 5.0, 10),
    (1.0, 10.0, 10),
    (0.5, 5.0, 10),
    (0.5, 10.0, 10),
]

results = []

for period, period_name in test_periods:
    test_df = df.tail(period + 100).reset_index(drop=True)
    print(f"\n{period_name} TEST ({len(test_df):,} bars)")
    print("-" * 80)
    print(f"{'TP%':>6} {'SL%':>6} {'Lev':>5} {'Trades':>8} {'Wins':>8} {'Win%':>8} {'Return':>12}")
    print("-" * 80)

    for tp, sl, lev in configs:
        r = test(test_df, tp, sl, lev)
        r['tp'] = tp
        r['sl'] = sl
        r['lev'] = lev
        r['period'] = period_name
        results.append(r)

        marker = "***" if r['win_rate'] >= 95 else ("**" if r['win_rate'] >= 90 else "")
        print(f"{tp:>6.1f} {sl:>6.1f} {lev:>5}x {r['trades']:>8} {r['wins']:>8} {r['win_rate']:>7.1f}% {r['return_pct']:>+11.1f}% {marker}")

    print("-" * 80)
    print(flush=True)

# Summary
print("\n" + "=" * 100)
print("RESULTS SUMMARY")
print("=" * 100)

# Best by win rate (95%+)
high_wr = [r for r in results if r['win_rate'] >= 95 and r['trades'] >= 5]
if high_wr:
    print(f"\n{'95%+ WIN RATE CONFIGS':^100}")
    print("-" * 100)
    for r in sorted(high_wr, key=lambda x: -x['win_rate']):
        print(f"{r['period']:<10} TP:{r['tp']}% SL:{r['sl']}% Lev:{r['lev']}x | Trades:{r['trades']} | Win:{r['win_rate']:.1f}% | Return:{r['return_pct']:+.1f}%")

# Best profitable
profitable = [r for r in results if r['return_pct'] > 0 and r['trades'] >= 5]
if profitable:
    print(f"\n{'PROFITABLE CONFIGS':^100}")
    print("-" * 100)
    for r in sorted(profitable, key=lambda x: -x['return_pct'])[:10]:
        print(f"{r['period']:<10} TP:{r['tp']}% SL:{r['sl']}% Lev:{r['lev']}x | Trades:{r['trades']} | Win:{r['win_rate']:.1f}% | Return:{r['return_pct']:+.1f}%")

# THE TRUTH about 95%+ win rate
print("\n" + "=" * 100)
print("THE MATH PROBLEM WITH 95%+ WIN RATE")
print("=" * 100)
print("""
To get 95%+ win rate, you need very WIDE stop loss and SMALL take profit.
But this creates a NEGATIVE expectancy:

Example with 95% win rate:
  - TP = 0.5%  (win $50 on $10k account with 10x leverage, 5% position)
  - SL = 10%   (lose $1000 on same position)

  Per 100 trades:
  - 95 wins x $50  = $4,750
  - 5 losses x $1000 = $5,000
  - NET = -$250 (LOSS!)

The ONLY way to be profitable with 95%+ win rate:
  1. Win rate must be VERY close to 100% (99%+)
  2. OR Stop loss must be smaller than 20x the take profit

For profitable trading, the BETTER approach is:
  - 75-85% win rate with 1:3 ratio (TP 1%, SL 3%) = breakeven to slight profit
  - 85%+ win rate with 1:5 ratio (TP 1%, SL 5%) = profitable if consistent
  - Focus on FILTERS to reduce bad trades rather than widening SL
""")

print("=" * 100)
