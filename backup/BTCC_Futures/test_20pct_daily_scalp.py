#!/usr/bin/env python3
"""
Test 20% Daily Scalping Strategy
=================================
Testing the best config: 50x TP:0.6% SL:3.0% PS:20%
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
print("20% DAILY SCALPING TEST - 50x TP:0.6% SL:3.0% PS:20%")
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


# SETTINGS
LEVERAGE = 50
TP_PCT = 0.6 / 100
SL_PCT = 3.0 / 100
POS_SIZE = 0.20
EMA_FAST = 2
EMA_SLOW = 5
RSI_PERIOD = 3
RSI_LONG_MIN = 35
RSI_LONG_MAX = 85
RSI_SHORT_MIN = 15
RSI_SHORT_MAX = 65
MAINT_MARGIN = 0.005

# Load data
symbol = 'XRPUSDT'
duration = 10080  # 7 days

file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)

close = df['close'].values
high = df['high'].values
low = df['low'].values

ema_f = ema(close, EMA_FAST)
ema_s = ema(close, EMA_SLOW)
rsi_val = rsi(close, RSI_PERIOD)

balance = 100.0
initial = balance
max_balance = balance
max_dd = 0

position = None
trades = []
daily_returns = []
day_start_balance = balance
current_day = 0

print(f"Config: {LEVERAGE}x | TP:{TP_PCT*100:.1f}% | SL:{SL_PCT*100:.1f}% | Position:{POS_SIZE*100:.0f}%")
print(f"Indicators: EMA {EMA_FAST}/{EMA_SLOW} | RSI {RSI_PERIOD}")
print(f"Starting Balance: ${balance:.2f}")
print()
print("-" * 100)
print(f"{'Bar':>6} {'Day':>4} {'Action':<8} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Balance':>12} {'Result':<10}")
print("-" * 100)

for i in range(50, min(duration, len(df))):
    # Track daily returns
    day = (i - 50) // 1440
    if day > current_day:
        daily_ret = (balance - day_start_balance) / day_start_balance * 100
        daily_returns.append(daily_ret)
        day_start_balance = balance
        current_day = day

    if balance > max_balance:
        max_balance = balance
    dd = (max_balance - balance) / max_balance * 100
    if dd > max_dd:
        max_dd = dd

    if balance <= 0:
        print(f"  *** ACCOUNT BLOWN ***")
        break

    if position:
        direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position
        dir_str = "LONG" if direction == 1 else "SHORT"

        # Liquidation check
        if direction == 1 and low[i] <= liq_price:
            pnl = -margin
            balance += pnl
            trades.append({'pnl': pnl, 'result': 'LIQ', 'day': day})
            print(f"{i:>6} {day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {liq_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'LIQ':<10}")
            position = None
            continue
        elif direction == -1 and high[i] >= liq_price:
            pnl = -margin
            balance += pnl
            trades.append({'pnl': pnl, 'result': 'LIQ', 'day': day})
            print(f"{i:>6} {day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {liq_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'LIQ':<10}")
            position = None
            continue

        # TP/SL check
        if direction == 1:
            if high[i] >= tp_price:
                pnl = ((tp_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                trades.append({'pnl': pnl, 'result': 'TP', 'day': day})
                print(f"{i:>6} {day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {tp_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'TP WIN':<10}")
                position = None
            elif low[i] <= sl_price:
                pnl = ((sl_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                trades.append({'pnl': pnl, 'result': 'SL', 'day': day})
                print(f"{i:>6} {day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {sl_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'SL LOSS':<10}")
                position = None
        else:
            if low[i] <= tp_price:
                pnl = ((entry - tp_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                trades.append({'pnl': pnl, 'result': 'TP', 'day': day})
                print(f"{i:>6} {day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {tp_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'TP WIN':<10}")
                position = None
            elif high[i] >= sl_price:
                pnl = ((entry - sl_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                trades.append({'pnl': pnl, 'result': 'SL', 'day': day})
                print(f"{i:>6} {day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {sl_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'SL LOSS':<10}")
                position = None

    # Open new position
    if position is None:
        signal = 0
        if ema_f[i] > ema_s[i] and RSI_LONG_MIN < rsi_val[i] < RSI_LONG_MAX:
            signal = 1
        elif ema_f[i] < ema_s[i] and RSI_SHORT_MIN < rsi_val[i] < RSI_SHORT_MAX:
            signal = -1

        if signal != 0:
            margin = balance * POS_SIZE
            entry = close[i]
            dir_str = "LONG" if signal == 1 else "SHORT"

            if signal == 1:
                tp_price = entry * (1 + TP_PCT)
                sl_price = entry * (1 - SL_PCT)
                liq_price = entry * (1 - (1/LEVERAGE) + MAINT_MARGIN)
            else:
                tp_price = entry * (1 - TP_PCT)
                sl_price = entry * (1 + SL_PCT)
                liq_price = entry * (1 + (1/LEVERAGE) - MAINT_MARGIN)

            position = (signal, entry, margin, tp_price, sl_price, liq_price, i)
            print(f"{i:>6} {day:>4} {'OPEN':<8} {dir_str:<6} {entry:>10.4f} {'---':>10} {'---':>10} ${balance:>11.2f} TP:{tp_price:.4f}")

# Close remaining
if position:
    direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position
    exit_price = close[-1]
    dir_str = "LONG" if direction == 1 else "SHORT"
    if direction == 1:
        pnl = ((exit_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
    else:
        pnl = ((entry - exit_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
    balance += pnl
    trades.append({'pnl': pnl, 'result': 'END', 'day': current_day})
    print(f"{len(df)-1:>6} {current_day:>4} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {exit_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {'END':<10}")

# Final daily return
daily_ret = (balance - day_start_balance) / day_start_balance * 100 if day_start_balance > 0 else 0
daily_returns.append(daily_ret)

# Stats
wins = len([t for t in trades if t['pnl'] > 0])
losses = len([t for t in trades if t['pnl'] <= 0])
liqs = len([t for t in trades if t['result'] == 'LIQ'])
total = wins + losses
win_rate = wins / total * 100 if total > 0 else 0
return_pct = (balance - initial) / initial * 100

print()
print("=" * 100)
print("FINAL RESULTS - 20% DAILY SCALPING STRATEGY")
print("=" * 100)
print(f"Settings:        {LEVERAGE}x | TP:{TP_PCT*100:.1f}% | SL:{SL_PCT*100:.1f}% | Position:{POS_SIZE*100:.0f}%")
print(f"Starting:        ${initial:.2f}")
print(f"Final:           ${balance:.2f}")
print(f"Profit/Loss:     ${balance - initial:+.2f}")
print(f"Total Return:    {return_pct:+.1f}%")
print(f"Max Drawdown:    {max_dd:.1f}%")
print()
print(f"Total Trades:    {total}")
print(f"Trades/Day:      {total/7:.1f}")
print(f"Wins:            {wins}")
print(f"Losses:          {losses}")
print(f"Liquidations:    {liqs}")
print(f"Win Rate:        {win_rate:.1f}%")

if wins > 0:
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0])
    print(f"Avg Win:         ${avg_win:.2f}")
if losses > 0:
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0])
    print(f"Avg Loss:        ${avg_loss:.2f}")

print()
print("=" * 100)
print("DAILY BREAKDOWN")
print("=" * 100)
for i, ret in enumerate(daily_returns):
    day_trades = len([t for t in trades if t.get('day', 0) == i])
    print(f"Day {i+1}: {ret:+.1f}% ({day_trades} trades)")

avg_daily = np.mean(daily_returns) if daily_returns else 0
print(f"\nAverage Daily Return: {avg_daily:+.1f}%")
print("=" * 100)
