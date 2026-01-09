#!/usr/bin/env python3
"""
$100 Capital XRPUSDT 30x Leverage - 7 Day Backtest
===================================================
Balanced leverage - higher returns but safer than 50x
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
print("$100 CAPITAL - XRPUSDT @ 30x LEVERAGE - 7 DAY BACKTEST")
print("=" * 100)
print()
print("Config: TP 1.0% | SL 10% | Leverage 30x | Position Size 10%")
print("Liquidation at: ~3.3% adverse move (vs 2% at 50x, 5% at 20x)")
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


symbol = 'XRPUSDT'
duration = 10080

file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)

close = df['close'].values
high = df['high'].values
low = df['low'].values

ema5 = ema(close, 5)
ema13 = ema(close, 13)
rsi7 = rsi(close, 7)

# 30x LEVERAGE
LEVERAGE = 30
TP_PCT = 1.0 / 100
SL_PCT = 10.0 / 100
POS_SIZE = 0.10
MAINT_MARGIN = 0.005

balance = 100.0
initial = balance
max_balance = balance
max_dd = 0

position = None
trades = []

print(f"\n{symbol}: {len(df):,} bars | Starting: ${balance:.2f} | Liq at ~{100/LEVERAGE - MAINT_MARGIN*100:.1f}% move")
print("-" * 100)
print(f"{'Bar':>6} {'Action':<8} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>10} {'Balance':>12} {'DD%':>8} {'Result':<12}")
print("-" * 100)

for i in range(100, min(duration, len(df))):
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

        # Check liquidation
        if direction == 1 and low[i] <= liq_price:
            pnl = -margin
            balance += pnl
            curr_dd = (max_balance - balance) / max_balance * 100
            trades.append({'pnl': pnl, 'result': 'LIQUIDATION'})
            print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {liq_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'LIQUIDATION':<12}")
            position = None
            continue
        elif direction == -1 and high[i] >= liq_price:
            pnl = -margin
            balance += pnl
            curr_dd = (max_balance - balance) / max_balance * 100
            trades.append({'pnl': pnl, 'result': 'LIQUIDATION'})
            print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {liq_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'LIQUIDATION':<12}")
            position = None
            continue

        if direction == 1:
            if high[i] >= tp_price:
                pnl = ((tp_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                curr_dd = 0 if balance >= max_balance else (max_balance - balance) / max_balance * 100
                trades.append({'pnl': pnl, 'result': 'TP'})
                print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {tp_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'TP WIN':<12}")
                position = None
            elif low[i] <= sl_price:
                pnl = ((sl_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                curr_dd = (max_balance - balance) / max_balance * 100
                trades.append({'pnl': pnl, 'result': 'SL'})
                print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {sl_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'SL LOSS':<12}")
                position = None
        else:
            if low[i] <= tp_price:
                pnl = ((entry - tp_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                curr_dd = 0 if balance >= max_balance else (max_balance - balance) / max_balance * 100
                trades.append({'pnl': pnl, 'result': 'TP'})
                print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {tp_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'TP WIN':<12}")
                position = None
            elif high[i] >= sl_price:
                pnl = ((entry - sl_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                balance += pnl
                curr_dd = (max_balance - balance) / max_balance * 100
                trades.append({'pnl': pnl, 'result': 'SL'})
                print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {sl_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'SL LOSS':<12}")
                position = None

    if position is None:
        signal = 0
        if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
            signal = 1
        elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
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
            curr_dd = 0 if balance >= max_balance else (max_balance - balance) / max_balance * 100
            print(f"{i:>6} {'OPEN':<8} {dir_str:<6} {entry:>10.4f} {'---':>10} {'---':>10} ${balance:>11.2f} {curr_dd:>7.1f}% TP:{tp_price:.4f}")

if position:
    direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position
    exit_price = close[-1]
    dir_str = "LONG" if direction == 1 else "SHORT"
    if direction == 1:
        pnl = ((exit_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
    else:
        pnl = ((entry - exit_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
    balance += pnl
    curr_dd = 0 if balance >= max_balance else (max_balance - balance) / max_balance * 100
    trades.append({'pnl': pnl, 'result': 'END'})
    print(f"{len(df)-1:>6} {'CLOSE':<8} {dir_str:<6} {entry:>10.4f} {exit_price:>10.4f} ${pnl:>+9.2f} ${balance:>11.2f} {curr_dd:>7.1f}% {'END':<12}")

wins = len([t for t in trades if t['pnl'] > 0])
losses = len([t for t in trades if t['pnl'] <= 0])
liquidations = len([t for t in trades if t['result'] == 'LIQUIDATION'])
total = wins + losses
win_rate = wins / total * 100 if total > 0 else 0
return_pct = (balance - initial) / initial * 100

print("-" * 100)
print()
print("=" * 100)
print(f"FINAL RESULTS - $100 @ {LEVERAGE}x XRPUSDT - 7 DAYS")
print("=" * 100)
print(f"Starting:     ${initial:.2f}")
print(f"Final:        ${balance:.2f}")
print(f"Profit/Loss:  ${balance - initial:+.2f}")
print(f"Return:       {return_pct:+.1f}%")
print(f"Max Drawdown: {max_dd:.1f}%")
print()
print(f"Trades:       {total}")
print(f"Wins:         {wins}")
print(f"Losses:       {losses}")
print(f"Liquidations: {liquidations}")
print(f"Win Rate:     {win_rate:.1f}%")
if wins > 0:
    print(f"Avg Win:      ${np.mean([t['pnl'] for t in trades if t['pnl'] > 0]):.2f}")
if losses > 0:
    print(f"Avg Loss:     ${np.mean([t['pnl'] for t in trades if t['pnl'] < 0]):.2f}")
print("=" * 100)
