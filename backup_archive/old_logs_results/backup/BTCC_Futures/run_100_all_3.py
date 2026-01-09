#!/usr/bin/env python3
"""
$100 Capital - BTC, ETH, XRP @ 20x - 7 Day Backtest
====================================================
Testing all 3 profitable assets together
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
print("$100 CAPITAL - BTC + ETH + XRP @ 20x - 7 DAY BACKTEST")
print("=" * 100)
print()
print("Config: TP 1.0% | SL 10% | Leverage 20x | Position Size 10%")
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


def run_backtest(symbol, duration=10080):
    """Run backtest for a single symbol"""

    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema5 = ema(close, 5)
    ema13 = ema(close, 13)
    rsi7 = rsi(close, 7)

    LEVERAGE = 20
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

    for i in range(100, min(duration, len(df))):
        if balance > max_balance:
            max_balance = balance
        dd = (max_balance - balance) / max_balance * 100
        if dd > max_dd:
            max_dd = dd

        if balance <= 0:
            break

        if position:
            direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position

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
                    pnl = ((tp_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP'})
                    position = None
                elif low[i] <= sl_price:
                    pnl = ((sl_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL'})
                    position = None
            else:
                if low[i] <= tp_price:
                    pnl = ((entry - tp_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'TP'})
                    position = None
                elif high[i] >= sl_price:
                    pnl = ((entry - sl_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({'pnl': pnl, 'result': 'SL'})
                    position = None

        # Open position
        if position is None:
            signal = 0
            if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
                signal = 1
            elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
                signal = -1

            if signal != 0:
                margin = balance * POS_SIZE
                entry = close[i]

                if signal == 1:
                    tp_price = entry * (1 + TP_PCT)
                    sl_price = entry * (1 - SL_PCT)
                    liq_price = entry * (1 - (1/LEVERAGE) + MAINT_MARGIN)
                else:
                    tp_price = entry * (1 - TP_PCT)
                    sl_price = entry * (1 + SL_PCT)
                    liq_price = entry * (1 + (1/LEVERAGE) - MAINT_MARGIN)

                position = (signal, entry, margin, tp_price, sl_price, liq_price, i)

    # Close remaining
    if position:
        direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position
        exit_price = close[-1]
        if direction == 1:
            pnl = ((exit_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
        else:
            pnl = ((entry - exit_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
        balance += pnl
        trades.append({'pnl': pnl, 'result': 'END'})

    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len([t for t in trades if t['pnl'] <= 0])
    liqs = len([t for t in trades if t['result'] == 'LIQ'])
    total = wins + losses

    return {
        'symbol': symbol,
        'balance': balance,
        'profit': balance - initial,
        'return_pct': (balance - initial) / initial * 100,
        'max_dd': max_dd,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'liquidations': liqs,
        'win_rate': wins / total * 100 if total > 0 else 0,
    }


# Run backtests
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
results = []

for symbol in symbols:
    print(f"\nTesting {symbol}...")
    r = run_backtest(symbol)
    if r:
        results.append(r)
        print(f"  Trades: {r['trades']} | Win Rate: {r['win_rate']:.1f}% | Return: {r['return_pct']:+.1f}%")

# Summary
print("\n" + "=" * 100)
print("INDIVIDUAL RESULTS - $100 EACH @ 20x - 7 DAYS")
print("=" * 100)
print(f"{'Symbol':<12} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'Win%':>8} {'Liqs':>6} {'Return':>10} {'Final Bal':>12}")
print("-" * 100)

total_profit = 0
for r in results:
    print(f"{r['symbol']:<12} {r['trades']:>8} {r['wins']:>8} {r['losses']:>8} {r['win_rate']:>7.1f}% {r['liquidations']:>6} {r['return_pct']:>+9.1f}% ${r['balance']:>11.2f}")
    total_profit += r['profit']

print("-" * 100)

# Combined stats
total_trades = sum(r['trades'] for r in results)
total_wins = sum(r['wins'] for r in results)
total_losses = sum(r['losses'] for r in results)
total_liqs = sum(r['liquidations'] for r in results)
combined_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
combined_balance = sum(r['balance'] for r in results)

print(f"{'COMBINED':<12} {total_trades:>8} {total_wins:>8} {total_losses:>8} {combined_wr:>7.1f}% {total_liqs:>6} {total_profit/3:>+9.1f}% ${combined_balance:>11.2f}")

print("\n" + "=" * 100)
print("PORTFOLIO SUMMARY")
print("=" * 100)
print(f"Total Capital Deployed:  $300 ($100 x 3 assets)")
print(f"Final Portfolio Value:   ${combined_balance:.2f}")
print(f"Total Profit:            ${total_profit:+.2f}")
print(f"Portfolio Return:        {total_profit/300*100:+.1f}%")
print(f"Average Return/Asset:    {total_profit/3:+.1f}%")
print()
print(f"Total Trades:            {total_trades}")
print(f"Overall Win Rate:        {combined_wr:.1f}%")
print(f"Liquidations:            {total_liqs}")
print("=" * 100)

# Best performer
best = max(results, key=lambda x: x['return_pct'])
print(f"\nBEST PERFORMER: {best['symbol']}")
print(f"  Return: {best['return_pct']:+.1f}% | Win Rate: {best['win_rate']:.1f}% | Trades: {best['trades']}")
print("=" * 100)
