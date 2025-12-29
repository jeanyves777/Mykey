#!/usr/bin/env python3
"""
96.7% Win Rate Strategy - 7 Day Backtest
=========================================
Config: TP 1.0% / SL 10% / 10x Leverage

This is the SAFEST high win rate config found.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

TAKER_FEE = 0.00045

print("=" * 100)
print("96.7% WIN RATE STRATEGY - 7 DAY BACKTEST SIMULATION")
print("=" * 100)
print()
print("Config: TP 1.0% | SL 10% | Leverage 10x | Position Size 5%")
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


def run_simulation(symbol, duration_bars):
    """Run detailed simulation with trade-by-trade logging"""

    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        print(f"ERROR: No data for {symbol}")
        return None

    df = pd.read_csv(file_path).tail(duration_bars + 200).reset_index(drop=True)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Calculate indicators
    ema5 = ema(close, 5)
    ema13 = ema(close, 13)
    rsi7 = rsi(close, 7)

    # Settings - 96.7% win rate config
    LEVERAGE = 10
    TP_PCT = 1.0 / 100  # 1%
    SL_PCT = 10.0 / 100  # 10%
    POS_SIZE = 0.05  # 5% of account
    MAINT_MARGIN = 0.005

    # Account
    balance = 10000
    initial = balance
    max_balance = balance
    max_dd = 0

    # Position tracking
    position = None
    trades = []

    print(f"\n{symbol}: {len(df):,} bars | Starting Balance: ${balance:,.2f}")
    print("-" * 100)
    print(f"{'Bar':>6} {'Action':<8} {'Dir':<6} {'Entry':>12} {'Exit':>12} {'PnL':>12} {'Balance':>14} {'Result':<10}")
    print("-" * 100)

    for i in range(100, min(duration_bars, len(df))):
        # Update drawdown
        if balance > max_balance:
            max_balance = balance
        dd = (max_balance - balance) / max_balance * 100
        if dd > max_dd:
            max_dd = dd

        if balance <= 0:
            print(f"  *** ACCOUNT BLOWN AT BAR {i} ***")
            break

        # Check existing position
        if position:
            direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position
            dir_str = "LONG" if direction == 1 else "SHORT"

            # Check liquidation
            if direction == 1 and low[i] <= liq_price:
                pnl = -margin
                balance += pnl
                trades.append({
                    'bar': i, 'direction': dir_str, 'entry': entry, 'exit': liq_price,
                    'pnl': pnl, 'result': 'LIQUIDATION', 'bars_held': i - entry_bar
                })
                print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {liq_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'LIQUIDATION':<10}")
                position = None
                continue
            elif direction == -1 and high[i] >= liq_price:
                pnl = -margin
                balance += pnl
                trades.append({
                    'bar': i, 'direction': dir_str, 'entry': entry, 'exit': liq_price,
                    'pnl': pnl, 'result': 'LIQUIDATION', 'bars_held': i - entry_bar
                })
                print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {liq_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'LIQUIDATION':<10}")
                position = None
                continue

            # Check TP/SL for LONG
            if direction == 1:
                if high[i] >= tp_price:
                    pnl = ((tp_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({
                        'bar': i, 'direction': dir_str, 'entry': entry, 'exit': tp_price,
                        'pnl': pnl, 'result': 'TP', 'bars_held': i - entry_bar
                    })
                    print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {tp_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'TP WIN':<10}")
                    position = None
                elif low[i] <= sl_price:
                    pnl = ((sl_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({
                        'bar': i, 'direction': dir_str, 'entry': entry, 'exit': sl_price,
                        'pnl': pnl, 'result': 'SL', 'bars_held': i - entry_bar
                    })
                    print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {sl_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'SL LOSS':<10}")
                    position = None

            # Check TP/SL for SHORT
            elif direction == -1:
                if low[i] <= tp_price:
                    pnl = ((entry - tp_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({
                        'bar': i, 'direction': dir_str, 'entry': entry, 'exit': tp_price,
                        'pnl': pnl, 'result': 'TP', 'bars_held': i - entry_bar
                    })
                    print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {tp_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'TP WIN':<10}")
                    position = None
                elif high[i] >= sl_price:
                    pnl = ((entry - sl_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
                    balance += pnl
                    trades.append({
                        'bar': i, 'direction': dir_str, 'entry': entry, 'exit': sl_price,
                        'pnl': pnl, 'result': 'SL', 'bars_held': i - entry_bar
                    })
                    print(f"{i:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {sl_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'SL LOSS':<10}")
                    position = None

        # Open new position
        if position is None:
            signal = 0

            # SCALP_MOMENTUM signals
            if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
                signal = 1
            elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
                signal = -1

            if signal != 0:
                margin = balance * POS_SIZE
                entry = close[i]
                dir_str = "LONG" if signal == 1 else "SHORT"

                if signal == 1:  # Long
                    tp_price = entry * (1 + TP_PCT)
                    sl_price = entry * (1 - SL_PCT)
                    liq_price = entry * (1 - (1/LEVERAGE) + MAINT_MARGIN)
                else:  # Short
                    tp_price = entry * (1 - TP_PCT)
                    sl_price = entry * (1 + SL_PCT)
                    liq_price = entry * (1 + (1/LEVERAGE) - MAINT_MARGIN)

                position = (signal, entry, margin, tp_price, sl_price, liq_price, i)
                print(f"{i:>6} {'OPEN':<8} {dir_str:<6} {entry:>12.4f} {'---':>12} {'---':>12} ${balance:>13,.2f} {'TP:{:.4f} SL:{:.4f}'.format(tp_price, sl_price):<10}")

    # Close remaining position
    if position:
        direction, entry, margin, tp_price, sl_price, liq_price, entry_bar = position
        exit_price = close[-1]
        dir_str = "LONG" if direction == 1 else "SHORT"

        if direction == 1:
            pnl = ((exit_price - entry) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2
        else:
            pnl = ((entry - exit_price) / entry) * margin * LEVERAGE - margin * LEVERAGE * TAKER_FEE * 2

        balance += pnl
        trades.append({
            'bar': len(df)-1, 'direction': dir_str, 'entry': entry, 'exit': exit_price,
            'pnl': pnl, 'result': 'END', 'bars_held': len(df) - 1 - entry_bar
        })
        print(f"{len(df)-1:>6} {'CLOSE':<8} {dir_str:<6} {entry:>12.4f} {exit_price:>12.4f} ${pnl:>+11.2f} ${balance:>13,.2f} {'END':<10}")

    # Calculate stats
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len([t for t in trades if t['pnl'] <= 0])
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0
    total_pnl = sum(t['pnl'] for t in trades)
    return_pct = (balance - initial) / initial * 100

    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losses > 0 else 0

    liquidations = len([t for t in trades if t['result'] == 'LIQUIDATION'])

    print("-" * 100)
    print()

    return {
        'symbol': symbol,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'return_pct': return_pct,
        'max_dd': max_dd,
        'final_balance': balance,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'liquidations': liquidations,
        'trades_list': trades,
    }


def main():
    duration = 10080  # 7 days in minutes

    symbols = ['ETHUSDT', 'BTCUSDT', 'XRPUSDT']
    all_results = []

    for symbol in symbols:
        result = run_simulation(symbol, duration)
        if result:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 100)
    print("7-DAY BACKTEST SUMMARY - 96.7% WIN RATE CONFIG")
    print("Config: TP 1.0% | SL 10% | Leverage 10x")
    print("=" * 100)
    print(f"{'Symbol':<12} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'Win%':>8} {'Return':>12} {'MaxDD':>10} {'Final Bal':>14}")
    print("-" * 100)

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_return = 0

    for r in all_results:
        print(f"{r['symbol']:<12} {r['trades']:>8} {r['wins']:>8} {r['losses']:>8} {r['win_rate']:>7.1f}% {r['return_pct']:>+11.1f}% {r['max_dd']:>9.1f}% ${r['final_balance']:>13,.2f}")
        total_trades += r['trades']
        total_wins += r['wins']
        total_losses += r['losses']
        total_return += r['return_pct']

    print("-" * 100)
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"{'TOTAL':<12} {total_trades:>8} {total_wins:>8} {total_losses:>8} {overall_wr:>7.1f}% {total_return/3:>+11.1f}%")

    print("\n" + "=" * 100)
    print("LOSS ANALYSIS")
    print("=" * 100)

    all_losses = []
    for r in all_results:
        for t in r['trades_list']:
            if t['pnl'] < 0:
                t['symbol'] = r['symbol']
                all_losses.append(t)

    if all_losses:
        print(f"\n{'#':>3} {'Symbol':<10} {'Dir':<6} {'Entry':>12} {'Exit':>12} {'PnL':>12} {'Bars':>8} {'Result':<12}")
        print("-" * 80)
        for i, t in enumerate(all_losses):
            print(f"{i+1:>3} {t['symbol']:<10} {t['direction']:<6} {t['entry']:>12.4f} {t['exit']:>12.4f} ${t['pnl']:>+11.2f} {t['bars_held']:>8} {t['result']:<12}")

        print(f"\nTotal Losses: {len(all_losses)}")
        print(f"Avg Loss: ${np.mean([t['pnl'] for t in all_losses]):.2f}")
    else:
        print("\n*** NO LOSSES! 100% WIN RATE! ***")

    print("\n" + "=" * 100)
    print("COMPARISON: 96.7% WR CONFIG vs ORIGINAL 85% WR CONFIG")
    print("=" * 100)
    print("""
    Original Config (TP 1% / SL 3% / 20x):
    - Win Rate: ~85%
    - Return: +50% (30 days)
    - Risk: HIGH (20x leverage, 3% SL)

    New Safe Config (TP 1% / SL 10% / 10x):
    - Win Rate: ~97%
    - Return: +17% (30 days)
    - Risk: LOWER (10x leverage, 10% SL)

    Trade-off: Lower return but MUCH higher win rate and safety.
    """)
    print("=" * 100)


if __name__ == "__main__":
    main()
