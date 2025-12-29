#!/usr/bin/env python3
"""
Find 95%+ Win Rate Strategy
============================
Test many TP/SL combinations to find consistent 95%+ win rate
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

TAKER_FEE = 0.00045


def ema(arr, period):
    e = np.zeros_like(arr)
    alpha = 2 / (period + 1)
    e[0] = arr[0]
    for i in range(1, len(arr)):
        e[i] = alpha * arr[i] + (1 - alpha) * e[i-1]
    return e


def sma(arr, period):
    s = np.zeros_like(arr)
    for i in range(period - 1, len(arr)):
        s[i] = np.mean(arr[i-period+1:i+1])
    return s


def rolling_std(arr, period):
    s = np.zeros_like(arr)
    for i in range(period - 1, len(arr)):
        s[i] = np.std(arr[i-period+1:i+1])
    return s


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


def test_config(df, tp_pct, sl_pct, leverage, strategy, ema200_filter=False):
    """Test a configuration and return results"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Calculate indicators
    ema5 = ema(close, 5)
    ema13 = ema(close, 13)
    ema200 = ema(close, 200) if ema200_filter else None
    rsi7 = rsi(close, 7)

    bb_mid = sma(close, 20)
    bb_std = rolling_std(close, 20)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    balance = 10000
    initial = balance
    max_bal = balance
    max_dd = 0
    position = None
    trades = []
    wins = 0
    losses = 0
    liquidations = 0

    tp = tp_pct / 100
    sl = sl_pct / 100
    pos_size = 0.05
    maint = 0.005

    for i in range(250, len(df)):
        # Update drawdown
        if balance > max_bal:
            max_bal = balance
        dd = (max_bal - balance) / max_bal * 100
        if dd > max_dd:
            max_dd = dd

        if balance <= 0:
            break

        # Check position
        if position:
            direction, entry, margin, tp_price, sl_price, liq, entry_bar = position

            # Check liquidation
            if direction == 1 and low[i] <= liq:
                balance -= margin
                liquidations += 1
                losses += 1
                trades.append(-margin)
                position = None
                continue
            elif direction == -1 and high[i] >= liq:
                balance -= margin
                liquidations += 1
                losses += 1
                trades.append(-margin)
                position = None
                continue

            # Check TP/SL
            if direction == 1:
                if high[i] >= tp_price:
                    pnl = ((tp_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    wins += 1
                    trades.append(pnl)
                    position = None
                elif low[i] <= sl_price:
                    pnl = ((sl_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    losses += 1
                    trades.append(pnl)
                    position = None
            else:
                if low[i] <= tp_price:
                    pnl = ((entry - tp_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    wins += 1
                    trades.append(pnl)
                    position = None
                elif high[i] >= sl_price:
                    pnl = ((entry - sl_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                    balance += pnl
                    losses += 1
                    trades.append(pnl)
                    position = None

        # Open new position
        if position is None:
            signal = 0

            if strategy == 'SCALP_MOMENTUM':
                if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
                    if ema200_filter:
                        if close[i] > ema200[i]:  # Only long above EMA200
                            signal = 1
                    else:
                        signal = 1
                elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
                    if ema200_filter:
                        if close[i] < ema200[i]:  # Only short below EMA200
                            signal = -1
                    else:
                        signal = -1

            elif strategy == 'BOLLINGER_MEAN':
                if close[i] < bb_lower[i] * 1.01:
                    if ema200_filter:
                        if close[i] > ema200[i]:
                            signal = 1
                    else:
                        signal = 1
                elif close[i] > bb_upper[i] * 0.99:
                    if ema200_filter:
                        if close[i] < ema200[i]:
                            signal = -1
                    else:
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
    ret_pct = (balance - initial) / initial * 100
    avg_win = np.mean([t for t in trades if t > 0]) if wins > 0 else 0
    avg_loss = np.mean([t for t in trades if t < 0]) if losses > 0 else 0

    return {
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'return_pct': ret_pct,
        'max_dd': max_dd,
        'liquidations': liquidations,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'balance': balance,
    }


def main():
    print("=" * 120)
    print("FINDING 95%+ WIN RATE STRATEGY")
    print("=" * 120)
    print()

    symbols = ['ETHUSDT', 'BTCUSDT', 'XRPUSDT']
    strategies = ['SCALP_MOMENTUM', 'BOLLINGER_MEAN']
    leverages = [10, 20]

    # TP/SL combinations to test (wider SL = higher win rate)
    configs = [
        # Standard
        (1.0, 3.0),
        (1.0, 5.0),
        (1.0, 7.0),
        (1.0, 10.0),
        # Smaller TP
        (0.5, 3.0),
        (0.5, 5.0),
        (0.5, 7.0),
        (0.5, 10.0),
        # Tiny TP
        (0.3, 3.0),
        (0.3, 5.0),
        (0.3, 7.0),
        (0.3, 10.0),
        # Micro TP
        (0.2, 5.0),
        (0.2, 7.0),
        (0.2, 10.0),
    ]

    durations = [
        (10080, '7 DAYS'),
        (43200, '30 DAYS'),
    ]

    results = []

    for symbol in symbols:
        file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        print(f"\nTesting {symbol}: {len(df):,} bars available")
        print("-" * 80)

        for duration, dur_name in durations:
            test_df = df.tail(duration + 300).reset_index(drop=True)

            for strategy in strategies:
                for leverage in leverages:
                    for tp, sl in configs:
                        for ema_filter in [False, True]:
                            r = test_config(test_df, tp, sl, leverage, strategy, ema_filter)

                            if r['trades'] >= 5:  # Minimum trades
                                r['symbol'] = symbol
                                r['strategy'] = strategy
                                r['leverage'] = leverage
                                r['tp'] = tp
                                r['sl'] = sl
                                r['duration'] = dur_name
                                r['ema_filter'] = ema_filter
                                results.append(r)

    # Filter for 90%+ win rate
    high_wr = [r for r in results if r['win_rate'] >= 90 and r['trades'] >= 5]
    high_wr.sort(key=lambda x: (-x['win_rate'], -x['return_pct']))

    print("\n" + "=" * 140)
    print(f"{'CONFIGURATIONS WITH 90%+ WIN RATE':^140}")
    print("=" * 140)
    print(f"{'Symbol':<10} {'Strategy':<15} {'Period':<8} {'Lev':>4} {'TP%':>5} {'SL%':>5} {'EMA':>4} {'Trades':>7} {'Win%':>7} {'Return':>10} {'MaxDD':>8} {'AvgWin':>9} {'AvgLoss':>10}")
    print("-" * 140)

    for r in high_wr[:50]:
        ema_str = 'Yes' if r['ema_filter'] else 'No'
        wr_marker = "***" if r['win_rate'] >= 95 else ""
        print(f"{r['symbol']:<10} {r['strategy']:<15} {r['duration']:<8} {r['leverage']:>4}x {r['tp']:>5.1f} {r['sl']:>5.1f} {ema_str:>4} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}% ${r['avg_win']:>8.2f} ${r['avg_loss']:>9.2f} {wr_marker}")

    # Find best profitable 95%+ configs
    print("\n" + "=" * 140)
    print(f"{'BEST 95%+ WIN RATE CONFIGS (PROFITABLE)':^140}")
    print("=" * 140)

    profitable_95 = [r for r in results if r['win_rate'] >= 95 and r['return_pct'] > 0 and r['trades'] >= 5]
    profitable_95.sort(key=lambda x: (-x['win_rate'], -x['return_pct']))

    if profitable_95:
        print(f"{'Symbol':<10} {'Strategy':<15} {'Period':<8} {'Lev':>4} {'TP%':>5} {'SL%':>5} {'EMA':>4} {'Trades':>7} {'Win%':>7} {'Return':>10} {'MaxDD':>8}")
        print("-" * 140)

        for r in profitable_95[:20]:
            ema_str = 'Yes' if r['ema_filter'] else 'No'
            print(f"{r['symbol']:<10} {r['strategy']:<15} {r['duration']:<8} {r['leverage']:>4}x {r['tp']:>5.1f} {r['sl']:>5.1f} {ema_str:>4} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}%")

        best = profitable_95[0]
        print("\n" + "=" * 140)
        print(f"{'*** BEST 95%+ WIN RATE PROFITABLE CONFIG ***':^140}")
        print("=" * 140)
        print(f"Symbol:        {best['symbol']}")
        print(f"Strategy:      {best['strategy']}")
        print(f"Period:        {best['duration']}")
        print(f"Leverage:      {best['leverage']}x")
        print(f"Take Profit:   {best['tp']}%")
        print(f"Stop Loss:     {best['sl']}%")
        print(f"EMA200 Filter: {'Yes' if best['ema_filter'] else 'No'}")
        print()
        print(f"Results:")
        print(f"  Trades:      {best['trades']}")
        print(f"  Win Rate:    {best['win_rate']:.1f}%")
        print(f"  Return:      {best['return_pct']:+.1f}%")
        print(f"  Max DD:      {best['max_dd']:.1f}%")
        print(f"  Avg Win:     ${best['avg_win']:.2f}")
        print(f"  Avg Loss:    ${best['avg_loss']:.2f}")
        print("=" * 140)
    else:
        print("\nNo profitable 95%+ win rate configs found!")
        print("\nThe math problem:")
        print("  - With 95% win rate: 95 wins, 5 losses per 100 trades")
        print("  - If TP=0.5%, SL=10%: Win=$50, Loss=$1000")
        print("  - 95 * $50 - 5 * $1000 = $4750 - $5000 = -$250 (LOSS!)")
        print()
        print("To be profitable at 95% win rate, need:")
        print("  - Even smaller SL (but then win rate drops)")
        print("  - OR larger TP (but then win rate drops)")
        print()
        print("The balance is tricky. Let's find what actually works...")

    # Show best overall profitable configs
    print("\n" + "=" * 140)
    print(f"{'TOP PROFITABLE CONFIGS (ANY WIN RATE, MIN 10 TRADES)':^140}")
    print("=" * 140)

    profitable = [r for r in results if r['return_pct'] > 0 and r['trades'] >= 10 and r['liquidations'] == 0]
    profitable.sort(key=lambda x: -x['return_pct'])

    print(f"{'Symbol':<10} {'Strategy':<15} {'Period':<8} {'Lev':>4} {'TP%':>5} {'SL%':>5} {'EMA':>4} {'Trades':>7} {'Win%':>7} {'Return':>10} {'MaxDD':>8}")
    print("-" * 140)

    for r in profitable[:30]:
        ema_str = 'Yes' if r['ema_filter'] else 'No'
        print(f"{r['symbol']:<10} {r['strategy']:<15} {r['duration']:<8} {r['leverage']:>4}x {r['tp']:>5.1f} {r['sl']:>5.1f} {ema_str:>4} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}%")

    print("=" * 140)


if __name__ == "__main__":
    main()
