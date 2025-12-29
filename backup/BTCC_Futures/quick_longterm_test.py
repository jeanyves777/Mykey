#!/usr/bin/env python3
"""
Quick Long-Term Strategy Test
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

TAKER_FEE = 0.00045
MAINT_MARGIN = 0.005

print("=" * 100)
print("LONG-TERM STRATEGY FINDER - 20x-50x Leverage, Max 50% DD")
print("=" * 100)
print(flush=True)

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

def get_signals(close, high, low, volume, strategy):
    signals = np.zeros(len(close))

    if strategy == 'EMA':
        ema9 = ema(close, 9)
        ema21 = ema(close, 21)
        for i in range(25, len(close)):
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
                signals[i] = 1
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]:
                signals[i] = -1

    elif strategy == 'BOLLINGER':
        sm = sma(close, 20)
        st = rolling_std(close, 20)
        upper = sm + 2 * st
        lower = sm - 2 * st
        for i in range(25, len(close)):
            if close[i-1] <= lower[i-1] and close[i] > lower[i]:
                signals[i] = 1
            elif close[i-1] >= upper[i-1] and close[i] < upper[i]:
                signals[i] = -1

    elif strategy == 'MACD':
        ema12 = ema(close, 12)
        ema26 = ema(close, 26)
        macd = ema12 - ema26
        signal_line = ema(macd, 9)
        for i in range(30, len(close)):
            if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1]:
                signals[i] = 1
            elif macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1]:
                signals[i] = -1

    return signals

def simulate(df, leverage, tp_pct, sl_pct, pos_size, strategy, balance=10000):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

    signals = get_signals(close, high, low, volume, strategy)

    initial = balance
    max_bal = balance
    max_dd = 0
    trades = []
    position = None  # (direction, entry_price, margin, tp, sl, liq)
    liquidations = 0

    for i in range(50, len(df)):
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
            direction, entry, margin, tp, sl, liq = position

            # Check liquidation
            if direction == 1:
                if low[i] <= liq:
                    balance -= margin
                    liquidations += 1
                    trades.append(-margin)
                    position = None
                    continue
                elif high[i] >= tp:
                    pnl = ((tp - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(pnl)
                    position = None
                elif low[i] <= sl:
                    pnl = ((sl - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(pnl)
                    position = None
            else:  # Short
                if high[i] >= liq:
                    balance -= margin
                    liquidations += 1
                    trades.append(-margin)
                    position = None
                    continue
                elif low[i] <= tp:
                    pnl = ((entry - tp) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(pnl)
                    position = None
                elif high[i] >= sl:
                    pnl = ((entry - sl) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(pnl)
                    position = None

        # Open new position
        if position is None and signals[i] != 0:
            margin = balance * (pos_size / 100)
            if margin < 1:
                continue

            entry = close[i]
            direction = int(signals[i])

            balance -= margin * leverage * TAKER_FEE  # Entry fee

            if direction == 1:
                tp = entry * (1 + tp_pct / 100)
                sl = entry * (1 - sl_pct / 100)
                liq = entry * (1 - (1/leverage) + MAINT_MARGIN)
            else:
                tp = entry * (1 - tp_pct / 100)
                sl = entry * (1 + sl_pct / 100)
                liq = entry * (1 + (1/leverage) - MAINT_MARGIN)

            position = (direction, entry, margin, tp, sl, liq)

    # Close remaining
    if position:
        direction, entry, margin, tp, sl, liq = position
        exit_price = close[-1]
        if direction == 1:
            pnl = ((exit_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE
        else:
            pnl = ((entry - exit_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE
        balance += pnl
        trades.append(pnl)

    wins = len([t for t in trades if t > 0])
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    ret_pct = ((balance - initial) / initial) * 100

    return {
        'trades': total,
        'wins': wins,
        'win_rate': win_rate,
        'return_pct': ret_pct,
        'max_dd': max_dd,
        'liquidations': liquidations,
        'final_balance': balance,
        'profitable': ret_pct > 0
    }

# Test configurations
symbols = ['AVAXUSDT', 'LINKUSDT', 'XRPUSDT', 'BTCUSDT']
strategies = ['EMA', 'BOLLINGER', 'MACD']
leverages = [20, 30, 50]
tp_pcts = [2.0, 3.0, 4.0, 5.0]
sl_pcts = [1.0, 1.5, 2.0]
pos_sizes = [3.0, 5.0]

results = []
test_period = 43200  # 30 days

for symbol in symbols:
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        print(f"SKIP: {symbol} - no data")
        continue

    df = pd.read_csv(file_path).tail(test_period).reset_index(drop=True)
    print(f"\n{symbol}: {len(df):,} candles loaded")
    print("-" * 60, flush=True)

    count = 0
    for strategy in strategies:
        for lev in leverages:
            for tp in tp_pcts:
                for sl in sl_pcts:
                    if tp <= sl:
                        continue
                    for ps in pos_sizes:
                        r = simulate(df, lev, tp, sl, ps, strategy)
                        count += 1

                        # Only keep if profitable, max 50% DD, no liquidations
                        if r['profitable'] and r['max_dd'] <= 50 and r['liquidations'] == 0 and r['trades'] >= 10:
                            r['symbol'] = symbol
                            r['strategy'] = strategy
                            r['leverage'] = lev
                            r['tp_pct'] = tp
                            r['sl_pct'] = sl
                            r['pos_size'] = ps
                            results.append(r)

    found = len([r for r in results if r['symbol'] == symbol])
    print(f"  Tested {count} configs, found {found} viable")

print("\n" + "=" * 100)
print(f"TOTAL VIABLE CONFIGS: {len(results)}")
print("=" * 100, flush=True)

if results:
    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)

    print(f"\n{'TOP 30 LONG-TERM PROFITABLE CONFIGURATIONS':^100}")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Strategy':<12} {'Lev':>5} {'TP%':>6} {'SL%':>6} {'Size':>5} {'Trades':>7} {'Win%':>7} {'Return':>10} {'MaxDD':>8} {'Liqs':>5}")
    print("-" * 100)

    for r in results[:30]:
        print(f"{r['symbol']:<10} {r['strategy']:<12} {r['leverage']:>5}x {r['tp_pct']:>6.1f} {r['sl_pct']:>6.1f} {r['pos_size']:>5.1f} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['return_pct']:>+9.1f}% {r['max_dd']:>7.1f}% {r['liquidations']:>5}")

    print("=" * 100)

    # Best overall
    best = results[0]
    print(f"\n{'*** BEST LONG-TERM CONFIGURATION ***':^100}")
    print("=" * 100)
    print(f"Symbol:        {best['symbol']}")
    print(f"Strategy:      {best['strategy']}")
    print(f"Leverage:      {best['leverage']}x")
    print(f"Take Profit:   {best['tp_pct']}%")
    print(f"Stop Loss:     {best['sl_pct']}%")
    print(f"Position Size: {best['pos_size']}%")
    print(f"\n30-Day Performance:")
    print(f"  Return:      {best['return_pct']:+.1f}%")
    print(f"  Win Rate:    {best['win_rate']:.1f}%")
    print(f"  Max DD:      {best['max_dd']:.1f}%")
    print(f"  Trades:      {best['trades']}")
    print(f"  Liquidations: {best['liquidations']}")
    print("=" * 100)
else:
    print("\nNO PROFITABLE CONFIGS FOUND!")
    print("The strategy loses money over 30 days at 20x-50x leverage.")
    print("Consider:")
    print("  1. Lower leverage (10x or less)")
    print("  2. Shorter trading periods (1-7 days)")
    print("  3. Different market conditions")
