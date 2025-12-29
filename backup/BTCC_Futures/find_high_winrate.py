#!/usr/bin/env python3
"""
Find HIGH WIN RATE Strategies (70%+ Win Rate)
==============================================
Goal: Find strategies with 70-80% win rate at 20x-50x leverage
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
print("HIGH WIN RATE STRATEGY FINDER - Target 70%+ Win Rate")
print("=" * 100)
print("\nKey insight: To get 70%+ win rate, TP must be SMALLER than SL")
print("  - Small TP = easier to hit = more wins")
print("  - Wider SL = less likely to stop out = fewer losses")
print("=" * 100, flush=True)

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
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    r = np.zeros(len(arr))
    if len(gains) < period:
        return r
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(arr) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            r[i + 1] = 100
        else:
            r[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))
    return r

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

    elif strategy == 'BOLLINGER_MEAN':
        # Mean reversion - enter when touching band, TP at middle band
        sm = sma(close, 20)
        st = rolling_std(close, 20)
        upper = sm + 2 * st
        lower = sm - 2 * st
        for i in range(25, len(close)):
            # Long when price at lower band
            if close[i] < lower[i] * 1.01:
                signals[i] = 1
            # Short when price at upper band
            elif close[i] > upper[i] * 0.99:
                signals[i] = -1

    elif strategy == 'RSI_OVERSOLD':
        # RSI oversold/overbought with quick TP
        r = rsi(close, 14)
        for i in range(20, len(close)):
            if r[i] < 30:  # Oversold
                signals[i] = 1
            elif r[i] > 70:  # Overbought
                signals[i] = -1

    elif strategy == 'SCALP_MOMENTUM':
        # Quick scalp on momentum
        ema5 = ema(close, 5)
        ema13 = ema(close, 13)
        r = rsi(close, 7)
        for i in range(20, len(close)):
            if ema5[i] > ema13[i] and r[i] > 50 and r[i] < 70:
                signals[i] = 1
            elif ema5[i] < ema13[i] and r[i] < 50 and r[i] > 30:
                signals[i] = -1

    elif strategy == 'TREND_SCALP':
        # Scalp in direction of trend
        ema20 = ema(close, 20)
        ema50 = ema(close, 50)
        for i in range(55, len(close)):
            # Only long in uptrend, short in downtrend
            if close[i] > ema20[i] and ema20[i] > ema50[i]:
                signals[i] = 1
            elif close[i] < ema20[i] and ema20[i] < ema50[i]:
                signals[i] = -1

    elif strategy == 'SUPPORT_BOUNCE':
        # Bounce from local support/resistance
        for i in range(25, len(close)):
            low_5 = np.min(low[i-5:i])
            high_5 = np.max(high[i-5:i])
            # Price bouncing from local low
            if low[i] <= low_5 * 1.002 and close[i] > open[i] if 'open' in dir() else close[i] > low[i]:
                signals[i] = 1
            # Price rejecting from local high
            elif high[i] >= high_5 * 0.998 and close[i] < high[i]:
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
    position = None
    liquidations = 0

    for i in range(50, len(df)):
        if balance > max_bal:
            max_bal = balance
        dd = (max_bal - balance) / max_bal * 100
        if dd > max_dd:
            max_dd = dd

        if balance <= 0:
            break

        if position:
            direction, entry, margin, tp, sl, liq = position

            if direction == 1:
                if low[i] <= liq:
                    balance -= margin
                    liquidations += 1
                    trades.append(('L', -margin))
                    position = None
                    continue
                elif high[i] >= tp:
                    pnl = ((tp - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(('W', pnl))
                    position = None
                elif low[i] <= sl:
                    pnl = ((sl - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(('L', pnl))
                    position = None
            else:
                if high[i] >= liq:
                    balance -= margin
                    liquidations += 1
                    trades.append(('L', -margin))
                    position = None
                    continue
                elif low[i] <= tp:
                    pnl = ((entry - tp) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(('W', pnl))
                    position = None
                elif high[i] >= sl:
                    pnl = ((entry - sl) / entry) * margin * leverage - margin * leverage * TAKER_FEE
                    balance += pnl
                    trades.append(('L', pnl))
                    position = None

        if position is None and signals[i] != 0:
            margin = balance * (pos_size / 100)
            if margin < 1:
                continue

            entry = close[i]
            direction = int(signals[i])

            balance -= margin * leverage * TAKER_FEE

            if direction == 1:
                tp = entry * (1 + tp_pct / 100)
                sl = entry * (1 - sl_pct / 100)
                liq = entry * (1 - (1/leverage) + MAINT_MARGIN)
            else:
                tp = entry * (1 - tp_pct / 100)
                sl = entry * (1 + sl_pct / 100)
                liq = entry * (1 + (1/leverage) - MAINT_MARGIN)

            position = (direction, entry, margin, tp, sl, liq)

    if position:
        direction, entry, margin, tp, sl, liq = position
        exit_price = close[-1]
        if direction == 1:
            pnl = ((exit_price - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE
        else:
            pnl = ((entry - exit_price) / entry) * margin * leverage - margin * leverage * TAKER_FEE
        balance += pnl
        if pnl > 0:
            trades.append(('W', pnl))
        else:
            trades.append(('L', pnl))

    wins = len([t for t in trades if t[0] == 'W'])
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    ret_pct = ((balance - initial) / initial) * 100

    gross_profit = sum(t[1] for t in trades if t[1] > 0)
    gross_loss = abs(sum(t[1] for t in trades if t[1] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'trades': total,
        'wins': wins,
        'win_rate': win_rate,
        'return_pct': ret_pct,
        'max_dd': max_dd,
        'liquidations': liquidations,
        'profit_factor': pf,
        'final_balance': balance,
        'profitable': ret_pct > 0
    }

# Test configurations for HIGH WIN RATE
# Key: TP < SL to get higher win rate
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'SOLUSDT']
strategies = ['EMA', 'BOLLINGER', 'BOLLINGER_MEAN', 'RSI_OVERSOLD', 'SCALP_MOMENTUM', 'TREND_SCALP']
leverages = [20, 30, 50]

# HIGH WIN RATE configs: Small TP, Wide SL
high_wr_configs = [
    # (tp_pct, sl_pct) - TP smaller than SL for higher win rate
    (0.5, 2.0),   # TP 0.5%, SL 2% - should hit TP 75%+ of time
    (0.8, 2.5),   # TP 0.8%, SL 2.5%
    (1.0, 3.0),   # TP 1%, SL 3%
    (0.5, 1.5),   # TP 0.5%, SL 1.5%
    (0.8, 2.0),   # TP 0.8%, SL 2%
    (1.0, 2.5),   # TP 1%, SL 2.5%
    (1.5, 3.0),   # TP 1.5%, SL 3%
    (0.3, 1.0),   # Ultra tight TP
    (0.5, 1.0),   # 1:2 risk reward inverted
]

pos_sizes = [3.0, 5.0]
test_period = 43200  # 30 days

results = []

for symbol in symbols:
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        print(f"SKIP: {symbol} - no data")
        continue

    df = pd.read_csv(file_path).tail(test_period).reset_index(drop=True)
    print(f"\n{symbol}: {len(df):,} candles", flush=True)

    for strategy in strategies:
        for lev in leverages:
            for tp, sl in high_wr_configs:
                for ps in pos_sizes:
                    r = simulate(df, lev, tp, sl, ps, strategy)

                    # Only keep if:
                    # 1. Win rate >= 60% (target high win rate)
                    # 2. Profitable
                    # 3. Max DD <= 50%
                    # 4. No liquidations
                    # 5. At least 20 trades
                    if (r['win_rate'] >= 60 and
                        r['profitable'] and
                        r['max_dd'] <= 50 and
                        r['liquidations'] == 0 and
                        r['trades'] >= 20):

                        r['symbol'] = symbol
                        r['strategy'] = strategy
                        r['leverage'] = lev
                        r['tp_pct'] = tp
                        r['sl_pct'] = sl
                        r['pos_size'] = ps
                        results.append(r)

    print(f"  Found {len([r for r in results if r['symbol'] == symbol])} high win rate configs")

print("\n" + "=" * 100)
print(f"TOTAL HIGH WIN RATE CONFIGS (60%+ WR): {len(results)}")
print("=" * 100, flush=True)

if results:
    # Sort by win rate first, then return
    results.sort(key=lambda x: (x['win_rate'], x['return_pct']), reverse=True)

    print(f"\n{'TOP 40 HIGH WIN RATE CONFIGURATIONS':^100}")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Strategy':<15} {'Lev':>4} {'TP%':>5} {'SL%':>5} {'Trades':>7} {'WIN%':>7} {'Return':>9} {'MaxDD':>7} {'PF':>6}")
    print("-" * 100)

    for r in results[:40]:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "INF"
        print(f"{r['symbol']:<10} {r['strategy']:<15} {r['leverage']:>4}x {r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['return_pct']:>+8.1f}% {r['max_dd']:>6.1f}% {pf_str:>6}")

    print("=" * 100)

    # Filter for 70%+ win rate
    high_wr = [r for r in results if r['win_rate'] >= 70]
    print(f"\n{'CONFIGURATIONS WITH 70%+ WIN RATE':^100}")
    print("-" * 100)

    if high_wr:
        print(f"{'Symbol':<10} {'Strategy':<15} {'Lev':>4} {'TP%':>5} {'SL%':>5} {'Trades':>7} {'WIN%':>7} {'Return':>9} {'MaxDD':>7}")
        print("-" * 100)
        for r in high_wr[:20]:
            print(f"{r['symbol']:<10} {r['strategy']:<15} {r['leverage']:>4}x {r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['return_pct']:>+8.1f}% {r['max_dd']:>6.1f}%")
    else:
        print("No configurations found with 70%+ win rate that are also profitable.")
        print("\nTRADE-OFF EXPLANATION:")
        print("  - High win rate (70%+) requires: Small TP, Wide SL")
        print("  - This means: Win often, but wins are small and losses are big")
        print("  - To be profitable with 70% WR, you need each win to cover losses from 30% losers")
        print("  - At 0.5% TP / 2% SL: You win $50, lose $200")
        print("  - Need 80%+ win rate to break even: 0.8 * $50 = $40 wins, 0.2 * $200 = $40 losses")

    print("=" * 100)

    # Best overall high win rate
    if high_wr:
        best = max(high_wr, key=lambda x: x['return_pct'])
        print(f"\n{'*** BEST HIGH WIN RATE CONFIGURATION ***':^100}")
        print("=" * 100)
        print(f"Symbol:        {best['symbol']}")
        print(f"Strategy:      {best['strategy']}")
        print(f"Leverage:      {best['leverage']}x")
        print(f"Take Profit:   {best['tp_pct']}%")
        print(f"Stop Loss:     {best['sl_pct']}%")
        print(f"\n30-Day Performance:")
        print(f"  Win Rate:    {best['win_rate']:.1f}%")
        print(f"  Return:      {best['return_pct']:+.1f}%")
        print(f"  Max DD:      {best['max_dd']:.1f}%")
        print(f"  Trades:      {best['trades']}")
        print("=" * 100)
else:
    print("\nNO HIGH WIN RATE PROFITABLE CONFIGS FOUND!")
    print("\nThe math reality:")
    print("  - To get 70%+ win rate, TP must be << SL")
    print("  - Example: TP 0.5%, SL 2% = 1:4 risk/reward")
    print("  - Even at 70% WR: 0.7 * 0.5 = 0.35% avg win, 0.3 * 2 = 0.6% avg loss")
    print("  - Net = -0.25% per trade = LOSING STRATEGY")
    print("\nRECOMMENDATION:")
    print("  - Accept 35-50% win rate with 2:1 or 3:1 reward/risk")
    print("  - This is how professional traders operate")
    print("  - Win rate is less important than EXPECTANCY (avg win * WR - avg loss * LR)")
