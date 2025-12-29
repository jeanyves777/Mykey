#!/usr/bin/env python3
"""
Detailed 7-Day Trade Analysis
==============================
Show EVERY trade with full details to find what to improve for 95%+ win rate
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


def ema(arr, period):
    """Calculate EMA"""
    e = np.zeros_like(arr)
    alpha = 2 / (period + 1)
    e[0] = arr[0]
    for i in range(1, len(arr)):
        e[i] = alpha * arr[i] + (1 - alpha) * e[i-1]
    return e


def sma(arr, period):
    """Calculate SMA"""
    s = np.zeros_like(arr)
    for i in range(period - 1, len(arr)):
        s[i] = np.mean(arr[i-period+1:i+1])
    return s


def rolling_std(arr, period):
    """Calculate rolling standard deviation"""
    s = np.zeros_like(arr)
    for i in range(period - 1, len(arr)):
        s[i] = np.std(arr[i-period+1:i+1])
    return s


def rsi(arr, period):
    """Calculate RSI"""
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


def run_detailed_analysis():
    """Run detailed trade-by-trade analysis"""

    print("=" * 120)
    print("DETAILED 7-DAY TRADE ANALYSIS - FINDING PATH TO 95%+ WIN RATE")
    print("=" * 120)
    print()

    # Configurations from the high win rate finder
    configs = [
        {'symbol': 'ETHUSDT', 'strategy': 'SCALP_MOMENTUM', 'leverage': 20, 'tp': 1.0, 'sl': 3.0},
        {'symbol': 'BTCUSDT', 'strategy': 'SCALP_MOMENTUM', 'leverage': 20, 'tp': 1.0, 'sl': 3.0},
        {'symbol': 'XRPUSDT', 'strategy': 'BOLLINGER_MEAN', 'leverage': 20, 'tp': 1.0, 'sl': 3.0},
    ]

    duration = 10080  # 7 days in minutes
    all_trades = []

    for cfg in configs:
        symbol = cfg['symbol']
        file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")

        if not os.path.exists(file_path):
            print(f"SKIP: {symbol} - no data")
            continue

        df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)
        print(f"\n{symbol}: {len(df):,} bars loaded | Strategy: {cfg['strategy']}")
        print("-" * 80)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Calculate indicators based on strategy
        if cfg['strategy'] == 'SCALP_MOMENTUM':
            ema5 = ema(close, 5)
            ema13 = ema(close, 13)
            rsi7 = rsi(close, 7)
        else:  # BOLLINGER_MEAN
            bb_mid = sma(close, 20)
            bb_std = rolling_std(close, 20)
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std

        # Simulation
        balance = 10000
        initial = balance
        position = None
        trades = []

        leverage = cfg['leverage']
        tp_pct = cfg['tp'] / 100
        sl_pct = cfg['sl'] / 100
        pos_size_pct = 0.05  # 5% of account
        maint_margin = 0.005

        for i in range(100, min(duration, len(df))):
            # Check existing position
            if position:
                direction, entry, margin, tp, sl, liq, entry_bar = position

                # Check liquidation
                if direction == 1 and low[i] <= liq:
                    pnl = -margin
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'bar': i,
                        'direction': 'LONG',
                        'entry': entry,
                        'exit': liq,
                        'pnl': pnl,
                        'result': 'LIQUIDATION',
                        'bars_held': i - entry_bar,
                        'move_pct': (liq - entry) / entry * 100,
                    })
                    position = None
                    continue
                elif direction == -1 and high[i] >= liq:
                    pnl = -margin
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'bar': i,
                        'direction': 'SHORT',
                        'entry': entry,
                        'exit': liq,
                        'pnl': pnl,
                        'result': 'LIQUIDATION',
                        'bars_held': i - entry_bar,
                        'move_pct': (entry - liq) / entry * 100,
                    })
                    position = None
                    continue

                # Check TP/SL for LONG
                if direction == 1:
                    if high[i] >= tp:
                        pnl = ((tp - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        balance += pnl
                        trades.append({
                            'symbol': symbol,
                            'bar': i,
                            'direction': 'LONG',
                            'entry': entry,
                            'exit': tp,
                            'pnl': pnl,
                            'result': 'TP',
                            'bars_held': i - entry_bar,
                            'move_pct': (tp - entry) / entry * 100,
                        })
                        position = None
                    elif low[i] <= sl:
                        pnl = ((sl - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        balance += pnl
                        trades.append({
                            'symbol': symbol,
                            'bar': i,
                            'direction': 'LONG',
                            'entry': entry,
                            'exit': sl,
                            'pnl': pnl,
                            'result': 'SL',
                            'bars_held': i - entry_bar,
                            'move_pct': (sl - entry) / entry * 100,
                        })
                        position = None

                # Check TP/SL for SHORT
                elif direction == -1:
                    if low[i] <= tp:
                        pnl = ((entry - tp) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        balance += pnl
                        trades.append({
                            'symbol': symbol,
                            'bar': i,
                            'direction': 'SHORT',
                            'entry': entry,
                            'exit': tp,
                            'pnl': pnl,
                            'result': 'TP',
                            'bars_held': i - entry_bar,
                            'move_pct': (entry - tp) / entry * 100,
                        })
                        position = None
                    elif high[i] >= sl:
                        pnl = ((entry - sl) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        balance += pnl
                        trades.append({
                            'symbol': symbol,
                            'bar': i,
                            'direction': 'SHORT',
                            'entry': entry,
                            'exit': sl,
                            'pnl': pnl,
                            'result': 'SL',
                            'bars_held': i - entry_bar,
                            'move_pct': (entry - sl) / entry * 100,
                        })
                        position = None

            # Open new position if none
            if position is None:
                signal = 0

                if cfg['strategy'] == 'SCALP_MOMENTUM':
                    # Long: EMA5 > EMA13 + RSI 50-70
                    if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
                        signal = 1
                    # Short: EMA5 < EMA13 + RSI 30-50
                    elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
                        signal = -1

                else:  # BOLLINGER_MEAN
                    # Long: Price below lower band
                    if close[i] < bb_lower[i] * 1.01:
                        signal = 1
                    # Short: Price above upper band
                    elif close[i] > bb_upper[i] * 0.99:
                        signal = -1

                if signal != 0:
                    margin = balance * pos_size_pct
                    entry = close[i]

                    if signal == 1:  # Long
                        tp = entry * (1 + tp_pct)
                        sl = entry * (1 - sl_pct)
                        liq = entry * (1 - (1/leverage) + maint_margin)
                    else:  # Short
                        tp = entry * (1 - tp_pct)
                        sl = entry * (1 + sl_pct)
                        liq = entry * (1 + (1/leverage) - maint_margin)

                    position = (signal, entry, margin, tp, sl, liq, i)

        # Add trades to all_trades
        all_trades.extend(trades)

        # Print stats for this symbol
        wins = len([t for t in trades if t['pnl'] > 0])
        losses = len([t for t in trades if t['pnl'] <= 0])
        win_rate = wins / len(trades) * 100 if trades else 0
        total_pnl = sum(t['pnl'] for t in trades)

        print(f"  Trades: {len(trades)} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${total_pnl:+.2f}")

    # Detailed analysis of ALL trades
    print("\n" + "=" * 120)
    print("ALL TRADES - DETAILED VIEW")
    print("=" * 120)
    print(f"{'#':>3} {'Symbol':<10} {'Dir':<6} {'Entry':>12} {'Exit':>12} {'Move%':>8} {'Bars':>6} {'Result':>10} {'PnL':>12}")
    print("-" * 120)

    for i, t in enumerate(all_trades):
        print(f"{i+1:>3} {t['symbol']:<10} {t['direction']:<6} {t['entry']:>12.4f} {t['exit']:>12.4f} {t['move_pct']:>+7.2f}% {t['bars_held']:>6} {t['result']:>10} ${t['pnl']:>+10.2f}")

    # LOSS ANALYSIS
    print("\n" + "=" * 120)
    print("LOSS ANALYSIS - WHAT WENT WRONG?")
    print("=" * 120)

    losses = [t for t in all_trades if t['pnl'] < 0]
    wins = [t for t in all_trades if t['pnl'] > 0]

    print(f"\nTotal Trades: {len(all_trades)}")
    print(f"Wins: {len(wins)} ({len(wins)/len(all_trades)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(all_trades)*100:.1f}%)")

    if losses:
        print(f"\n{'LOSING TRADES BREAKDOWN':^120}")
        print("-" * 120)

        for i, t in enumerate(losses):
            print(f"\nLOSS #{i+1}: {t['symbol']} {t['direction']}")
            print(f"  Entry: {t['entry']:.4f} | Exit: {t['exit']:.4f}")
            print(f"  Move: {t['move_pct']:+.2f}% | Bars Held: {t['bars_held']}")
            print(f"  PnL: ${t['pnl']:.2f} | Result: {t['result']}")

        # Loss statistics
        print(f"\n{'LOSS STATISTICS':^120}")
        print("-" * 120)

        avg_loss = np.mean([t['pnl'] for t in losses])
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss_move = np.mean([abs(t['move_pct']) for t in losses])
        avg_win_move = np.mean([t['move_pct'] for t in wins]) if wins else 0
        avg_loss_bars = np.mean([t['bars_held'] for t in losses])
        avg_win_bars = np.mean([t['bars_held'] for t in wins]) if wins else 0

        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x")
        print(f"\nAverage Adverse Move on Loss: {avg_loss_move:.2f}%")
        print(f"Average Favorable Move on Win: {avg_win_move:.2f}%")
        print(f"\nAverage Bars Held (Loss): {avg_loss_bars:.0f} minutes")
        print(f"Average Bars Held (Win): {avg_win_bars:.0f} minutes")

        # Loss by symbol
        print(f"\n{'LOSSES BY SYMBOL':^120}")
        print("-" * 120)
        for symbol in ['ETHUSDT', 'BTCUSDT', 'XRPUSDT']:
            sym_losses = [t for t in losses if t['symbol'] == symbol]
            sym_total = len([t for t in all_trades if t['symbol'] == symbol])
            if sym_losses:
                print(f"{symbol}: {len(sym_losses)} losses out of {sym_total} trades ({len(sym_losses)/sym_total*100:.1f}% loss rate)")

        # Loss by direction
        print(f"\n{'LOSSES BY DIRECTION':^120}")
        print("-" * 120)
        long_losses = [t for t in losses if t['direction'] == 'LONG']
        short_losses = [t for t in losses if t['direction'] == 'SHORT']
        long_total = len([t for t in all_trades if t['direction'] == 'LONG'])
        short_total = len([t for t in all_trades if t['direction'] == 'SHORT'])

        print(f"LONG losses: {len(long_losses)} out of {long_total} ({len(long_losses)/long_total*100:.1f}% loss rate)" if long_total else "No LONG trades")
        print(f"SHORT losses: {len(short_losses)} out of {short_total} ({len(short_losses)/short_total*100:.1f}% loss rate)" if short_total else "No SHORT trades")

    # RECOMMENDATIONS FOR 95%+ WIN RATE
    print("\n" + "=" * 120)
    print("RECOMMENDATIONS FOR 95%+ WIN RATE")
    print("=" * 120)

    print("""
    CURRENT PROBLEM:
    - We have ~73% win rate with 1% TP / 3% SL
    - Each loss = 3x the gain of each win
    - Need 75%+ win rate just to break even!

    TO ACHIEVE 95%+ WIN RATE, TRY:

    1. EVEN WIDER STOP LOSS:
       - Current: 3% SL
       - Try: 5% or even 10% SL
       - Risk: Huge losses when hit, but hit less often

    2. EVEN SMALLER TAKE PROFIT:
       - Current: 1% TP
       - Try: 0.5% or 0.3% TP
       - Pro: More wins, faster exits
       - Con: Fees eat into profits

    3. ADD MORE FILTERS:
       - Volume confirmation
       - Trend alignment (higher timeframe)
       - Volatility filter (avoid high volatility)
       - Time filter (avoid news hours)

    4. ONLY TRADE WITH THE TREND:
       - Don't counter-trend
       - Use EMA200 as trend filter
       - Only long above EMA200, short below

    5. REDUCE LEVERAGE:
       - 20x is aggressive
       - 10x gives more room for error
       - Less liquidation risk
    """)

    # Test alternative configs
    print("\n" + "=" * 120)
    print("TESTING ALTERNATIVE CONFIGS FOR 95%+ WIN RATE")
    print("=" * 120)

    # Test with wider SL and smaller TP
    test_configs = [
        {'tp': 0.5, 'sl': 5.0, 'name': 'Ultra Wide: TP 0.5% / SL 5%'},
        {'tp': 0.5, 'sl': 10.0, 'name': 'Extreme Wide: TP 0.5% / SL 10%'},
        {'tp': 0.3, 'sl': 5.0, 'name': 'Tiny TP: TP 0.3% / SL 5%'},
        {'tp': 1.0, 'sl': 5.0, 'name': 'Balanced: TP 1.0% / SL 5%'},
        {'tp': 1.0, 'sl': 10.0, 'name': 'Very Wide: TP 1.0% / SL 10%'},
    ]

    symbol = 'ETHUSDT'  # Test on best performer
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    df = pd.read_csv(file_path).tail(duration + 200).reset_index(drop=True)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    ema5 = ema(close, 5)
    ema13 = ema(close, 13)
    rsi7 = rsi(close, 7)

    print(f"\nTesting on {symbol} - 7 days")
    print("-" * 80)
    print(f"{'Config':<35} {'Trades':>7} {'Wins':>7} {'Win%':>8} {'PnL':>12} {'Avg Win':>10} {'Avg Loss':>10}")
    print("-" * 120)

    for tcfg in test_configs:
        tp_pct = tcfg['tp'] / 100
        sl_pct = tcfg['sl'] / 100
        leverage = 20

        balance = 10000
        position = None
        trades = []

        for i in range(100, min(duration, len(df))):
            if position:
                direction, entry, margin, tp, sl, liq, entry_bar = position

                # Check TP/SL
                if direction == 1:
                    if high[i] >= tp:
                        pnl = ((tp - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        trades.append(pnl)
                        position = None
                    elif low[i] <= sl:
                        pnl = ((sl - entry) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        trades.append(pnl)
                        position = None
                else:
                    if low[i] <= tp:
                        pnl = ((entry - tp) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        trades.append(pnl)
                        position = None
                    elif high[i] >= sl:
                        pnl = ((entry - sl) / entry) * margin * leverage - margin * leverage * TAKER_FEE * 2
                        trades.append(pnl)
                        position = None

            if position is None:
                if ema5[i] > ema13[i] and 50 < rsi7[i] < 70:
                    signal = 1
                elif ema5[i] < ema13[i] and 30 < rsi7[i] < 50:
                    signal = -1
                else:
                    signal = 0

                if signal != 0:
                    margin = balance * 0.05
                    entry = close[i]
                    maint = 0.005

                    if signal == 1:
                        tp = entry * (1 + tp_pct)
                        sl = entry * (1 - sl_pct)
                        liq = entry * (1 - (1/leverage) + maint)
                    else:
                        tp = entry * (1 - tp_pct)
                        sl = entry * (1 + sl_pct)
                        liq = entry * (1 + (1/leverage) - maint)

                    position = (signal, entry, margin, tp, sl, liq, i)

        if trades:
            wins = len([t for t in trades if t > 0])
            win_rate = wins / len(trades) * 100
            total_pnl = sum(trades)
            avg_win = np.mean([t for t in trades if t > 0]) if wins else 0
            avg_loss = np.mean([t for t in trades if t < 0]) if len(trades) - wins > 0 else 0

            wr_color = "***" if win_rate >= 90 else ""
            print(f"{tcfg['name']:<35} {len(trades):>7} {wins:>7} {win_rate:>7.1f}% ${total_pnl:>+10.2f} ${avg_win:>9.2f} ${avg_loss:>9.2f} {wr_color}")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    run_detailed_analysis()
