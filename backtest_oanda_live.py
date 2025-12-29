"""
Backtest Optimized Strategy with LIVE OANDA Data
Pulls ~3-4 days of 1-minute data directly from OANDA API
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from trading_system.Forex_Trading.engine.oanda_client import OandaClient

# Optimized settings per pair
PAIR_SETTINGS = {
    'EUR_USD': {'strategy': 'RSI_30_70', 'tp_pips': 5, 'sl_pips': 20, 'pip_mult': 10000},
    'GBP_USD': {'strategy': 'STRONG_TREND', 'tp_pips': 5, 'sl_pips': 20, 'pip_mult': 10000},
    'USD_JPY': {'strategy': 'RSI_30_70', 'tp_pips': 5, 'sl_pips': 20, 'pip_mult': 100},
    'USD_CHF': {'strategy': 'STRONG_TREND', 'tp_pips': 8, 'sl_pips': 24, 'pip_mult': 10000},
    'USD_CAD': {'strategy': 'RSI_30_70', 'tp_pips': 5, 'sl_pips': 20, 'pip_mult': 10000},
}

print("=" * 70)
print("BACKTEST WITH LIVE OANDA DATA")
print("Pulling ~3-4 days of 1-minute candles from OANDA API")
print("=" * 70)

# Initialize OANDA client
oanda = OandaClient(account_type="practice")

def fetch_oanda_data(instrument, count=5000):
    """Fetch candles from OANDA and convert to DataFrame"""
    print(f"  Fetching {count} M1 candles for {instrument}...")
    candles = oanda.get_candles(instrument, granularity="M1", count=count)

    if not candles:
        print(f"  ERROR: No candles returned for {instrument}")
        return None

    df = pd.DataFrame(candles)
    print(f"  Got {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Calculate indicators
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']

    return df

def check_rsi_30_70(df, i):
    """RSI 30/70 reversal strategy"""
    if i < 1:
        return None

    rsi = df.iloc[i]['rsi']
    prev_rsi = df.iloc[i-1]['rsi']

    if prev_rsi < 30 and rsi >= 30 and df.iloc[i]['is_green']:
        return 'BUY'
    if prev_rsi > 70 and rsi <= 70 and df.iloc[i]['is_red']:
        return 'SELL'
    return None

def check_strong_trend(df, i):
    """Strong trend pullback strategy"""
    ema9 = df.iloc[i]['ema9']
    ema21 = df.iloc[i]['ema21']
    ema50 = df.iloc[i]['ema50']
    rsi = df.iloc[i]['rsi']

    if ema9 > ema21 > ema50:
        if 35 <= rsi <= 50 and df.iloc[i]['is_green']:
            return 'BUY'
    if ema9 < ema21 < ema50:
        if 50 <= rsi <= 65 and df.iloc[i]['is_red']:
            return 'SELL'
    return None

def backtest_pair(instrument, df, settings):
    """Backtest a single pair"""
    strategy = settings['strategy']
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    pip_mult = settings['pip_mult']

    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult

    trades = []
    cooldown = 0

    for i in range(50, len(df) - 100):
        if i < cooldown:
            continue

        if strategy == 'RSI_30_70':
            signal = check_rsi_30_70(df, i)
        else:
            signal = check_strong_trend(df, i)

        if not signal:
            continue

        entry = df.iloc[i]['close']
        entry_time = df.iloc[i]['time']

        for j in range(i + 1, min(i + 200, len(df))):
            h = df.iloc[j]['high']
            l = df.iloc[j]['low']

            if signal == 'BUY':
                if h >= entry + tp_dist:
                    trades.append({'result': 'WIN', 'pips': tp_pips, 'direction': 'BUY', 'entry_time': entry_time})
                    cooldown = j + 30
                    break
                if l <= entry - sl_dist:
                    trades.append({'result': 'LOSS', 'pips': -sl_pips, 'direction': 'BUY', 'entry_time': entry_time})
                    cooldown = j + 30
                    break
            else:
                if l <= entry - tp_dist:
                    trades.append({'result': 'WIN', 'pips': tp_pips, 'direction': 'SELL', 'entry_time': entry_time})
                    cooldown = j + 30
                    break
                if h >= entry + sl_dist:
                    trades.append({'result': 'LOSS', 'pips': -sl_pips, 'direction': 'SELL', 'entry_time': entry_time})
                    cooldown = j + 30
                    break

    return trades

# Run backtest for all pairs
all_trades = []
pair_results = {}

for instrument, settings in PAIR_SETTINGS.items():
    print(f"\nBacktesting {instrument} ({settings['strategy']})...")

    df = fetch_oanda_data(instrument, count=5000)
    if df is None or len(df) < 100:
        print(f"  Skipping {instrument} - insufficient data")
        pair_results[instrument] = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0}
        continue

    trades = backtest_pair(instrument, df, settings)
    all_trades.extend(trades)

    if trades:
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        total_pips = sum(t['pips'] for t in trades)
        win_rate = wins / len(trades) * 100

        pair_results[instrument] = {
            'trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate': win_rate,
            'total_pips': total_pips
        }
        print(f"  Trades: {len(trades)} | Wins: {wins} ({win_rate:.1f}%) | P&L: {total_pips:+.0f} pips")
    else:
        print(f"  No trades")
        pair_results[instrument] = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0}

# Summary
print("\n" + "=" * 70)
print("BACKTEST SUMMARY - LIVE OANDA DATA (~3-4 days)")
print("=" * 70)
print(f"\n{'PAIR':<10} {'STRATEGY':<15} {'TP':<4} {'SL':<4} {'TRADES':<8} {'WIN%':<8} {'P&L'}")
print("-" * 70)

total_trades = 0
total_wins = 0
total_pips = 0

for instrument, settings in PAIR_SETTINGS.items():
    r = pair_results[instrument]
    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']

    print(f"{instrument:<10} {settings['strategy']:<15} {settings['tp_pips']:<4} {settings['sl_pips']:<4} "
          f"{r['trades']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:+.0f}p")

print("-" * 70)
overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {'':<15} {'':<4} {'':<4} {total_trades:<8} {overall_wr:<7.1f}% {total_pips:+.0f}p")
print("=" * 70)

# Profit factor
if all_trades:
    wins_pips = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    losses_pips = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    profit_factor = wins_pips / losses_pips if losses_pips > 0 else 0

    print(f"\nTotal Winning Pips: +{wins_pips:.0f}")
    print(f"Total Losing Pips: -{losses_pips:.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Net P&L: {total_pips:+.0f} pips (${total_pips:+.2f} at $1/pip)")
