"""
Strategy Optimizer - Find BEST strategy for EACH pair
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data")

# ALL 5 PAIRS
PAIRS = {
    'EUR_USD': ('EURUSD', 10000),
    'GBP_USD': ('GBPUSD', 10000),
    'USD_JPY': ('USDJPY', 100),
    'USD_CHF': ('USDCHF', 10000),
    'USD_CAD': ('USDCAD', 10000),
}

print('='*70)
print('STRATEGY OPTIMIZER - ALL 5 PAIRS')
print('Finding BEST strategy for EACH pair')
print('='*70)

def load_and_calc(pair):
    prefix, pip_mult = PAIRS[pair]
    df = pd.read_csv(DATA_DIR / f'{prefix}5.csv', sep='\t', names=['time','open','high','low','close','volume'])
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2*df['bb_std']
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    return df, pip_mult

def test_strategy(df, pip_mult, strategy_func, tp_pips, sl_pips, sample=50):
    trades = []
    cooldown = 0
    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult

    for i in range(100, len(df)-100, sample):
        if i < cooldown:
            continue
        signal = strategy_func(df, i)
        if not signal:
            continue
        entry = df.iloc[i]['close']
        for j in range(i+1, min(i+100, len(df))):
            h, l = df.iloc[j]['high'], df.iloc[j]['low']
            if signal == 'BUY':
                if h >= entry + tp_dist:
                    trades.append({'won': True, 'pnl': tp_pips})
                    cooldown = j + 5
                    break
                if l <= entry - sl_dist:
                    trades.append({'won': False, 'pnl': -sl_pips})
                    cooldown = j + 5
                    break
            else:
                if l <= entry - tp_dist:
                    trades.append({'won': True, 'pnl': tp_pips})
                    cooldown = j + 5
                    break
                if h >= entry + sl_dist:
                    trades.append({'won': False, 'pnl': -sl_pips})
                    cooldown = j + 5
                    break

    if len(trades) < 20:
        return None
    wins = sum(1 for t in trades if t['won'])
    pnl = sum(t['pnl'] for t in trades)
    return {'trades': len(trades), 'wins': wins, 'wr': wins/len(trades)*100, 'pnl': pnl}

# Strategy definitions
def strat_rsi_30_70(df, i):
    rsi = df.iloc[i]['rsi']
    prev_rsi = df.iloc[i-1]['rsi']
    if prev_rsi < 30 and rsi >= 30 and df.iloc[i]['is_green']:
        return 'BUY'
    if prev_rsi > 70 and rsi <= 70 and df.iloc[i]['is_red']:
        return 'SELL'
    return None

def strat_rsi_25_75(df, i):
    rsi = df.iloc[i]['rsi']
    prev_rsi = df.iloc[i-1]['rsi']
    if prev_rsi < 25 and rsi >= 25 and df.iloc[i]['is_green']:
        return 'BUY'
    if prev_rsi > 75 and rsi <= 75 and df.iloc[i]['is_red']:
        return 'SELL'
    return None

def strat_ema_cross(df, i):
    if df.iloc[i-1]['ema9'] < df.iloc[i-1]['ema21'] and df.iloc[i]['ema9'] >= df.iloc[i]['ema21']:
        return 'BUY'
    if df.iloc[i-1]['ema9'] > df.iloc[i-1]['ema21'] and df.iloc[i]['ema9'] <= df.iloc[i]['ema21']:
        return 'SELL'
    return None

def strat_trend_rsi(df, i):
    ema_trend = 1 if df.iloc[i]['ema9'] > df.iloc[i]['ema21'] else -1
    rsi = df.iloc[i]['rsi']
    if ema_trend == 1 and 40 <= rsi <= 55 and df.iloc[i]['is_green']:
        return 'BUY'
    if ema_trend == -1 and 45 <= rsi <= 60 and df.iloc[i]['is_red']:
        return 'SELL'
    return None

def strat_strong_trend(df, i):
    if df.iloc[i]['ema9'] > df.iloc[i]['ema21'] > df.iloc[i]['ema50']:
        rsi = df.iloc[i]['rsi']
        if 35 <= rsi <= 50 and df.iloc[i]['is_green']:
            return 'BUY'
    if df.iloc[i]['ema9'] < df.iloc[i]['ema21'] < df.iloc[i]['ema50']:
        rsi = df.iloc[i]['rsi']
        if 50 <= rsi <= 65 and df.iloc[i]['is_red']:
            return 'SELL'
    return None

def strat_bb_bounce(df, i):
    close = df.iloc[i]['close']
    if close < df.iloc[i]['bb_lower'] and df.iloc[i]['is_green']:
        return 'BUY'
    if close > df.iloc[i]['bb_upper'] and df.iloc[i]['is_red']:
        return 'SELL'
    return None

strategies = [
    ('RSI_30_70', strat_rsi_30_70),
    ('RSI_25_75', strat_rsi_25_75),
    ('EMA_Cross', strat_ema_cross),
    ('Trend_RSI', strat_trend_rsi),
    ('Strong_Trend', strat_strong_trend),
    ('BB_Bounce', strat_bb_bounce),
]

tp_sl_combos = [
    (5, 15), (5, 20), (6, 12), (8, 16), (8, 24), (10, 20),
]

# Store best for each pair
best_per_pair = {}

for pair in PAIRS.keys():
    print(f'\nTesting {pair}...')
    df, pip_mult = load_and_calc(pair)

    best = None
    for strat_name, strat_func in strategies:
        for tp, sl in tp_sl_combos:
            r = test_strategy(df, pip_mult, strat_func, tp, sl)
            if r:
                if best is None or r['wr'] > best['wr']:
                    best = {
                        'pair': pair,
                        'strategy': strat_name,
                        'tp': tp,
                        'sl': sl,
                        **r
                    }

    if best:
        best_per_pair[pair] = best
        status = "70%+" if best['wr'] >= 70 else "<70%"
        print(f"  BEST: {best['strategy']} | TP={best['tp']} SL={best['sl']} | {best['wr']:.1f}% win | {best['trades']} trades | {best['pnl']:+.0f} pips [{status}]")

# Summary
print('\n' + '='*70)
print('BEST STRATEGY FOR EACH PAIR')
print('='*70)
print(f"{'PAIR':<10} {'STRATEGY':<15} {'TP':<4} {'SL':<4} {'WIN%':<7} {'TRADES':<8} {'P&L':<10} {'STATUS'}")
print('-'*70)

total_pnl = 0
for pair in PAIRS.keys():
    if pair in best_per_pair:
        b = best_per_pair[pair]
        status = '[70%+ OK]' if b['wr'] >= 70 else '[< 70%]'
        print(f"{pair:<10} {b['strategy']:<15} {b['tp']:<4} {b['sl']:<4} {b['wr']:<6.1f}% {b['trades']:<8} {b['pnl']:+.0f}p      {status}")
        total_pnl += b['pnl']

print('-'*70)
print(f'Total P&L if trading all pairs with best settings: {total_pnl:+.0f} pips')
print('='*70)
