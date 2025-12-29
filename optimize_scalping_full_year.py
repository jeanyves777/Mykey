"""
OPTIMIZATION: SCALPING STRATEGY - FULL YEAR 2024 DATA
======================================================

Test HUNDREDS of combinations on FULL YEAR 2024 to find what ACTUALLY works:
- TP: 5, 8, 10, 12, 15, 20 pips
- SL: 15, 20, 25, 30 pips
- RSI zones: Different ranges
- Cooldown: 0, 15, 30 minutes
- Session filter: ON/OFF
- Indicator requirements: Different combinations

~300+ combinations PER PAIR on 370K+ candles = REAL optimization
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product

# Data directory
DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024")

# File map
FILE_MAP = {
    'EUR_USD': 'DAT_MT_EURUSD_M1_2024.csv',
    'GBP_USD': 'DAT_MT_GBPUSD_M1_2024.csv',
    'USD_CHF': 'DAT_MT_USDCHF_M1_2024.csv',
    'USD_CAD': 'DAT_MT_USDCAD_M1_2024.csv',
    'NZD_USD': 'DAT_MT_NZDUSD_M1_2024.csv',
    'AUD_JPY': 'DAT_MT_AUDJPY_M1_2024.csv',
    'USD_JPY': 'DAT_MT_USDJPY_M1_2024.csv',
    'EUR_GBP': 'DAT_MT_EURGBP_M1_2024.csv',
    'AUD_CHF': 'DAT_MT_AUDCHF_M1_2024.csv',
    'EUR_CAD': 'DAT_MT_EURCAD_M1_2024.csv',
}

SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}


def load_histdata(pair: str) -> pd.DataFrame:
    """Load HistData CSV."""
    filename = FILE_MAP.get(pair)
    if not filename:
        return pd.DataFrame()
    
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return pd.DataFrame()
    
    print(f"  Loading {filename}...")
    df = pd.read_csv(filepath, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df = df.drop(columns=['date', 'volume'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"  Loaded {len(df):,} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators."""
    df = df.copy()
    
    # EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Candle patterns
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / df['candle_range'].replace(0, 1)
    
    return df


def check_signal_flexible(df: pd.DataFrame, rsi_low: int, rsi_high: int, 
                          require_macd: bool, require_strong_candle: bool) -> str:
    """Flexible signal check."""
    if len(df) < 2:
        return None
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']
    rsi = current['rsi']
    macd = current['macd']
    macd_signal = current['macd_signal']
    macd_hist = current['macd_hist']
    prev_hist = prev['macd_hist']
    body_ratio = current['body_ratio']
    
    # BUY signal
    trend_up = ema9 > ema21 > ema50
    rsi_buy_zone = rsi_low <= rsi <= rsi_high
    macd_bullish = macd > macd_signal and macd_hist > prev_hist
    strong_green = current['is_green'] and (not require_strong_candle or body_ratio > 0.4)
    
    if trend_up and rsi_buy_zone and strong_green:
        if not require_macd or macd_bullish:
            return 'BUY'
    
    # SELL signal
    trend_down = ema9 < ema21 < ema50
    rsi_sell_zone = rsi_low <= rsi <= rsi_high
    macd_bearish = macd < macd_signal and macd_hist < prev_hist
    strong_red = current['is_red'] and (not require_strong_candle or body_ratio > 0.4)
    
    if trend_down and rsi_sell_zone and strong_red:
        if not require_macd or macd_bearish:
            return 'SELL'
    
    return None


def backtest_config(df: pd.DataFrame, pair: str, config: dict) -> dict:
    """Backtest with specific configuration."""
    tp_pips = config['tp_pips']
    sl_pips = config['sl_pips']
    cooldown_minutes = config['cooldown_minutes']
    session_hours = config['session_hours']
    rsi_low = config['rsi_low']
    rsi_high = config['rsi_high']
    require_macd = config['require_macd']
    require_strong_candle = config['require_strong_candle']
    
    pip_mult = 100 if 'JPY' in pair else 10000
    tp_price_diff = tp_pips / pip_mult
    sl_price_diff = sl_pips / pip_mult
    spread = SPREAD_PIPS.get(pair, 1.5) / pip_mult
    
    trades = []
    in_position = False
    entry_price = 0
    direction = None
    tp_price = 0
    sl_price = 0
    entry_time = None
    last_exit_time = None
    
    for i in range(200, len(df)):
        row = df.iloc[i]
        current_time = row['time']
        hour_utc = current_time.hour
        
        if in_position:
            high = row['high']
            low = row['low']
            
            exit_reason = None
            exit_pnl = 0
            
            if direction == 'BUY':
                if low <= sl_price:
                    exit_reason = 'SL'
                    exit_pnl = -sl_pips
                elif high >= tp_price:
                    exit_reason = 'TP'
                    exit_pnl = tp_pips
            else:
                if high >= sl_price:
                    exit_reason = 'SL'
                    exit_pnl = -sl_pips
                elif low <= tp_price:
                    exit_reason = 'TP'
                    exit_pnl = tp_pips
            
            if exit_reason:
                trades.append({'pnl': exit_pnl, 'won': exit_pnl > 0})
                in_position = False
                last_exit_time = current_time
        
        else:
            # Session filter
            if session_hours and hour_utc not in session_hours:
                continue
            
            # Cooldown
            if cooldown_minutes > 0 and last_exit_time:
                time_since = (current_time - last_exit_time).total_seconds() / 60
                if time_since < cooldown_minutes:
                    continue
            
            # Get signal
            signal = check_signal_flexible(
                df.iloc[:i+1], 
                rsi_low, 
                rsi_high,
                require_macd,
                require_strong_candle
            )
            
            if signal:
                in_position = True
                direction = signal
                entry_time = current_time
                
                if signal == 'BUY':
                    entry_price = row['close'] + spread
                    tp_price = entry_price + tp_price_diff
                    sl_price = entry_price - sl_price_diff
                else:
                    entry_price = row['close'] - spread
                    tp_price = entry_price - tp_price_diff
                    sl_price = entry_price + sl_price_diff
    
    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0}
    
    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    
    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
    }


def optimize_pair(pair: str):
    """Optimize all parameters for a single pair."""
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZING {pair} - FULL YEAR 2024")
    print(f"{'=' * 70}")
    
    df = load_histdata(pair)
    if len(df) < 1000:
        print(f"  Insufficient data")
        return None
    
    print(f"  Calculating indicators...")
    df = calculate_indicators(df)
    
    # Parameter combinations
    tp_options = [5, 8, 10, 12, 15, 20]
    sl_options = [15, 20, 25, 30]
    cooldown_options = [0, 15, 30]
    session_options = [
        None,  # 24/7
        list(range(12, 20)),  # London/NY
    ]
    rsi_options = [
        (30, 70),  # Classic oversold/overbought
        (35, 65),  # Slightly tighter
        (40, 60),  # Medium
        (35, 45),  # Pullback low
        (55, 65),  # Pullback high
    ]
    macd_options = [True, False]
    candle_options = [True, False]
    
    print(f"  Testing combinations...")
    
    best_configs = []
    total_tested = 0
    
    for tp, sl, cd, session, rsi, macd_req, candle_req in product(
        tp_options, sl_options, cooldown_options, session_options,
        rsi_options, macd_options, candle_options
    ):
        # Skip bad risk:reward
        if sl / tp > 4:
            continue
        
        total_tested += 1
        
        config = {
            'tp_pips': tp,
            'sl_pips': sl,
            'cooldown_minutes': cd,
            'session_hours': session,
            'rsi_low': rsi[0],
            'rsi_high': rsi[1],
            'require_macd': macd_req,
            'require_strong_candle': candle_req,
        }
        
        result = backtest_config(df, pair, config)
        
        # Filter: min 20 trades, 65% WR, positive pips
        if result['trades'] >= 20 and result['win_rate'] >= 65 and result['total_pips'] > 0:
            best_configs.append({
                'config': config,
                'result': result
            })
        
        # Progress update
        if total_tested % 100 == 0:
            print(f"    Tested {total_tested} combinations, found {len(best_configs)} winners...")
    
    best_configs.sort(key=lambda x: x['result']['total_pips'], reverse=True)
    
    print(f"\n  ✅ Tested {total_tested} combinations")
    print(f"  ✅ Found {len(best_configs)} profitable configs (≥20 trades, ≥65% WR, +pips)")
    
    if not best_configs:
        print(f"  ❌ No winning configuration found!")
        return None
    
    # Show top 5
    print(f"\n  TOP 5 CONFIGURATIONS:")
    print(f"  {'#':<3} {'Pips':<9} {'Trades':<8} {'WR%':<7} {'TP':<4} {'SL':<4} {'CD':<4} {'SF':<4} {'RSI':<9} {'MACD':<5} {'Cndl'}")
    print(f"  {'-' * 95}")
    
    for i, item in enumerate(best_configs[:5], 1):
        c = item['config']
        r = item['result']
        sf = 'ON' if c['session_hours'] else 'OFF'
        rsi_str = f"{c['rsi_low']}-{c['rsi_high']}"
        macd_str = 'Y' if c['require_macd'] else 'N'
        candle_str = 'Y' if c['require_strong_candle'] else 'N'
        
        print(f"  {i:<3} {r['total_pips']:>+8.0f} {r['trades']:<8} {r['win_rate']:<6.1f}% "
              f"{c['tp_pips']:<4} {c['sl_pips']:<4} {c['cooldown_minutes']:<4} {sf:<4} "
              f"{rsi_str:<9} {macd_str:<5} {candle_str}")
    
    return best_configs[0]


# =============================================================================
# MAIN
# =============================================================================
print("=" * 80)
print("SCALPING STRATEGY OPTIMIZATION - FULL YEAR 2024")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nData: {DATA_DIR}")
print(f"\nTesting ~300+ combinations per pair:")
print(f"  TP: 5, 8, 10, 12, 15, 20 pips")
print(f"  SL: 15, 20, 25, 30 pips")
print(f"  Cooldown: 0, 15, 30 min")
print(f"  Session: OFF, ON (12-20 UTC)")
print(f"  RSI zones: (30-70), (35-65), (40-60), (35-45), (55-65)")
print(f"  MACD: Required / Optional")
print(f"  Candle: Strong / Any")
print(f"\nMinimum requirements:")
print(f"  - 20+ trades (statistical significance)")
print(f"  - 65%+ win rate")
print(f"  - Positive total pips")
print("=" * 80)

best_configs = {}

for pair in FILE_MAP.keys():
    best = optimize_pair(pair)
    if best:
        best_configs[pair] = best

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMIZATION RESULTS - BEST CONFIG PER PAIR")
print("=" * 80)

if not best_configs:
    print("\n❌ NO PROFITABLE CONFIGURATIONS FOUND!")
    print("\nThe scalping strategy requirements are too strict.")
    print("Recommendation: Use the existing optimized_strategy.py")
else:
    total_pips = sum(item['result']['total_pips'] for item in best_configs.values())
    total_trades = sum(item['result']['trades'] for item in best_configs.values())
    total_wins = sum(item['result']['wins'] for item in best_configs.values())
    
    print(f"\n✅ Found winning configs for {len(best_configs)}/{len(FILE_MAP)} pairs")
    print(f"\nAGGREGATE PERFORMANCE (if all pairs traded):")
    print(f"  Total Pips: {total_pips:+,.0f}")
    print(f"  Total Trades: {total_trades:,}")
    print(f"  Overall Win Rate: {total_wins/total_trades*100:.1f}%")
    print(f"  Avg per month: {total_pips/12:+,.0f} pips")
    print(f"\nPER PAIR BREAKDOWN:")
    print(f"{'PAIR':<10} {'PIPS':<10} {'TRADES':<8} {'WR%':<7} {'TP':<4} {'SL':<4} {'CD':<4} {'SF':<4} {'RSI':<9} {'MACD':<5} {'Cndl'}")
    print("-" * 95)
    
    for pair in sorted(best_configs.keys(), key=lambda p: best_configs[p]['result']['total_pips'], reverse=True):
        item = best_configs[pair]
        c = item['config']
        r = item['result']
        sf = 'ON' if c['session_hours'] else 'OFF'
        rsi_str = f"{c['rsi_low']}-{c['rsi_high']}"
        macd_str = 'Y' if c['require_macd'] else 'N'
        candle_str = 'Y' if c['require_strong_candle'] else 'N'
        
        print(f"{pair:<10} {r['total_pips']:>+9.0f} {r['trades']:<8} {r['win_rate']:<6.1f}% "
              f"{c['tp_pips']:<4} {c['sl_pips']:<4} {c['cooldown_minutes']:<4} {sf:<4} "
              f"{rsi_str:<9} {macd_str:<5} {candle_str}")

print("\n" + "=" * 80)
