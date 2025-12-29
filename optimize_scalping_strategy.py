"""
OPTIMIZATION: SCALPING STRATEGY - FIND BEST PARAMETERS
=======================================================

Test HUNDREDS of combinations to find what ACTUALLY works:
- TP: 5, 8, 10, 12, 15 pips
- SL: 15, 20, 25, 30 pips
- RSI zones: Different ranges (35-45, 40-50, 40-55, 40-60, 45-60)
- Cooldown: 0, 15, 30 minutes
- Session filter: ON/OFF
- Indicator requirements: ALL 5 vs relaxed

This will test ~300+ combinations PER PAIR to find actual winners.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product

# Import OANDA client
try:
    sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading' / 'engine'))
    from oanda_client import OandaClient
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

# Spread simulation
SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_USD': 1.3, 'USD_JPY': 1.0, 'EUR_GBP': 1.2,
    'EUR_JPY': 1.5, 'GBP_JPY': 2.0, 'AUD_JPY': 1.8,
}

# Test pairs
TEST_PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD', 
              'EUR_GBP', 'NZD_USD', 'AUD_USD', 'EUR_JPY', 'GBP_JPY']


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
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Candle patterns
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / df['candle_range'].replace(0, 1)
    
    return df


def check_signal_flexible(df: pd.DataFrame, rsi_low: int, rsi_high: int, 
                          require_macd: bool, require_strong_candle: bool) -> str:
    """
    Flexible signal check with configurable requirements.
    """
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
    
    # Test BUY signal
    trend_up = ema9 > ema21 > ema50
    rsi_buy_zone = rsi_low <= rsi <= rsi_high
    macd_bullish = macd > macd_signal and macd_hist > prev_hist
    strong_green = current['is_green'] and (not require_strong_candle or body_ratio > 0.4)
    
    if trend_up and rsi_buy_zone and strong_green:
        if not require_macd or macd_bullish:
            return 'BUY'
    
    # Test SELL signal
    trend_down = ema9 < ema21 < ema50
    rsi_sell_zone = rsi_low <= rsi <= rsi_high
    macd_bearish = macd < macd_signal and macd_hist < prev_hist
    strong_red = current['is_red'] and (not require_strong_candle or body_ratio > 0.4)
    
    if trend_down and rsi_sell_zone and strong_red:
        if not require_macd or macd_bearish:
            return 'SELL'
    
    return None


def backtest_config(df: pd.DataFrame, pair: str, config: dict) -> dict:
    """
    Backtest with specific configuration.
    """
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
            # Check exits
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
                trades.append({
                    'pnl': exit_pnl,
                    'won': exit_pnl > 0,
                    'exit_reason': exit_reason
                })
                in_position = False
                last_exit_time = current_time
        
        else:
            # Check entry
            
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
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0, 'avg_pips': 0}
    
    wins = sum(1 for t in trades if t['won'])
    total_pips = sum(t['pnl'] for t in trades)
    
    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips,
        'avg_pips': total_pips / len(trades)
    }


def fetch_oanda_data(pair: str) -> pd.DataFrame:
    """Fetch OANDA data."""
    if not OANDA_AVAILABLE:
        return pd.DataFrame()
    
    client = OandaClient()
    print(f"  Fetching data for {pair}...")
    candles = client.get_candles(pair, 'M1', count=5000)
    
    if not candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"  Got {len(df)} candles")
    return df


def optimize_pair(pair: str):
    """
    Optimize all parameters for a single pair.
    """
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZING {pair}")
    print(f"{'=' * 70}")
    
    # Fetch data
    df = fetch_oanda_data(pair)
    if len(df) < 500:
        print(f"  Insufficient data")
        return None
    
    # Calculate indicators once
    df = calculate_indicators(df)
    
    # Parameter combinations to test
    tp_options = [5, 8, 10, 12, 15]
    sl_options = [15, 20, 25, 30]
    cooldown_options = [0, 15, 30]
    session_options = [
        None,  # No filter (24/7)
        list(range(12, 20)),  # London/NY overlap
    ]
    rsi_options = [
        (35, 45),  # Tight low
        (40, 50),  # Medium low
        (40, 55),  # Medium
        (40, 60),  # Wide
        (45, 60),  # Medium high
    ]
    macd_options = [True, False]  # Require MACD or not
    candle_options = [True, False]  # Require strong candle or not
    
    all_configs = []
    
    # Generate all combinations
    for tp, sl, cd, session, rsi, macd_req, candle_req in product(
        tp_options, sl_options, cooldown_options, session_options,
        rsi_options, macd_options, candle_options
    ):
        # Skip bad risk:reward ratios
        if sl / tp > 4:  # Max 1:4 risk:reward
            continue
        
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
        
        # Filters for good results
        if result['trades'] >= 5 and result['win_rate'] >= 60 and result['total_pips'] > 0:
            all_configs.append({
                'config': config,
                'result': result
            })
    
    # Sort by total pips
    all_configs.sort(key=lambda x: x['result']['total_pips'], reverse=True)
    
    print(f"\n  Tested {len(list(product(tp_options, sl_options, cooldown_options, session_options, rsi_options, macd_options, candle_options)))} combinations")
    print(f"  Found {len(all_configs)} profitable configs (≥5 trades, ≥60% WR, +pips)")
    
    if not all_configs:
        print(f"  ❌ No winning configuration found!")
        return None
    
    # Show top 5
    print(f"\n  TOP 5 CONFIGURATIONS:")
    print(f"  {'Rank':<6} {'Pips':<8} {'Trades':<8} {'WR%':<8} {'TP':<5} {'SL':<5} {'CD':<5} {'SF':<5} {'RSI':<10} {'MACD':<6} {'Candle'}")
    print(f"  {'-' * 100}")
    
    for i, item in enumerate(all_configs[:5], 1):
        c = item['config']
        r = item['result']
        sf = 'ON' if c['session_hours'] else 'OFF'
        rsi_str = f"{c['rsi_low']}-{c['rsi_high']}"
        macd_str = 'REQ' if c['require_macd'] else 'OPT'
        candle_str = 'REQ' if c['require_strong_candle'] else 'OPT'
        
        print(f"  {i:<6} {r['total_pips']:>+7.0f} {r['trades']:<8} {r['win_rate']:<7.1f}% "
              f"{c['tp_pips']:<5} {c['sl_pips']:<5} {c['cooldown_minutes']:<5} {sf:<5} "
              f"{rsi_str:<10} {macd_str:<6} {candle_str}")
    
    return all_configs[0]  # Return best config


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================
print("=" * 80)
print("SCALPING STRATEGY OPTIMIZATION - FIND BEST PARAMETERS")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"\nTesting parameters:")
print(f"  TP: 5, 8, 10, 12, 15 pips")
print(f"  SL: 15, 20, 25, 30 pips")
print(f"  Cooldown: 0, 15, 30 minutes")
print(f"  Session: OFF, ON (12-20 UTC)")
print(f"  RSI zones: (35-45), (40-50), (40-55), (40-60), (45-60)")
print(f"  MACD: Required vs Optional")
print(f"  Candle: Strong body vs Any")
print(f"\nFilters:")
print(f"  - Minimum 5 trades")
print(f"  - Minimum 60% win rate")
print(f"  - Positive total pips")
print(f"  - Max risk:reward 1:4")
print("=" * 80)

best_configs = {}

for pair in TEST_PAIRS:
    best = optimize_pair(pair)
    if best:
        best_configs[pair] = best

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE - BEST CONFIGURATION PER PAIR")
print("=" * 80)

if not best_configs:
    print("\n❌ NO PROFITABLE CONFIGURATIONS FOUND!")
    print("\nThe multi-indicator scalping strategy is TOO STRICT.")
    print("Consider:")
    print("  - Using the existing optimized_strategy.py instead")
    print("  - Relaxing indicator requirements")
    print("  - Using longer TP targets (15-20 pips)")
else:
    print(f"\n✅ Found winning configurations for {len(best_configs)}/{len(TEST_PAIRS)} pairs\n")
    
    for pair, item in best_configs.items():
        c = item['config']
        r = item['result']
        sf = 'ON' if c['session_hours'] else 'OFF'
        macd_str = 'REQ' if c['require_macd'] else 'OPT'
        candle_str = 'REQ' if c['require_strong_candle'] else 'OPT'
        
        print(f"{pair}:")
        print(f"  Performance: {r['total_pips']:+.0f} pips | {r['trades']} trades | {r['win_rate']:.1f}% WR")
        print(f"  TP/SL: {c['tp_pips']}p / {c['sl_pips']}p | Cooldown: {c['cooldown_minutes']}m | Session: {sf}")
        print(f"  RSI: {c['rsi_low']}-{c['rsi_high']} | MACD: {macd_str} | Candle: {candle_str}")
        print()

print("=" * 80)
