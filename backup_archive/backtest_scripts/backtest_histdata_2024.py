"""
Backtest the Optimized Strategy with HistData 1-minute data
Uses the same strategy logic from optimized_strategy.py
Configuration loaded from optimized_paper_config.py
Data Period: 2024 (full year)
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("BACKTEST INITIALIZATION")
print("=" * 70)
print(f"[{datetime.now()}] Starting backtest script...")
print(f"[{datetime.now()}] Python version: {sys.version}")
print(f"[{datetime.now()}] Working directory: {os.getcwd()}")

# Import config (shared with paper trading)
print(f"\n[{datetime.now()}] Loading config module...")
try:
    from trading_system.Forex_Trading.config import optimized_paper_config as config
    print(f"[{datetime.now()}] Config loaded successfully")
    print(f"[{datetime.now()}] OPTIMIZED_PAIRS: {config.OPTIMIZED_PAIRS}")
except Exception as e:
    print(f"[{datetime.now()}] ERROR loading config: {e}")
    sys.exit(1)

# Import strategy logic
print(f"\n[{datetime.now()}] Loading strategy module...")
try:
    from trading_system.Forex_Trading.strategies.optimized_strategy import (
        calculate_indicators,
        get_signal,
        print_strategy_info
    )
    print(f"[{datetime.now()}] Strategy module loaded successfully")
except Exception as e:
    print(f"[{datetime.now()}] ERROR loading strategy: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Data directory for HistData
HISTDATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024')
print(f"\n[{datetime.now()}] Data directory: {HISTDATA_DIR}")
print(f"[{datetime.now()}] Directory exists: {HISTDATA_DIR.exists()}")

# Map our pair format to HistData format
PAIR_TO_HISTDATA = {
    'EUR_USD': 'EURUSD',
    'GBP_USD': 'GBPUSD',
    'USD_JPY': 'USDJPY',
    'USD_CHF': 'USDCHF',
    'USD_CAD': 'USDCAD',
    'NZD_USD': 'NZDUSD',
    'AUD_JPY': 'AUDJPY',
}

# Spread simulation (in pips) - OANDA typical spreads
# This makes backtest more realistic by accounting for bid/ask spread
SPREAD_PIPS = {
    'EUR_USD': 1.0,  # ~1 pip typical
    'GBP_USD': 1.2,  # slightly wider
    'USD_JPY': 1.0,  # ~1 pip typical
    'USD_CHF': 1.5,  # slightly wider
    'USD_CAD': 1.2,  # slightly wider
    'NZD_USD': 1.5,  # slightly wider
    'AUD_JPY': 1.8,  # JPY cross wider spread
}

# Set to True to simulate spread costs (more realistic)
SIMULATE_SPREAD = True

# Print config info
print("\n" + "=" * 70)
print("STRATEGY CONFIGURATION")
print("=" * 70)
print_strategy_info(config)
print("\nBACKTEST MODE - Using HistData 1-minute data (2024)")
print("=" * 70)


def load_histdata(pair):
    """Load 1-minute CSV data from HistData download"""
    histdata_pair = PAIR_TO_HISTDATA.get(pair, pair.replace('_', ''))
    csv_path = HISTDATA_DIR / f"DAT_MT_{histdata_pair}_M1_2024.csv"

    print(f"\n[{datetime.now()}] LOAD DATA: {pair}")
    print(f"  File path: {csv_path}")
    print(f"  File exists: {csv_path.exists()}")

    if not csv_path.exists():
        print(f"  ERROR: Data file not found!")
        return None

    # Get file size
    file_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")

    print(f"  Reading CSV...")
    # HistData format: Date,Time,Open,High,Low,Close,Volume
    # No header in file
    df = pd.read_csv(csv_path, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Sample data (first 3 rows):")
    print(df.head(3).to_string())

    # Create timestamp column
    print(f"  Creating timestamp column...")
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Calculate indicators using the same function as live trading
    print(f"  Calculating indicators (RSI, EMA, MACD, Bollinger)...")
    df = calculate_indicators(df)
    print(f"  Indicators calculated. Columns: {list(df.columns)}")

    # Show sample of calculated indicators
    print(f"  Sample indicators (last row):")
    last_row = df.iloc[-1]
    print(f"    RSI: {last_row.get('rsi', 'N/A'):.2f}" if 'rsi' in df.columns else "    RSI: N/A")
    print(f"    EMA20: {last_row.get('ema_20', 'N/A')}" if 'ema_20' in df.columns else "    EMA20: N/A")

    print(f"  Data load COMPLETE for {pair}")
    return df


def backtest_pair(pair):
    """Backtest a single pair - OPTIMIZED for speed using numpy arrays"""
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {pair}")
    print(f"{'='*70}")

    df = load_histdata(pair)

    if df is None or len(df) < 100:
        print(f"  ERROR: Insufficient data for {pair}")
        return []

    settings = config.get_pair_settings(pair)
    print(f"\n[{datetime.now()}] PAIR SETTINGS:")
    print(f"  Strategy: {settings['strategy']}")
    print(f"  TP: {settings['tp_pips']} pips")
    print(f"  SL: {settings['sl_pips']} pips")
    if settings['strategy'] == 'RSI_30_70' and ('rsi_oversold' in settings or 'rsi_overbought' in settings):
        print(f"  RSI Oversold: {settings.get('rsi_oversold', 30)}")
        print(f"  RSI Overbought: {settings.get('rsi_overbought', 70)}")

    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    pip_mult = config.PIP_MULTIPLIERS[pair]
    print(f"  Pip multiplier: {pip_mult}")

    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult

    # Calculate spread cost in price terms
    spread_pips = SPREAD_PIPS.get(pair, 1.0) if SIMULATE_SPREAD else 0
    spread_dist = spread_pips / pip_mult

    print(f"  TP distance (price): {tp_dist:.5f}")
    print(f"  SL distance (price): {sl_dist:.5f}")
    if SIMULATE_SPREAD:
        print(f"  Spread simulation: {spread_pips} pips ({spread_dist:.5f} in price)")
        print(f"  Effect: TP needs +{spread_pips}p more movement, SL triggers {spread_pips}p earlier")

    trades = []
    cooldown = 0
    signals_found = 0
    last_pct = 0

    # Pre-extract numpy arrays for FAST iteration (avoid df.iloc[] overhead)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    timestamps = df['timestamp'].values
    rsi_arr = df['rsi'].values
    ema9_arr = df['ema9'].values
    ema21_arr = df['ema21'].values
    ema50_arr = df['ema50'].values
    is_green_arr = df['is_green'].values
    is_red_arr = df['is_red'].values
    macd_arr = df['macd'].values
    macd_signal_arr = df['macd_signal'].values

    total_bars = len(df) - 150
    print(f"\n[{datetime.now()}] STARTING BACKTEST LOOP (OPTIMIZED)")
    print(f"  Total bars to process: {total_bars:,}")
    print(f"  Start index: 50")
    print(f"  End index: {len(df) - 100}")

    for i in range(50, len(df) - 100):
        # Progress logging every 10%
        pct = int((i - 50) / total_bars * 100)
        if pct >= last_pct + 10:
            print(f"  [{datetime.now()}] {pair}: {pct}% | Bar {i:,}/{len(df):,} | Signals: {signals_found} | Trades: {len(trades)}")
            last_pct = pct

        if i < cooldown:
            continue

        # FAST signal check using pre-extracted numpy arrays (100x faster than df.iloc)
        signal = None
        reason = ""

        rsi = rsi_arr[i]
        prev_rsi = rsi_arr[i-1]

        if settings['strategy'] == 'RSI_30_70':
            # RSI Reversal with custom thresholds (default 30/70)
            oversold = settings.get('rsi_oversold', 30)
            overbought = settings.get('rsi_overbought', 70)

            if prev_rsi < oversold and rsi >= oversold and is_green_arr[i]:
                signal = 'BUY'
                reason = f"RSI crossed UP through {oversold} ({prev_rsi:.1f} -> {rsi:.1f}) + Green candle"
            elif prev_rsi > overbought and rsi <= overbought and is_red_arr[i]:
                signal = 'SELL'
                reason = f"RSI crossed DOWN through {overbought} ({prev_rsi:.1f} -> {rsi:.1f}) + Red candle"

        elif settings['strategy'] == 'STRONG_TREND':
            # Strong Trend Pullback: trade pullbacks in EMA-aligned trends
            ema9 = ema9_arr[i]
            ema21 = ema21_arr[i]
            ema50 = ema50_arr[i]

            if ema9 > ema21 > ema50:  # Strong uptrend
                if 35 <= rsi <= 50 and is_green_arr[i]:
                    signal = 'BUY'
                    reason = f"Strong uptrend pullback (RSI: {rsi:.1f}, EMA9>EMA21>EMA50)"
            elif ema9 < ema21 < ema50:  # Strong downtrend
                if 50 <= rsi <= 65 and is_red_arr[i]:
                    signal = 'SELL'
                    reason = f"Strong downtrend bounce (RSI: {rsi:.1f}, EMA9<EMA21<EMA50)"

        elif settings['strategy'] == 'MACD_CROSS':
            # MACD Crossover: trade when MACD crosses signal line
            macd = macd_arr[i]
            macd_signal = macd_signal_arr[i]
            prev_macd = macd_arr[i-1]
            prev_macd_signal = macd_signal_arr[i-1]

            if prev_macd <= prev_macd_signal and macd > macd_signal and is_green_arr[i]:
                signal = 'BUY'
                reason = f"MACD crossed ABOVE signal ({macd:.5f} > {macd_signal:.5f}) + Green candle"
            elif prev_macd >= prev_macd_signal and macd < macd_signal and is_red_arr[i]:
                signal = 'SELL'
                reason = f"MACD crossed BELOW signal ({macd:.5f} < {macd_signal:.5f}) + Red candle"

        if signal is None:
            continue

        signals_found += 1
        entry = closes[i]
        entry_time = timestamps[i]

        # Log first few signals in detail
        if signals_found <= 3:
            print(f"\n  [SIGNAL #{signals_found}] {signal} at {entry_time}")
            print(f"    Entry price: {entry:.5f}")
            print(f"    Reason: {reason}")
            print(f"    TP target: {entry + tp_dist:.5f}" if signal == 'BUY' else f"    TP target: {entry - tp_dist:.5f}")
            print(f"    SL target: {entry - sl_dist:.5f}" if signal == 'BUY' else f"    SL target: {entry + sl_dist:.5f}")

        # Simulate trade outcome - look ahead up to 200 bars (200 minutes max)
        # Using numpy arrays for fast access
        trade_closed = False
        max_j = min(i + 200, len(highs))

        if signal == 'BUY':
            tp_target = entry + tp_dist + spread_dist
            sl_target = entry - sl_dist
            net_win = tp_pips - spread_pips
            net_loss = -sl_pips - spread_pips

            for j in range(i + 1, max_j):
                h = highs[j]
                l = lows[j]

                if h >= tp_target:
                    trades.append({
                        'pair': pair, 'direction': 'BUY', 'entry_time': entry_time,
                        'entry': entry, 'exit': tp_target, 'pips': net_win, 'result': 'WIN'
                    })
                    cooldown = j + 30
                    trade_closed = True
                    if signals_found <= 3:
                        print(f"    Result: WIN (+{net_win:.1f} pips after spread) after {j-i} bars")
                    break
                if l <= sl_target:
                    trades.append({
                        'pair': pair, 'direction': 'BUY', 'entry_time': entry_time,
                        'entry': entry, 'exit': sl_target, 'pips': net_loss, 'result': 'LOSS'
                    })
                    cooldown = j + 30
                    trade_closed = True
                    if signals_found <= 3:
                        print(f"    Result: LOSS ({net_loss:.1f} pips with spread) after {j-i} bars")
                    break
        else:  # SELL
            tp_target = entry - tp_dist - spread_dist
            sl_target = entry + sl_dist
            net_win = tp_pips - spread_pips
            net_loss = -sl_pips - spread_pips

            for j in range(i + 1, max_j):
                h = highs[j]
                l = lows[j]

                if l <= tp_target:
                    trades.append({
                        'pair': pair, 'direction': 'SELL', 'entry_time': entry_time,
                        'entry': entry, 'exit': tp_target, 'pips': net_win, 'result': 'WIN'
                    })
                    cooldown = j + 30
                    trade_closed = True
                    if signals_found <= 3:
                        print(f"    Result: WIN (+{net_win:.1f} pips after spread) after {j-i} bars")
                    break
                if h >= sl_target:
                    trades.append({
                        'pair': pair, 'direction': 'SELL', 'entry_time': entry_time,
                        'entry': entry, 'exit': sl_target, 'pips': net_loss, 'result': 'LOSS'
                    })
                    cooldown = j + 30
                    trade_closed = True
                    if signals_found <= 3:
                        print(f"    Result: LOSS ({net_loss:.1f} pips with spread) after {j-i} bars")
                    break

        if not trade_closed and signals_found <= 3:
            print(f"    Result: EXPIRED (no TP/SL hit in 200 bars)")

    print(f"\n[{datetime.now()}] {pair} BACKTEST COMPLETE")
    print(f"  Total signals found: {signals_found}")
    print(f"  Total trades closed: {len(trades)}")
    if trades:
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        print(f"  Wins: {wins} | Losses: {len(trades) - wins}")
        print(f"  Win rate: {wins/len(trades)*100:.1f}%")
        print(f"  Total P&L: {sum(t['pips'] for t in trades):+.0f} pips")

    return trades


# Check if data exists
print("\n" + "=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)
print(f"Scanning directory: {HISTDATA_DIR}")
data_files = list(HISTDATA_DIR.glob("DAT_MT_*_M1_2024.csv"))

if not data_files:
    print(f"\nERROR: No data files found in {HISTDATA_DIR}")
    print("Please download M1 data from HistData.com for 2024")
    sys.exit(1)

print(f"Found {len(data_files)} data files:")
for f in data_files:
    print(f"  {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

# Run backtest for all pairs
print("\n" + "=" * 70)
print("RUNNING BACKTESTS")
print("=" * 70)
all_trades = []
pair_results = {}

for pair in config.OPTIMIZED_PAIRS:
    print(f"\n>>> Processing {pair}...")
    trades = backtest_pair(pair)
    all_trades.extend(trades)

    if trades:
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        total_pips = sum(t['pips'] for t in trades)
        win_rate = wins / len(trades) * 100

        pair_results[pair] = {
            'trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate': win_rate,
            'total_pips': total_pips
        }
    else:
        pair_results[pair] = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0}

# Summary
print("\n" + "=" * 70)
print("BACKTEST SUMMARY - OPTIMIZED STRATEGY (HISTDATA 2024 1-MIN DATA)")
if SIMULATE_SPREAD:
    print("*** SPREAD SIMULATION ENABLED (~1 pip OANDA typical) ***")
else:
    print("*** NO SPREAD SIMULATION (results may be optimistic) ***")
print("=" * 70)
print(f"\n{'PAIR':<10} {'STRATEGY':<15} {'TP':<4} {'SL':<4} {'TRADES':<8} {'WIN%':<8} {'P&L'}")
print("-" * 70)

total_trades = 0
total_wins = 0
total_pips = 0

for pair in config.OPTIMIZED_PAIRS:
    settings = config.get_pair_settings(pair)
    r = pair_results[pair]
    total_trades += r['trades']
    total_wins += r['wins']
    total_pips += r['total_pips']

    print(f"{pair:<10} {settings['strategy']:<15} {settings['tp_pips']:<4} {settings['sl_pips']:<4} "
          f"{r['trades']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:+.0f}p")

print("-" * 70)
overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
print(f"{'TOTAL':<10} {'':<15} {'':<4} {'':<4} {total_trades:<8} {overall_wr:<7.1f}% {total_pips:+.0f}p")
print("=" * 70)

# Profit factor and additional stats
if all_trades:
    wins_pips = sum(t['pips'] for t in all_trades if t['pips'] > 0)
    losses_pips = abs(sum(t['pips'] for t in all_trades if t['pips'] < 0))
    profit_factor = wins_pips / losses_pips if losses_pips > 0 else 0

    print(f"\nTotal Winning Pips: +{wins_pips:.0f}")
    print(f"Total Losing Pips: -{losses_pips:.0f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Net P&L: {total_pips:+.0f} pips (${total_pips:+.2f} at $1/pip)")

    # Monthly breakdown
    print("\n" + "=" * 70)
    print("MONTHLY BREAKDOWN")
    print("=" * 70)

    df_trades = pd.DataFrame(all_trades)
    df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M')

    for month in sorted(df_trades['month'].unique()):
        month_trades = df_trades[df_trades['month'] == month]
        month_wins = (month_trades['result'] == 'WIN').sum()
        month_pips = month_trades['pips'].sum()
        month_wr = month_wins / len(month_trades) * 100
        print(f"{month}: {len(month_trades):>3} trades | {month_wr:>5.1f}% win | {month_pips:>+6.0f} pips")

    # Trade distribution
    print("\n" + "=" * 70)
    print("TRADE DISTRIBUTION")
    print("=" * 70)

    buy_count = sum(1 for t in all_trades if t['direction'] == 'BUY')
    sell_count = sum(1 for t in all_trades if t['direction'] == 'SELL')
    print(f"BUY trades: {buy_count} ({buy_count/len(all_trades)*100:.1f}%)")
    print(f"SELL trades: {sell_count} ({sell_count/len(all_trades)*100:.1f}%)")

    # By pair
    print(f"\nBy Pair:")
    for pair in config.OPTIMIZED_PAIRS:
        pair_trades = [t for t in all_trades if t['pair'] == pair]
        if pair_trades:
            pair_wins = sum(1 for t in pair_trades if t['result'] == 'WIN')
            print(f"  {pair}: {len(pair_trades)} trades, {pair_wins/len(pair_trades)*100:.1f}% win rate")

print(f"\n[{datetime.now()}] BACKTEST COMPLETE")
