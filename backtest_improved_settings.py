"""
Quick backtest to verify improved settings work better
Tests:
1. Wider TP for EUR_USD and USD_CAD (5 -> 8 pips)
2. Session filters (only trade good hours)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading'))
from config import improved_config as config

HISTDATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024')

PAIR_TO_HISTDATA = {
    'EUR_USD': 'EURUSD',
    'GBP_USD': 'GBPUSD',
    'USD_JPY': 'USDJPY',
    'USD_CHF': 'USDCHF',
    'USD_CAD': 'USDCAD',
    'NZD_USD': 'NZDUSD',
    'AUD_JPY': 'AUDJPY',
}

SPREAD_PIPS = {
    'EUR_USD': 1.0,
    'GBP_USD': 1.2,
    'USD_JPY': 1.0,
    'USD_CHF': 1.5,
    'USD_CAD': 1.2,
    'NZD_USD': 1.5,
    'AUD_JPY': 1.8,
}


def load_data(pair: str) -> pd.DataFrame:
    """Load and prepare data."""
    histdata_name = PAIR_TO_HISTDATA.get(pair)
    if not histdata_name:
        return pd.DataFrame()

    data_file = None
    for f in HISTDATA_DIR.glob(f'DAT_MT_{histdata_name}_M1_*.csv'):
        data_file = f
        break

    if not data_file:
        return pd.DataFrame()

    df = pd.read_csv(data_file, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df = df.drop(columns=['date', 'time'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Indicators
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['hour'] = df['timestamp'].dt.hour

    return df


def backtest_pair(pair: str, df: pd.DataFrame, use_session_filter: bool = True) -> dict:
    """Run backtest with improved settings."""
    if len(df) < 200:
        return {'trades': 0, 'wins': 0, 'pips': 0}

    settings = config.get_pair_settings(pair)
    pip_mult = config.PIP_MULTIPLIERS[pair]
    spread_pips = SPREAD_PIPS.get(pair, 1.0)

    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult
    spread_dist = spread_pips / pip_mult

    # Pre-extract arrays
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    rsi_arr = df['rsi'].values
    is_green_arr = df['is_green'].values
    is_red_arr = df['is_red'].values
    macd_arr = df['macd'].values
    macd_signal_arr = df['macd_signal'].values
    ema9_arr = df['ema9'].values
    ema21_arr = df['ema21'].values
    ema50_arr = df['ema50'].values
    hour_arr = df['hour'].values

    trades = 0
    wins = 0
    total_pips = 0

    for i in range(50, len(df) - 100):
        # Session filter
        if use_session_filter:
            hour = hour_arr[i]
            if not config.is_allowed_hour(pair, hour):
                continue

        # Signal check
        signal = None
        rsi = rsi_arr[i]
        prev_rsi = rsi_arr[i-1]

        if settings['strategy'] == 'RSI_30_70':
            oversold = settings.get('rsi_oversold', 30)
            overbought = settings.get('rsi_overbought', 70)

            if prev_rsi < oversold and rsi >= oversold and is_green_arr[i]:
                signal = 'BUY'
            elif prev_rsi > overbought and rsi <= overbought and is_red_arr[i]:
                signal = 'SELL'

        elif settings['strategy'] == 'MACD_CROSS':
            macd = macd_arr[i]
            macd_signal = macd_signal_arr[i]
            prev_macd = macd_arr[i-1]
            prev_macd_signal = macd_signal_arr[i-1]

            if prev_macd <= prev_macd_signal and macd > macd_signal and is_green_arr[i]:
                signal = 'BUY'
            elif prev_macd >= prev_macd_signal and macd < macd_signal and is_red_arr[i]:
                signal = 'SELL'

        elif settings['strategy'] == 'STRONG_TREND':
            ema9 = ema9_arr[i]
            ema21 = ema21_arr[i]
            ema50 = ema50_arr[i]

            if ema9 > ema21 > ema50 and 35 <= rsi <= 50 and is_green_arr[i]:
                signal = 'BUY'
            elif ema9 < ema21 < ema50 and 50 <= rsi <= 65 and is_red_arr[i]:
                signal = 'SELL'

        if signal is None:
            continue

        # Entry with spread
        entry = closes[i]
        if signal == 'BUY':
            entry += spread_dist
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            entry -= spread_dist
            tp = entry - tp_dist
            sl = entry + sl_dist

        # Simulate outcome
        for j in range(i + 1, min(i + 500, len(df))):
            high = highs[j]
            low = lows[j]

            if signal == 'BUY':
                if low <= sl:
                    trades += 1
                    total_pips -= sl_pips
                    break
                elif high >= tp:
                    trades += 1
                    wins += 1
                    total_pips += tp_pips
                    break
            else:
                if high >= sl:
                    trades += 1
                    total_pips -= sl_pips
                    break
                elif low <= tp:
                    trades += 1
                    wins += 1
                    total_pips += tp_pips
                    break

    return {
        'trades': trades,
        'wins': wins,
        'losses': trades - wins,
        'pips': total_pips,
        'win_rate': (wins / trades * 100) if trades > 0 else 0
    }


if __name__ == '__main__':
    print("=" * 80)
    print("BACKTEST: IMPROVED SETTINGS vs OLD SETTINGS")
    print("=" * 80)
    print(f"\nKey Changes:")
    print("  - EUR_USD: TP 5 -> 8 pips")
    print("  - USD_CAD: TP 5 -> 8 pips")
    print("  - Session filters: Trade only during high win-rate hours")
    print("  - USD_JPY & AUD_JPY: Changed from MACD to RSI_30_70")

    # Test problem pairs first
    problem_pairs = ['EUR_USD', 'USD_CAD', 'USD_JPY', 'AUD_JPY']
    good_pairs = ['NZD_USD', 'GBP_USD', 'USD_CHF']

    print("\n" + "=" * 80)
    print("PROBLEM PAIRS - WITH vs WITHOUT SESSION FILTER")
    print("=" * 80)

    for pair in problem_pairs:
        print(f"\n[Loading {pair}...]")
        df = load_data(pair)
        if len(df) == 0:
            continue

        settings = config.get_pair_settings(pair)

        # Without filter (all hours)
        result_all = backtest_pair(pair, df, use_session_filter=False)

        # With session filter
        result_filtered = backtest_pair(pair, df, use_session_filter=True)

        print(f"\n{pair} (TP: {settings['tp_pips']}p, SL: {settings['sl_pips']}p)")
        print(f"  ALL HOURS:     {result_all['trades']:,} trades | {result_all['win_rate']:.1f}% WR | {result_all['pips']:+,.0f} pips")
        print(f"  FILTERED:      {result_filtered['trades']:,} trades | {result_filtered['win_rate']:.1f}% WR | {result_filtered['pips']:+,.0f} pips")

        improvement = result_filtered['pips'] - result_all['pips']
        print(f"  IMPROVEMENT:   {improvement:+,.0f} pips")

    print("\n" + "=" * 80)
    print("GOOD PAIRS - VERIFICATION")
    print("=" * 80)

    for pair in good_pairs:
        print(f"\n[Loading {pair}...]")
        df = load_data(pair)
        if len(df) == 0:
            continue

        settings = config.get_pair_settings(pair)
        result = backtest_pair(pair, df, use_session_filter=True)

        print(f"\n{pair} (TP: {settings['tp_pips']}p, SL: {settings['sl_pips']}p)")
        print(f"  {result['trades']:,} trades | {result['win_rate']:.1f}% WR | {result['pips']:+,.0f} pips")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    all_pairs = problem_pairs + good_pairs
    total_pips = 0

    print(f"\n{'PAIR':<10} {'TRADES':<10} {'WIN%':<8} {'PIPS':<12} {'STATUS'}")
    print("-" * 55)

    for pair in all_pairs:
        df = load_data(pair)
        if len(df) == 0:
            continue
        result = backtest_pair(pair, df, use_session_filter=True)
        total_pips += result['pips']
        status = "KEEP" if result['pips'] > 0 else "REMOVE"
        print(f"{pair:<10} {result['trades']:<10,} {result['win_rate']:<7.1f}% {result['pips']:+11,.0f} {status}")

    print("-" * 55)
    print(f"{'TOTAL':<10} {'':<10} {'':<8} {total_pips:+11,.0f}")

    print(f"\n[DONE]")
