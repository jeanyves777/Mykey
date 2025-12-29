"""
Backtest using OANDA's live 3-day data
Tests the improved strategy settings on real recent market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading'))
sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading' / 'engine'))
from config import optimized_paper_config as config
from oanda_client import OandaClient

# Initialize OANDA client
client = OandaClient()

# Spread simulation (OANDA typical)
SPREAD_PIPS = {
    'EUR_USD': 1.0,
    'GBP_USD': 1.2,
    'USD_CHF': 1.5,
    'USD_CAD': 1.2,
    'NZD_USD': 1.5,
    'AUD_JPY': 1.8,
    'EUR_GBP': 1.2,
    'AUD_CHF': 2.0,
    'EUR_CAD': 2.0,
}


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    """Fetch candle data from OANDA."""
    print(f"  Fetching {count} candles from OANDA...")

    candles = client.get_candles(pair, 'M1', count=count)

    if not candles:
        print(f"  ERROR: No data returned for {pair}")
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['time'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Got {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calculate indicators
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
    df['date'] = df['timestamp'].dt.date

    return df


def backtest_pair(pair: str, df: pd.DataFrame, use_session_filter: bool = True) -> dict:
    """Run backtest on OANDA data."""
    if len(df) < 100:
        return {'trades': [], 'summary': {}}

    settings = config.get_pair_settings(pair)
    pip_mult = config.PIP_MULTIPLIERS.get(pair, 10000)
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
    timestamps = df['timestamp'].values
    rsi_arr = df['rsi'].values
    is_green_arr = df['is_green'].values
    is_red_arr = df['is_red'].values
    macd_arr = df['macd'].values
    macd_signal_arr = df['macd_signal'].values
    ema9_arr = df['ema9'].values
    ema21_arr = df['ema21'].values
    ema50_arr = df['ema50'].values
    hour_arr = df['hour'].values

    trades = []

    for i in range(50, len(df) - 50):
        # Session filter
        if use_session_filter and hasattr(config, 'is_allowed_hour'):
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
        entry_time = pd.Timestamp(timestamps[i])

        if signal == 'BUY':
            entry += spread_dist
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            entry -= spread_dist
            tp = entry - tp_dist
            sl = entry + sl_dist

        # Simulate outcome
        result = None
        exit_price = entry
        exit_time = entry_time

        for j in range(i + 1, min(i + 300, len(df))):
            high = highs[j]
            low = lows[j]

            if signal == 'BUY':
                if low <= sl:
                    result = 'LOSS'
                    exit_price = sl
                    exit_time = pd.Timestamp(timestamps[j])
                    break
                elif high >= tp:
                    result = 'WIN'
                    exit_price = tp
                    exit_time = pd.Timestamp(timestamps[j])
                    break
            else:
                if high >= sl:
                    result = 'LOSS'
                    exit_price = sl
                    exit_time = pd.Timestamp(timestamps[j])
                    break
                elif low <= tp:
                    result = 'WIN'
                    exit_price = tp
                    exit_time = pd.Timestamp(timestamps[j])
                    break

        if result is None:
            continue

        # Calculate P&L
        if signal == 'BUY':
            pnl_pips = (exit_price - entry) * pip_mult
        else:
            pnl_pips = (entry - exit_price) * pip_mult

        duration = (exit_time - entry_time).total_seconds() / 60

        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'signal': signal,
            'entry': entry,
            'exit': exit_price,
            'result': result,
            'pnl_pips': pnl_pips,
            'duration_min': duration
        })

    # Summary
    if trades:
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        total_pips = sum(t['pnl_pips'] for t in trades)
        avg_duration = np.mean([t['duration_min'] for t in trades])

        summary = {
            'trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate': wins / len(trades) * 100,
            'total_pips': total_pips,
            'avg_duration_min': avg_duration
        }
    else:
        summary = {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pips': 0}

    return {'trades': trades, 'summary': summary}


if __name__ == '__main__':
    print("=" * 80)
    print("BACKTEST: OANDA 3-DAY LIVE DATA")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print(f"\nUsing IMPROVED settings from config:")
    print(f"  - Session filters enabled")
    print(f"  - Wider TP for EUR_USD & USD_CAD (8 pips)")
    print(f"  - USD_JPY removed")

    all_results = {}

    for pair in config.OPTIMIZED_PAIRS:
        print(f"\n{'=' * 60}")
        print(f"[{pair}]")
        print(f"{'=' * 60}")

        settings = config.get_pair_settings(pair)
        print(f"Strategy: {settings['strategy']}")
        print(f"TP: {settings['tp_pips']} pips | SL: {settings['sl_pips']} pips")

        if hasattr(config, 'TRADING_SESSIONS'):
            session = config.TRADING_SESSIONS.get(pair, {})
            hours = session.get('allowed_hours', list(range(24)))
            print(f"Trading Hours: {min(hours):02d}:00-{max(hours):02d}:00 UTC")

        # Fetch OANDA data
        df = fetch_oanda_data(pair, count=5000)  # ~3.5 days of M1 data

        if len(df) < 100:
            print(f"  Insufficient data")
            continue

        # Run backtest with session filter
        result = backtest_pair(pair, df, use_session_filter=True)
        summary = result['summary']
        all_results[pair] = summary

        print(f"\nRESULTS (with session filter):")
        print(f"  Trades: {summary['trades']}")
        print(f"  Win Rate: {summary['win_rate']:.1f}% ({summary['wins']}W/{summary['losses']}L)")
        print(f"  Total Pips: {summary['total_pips']:+.1f}")
        if summary['trades'] > 0:
            print(f"  Avg Duration: {summary.get('avg_duration_min', 0):.1f} min")

        # Show recent trades
        trades = result['trades']
        if trades:
            print(f"\n  Recent trades:")
            for t in trades[-5:]:
                emoji = "WIN" if t['result'] == 'WIN' else "LOSS"
                print(f"    [{emoji}] {t['entry_time'].strftime('%m/%d %H:%M')} {t['signal']} ({t['pnl_pips']:+.1f}p, {t['duration_min']:.0f}m)")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - 3 DAY OANDA BACKTEST")
    print("=" * 80)

    print(f"\n{'PAIR':<10} {'TRADES':<8} {'WIN%':<8} {'PIPS':<12} {'STATUS'}")
    print("-" * 50)

    total_trades = 0
    total_wins = 0
    total_pips = 0

    for pair in config.OPTIMIZED_PAIRS:
        if pair in all_results:
            r = all_results[pair]
            status = "OK" if r['total_pips'] >= 0 else "CHECK"
            print(f"{pair:<10} {r['trades']:<8} {r['win_rate']:<7.1f}% {r['total_pips']:+11.1f} {status}")
            total_trades += r['trades']
            total_wins += r['wins']
            total_pips += r['total_pips']

    print("-" * 50)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"{'TOTAL':<10} {total_trades:<8} {overall_wr:<7.1f}% {total_pips:+11.1f}")

    # Compare to expected
    print("\n" + "=" * 80)
    print("EXPECTED vs ACTUAL")
    print("=" * 80)

    print(f"\n{'PAIR':<10} {'EXP WR%':<10} {'ACT WR%':<10} {'DIFF'}")
    print("-" * 40)

    for pair in config.OPTIMIZED_PAIRS:
        if pair in all_results:
            r = all_results[pair]
            settings = config.get_pair_settings(pair)
            exp_wr = settings.get('expected_wr', 75)
            diff = r['win_rate'] - exp_wr
            status = "OK" if diff >= -10 else "BELOW"
            print(f"{pair:<10} {exp_wr:<10.1f} {r['win_rate']:<10.1f} {diff:+.1f}% {status}")

    print(f"\n[{datetime.now()}] Backtest complete!")
