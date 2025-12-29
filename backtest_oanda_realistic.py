"""
REALISTIC BACKTEST: One position per symbol at a time
======================================================
This reflects actual trading: you can only have 1 open position per symbol.
Compare NO COOLDOWN vs 15-MIN vs 60-MIN cooldown AFTER trade closes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading'))
sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading' / 'engine'))
from config import optimized_paper_config as config
from oanda_client import OandaClient

client = OandaClient()

SPREAD_PIPS = {
    'EUR_USD': 1.0, 'GBP_USD': 1.2, 'USD_CHF': 1.5, 'USD_CAD': 1.2,
    'NZD_USD': 1.5, 'AUD_JPY': 1.8, 'EUR_GBP': 1.2, 'AUD_CHF': 2.0,
    'EUR_CAD': 2.0, 'USD_JPY': 1.0,
}


def fetch_oanda_data(pair: str, count: int = 5000) -> pd.DataFrame:
    print(f"  Fetching {count} candles from OANDA...")
    candles = client.get_candles(pair, 'M1', count=count)
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df['timestamp'] = pd.to_datetime(df['time'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Got {len(df)} candles")

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
    return df


def backtest_realistic(pair: str, df: pd.DataFrame, cooldown_minutes: int) -> dict:
    """
    REALISTIC backtest: Only ONE position at a time per symbol.
    Cooldown is applied AFTER trade closes (not entry).
    """
    if len(df) < 100:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0}

    settings = config.get_pair_settings(pair)
    pip_mult = config.PIP_MULTIPLIERS.get(pair, 10000)
    spread_pips = SPREAD_PIPS.get(pair, 1.0)

    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult
    spread_dist = spread_pips / pip_mult

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
    in_position = False
    entry_price = 0
    entry_time = None
    direction = None
    tp_price = 0
    sl_price = 0
    last_exit_time = None

    for i in range(50, len(df)):
        current_time = pd.Timestamp(timestamps[i])

        if in_position:
            # Check if position closes
            high = highs[i]
            low = lows[i]

            if direction == 'BUY':
                if low <= sl_price:
                    # Stop loss hit
                    trades.append({'result': 'LOSS', 'pnl': -sl_pips, 'exit_time': current_time})
                    in_position = False
                    last_exit_time = current_time
                elif high >= tp_price:
                    # Take profit hit
                    trades.append({'result': 'WIN', 'pnl': tp_pips, 'exit_time': current_time})
                    in_position = False
                    last_exit_time = current_time
            else:  # SELL
                if high >= sl_price:
                    trades.append({'result': 'LOSS', 'pnl': -sl_pips, 'exit_time': current_time})
                    in_position = False
                    last_exit_time = current_time
                elif low <= tp_price:
                    trades.append({'result': 'WIN', 'pnl': tp_pips, 'exit_time': current_time})
                    in_position = False
                    last_exit_time = current_time
        else:
            # Not in position - check for entry

            # Session filter
            if hasattr(config, 'is_allowed_hour'):
                hour = hour_arr[i]
                if not config.is_allowed_hour(pair, hour):
                    continue

            # Cooldown check (after last EXIT)
            if cooldown_minutes > 0 and last_exit_time is not None:
                time_since = (current_time - last_exit_time).total_seconds() / 60
                if time_since < cooldown_minutes:
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

            if signal:
                in_position = True
                direction = signal
                entry_time = current_time
                entry_price = closes[i]

                if signal == 'BUY':
                    entry_price += spread_dist
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    entry_price -= spread_dist
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0, 'total_pips': 0}

    wins = sum(1 for t in trades if t['result'] == 'WIN')
    total_pips = sum(t['pnl'] for t in trades)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': wins / len(trades) * 100,
        'total_pips': total_pips
    }


print("=" * 80)
print("REALISTIC BACKTEST: ONE POSITION PER SYMBOL AT A TIME")
print("=" * 80)
print(f"Time: {datetime.now()}")
print("This reflects ACTUAL trading - only 1 position per symbol allowed.")
print("Cooldown is applied AFTER trade closes.")
print("=" * 80)

results = {0: {}, 15: {}, 60: {}}

for pair in config.OPTIMIZED_PAIRS:
    print(f"\n{pair}:")
    df = fetch_oanda_data(pair, 5000)

    for cd in [0, 15, 60]:
        result = backtest_realistic(pair, df, cd)
        results[cd][pair] = result

    r0 = results[0][pair]
    r15 = results[15][pair]
    r60 = results[60][pair]

    print(f"  NO COOLDOWN:  {r0['trades']:>4}t  {r0['win_rate']:>5.1f}% WR  {r0['total_pips']:>+7.1f}p")
    print(f"  15-MIN CD:    {r15['trades']:>4}t  {r15['win_rate']:>5.1f}% WR  {r15['total_pips']:>+7.1f}p  [{r15['total_pips']-r0['total_pips']:+.0f}p]")
    print(f"  60-MIN CD:    {r60['trades']:>4}t  {r60['win_rate']:>5.1f}% WR  {r60['total_pips']:>+7.1f}p  [{r60['total_pips']-r0['total_pips']:+.0f}p]")

# Summary
print("\n" + "=" * 80)
print("SUMMARY - REALISTIC (1 position per symbol)")
print("=" * 80)
print(f"{'PAIR':<10} {'NO COOLDOWN':<20} {'15-MIN CD':<20} {'60-MIN CD':<20}")
print("-" * 75)

for pair in config.OPTIMIZED_PAIRS:
    r0 = results[0][pair]
    r15 = results[15][pair]
    r60 = results[60][pair]
    print(f"{pair:<10} "
          f"{r0['trades']:>3}t {r0['win_rate']:>5.1f}% {r0['total_pips']:>+6.0f}p   "
          f"{r15['trades']:>3}t {r15['win_rate']:>5.1f}% {r15['total_pips']:>+6.0f}p   "
          f"{r60['trades']:>3}t {r60['win_rate']:>5.1f}% {r60['total_pips']:>+6.0f}p")

print("-" * 75)

# Totals
for cd in [0, 15, 60]:
    total_trades = sum(r['trades'] for r in results[cd].values())
    total_wins = sum(r['wins'] for r in results[cd].values())
    total_pips = sum(r['total_pips'] for r in results[cd].values())
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    results[cd]['_total'] = {'trades': total_trades, 'wins': total_wins, 'win_rate': total_wr, 'total_pips': total_pips}

r0 = results[0]['_total']
r15 = results[15]['_total']
r60 = results[60]['_total']

print(f"{'TOTAL':<10} "
      f"{r0['trades']:>3}t {r0['win_rate']:>5.1f}% {r0['total_pips']:>+6.0f}p   "
      f"{r15['trades']:>3}t {r15['win_rate']:>5.1f}% {r15['total_pips']:>+6.0f}p   "
      f"{r60['trades']:>3}t {r60['win_rate']:>5.1f}% {r60['total_pips']:>+6.0f}p")

print("\n" + "=" * 80)
print("FINAL COMPARISON - REALISTIC TRADING")
print("=" * 80)
print(f"\n  NO COOLDOWN (trade immediately after exit):")
print(f"    {r0['trades']} trades, {r0['win_rate']:.1f}% WR, {r0['total_pips']:+,.0f} pips")

print(f"\n  15-MINUTE COOLDOWN (wait 15 min after exit):")
print(f"    {r15['trades']} trades, {r15['win_rate']:.1f}% WR, {r15['total_pips']:+,.0f} pips")
diff15 = r15['total_pips'] - r0['total_pips']
print(f"    Difference: {diff15:+,.0f} pips ({diff15/r0['total_pips']*100 if r0['total_pips'] != 0 else 0:+.1f}%)")

print(f"\n  60-MINUTE COOLDOWN (wait 1 hour after exit):")
print(f"    {r60['trades']} trades, {r60['win_rate']:.1f}% WR, {r60['total_pips']:+,.0f} pips")
diff60 = r60['total_pips'] - r0['total_pips']
print(f"    Difference: {diff60:+,.0f} pips ({diff60/r0['total_pips']*100 if r0['total_pips'] != 0 else 0:+.1f}%)")

best = max([('NO COOLDOWN', r0['total_pips']), ('15-MIN CD', r15['total_pips']), ('60-MIN CD', r60['total_pips'])], key=lambda x: x[1])
print(f"\n  >>> BEST: {best[0]} with {best[1]:+,.0f} pips <<<")
print("=" * 80)
