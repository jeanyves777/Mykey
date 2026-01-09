"""
Analyze winning patterns from forex backtest
Identifies what characteristics make trades profitable
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

print("=" * 80)
print("ANALYZING WINNING PATTERNS FROM BACKTEST")
print("=" * 80)

# Initialize
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

# Download data
print("\nDownloading data...")
all_data = {}

for instrument in MAJOR_PAIRS:
    try:
        candles_1min = client.get_candles(instrument, "M1", count=5000)
        df_1min = pd.DataFrame(candles_1min)

        candles_5min = client.get_candles(instrument, "M5", count=2000)
        df_5min = pd.DataFrame(candles_5min)

        candles_15min = client.get_candles(instrument, "M15", count=500)
        df_15min = pd.DataFrame(candles_15min)

        candles_30min = client.get_candles(instrument, "M30", count=500)
        df_30min = pd.DataFrame(candles_30min)

        all_data[instrument] = {
            "1min": df_1min,
            "5min": df_5min,
            "15min": df_15min,
            "30min": df_30min
        }
    except Exception as e:
        print(f"Error loading {instrument}: {e}")

valid_pairs = list(all_data.keys())
print(f"Loaded data for {len(valid_pairs)} pairs")

# Initialize strategies
strategies = {}
for instrument in valid_pairs:
    params = get_scalping_params(instrument)
    strategies[instrument] = ForexScalpingStrategy(
        instruments=[instrument],
        max_trades_per_day=10,
        daily_profit_target=0.05,
        trade_size_pct=0.05,
        take_profit_pct=params["take_profit_pct"],
        stop_loss_pct=params["stop_loss_pct"],
        trailing_stop_trigger=params["trailing_stop_trigger"],
        trailing_stop_distance=params["trailing_stop_distance"],
        require_htf_strict=True,
        pullback_required=True,
        min_consensus_score=1
    )

# Run backtest and CAPTURE DETAILED SIGNAL DATA
print("\nRunning backtest with pattern analysis...")

balance = 10000.0
max_concurrent = 5
open_positions = {}
trades = []
daily_trades = {pair: 0 for pair in valid_pairs}
current_date = None

# Track signal details for each trade
trade_signals = []

reference_pair = "EUR_USD" if "EUR_USD" in valid_pairs else valid_pairs[0]
df_time_ref = all_data[reference_pair]["1min"]

for i in range(100, len(df_time_ref)):
    current_time = df_time_ref.iloc[i]['time']

    if current_date is None:
        current_date = current_time.date()

    if current_time.date() > current_date:
        current_date = current_time.date()
        daily_trades = {pair: 0 for pair in valid_pairs}

    # Check for entry signals
    if len(open_positions) < max_concurrent:
        for instrument in valid_pairs:
            if instrument in open_positions:
                continue
            if daily_trades[instrument] >= 10:
                continue

            df_1min = all_data[instrument]["1min"]
            df_5min = all_data[instrument]["5min"]
            df_15min = all_data[instrument]["15min"]
            df_30min = all_data[instrument]["30min"]

            pair_bars = df_1min[df_1min['time'] <= current_time]
            if len(pair_bars) < 100:
                continue

            idx = len(pair_bars) - 1
            current_bar = df_1min.iloc[idx]

            df_1min_slice = df_1min.iloc[max(0, idx-200):idx+1]
            df_5min_slice = df_5min[df_5min['time'] <= current_time].tail(100)
            df_15min_slice = df_15min[df_15min['time'] <= current_time].tail(100)
            df_30min_slice = df_30min[df_30min['time'] <= current_time].tail(100)

            if len(df_1min_slice) < 50:
                continue

            daily_pl_pct = (balance - 10000) / 10000

            signal = strategies[instrument].should_enter_trade(
                instrument=instrument,
                df_1min=df_1min_slice,
                df_5min=df_5min_slice,
                df_15min=df_15min_slice,
                df_30min=df_30min_slice,
                current_positions=0,
                trades_today=daily_trades[instrument],
                daily_pl_pct=daily_pl_pct
            )

            if signal["action"] in ["BUY", "SELL"]:
                if len(open_positions) >= max_concurrent:
                    break

                entry_price = current_bar['close']
                units = strategies[instrument].calculate_position_size(balance, entry_price, instrument)

                if signal["action"] == "SELL":
                    units = -units

                sl, tp = strategies[instrument].calculate_stop_loss_take_profit(entry_price, signal["action"], instrument)
                position_value = abs(units * entry_price)

                # CAPTURE SIGNAL DETAILS
                analysis = signal.get('analysis', {})

                open_positions[instrument] = {
                    'entry_bar': i,
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'units': units,
                    'direction': signal["action"],
                    'sl': sl,
                    'tp': tp,
                    'position_value': position_value,
                    'signal_analysis': analysis  # Store analysis
                }

                daily_trades[instrument] += 1

    # Check for exits
    to_close = []
    for instrument, pos in open_positions.items():
        df_1min = all_data[instrument]["1min"]
        pair_bars = df_1min[df_1min['time'] <= current_time]

        if len(pair_bars) == 0:
            continue

        current_bar = pair_bars.iloc[-1]

        hit_tp = False
        hit_sl = False

        if pos['direction'] == "BUY":
            if current_bar['high'] >= pos['tp']:
                hit_tp = True
                exit_price = pos['tp']
            elif current_bar['low'] <= pos['sl']:
                hit_sl = True
                exit_price = pos['sl']
        else:
            if current_bar['low'] <= pos['tp']:
                hit_tp = True
                exit_price = pos['tp']
            elif current_bar['high'] >= pos['sl']:
                hit_sl = True
                exit_price = pos['sl']

        if hit_tp or hit_sl:
            pnl = (exit_price - pos['entry_price']) * pos['units']
            balance += pnl
            pnl_pct = (pnl / pos['position_value']) * 100
            pips = abs(exit_price - pos['entry_price']) * 10000

            # Store trade with signal analysis
            trade_record = {
                'instrument': instrument,
                'entry_time': pos['entry_time'],
                'exit_time': current_time,
                'direction': pos['direction'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'pips': pips,
                'balance_after': balance,
                'exit_reason': 'TP' if hit_tp else 'SL',
                'won': hit_tp,
                'analysis': pos['signal_analysis']
            }

            trades.append(trade_record)
            to_close.append(instrument)

    for inst in to_close:
        del open_positions[inst]

# ANALYZE PATTERNS
print("\n" + "=" * 80)
print("PATTERN ANALYSIS - WINNERS VS LOSERS")
print("=" * 80)

winners = [t for t in trades if t['won']]
losers = [t for t in trades if not t['won']]

print(f"\nTotal Trades: {len(trades)}")
print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
print(f"Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")

# Analyze by pair
print("\n" + "=" * 80)
print("WINNING PATTERNS BY PAIR")
print("=" * 80)

for pair in valid_pairs:
    pair_winners = [t for t in winners if t['instrument'] == pair]
    pair_losers = [t for t in losers if t['instrument'] == pair]
    total_pair = len(pair_winners) + len(pair_losers)

    if total_pair == 0:
        continue

    wr = len(pair_winners) / total_pair * 100

    print(f"\n{pair}: {total_pair} trades | Win Rate: {wr:.0f}%")

    if len(pair_winners) > 0:
        # Analyze momentum scores
        mom_scores_win = []
        for t in pair_winners:
            mom = t['analysis'].get('momentum', {})
            score = mom.get('score', 0)
            mom_scores_win.append(score)

        mom_scores_loss = []
        for t in pair_losers:
            mom = t['analysis'].get('momentum', {})
            score = mom.get('score', 0)
            mom_scores_loss.append(score)

        if mom_scores_win:
            print(f"  Winners Avg Momentum Score: {np.mean(mom_scores_win):.1f}")
        if mom_scores_loss:
            print(f"  Losers Avg Momentum Score: {np.mean(mom_scores_loss):.1f}")

        # Direction bias
        long_wins = len([t for t in pair_winners if t['direction'] == 'BUY'])
        short_wins = len([t for t in pair_winners if t['direction'] == 'SELL'])

        long_loss = len([t for t in pair_losers if t['direction'] == 'BUY'])
        short_loss = len([t for t in pair_losers if t['direction'] == 'SELL'])

        print(f"  Winners: {long_wins} LONG | {short_wins} SHORT")
        print(f"  Losers: {long_loss} LONG | {short_loss} SHORT")

        # Best direction
        if long_wins + long_loss > 0:
            long_wr = long_wins / (long_wins + long_loss) * 100
        else:
            long_wr = 0

        if short_wins + short_loss > 0:
            short_wr = short_wins / (short_wins + short_loss) * 100
        else:
            short_wr = 0

        if long_wr > short_wr and long_wr > 55:
            print(f"  [+] LONG BIAS: {long_wr:.0f}% win rate")
        elif short_wr > long_wr and short_wr > 55:
            print(f"  [+] SHORT BIAS: {short_wr:.0f}% win rate")

# Analyze RSI patterns
print("\n" + "=" * 80)
print("RSI PATTERNS")
print("=" * 80)

rsi_winners = []
rsi_losers = []

for t in winners:
    rsi = t['analysis'].get('rsi', 0)
    rsi_winners.append(rsi)

for t in losers:
    rsi = t['analysis'].get('rsi', 0)
    rsi_losers.append(rsi)

if rsi_winners:
    print(f"\nWinners Avg RSI: {np.mean(rsi_winners):.1f}")
    print(f"  RSI Range: {min(rsi_winners):.1f} - {max(rsi_winners):.1f}")

if rsi_losers:
    print(f"\nLosers Avg RSI: {np.mean(rsi_losers):.1f}")
    print(f"  RSI Range: {min(rsi_losers):.1f} - {max(rsi_losers):.1f}")

# Analyze HTF alignment
print("\n" + "=" * 80)
print("HTF TREND PATTERNS")
print("=" * 80)

htf_winners = defaultdict(int)
htf_losers = defaultdict(int)

for t in winners:
    htf = t['analysis'].get('htf', {})
    trend = htf.get('trend', 'NEUTRAL')
    htf_winners[trend] += 1

for t in losers:
    htf = t['analysis'].get('htf', {})
    trend = htf.get('trend', 'NEUTRAL')
    htf_losers[trend] += 1

print("\nWinners by HTF Trend:")
for trend, count in htf_winners.items():
    pct = count / len(winners) * 100
    print(f"  {trend}: {count} ({pct:.0f}%)")

print("\nLosers by HTF Trend:")
for trend, count in htf_losers.items():
    pct = count / len(losers) * 100
    print(f"  {trend}: {count} ({pct:.0f}%)")

# Time of day analysis
print("\n" + "=" * 80)
print("TIME OF DAY PATTERNS")
print("=" * 80)

def get_hour_bucket(dt):
    hour = dt.hour
    if 0 <= hour < 6:
        return "00-06 (Asian)"
    elif 6 <= hour < 12:
        return "06-12 (London Open)"
    elif 12 <= hour < 18:
        return "12-18 (NY Open)"
    else:
        return "18-24 (Late NY)"

hour_winners = defaultdict(int)
hour_losers = defaultdict(int)

for t in winners:
    bucket = get_hour_bucket(t['entry_time'])
    hour_winners[bucket] += 1

for t in losers:
    bucket = get_hour_bucket(t['entry_time'])
    hour_losers[bucket] += 1

print("\nWinners by Time:")
for bucket in sorted(hour_winners.keys()):
    count = hour_winners[bucket]
    total = count + hour_losers.get(bucket, 0)
    wr = count / total * 100 if total > 0 else 0
    print(f"  {bucket}: {count}W/{hour_losers.get(bucket, 0)}L ({wr:.0f}% WR)")

# RECOMMENDATIONS
print("\n" + "=" * 80)
print("RECOMMENDATIONS TO IMPROVE STRATEGY")
print("=" * 80)

# 1. Remove underperforming pairs
print("\n1. PAIR SELECTION:")
pair_performance = {}
for pair in valid_pairs:
    pair_trades = [t for t in trades if t['instrument'] == pair]
    if pair_trades:
        pair_wins = len([t for t in pair_trades if t['won']])
        wr = pair_wins / len(pair_trades) * 100
        pair_pnl = sum([t['pnl'] for t in pair_trades])
        pair_performance[pair] = (wr, pair_pnl, len(pair_trades))

for pair, (wr, pnl, count) in sorted(pair_performance.items(), key=lambda x: x[1][0], reverse=True):
    status = "[KEEP]" if wr >= 45 else "[REMOVE]"
    print(f"  {status} {pair}: {wr:.0f}% WR | ${pnl:+.2f} | {count} trades")

# 2. Momentum threshold
print("\n2. MOMENTUM SCORE THRESHOLD:")
if rsi_winners and rsi_losers:
    avg_mom_win = np.mean([t['analysis'].get('momentum', {}).get('score', 0) for t in winners])
    avg_mom_loss = np.mean([t['analysis'].get('momentum', {}).get('score', 0) for t in losers])

    print(f"  Winners avg momentum: {avg_mom_win:.1f}")
    print(f"  Losers avg momentum: {avg_mom_loss:.1f}")

    if avg_mom_win > avg_mom_loss:
        recommended_threshold = (avg_mom_win + avg_mom_loss) / 2
        print(f"  [+] INCREASE min_consensus_score to {int(recommended_threshold)}")

# 3. Best trading hours
print("\n3. TRADING HOURS:")
best_session = None
best_wr = 0
for bucket in sorted(hour_winners.keys()):
    count = hour_winners[bucket]
    total = count + hour_losers.get(bucket, 0)
    wr = count / total * 100 if total > 0 else 0
    if wr > best_wr and total >= 5:
        best_wr = wr
        best_session = bucket

if best_session:
    print(f"  [+] Focus trading on {best_session} session ({best_wr:.0f}% WR)")

print("\n" + "=" * 80)
