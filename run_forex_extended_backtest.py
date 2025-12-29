"""
Extended Forex Scalping Backtest - Real 1-min Data
Downloads REAL 1-min data in multiple chunks to backtest over longer period
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
import pytz

print("=" * 80)
print("EXTENDED FOREX SCALPING BACKTEST - REAL 1-MIN DATA")
print("=" * 80)
print(f"\nTrading {len(MAJOR_PAIRS)} Major Pairs:")
for pair in MAJOR_PAIRS:
    print(f"  - {pair}")

# Initialize
print("\n[1/5] Connecting to OANDA...")
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')
print(f"      Account Balance: ${client.get_balance():,.2f}")

# Download REAL 1-min data (max 5000 bars per request = ~3.5 days)
print(f"\n[2/5] Downloading REAL 1-min historical data...")
print("      NOTE: OANDA 1-min data limit = 5000 bars (~3.5 days)")
print("      Using maximum available period")

all_data = {}

for instrument in MAJOR_PAIRS:
    print(f"\n      [{instrument}]")
    try:
        # Fetch REAL 1-min data (not synthetic)
        print(f"        Fetching 1-min candles (5000 bars = ~3.5 days)...")
        candles_1min = client.get_candles(instrument, "M1", count=5000)
        df_1min = pd.DataFrame(candles_1min)

        print(f"        Fetching 5-min candles...")
        candles_5min = client.get_candles(instrument, "M5", count=2000)
        df_5min = pd.DataFrame(candles_5min)

        print(f"        Fetching 15-min candles...")
        candles_15min = client.get_candles(instrument, "M15", count=500)
        df_15min = pd.DataFrame(candles_15min)

        print(f"        Fetching 30-min candles...")
        candles_30min = client.get_candles(instrument, "M30", count=500)
        df_30min = pd.DataFrame(candles_30min)

        all_data[instrument] = {
            "1min": df_1min,
            "5min": df_5min,
            "15min": df_15min,
            "30min": df_30min
        }

        days = len(df_1min) / 1440
        print(f"        Loaded {len(df_1min)} 1-min bars (~{days:.1f} days)")

    except Exception as e:
        print(f"        ERROR: {e}")
        all_data[instrument] = None

# Summary
print(f"\n[3/5] Data Summary:")
valid_pairs = [k for k, v in all_data.items() if v is not None]
print(f"      Successfully loaded: {len(valid_pairs)}/{len(MAJOR_PAIRS)} pairs")

if len(valid_pairs) == 0:
    print("\nERROR: No data loaded. Cannot run backtest.")
    sys.exit(1)

# Find common date range
min_bars_1min = min([len(all_data[pair]["1min"]) for pair in valid_pairs])
print(f"      Common 1-min bars available: {min_bars_1min}")
print(f"      Backtest period: ~{min_bars_1min / 1440:.1f} days")

# Get date range
start_time = min([all_data[p]["1min"]['time'].min() for p in valid_pairs])
end_time = max([all_data[p]["1min"]['time'].max() for p in valid_pairs])
print(f"      Date range: {start_time} to {end_time}")

# Initialize strategy for each pair
print("\n[4/5] Initializing Scalping Strategies...")
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
        pullback_required=True,  # WITH PULLBACK DETECTION
        min_consensus_score=1
    )

print(f"      Initialized {len(strategies)} strategies with pullback detection")

# Run backtest with RELAXED HTF requirement to test pullback alone
print("\n[5/5] Running extended multi-symbol backtest...")
print("=" * 80)

balance = 10000.0
max_concurrent = 5
open_positions = {}
trades = []
daily_trades = {pair: 0 for pair in valid_pairs}
current_date = None

print(f"\nBacktest Period: {start_time} to {end_time}")
print(f"Duration: {(end_time - start_time).days} days\n")

# Use EUR_USD as time reference
reference_pair = "EUR_USD" if "EUR_USD" in valid_pairs else valid_pairs[0]
df_time_ref = all_data[reference_pair]["1min"]

# Progress tracking
total_bars = len(df_time_ref)
progress_step = total_bars // 20

# Start from bar 100 for indicator warmup
for i in range(100, len(df_time_ref)):
    # Progress
    if i % progress_step == 0:
        pct = (i / total_bars) * 100
        print(f"[PROGRESS] {pct:.0f}% | Balance: ${balance:,.0f} | Trades: {len(trades)} | Open: {len(open_positions)}")

    current_time = df_time_ref.iloc[i]['time']

    # Daily reset
    if current_date is None:
        current_date = current_time.date()

    if current_time.date() > current_date:
        daily_pl = balance - (trades[-1]['balance_after'] if trades else 10000.0)
        total_trades_today = sum(daily_trades.values())

        print(f"\n[DAY] {current_date} | Trades: {total_trades_today} | P&L: ${daily_pl:+,.2f} | Balance: ${balance:,.2f}")
        if total_trades_today > 0:
            pairs_traded = {k: v for k, v in daily_trades.items() if v > 0}
            for pair, count in pairs_traded.items():
                print(f"      {pair}: {count} trades")

        current_date = current_time.date()
        daily_trades = {pair: 0 for pair in valid_pairs}

    # Check for entry signals
    if len(open_positions) < max_concurrent:
        for instrument in valid_pairs:
            if instrument in open_positions:
                continue

            if daily_trades[instrument] >= 10:
                continue

            # Get data slices for this pair
            df_1min = all_data[instrument]["1min"]
            df_5min = all_data[instrument]["5min"]
            df_15min = all_data[instrument]["15min"]
            df_30min = all_data[instrument]["30min"]

            # Find current bar index in this pair's data
            pair_bars = df_1min[df_1min['time'] <= current_time]
            if len(pair_bars) < 100:
                continue

            idx = len(pair_bars) - 1
            current_bar = df_1min.iloc[idx]

            # Get slices
            df_1min_slice = df_1min.iloc[max(0, idx-200):idx+1]
            df_5min_slice = df_5min[df_5min['time'] <= current_time].tail(100)
            df_15min_slice = df_15min[df_15min['time'] <= current_time].tail(100)
            df_30min_slice = df_30min[df_30min['time'] <= current_time].tail(100)

            if len(df_1min_slice) < 50:
                continue

            # Check for signal
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

                open_positions[instrument] = {
                    'entry_bar': i,
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'units': units,
                    'direction': signal["action"],
                    'sl': sl,
                    'tp': tp,
                    'position_value': position_value
                }

                daily_trades[instrument] += 1

                print(f"\n[ENTRY #{len(trades)+1}] {current_time.strftime('%m-%d %H:%M')} | {instrument} {signal['action']} @ {entry_price:.5f}")
                print(f"           Size: ${position_value:,.0f} | TP: {abs(tp-entry_price)*10000:.0f}p | SL: {abs(sl-entry_price)*10000:.0f}p | Open: {len(open_positions)}")

    # Check for exits
    to_close = []
    for instrument, pos in open_positions.items():
        # Get current bar for this pair
        df_1min = all_data[instrument]["1min"]
        pair_bars = df_1min[df_1min['time'] <= current_time]

        if len(pair_bars) == 0:
            continue

        current_bar = pair_bars.iloc[-1]
        current_price = current_bar['close']

        # Check TP/SL
        hit_tp = False
        hit_sl = False

        if pos['direction'] == "BUY":
            if current_bar['high'] >= pos['tp']:
                hit_tp = True
                exit_price = pos['tp']
            elif current_bar['low'] <= pos['sl']:
                hit_sl = True
                exit_price = pos['sl']
        else:  # SELL
            if current_bar['low'] <= pos['tp']:
                hit_tp = True
                exit_price = pos['tp']
            elif current_bar['high'] >= pos['sl']:
                hit_sl = True
                exit_price = pos['sl']

        if hit_tp or hit_sl:
            # Close position
            pnl = (exit_price - pos['entry_price']) * pos['units']
            balance += pnl
            pnl_pct = (pnl / pos['position_value']) * 100
            pips = abs(exit_price - pos['entry_price']) * 10000

            trades.append({
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
                'exit_reason': 'TP' if hit_tp else 'SL'
            })

            print(f"[EXIT  #{len(trades)}] {current_time.strftime('%m-%d %H:%M')} | {instrument} {'TP' if hit_tp else 'SL'} @ {exit_price:.5f} | P&L: ${pnl:+.0f} ({pnl_pct:+.1f}%) | Bal: ${balance:,.0f}")

            to_close.append(instrument)

    for inst in to_close:
        del open_positions[inst]

# Final summary
print("\n" + "=" * 80)
print("[FINAL] EXTENDED BACKTEST RESULTS")
print("=" * 80)

if len(trades) == 0:
    print("\nNo trades executed.")
    print("\n⚠️  STRATEGY TOO RESTRICTIVE - Consider:")
    print("   1. Relax HTF requirement (require_htf_strict=False)")
    print("   2. Lower pullback threshold (score >= 3 instead of 4)")
    print("   3. Reduce consensus requirement")
else:
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]

    print(f"\nOverall Performance:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100 if trades else 0:.1f}%)")
    print(f"  Losers: {len(losers)}")
    if winners:
        print(f"  Avg Win: ${np.mean([t['pnl'] for t in winners]):.2f} (+{np.mean([t['pips'] for t in winners]):.1f} pips)")
    if losers:
        print(f"  Avg Loss: ${np.mean([t['pnl'] for t in losers]):.2f} (+{np.mean([t['pips'] for t in losers]):.1f} pips)")

    total_profit = sum([t['pnl'] for t in winners]) if winners else 0
    total_loss = abs(sum([t['pnl'] for t in losers])) if losers else 0
    pf = total_profit / total_loss if total_loss > 0 else float('inf')
    print(f"  Profit Factor: {pf:.2f}")

    # Calculate max drawdown
    peak = 10000
    max_dd = 0
    for t in trades:
        if t['balance_after'] > peak:
            peak = t['balance_after']
        dd = ((peak - t['balance_after']) / peak) * 100
        if dd > max_dd:
            max_dd = dd
    print(f"  Max Drawdown: {max_dd:.2f}%")

    print(f"\nReturns:")
    print(f"  Initial: $10,000.00")
    print(f"  Final: ${balance:,.2f}")
    print(f"  P&L: ${balance - 10000:+,.2f}")
    print(f"  Return: {((balance - 10000) / 10000) * 100:+.2f}%")

    # Per-symbol breakdown
    print(f"\nPer-Symbol Breakdown:")
    for pair in valid_pairs:
        pair_trades = [t for t in trades if t['instrument'] == pair]
        if pair_trades:
            pair_wins = len([t for t in pair_trades if t['pnl'] > 0])
            pair_pnl = sum([t['pnl'] for t in pair_trades])
            win_rate = (pair_wins / len(pair_trades)) * 100 if pair_trades else 0
            print(f"  {pair}: {len(pair_trades)} trades | {pair_wins}W-{len(pair_trades)-pair_wins}L ({win_rate:.0f}%) | ${pair_pnl:+.2f}")

    days_traded = (end_time - start_time).days
    print(f"\nTrading Stats:")
    print(f"  Days: {days_traded}")
    print(f"  Trades/Day: {len(trades)/days_traded:.1f}")
    print(f"  Symbols Traded: {len(set([t['instrument'] for t in trades]))}/{len(valid_pairs)}")

print("\n" + "=" * 80)
print("Extended Backtest Complete!")
print("=" * 80)
