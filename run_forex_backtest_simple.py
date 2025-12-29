"""
Simple OANDA Forex Backtest
Uses count-based data retrieval (works with OANDA limits)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.multi_timeframe_momentum import MultiTimeframeMomentumStrategy
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

print("="*80)
print("FOREX BACKTEST - OANDA Real Data")
print("="*80)

# Initialize OANDA client
print("\n[1/6] Connecting to OANDA...")
client = OandaClient('practice')
print(f"      Account Balance: ${client.get_balance():,.2f}")

# Get recent historical data (last 5000 bars for each timeframe)
instrument = "EUR_USD"
print(f"\n[2/6] Downloading {instrument} historical data...")

print("      Fetching 1-min candles (last 5000)...")
candles_1min = client.get_candles(instrument, "M1", count=5000)
df_1min = pd.DataFrame(candles_1min)

print("      Fetching 5-min candles (last 2000)...")
candles_5min = client.get_candles(instrument, "M5", count=2000)
df_5min = pd.DataFrame(candles_5min)

print("      Fetching 30-min candles (last 500)...")
candles_30min = client.get_candles(instrument, "M30", count=500)
df_30min = pd.DataFrame(candles_30min)

print("      Fetching 1-hour candles (last 500)...")
candles_1hour = client.get_candles(instrument, "H1", count=500)
df_1hour = pd.DataFrame(candles_1hour)

print(f"\n[3/6] Data Summary:")
print(f"      1-min bars: {len(df_1min)} (covers ~{len(df_1min)/1440:.1f} days)")
print(f"      5-min bars: {len(df_5min)}")
print(f"      30-min bars: {len(df_30min)}")
print(f"      1-hour bars: {len(df_1hour)}")

if len(df_1min) > 0:
    print(f"      Date range: {df_1min.iloc[0]['time'].date()} to {df_1min.iloc[-1]['time'].date()}")

# Initialize strategy
print("\n[4/6] Initializing Multi-Timeframe Momentum Strategy...")
strategy = MultiTimeframeMomentumStrategy(
    instruments=[instrument],
    max_trades_per_day=3,
    daily_profit_target=0.02,
    trade_size_pct=0.10,
    take_profit_pct=0.015,
    stop_loss_pct=0.01,
    trailing_stop_trigger=0.006,
    trailing_stop_distance=0.004
)

# Run backtest
print("\n[5/6] Running backtest simulation...")
print("="*80)

balance = 10000
initial_balance = balance
positions = {}
trades = []
daily_trades = 0
daily_start_balance = balance
current_date = None

# Start from bar 500 to have enough history for indicators
for i in range(500, len(df_1min)):
    current_time = df_1min.iloc[i]['time']
    current_bar = df_1min.iloc[i]

    # Daily reset
    if current_date is None or current_time.date() > current_date:
        if current_date is not None:
            daily_pl = balance - daily_start_balance
            daily_pl_pct = daily_pl / daily_start_balance * 100
            print(f"\n[DAY] {current_date} | Trades: {daily_trades} | P&L: ${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%) | Balance: ${balance:,.2f}")

        current_date = current_time.date()
        daily_trades = 0
        daily_start_balance = balance

    # Get data slices
    df_1min_slice = df_1min.iloc[max(0, i-500):i+1]
    df_5min_slice = df_5min[df_5min['time'] <= current_time].tail(100)
    df_30min_slice = df_30min[df_30min['time'] <= current_time].tail(50)
    df_1hour_slice = df_1hour[df_1hour['time'] <= current_time].tail(50)

    # Check for entry
    if len(positions) == 0 and daily_trades < 3:
        daily_pl_pct = (balance - daily_start_balance) / daily_start_balance

        signal = strategy.should_enter_trade(
            instrument=instrument,
            df_1min=df_1min_slice,
            df_5min=df_5min_slice,
            df_30min=df_30min_slice,
            df_1hour=df_1hour_slice,
            current_positions=len(positions),
            trades_today=daily_trades,
            daily_pl_pct=daily_pl_pct
        )

        if signal["action"] in ["BUY", "SELL"]:
            entry_price = current_bar['close']
            units = strategy.calculate_position_size(balance, entry_price, instrument)
            stop_loss, take_profit = strategy.calculate_stop_loss_take_profit(entry_price, signal["action"])

            commission = abs(units) * entry_price * 0.00002
            balance -= commission

            position = {
                "direction": "LONG" if signal["action"] == "BUY" else "SHORT",
                "entry_price": entry_price,
                "entry_time": current_time,
                "units": units,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "commission": commission,
                "trailing_stop_triggered": False,
                "highest_price": entry_price if signal["action"] == "BUY" else None,
                "lowest_price": entry_price if signal["action"] == "SELL" else None,
            }

            positions[instrument] = position
            daily_trades += 1

            print(f"\n{'='*80}")
            print(f"[ENTRY] {current_time.strftime('%Y-%m-%d %H:%M')} | {position['direction']} @ {entry_price:.5f}")
            print(f"        Units: {units:,} | Size: ${abs(units*entry_price):,.2f}")
            print(f"        SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            print(f"        Reason: {signal['reason']}")
            print(f"        Confidence: {signal['confidence']}")
            print(f"{'='*80}")

    # Manage positions
    if instrument in positions:
        pos = positions[instrument]
        current_price = current_bar['close']

        # Update highest/lowest
        if pos["direction"] == "LONG":
            if pos["highest_price"] is None or current_bar['high'] > pos["highest_price"]:
                pos["highest_price"] = current_bar['high']
        else:
            if pos["lowest_price"] is None or current_bar['low'] < pos["lowest_price"]:
                pos["lowest_price"] = current_bar['low']

        # Calculate P&L
        if pos["direction"] == "LONG":
            pl = (current_price - pos["entry_price"]) * pos["units"]
        else:
            pl = (pos["entry_price"] - current_price) * abs(pos["units"])

        pl_pct = pl / (abs(pos["units"]) * pos["entry_price"])

        # Check trailing stop
        if not pos["trailing_stop_triggered"] and pl_pct >= 0.006:
            pos["trailing_stop_triggered"] = True
            print(f"[TRAILING] Trailing stop activated at +{pl_pct*100:.2f}%")

        # Check exits
        exit_price = None
        exit_reason = None

        if pos["direction"] == "LONG":
            if current_bar['high'] >= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TAKE PROFIT"
            elif current_bar['low'] <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "STOP LOSS"
            elif pos["trailing_stop_triggered"]:
                trailing_stop = pos["highest_price"] * (1 - 0.004)
                if current_bar['low'] <= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "TRAILING STOP"
        else:
            if current_bar['low'] <= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TAKE PROFIT"
            elif current_bar['high'] >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "STOP LOSS"
            elif pos["trailing_stop_triggered"]:
                trailing_stop = pos["lowest_price"] * (1 + 0.004)
                if current_bar['high'] >= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "TRAILING STOP"

        if exit_price:
            if pos["direction"] == "LONG":
                final_pl = (exit_price - pos["entry_price"]) * pos["units"]
            else:
                final_pl = (pos["entry_price"] - exit_price) * abs(pos["units"])

            final_pl_pct = final_pl / (abs(pos["units"]) * pos["entry_price"])
            balance += final_pl

            trade = {
                "instrument": instrument,
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "entry_time": pos["entry_time"],
                "exit_price": exit_price,
                "exit_time": current_time,
                "units": pos["units"],
                "pl": final_pl,
                "pl_pct": final_pl_pct,
                "exit_reason": exit_reason,
                "commission": pos["commission"],
                "balance": balance
            }

            trades.append(trade)

            print(f"\n{'='*80}")
            print(f"[EXIT] {current_time.strftime('%Y-%m-%d %H:%M')} | {pos['direction']} @ {exit_price:.5f}")
            print(f"       P&L: ${final_pl:+,.2f} ({final_pl_pct*100:+.2f}%) | Reason: {exit_reason}")
            print(f"       Balance: ${balance:,.2f} | Total Return: {((balance/initial_balance)-1)*100:+.2f}%")
            print(f"{'='*80}")

            del positions[instrument]

    # Progress update
    if i % 1000 == 0:
        progress = (i - 500) / (len(df_1min) - 500) * 100
        print(f"[PROGRESS] {progress:.1f}% | Balance: ${balance:,.2f} | Trades: {len(trades)}")

# Print results
print("\n" + "="*80)
print("[6/6] BACKTEST RESULTS")
print("="*80)

if len(trades) == 0:
    print("\nNo trades executed during backtest period.")
    print("\nThis means the strategy is very strict with its filters.")
    print("The multi-timeframe validation pipeline blocked all potential trades.")
    print("\nTo see trades, try:")
    print("  - Longer data period (more bars)")
    print("  - Different market conditions")
    print("  - Adjust strategy parameters")
else:
    winning_trades = [t for t in trades if t["pl"] > 0]
    losing_trades = [t for t in trades if t["pl"] < 0]

    total_pl = balance - initial_balance
    total_return = (balance / initial_balance - 1) * 100
    win_rate = len(winning_trades) / len(trades) * 100

    avg_win = np.mean([t["pl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["pl"] for t in losing_trades]) if losing_trades else 0

    profit_factor = abs(sum(t["pl"] for t in winning_trades) / sum(t["pl"] for t in losing_trades)) if losing_trades else float('inf')

    print(f"\n📊 Performance Metrics:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"   Losing Trades: {len(losing_trades)}")
    print(f"   Average Win: ${avg_win:,.2f}")
    print(f"   Average Loss: ${avg_loss:,.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"\n💰 Returns:")
    print(f"   Initial Capital: ${initial_balance:,.2f}")
    print(f"   Final Balance: ${balance:,.2f}")
    print(f"   Total P&L: ${total_pl:+,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")

    print("\n" + "="*80)
    print("TRADE LOG")
    print("="*80)
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade #{i}:")
        print(f"  {trade['direction']} {trade['instrument']}")
        print(f"  Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {trade['entry_price']:.5f}")
        print(f"  Exit:  {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} @ {trade['exit_price']:.5f}")
        print(f"  P&L: ${trade['pl']:+,.2f} ({trade['pl_pct']*100:+.2f}%)")
        print(f"  Reason: {trade['exit_reason']}")

print("\n" + "="*80)
print("Backtest Complete!")
print("="*80)
