"""
Forex Scalping Backtest
Tests scalping strategy with OANDA real data
Optimized for multiple trades per day
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

print("="*80)
print("FOREX SCALPING BACKTEST - OANDA Real Data")
print("="*80)

# Initialize
print("\n[1/6] Connecting to OANDA...")
client = OandaClient('practice')
print(f"      Account Balance: ${client.get_balance():,.2f}")

# Get data
instrument = "EUR_USD"
print(f"\n[2/6] Downloading {instrument} historical data...")

print("      Fetching 1-min candles...")
candles_1min = client.get_candles(instrument, "M1", count=5000)
df_1min = pd.DataFrame(candles_1min)

print("      Fetching 5-min candles...")
candles_5min = client.get_candles(instrument, "M5", count=2000)
df_5min = pd.DataFrame(candles_5min)

print("      Fetching 15-min candles...")
candles_15min = client.get_candles(instrument, "M15", count=500)
df_15min = pd.DataFrame(candles_15min)

print("      Fetching 30-min candles...")
candles_30min = client.get_candles(instrument, "M30", count=500)
df_30min = pd.DataFrame(candles_30min)

print(f"\n[3/6] Data Summary:")
print(f"      1-min bars: {len(df_1min)} (covers ~{len(df_1min)/1440:.1f} days)")
print(f"      5-min bars: {len(df_5min)}")
print(f"      15-min bars: {len(df_15min)}")
print(f"      30-min bars: {len(df_30min)}")

if len(df_1min) > 0:
    print(f"      Date range: {df_1min.iloc[0]['time'].date()} to {df_1min.iloc[-1]['time'].date()}")
    price_range = df_1min['high'].max() - df_1min['low'].min()
    print(f"      Price range: {price_range:.5f} ({price_range*10000:.1f} pips)")

# Initialize SCALPING strategy
print("\n[4/6] Initializing Scalping Strategy...")
print("      Configuration: AGGRESSIVE (More Trades)")
strategy = ForexScalpingStrategy(
    instruments=[instrument],
    max_trades_per_day=10,  # Up to 10 trades/day
    daily_profit_target=0.03,  # 3%
    trade_size_pct=0.10,  # 10% per trade (reduced risk)
    take_profit_pct=0.003,  # 30 pips TP (tighter target - more reachable)
    stop_loss_pct=0.002,  # 20 pips SL (tighter stop)
    trailing_stop_trigger=0.002,  # 20 pips (start earlier)
    trailing_stop_distance=0.001,  # 10 pips (trail closer)
    require_htf_strict=True,  # STRICT: HTF must align (better quality)
    pullback_required=False,  # Don't wait for pullback
    min_consensus_score=1  # Only need 1 signal
)

# Run backtest
print("\n[5/6] Running scalping simulation...")
print("="*80)

balance = 10000
initial_balance = balance
positions = {}
trades = []
daily_trades = 0
daily_start_balance = balance
current_date = None

# Start from bar 100
for i in range(100, len(df_1min)):
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
    df_1min_slice = df_1min.iloc[max(0, i-200):i+1]
    df_5min_slice = df_5min[df_5min['time'] <= current_time].tail(100)
    df_15min_slice = df_15min[df_15min['time'] <= current_time].tail(100)
    df_30min_slice = df_30min[df_30min['time'] <= current_time].tail(100)

    # Check for entry
    if len(positions) == 0 and daily_trades < 10:
        daily_pl_pct = (balance - daily_start_balance) / daily_start_balance

        signal = strategy.should_enter_trade(
            instrument=instrument,
            df_1min=df_1min_slice,
            df_5min=df_5min_slice,
            df_15min=df_15min_slice,
            df_30min=df_30min_slice,
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

            pips_to_tp = abs(take_profit - entry_price) * 10000
            pips_to_sl = abs(stop_loss - entry_price) * 10000

            print(f"\n[ENTRY #{len(trades)+1}] {current_time.strftime('%m-%d %H:%M')} | {position['direction']} @ {entry_price:.5f}")
            print(f"           Size: ${abs(units*entry_price):,.0f} | TP: {pips_to_tp:.0f}p | SL: {pips_to_sl:.0f}p")
            print(f"           {signal['reason']}")

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
        if not pos["trailing_stop_triggered"] and pl_pct >= 0.003:
            pos["trailing_stop_triggered"] = True

        # Check exits
        exit_price = None
        exit_reason = None

        if pos["direction"] == "LONG":
            if current_bar['high'] >= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TP"
            elif current_bar['low'] <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "SL"
            elif pos["trailing_stop_triggered"]:
                trailing_stop = pos["highest_price"] * (1 - 0.002)
                if current_bar['low'] <= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "TRAIL"
        else:
            if current_bar['low'] <= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TP"
            elif current_bar['high'] >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "SL"
            elif pos["trailing_stop_triggered"]:
                trailing_stop = pos["lowest_price"] * (1 + 0.002)
                if current_bar['high'] >= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "TRAIL"

        if exit_price:
            if pos["direction"] == "LONG":
                final_pl = (exit_price - pos["entry_price"]) * pos["units"]
            else:
                final_pl = (pos["entry_price"] - exit_price) * abs(pos["units"])

            final_pl_pct = final_pl / (abs(pos["units"]) * pos["entry_price"])
            balance += final_pl

            pips = abs(exit_price - pos["entry_price"]) * 10000

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
                "balance": balance,
                "pips": pips if final_pl > 0 else -pips
            }

            trades.append(trade)

            print(f"[EXIT  #{len(trades)}] {current_time.strftime('%m-%d %H:%M')} | {exit_reason} @ {exit_price:.5f} | P&L: ${final_pl:+,.0f} ({final_pl_pct*100:+.1f}%) {pips:+.0f}p | Bal: ${balance:,.0f}")

            del positions[instrument]

    # Progress
    if i % 1000 == 0:
        progress = (i - 100) / (len(df_1min) - 100) * 100
        print(f"[PROGRESS] {progress:.0f}% | Balance: ${balance:,.0f} | Trades: {len(trades)}")

# Results
print("\n" + "="*80)
print("[6/6] SCALPING BACKTEST RESULTS")
print("="*80)

if len(trades) == 0:
    print("\nNo trades executed.")
else:
    winning_trades = [t for t in trades if t["pl"] > 0]
    losing_trades = [t for t in trades if t["pl"] < 0]

    total_pl = balance - initial_balance
    total_return = (balance / initial_balance - 1) * 100
    win_rate = len(winning_trades) / len(trades) * 100

    avg_win = np.mean([t["pl"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["pl"] for t in losing_trades]) if losing_trades else 0
    avg_win_pips = np.mean([t["pips"] for t in winning_trades]) if winning_trades else 0
    avg_loss_pips = np.mean([abs(t["pips"]) for t in losing_trades]) if losing_trades else 0

    profit_factor = abs(sum(t["pl"] for t in winning_trades) / sum(t["pl"] for t in losing_trades)) if losing_trades else float('inf')

    # Max drawdown
    equity_values = [initial_balance] + [t["balance"] for t in trades]
    peak = equity_values[0]
    max_dd = 0
    for value in equity_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd

    print(f"\nPerformance:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Winners: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"  Losers: {len(losing_trades)}")
    print(f"  Avg Win: ${avg_win:,.2f} ({avg_win_pips:+.1f} pips)")
    print(f"  Avg Loss: ${avg_loss:,.2f} ({avg_loss_pips:+.1f} pips)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"\nReturns:")
    print(f"  Initial: ${initial_balance:,.2f}")
    print(f"  Final: ${balance:,.2f}")
    print(f"  P&L: ${total_pl:+,.2f}")
    print(f"  Return: {total_return:+.2f}%")

    # Trades per day
    days = (df_1min.iloc[-1]['time'] - df_1min.iloc[0]['time']).days + 1
    trades_per_day = len(trades) / days if days > 0 else 0
    print(f"\nTrading Stats:")
    print(f"  Days: {days}")
    print(f"  Trades/Day: {trades_per_day:.1f}")

print("\n" + "="*80)
print("Backtest Complete!")
print("="*80)
