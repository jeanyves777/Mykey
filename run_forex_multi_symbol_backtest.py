"""
Multi-Symbol Forex Scalping Backtest
Tests scalping strategy on ALL MAJOR FOREX PAIRS simultaneously
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

print("=" * 80)
print("MULTI-SYMBOL FOREX SCALPING BACKTEST")
print("=" * 80)
print(f"\nTrading {len(MAJOR_PAIRS)} Major Pairs:")
for pair in MAJOR_PAIRS:
    print(f"  - {pair}")

# Initialize
print("\n[1/6] Connecting to OANDA...")
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')
print(f"      Account Balance: ${client.get_balance():,.2f}")

# Get data for all pairs
print(f"\n[2/6] Downloading historical data for {len(MAJOR_PAIRS)} pairs...")
all_data = {}

for instrument in MAJOR_PAIRS:
    print(f"\n      [{instrument}]")
    try:
        print(f"        Fetching 1-min candles...")
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

        print(f"        Loaded {len(df_1min)} 1-min bars (~{len(df_1min)/1440:.1f} days)")

    except Exception as e:
        print(f"        ERROR: {e}")
        all_data[instrument] = None

# Summary
print(f"\n[3/6] Data Summary:")
valid_pairs = [k for k, v in all_data.items() if v is not None]
print(f"      Successfully loaded: {len(valid_pairs)}/{len(MAJOR_PAIRS)} pairs")

if len(valid_pairs) == 0:
    print("\nERROR: No data loaded. Cannot run backtest.")
    sys.exit(1)

# Find common date range
min_bars = min([len(all_data[pair]["1min"]) for pair in valid_pairs])
print(f"      Common bars available: {min_bars}")

# Initialize strategy for each pair
print("\n[4/6] Initializing Scalping Strategies with pair-specific settings...")
strategies = {}

# Import pair-specific settings
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params

for instrument in valid_pairs:
    # Get pair-specific parameters
    params = get_scalping_params(instrument)

    strategies[instrument] = ForexScalpingStrategy(
        instruments=[instrument],
        max_trades_per_day=10,  # 10 per symbol - aggressive scalping
        daily_profit_target=0.05,  # 5% across all
        trade_size_pct=0.05,  # 5% per trade (lower since multi-symbol)
        take_profit_pct=params["take_profit_pct"],
        stop_loss_pct=params["stop_loss_pct"],
        trailing_stop_trigger=params["trailing_stop_trigger"],
        trailing_stop_distance=params["trailing_stop_distance"],
        require_htf_strict=True,
        pullback_required=True,  # ENABLE PULLBACK DETECTION
        min_consensus_score=1
    )

print(f"      Initialized {len(strategies)} strategies")

# Run backtest
print("\n[5/6] Running multi-symbol backtest...")
print("=" * 80)

balance = 10000
initial_balance = balance
positions = {}
trades = []
daily_trades_per_symbol = {pair: 0 for pair in valid_pairs}
daily_start_balance = balance
current_date = None
max_concurrent = 5  # Max 5 positions at once

# Use EUR_USD as time reference (most liquid)
reference_pair = "EUR_USD" if "EUR_USD" in valid_pairs else valid_pairs[0]
df_time_ref = all_data[reference_pair]["1min"]

# Start from bar 100
for i in range(100, len(df_time_ref)):
    current_time = df_time_ref.iloc[i]['time']

    # Daily reset
    if current_date is None or current_time.date() > current_date:
        if current_date is not None:
            daily_pl = balance - daily_start_balance
            daily_pl_pct = daily_pl / daily_start_balance * 100
            total_trades = sum(daily_trades_per_symbol.values())
            print(f"\n[DAY] {current_date} | Total Trades: {total_trades} | P&L: ${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%) | Balance: ${balance:,.2f}")

            # Show per-symbol breakdown
            for pair in valid_pairs:
                if daily_trades_per_symbol[pair] > 0:
                    print(f"      {pair}: {daily_trades_per_symbol[pair]} trades")

        current_date = current_time.date()
        daily_trades_per_symbol = {pair: 0 for pair in valid_pairs}
        daily_start_balance = balance

    # Check for entries on ALL pairs
    if len(positions) < max_concurrent:  # Only if we have room
        for instrument in valid_pairs:
            # Skip if already have position in this pair
            if instrument in positions:
                continue

            # Skip if daily limit reached for this pair
            if daily_trades_per_symbol[instrument] >= 3:
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
            daily_pl_pct = (balance - daily_start_balance) / daily_start_balance

            signal = strategies[instrument].should_enter_trade(
                instrument=instrument,
                df_1min=df_1min_slice,
                df_5min=df_5min_slice,
                df_15min=df_15min_slice,
                df_30min=df_30min_slice,
                current_positions=len(positions),
                trades_today=daily_trades_per_symbol[instrument],
                daily_pl_pct=daily_pl_pct
            )

            if signal["action"] in ["BUY", "SELL"]:
                # Check if we can still enter
                if len(positions) >= max_concurrent:
                    break

                entry_price = current_bar['close']
                units = strategies[instrument].calculate_position_size(balance, entry_price, instrument)
                stop_loss, take_profit = strategies[instrument].calculate_stop_loss_take_profit(entry_price, signal["action"])

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
                daily_trades_per_symbol[instrument] += 1

                pips_to_tp = abs(take_profit - entry_price) * 10000
                pips_to_sl = abs(stop_loss - entry_price) * 10000

                print(f"\n[ENTRY #{len(trades)+1}] {current_time.strftime('%m-%d %H:%M')} | {instrument} {position['direction']} @ {entry_price:.5f}")
                print(f"           Size: ${abs(units*entry_price):,.0f} | TP: {pips_to_tp:.0f}p | SL: {pips_to_sl:.0f}p | Open: {len(positions)}")

    # Manage all open positions
    for instrument in list(positions.keys()):
        pos = positions[instrument]

        # Get current bar for this pair
        df_1min = all_data[instrument]["1min"]
        pair_bars = df_1min[df_1min['time'] <= current_time]

        if len(pair_bars) == 0:
            continue

        current_bar = pair_bars.iloc[-1]
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
        if not pos["trailing_stop_triggered"] and pl_pct >= 0.002:
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
                trailing_stop = pos["highest_price"] * (1 - 0.001)
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
                trailing_stop = pos["lowest_price"] * (1 + 0.001)
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

            print(f"[EXIT  #{len(trades)}] {current_time.strftime('%m-%d %H:%M')} | {instrument} {exit_reason} @ {exit_price:.5f} | P&L: ${final_pl:+,.0f} ({final_pl_pct*100:+.1f}%) {pips:+.0f}p | Bal: ${balance:,.0f}")

            del positions[instrument]

    # Progress
    if i % 1000 == 0:
        progress = (i - 100) / (len(df_time_ref) - 100) * 100
        print(f"[PROGRESS] {progress:.0f}% | Balance: ${balance:,.0f} | Trades: {len(trades)} | Open: {len(positions)}")

# Results
print("\n" + "=" * 80)
print("[6/6] MULTI-SYMBOL BACKTEST RESULTS")
print("=" * 80)

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

    # Per-symbol stats
    trades_per_symbol = {}
    for t in trades:
        symbol = t["instrument"]
        if symbol not in trades_per_symbol:
            trades_per_symbol[symbol] = []
        trades_per_symbol[symbol].append(t)

    print(f"\nOverall Performance:")
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

    print(f"\nPer-Symbol Breakdown:")
    for symbol in sorted(trades_per_symbol.keys()):
        symbol_trades = trades_per_symbol[symbol]
        symbol_wins = [t for t in symbol_trades if t["pl"] > 0]
        symbol_pl = sum(t["pl"] for t in symbol_trades)
        symbol_wr = len(symbol_wins) / len(symbol_trades) * 100 if symbol_trades else 0
        print(f"  {symbol}: {len(symbol_trades)} trades | {len(symbol_wins)}W-{len(symbol_trades)-len(symbol_wins)}L ({symbol_wr:.0f}%) | ${symbol_pl:+,.2f}")

    # Trading stats
    days = (df_time_ref.iloc[-1]['time'] - df_time_ref.iloc[0]['time']).days + 1
    trades_per_day = len(trades) / days if days > 0 else 0
    print(f"\nTrading Stats:")
    print(f"  Days: {days}")
    print(f"  Trades/Day: {trades_per_day:.1f}")
    print(f"  Symbols Traded: {len(trades_per_symbol)}/{len(valid_pairs)}")

print("\n" + "=" * 80)
print("Multi-Symbol Backtest Complete!")
print("=" * 80)
