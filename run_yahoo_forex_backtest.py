"""
Yahoo Finance Forex Backtest - ALL 7 PAIRS
Uses 2 months of 1-hour data from Yahoo Finance
Tests NEW wider stop loss settings
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params

print("=" * 80)
print("YAHOO FINANCE FOREX BACKTEST - 2 MONTHS ALL PAIRS")
print("=" * 80)

# Yahoo Finance symbols
YAHOO_SYMBOLS = {
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "USD_CHF": "USDCHF=X",
    "AUD_USD": "AUDUSD=X",
    "USD_CAD": "USDCAD=X",
    "NZD_USD": "NZDUSD=X"
}

print(f"\n[1/5] Downloading 2 months of hourly data for {len(MAJOR_PAIRS)} pairs...")
all_data = {}

for pair in MAJOR_PAIRS:
    yahoo_symbol = YAHOO_SYMBOLS.get(pair)
    if not yahoo_symbol:
        print(f"  {pair}: No Yahoo symbol mapping")
        continue

    try:
        print(f"  {pair}: Downloading {yahoo_symbol}...")
        ticker = yf.Ticker(yahoo_symbol)

        # Get 2 months of 1-hour data
        df = ticker.history(period='2mo', interval='1h')

        if len(df) == 0:
            print(f"    ERROR: No data returned")
            continue

        # Rename columns to match our format
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'datetime': 'time'})

        # Keep only OHLC
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        all_data[pair] = df

        print(f"    Downloaded {len(df)} bars ({df['time'].min()} to {df['time'].max()})")

    except Exception as e:
        print(f"    ERROR: {e}")

if len(all_data) == 0:
    print("\nERROR: No data downloaded")
    sys.exit(1)

print(f"\n[2/5] Successfully loaded {len(all_data)}/{len(MAJOR_PAIRS)} pairs")

# Find common date range
min_bars = min([len(df) for df in all_data.values()])
print(f"      Common bars: {min_bars}")

# Initialize strategies with NEW wider stops
print("\n[3/5] Initializing strategies with WIDER STOPS...")
strategies = {}

for pair in all_data.keys():
    params = get_scalping_params(pair)

    strategies[pair] = ForexScalpingStrategy(
        instruments=[pair],
        max_trades_per_day=10,
        daily_profit_target=0.05,
        trade_size_pct=0.05,
        take_profit_pct=params["take_profit_pct"],
        stop_loss_pct=params["stop_loss_pct"],
        trailing_stop_trigger=params["trailing_stop_trigger"],
        trailing_stop_distance=params["trailing_stop_distance"],
        require_htf_strict=False,  # Less strict on 1h data
        pullback_required=False,
        min_consensus_score=2  # Require 2+ indicators
    )

    print(f"  {pair}: TP={params['tp_pips']}p SL={params['sl_pips']}p")

# Run backtest
print("\n[4/5] Running backtest simulation...")
print("=" * 80)

balance = 10000
initial_balance = balance
positions = {}
trades = []
daily_trades_per_symbol = {pair: 0 for pair in all_data.keys()}
daily_start_balance = balance
current_date = None
max_concurrent = 5

# Use EUR_USD as time reference
reference_pair = "EUR_USD" if "EUR_USD" in all_data else list(all_data.keys())[0]
df_time_ref = all_data[reference_pair]

for i in range(50, len(df_time_ref)):
    current_time = pd.to_datetime(df_time_ref.iloc[i]['time'])

    # Daily reset
    if current_date is None or current_time.date() > current_date:
        if current_date is not None:
            daily_pl = balance - daily_start_balance
            daily_pl_pct = daily_pl / daily_start_balance * 100
            total_trades = sum(daily_trades_per_symbol.values())
            print(f"\n[DAY] {current_date} | Trades: {total_trades} | P&L: ${daily_pl:+.2f} ({daily_pl_pct:+.2f}%) | Bal: ${balance:,.2f}")

            for pair in all_data.keys():
                if daily_trades_per_symbol[pair] > 0:
                    print(f"      {pair}: {daily_trades_per_symbol[pair]} trades")

        current_date = current_time.date()
        daily_trades_per_symbol = {pair: 0 for pair in all_data.keys()}
        daily_start_balance = balance

    # Check for entries
    if len(positions) < max_concurrent:
        for pair in all_data.keys():
            if pair in positions:
                continue

            if daily_trades_per_symbol[pair] >= 3:
                continue

            df = all_data[pair]

            # Find current bar in this pair's data
            pair_bars = df[df['time'] <= current_time]
            if len(pair_bars) < 50:
                continue

            idx = len(pair_bars) - 1
            current_bar = df.iloc[idx]

            # Get slice (1h bars, use as both 1min and 5min for signal generation)
            df_slice = df.iloc[max(0, idx-200):idx+1].copy()

            if len(df_slice) < 50:
                continue

            # Simple momentum signal on 1h data
            df_slice['sma_20'] = df_slice['close'].rolling(20).mean()
            df_slice['sma_50'] = df_slice['close'].rolling(50).mean()
            df_slice['rsi'] = 50  # Simplified

            last_bar = df_slice.iloc[-1]

            # Generate signal
            signal = None

            # LONG: Price above both MAs and recent momentum
            if (last_bar['close'] > last_bar['sma_20'] and
                last_bar['close'] > last_bar['sma_50'] and
                last_bar['sma_20'] > last_bar['sma_50']):
                signal = {"action": "BUY"}

            # SHORT: Price below both MAs and recent momentum
            elif (last_bar['close'] < last_bar['sma_20'] and
                  last_bar['close'] < last_bar['sma_50'] and
                  last_bar['sma_20'] < last_bar['sma_50']):
                signal = {"action": "SELL"}

            if signal:
                if len(positions) >= max_concurrent:
                    break

                entry_price = current_bar['close']

                # Calculate position size
                units = int((balance * 0.15) / entry_price)  # 15% per position
                if units < 1:
                    units = 1

                # Get stop/target from strategy
                params = get_scalping_params(pair)

                if signal["action"] == "BUY":
                    stop_loss = entry_price * (1 - params["stop_loss_pct"])
                    take_profit = entry_price * (1 + params["take_profit_pct"])
                else:
                    stop_loss = entry_price * (1 + params["stop_loss_pct"])
                    take_profit = entry_price * (1 - params["take_profit_pct"])

                commission = units * entry_price * 0.00002
                balance -= commission

                position = {
                    "direction": "LONG" if signal["action"] == "BUY" else "SHORT",
                    "entry_price": entry_price,
                    "entry_time": current_time,
                    "units": units,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "commission": commission,
                    "highest_price": entry_price if signal["action"] == "BUY" else None,
                    "lowest_price": entry_price if signal["action"] == "SELL" else None,
                }

                positions[pair] = position
                daily_trades_per_symbol[pair] += 1

                pips_to_tp = abs(take_profit - entry_price) * 10000
                pips_to_sl = abs(stop_loss - entry_price) * 10000
                if "JPY" in pair:
                    pips_to_tp /= 100
                    pips_to_sl /= 100

                print(f"\n[ENTRY #{len(trades)+1}] {current_time.strftime('%m-%d %H:%M')} | {pair} {position['direction']} @ {entry_price:.5f}")
                print(f"           Size: ${units*entry_price:,.0f} ({units} units) | TP: {pips_to_tp:.0f}p | SL: {pips_to_sl:.0f}p")

    # Manage positions
    for pair in list(positions.keys()):
        pos = positions[pair]

        df = all_data[pair]
        pair_bars = df[df['time'] <= current_time]

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
        else:
            if current_bar['low'] <= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TP"
            elif current_bar['high'] >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "SL"

        if exit_price:
            if pos["direction"] == "LONG":
                final_pl = (exit_price - pos["entry_price"]) * pos["units"]
            else:
                final_pl = (pos["entry_price"] - exit_price) * pos["units"]

            final_pl_pct = final_pl / (pos["units"] * pos["entry_price"])
            balance += final_pl

            pips = abs(exit_price - pos["entry_price"]) * 10000
            if "JPY" in pair:
                pips /= 100

            trade = {
                "pair": pair,
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "entry_time": pos["entry_time"],
                "exit_price": exit_price,
                "exit_time": current_time,
                "units": pos["units"],
                "pl": final_pl,
                "pl_pct": final_pl_pct,
                "exit_reason": exit_reason,
                "pips": pips if final_pl > 0 else -pips
            }

            trades.append(trade)

            print(f"[EXIT  #{len(trades)}] {current_time.strftime('%m-%d %H:%M')} | {pair} {exit_reason} @ {exit_price:.5f} | P&L: ${final_pl:+.0f} ({final_pl_pct*100:+.1f}%) {pips:+.0f}p | Bal: ${balance:,.0f}")

            del positions[pair]

    # Progress
    if i % 100 == 0:
        progress = (i - 50) / (len(df_time_ref) - 50) * 100
        print(f"[PROGRESS] {progress:.0f}% | Bal: ${balance:,.0f} | Trades: {len(trades)} | Open: {len(positions)}")

# Results
print("\n" + "=" * 80)
print("[5/5] YAHOO FINANCE BACKTEST RESULTS - WIDER STOPS")
print("=" * 80)

if len(trades) == 0:
    print("\nNo trades executed")
else:
    winners = [t for t in trades if t["pl"] > 0]
    losers = [t for t in trades if t["pl"] < 0]

    total_pl = balance - initial_balance
    total_return = (balance / initial_balance - 1) * 100
    win_rate = len(winners) / len(trades) * 100

    avg_win = np.mean([t["pl"] for t in winners]) if winners else 0
    avg_loss = np.mean([t["pl"] for t in losers]) if losers else 0
    avg_win_pips = np.mean([t["pips"] for t in winners]) if winners else 0
    avg_loss_pips = np.mean([abs(t["pips"]) for t in losers]) if losers else 0

    profit_factor = abs(sum(t["pl"] for t in winners) / sum(t["pl"] for t in losers)) if losers else float('inf')

    # Max drawdown
    equity = [initial_balance] + [t["pl"] for t in trades]
    cumulative = np.cumsum(equity)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    # Per-pair stats
    trades_per_pair = {}
    for t in trades:
        pair = t["pair"]
        if pair not in trades_per_pair:
            trades_per_pair[pair] = []
        trades_per_pair[pair].append(t)

    print(f"\nPerformance:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers: {len(losers)}")
    print(f"  Avg Win: ${avg_win:.2f} (+{avg_win_pips:.1f} pips)")
    print(f"  Avg Loss: ${avg_loss:.2f} ({avg_loss_pips:.1f} pips)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")

    print(f"\nReturns:")
    print(f"  Initial: ${initial_balance:,.2f}")
    print(f"  Final: ${balance:,.2f}")
    print(f"  P&L: ${total_pl:+,.2f}")
    print(f"  Return: {total_return:+.2f}%")

    print(f"\nPer-Pair Breakdown:")
    for pair in sorted(trades_per_pair.keys()):
        pair_trades = trades_per_pair[pair]
        pair_wins = [t for t in pair_trades if t["pl"] > 0]
        pair_pl = sum(t["pl"] for t in pair_trades)
        pair_wr = len(pair_wins) / len(pair_trades) * 100 if pair_trades else 0
        print(f"  {pair}: {len(pair_trades)} trades | {len(pair_wins)}W-{len(pair_trades)-len(pair_wins)}L ({pair_wr:.0f}%) | ${pair_pl:+.2f}")

    days = (df_time_ref.iloc[-1]['time'] - df_time_ref.iloc[0]['time']).days
    print(f"\nTrading Stats:")
    print(f"  Period: {days} days (~2 months)")
    print(f"  Trades/Day: {len(trades)/days:.1f}")
    print(f"  Pairs Traded: {len(trades_per_pair)}/{len(all_data)}")

print("\n" + "=" * 80)
print("Backtest Complete!")
print("=" * 80)
