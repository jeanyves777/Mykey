"""
Quick Yahoo Finance Backtest - 1 WEEK, ALL 7 PAIRS
Tests NEW settings: Wider stops (16-25 pips) + Leverage-based sizing (20:1)
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params

print("=" * 80)
print("QUICK YAHOO BACKTEST - 1 WEEK WITH NEW SETTINGS")
print("=" * 80)
print("\nNEW SETTINGS:")
print("  - Stop Loss: 12-25 pips (WIDER - was 6-12 pips)")
print("  - Take Profit: 15-30 pips (WIDER - was 10-20 pips)")
print("  - Position Size: 20:1 leverage (15% margin usage)")
print()

YAHOO_SYMBOLS = {
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "USD_CHF": "USDCHF=X",
    "AUD_USD": "AUDUSD=X",
    "USD_CAD": "USDCAD=X",
    "NZD_USD": "NZDUSD=X"
}

print(f"[1/4] Downloading 1 week of 5-minute data for {len(MAJOR_PAIRS)} pairs...")
all_data = {}

for pair in MAJOR_PAIRS:
    yahoo_symbol = YAHOO_SYMBOLS.get(pair)
    if not yahoo_symbol:
        continue

    try:
        print(f"  {pair}: ", end="", flush=True)
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period='7d', interval='5m')

        if len(df) == 0:
            print("NO DATA")
            continue

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'datetime': 'time'})
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        all_data[pair] = df
        print(f"{len(df)} bars")

    except Exception as e:
        print(f"ERROR: {e}")

print(f"\n[2/4] Loaded {len(all_data)}/{len(MAJOR_PAIRS)} pairs")

if len(all_data) == 0:
    print("ERROR: No data")
    sys.exit(1)

print(f"\n[3/4] Running backtest with NEW settings...")
print("=" * 80)

balance = 10000
initial_balance = balance
positions = {}
trades = []
leverage = 20  # OANDA leverage

# Get date range
reference_pair = "EUR_USD" if "EUR_USD" in all_data else list(all_data.keys())[0]
df_ref = all_data[reference_pair]

for i in range(20, len(df_ref)):
    current_time = pd.to_datetime(df_ref.iloc[i]['time'])

    # Check for entries
    if len(positions) < 5:
        for pair in all_data.keys():
            if pair in positions:
                continue

            df = all_data[pair]
            pair_bars = df[df['time'] <= current_time]
            if len(pair_bars) < 20:
                continue

            idx = len(pair_bars) - 1
            current_bar = df.iloc[idx]
            df_slice = df.iloc[max(0, idx-50):idx+1].copy()

            # Simple momentum signal
            df_slice['sma_10'] = df_slice['close'].rolling(10).mean()
            df_slice['sma_20'] = df_slice['close'].rolling(20).mean()

            last = df_slice.iloc[-1]
            signal = None

            if last['close'] > last['sma_10'] and last['sma_10'] > last['sma_20']:
                signal = {"action": "BUY"}
            elif last['close'] < last['sma_10'] and last['sma_10'] < last['sma_20']:
                signal = {"action": "SELL"}

            if signal:
                entry_price = current_bar['close']

                # NEW: Leverage-based position sizing
                target_margin = balance * 0.15
                notional_value = target_margin * leverage
                units = int(notional_value / entry_price)
                if units < 1:
                    units = 1

                # NEW: Wider stops
                params = get_scalping_params(pair)
                if signal["action"] == "BUY":
                    stop_loss = entry_price * (1 - params["stop_loss_pct"])
                    take_profit = entry_price * (1 + params["take_profit_pct"])
                else:
                    stop_loss = entry_price * (1 + params["stop_loss_pct"])
                    take_profit = entry_price * (1 - params["take_profit_pct"])

                positions[pair] = {
                    "direction": "LONG" if signal["action"] == "BUY" else "SHORT",
                    "entry_price": entry_price,
                    "entry_time": current_time,
                    "units": units,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

                pips_tp = abs(take_profit - entry_price) * (100 if "JPY" in pair else 10000)
                pips_sl = abs(stop_loss - entry_price) * (100 if "JPY" in pair else 10000)

                print(f"\n[ENTRY #{len(trades)+1}] {current_time.strftime('%m-%d %H:%M')} | {pair} {positions[pair]['direction']}")
                print(f"  {units:,} units @ {entry_price:.5f} | TP: {pips_tp:.0f}p | SL: {pips_sl:.0f}p")

    # Manage positions
    for pair in list(positions.keys()):
        pos = positions[pair]
        df = all_data[pair]
        pair_bars = df[df['time'] <= current_time]

        if len(pair_bars) == 0:
            continue

        current_bar = pair_bars.iloc[-1]

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
                pl = (exit_price - pos["entry_price"]) * pos["units"]
            else:
                pl = (pos["entry_price"] - exit_price) * pos["units"]

            balance += pl

            pips = abs(exit_price - pos["entry_price"]) * (100 if "JPY" in pair else 10000)
            if pl < 0:
                pips = -pips

            trades.append({
                "pair": pair,
                "pl": pl,
                "pips": pips,
                "exit_reason": exit_reason
            })

            print(f"[EXIT  #{len(trades)}] {current_time.strftime('%m-%d %H:%M')} | {pair} {exit_reason} | ${pl:+.0f} ({pips:+.0f}p)")

            del positions[pair]

print("\n" + "=" * 80)
print("[4/4] BACKTEST RESULTS - NEW WIDER STOPS & LEVERAGE SIZING")
print("=" * 80)

if len(trades) == 0:
    print("\nNo trades executed")
else:
    winners = [t for t in trades if t["pl"] > 0]
    losers = [t for t in trades if t["pl"] < 0]

    total_pl = balance - initial_balance
    win_rate = len(winners) / len(trades) * 100

    avg_win = np.mean([t["pl"] for t in winners]) if winners else 0
    avg_loss = np.mean([t["pl"] for t in losers]) if losers else 0
    avg_win_pips = np.mean([t["pips"] for t in winners]) if winners else 0
    avg_loss_pips = np.mean([abs(t["pips"]) for t in losers]) if losers else 0

    profit_factor = abs(sum(t["pl"] for t in winners) / sum(t["pl"] for t in losers)) if losers else float('inf')

    # Per-pair breakdown
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

    print(f"\nReturns:")
    print(f"  Initial: ${initial_balance:,.2f}")
    print(f"  Final: ${balance:,.2f}")
    print(f"  P&L: ${total_pl:+,.2f}")
    print(f"  Return: {(balance/initial_balance-1)*100:+.2f}%")

    print(f"\nPer-Pair Breakdown:")
    for pair in sorted(trades_per_pair.keys()):
        pair_trades = trades_per_pair[pair]
        pair_wins = [t for t in pair_trades if t["pl"] > 0]
        pair_pl = sum(t["pl"] for t in pair_trades)
        pair_wr = len(pair_wins) / len(pair_trades) * 100 if pair_trades else 0
        print(f"  {pair}: {len(pair_trades)} trades | {len(pair_wins)}W-{len(pair_trades)-len(pair_wins)}L ({pair_wr:.0f}%) | ${pair_pl:+.2f}")

print("\n" + "=" * 80)
print("NEW SETTINGS SUMMARY:")
print(f"  EUR/USD: TP={get_scalping_params('EUR_USD')['tp_pips']}p SL={get_scalping_params('EUR_USD')['sl_pips']}p")
print(f"  GBP/USD: TP={get_scalping_params('GBP_USD')['tp_pips']}p SL={get_scalping_params('GBP_USD')['sl_pips']}p")
print(f"  USD/JPY: TP={get_scalping_params('USD_JPY')['tp_pips']}p SL={get_scalping_params('USD_JPY')['sl_pips']}p")
print("  Position sizing: 15% margin @ 20:1 leverage")
print("=" * 80)
