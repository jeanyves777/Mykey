"""
1-MONTH BACKTEST - MACD CROSS STRATEGY
Confirming the winning strategy on all 7 pairs with 1 month of data
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params

YAHOO_SYMBOLS = {
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "USD_CHF": "USDCHF=X",
    "AUD_USD": "AUDUSD=X",
    "USD_CAD": "USDCAD=X",
    "NZD_USD": "NZDUSD=X"
}

print("=" * 90)
print("1-MONTH BACKTEST - MACD CROSS STRATEGY (ALL 7 PAIRS)")
print("=" * 90)

def macd_cross_signal(df):
    """MACD Crossover Entry Signal"""
    df = df.copy()

    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # MACD crosses above signal = BUY
    if last['macd'] > last['signal'] and prev['macd'] <= prev['signal']:
        return "BUY"
    # MACD crosses below signal = SELL
    elif last['macd'] < last['signal'] and prev['macd'] >= prev['signal']:
        return "SELL"
    return None

# Download 1 month of 15min data
print(f"\n[1/4] Downloading 1 month of 15-minute data for {len(MAJOR_PAIRS)} pairs...")
all_data = {}

for pair in MAJOR_PAIRS:
    yahoo_symbol = YAHOO_SYMBOLS.get(pair)
    try:
        print(f"  {pair}: ", end="", flush=True)
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period='1mo', interval='15m')

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

print(f"\n[3/4] Running 1-month backtest with MACD Cross strategy...")
print("=" * 90)

balance = 10000
initial_balance = balance
positions = {}
trades = []
leverage = 20
max_concurrent = 5

reference_pair = list(all_data.keys())[0]
df_ref = all_data[reference_pair]

entry_count = 0
exit_count = 0

for i in range(60, len(df_ref)):
    current_time = pd.to_datetime(df_ref.iloc[i]['time'])

    # Check for entries
    if len(positions) < max_concurrent:
        for pair in all_data.keys():
            if pair in positions:
                continue

            df = all_data[pair]
            pair_bars = df[df['time'] <= current_time]
            if len(pair_bars) < 60:
                continue

            idx = len(pair_bars) - 1
            df_slice = df.iloc[max(0, idx-60):idx+1].copy()

            if len(df_slice) < 50:
                continue

            # MACD Cross signal
            signal = macd_cross_signal(df_slice)

            if signal:
                current_bar = df.iloc[idx]
                entry_price = current_bar['close']

                # Position sizing with leverage
                target_margin = balance * 0.15
                notional_value = target_margin * leverage
                units = int(notional_value / entry_price)
                if units < 1:
                    units = 1

                # Stops (wider)
                params = get_scalping_params(pair)
                if signal == "BUY":
                    stop_loss = entry_price * (1 - params["stop_loss_pct"])
                    take_profit = entry_price * (1 + params["take_profit_pct"])
                else:
                    stop_loss = entry_price * (1 + params["stop_loss_pct"])
                    take_profit = entry_price * (1 - params["take_profit_pct"])

                positions[pair] = {
                    "direction": "LONG" if signal == "BUY" else "SHORT",
                    "entry_price": entry_price,
                    "entry_time": current_time,
                    "units": units,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

                entry_count += 1
                if entry_count <= 10:  # Show first 10 entries
                    pips_tp = abs(take_profit - entry_price) * (100 if "JPY" in pair else 10000)
                    pips_sl = abs(stop_loss - entry_price) * (100 if "JPY" in pair else 10000)
                    print(f"[ENTRY #{entry_count}] {current_time.strftime('%m-%d %H:%M')} | {pair} {positions[pair]['direction']} @ {entry_price:.5f}")

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

            exit_count += 1
            if exit_count <= 10:  # Show first 10 exits
                print(f"[EXIT  #{exit_count}] {current_time.strftime('%m-%d %H:%M')} | {pair} {exit_reason} | ${pl:+.0f} ({pips:+.0f}p)")

            del positions[pair]

    # Progress every 500 bars
    if i % 500 == 0:
        progress = (i - 60) / (len(df_ref) - 60) * 100
        print(f"[PROGRESS] {progress:.0f}% | Bal: ${balance:,.0f} | Trades: {len(trades)} | Open: {len(positions)}")

print("\n" + "=" * 90)
print("[4/4] 1-MONTH MACD CROSS BACKTEST RESULTS")
print("=" * 90)

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

    # Max drawdown
    equity_curve = [initial_balance]
    running_balance = initial_balance
    for t in trades:
        running_balance += t["pl"]
        equity_curve.append(running_balance)

    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd

    print(f"  Max Drawdown: {max_dd:.2f}%")

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

    # Calculate days
    first_time = pd.to_datetime(df_ref.iloc[60]['time'])
    last_time = pd.to_datetime(df_ref.iloc[-1]['time'])
    days = (last_time - first_time).days

    print(f"\nTrading Stats:")
    print(f"  Period: {days} days")
    print(f"  Trades/Day: {len(trades)/days:.1f}")
    print(f"  Pairs Traded: {len(trades_per_pair)}/{len(all_data)}")

    print("\n" + "=" * 90)
    if total_pl > 0 and win_rate > 50 and profit_factor > 1.0:
        print("SUCCESS: MACD Cross strategy is PROFITABLE and ready for live trading!")
        print(f"  Win Rate: {win_rate:.1f}% (> 50%)")
        print(f"  Profit Factor: {profit_factor:.2f} (> 1.0)")
        print(f"  Return: {(balance/initial_balance-1)*100:+.2f}%")
        print("\nNext step: Update forex_scalping.py to use MACD Cross entry signals")
    else:
        print("NEEDS IMPROVEMENT:")
        if win_rate <= 50:
            print(f"  - Win rate {win_rate:.1f}% is too low (need > 50%)")
        if profit_factor <= 1.0:
            print(f"  - Profit factor {profit_factor:.2f} is too low (need > 1.0)")
        if total_pl <= 0:
            print(f"  - Strategy is unprofitable (${total_pl:+,.2f})")
        print("\nConsider combining MACD with trend filter or other confirmations")
    print("=" * 90)
