"""
1-MONTH BACKTEST - ALL 5 STRATEGIES - ALL 7 PAIRS
Comprehensive test to find the best strategy
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

print("=" * 100)
print("1-MONTH BACKTEST - ALL 5 STRATEGIES - ALL 7 PAIRS")
print("=" * 100)

# Download 1 month of 15min data
print(f"\n[1/3] Downloading 1 month of 15-minute data for {len(MAJOR_PAIRS)} pairs...")
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

print(f"\nLoaded {len(all_data)}/{len(MAJOR_PAIRS)} pairs")

if len(all_data) == 0:
    print("ERROR: No data")
    sys.exit(1)

# Define all 5 strategies
def strategy_1_sma_cross(df):
    """Simple MA Crossover"""
    df = df.copy()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last['sma_10'] > last['sma_20'] and prev['sma_10'] <= prev['sma_20']:
        return "BUY"
    elif last['sma_10'] < last['sma_20'] and prev['sma_10'] >= prev['sma_20']:
        return "SELL"
    return None

def strategy_2_rsi_oversold(df):
    """RSI Oversold/Overbought"""
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    last = df.iloc[-1]
    if last['rsi'] < 30:
        return "BUY"
    elif last['rsi'] > 70:
        return "SELL"
    return None

def strategy_3_macd_cross(df):
    """MACD Crossover"""
    df = df.copy()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last['macd'] > last['signal'] and prev['macd'] <= prev['signal']:
        return "BUY"
    elif last['macd'] < last['signal'] and prev['macd'] >= prev['signal']:
        return "SELL"
    return None

def strategy_4_breakout(df):
    """Price Breakout (20-bar high/low)"""
    df = df.copy()
    df['highest_20'] = df['high'].rolling(20).max()
    df['lowest_20'] = df['low'].rolling(20).min()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last['close'] > prev['highest_20']:
        return "BUY"
    elif last['close'] < prev['lowest_20']:
        return "SELL"
    return None

def strategy_5_macd_trend_filter(df):
    """MACD + Trend Filter (IMPROVED)"""
    df = df.copy()

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Trend filter (50 SMA)
    df['sma_50'] = df['close'].rolling(50).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # MACD cross + trend confirmation
    macd_cross_up = last['macd'] > last['signal'] and prev['macd'] <= prev['signal']
    macd_cross_down = last['macd'] < last['signal'] and prev['macd'] >= prev['signal']

    # Only trade in direction of trend
    if macd_cross_up and last['close'] > last['sma_50']:
        return "BUY"
    elif macd_cross_down and last['close'] < last['sma_50']:
        return "SELL"
    return None

strategies = {
    "Strategy 1: SMA Cross": strategy_1_sma_cross,
    "Strategy 2: RSI Oversold": strategy_2_rsi_oversold,
    "Strategy 3: MACD Cross": strategy_3_macd_cross,
    "Strategy 4: Breakout": strategy_4_breakout,
    "Strategy 5: MACD + Trend Filter": strategy_5_macd_trend_filter,
}

print(f"\n[2/3] Testing {len(strategies)} strategies over 1 month...")
print("=" * 100)

results = {}

for strategy_name, strategy_func in strategies.items():
    print(f"\n{strategy_name}:")
    print("-" * 100)

    balance = 10000
    initial_balance = balance
    positions = {}
    trades = []
    leverage = 20
    max_concurrent = 5

    reference_pair = list(all_data.keys())[0]
    df_ref = all_data[reference_pair]

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

                # Get signal
                signal = strategy_func(df_slice)

                if signal:
                    current_bar = df.iloc[idx]
                    entry_price = current_bar['close']

                    # Position sizing
                    target_margin = balance * 0.15
                    notional_value = target_margin * leverage
                    units = int(notional_value / entry_price)
                    if units < 1:
                        units = 1

                    # Stops
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

                del positions[pair]

        # Progress
        if i % 500 == 0:
            progress = (i - 60) / (len(df_ref) - 60) * 100
            print(f"  {progress:.0f}% | Bal: ${balance:,.0f} | Trades: {len(trades)}", end="\r")

    # Calculate results
    if len(trades) > 0:
        winners = [t for t in trades if t["pl"] > 0]
        win_rate = len(winners) / len(trades) * 100
        total_pl = balance - initial_balance

        profit_factor = abs(sum(t["pl"] for t in winners) / sum(t["pl"] for t in trades if t["pl"] < 0)) if any(t["pl"] < 0 for t in trades) else float('inf')

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

        results[strategy_name] = {
            "trades": len(trades),
            "win_rate": win_rate,
            "pl": total_pl,
            "return_pct": (balance / initial_balance - 1) * 100,
            "profit_factor": profit_factor,
            "max_dd": max_dd,
            "balance": balance
        }

        print(f"  100% DONE | Trades: {len(trades)} | Win Rate: {win_rate:.1f}% | P&L: ${total_pl:+,.0f} ({results[strategy_name]['return_pct']:+.2f}%) | PF: {profit_factor:.2f} | DD: {max_dd:.1f}%")
    else:
        results[strategy_name] = {
            "trades": 0,
            "win_rate": 0,
            "pl": 0,
            "return_pct": 0,
            "profit_factor": 0,
            "max_dd": 0,
            "balance": initial_balance
        }
        print(f"  100% DONE | No trades")

print("\n" + "=" * 100)
print("[3/3] FINAL COMPARISON - ALL STRATEGIES")
print("=" * 100)

# Sort by return
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'Strategy':<35} {'Trades':<9} {'Win Rate':<11} {'Return':<14} {'PF':<7} {'Max DD'}")
print("-" * 100)

for strategy_name, result in sorted_results:
    print(f"{strategy_name:<35} {result['trades']:<9} {result['win_rate']:<10.1f}% "
          f"{result['return_pct']:+12.2f}% {result['profit_factor']:>6.2f} {result['max_dd']:>6.1f}%")

print("\n" + "=" * 100)
print("WINNER")
print("=" * 100)

best_strategy = sorted_results[0]
print(f"\nBEST STRATEGY: {best_strategy[0]}")
print(f"  Trades: {best_strategy[1]['trades']}")
print(f"  Win Rate: {best_strategy[1]['win_rate']:.1f}%")
print(f"  Return: {best_strategy[1]['return_pct']:+.2f}%")
print(f"  Profit Factor: {best_strategy[1]['profit_factor']:.2f}")
print(f"  Max Drawdown: {best_strategy[1]['max_dd']:.1f}%")

if best_strategy[1]['return_pct'] > 0 and best_strategy[1]['win_rate'] > 45:
    print(f"\nSTATUS: PROFITABLE AND READY FOR LIVE TRADING")
    print(f"Next step: Implement '{best_strategy[0]}' in forex_scalping.py")
else:
    print(f"\nSTATUS: NEEDS MORE WORK")
    if best_strategy[1]['return_pct'] <= 0:
        print(f"  - Still unprofitable ({best_strategy[1]['return_pct']:+.2f}%)")
    if best_strategy[1]['win_rate'] <= 45:
        print(f"  - Win rate too low ({best_strategy[1]['win_rate']:.1f}%)")
    print("\nRecommendations:")
    print("  1. Add more filters (volatility, time of day)")
    print("  2. Optimize entry timing")
    print("  3. Consider only trading best performing pairs")

print("=" * 100)
