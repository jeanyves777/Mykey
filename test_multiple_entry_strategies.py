"""
Test Multiple Entry Signal Strategies
Find the best performing entry logic, then use it for live trading
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
print("TESTING MULTIPLE ENTRY STRATEGIES - FINDING THE BEST ONE")
print("=" * 90)

# Download 1 week of 15min data
print(f"\n[1/4] Downloading 1 week of 15-minute data for {len(MAJOR_PAIRS)} pairs...")
all_data = {}

for pair in MAJOR_PAIRS[:3]:  # Test on 3 pairs first for speed
    yahoo_symbol = YAHOO_SYMBOLS.get(pair)
    try:
        print(f"  {pair}: ", end="", flush=True)
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period='7d', interval='15m')

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

print(f"\n[2/4] Testing 5 different entry strategies...")
print("=" * 90)

# Define entry strategies
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

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]

    if last['rsi'] < 30:  # Oversold
        return "BUY"
    elif last['rsi'] > 70:  # Overbought
        return "SELL"
    return None

def strategy_3_macd_cross(df):
    """MACD Crossover"""
    df = df.copy()

    # Calculate MACD
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

def strategy_5_momentum_confluence(df):
    """Multi-Indicator Confluence (Best Signals)"""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    last = df.iloc[-1]

    # BULLISH: RSI > 50, MACD > Signal, Price > SMA20 > SMA50
    bullish_score = 0
    if last['rsi'] > 50:
        bullish_score += 1
    if last['macd'] > last['signal']:
        bullish_score += 1
    if last['close'] > last['sma_20'] and last['sma_20'] > last['sma_50']:
        bullish_score += 1

    # BEARISH: RSI < 50, MACD < Signal, Price < SMA20 < SMA50
    bearish_score = 0
    if last['rsi'] < 50:
        bearish_score += 1
    if last['macd'] < last['signal']:
        bearish_score += 1
    if last['close'] < last['sma_20'] and last['sma_20'] < last['sma_50']:
        bearish_score += 1

    # Require 2+ confirmations
    if bullish_score >= 2:
        return "BUY"
    elif bearish_score >= 2:
        return "SELL"
    return None

strategies = {
    "Strategy 1: SMA Cross": strategy_1_sma_cross,
    "Strategy 2: RSI Oversold/Overbought": strategy_2_rsi_oversold,
    "Strategy 3: MACD Cross": strategy_3_macd_cross,
    "Strategy 4: Breakout": strategy_4_breakout,
    "Strategy 5: Momentum Confluence": strategy_5_momentum_confluence,
}

# Run backtest for each strategy
results = {}

for strategy_name, strategy_func in strategies.items():
    print(f"\n{strategy_name}")
    print("-" * 90)

    balance = 10000
    initial_balance = balance
    positions = {}
    trades = []
    leverage = 20

    reference_pair = list(all_data.keys())[0]
    df_ref = all_data[reference_pair]

    for i in range(60, len(df_ref)):
        current_time = pd.to_datetime(df_ref.iloc[i]['time'])

        # Check for entries
        if len(positions) < 3:
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

                # Get signal from strategy
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

    # Results
    if len(trades) > 0:
        winners = [t for t in trades if t["pl"] > 0]
        win_rate = len(winners) / len(trades) * 100
        total_pl = balance - initial_balance

        profit_factor = abs(sum(t["pl"] for t in winners) / sum(t["pl"] for t in trades if t["pl"] < 0)) if any(t["pl"] < 0 for t in trades) else float('inf')

        results[strategy_name] = {
            "trades": len(trades),
            "win_rate": win_rate,
            "pl": total_pl,
            "return_pct": (balance / initial_balance - 1) * 100,
            "profit_factor": profit_factor,
            "balance": balance
        }

        print(f"  Trades: {len(trades)} | Win Rate: {win_rate:.1f}% | P&L: ${total_pl:+,.0f} ({results[strategy_name]['return_pct']:+.2f}%) | PF: {profit_factor:.2f}")
    else:
        results[strategy_name] = {
            "trades": 0,
            "win_rate": 0,
            "pl": 0,
            "return_pct": 0,
            "profit_factor": 0,
            "balance": initial_balance
        }
        print(f"  No trades")

print("\n" + "=" * 90)
print("[3/4] STRATEGY COMPARISON")
print("=" * 90)

# Sort by return
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'Strategy':<40} {'Trades':<10} {'Win Rate':<12} {'Return':<15} {'Profit Factor'}")
print("-" * 90)

for strategy_name, result in sorted_results:
    print(f"{strategy_name:<40} {result['trades']:<10} {result['win_rate']:<11.1f}% "
          f"{result['return_pct']:+11.2f}% {result['profit_factor']:>13.2f}")

print("\n" + "=" * 90)
print("[4/4] RECOMMENDATION")
print("=" * 90)

best_strategy = sorted_results[0]
print(f"\nBEST PERFORMING STRATEGY: {best_strategy[0]}")
print(f"  Win Rate: {best_strategy[1]['win_rate']:.1f}%")
print(f"  Return: {best_strategy[1]['return_pct']:+.2f}%")
print(f"  Profit Factor: {best_strategy[1]['profit_factor']:.2f}")
print(f"  Total Trades: {best_strategy[1]['trades']}")

if best_strategy[1]['return_pct'] > 0:
    print(f"\nRECOMMENDATION: Use '{best_strategy[0]}' for live trading!")
    print("Next step: Run 1-month backtest with this strategy to confirm.")
else:
    print(f"\nWARNING: Even the best strategy is unprofitable ({best_strategy[1]['return_pct']:+.2f}%)")
    print("Consider:")
    print("  - Combine multiple strategies")
    print("  - Add filters (trend, volatility)")
    print("  - Test different timeframes")

print("=" * 90)
