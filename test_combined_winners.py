"""
BACKTEST: COMBINING THE 3 WINNING STRATEGIES
Range Scalping + Bollinger Bands + RSI Oversold
Testing if combining signals improves results
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
print("BACKTEST: COMBINED WINNERS - Range + Bollinger + RSI")
print("=" * 100)

# Download data
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

# ===========================
# COMBINED STRATEGY VARIATIONS
# ===========================

def combined_strategy_v1_any_signal(df):
    """Version 1: ANY of the 3 signals triggers entry (OR logic)
    Most aggressive - highest trade frequency
    """
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    period = 20
    df['bb_sma'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_sma'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_sma'] - (2 * df['bb_std'])

    # Range levels
    lookback = 50
    df['support'] = df['low'].rolling(lookback).min()
    df['resistance'] = df['high'].rolling(lookback).max()

    # Stochastic
    stoch_period = 14
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BULLISH signals
    rsi_buy = last['rsi'] < 30
    bb_buy = last['close'] <= last['bb_lower'] and prev['close'] > prev['bb_lower']

    dist_to_support = (last['close'] - last['support']) / last['support']
    range_buy = dist_to_support < 0.002 and last['rsi'] < 35 and last['stoch'] < 35

    if rsi_buy or bb_buy or range_buy:
        return "BUY"

    # BEARISH signals
    rsi_sell = last['rsi'] > 70
    bb_sell = last['close'] >= last['bb_upper'] and prev['close'] < prev['bb_upper']

    dist_to_resistance = (last['resistance'] - last['close']) / last['close']
    range_sell = dist_to_resistance < 0.002 and last['rsi'] > 65 and last['stoch'] > 65

    if rsi_sell or bb_sell or range_sell:
        return "SELL"

    return None


def combined_strategy_v2_two_signals(df):
    """Version 2: ANY 2 of the 3 signals must agree (Moderate)
    Requires 2/3 confirmation
    """
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    period = 20
    df['bb_sma'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_sma'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_sma'] - (2 * df['bb_std'])

    # Range levels
    lookback = 50
    df['support'] = df['low'].rolling(lookback).min()
    df['resistance'] = df['high'].rolling(lookback).max()

    # Stochastic
    stoch_period = 14
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BULLISH signals
    bullish_count = 0

    if last['rsi'] < 30:
        bullish_count += 1

    if last['close'] <= last['bb_lower'] and prev['close'] > prev['bb_lower']:
        bullish_count += 1

    dist_to_support = (last['close'] - last['support']) / last['support']
    if dist_to_support < 0.002 and last['rsi'] < 35 and last['stoch'] < 35:
        bullish_count += 1

    if bullish_count >= 2:
        return "BUY"

    # BEARISH signals
    bearish_count = 0

    if last['rsi'] > 70:
        bearish_count += 1

    if last['close'] >= last['bb_upper'] and prev['close'] < prev['bb_upper']:
        bearish_count += 1

    dist_to_resistance = (last['resistance'] - last['close']) / last['close']
    if dist_to_resistance < 0.002 and last['rsi'] > 65 and last['stoch'] > 65:
        bearish_count += 1

    if bearish_count >= 2:
        return "SELL"

    return None


def combined_strategy_v3_all_signals(df):
    """Version 3: ALL 3 signals must agree (AND logic)
    Most conservative - highest quality signals
    """
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    period = 20
    df['bb_sma'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_sma'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_sma'] - (2 * df['bb_std'])

    # Range levels
    lookback = 50
    df['support'] = df['low'].rolling(lookback).min()
    df['resistance'] = df['high'].rolling(lookback).max()

    # Stochastic
    stoch_period = 14
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BULLISH: ALL 3 must agree
    rsi_buy = last['rsi'] < 30
    bb_buy = last['close'] <= last['bb_lower'] and prev['close'] > prev['bb_lower']

    dist_to_support = (last['close'] - last['support']) / last['support']
    range_buy = dist_to_support < 0.002 and last['rsi'] < 35 and last['stoch'] < 35

    if rsi_buy and bb_buy and range_buy:
        return "BUY"

    # BEARISH: ALL 3 must agree
    rsi_sell = last['rsi'] > 70
    bb_sell = last['close'] >= last['bb_upper'] and prev['close'] < prev['bb_upper']

    dist_to_resistance = (last['resistance'] - last['close']) / last['close']
    range_sell = dist_to_resistance < 0.002 and last['rsi'] > 65 and last['stoch'] > 65

    if rsi_sell and bb_sell and range_sell:
        return "SELL"

    return None


def combined_strategy_v4_weighted(df):
    """Version 4: Weighted scoring system
    Range Scalping: 40% (best performer)
    Bollinger Bands: 35% (highest win rate)
    RSI: 25%
    """
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    period = 20
    df['bb_sma'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_sma'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_sma'] - (2 * df['bb_std'])

    # Range levels
    lookback = 50
    df['support'] = df['low'].rolling(lookback).min()
    df['resistance'] = df['high'].rolling(lookback).max()

    # Stochastic
    stoch_period = 14
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BULLISH scoring
    bullish_score = 0

    # RSI (25%)
    if last['rsi'] < 30:
        bullish_score += 25

    # Bollinger Bands (35%)
    if last['close'] <= last['bb_lower'] and prev['close'] > prev['bb_lower']:
        bullish_score += 35

    # Range Scalping (40%)
    dist_to_support = (last['close'] - last['support']) / last['support']
    if dist_to_support < 0.002 and last['rsi'] < 35 and last['stoch'] < 35:
        bullish_score += 40

    if bullish_score >= 60:  # Need 60%+ score to enter
        return "BUY"

    # BEARISH scoring
    bearish_score = 0

    # RSI (25%)
    if last['rsi'] > 70:
        bearish_score += 25

    # Bollinger Bands (35%)
    if last['close'] >= last['bb_upper'] and prev['close'] < prev['bb_upper']:
        bearish_score += 35

    # Range Scalping (40%)
    dist_to_resistance = (last['resistance'] - last['close']) / last['close']
    if dist_to_resistance < 0.002 and last['rsi'] > 65 and last['stoch'] > 65:
        bearish_score += 40

    if bearish_score >= 60:
        return "SELL"

    return None


# ===========================
# BACKTEST ENGINE
# ===========================

strategies = {
    "Combined V1: ANY Signal (OR)": combined_strategy_v1_any_signal,
    "Combined V2: 2 of 3 Signals": combined_strategy_v2_two_signals,
    "Combined V3: ALL Signals (AND)": combined_strategy_v3_all_signals,
    "Combined V4: Weighted Score (60%+)": combined_strategy_v4_weighted,
}

print(f"\n[2/3] Testing {len(strategies)} combined strategy variations...")
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

                    # Position sizing with leverage
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
print("[3/3] FINAL COMPARISON - COMBINED VS INDIVIDUAL STRATEGIES")
print("=" * 100)

# Sort by return
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'Strategy':<45} {'Trades':<9} {'Win Rate':<11} {'Return':<14} {'PF':<7} {'Max DD'}")
print("-" * 100)

for strategy_name, result in sorted_results:
    print(f"{strategy_name:<45} {result['trades']:<9} {result['win_rate']:<10.1f}% "
          f"{result['return_pct']:+12.2f}% {result['profit_factor']:>6.2f} {result['max_dd']:>6.1f}%")

print("\n" + "=" * 100)
print("COMPARISON WITH INDIVIDUAL STRATEGIES:")
print("-" * 100)
print("  Range Scalping (individual):     +22.39% | 51.1% WR | 1.27 PF | 8.8% DD | 309 trades")
print("  Bollinger Bands (individual):    +20.69% | 53.2% WR | 1.39 PF | 5.3% DD | 218 trades")
print("  RSI Oversold (individual):       +8.15%  | 48.3% WR | 1.11 PF | 7.1% DD | 269 trades")
print("=" * 100)

best_strategy = sorted_results[0]
print(f"\nBEST COMBINED STRATEGY: {best_strategy[0]}")
print(f"  Trades: {best_strategy[1]['trades']}")
print(f"  Win Rate: {best_strategy[1]['win_rate']:.1f}%")
print(f"  Return: {best_strategy[1]['return_pct']:+.2f}%")
print(f"  Profit Factor: {best_strategy[1]['profit_factor']:.2f}")
print(f"  Max Drawdown: {best_strategy[1]['max_dd']:.1f}%")

if best_strategy[1]['return_pct'] > 22.39:  # Beat Range Scalping
    print(f"\nRESULT: COMBINED STRATEGY WINS!")
    print(f"Improvement: {best_strategy[1]['return_pct'] - 22.39:+.2f}% over best individual strategy")
    print(f"Next step: Implement combined strategy in forex_scalping.py")
else:
    print(f"\nRESULT: Individual Range Scalping still performs better")
    print(f"Recommendation: Use Range Scalping alone (+22.39%)")

print("=" * 100)
