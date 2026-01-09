"""
BACKTEST: COMBINED STRATEGY WITH PULLBACK LOGIC
Testing if adding pullback entry improves the winning Combined V2 strategy
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
print("BACKTEST: COMBINED STRATEGY + PULLBACK LOGIC")
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
# STRATEGY DEFINITIONS
# ===========================

def check_pullback(df, signal_direction):
    """
    Check if price has pulled back after initial signal
    BULLISH: Price pulled back to EMA after initial push up
    BEARISH: Price pulled back to EMA after initial push down
    """
    if len(df) < 20:
        return False

    close = df['close']
    ema9 = close.ewm(span=9, adjust=False).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if signal_direction == "BUY":
        # For bullish: Price should be near or touching EMA from above
        # This means we're buying on a dip in an uptrend
        price_near_ema = abs(last['close'] - ema9.iloc[-1]) / last['close'] < 0.0003  # Within 3 pips
        price_above_ema = last['close'] >= ema9.iloc[-1]

        # Check if there was a recent pullback (price dipped but stayed above EMA)
        pullback_happened = (prev['close'] < df.iloc[-3]['close'] and
                            last['close'] > prev['close'])

        return (price_near_ema and price_above_ema) or pullback_happened

    else:  # SELL
        # For bearish: Price should be near or touching EMA from below
        price_near_ema = abs(last['close'] - ema9.iloc[-1]) / last['close'] < 0.0003
        price_below_ema = last['close'] <= ema9.iloc[-1]

        # Check if there was a recent pullback (price bounced but stayed below EMA)
        pullback_happened = (prev['close'] > df.iloc[-3]['close'] and
                            last['close'] < prev['close'])

        return (price_near_ema and price_below_ema) or pullback_happened


def combined_strategy_no_pullback(df):
    """ORIGINAL: Combined V2 - 2 of 3 Signals (NO pullback requirement)"""
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


def combined_strategy_with_pullback(df):
    """NEW: Combined V2 + Pullback Requirement"""
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
        # CHECK PULLBACK before entering
        if check_pullback(df, "BUY"):
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
        # CHECK PULLBACK before entering
        if check_pullback(df, "SELL"):
            return "SELL"

    return None


# ===========================
# BACKTEST ENGINE
# ===========================

strategies = {
    "Combined V2: No Pullback (Original)": combined_strategy_no_pullback,
    "Combined V2: With Pullback": combined_strategy_with_pullback,
}

print(f"\n[2/3] Testing combined strategy with/without pullback...")
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
print("[3/3] RESULTS: PULLBACK vs NO PULLBACK")
print("=" * 100)

# Sort by return
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'Strategy':<45} {'Trades':<9} {'Win Rate':<11} {'Return':<14} {'PF':<7} {'Max DD'}")
print("-" * 100)

for strategy_name, result in sorted_results:
    print(f"{strategy_name:<45} {result['trades']:<9} {result['win_rate']:<10.1f}% "
          f"{result['return_pct']:+12.2f}% {result['profit_factor']:>6.2f} {result['max_dd']:>6.1f}%")

print("\n" + "=" * 100)

no_pullback = results["Combined V2: No Pullback (Original)"]
with_pullback = results["Combined V2: With Pullback"]

print("COMPARISON:")
print("-" * 100)
print(f"WITHOUT Pullback: {no_pullback['trades']} trades | {no_pullback['win_rate']:.1f}% WR | {no_pullback['return_pct']:+.2f}% return | {no_pullback['profit_factor']:.2f} PF")
print(f"WITH Pullback:    {with_pullback['trades']} trades | {with_pullback['win_rate']:.1f}% WR | {with_pullback['return_pct']:+.2f}% return | {with_pullback['profit_factor']:.2f} PF")

if with_pullback['return_pct'] > no_pullback['return_pct']:
    improvement = with_pullback['return_pct'] - no_pullback['return_pct']
    print(f"\nRESULT: PULLBACK WINS! (+{improvement:.2f}% improvement)")
    print(f"Benefits:")
    if with_pullback['win_rate'] > no_pullback['win_rate']:
        print(f"  - Higher win rate: {with_pullback['win_rate']:.1f}% vs {no_pullback['win_rate']:.1f}%")
    if with_pullback['profit_factor'] > no_pullback['profit_factor']:
        print(f"  - Better profit factor: {with_pullback['profit_factor']:.2f} vs {no_pullback['profit_factor']:.2f}")
    if with_pullback['max_dd'] < no_pullback['max_dd']:
        print(f"  - Lower drawdown: {with_pullback['max_dd']:.1f}% vs {no_pullback['max_dd']:.1f}%")
    print(f"\nRecommendation: Use Combined V2 WITH pullback requirement")
else:
    decline = no_pullback['return_pct'] - with_pullback['return_pct']
    print(f"\nRESULT: NO PULLBACK WINS! (Pullback caused -{decline:.2f}% decline)")
    print(f"Why pullback hurt performance:")
    if with_pullback['trades'] < no_pullback['trades']:
        missed_trades = no_pullback['trades'] - with_pullback['trades']
        print(f"  - Missed {missed_trades} trades ({missed_trades/no_pullback['trades']*100:.1f}% of opportunities)")
    if with_pullback['win_rate'] < no_pullback['win_rate']:
        print(f"  - Lower win rate: {with_pullback['win_rate']:.1f}% vs {no_pullback['win_rate']:.1f}%")
    print(f"\nRecommendation: Use Combined V2 WITHOUT pullback requirement")

print("=" * 100)
