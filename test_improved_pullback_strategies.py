"""
BACKTEST: IMPROVED PULLBACK STRATEGIES
Testing 5 different pullback entry methods to improve Combined V2
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
print("BACKTEST: IMPROVED PULLBACK STRATEGIES")
print("=" * 100)

# Download data - need 5-min for pullback detection
print(f"\n[1/3] Downloading 1 month of 5-minute data for {len(MAJOR_PAIRS)} pairs...")
all_data_5min = {}
all_data_15min = {}

for pair in MAJOR_PAIRS:
    yahoo_symbol = YAHOO_SYMBOLS.get(pair)
    try:
        print(f"  {pair}: ", end="", flush=True)

        # 5-min data for pullback detection
        ticker = yf.Ticker(yahoo_symbol)
        df_5min = ticker.history(period='1mo', interval='5m')

        if len(df_5min) == 0:
            print("NO DATA")
            continue

        df_5min = df_5min.reset_index()
        df_5min.columns = [c.lower() for c in df_5min.columns]
        df_5min = df_5min.rename(columns={'datetime': 'time'})
        df_5min = df_5min[['time', 'open', 'high', 'low', 'close', 'volume']]
        all_data_5min[pair] = df_5min

        # 15-min data for strategy signals
        df_15min = ticker.history(period='1mo', interval='15m')
        df_15min = df_15min.reset_index()
        df_15min.columns = [c.lower() for c in df_15min.columns]
        df_15min = df_15min.rename(columns={'datetime': 'time'})
        df_15min = df_15min[['time', 'open', 'high', 'low', 'close', 'volume']]
        all_data_15min[pair] = df_15min

        print(f"{len(df_5min)} bars (5m) / {len(df_15min)} bars (15m)")

    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nLoaded {len(all_data_5min)}/{len(MAJOR_PAIRS)} pairs")

if len(all_data_5min) == 0:
    print("ERROR: No data")
    sys.exit(1)

# ===========================
# PULLBACK STRATEGIES
# ===========================

def pullback_v1_fib_retracement(df_5min, signal_direction):
    """
    Fibonacci Retracement Pullback
    Wait for 38.2% - 61.8% retracement of recent swing
    """
    if len(df_5min) < 20:
        return False

    recent = df_5min.tail(20)

    if signal_direction == "BUY":
        # Find recent swing: low to high
        swing_low = recent['low'].min()
        swing_high = recent['high'].max()
        swing_range = swing_high - swing_low

        # Fibonacci levels
        fib_382 = swing_high - (swing_range * 0.382)
        fib_618 = swing_high - (swing_range * 0.618)

        current_price = recent['close'].iloc[-1]

        # Enter if price pulled back to 38.2-61.8% and now bouncing
        in_fib_zone = fib_618 <= current_price <= fib_382
        bouncing = recent['close'].iloc[-1] > recent['close'].iloc[-2]

        return in_fib_zone and bouncing

    else:  # SELL
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        swing_range = swing_high - swing_low

        fib_382 = swing_low + (swing_range * 0.382)
        fib_618 = swing_low + (swing_range * 0.618)

        current_price = recent['close'].iloc[-1]

        in_fib_zone = fib_382 <= current_price <= fib_618
        dropping = recent['close'].iloc[-1] < recent['close'].iloc[-2]

        return in_fib_zone and dropping


def pullback_v2_ema_touch(df_5min, signal_direction):
    """
    EMA Touch Pullback (IMPROVED)
    Wait for price to touch EMA20 on 5-min, then bounce
    More room than original 3-pip EMA9
    """
    if len(df_5min) < 25:
        return False

    close = df_5min['close']
    ema20 = close.ewm(span=20, adjust=False).mean()

    last = df_5min.iloc[-1]
    prev = df_5min.iloc[-2]

    if signal_direction == "BUY":
        # Price touched EMA from above and bouncing
        touched_ema = abs(prev['close'] - ema20.iloc[-2]) / prev['close'] < 0.0010  # Within 10 pips
        price_above_ema = last['close'] > ema20.iloc[-1]
        bouncing = last['close'] > prev['close']

        return touched_ema and price_above_ema and bouncing

    else:  # SELL
        touched_ema = abs(prev['close'] - ema20.iloc[-2]) / prev['close'] < 0.0010
        price_below_ema = last['close'] < ema20.iloc[-1]
        dropping = last['close'] < prev['close']

        return touched_ema and price_below_ema and dropping


def pullback_v3_higher_low_lower_high(df_5min, signal_direction):
    """
    Higher Low / Lower High Pattern
    BULLISH: Wait for higher low formation (classic pullback)
    BEARISH: Wait for lower high formation
    """
    if len(df_5min) < 15:
        return False

    recent = df_5min.tail(15)

    if signal_direction == "BUY":
        # Look for higher low pattern in last 10 bars
        lows = recent['low'].values[-10:]

        # Find the lowest low and a subsequent higher low
        min_idx = np.argmin(lows)

        # Check if we have a higher low after the lowest point
        if min_idx < len(lows) - 3:  # Need at least 3 bars after
            subsequent_lows = lows[min_idx+1:]
            if len(subsequent_lows) > 0:
                # Current low should be higher than previous low
                current_low = recent['low'].iloc[-1]
                prev_low = recent['low'].iloc[-3]

                higher_low_formed = current_low > prev_low
                price_rising = recent['close'].iloc[-1] > recent['close'].iloc[-2]

                return higher_low_formed and price_rising

        return False

    else:  # SELL
        highs = recent['high'].values[-10:]
        max_idx = np.argmax(highs)

        if max_idx < len(highs) - 3:
            subsequent_highs = highs[max_idx+1:]
            if len(subsequent_highs) > 0:
                current_high = recent['high'].iloc[-1]
                prev_high = recent['high'].iloc[-3]

                lower_high_formed = current_high < prev_high
                price_dropping = recent['close'].iloc[-1] < recent['close'].iloc[-2]

                return lower_high_formed and price_dropping

        return False


def pullback_v4_rsi_divergence(df_5min, signal_direction):
    """
    RSI Divergence Pullback
    Look for RSI pulling back from extreme while price consolidates
    More sophisticated entry
    """
    if len(df_5min) < 20:
        return False

    close = df_5min['close']

    # Calculate RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    last_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-5]  # 5 bars ago

    if signal_direction == "BUY":
        # RSI was oversold, now recovering
        # Price made lower low, but RSI made higher low (bullish divergence)
        rsi_recovering = last_rsi > prev_rsi and last_rsi > 35 and last_rsi < 55
        price_stable = abs(close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] < 0.002

        return rsi_recovering and price_stable

    else:  # SELL
        # RSI was overbought, now declining
        rsi_declining = last_rsi < prev_rsi and last_rsi < 65 and last_rsi > 45
        price_stable = abs(close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] < 0.002

        return rsi_declining and price_stable


def pullback_v5_volume_confirmation(df_5min, signal_direction):
    """
    Volume-Confirmed Pullback
    Wait for low volume on pullback, high volume on resumption
    Shows profit-taking followed by renewed interest
    """
    if len(df_5min) < 20:
        return False

    recent = df_5min.tail(20)

    # Volume analysis
    avg_volume = recent['volume'].mean()
    last_3_volume = recent['volume'].tail(3).mean()
    current_volume = recent['volume'].iloc[-1]

    close = recent['close']
    ema9 = close.ewm(span=9, adjust=False).mean()

    if signal_direction == "BUY":
        # Low volume pullback, high volume bounce
        pullback_low_volume = last_3_volume < avg_volume * 0.8
        bounce_high_volume = current_volume > avg_volume * 1.2

        price_above_ema = close.iloc[-1] > ema9.iloc[-1]
        price_rising = close.iloc[-1] > close.iloc[-2]

        return pullback_low_volume and bounce_high_volume and price_above_ema and price_rising

    else:  # SELL
        pullback_low_volume = last_3_volume < avg_volume * 0.8
        drop_high_volume = current_volume > avg_volume * 1.2

        price_below_ema = close.iloc[-1] < ema9.iloc[-1]
        price_dropping = close.iloc[-1] < close.iloc[-2]

        return pullback_low_volume and drop_high_volume and price_below_ema and price_dropping


def get_combined_signal(df_15min):
    """Combined V2 signal from 15-min data"""
    df = df_15min.copy()

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


# ===========================
# BACKTEST ENGINE
# ===========================

pullback_methods = {
    "No Pullback (Original)": None,
    "V1: Fibonacci Retracement": pullback_v1_fib_retracement,
    "V2: EMA20 Touch + Bounce": pullback_v2_ema_touch,
    "V3: Higher Low / Lower High": pullback_v3_higher_low_lower_high,
    "V4: RSI Divergence": pullback_v4_rsi_divergence,
    "V5: Volume Confirmation": pullback_v5_volume_confirmation,
}

print(f"\n[2/3] Testing {len(pullback_methods)} pullback strategies...")
print("=" * 100)

results = {}

for method_name, pullback_func in pullback_methods.items():
    print(f"\n{method_name}:")
    print("-" * 100)

    balance = 10000
    initial_balance = balance
    positions = {}
    trades = []
    leverage = 20
    max_concurrent = 5

    reference_pair = list(all_data_15min.keys())[0]
    df_ref = all_data_15min[reference_pair]

    for i in range(60, len(df_ref)):
        current_time = pd.to_datetime(df_ref.iloc[i]['time'])

        # Check for entries
        if len(positions) < max_concurrent:
            for pair in all_data_15min.keys():
                if pair in positions:
                    continue

                # Get 15-min data for signal
                df_15min = all_data_15min[pair]
                pair_bars_15min = df_15min[df_15min['time'] <= current_time]
                if len(pair_bars_15min) < 60:
                    continue

                idx_15min = len(pair_bars_15min) - 1
                df_slice_15min = df_15min.iloc[max(0, idx_15min-60):idx_15min+1].copy()

                if len(df_slice_15min) < 50:
                    continue

                # Get combined signal
                signal = get_combined_signal(df_slice_15min)

                if signal:
                    # If pullback required, check 5-min data
                    if pullback_func is not None:
                        df_5min = all_data_5min[pair]
                        pair_bars_5min = df_5min[df_5min['time'] <= current_time]

                        if len(pair_bars_5min) < 30:
                            continue

                        df_slice_5min = pair_bars_5min.tail(30)

                        # Check pullback on 5-min data
                        if not pullback_func(df_slice_5min, signal):
                            continue  # Skip if pullback not confirmed

                    # Enter trade
                    current_bar = df_15min.iloc[idx_15min]
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
            df = all_data_15min[pair]
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

        results[method_name] = {
            "trades": len(trades),
            "win_rate": win_rate,
            "pl": total_pl,
            "return_pct": (balance / initial_balance - 1) * 100,
            "profit_factor": profit_factor,
            "max_dd": max_dd,
            "balance": balance
        }

        print(f"  100% DONE | Trades: {len(trades)} | Win Rate: {win_rate:.1f}% | P&L: ${total_pl:+,.0f} ({results[method_name]['return_pct']:+.2f}%) | PF: {profit_factor:.2f} | DD: {max_dd:.1f}%")
    else:
        results[method_name] = {
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
print("[3/3] FINAL COMPARISON - BEST PULLBACK METHOD")
print("=" * 100)

# Sort by return
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'Pullback Method':<45} {'Trades':<9} {'Win Rate':<11} {'Return':<14} {'PF':<7} {'Max DD'}")
print("-" * 100)

for method_name, result in sorted_results:
    print(f"{method_name:<45} {result['trades']:<9} {result['win_rate']:<10.1f}% "
          f"{result['return_pct']:+12.2f}% {result['profit_factor']:>6.2f} {result['max_dd']:>6.1f}%")

print("\n" + "=" * 100)
print("WINNER")
print("=" * 100)

best = sorted_results[0]
print(f"\nBEST PULLBACK METHOD: {best[0]}")
print(f"  Trades: {best[1]['trades']}")
print(f"  Win Rate: {best[1]['win_rate']:.1f}%")
print(f"  Return: {best[1]['return_pct']:+.2f}%")
print(f"  Profit Factor: {best[1]['profit_factor']:.2f}")
print(f"  Max Drawdown: {best[1]['max_dd']:.1f}%")

original = results["No Pullback (Original)"]
if best[1]['return_pct'] > original['return_pct']:
    improvement = best[1]['return_pct'] - original['return_pct']
    print(f"\nRESULT: PULLBACK IMPROVES PERFORMANCE!")
    print(f"  Improvement: +{improvement:.2f}% over no pullback")
    print(f"  Recommendation: Use '{best[0]}' for live trading")
else:
    print(f"\nRESULT: Original (no pullback) still best at {original['return_pct']:+.2f}%")
    print(f"  Recommendation: Keep Combined V2 without pullback")

print("=" * 100)
