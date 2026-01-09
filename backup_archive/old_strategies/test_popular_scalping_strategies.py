"""
BACKTEST: 6 MOST POPULAR FOREX SCALPING STRATEGIES
Testing all on 1-month, 15-minute data, all 7 pairs
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
print("BACKTEST: 6 POPULAR FOREX SCALPING STRATEGIES - 1 MONTH - ALL 7 PAIRS")
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

def strategy_1_ema_triple(df):
    """EMA Triple Scalping (ForexFactory - 95% win rate claim)
    EMAs: 21, 13, 8
    H1 trend confirmation, M5 entry
    Since we use 15min data, we'll adapt:
    - Use EMAs to detect trend alignment
    - Enter when all EMAs aligned
    """
    df = df.copy()

    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()

    last = df.iloc[-1]

    # BULLISH: Price > EMA8 > EMA13 > EMA21 (all aligned upward)
    if (last['close'] > last['ema_8'] and
        last['ema_8'] > last['ema_13'] and
        last['ema_13'] > last['ema_21']):
        return "BUY"

    # BEARISH: Price < EMA8 < EMA13 < EMA21 (all aligned downward)
    elif (last['close'] < last['ema_8'] and
          last['ema_8'] < last['ema_13'] and
          last['ema_13'] < last['ema_21']):
        return "SELL"

    return None


def strategy_2_stochastic_ema(df):
    """1-Minute Stochastic + EMA Strategy
    Stochastic Oscillator + EMA (13, 26)
    Entry when Stochastic crosses in overbought/oversold + EMA confirms trend
    """
    df = df.copy()

    # Stochastic Oscillator
    period = 14
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # EMAs for trend
    df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BULLISH: Stoch crosses up from oversold (<30) + Price > EMA13 > EMA26
    if (last['stoch_k'] < 30 and prev['stoch_k'] < last['stoch_k'] and
        last['close'] > last['ema_13'] and last['ema_13'] > last['ema_26']):
        return "BUY"

    # BEARISH: Stoch crosses down from overbought (>70) + Price < EMA13 < EMA26
    elif (last['stoch_k'] > 70 and prev['stoch_k'] > last['stoch_k'] and
          last['close'] < last['ema_13'] and last['ema_13'] < last['ema_26']):
        return "SELL"

    return None


def strategy_3_ma_ribbon(df):
    """Moving Average Ribbon (5-8-13 SMA)
    Trade when ribbons align showing strong trend
    """
    df = df.copy()

    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_8'] = df['close'].rolling(8).mean()
    df['sma_13'] = df['close'].rolling(13).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BULLISH: All SMAs slope upward (current > previous) and aligned
    if (last['sma_5'] > prev['sma_5'] and
        last['sma_8'] > prev['sma_8'] and
        last['sma_13'] > prev['sma_13'] and
        last['sma_5'] > last['sma_8'] > last['sma_13']):
        return "BUY"

    # BEARISH: All SMAs slope downward and aligned
    elif (last['sma_5'] < prev['sma_5'] and
          last['sma_8'] < prev['sma_8'] and
          last['sma_13'] < prev['sma_13'] and
          last['sma_5'] < last['sma_8'] < last['sma_13']):
        return "SELL"

    return None


def strategy_4_bollinger_bands(df):
    """Bollinger Bands Scalping
    Buy at lower band, Sell at upper band
    Works best in ranging markets
    """
    df = df.copy()

    period = 20
    df['sma'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['upper'] = df['sma'] + (2 * df['std'])
    df['lower'] = df['sma'] - (2 * df['std'])

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # BUY: Price touches or crosses below lower band (oversold)
    if last['close'] <= last['lower'] and prev['close'] > prev['lower']:
        return "BUY"

    # SELL: Price touches or crosses above upper band (overbought)
    elif last['close'] >= last['upper'] and prev['close'] < prev['upper']:
        return "SELL"

    return None


def strategy_5_breakout(df):
    """Breakout Scalping
    Identify consolidation patterns and trade breakouts
    """
    df = df.copy()

    # Calculate 20-bar range
    lookback = 20
    df['highest'] = df['high'].rolling(lookback).max()
    df['lowest'] = df['low'].rolling(lookback).min()
    df['range'] = df['highest'] - df['lowest']

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Measure volatility (consolidation = low volatility)
    avg_range = df['range'].rolling(lookback).mean().iloc[-1]
    current_range = last['range']

    # Only trade breakouts after consolidation (range < 80% of avg)
    if current_range > avg_range * 0.8:
        return None

    # BULLISH BREAKOUT: Price breaks above recent high
    if last['close'] > prev['highest']:
        return "BUY"

    # BEARISH BREAKOUT: Price breaks below recent low
    elif last['close'] < prev['lowest']:
        return "SELL"

    return None


def strategy_6_range_scalping(df):
    """Range Scalping
    Buy at support, Sell at resistance
    Use RSI + Stochastic for confirmation
    """
    df = df.copy()

    # Identify range
    lookback = 50
    df['support'] = df['low'].rolling(lookback).min()
    df['resistance'] = df['high'].rolling(lookback).max()
    df['mid'] = (df['support'] + df['resistance']) / 2

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic
    period = 14
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    last = df.iloc[-1]

    # Calculate distance to support/resistance
    dist_to_support = (last['close'] - last['support']) / last['support']
    dist_to_resistance = (last['resistance'] - last['close']) / last['close']

    # BUY: Near support (<2% above) + RSI oversold + Stoch oversold
    if dist_to_support < 0.002 and last['rsi'] < 35 and last['stoch'] < 35:
        return "BUY"

    # SELL: Near resistance (<2% below) + RSI overbought + Stoch overbought
    elif dist_to_resistance < 0.002 and last['rsi'] > 65 and last['stoch'] > 65:
        return "SELL"

    return None


# ===========================
# BACKTEST ENGINE
# ===========================

strategies = {
    "Strategy 1: EMA Triple (ForexFactory)": strategy_1_ema_triple,
    "Strategy 2: Stochastic + EMA": strategy_2_stochastic_ema,
    "Strategy 3: MA Ribbon (5-8-13)": strategy_3_ma_ribbon,
    "Strategy 4: Bollinger Bands": strategy_4_bollinger_bands,
    "Strategy 5: Breakout Scalping": strategy_5_breakout,
    "Strategy 6: Range Scalping": strategy_6_range_scalping,
}

print(f"\n[2/3] Testing {len(strategies)} popular scalping strategies...")
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
print("[3/3] FINAL COMPARISON - ALL 6 POPULAR STRATEGIES")
print("=" * 100)

# Sort by return
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'Strategy':<40} {'Trades':<9} {'Win Rate':<11} {'Return':<14} {'PF':<7} {'Max DD'}")
print("-" * 100)

for strategy_name, result in sorted_results:
    print(f"{strategy_name:<40} {result['trades']:<9} {result['win_rate']:<10.1f}% "
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

print("\n" + "=" * 100)
print("COMPARISON WITH PREVIOUS WINNER (RSI Oversold):")
print("  RSI Oversold: +8.15% return, 48.3% WR, 1.11 PF, 7.1% DD")
print("=" * 100)
