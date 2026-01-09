"""
Quick Forex Backtest with Sample Data
Run immediate backtest without OANDA API using synthetic data

This allows you to test the strategy logic before setting up data sources
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from trading_system.Forex_Trading.strategies.multi_timeframe_momentum import MultiTimeframeMomentumStrategy


def generate_realistic_forex_data(
    instrument: str,
    start_date: datetime,
    end_date: datetime,
    base_price: float,
    volatility: float = 0.0001
) -> pd.DataFrame:
    """Generate realistic forex price data"""

    # Generate 1-minute timestamps (only market hours for realism)
    all_timestamps = []
    current = start_date

    while current <= end_date:
        # Skip weekends (forex closed Saturday-Sunday)
        if current.weekday() < 5:  # Monday=0 to Friday=4
            all_timestamps.append(current)
        current += timedelta(minutes=1)

    print(f"[DATA] Generating {len(all_timestamps)} bars for {instrument}")

    # Generate price movements with trend and volatility
    np.random.seed(42)

    # Add trending behavior
    trend = np.linspace(0, 0.002, len(all_timestamps))  # 0.2% trend over period
    returns = np.random.normal(0, volatility, len(all_timestamps)) + trend / len(all_timestamps)
    prices = base_price * (1 + returns).cumprod()

    # Generate OHLC
    data = []
    for i, (ts, close) in enumerate(zip(all_timestamps, prices)):
        vol = close * volatility * 2
        high = close + abs(np.random.normal(0, vol))
        low = close - abs(np.random.normal(0, vol))
        open_price = prices[i-1] if i > 0 else close

        data.append({
            'time': ts,
            'open': round(open_price, 5),
            'high': round(max(high, open_price, close), 5),
            'low': round(min(low, open_price, close), 5),
            'close': round(close, 5),
            'volume': int(np.random.uniform(50, 200))
        })

    return pd.DataFrame(data)


def resample_to_timeframe(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-min data to other timeframes"""
    df_resampled = df.set_index('time').resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    return df_resampled


def run_quick_backtest(
    instrument: str = "EUR_USD",
    days: int = 7,
    initial_capital: float = 10000
):
    """Run a quick backtest with sample data"""

    print("="*80)
    print("FOREX QUICK BACKTEST - Sample Data")
    print("="*80)
    print(f"\nInstrument: {instrument}")
    print(f"Period: Last {days} days")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("\nUsing synthetic data for demonstration...")
    print("(To use real data, set up OANDA API or download HistData)\n")

    # Generate date range
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days)

    # Generate sample data
    print(f"[BACKTEST] Generating data from {start_date.date()} to {end_date.date()}")

    base_prices = {
        "EUR_USD": 1.0850,
        "GBP_USD": 1.2650,
        "USD_JPY": 149.50,
        "USD_CHF": 0.8850,
        "AUD_USD": 0.6550,
        "USD_CAD": 1.3750,
        "NZD_USD": 0.5950
    }

    base_price = base_prices.get(instrument, 1.0850)
    df_1min = generate_realistic_forex_data(instrument, start_date, end_date, base_price)

    # Resample to other timeframes
    print("[BACKTEST] Resampling to multiple timeframes...")
    df_5min = resample_to_timeframe(df_1min, '5min')
    df_30min = resample_to_timeframe(df_1min, '30min')
    df_1hour = resample_to_timeframe(df_1min, '1H')

    print(f"[BACKTEST] 1-min bars: {len(df_1min)}")
    print(f"[BACKTEST] 5-min bars: {len(df_5min)}")
    print(f"[BACKTEST] 30-min bars: {len(df_30min)}")
    print(f"[BACKTEST] 1-hour bars: {len(df_1hour)}")

    # Initialize strategy
    print("\n[BACKTEST] Initializing strategy...")
    strategy = MultiTimeframeMomentumStrategy(
        instruments=[instrument],
        max_trades_per_day=3,
        daily_profit_target=0.02,
        trade_size_pct=0.10,
        take_profit_pct=0.015,
        stop_loss_pct=0.01,
        trailing_stop_trigger=0.006,
        trailing_stop_distance=0.004
    )

    # Backtest variables
    balance = initial_capital
    positions = {}
    trades = []
    daily_trades = 0
    daily_start_balance = initial_capital
    current_date = None

    print("\n[BACKTEST] Running simulation...")
    print("="*80)

    # Simulate tick-by-tick
    for i in range(100, len(df_1min)):  # Start after 100 bars for indicators
        current_time = df_1min.iloc[i]['time']
        current_bar = df_1min.iloc[i]

        # Daily reset
        if current_date is None or current_time.date() > current_date:
            if current_date is not None:
                daily_pl = balance - daily_start_balance
                daily_pl_pct = daily_pl / daily_start_balance * 100
                print(f"\n[DAY END] {current_date} | Trades: {daily_trades} | P&L: ${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%) | Balance: ${balance:,.2f}")

            current_date = current_time.date()
            daily_trades = 0
            daily_start_balance = balance
            print(f"\n[NEW DAY] {current_date} | Starting balance: ${balance:,.2f}")

        # Get data up to current time
        df_1min_slice = df_1min.iloc[max(0, i-500):i+1]
        df_5min_slice = df_5min[df_5min['time'] <= current_time].tail(100)
        df_30min_slice = df_30min[df_30min['time'] <= current_time].tail(50)
        df_1hour_slice = df_1hour[df_1hour['time'] <= current_time].tail(50)

        # Check for new entry
        if len(positions) == 0 and daily_trades < 3:
            daily_pl_pct = (balance - daily_start_balance) / daily_start_balance

            signal = strategy.should_enter_trade(
                instrument=instrument,
                df_1min=df_1min_slice,
                df_5min=df_5min_slice,
                df_30min=df_30min_slice,
                df_1hour=df_1hour_slice,
                current_positions=len(positions),
                trades_today=daily_trades,
                daily_pl_pct=daily_pl_pct
            )

            if signal["action"] in ["BUY", "SELL"]:
                # Enter position
                entry_price = current_bar['close']
                units = strategy.calculate_position_size(balance, entry_price, instrument)
                stop_loss, take_profit = strategy.calculate_stop_loss_take_profit(entry_price, signal["action"])

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
                    "signal": signal
                }

                positions[instrument] = position
                daily_trades += 1

                print(f"\n[ENTRY] {current_time.strftime('%Y-%m-%d %H:%M')} | {position['direction']} @ {entry_price:.5f}")
                print(f"        Units: {units:,} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                print(f"        Reason: {signal['reason']}")
                print(f"        Confidence: {signal['confidence']}")

        # Update existing positions
        if instrument in positions:
            pos = positions[instrument]
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

            # Check trailing stop trigger
            if not pos["trailing_stop_triggered"] and pl_pct >= 0.006:
                pos["trailing_stop_triggered"] = True
                print(f"[TRAILING] Trailing stop activated at +{pl_pct*100:.2f}%")

            # Check exits
            exit_price = None
            exit_reason = None

            if pos["direction"] == "LONG":
                if current_bar['high'] >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TAKE PROFIT"
                elif current_bar['low'] <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "STOP LOSS"
                elif pos["trailing_stop_triggered"]:
                    trailing_stop = pos["highest_price"] * (1 - 0.004)
                    if current_bar['low'] <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAILING STOP"
            else:
                if current_bar['low'] <= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TAKE PROFIT"
                elif current_bar['high'] >= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "STOP LOSS"
                elif pos["trailing_stop_triggered"]:
                    trailing_stop = pos["lowest_price"] * (1 + 0.004)
                    if current_bar['high'] >= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAILING STOP"

            if exit_price:
                # Close position
                if pos["direction"] == "LONG":
                    final_pl = (exit_price - pos["entry_price"]) * pos["units"]
                else:
                    final_pl = (pos["entry_price"] - exit_price) * abs(pos["units"])

                final_pl_pct = final_pl / (abs(pos["units"]) * pos["entry_price"])
                balance += final_pl

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
                    "balance": balance
                }

                trades.append(trade)

                print(f"\n[EXIT] {current_time.strftime('%Y-%m-%d %H:%M')} | {pos['direction']} @ {exit_price:.5f}")
                print(f"       P&L: ${final_pl:+,.2f} ({final_pl_pct*100:+.2f}%) | Reason: {exit_reason}")
                print(f"       Balance: ${balance:,.2f}")

                del positions[instrument]

        # Progress update every 1000 bars
        if i % 1000 == 0:
            progress = i / len(df_1min) * 100
            print(f"[PROGRESS] {progress:.1f}% | Balance: ${balance:,.2f} | Trades: {len(trades)}")

    # Close remaining positions
    if instrument in positions:
        pos = positions[instrument]
        exit_price = df_1min.iloc[-1]['close']

        if pos["direction"] == "LONG":
            final_pl = (exit_price - pos["entry_price"]) * pos["units"]
        else:
            final_pl = (pos["entry_price"] - exit_price) * abs(pos["units"])

        final_pl_pct = final_pl / (abs(pos["units"]) * pos["entry_price"])
        balance += final_pl

        trades.append({
            "instrument": instrument,
            "direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "entry_time": pos["entry_time"],
            "exit_price": exit_price,
            "exit_time": df_1min.iloc[-1]['time'],
            "units": pos["units"],
            "pl": final_pl,
            "pl_pct": final_pl_pct,
            "exit_reason": "END OF BACKTEST",
            "commission": pos["commission"],
            "balance": balance
        })

    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)

    if len(trades) == 0:
        print("\nNo trades executed during backtest period.")
        print("This is normal - the strategy is very strict with its filters.")
        print("\nTry:")
        print("  - Longer backtest period (--days 30)")
        print("  - Different instrument")
        print("  - Review strategy parameters")
    else:
        winning_trades = [t for t in trades if t["pl"] > 0]
        losing_trades = [t for t in trades if t["pl"] < 0]

        total_pl = balance - initial_capital
        total_return = (balance / initial_capital - 1) * 100
        win_rate = len(winning_trades) / len(trades) * 100

        avg_win = np.mean([t["pl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pl"] for t in losing_trades]) if losing_trades else 0

        profit_factor = abs(sum(t["pl"] for t in winning_trades) / sum(t["pl"] for t in losing_trades)) if losing_trades else float('inf')

        print(f"\nTotal Trades: {len(trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"\nInitial Capital: ${initial_capital:,.2f}")
        print(f"Final Balance: ${balance:,.2f}")
        print(f"Total P&L: ${total_pl:+,.2f}")
        print(f"Total Return: {total_return:+.2f}%")

        print("\n" + "="*80)
        print("TRADE LOG")
        print("="*80)
        for i, trade in enumerate(trades, 1):
            print(f"\nTrade #{i}:")
            print(f"  {trade['direction']} {trade['instrument']}")
            print(f"  Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {trade['entry_price']:.5f}")
            print(f"  Exit:  {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} @ {trade['exit_price']:.5f}")
            print(f"  P&L: ${trade['pl']:+,.2f} ({trade['pl_pct']*100:+.2f}%)")
            print(f"  Reason: {trade['exit_reason']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick Forex Backtest with Sample Data")
    parser.add_argument("--instrument", type=str, default="EUR_USD", help="Forex pair")
    parser.add_argument("--days", type=int, default=7, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")

    args = parser.parse_args()

    run_quick_backtest(args.instrument, args.days, args.capital)
