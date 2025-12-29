"""
Forex Scalping - Live Paper Trading
Runs the profitable scalping strategy on OANDA demo account in real-time

PROFITABLE STRATEGY:
- Win Rate: 57%
- Profit Factor: 2.01
- Trades/Day: ~2
- TP: 30 pips | SL: 20 pips
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
import pandas as pd
import time
from datetime import datetime
import pytz

print("="*80)
print("FOREX SCALPING - LIVE PAPER TRADING")
print("="*80)
print("\nProfitable Strategy Configuration:")
print("  - TP: 30 pips | SL: 20 pips")
print("  - Win Rate: 57% (backtest)")
print("  - Profit Factor: 2.01")
print("  - Max Trades/Day: 10")
print("  - Position Size: 10% per trade")
print("="*80)

# Initialize OANDA
print("\n[1/3] Connecting to OANDA Demo Account...")
import os
# Force correct account ID
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

account_info = client.get_account_info()
if account_info:
    balance = client.get_balance()
    print(f"      Account: {client.account_id}")
    print(f"      Balance: ${balance:,.2f}")
    print(f"      Status: CONNECTED")
else:
    print("      ERROR: Could not connect to OANDA")
    sys.exit(1)

# Initialize strategy
print("\n[2/3] Initializing Scalping Strategy...")
instrument = "EUR_USD"

strategy = ForexScalpingStrategy(
    instruments=[instrument],
    max_trades_per_day=10,
    daily_profit_target=0.03,
    trade_size_pct=0.10,  # 10% per trade
    take_profit_pct=0.003,  # 30 pips
    stop_loss_pct=0.002,  # 20 pips
    trailing_stop_trigger=0.002,
    trailing_stop_distance=0.001,
    require_htf_strict=True,  # Quality over quantity
    pullback_required=False,
    min_consensus_score=1
)

# Trading state
positions = {}
trades_today = 0
daily_start_balance = balance
start_time = datetime.now(pytz.UTC)
current_date = start_time.date()

print("\n[3/3] Starting Live Trading...")
print("="*80)
print(f"Instrument: {instrument}")
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Initial Balance: ${balance:,.2f}")
print("\nPress Ctrl+C to stop trading\n")
print("="*80)

try:
    iteration = 0
    while True:
        iteration += 1
        now = datetime.now(pytz.UTC)

        # Daily reset
        if now.date() > current_date:
            daily_pl = balance - daily_start_balance
            daily_pl_pct = daily_pl / daily_start_balance * 100

            print(f"\n{'='*80}")
            print(f"[DAY END] {current_date}")
            print(f"  Trades: {trades_today}")
            print(f"  P&L: ${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%)")
            print(f"  Balance: ${balance:,.2f}")
            print(f"{'='*80}\n")

            current_date = now.date()
            trades_today = 0
            daily_start_balance = balance

            print(f"[NEW DAY] {current_date} | Starting Balance: ${balance:,.2f}\n")

        # Get current price
        price_data = client.get_current_price(instrument)

        if not price_data or 'bid' not in price_data:
            print(f"[{now.strftime('%H:%M:%S')}] Waiting for price data...")
            time.sleep(5)
            continue

        current_price = (price_data['bid'] + price_data['ask']) / 2
        spread = price_data['spread'] * 10000  # in pips

        # Check open positions first
        open_trades = client.get_open_trades()

        if len(open_trades) > 0:
            for trade in open_trades:
                trade_id = trade['id']
                unrealized_pl = trade['unrealized_pl']

                # Update display every 10 iterations
                if iteration % 10 == 0:
                    print(f"[{now.strftime('%H:%M:%S')}] Position OPEN | Price: {current_price:.5f} | P&L: ${unrealized_pl:+.2f}")

            time.sleep(5)  # Check every 5 seconds when position open
            continue

        # No position - check for entry signal
        # Only check every 60 seconds to avoid rate limits
        if iteration % 12 != 0:  # 12 * 5 seconds = 60 seconds
            time.sleep(5)
            continue

        print(f"\n[{now.strftime('%H:%M:%S')}] Checking for entry signal...")

        # Get historical data
        try:
            candles_1min = client.get_candles(instrument, "M1", count=200)
            candles_5min = client.get_candles(instrument, "M5", count=100)
            candles_15min = client.get_candles(instrument, "M15", count=100)
            candles_30min = client.get_candles(instrument, "M30", count=100)

            df_1min = pd.DataFrame(candles_1min)
            df_5min = pd.DataFrame(candles_5min)
            df_15min = pd.DataFrame(candles_15min)
            df_30min = pd.DataFrame(candles_30min)

            if len(df_1min) < 50:
                print("  Insufficient data, waiting...")
                time.sleep(60)
                continue

            # Check for signal
            daily_pl_pct = (balance - daily_start_balance) / daily_start_balance

            signal = strategy.should_enter_trade(
                instrument=instrument,
                df_1min=df_1min,
                df_5min=df_5min,
                df_15min=df_15min,
                df_30min=df_30min,
                current_positions=len(open_trades),
                trades_today=trades_today,
                daily_pl_pct=daily_pl_pct
            )

            if signal["action"] == "SKIP":
                print(f"  NO ENTRY - {signal['reason']}")
                print(f"  Price: {current_price:.5f} | Spread: {spread:.1f} pips")
                time.sleep(60)
                continue

            # ENTER TRADE
            if signal["action"] in ["BUY", "SELL"]:
                print(f"\n{'='*80}")
                print(f"[ENTRY SIGNAL] {signal['action']} {instrument}")
                print(f"  Price: {current_price:.5f}")
                print(f"  Reason: {signal['reason']}")
                print(f"  Confidence: {signal['confidence']}")

                # Calculate position size
                units = strategy.calculate_position_size(balance, current_price, instrument)

                # Set direction
                if signal["action"] == "SELL":
                    units = -units  # Negative for SHORT

                # Calculate SL and TP
                stop_loss, take_profit = strategy.calculate_stop_loss_take_profit(
                    current_price, signal["action"]
                )

                # Calculate trailing stop distance in price
                trailing_distance = current_price * 0.001  # 10 pips

                print(f"\n  Placing Order:")
                print(f"    Units: {abs(units):,}")
                print(f"    Position Size: ${abs(units * current_price):,.2f}")
                print(f"    Stop Loss: {stop_loss:.5f} ({abs(stop_loss-current_price)*10000:.1f} pips)")
                print(f"    Take Profit: {take_profit:.5f} ({abs(take_profit-current_price)*10000:.1f} pips)")
                print(f"    Trailing Stop: {trailing_distance*10000:.1f} pips")

                # Place order
                result = client.place_market_order(
                    instrument=instrument,
                    units=units,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_distance=trailing_distance
                )

                if result.get('success'):
                    trades_today += 1
                    print(f"\n  ORDER FILLED!")
                    print(f"    Order ID: {result.get('order_id')}")
                    print(f"    Trade ID: {result.get('trade_id')}")
                    print(f"    Fill Price: {result.get('filled_price'):.5f}")
                    print(f"    Trades Today: {trades_today}/{strategy.max_trades_per_day}")
                else:
                    print(f"\n  ORDER FAILED: {result.get('error')}")

                print(f"{'='*80}\n")

        except Exception as e:
            print(f"  ERROR: {e}")
            time.sleep(60)
            continue

        time.sleep(5)

except KeyboardInterrupt:
    print(f"\n\n{'='*80}")
    print("STOPPING LIVE TRADING")
    print(f"{'='*80}")

    # Close any open positions
    open_trades = client.get_open_trades()

    if len(open_trades) > 0:
        print(f"\nClosing {len(open_trades)} open position(s)...")

        for trade in open_trades:
            result = client.close_trade(trade['id'])
            if result.get('success'):
                print(f"  Trade {trade['id']} closed at {result['closed_price']:.5f}")
                print(f"  P&L: ${result['pl']:+,.2f}")

    # Final summary
    final_balance = client.get_balance()
    total_pl = final_balance - balance
    total_return = (final_balance / balance - 1) * 100
    duration = datetime.now(pytz.UTC) - start_time

    print(f"\n{'='*80}")
    print("SESSION SUMMARY")
    print(f"{'='*80}")
    print(f"Duration: {duration}")
    print(f"Trades Executed: {trades_today}")
    print(f"Initial Balance: ${balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Total P&L: ${total_pl:+,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"{'='*80}\n")

except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nLive trading stopped.")
