"""
Multi-Symbol Forex Trading - TREND-FOLLOWING V3
Trades ALL MAJOR FOREX PAIRS simultaneously on OANDA demo account

STRATEGY: Breakout + Pullback Entry (WITH the trend only!)
- ONLY BUY in uptrends (EMA9 > EMA21 on 15min+30min)
- ONLY SELL in downtrends (EMA9 < EMA21 on 15min+30min)
- Entry 1: Breakout (price breaks key level with momentum)
- Entry 2: Pullback (RSI retraces to 40-60 zone in trend direction)

TRAILING STOP: DELAYED ACTIVATION (prevents premature exits)
- NOT set on order entry - managed manually
- Only activates when profit reaches trigger level
- Volatile pairs (GBP, JPY, AUD, NZD): 88-91% of TP before trailing
- Moderate pairs (EUR, CHF, CAD): 78-80% of TP before trailing

TIME-BASED COOLDOWN: 15-30 min between trades per symbol (session dependent)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
from trading_system.Forex_Trading.utils.trade_logger import ForexTradeLogger
import pandas as pd
import time
from datetime import datetime
import pytz

print("=" * 80)
print("MULTI-SYMBOL FOREX SCALPING - LIVE PAPER TRADING")
print("=" * 80)
print(f"\nTrading {len(MAJOR_PAIRS)} Major Pairs:")
for pair in MAJOR_PAIRS:
    print(f"  - {pair}")

print("\nStrategy: TREND-FOLLOWING V3 (Breakout + Pullback)")
print("  - ONLY trade WITH the trend (no counter-trend!)")
print("  - Entry 1: Breakout (price breaks key level + momentum)")
print("  - Entry 2: Pullback (RSI retraces in established trend)")
print("  - Max Trades: 10 per symbol/day")
print("  - Max Concurrent: 5 positions")
print("\nPosition Sizing: $1 PER PIP (consistent risk)")
print("  - EUR/USD, GBP/USD, AUD/USD, NZD/USD: 10,000 units")
print("  - USD/JPY: ~15,500 units (varies with rate)")
print("  - USD/CHF: ~8,000 units (varies with rate)")
print("  - USD/CAD: ~13,800 units (varies with rate)")
print("  - Risk Example: 20 pip SL = $20 risk per trade")
print("=" * 80)

# Initialize OANDA
print("\n[1/4] Connecting to OANDA Demo Account...")
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

# Initialize Trade Logger
print("\n[2/4] Initializing Trade Logger...")
logger = ForexTradeLogger(log_dir="forex_logs")
print(f"      Logs will be saved to: forex_logs/")
print(f"      - Trades: trades_{datetime.now().strftime('%Y%m%d')}.jsonl")
print(f"      - Market: market_data_{datetime.now().strftime('%Y%m%d')}.jsonl")
print(f"      - Summary: daily_summary_{datetime.now().strftime('%Y%m%d')}.json")

# Initialize strategies for each pair
print(f"\n[3/4] Initializing {len(MAJOR_PAIRS)} Strategies with pair-specific settings...")
strategies = {}

# Import pair settings
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params

print("\n      Pair-Specific Settings:")
print(f"      {'Pair':<10} {'TP':<8} {'SL':<8} {'Trailing':<12} {'R:R'}")
print(f"      {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*6}")

for instrument in MAJOR_PAIRS:
    # Get pair-specific parameters
    params = get_scalping_params(instrument)

    strategies[instrument] = ForexScalpingStrategy(
        instruments=[instrument],
        max_trades_per_day=10,  # 10 per symbol - aggressive scalping
        daily_profit_target=0.05,  # 5% daily target
        trade_size_pct=0.05,  # 5% per trade
        take_profit_pct=params["take_profit_pct"],
        stop_loss_pct=params["stop_loss_pct"],
        trailing_stop_trigger=params["trailing_stop_trigger"],
        trailing_stop_distance=params["trailing_stop_distance"],
        require_htf_strict=True,
        pullback_required=True,  # Wait for pullback for better entries
        min_consensus_score=1
    )

    # Display settings
    print(f"      {instrument:<10} {params['tp_pips']:<3}p     {params['sl_pips']:<3}p     "
          f"{params['trail_trigger_pips']}p / {params['trail_distance_pips']}p       "
          f"{params['risk_reward_ratio']:.2f}:1")

print(f"\n      Initialized {len(strategies)} strategies with optimized settings")

# Trading state
trades_today_per_symbol = {pair: 0 for pair in MAJOR_PAIRS}
daily_start_balance = balance
start_time = datetime.now(pytz.UTC)
current_date = start_time.date()
max_concurrent = 5

# Trailing stop activation tracking
# Key = trade_id, Value = {trigger_price, trail_distance, activated}
trailing_stop_tracking = {}

# Check for existing open trades
print("\n[4/4] Checking for existing open positions...")
existing_trades = client.get_open_trades() or []

if len(existing_trades) > 0:
    print(f"      Found {len(existing_trades)} existing open position(s):")
    for trade in existing_trades:
        symbol = trade['instrument']
        units = trade['units']
        entry_price = trade['price']
        unrealized_pl = trade['unrealized_pl']
        direction = "LONG" if units > 0 else "SHORT"
        print(f"      - {symbol}: {direction} {abs(units):,} units @ {entry_price:.5f} | P&L: ${unrealized_pl:+.2f}")
    print("      These positions will be managed automatically.")
else:
    print("      No existing positions found.")

# Get initial session info
initial_session = strategies[MAJOR_PAIRS[0]].get_current_session(start_time)

print("\n[4/4] Starting Multi-Symbol Live Trading...")
print("=" * 80)
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Current Session: {initial_session}")
print(f"Initial Balance: ${balance:,.2f}")
print(f"Max Concurrent Positions: {max_concurrent}")
if len(existing_trades) > 0:
    print(f"Existing Positions: {len(existing_trades)}")

print("\nTime-Based Cooldown: ENABLED")
print("  - Tokyo: 30min | London: 20min | New York: 15min")
print("  - Multiple trades per session allowed (wait cooldown between)")
print("\nTrailing Stop: DELAYED ACTIVATION")
print("  - Trailing stop NOT set on entry (prevents premature exits)")
print("  - Activates ONLY when profit reaches trigger level (80-90% of TP)")
print("  - Volatile pairs (GBP, JPY, AUD, NZD): 88-91% of TP trigger")
print("  - Moderate pairs (EUR, CHF, CAD): 78-80% of TP trigger")

print("\nPress Ctrl+C to stop trading\n")
print("=" * 80)

try:
    iteration = 0
    tracked_trade_ids = set()  # Track open trades to detect closures
    trade_entry_data = {}  # Store entry data for exit logging
    last_known_trades = []  # Cache last successful trade fetch
    consecutive_errors = 0  # Track consecutive API errors

    while True:
        iteration += 1
        now = datetime.now(pytz.UTC)

        # Get all open positions first (needed for Friday check)
        open_trades = client.get_open_trades()

        # Handle API errors gracefully - don't assume trades are closed on error
        if open_trades is None:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                print(f"[WARNING] {consecutive_errors} consecutive API errors - using cached trade data")
            open_trades = last_known_trades  # Use cached data
        else:
            consecutive_errors = 0
            last_known_trades = open_trades  # Update cache on success

        # FRIDAY MARKET CLOSE PROTECTION
        # Forex market closes Friday 5:00 PM EST (22:00 UTC)
        # Stop opening new trades in last hour (21:00 UTC Friday)
        # Close all trades 10 minutes before close (21:50 UTC Friday)
        is_friday = now.weekday() == 4  # Friday = 4
        hour_utc = now.hour
        minute_utc = now.minute

        # Flag: Don't open new trades in last hour on Friday
        friday_no_new_trades = is_friday and hour_utc >= 21

        # Flag: Close all trades 10 min before market close
        friday_close_all = is_friday and (hour_utc == 21 and minute_utc >= 50)

        if friday_close_all and len(open_trades) > 0:
            print(f"\n{'=' * 80}")
            print(f"[FRIDAY MARKET CLOSE] Closing all {len(open_trades)} positions before market close")
            print(f"  Current Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (Friday 21:50 UTC)")
            print(f"  Market Closes: Friday 22:00 UTC (5:00 PM EST)")
            print(f"{'=' * 80}\n")

            for trade in open_trades:
                trade_id = trade['id']
                instrument = trade['instrument']
                units = trade['units']
                unrealized_pl = trade.get('unrealized_pl', 0)

                print(f"  Closing: {instrument} (Trade #{trade_id}) | P&L: ${unrealized_pl:+.2f}")

                result = client.close_trade(trade_id)
                if result.get('success'):
                    print(f"    ✓ Closed successfully")
                else:
                    print(f"    ✗ Failed to close: {result.get('error', 'Unknown error')}")

            print(f"\n{'=' * 80}")
            print(f"[WEEKEND MODE] All positions closed. No new trades until Monday.")
            print(f"  Next trading starts: Monday ~00:00 UTC")
            print(f"{'=' * 80}\n")

        # Daily reset
        if now.date() > current_date:
            daily_pl = balance - daily_start_balance
            daily_pl_pct = daily_pl / daily_start_balance * 100
            total_trades = sum(trades_today_per_symbol.values())

            print(f"\n{'=' * 80}")
            print(f"[DAY END] {current_date}")
            print(f"  Total Trades: {total_trades}")
            print(f"  P&L: ${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%)")
            print(f"  Balance: ${balance:,.2f}")

            # Per-symbol breakdown
            for pair in MAJOR_PAIRS:
                if trades_today_per_symbol[pair] > 0:
                    print(f"    {pair}: {trades_today_per_symbol[pair]} trades")

            print(f"{'=' * 80}\n")

            # Generate daily summary
            logger.generate_daily_summary()
            logger.export_to_csv()

            # Get monthly stats if available
            logger.get_monthly_stats()

            current_date = now.date()
            trades_today_per_symbol = {pair: 0 for pair in MAJOR_PAIRS}
            daily_start_balance = balance

            # Reset session cooldowns for all strategies
            for instrument in MAJOR_PAIRS:
                strategies[instrument].reset_daily_cooldowns()

            print(f"[NEW DAY] {current_date} | Starting Balance: ${balance:,.2f}")
            print(f"[NEW DAY] Session cooldowns reset for all symbols\n")

        # Organize positions by instrument (already fetched above for Friday check)
        positions_by_instrument = {t['instrument']: t for t in open_trades}

        # Detect closed trades (for logging)
        current_trade_ids = {t['id'] for t in open_trades}
        closed_trade_ids = tracked_trade_ids - current_trade_ids

        if closed_trade_ids:
            # Some trades were closed - fetch recent trades to get exit data
            print(f"\n[TRADES CLOSED] Detected {len(closed_trade_ids)} closed position(s)")

            for trade_id in closed_trade_ids:
                # Get entry data if we have it
                if trade_id in trade_entry_data:
                    entry_data = trade_entry_data[trade_id]

                    # Fetch closed trade data from OANDA
                    closed_trades = client.get_trade_history(count=50)

                    for closed_trade in closed_trades:
                        if closed_trade.get('id') == trade_id:
                            # Calculate P&L
                            pnl = closed_trade.get('realized_pl', 0)
                            exit_price = closed_trade.get('average_close_price', entry_data['entry_price'])

                            # Get exit reason from OANDA (now properly detected)
                            exit_reason = closed_trade.get('exit_reason', 'MANUAL')

                            # Calculate metrics
                            position_value = abs(entry_data['units'] * entry_data['entry_price'])
                            pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0

                            # Get current balance
                            current_balance = client.get_balance()

                            # LOG TRADE EXIT
                            logger.log_trade_exit(
                                instrument=entry_data['instrument'],
                                direction=entry_data['direction'],
                                entry_price=entry_data['entry_price'],
                                entry_time=entry_data['entry_time'],
                                exit_price=exit_price,
                                exit_time=datetime.now(pytz.UTC),
                                exit_reason=exit_reason,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                account_balance=current_balance,
                                trade_id=trade_id
                            )

                            print(f"  {entry_data['instrument']} {exit_reason}: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                            break

                    # Remove from tracking
                    del trade_entry_data[trade_id]

        # Update tracked trades
        tracked_trade_ids = current_trade_ids

        # Clean up trailing stop tracking for closed trades
        closed_trailing_ids = set(trailing_stop_tracking.keys()) - current_trade_ids
        for closed_id in closed_trailing_ids:
            if closed_id in trailing_stop_tracking:
                del trailing_stop_tracking[closed_id]

        # ============================================================
        # TRAILING STOP MONITORING - Check if we should activate
        # Trailing stop only activates when profit reaches trigger level
        # This prevents premature exits on volatile pairs
        # ============================================================
        for trade in open_trades:
            trade_id = trade['id']

            # Skip if not tracked or already activated
            if trade_id not in trailing_stop_tracking:
                continue
            if trailing_stop_tracking[trade_id]['activated']:
                continue

            ts_data = trailing_stop_tracking[trade_id]
            instrument = ts_data['instrument']
            entry_price = ts_data['entry_price']
            direction = ts_data['direction']
            trigger_distance = ts_data['trigger_distance']
            trail_distance = ts_data['trail_distance']

            # Get current price
            price_data = client.get_current_price(instrument)
            if not price_data:
                continue

            # Use bid for LONG, ask for SHORT
            if direction == "BUY":
                current_price = price_data['bid']
                profit_distance = current_price - entry_price
            else:  # SELL
                current_price = price_data['ask']
                profit_distance = entry_price - current_price

            pip_mult = 100 if 'JPY' in instrument else 10000

            # Check if profit has reached trigger level
            if profit_distance >= trigger_distance:
                # Activate trailing stop!
                print(f"\n[TRAILING ACTIVATED] {instrument} reached {profit_distance*pip_mult:.1f}p profit (trigger: {trigger_distance*pip_mult:.1f}p)")
                print(f"  Adding trailing stop: {trail_distance*pip_mult:.1f} pips")

                # Modify trade to add trailing stop
                result = client.modify_trade(
                    trade_id=trade_id,
                    trailing_stop_distance=trail_distance
                )

                if result.get('success'):
                    trailing_stop_tracking[trade_id]['activated'] = True
                    print(f"  ✓ Trailing stop activated successfully")
                else:
                    print(f"  ✗ Failed to activate: {result.get('error', 'Unknown')}")

        # Display open positions (every 10 iterations)
        if len(open_trades) > 0 and iteration % 10 == 0:
            # Get current balance for display
            balance = client.get_balance()
            print(f"\n[{now.strftime('%H:%M:%S')}] Balance: ${balance:,.2f} | OPEN POSITIONS ({len(open_trades)}):")
            for trade in open_trades:
                symbol = trade['instrument']
                unrealized_pl = trade['unrealized_pl']
                entry_price = trade['price']
                units = trade['units']

                # Get current price for this symbol
                price_data = client.get_current_price(symbol)
                if price_data:
                    # Use bid for LONG, ask for SHORT
                    if units > 0:  # LONG
                        current_price = price_data['bid']
                    else:  # SHORT
                        current_price = price_data['ask']
                else:
                    current_price = entry_price

                # Calculate pips
                pip_multiplier = 10000 if 'JPY' not in symbol else 100
                if units > 0:  # LONG
                    pips = (current_price - entry_price) * pip_multiplier
                else:  # SHORT
                    pips = (entry_price - current_price) * pip_multiplier

                # Check if trailing stop is active from our tracking
                trailing_status = ""
                trade_id = trade['id']
                if trade.get('trailing_stop'):
                    trailing_dist_val = float(trade['trailing_stop']) * pip_multiplier if trade['trailing_stop'] else 0
                    trailing_status = f" [TRAILING: {trailing_dist_val:.1f}p]"
                elif trade_id in trailing_stop_tracking:
                    ts_data = trailing_stop_tracking[trade_id]
                    trigger_pips = ts_data['trigger_distance'] * pip_multiplier
                    if ts_data['activated']:
                        trailing_status = f" [TRAILING: {ts_data['trail_distance']*pip_multiplier:.1f}p]"
                    else:
                        # Show progress toward trigger
                        progress_pct = (pips / trigger_pips) * 100 if trigger_pips > 0 else 0
                        trailing_status = f" [Trail @ {trigger_pips:.0f}p: {progress_pct:.0f}%]"

                # Show TP and SL distances
                tp_price = trade.get('take_profit', 0)
                sl_price = trade.get('stop_loss', 0)

                tp_dist = ""
                sl_dist = ""
                if tp_price > 0:
                    if units > 0:  # LONG
                        tp_pips = (tp_price - current_price) * pip_multiplier
                    else:  # SHORT
                        tp_pips = (current_price - tp_price) * pip_multiplier
                    tp_dist = f" | TP: {tp_pips:.1f}p away"

                if sl_price > 0:
                    if units > 0:  # LONG
                        sl_pips = (current_price - sl_price) * pip_multiplier
                    else:  # SHORT
                        sl_pips = (sl_price - current_price) * pip_multiplier
                    sl_dist = f" | SL: {sl_pips:.1f}p away"

                print(f"  {symbol}: ${unrealized_pl:+.2f} ({pips:+.1f} pips){tp_dist}{sl_dist}{trailing_status}")

        # Check for entries on each pair (only every 60 seconds)
        if iteration % 12 == 0:  # 12 * 5 sec = 60 sec
            # Get fresh balance before checking for signals
            balance = client.get_balance()

            # Get current session
            current_session = strategies[MAJOR_PAIRS[0]].get_current_session(now)

            # Check if we should skip new trades (Friday last hour)
            if friday_no_new_trades:
                print(f"\n[{now.strftime('%H:%M:%S')}] Balance: ${balance:,.2f} | Session: {current_session} | [FRIDAY] No new trades in last hour")
            else:
                print(f"\n[{now.strftime('%H:%M:%S')}] Balance: ${balance:,.2f} | Session: {current_session} | Checking signals...")

            # Only check if we have room for more positions AND not Friday last hour
            if len(open_trades) < max_concurrent and not friday_no_new_trades:
                for instrument in MAJOR_PAIRS:
                    # Skip if already have position
                    if instrument in positions_by_instrument:
                        continue

                    # Skip if daily limit reached
                    if trades_today_per_symbol[instrument] >= 10:
                        continue

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
                            continue

                        # Get current price
                        price_data = client.get_current_price(instrument)
                        if not price_data:
                            continue

                        current_price = (price_data['bid'] + price_data['ask']) / 2
                        spread = price_data['spread'] * 10000

                        # Check for signal
                        daily_pl_pct = (balance - daily_start_balance) / daily_start_balance

                        signal = strategies[instrument].should_enter_trade(
                            instrument=instrument,
                            df_1min=df_1min,
                            df_5min=df_5min,
                            df_15min=df_15min,
                            df_30min=df_30min,
                            current_positions=0,  # Already filtered out instruments with positions above
                            trades_today=trades_today_per_symbol[instrument],
                            daily_pl_pct=daily_pl_pct
                        )

                        # Show signal validation details
                        if signal["action"] == "SKIP":
                            # Show why signal was rejected
                            print(f"  {instrument}: {signal['reason']}")

                            # Log skipped signal
                            logger.log_signal_skipped(
                                instrument=instrument,
                                reason=signal['reason'],
                                timestamp=datetime.now(pytz.UTC),
                                signal_analysis=signal.get('analysis', {})
                            )
                        elif signal["action"] in ["BUY", "SELL"]:
                            # Show detailed analysis for valid signals
                            analysis = signal.get('analysis', {})
                            momentum = analysis.get('momentum', {})
                            htf = analysis.get('htf', {})
                            rsi = analysis.get('rsi', 0)

                            print(f"\n  {instrument} [{signal['action']}]:")
                            print(f"    Momentum: {momentum.get('signal', 'N/A')} (score: {momentum.get('score', 0)})")
                            print(f"    HTF Trend: {htf.get('trend', 'N/A')}")
                            print(f"    RSI: {rsi:.1f}")
                            print(f"    Confidence: {signal['confidence']}")

                        if signal["action"] in ["BUY", "SELL"]:
                            # Check if we still have room
                            if len(open_trades) >= max_concurrent:
                                print(f"  {instrument}: Signal found but max concurrent reached")
                                break

                            print(f"\n{'=' * 80}")
                            print(f"[ENTRY SIGNAL] {signal['action']} {instrument}")
                            print(f"  Price: {current_price:.5f}")
                            print(f"  Spread: {spread:.1f} pips")
                            print(f"  Reason: {signal['reason']}")
                            print(f"  Confidence: {signal['confidence']}")

                            # Get fresh balance right before placing order for accurate position sizing
                            balance = client.get_balance()

                            # Calculate position size with current balance
                            units = strategies[instrument].calculate_position_size(balance, current_price, instrument)

                            # Set direction
                            if signal["action"] == "SELL":
                                units = -units

                            # Calculate SL and TP (pair-specific)
                            stop_loss, take_profit = strategies[instrument].calculate_stop_loss_take_profit(
                                current_price, signal["action"], instrument
                            )

                            # Get pair-specific trailing stop settings
                            trail_trigger_pct, trail_dist_pct = strategies[instrument].get_pair_trailing_settings(instrument)
                            trailing_distance = trail_dist_pct  # Absolute distance
                            trailing_trigger = trail_trigger_pct  # Trigger level

                            # Calculate pip multiplier for display
                            pip_mult = 100 if 'JPY' in instrument else 10000

                            # Calculate actual risk in dollars
                            sl_pips = abs(stop_loss - current_price) * pip_mult
                            tp_pips = abs(take_profit - current_price) * pip_mult
                            risk_dollars = sl_pips * 1.0  # $1 per pip
                            reward_dollars = tp_pips * 1.0

                            print(f"\n  Placing Order:")
                            print(f"    Units: {abs(units):,} ($1/pip position sizing)")
                            print(f"    Stop Loss: {stop_loss:.5f} ({sl_pips:.1f} pips = ${risk_dollars:.2f} risk)")
                            print(f"    Take Profit: {take_profit:.5f} ({tp_pips:.1f} pips = ${reward_dollars:.2f} target)")
                            print(f"    Trailing Stop: {trailing_distance*pip_mult:.1f}p (activates at {trailing_trigger*pip_mult:.1f}p profit)")

                            # Place order WITHOUT trailing stop - we'll add it later when trigger is reached
                            # This prevents premature trailing stop exits on volatile pairs
                            result = client.place_market_order(
                                instrument=instrument,
                                units=units,
                                stop_loss=stop_loss,
                                take_profit=take_profit
                                # trailing_stop_distance NOT set here - managed manually
                            )

                            if result.get('success'):
                                trades_today_per_symbol[instrument] += 1
                                total_today = sum(trades_today_per_symbol.values())

                                # Record session cooldown for this symbol
                                strategies[instrument].record_trade_session(instrument, now)
                                current_session = strategies[instrument].get_current_session(now)

                                print(f"\n  ORDER FILLED!")
                                print(f"    Order ID: {result.get('order_id')}")
                                print(f"    Trade ID: {result.get('trade_id')}")
                                print(f"    Fill Price: {result.get('filled_price'):.5f}")
                                print(f"    Session: {current_session} (cooldown until next session)")
                                print(f"    Trades Today: {trades_today_per_symbol[instrument]}/10 ({instrument})")
                                print(f"    Total Trades Today: {total_today}")
                                print(f"    Open Positions: {len(open_trades)+1}/{max_concurrent}")

                                # LOG TRADE ENTRY
                                entry_time = datetime.now(pytz.UTC)
                                filled_price = result.get('filled_price', current_price)

                                logger.log_trade_entry(
                                    instrument=instrument,
                                    direction=signal["action"],
                                    entry_price=filled_price,
                                    entry_time=entry_time,
                                    units=abs(units),
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    trailing_distance=trailing_distance,
                                    signal_analysis=signal.get('analysis', {}),
                                    account_balance=balance,
                                    trade_id=result.get('trade_id')
                                )

                                # Store entry data for exit logging
                                trade_id = result.get('trade_id')
                                if trade_id:
                                    trade_entry_data[trade_id] = {
                                        'instrument': instrument,
                                        'direction': signal["action"],
                                        'entry_price': filled_price,
                                        'entry_time': entry_time,
                                        'units': abs(units)
                                    }
                                    tracked_trade_ids.add(trade_id)

                                    # Track trailing stop for later activation
                                    trailing_stop_tracking[trade_id] = {
                                        'instrument': instrument,
                                        'direction': signal["action"],
                                        'entry_price': filled_price,
                                        'trigger_distance': trailing_trigger,  # Absolute price distance
                                        'trail_distance': trailing_distance,   # Absolute trail distance
                                        'activated': False
                                    }
                            else:
                                print(f"\n  ORDER FAILED: {result.get('error')}")

                            print(f"{'=' * 80}\n")

                            # Refresh open trades
                            open_trades = client.get_open_trades()

                            # Break if we hit max concurrent
                            if len(open_trades) >= max_concurrent:
                                break

                    except Exception as e:
                        print(f"  ERROR checking {instrument}: {e}")
                        continue

            else:
                print(f"  Max concurrent positions reached ({max_concurrent})")

        time.sleep(5)

except KeyboardInterrupt:
    print(f"\n\n{'=' * 80}")
    print("STOPPING MULTI-SYMBOL LIVE TRADING")
    print(f"{'=' * 80}")

    # Check for open positions
    open_trades = client.get_open_trades() or []

    if len(open_trades) > 0:
        print(f"\nYou have {len(open_trades)} open position(s):")

        total_unrealized_pl = 0
        for trade in open_trades:
            symbol = trade['instrument']
            units = trade['units']
            entry_price = trade['price']
            unrealized_pl = trade['unrealized_pl']
            direction = "LONG" if units > 0 else "SHORT"
            total_unrealized_pl += unrealized_pl

            # Get current price
            price_data = client.get_current_price(symbol)
            if price_data:
                current_price = price_data['bid'] if units > 0 else price_data['ask']
            else:
                current_price = entry_price

            # Calculate pips
            pip_multiplier = 10000 if 'JPY' not in symbol else 100
            if units > 0:
                pips = (current_price - entry_price) * pip_multiplier
            else:
                pips = (entry_price - current_price) * pip_multiplier

            print(f"  - {symbol}: {direction} @ {entry_price:.5f} | P&L: ${unrealized_pl:+.2f} ({pips:+.1f} pips)")

        print(f"\nTotal Unrealized P&L: ${total_unrealized_pl:+.2f}")
        print("\nWhat would you like to do with these positions?")
        print("  [1] Close all positions now")
        print("  [2] Keep positions open (TP/SL will remain active)")

        try:
            choice = input("\nEnter choice (1 or 2): ").strip()

            if choice == "1":
                print(f"\nClosing {len(open_trades)} position(s)...")
                for trade in open_trades:
                    result = client.close_trade(trade['id'])
                    if result.get('success'):
                        print(f"  {trade['instrument']} - Trade {trade['id']} closed at {result['closed_price']:.5f}")
                        print(f"    P&L: ${result['pl']:+,.2f}")
                print("\nAll positions closed.")
            else:
                print(f"\nKeeping {len(open_trades)} position(s) open.")
                print("Take Profit and Stop Loss orders remain active on OANDA.")
                print("You can restart this script later to continue monitoring.")
        except:
            print("\nKeeping positions open by default.")
    else:
        print("\nNo open positions to manage.")

    # Final summary
    final_balance = client.get_balance()
    total_pl = final_balance - balance
    total_return = (final_balance / balance - 1) * 100
    duration = datetime.now(pytz.UTC) - start_time
    total_trades = sum(trades_today_per_symbol.values())

    print(f"\n{'=' * 80}")
    print("SESSION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Duration: {duration}")
    print(f"Total Trades: {total_trades}")

    # Per-symbol breakdown
    print(f"\nTrades per Symbol:")
    for pair in MAJOR_PAIRS:
        if trades_today_per_symbol[pair] > 0:
            print(f"  {pair}: {trades_today_per_symbol[pair]} trades")

    print(f"\nFinancial Summary:")
    print(f"  Initial Balance: ${balance:,.2f}")
    print(f"  Final Balance: ${final_balance:,.2f}")
    print(f"  Total P&L: ${total_pl:+,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"{'=' * 80}\n")

except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nMulti-symbol live trading stopped.")
