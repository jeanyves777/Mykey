"""
OPTIMIZED Multi-Symbol Forex LIVE TRADING - 86%+ WIN RATE STRATEGY
Trades 7 OPTIMIZED FOREX PAIRS with Session + Volume + Trend Filters

!!! REAL MONEY TRADING - USE WITH CAUTION !!!

Configuration loaded from optimized_live_config.py
Strategy logic from optimized_strategy.py

Optimization Results: +455 pips (3-day backtest)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.utils.trade_logger import ForexTradeLogger

# Import LIVE config
from trading_system.Forex_Trading.config import optimized_live_config as config

# Import strategy logic
from trading_system.Forex_Trading.strategies.optimized_strategy import (
    calculate_indicators,
    get_signal,
    print_strategy_info
)

import pandas as pd
import time
from datetime import datetime, timedelta
import pytz

# =============================================================================
# SESSION DEFINITIONS FOR LOSS COOLDOWN
# =============================================================================
# After 1 loss, pair waits until next session opens
# Sessions (UTC): ASIAN 0-8, LONDON 8-16, NEW_YORK 13-21
SESSION_START_HOURS = {
    'ASIAN': 0,      # 00:00 UTC (Tokyo/Sydney open)
    'LONDON': 8,     # 08:00 UTC
    'NEW_YORK': 13,  # 13:00 UTC
}

def get_current_session(hour_utc: int) -> str:
    """Determine current trading session based on UTC hour."""
    if 0 <= hour_utc < 8:
        return 'ASIAN'
    elif 8 <= hour_utc < 13:
        return 'LONDON'
    elif 13 <= hour_utc < 21:
        return 'NEW_YORK'
    else:
        return 'ASIAN'  # Late night = next Asian session

def get_next_session_start(now_utc: datetime) -> tuple:
    """
    Get the next session start time after current time.
    Returns (session_name, start_datetime)
    """
    hour = now_utc.hour

    # Determine next session based on current hour
    if hour < 8:
        # Before London -> Next is LONDON at 08:00 today
        next_session = 'LONDON'
        next_start = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)
    elif hour < 13:
        # Before NY -> Next is NEW_YORK at 13:00 today
        next_session = 'NEW_YORK'
        next_start = now_utc.replace(hour=13, minute=0, second=0, microsecond=0)
    elif hour < 21:
        # During NY -> Next is ASIAN at 00:00 tomorrow
        next_session = 'ASIAN'
        next_start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # After NY close -> Next is ASIAN at 00:00 tomorrow
        next_session = 'ASIAN'
        next_start = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    return next_session, next_start

# ============================================================================
# REAL MONEY WARNING
# ============================================================================
print("\n" + "!" * 80)
print("!" * 80)
print("!!!")
print("!!!   WARNING: REAL MONEY TRADING MODE")
print("!!!")
print("!!!   This script will trade with REAL MONEY on your OANDA LIVE account.")
print("!!!   All losses are REAL and PERMANENT.")
print("!!!")
print("!!!   Make sure you understand the risks before proceeding.")
print("!!!")
print("!" * 80)
print("!" * 80 + "\n")

# Require explicit confirmation
print("To start live trading, type 'I ACCEPT THE RISK' exactly:")
confirmation = input("> ").strip()

if confirmation != "I ACCEPT THE RISK":
    print("\nConfirmation not received. Exiting for safety.")
    print("If you want to practice, use: run_optimized_forex_paper.py")
    sys.exit(0)

print("\nConfirmation received. Starting LIVE trading...\n")

# Print strategy configuration from config
print_strategy_info(config)

# Get typical SL from first pair for display
first_pair = config.OPTIMIZED_PAIRS[0]
typical_sl = config.get_pair_settings(first_pair)['sl_pips']

print("\nPosition Sizing: $1 PER PIP (consistent risk)")
print(f"  - Risk Example: {typical_sl} pip SL = ${typical_sl} risk per trade")
print(f"  - Max Concurrent: {config.MAX_CONCURRENT_POSITIONS} positions")
print(f"  - Cooldown: {config.COOLDOWN_MINUTES} min between trades per symbol")
print("=" * 80)

# Initialize OANDA LIVE ACCOUNT
print("\n" + "=" * 80)
print(f"***  {config.ACCOUNT_TYPE.upper()} TRADING MODE - OANDA {config.ACCOUNT_TYPE.upper()} ACCOUNT - REAL MONEY  ***")
print("=" * 80)
print(f"\n[1/4] Connecting to OANDA {config.ACCOUNT_TYPE.upper()} Account...")
client = OandaClient(config.ACCOUNT_TYPE)

account_info = client.get_account_info()
if account_info:
    balance = client.get_balance()
    print(f"      Account: {client.account_id}")
    print(f"      Balance: ${balance:,.2f} (REAL MONEY)")
    print(f"      Status: CONNECTED TO LIVE")
else:
    print("      ERROR: Could not connect to OANDA LIVE")
    sys.exit(1)

# Second confirmation with balance
print(f"\n" + "!" * 70)
print(f"You are about to trade with ${balance:,.2f} of REAL MONEY")
print("!" * 70)
print("\nType 'START' to begin live trading:")
start_confirm = input("> ").strip()

if start_confirm != "START":
    print("\nCancelled. Use run_optimized_forex_paper.py to practice safely.")
    sys.exit(0)

# Initialize Trade Logger
print("\n[2/4] Initializing Trade Logger...")
logger = ForexTradeLogger(log_dir=config.LOG_DIR)
print(f"      Logs will be saved to: {config.LOG_DIR}/")

# Trading state
trades_today_per_symbol = {pair: 0 for pair in config.OPTIMIZED_PAIRS}
daily_start_balance = balance
start_time = datetime.now(pytz.UTC)
current_date = start_time.date()

# Cooldown tracking (last trade time per symbol)
last_trade_time = {pair: None for pair in config.OPTIMIZED_PAIRS}

# SESSION-BASED LOSS COOLDOWN
# After 1 loss, pair is blocked until next session opens
pair_blocked_until = {pair: None for pair in config.OPTIMIZED_PAIRS}  # datetime when pair can trade again

# Track open trades
tracked_trade_ids = set()
trade_entry_data = {}
last_known_trades = []
consecutive_errors = 0

# SESSION STATISTICS TRACKING
session_stats = {
    'total_trades': 0,
    'wins': 0,
    'losses': 0,
    'total_pnl': 0.0,
    'by_pair': {pair: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0} for pair in config.OPTIMIZED_PAIRS}
}

# Check for existing open trades
print("\n[3/4] Checking for existing open positions...")
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
        tracked_trade_ids.add(trade['id'])
    print("      These positions will be managed automatically.")
else:
    print("      No existing positions found.")

print("\n[4/4] Starting LIVE Trading (REAL MONEY)...")
print("=" * 80)
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Initial Balance: ${balance:,.2f} (REAL)")
print(f"Max Concurrent Positions: {config.MAX_CONCURRENT_POSITIONS}")
print(f"Cooldown: {config.COOLDOWN_MINUTES} minutes between trades per symbol")
if len(existing_trades) > 0:
    print(f"Existing Positions: {len(existing_trades)}")

print("\nPress Ctrl+C to stop trading\n")
print("=" * 80)

try:
    iteration = 0

    while True:
        iteration += 1
        now = datetime.now(pytz.UTC)

        # Get all open positions
        open_trades = client.get_open_trades()

        # Handle API errors gracefully
        if open_trades is None:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                print(f"[WARNING] {consecutive_errors} consecutive API errors - using cached data")
            open_trades = last_known_trades
        else:
            consecutive_errors = 0
            last_known_trades = open_trades

        # FRIDAY MARKET CLOSE PROTECTION
        # Forex market closes at 22:00 UTC on Friday
        # - No new trades after 21:00 UTC (1 hour before close)
        # - Force close all positions at 21:50 UTC (10 min before close)
        is_friday = now.weekday() == 4
        hour_utc = now.hour
        minute_utc = now.minute
        friday_no_new_trades = is_friday and hour_utc >= 21
        # Close if 21:50+ OR if somehow we're past 22:00 on Friday (safety net)
        friday_close_all = is_friday and ((hour_utc == 21 and minute_utc >= 50) or hour_utc >= 22)

        if friday_close_all and len(open_trades) > 0:
            print(f"\n{'=' * 80}")
            print(f"[FRIDAY MARKET CLOSE] Closing all {len(open_trades)} positions")
            for trade in open_trades:
                result = client.close_trade(trade['id'])
                print(f"  {trade['instrument']}: {'Closed' if result.get('success') else 'Failed'}")
            print(f"{'=' * 80}\n")

        # Daily reset
        if now.date() > current_date:
            daily_pl = balance - daily_start_balance
            daily_pl_pct = daily_pl / daily_start_balance * 100
            total_trades = sum(trades_today_per_symbol.values())

            print(f"\n{'=' * 80}")
            print(f"[DAY END] {current_date} - LIVE ACCOUNT")
            print(f"  Total Trades: {total_trades}")
            print(f"  P&L: ${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%) REAL MONEY")
            print(f"  Balance: ${balance:,.2f}")

            for pair in config.OPTIMIZED_PAIRS:
                if trades_today_per_symbol[pair] > 0:
                    print(f"    {pair}: {trades_today_per_symbol[pair]} trades")

            print(f"{'=' * 80}\n")

            # Generate daily summary
            logger.generate_daily_summary()
            logger.export_to_csv()

            current_date = now.date()
            trades_today_per_symbol = {pair: 0 for pair in config.OPTIMIZED_PAIRS}
            daily_start_balance = balance
            last_trade_time = {pair: None for pair in config.OPTIMIZED_PAIRS}
            pair_blocked_until = {pair: None for pair in config.OPTIMIZED_PAIRS}  # Reset session blocks

            print(f"[NEW DAY] {current_date} | Starting Balance: ${balance:,.2f}\n")

        # Organize positions by instrument
        positions_by_instrument = {t['instrument']: t for t in open_trades}

        # Detect closed trades
        current_trade_ids = {t['id'] for t in open_trades}
        closed_trade_ids = tracked_trade_ids - current_trade_ids

        if closed_trade_ids:
            print(f"\n[TRADES CLOSED] Detected {len(closed_trade_ids)} closed position(s)")

            # Fetch trade history from OANDA for accurate exit data
            closed_trades = client.get_trade_history(count=50)

            for trade_id in closed_trade_ids:
                if trade_id in trade_entry_data:
                    entry_data = trade_entry_data[trade_id]
                    instrument = entry_data['instrument']

                    # Find this trade in OANDA's closed trades
                    oanda_trade = None
                    for ct in closed_trades:
                        if ct.get('id') == trade_id:
                            oanda_trade = ct
                            break

                    if oanda_trade:
                        # Use OANDA's actual data for accuracy
                        pnl = oanda_trade.get('realized_pl', 0)
                        exit_reason = oanda_trade.get('exit_reason', 'MANUAL')
                        exit_price = oanda_trade.get('average_close_price', entry_data['entry_price'])
                        exit_time = oanda_trade.get('close_time', datetime.now(pytz.UTC))
                        open_price = oanda_trade.get('open_price', entry_data['entry_price'])

                        # Calculate pips accurately
                        pip_mult = 100 if 'JPY' in instrument else 10000
                        if entry_data['direction'] == 'BUY':
                            pips = (exit_price - open_price) * pip_mult
                        else:
                            pips = (open_price - exit_price) * pip_mult

                        logger.log_trade_exit(
                            instrument=instrument,
                            direction=entry_data['direction'],
                            entry_price=open_price,
                            entry_time=entry_data['entry_time'],
                            exit_price=exit_price,
                            exit_time=exit_time,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(pnl / balance * 100) if balance > 0 else 0,
                            account_balance=client.get_balance(),
                            trade_id=trade_id
                        )

                        # UPDATE SESSION STATISTICS
                        session_stats['total_trades'] += 1
                        session_stats['total_pnl'] += pnl
                        session_stats['by_pair'][instrument]['trades'] += 1
                        session_stats['by_pair'][instrument]['pnl'] += pnl
                        if pnl > 0:
                            session_stats['wins'] += 1
                            session_stats['by_pair'][instrument]['wins'] += 1
                        else:
                            session_stats['losses'] += 1
                            session_stats['by_pair'][instrument]['losses'] += 1

                        result_emoji = "WIN" if pnl > 0 else "LOSS"
                        print(f"  {instrument} {exit_reason}: ${pnl:+.2f} ({pips:+.1f} pips) [{result_emoji}] REAL")

                        # SESSION-BASED LOSS COOLDOWN
                        # After a loss, block this pair until next session opens
                        if pnl < 0:
                            next_session, next_start = get_next_session_start(now)
                            pair_blocked_until[instrument] = next_start
                            current_session = get_current_session(now.hour)
                            print(f"  >>> {instrument} BLOCKED until {next_session} session ({next_start.strftime('%H:%M UTC')})")
                    else:
                        # Trade not found in OANDA history yet - use basic logging
                        print(f"  {instrument}: Trade {trade_id} closed (awaiting OANDA data) REAL")

                    del trade_entry_data[trade_id]

            # DISPLAY PER-SYMBOL STATS AFTER TRADE CLOSES
            session_wr = (session_stats['wins'] / session_stats['total_trades'] * 100) if session_stats['total_trades'] > 0 else 0
            print(f"\n  {'=' * 65}")
            print(f"  SESSION STATS: {session_stats['wins']}W/{session_stats['losses']}L ({session_wr:.1f}%) | Total P&L: ${session_stats['total_pnl']:+.2f} REAL")
            print(f"  {'-' * 65}")
            print(f"  {'PAIR':<10} {'TRADES':<8} {'WINS':<6} {'LOSSES':<8} {'WIN%':<8} {'P&L'}")
            print(f"  {'-' * 65}")
            for pair in config.OPTIMIZED_PAIRS:
                ps = session_stats['by_pair'][pair]
                if ps['trades'] > 0:
                    pair_wr = (ps['wins'] / ps['trades'] * 100) if ps['trades'] > 0 else 0
                    print(f"  {pair:<10} {ps['trades']:<8} {ps['wins']:<6} {ps['losses']:<8} {pair_wr:<7.1f}% ${ps['pnl']:+.2f}")
            print(f"  {'-' * 65}")
            print(f"  {'TOTAL':<10} {session_stats['total_trades']:<8} {session_stats['wins']:<6} {session_stats['losses']:<8} {session_wr:<7.1f}% ${session_stats['total_pnl']:+.2f}")
            print(f"  {'=' * 65}\n")

        tracked_trade_ids = current_trade_ids

        # Display open positions (every 10 iterations)
        if len(open_trades) > 0 and iteration % 10 == 0:
            balance = client.get_balance()

            # Calculate session win rate
            session_wr = (session_stats['wins'] / session_stats['total_trades'] * 100) if session_stats['total_trades'] > 0 else 0

            # Calculate total unrealized P&L from open positions
            total_unrealized_pl = sum(t['unrealized_pl'] for t in open_trades)

            print(f"\n[{now.strftime('%H:%M:%S')}] Balance: ${balance:,.2f} REAL | OPEN POSITIONS ({len(open_trades)}) | Open P&L: ${total_unrealized_pl:+.2f}:")

            for trade in open_trades:
                symbol = trade['instrument']
                unrealized_pl = trade['unrealized_pl']
                entry_price = trade['price']
                units = trade['units']

                price_data = client.get_current_price(symbol)
                if price_data:
                    current_price = price_data['bid'] if units > 0 else price_data['ask']
                else:
                    current_price = entry_price

                pip_mult = 100 if 'JPY' in symbol else 10000
                pips = (current_price - entry_price) * pip_mult if units > 0 else (entry_price - current_price) * pip_mult

                tp_price = trade.get('take_profit', 0)
                sl_price = trade.get('stop_loss', 0)

                tp_dist = ""
                sl_dist = ""
                if tp_price > 0:
                    tp_pips = (tp_price - current_price) * pip_mult if units > 0 else (current_price - tp_price) * pip_mult
                    tp_dist = f" | TP: {tp_pips:.1f}p away"
                if sl_price > 0:
                    sl_pips = (current_price - sl_price) * pip_mult if units > 0 else (sl_price - current_price) * pip_mult
                    sl_dist = f" | SL: {sl_pips:.1f}p away"

                print(f"  {symbol}: ${unrealized_pl:+.2f} ({pips:+.1f} pips){tp_dist}{sl_dist}")

            # DISPLAY SESSION STATISTICS TABLE
            if session_stats['total_trades'] > 0:
                print(f"\n  {'=' * 60}")
                print(f"  SESSION STATS: {session_stats['wins']}W/{session_stats['losses']}L ({session_wr:.1f}%) | Total P&L: ${session_stats['total_pnl']:+.2f} REAL")
                print(f"  {'-' * 60}")
                print(f"  {'PAIR':<10} {'TRADES':<8} {'WINS':<6} {'LOSSES':<8} {'WIN%':<8} {'P&L'}")
                print(f"  {'-' * 60}")

                # Show all pairs with trades
                for pair in config.OPTIMIZED_PAIRS:
                    ps = session_stats['by_pair'][pair]
                    if ps['trades'] > 0:
                        pair_wr = (ps['wins'] / ps['trades'] * 100) if ps['trades'] > 0 else 0
                        print(f"  {pair:<10} {ps['trades']:<8} {ps['wins']:<6} {ps['losses']:<8} {pair_wr:<7.1f}% ${ps['pnl']:+.2f}")

                print(f"  {'-' * 60}")
                print(f"  {'TOTAL':<10} {session_stats['total_trades']:<8} {session_stats['wins']:<6} {session_stats['losses']:<8} {session_wr:<7.1f}% ${session_stats['total_pnl']:+.2f}")
                print(f"  {'=' * 60}")

        # Check for entries (every 60 seconds)
        if iteration % 12 == 0:
            balance = client.get_balance()

            if friday_no_new_trades:
                print(f"\n[{now.strftime('%H:%M:%S')}] Balance: ${balance:,.2f} REAL | [FRIDAY] No new trades in last hour")
            else:
                print(f"\n[{now.strftime('%H:%M:%S')}] Balance: ${balance:,.2f} REAL | Checking signals...")

            if len(open_trades) < config.MAX_CONCURRENT_POSITIONS and not friday_no_new_trades:
                for instrument in config.OPTIMIZED_PAIRS:
                    # Skip if already have position
                    if instrument in positions_by_instrument:
                        continue

                    # Check cooldown
                    if last_trade_time[instrument]:
                        time_since_trade = (now - last_trade_time[instrument]).total_seconds() / 60
                        if time_since_trade < config.COOLDOWN_MINUTES:
                            remaining = config.COOLDOWN_MINUTES - time_since_trade
                            print(f"  {instrument}: Cooldown ({remaining:.0f}m remaining)")
                            continue

                    # SESSION-BASED LOSS COOLDOWN
                    # After a loss, pair is blocked until next session opens
                    if pair_blocked_until[instrument]:
                        if now < pair_blocked_until[instrument]:
                            next_session, _ = get_next_session_start(now)
                            time_remaining = pair_blocked_until[instrument] - now
                            hours_remaining = time_remaining.total_seconds() / 3600
                            print(f"  {instrument}: BLOCKED after loss (wait {hours_remaining:.1f}h for {next_session})")
                            continue
                        else:
                            # Block expired, clear it
                            pair_blocked_until[instrument] = None
                            print(f"  {instrument}: Session block expired, trading enabled")

                    # SESSION FILTER - Only trade during high win-rate hours
                    if hasattr(config, 'is_allowed_hour'):
                        if not config.is_allowed_hour(instrument, hour_utc):
                            session_info = config.TRADING_SESSIONS.get(instrument, {})
                            allowed = session_info.get('allowed_hours', [])
                            if allowed:
                                print(f"  {instrument}: Outside trading hours (allowed: {min(allowed):02d}:00-{max(allowed):02d}:00 UTC)")
                            continue

                    try:
                        # Get candles (timeframe from config)
                        candles = client.get_candles(instrument, config.TIMEFRAME, count=config.CANDLE_COUNT)

                        if len(candles) < 50:
                            continue

                        df = pd.DataFrame(candles)
                        df = calculate_indicators(df)

                        # Get signal based on pair's strategy (using config)
                        signal, reason = get_signal(instrument, df, config)

                        if signal is None:
                            print(f"  {instrument}: {reason}")
                            continue

                        # We have a signal!
                        price_data = client.get_current_price(instrument)
                        if not price_data:
                            continue

                        current_price = (price_data['bid'] + price_data['ask']) / 2
                        spread = price_data['spread'] * 10000

                        settings = config.get_pair_settings(instrument)

                        print(f"\n{'=' * 80}")
                        print(f"[LIVE ENTRY SIGNAL] {signal} {instrument} ({settings['strategy']})")
                        print(f"  Price: {current_price:.5f}")
                        print(f"  Spread: {spread:.1f} pips")
                        print(f"  Reason: {reason}")
                        print(f"  Expected WR: {settings['expected_wr']:.1f}%")

                        # Calculate position size (from config)
                        units = config.calculate_position_size(instrument, current_price)
                        if signal == "SELL":
                            units = -units

                        # Calculate TP and SL (from config)
                        take_profit, stop_loss = config.calculate_tp_sl(instrument, current_price, signal)

                        tp_pips = settings['tp_pips']
                        sl_pips = settings['sl_pips']

                        print(f"\n  Placing REAL Order:")
                        print(f"    Units: {abs(units):,} ($1/pip)")
                        print(f"    Take Profit: {take_profit:.5f} (+{tp_pips} pips = ${tp_pips:.2f})")
                        print(f"    Stop Loss: {stop_loss:.5f} (-{sl_pips} pips = ${sl_pips:.2f})")

                        # Place order
                        result = client.place_market_order(
                            instrument=instrument,
                            units=units,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )

                        if result.get('success'):
                            trades_today_per_symbol[instrument] += 1
                            total_today = sum(trades_today_per_symbol.values())
                            last_trade_time[instrument] = now

                            print(f"\n  REAL ORDER FILLED!")
                            print(f"    Order ID: {result.get('order_id')}")
                            print(f"    Trade ID: {result.get('trade_id')}")
                            print(f"    Fill Price: {result.get('filled_price'):.5f}")
                            print(f"    Trades Today: {trades_today_per_symbol[instrument]} ({instrument})")
                            print(f"    Total Trades Today: {total_today}")
                            print(f"    Open Positions: {len(open_trades)+1}/{config.MAX_CONCURRENT_POSITIONS}")

                            # Log trade entry
                            entry_time = datetime.now(pytz.UTC)
                            filled_price = result.get('filled_price', current_price)

                            logger.log_trade_entry(
                                instrument=instrument,
                                direction=signal,
                                entry_price=filled_price,
                                entry_time=entry_time,
                                units=abs(units),
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                trailing_distance=0,
                                signal_analysis={'strategy': settings['strategy'], 'reason': reason},
                                account_balance=balance,
                                trade_id=result.get('trade_id')
                            )

                            # Store for exit logging
                            trade_id = result.get('trade_id')
                            if trade_id:
                                trade_entry_data[trade_id] = {
                                    'instrument': instrument,
                                    'direction': signal,
                                    'entry_price': filled_price,
                                    'entry_time': entry_time,
                                    'units': abs(units)
                                }
                                tracked_trade_ids.add(trade_id)
                        else:
                            print(f"\n  ORDER FAILED: {result.get('error')}")

                        print(f"{'=' * 80}\n")

                        # Refresh open trades
                        open_trades = client.get_open_trades() or []
                        positions_by_instrument = {t['instrument']: t for t in open_trades}

                        if len(open_trades) >= config.MAX_CONCURRENT_POSITIONS:
                            break

                    except Exception as e:
                        print(f"  ERROR checking {instrument}: {e}")
                        continue

            else:
                if len(open_trades) >= config.MAX_CONCURRENT_POSITIONS:
                    print(f"  Max concurrent positions reached ({config.MAX_CONCURRENT_POSITIONS})")

        time.sleep(5)

except KeyboardInterrupt:
    print(f"\n\n{'=' * 80}")
    print("STOPPING LIVE TRADING (REAL MONEY)")
    print(f"{'=' * 80}")

    open_trades = client.get_open_trades() or []

    if len(open_trades) > 0:
        print(f"\nYou have {len(open_trades)} REAL open position(s):")

        total_unrealized_pl = 0
        for trade in open_trades:
            symbol = trade['instrument']
            units = trade['units']
            entry_price = trade['price']
            unrealized_pl = trade['unrealized_pl']
            direction = "LONG" if units > 0 else "SHORT"
            total_unrealized_pl += unrealized_pl

            price_data = client.get_current_price(symbol)
            if price_data:
                current_price = price_data['bid'] if units > 0 else price_data['ask']
            else:
                current_price = entry_price

            pip_mult = 10000 if 'JPY' not in symbol else 100
            pips = (current_price - entry_price) * pip_mult if units > 0 else (entry_price - current_price) * pip_mult

            print(f"  - {symbol}: {direction} @ {entry_price:.5f} | P&L: ${unrealized_pl:+.2f} ({pips:+.1f} pips) REAL")

        print(f"\nTotal Unrealized P&L: ${total_unrealized_pl:+.2f} (REAL MONEY)")
        print("\nWhat would you like to do with these positions?")
        print("  [1] Close all positions now")
        print("  [2] Keep positions open (TP/SL will remain active)")

        try:
            choice = input("\nEnter choice (1 or 2): ").strip()

            if choice == "1":
                print(f"\nClosing {len(open_trades)} REAL position(s)...")
                for trade in open_trades:
                    result = client.close_trade(trade['id'])
                    if result.get('success'):
                        print(f"  {trade['instrument']} - Closed at {result['closed_price']:.5f} | P&L: ${result['pl']:+,.2f} REAL")
                print("\nAll positions closed.")
            else:
                print(f"\nKeeping {len(open_trades)} position(s) open.")
                print("TP and SL orders remain active on OANDA.")
        except:
            print("\nKeeping positions open by default.")
    else:
        print("\nNo open positions to manage.")

    # Final summary
    final_balance = client.get_balance()
    total_pl = final_balance - balance
    total_return = (final_balance / balance - 1) * 100
    duration = datetime.now(pytz.UTC) - start_time

    # Calculate final session win rate
    session_wr = (session_stats['wins'] / session_stats['total_trades'] * 100) if session_stats['total_trades'] > 0 else 0

    print(f"\n{'=' * 80}")
    print("LIVE SESSION SUMMARY (REAL MONEY)")
    print(f"{'=' * 80}")
    print(f"Duration: {duration}")
    print(f"Total Closed Trades: {session_stats['total_trades']}")
    print(f"Wins: {session_stats['wins']} | Losses: {session_stats['losses']} | Win Rate: {session_wr:.1f}%")
    print(f"Total P&L from Closed Trades: ${session_stats['total_pnl']:+.2f} REAL")

    print(f"\nPerformance by Pair:")
    print(f"{'PAIR':<10} {'TRADES':<8} {'WINS':<6} {'LOSSES':<8} {'WIN%':<8} {'P&L'}")
    print("-" * 55)
    for pair in config.OPTIMIZED_PAIRS:
        ps = session_stats['by_pair'][pair]
        if ps['trades'] > 0:
            pair_wr = (ps['wins'] / ps['trades'] * 100) if ps['trades'] > 0 else 0
            print(f"{pair:<10} {ps['trades']:<8} {ps['wins']:<6} {ps['losses']:<8} {pair_wr:<7.1f}% ${ps['pnl']:+.2f}")
    print("-" * 55)

    print(f"\nFinancial Summary (REAL):")
    print(f"  Initial Balance: ${balance:,.2f}")
    print(f"  Final Balance: ${final_balance:,.2f}")
    print(f"  Total P&L: ${total_pl:+,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"{'=' * 80}\n")

    # Log session end with final stats
    logger.log_session_end(final_stats={
        'total_trades': session_stats['total_trades'],
        'wins': session_stats['wins'],
        'losses': session_stats['losses'],
        'win_rate': session_wr,
        'total_pnl': session_stats['total_pnl'],
        'initial_balance': balance,
        'final_balance': final_balance,
        'total_return_pct': total_return,
        'account_type': 'LIVE'
    })

    # Generate session summary and export
    logger.generate_daily_summary()
    logger.export_to_csv()

except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nLive trading stopped.")
