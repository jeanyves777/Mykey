"""
FundedNext Forex Futures Backtest Script

Simulates trading on $25K FundedNext challenge account
Uses OANDA forex data as proxy for futures (same price action)

SYMBOLS (OANDA → Futures mapping):
- EUR_USD → M6E
- GBP_USD → M6B
- USD_JPY → MJY
- USD_CAD → MCD
- USD_CHF → MSF
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Import OANDA client to get forex data
from trading_system.Forex_Trading.engine.oanda_client import OandaClient

# Import our FundedNext strategy
from trading_system.Forex_trading_ninjatrader.fundednext_strategy import FundedNextStrategy

def get_forex_data(client: OandaClient, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get 15min forex data from OANDA"""
    print(f"  Fetching {symbol} data (last 2000 candles)...")

    candles = client.get_candles(
        instrument=symbol,
        granularity='M15',
        count=2000  # Last ~20 days of 15min data
    )

    if not candles or len(candles) == 0:
        print(f"  WARNING: No data for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Set index to time column
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    print(f"  Loaded {len(df)} bars for {symbol} ({df.index.min()} to {df.index.max()})")
    return df

def run_backtest(
    symbols_map: dict,
    start_date: str,
    end_date: str,
    contracts_per_trade: int = 1,
    daily_loss_limit: float = -500,
    is_challenge_mode: bool = True
):
    """
    Run backtest simulation

    symbols_map: {'M6E': 'EUR_USD', 'M6B': 'GBP_USD', ...}
    """
    print("=" * 80)
    print("FUNDEDNEXT $25K CHALLENGE - BACKTEST SIMULATION")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Symbols: {list(symbols_map.keys())}")
    print(f"Mode: {'CHALLENGE' if is_challenge_mode else 'FUNDED'}")
    print("=" * 80)

    # Initialize OANDA client
    print("\n[1/5] Connecting to OANDA...")
    client = OandaClient()

    # Load data for all symbols
    print("\n[2/5] Loading historical data...")
    data = {}
    for futures_symbol, oanda_symbol in symbols_map.items():
        df = get_forex_data(client, oanda_symbol, start_date, end_date)
        if len(df) > 0:
            data[futures_symbol] = df

    if not data:
        print("\nERROR: No data loaded!")
        return

    print(f"\n[3/5] Loaded data for {len(data)} symbols")

    # Initialize strategy
    print("\n[4/5] Initializing strategy...")
    strategy = FundedNextStrategy(
        symbols=list(data.keys()),
        contracts_per_trade=contracts_per_trade,
        daily_loss_limit=daily_loss_limit,
        max_concurrent=5,
        max_trades_per_day=50,
        max_trades_per_symbol=10,
        is_challenge_mode=is_challenge_mode
    )

    # Calculate indicators for all symbols
    print("\n[5/5] Calculating indicators...")
    for symbol in data.keys():
        data[symbol] = strategy.calculate_indicators(data[symbol])
        print(f"  {symbol}: Indicators calculated")

    # Run simulation
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST...")
    print("=" * 80)

    # Get all timestamps (union of all symbols)
    all_timestamps = set()
    for df in data.values():
        all_timestamps.update(df.index.tolist())

    all_timestamps = sorted(list(all_timestamps))

    print(f"\nProcessing {len(all_timestamps)} timestamps...")

    last_percent = -1
    for i, timestamp in enumerate(all_timestamps):
        # Progress
        percent = int(i / len(all_timestamps) * 100)
        if percent != last_percent and percent % 10 == 0:
            print(f"  Progress: {percent}% ({i}/{len(all_timestamps)})")
            last_percent = percent

        # Check exits first (for all symbols at this timestamp)
        for symbol in list(strategy.open_positions.keys()):
            if timestamp in data[symbol].index:
                current_price = data[symbol].loc[timestamp, 'close']
                trade_result = strategy.check_exit(symbol, current_price, timestamp)

                if trade_result:
                    pnl_str = f"+${trade_result['net_pnl']:.2f}" if trade_result['net_pnl'] > 0 else f"-${abs(trade_result['net_pnl']):.2f}"
                    print(f"  [{timestamp}] {symbol} {trade_result['signal']} EXIT ({trade_result['exit_reason']}): {pnl_str} | Balance: ${strategy.current_balance:,.2f}")

        # Check entries (for each symbol)
        for symbol in data.keys():
            if timestamp not in data[symbol].index:
                continue

            df_up_to_now = data[symbol].loc[:timestamp]

            if len(df_up_to_now) < 50:
                continue

            # Check for entry signal
            signal = strategy.check_entry_signals(df_up_to_now, symbol)

            if signal:
                current_price = data[symbol].loc[timestamp, 'close']

                # Check if we should enter
                if strategy.should_enter_trade(symbol, signal, current_price, timestamp):
                    strategy.enter_trade(symbol, signal, current_price, timestamp)

                    pos = strategy.open_positions[symbol]
                    print(f"  [{timestamp}] {symbol} {signal} ENTRY @ {current_price:.5f} | TP: {pos['tp_price']:.5f}, SL: {pos['sl_price']:.5f}")

        # Check if challenge ended
        if strategy.total_profit >= strategy.PROFIT_TARGET:
            print(f"\n{'=' * 80}")
            print("CHALLENGE PASSED! PROFIT TARGET HIT!")
            print(f"{'=' * 80}")
            break

        if strategy.current_balance <= strategy.current_threshold:
            print(f"\n{'=' * 80}")
            print("ACCOUNT FAILED! MAX LOSS LIMIT BREACHED!")
            print(f"{'=' * 80}")
            break

    # Close any remaining positions at end of backtest
    print("\n[BACKTEST END] Closing remaining positions...")
    for symbol in list(strategy.open_positions.keys()):
        final_price = data[symbol].iloc[-1]['close']
        final_time = data[symbol].index[-1]
        trade_result = strategy.exit_trade(symbol, final_price, final_time, 'EOD')
        print(f"  Closed {symbol} @ {final_price:.5f}: ${trade_result['net_pnl']:.2f}")

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    stats = strategy.get_performance_stats()

    print(f"\nACCOUNT PERFORMANCE:")
    print(f"  Initial Balance:    ${strategy.INITIAL_BALANCE:,.2f}")
    print(f"  Final Balance:      ${stats['final_balance']:,.2f}")
    print(f"  Total Profit:       ${stats['total_profit']:+,.2f}")
    print(f"  Total Return:       {stats['total_return_pct']:+.2f}%")

    print(f"\nTRADING STATISTICS:")
    print(f"  Total Trades:       {stats['total_trades']}")
    print(f"  Winners:            {stats['winners']} ({stats['win_rate']:.1f}%)")
    print(f"  Losers:             {stats['losers']}")
    print(f"  Profit Factor:      {stats['profit_factor']:.2f}")
    print(f"  Avg Win:            ${stats['avg_win']:.2f}")
    print(f"  Avg Loss:           ${stats['avg_loss']:.2f}")
    print(f"  Avg Trade:          ${stats['avg_trade']:.2f}")

    print(f"\nFUNDEDNEXT COMPLIANCE:")
    print(f"  Profit Target:      ${strategy.PROFIT_TARGET:,.2f}")
    print(f"  Target Achieved:    {'YES ✓' if stats['challenge_passed'] else 'NO'}")
    print(f"  Max Loss Limit:     ${strategy.MAX_LOSS_LIMIT:,.2f}")
    print(f"  Account Failed:     {'YES (BREACH)' if stats['account_failed'] else 'NO ✓'}")
    print(f"  Max Day Profit:     ${stats['max_day_profit']:.2f} ({stats['max_day_pct']:.1f}% of total)")
    print(f"  Consistency Rule:   {'PASS ✓' if stats['consistency_compliant'] else 'FAIL (>40%)'}")

    # Per-symbol breakdown
    print(f"\nPER-SYMBOL BREAKDOWN:")
    df_trades = pd.DataFrame(strategy.trades_log)
    if len(df_trades) > 0:
        for symbol in strategy.symbols:
            symbol_trades = df_trades[df_trades['symbol'] == symbol]
            if len(symbol_trades) > 0:
                symbol_profit = symbol_trades['net_pnl'].sum()
                symbol_winners = len(symbol_trades[symbol_trades['net_pnl'] > 0])
                symbol_losers = len(symbol_trades[symbol_trades['net_pnl'] < 0])
                symbol_wr = (symbol_winners / len(symbol_trades) * 100) if len(symbol_trades) > 0 else 0

                print(f"  {symbol}: {len(symbol_trades)} trades, {symbol_wr:.1f}% WR, ${symbol_profit:+.2f}")

    # Daily breakdown
    if strategy.daily_profits:
        print(f"\nDAILY PROFIT BREAKDOWN:")
        for date, profit in sorted(strategy.daily_profits.items()):
            pct_of_total = (profit / stats['total_profit'] * 100) if stats['total_profit'] != 0 else 0
            print(f"  {date}: ${profit:+8.2f} ({pct_of_total:5.1f}% of total)")

    # Save detailed results
    print(f"\n[SAVE] Exporting detailed trade log...")
    if len(df_trades) > 0:
        output_file = 'trading_system/Forex_trading_ninjatrader/backtest_results.csv'
        df_trades.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    return strategy, stats


if __name__ == "__main__":
    # Symbol mapping: Futures → OANDA
    symbols_map = {
        'M6E': 'EUR_USD',
        'M6B': 'GBP_USD',
        'MJY': 'USD_JPY',
        'MCD': 'USD_CAD',
        'MSF': 'USD_CHF'
    }

    # Backtest period (Nov 1 - Dec 1, 2024)
    start_str = '2024-11-01'
    end_str = '2024-12-01'

    # Run backtest
    strategy, stats = run_backtest(
        symbols_map=symbols_map,
        start_date=start_str,
        end_date=end_str,
        contracts_per_trade=1,
        daily_loss_limit=-500,
        is_challenge_mode=True
    )

    # Summary
    print(f"\n{'=' * 80}")
    print("FUNDEDNEXT CHALLENGE PROJECTION")
    print(f"{'=' * 80}")

    if stats['challenge_passed']:
        print(f"\n✓ CHALLENGE PASSED!")
        print(f"  - Final Profit: ${stats['total_profit']:,.2f}")
        print(f"  - Total Trades: {stats['total_trades']}")
        print(f"  - Win Rate: {stats['win_rate']:.1f}%")
        print(f"  - All FundedNext rules complied ✓")
        print(f"\n  Next step: Run same strategy in NinjaTrader for validation")

    elif stats['account_failed']:
        print(f"\n❌ CHALLENGE FAILED - Max Loss Limit Breached")
        print(f"  - Loss: ${stats['total_profit']:,.2f}")
        print(f"  - Suggestions:")
        print(f"    1. Reduce daily loss limit to -$300")
        print(f"    2. Trade fewer pairs (focus on EUR/USD + USD/CAD)")
        print(f"    3. Increase win rate filter (wait for 3/3 signals)")

    else:
        print(f"\n⚠ Challenge Not Yet Passed (need ${strategy.PROFIT_TARGET - stats['total_profit']:.2f} more)")
        print(f"  - Current Profit: ${stats['total_profit']:,.2f}")
        print(f"  - Estimated days to pass: {int((strategy.PROFIT_TARGET - stats['total_profit']) / stats['avg_trade'] / 10)} days")
        print(f"  - Keep trading with same settings")

    print(f"\n{'=' * 80}\n")
