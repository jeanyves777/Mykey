#!/usr/bin/env python3
"""
Fetch All Trade History from Binance
=====================================
Fetches past trades and realized PNL from Binance API
and logs them to the trade log JSON file.
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient
from config.trading_config import FUTURES_SYMBOLS, LOGGING_CONFIG, STRATEGY_CONFIG, RISK_CONFIG


def estimate_dca_level_from_size(position_value: float, symbol_budget: float, leverage: int = 20) -> int:
    """
    Estimate DCA level from position size.
    Position value = margin * leverage
    Margin used = position_value / leverage

    DCA allocation (cumulative margin %):
    - No DCA: 10%
    - DCA 1: 10% + 15% = 25%
    - DCA 2: 25% + 20% = 45%
    - DCA 3: 45% + 25% = 70%
    - DCA 4: 70% + 30% = 100%
    """
    if symbol_budget <= 0:
        return 0

    margin_used = position_value / leverage
    margin_pct = margin_used / symbol_budget

    # Cumulative thresholds with some tolerance (±10%)
    if margin_pct > 0.65:  # 70% threshold - DCA level 4
        return 4
    elif margin_pct > 0.40:  # 45% threshold - DCA level 3
        return 3
    elif margin_pct > 0.22:  # 25% threshold - DCA level 2
        return 2
    elif margin_pct > 0.12:  # 10% threshold - DCA level 1
        return 1
    else:
        return 0  # No DCA - initial entry only


def fetch_and_log_trades(days_back: int = 7, testnet: bool = True):
    """Fetch trade history and log to JSON file"""

    client = BinanceClient(testnet=testnet)
    mode = "DEMO" if testnet else "MAINNET"

    print("=" * 70)
    print(f"FETCHING TRADE HISTORY - {mode}")
    print("=" * 70)

    # Get log file path
    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        LOGGING_CONFIG["log_dir"]
    )
    os.makedirs(log_dir, exist_ok=True)
    trade_log_path = os.path.join(log_dir, LOGGING_CONFIG["trade_log_file"])

    # Load existing trades
    existing_trades = []
    if os.path.exists(trade_log_path):
        try:
            with open(trade_log_path, 'r') as f:
                existing_trades = json.load(f)
            print(f"Loaded {len(existing_trades)} existing trades from log")
        except:
            existing_trades = []

    # Get existing trade timestamps to avoid duplicates
    existing_timestamps = set()
    for trade in existing_trades:
        if "timestamp" in trade:
            existing_timestamps.add(trade["timestamp"][:19])  # Compare up to seconds

    all_new_trades = []
    total_pnl = 0.0

    # Get balance to estimate symbol budget for DCA level detection
    try:
        balance = client.get_balance()
        num_symbols = len(FUTURES_SYMBOLS)
        buffer_pct = RISK_CONFIG.get("allocation_buffer_pct", 0.05)
        symbol_budget = balance * (1 - buffer_pct) / num_symbols
        print(f"Balance: ${balance:,.2f} | Budget per symbol: ${symbol_budget:,.2f}")
    except:
        balance = 0
        symbol_budget = 600  # Default estimate

    # Calculate time range - Binance API has 7 day max per request
    # So we fetch in 7-day chunks
    print(f"\nFetching trades from last {days_back} days...")
    print(f"(Binance API limit: 7 days per request, fetching in chunks)")
    print()

    leverage = STRATEGY_CONFIG.get("leverage", 20)

    for symbol in FUTURES_SYMBOLS:
        try:
            print(f"Fetching {symbol}...", end=" ")

            # Fetch in 7-day chunks (Binance API limit)
            all_symbol_trades = []
            all_income_records = []

            chunk_days = 7
            num_chunks = (days_back + chunk_days - 1) // chunk_days  # Ceiling division

            for chunk in range(num_chunks):
                chunk_end = datetime.now() - timedelta(days=chunk * chunk_days)
                chunk_start = chunk_end - timedelta(days=chunk_days)

                # Don't go beyond days_back
                if chunk_start < datetime.now() - timedelta(days=days_back):
                    chunk_start = datetime.now() - timedelta(days=days_back)

                start_time = int(chunk_start.timestamp() * 1000)
                end_time = int(chunk_end.timestamp() * 1000)

                # Get account trades for this symbol in this chunk
                try:
                    trades = client._request(
                        "GET",
                        "/fapi/v1/userTrades",
                        {
                            "symbol": symbol,
                            "startTime": start_time,
                            "endTime": end_time,
                            "limit": 1000
                        },
                        signed=True
                    )
                    if trades and isinstance(trades, list):
                        all_symbol_trades.extend(trades)
                except:
                    pass

                # Get income history (realized PNL) for this symbol in this chunk
                try:
                    income_records = client._request(
                        "GET",
                        "/fapi/v1/income",
                        {
                            "symbol": symbol,
                            "incomeType": "REALIZED_PNL",
                            "startTime": start_time,
                            "endTime": end_time,
                            "limit": 1000
                        },
                        signed=True
                    )
                    if income_records and isinstance(income_records, list):
                        all_income_records.extend(income_records)
                except:
                    pass

            trades = all_symbol_trades
            income_records = all_income_records

            if not trades:
                print("No trades")
                continue

            # Create a map of time -> PNL
            pnl_map = {}
            if income_records and isinstance(income_records, list):
                for record in income_records:
                    time_key = record.get("time", 0)
                    pnl = float(record.get("income", 0))
                    pnl_map[time_key] = pnl

            # Group trades by position (entry/exit pairs)
            # Trades are ordered by time
            symbol_trades = []
            current_position = None

            for trade in trades:
                trade_time = trade.get("time", 0)
                side = trade.get("side", "")  # BUY or SELL
                price = float(trade.get("price", 0))
                qty = float(trade.get("qty", 0))
                realized_pnl = float(trade.get("realizedPnl", 0))
                commission = float(trade.get("commission", 0))
                position_side = trade.get("positionSide", "BOTH")

                # Check if this is a closing trade (has realized PNL)
                if abs(realized_pnl) > 0.001:
                    # This is a closing trade
                    trade_timestamp = datetime.fromtimestamp(trade_time / 1000).isoformat()

                    # Skip if already logged
                    if trade_timestamp[:19] in existing_timestamps:
                        continue

                    # Determine trade side (what was the position)
                    if side == "SELL":
                        pos_side = "LONG"  # Selling to close a long
                    else:
                        pos_side = "SHORT"  # Buying to close a short

                    # Determine exit type
                    if realized_pnl > 0:
                        exit_type = "TP"
                    else:
                        exit_type = "SL"

                    # Estimate DCA level from position size
                    position_value = price * qty
                    estimated_dca = estimate_dca_level_from_size(position_value, symbol_budget, leverage)

                    trade_record = {
                        "timestamp": trade_timestamp,
                        "symbol": symbol,
                        "side": pos_side,
                        "entry_price": 0,  # Not available from this API
                        "exit_price": price,
                        "quantity": qty,
                        "pnl": realized_pnl,
                        "pnl_pct": 0,  # Can't calculate without entry
                        "exit_type": exit_type,
                        "dca_level": estimated_dca,  # Estimated from position size
                        "leverage": leverage,
                        "testnet": testnet,
                        "commission": commission,
                        "binance_trade_id": trade.get("id", 0)
                    }

                    symbol_trades.append(trade_record)
                    total_pnl += realized_pnl

            if symbol_trades:
                print(f"{len(symbol_trades)} closed trades, PNL: ${sum(t['pnl'] for t in symbol_trades):+.2f}")
                all_new_trades.extend(symbol_trades)
            else:
                print("No closed trades")

        except Exception as e:
            print(f"Error: {e}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"New trades found: {len(all_new_trades)}")
    print(f"Total PNL from new trades: ${total_pnl:+.2f}")

    if all_new_trades:
        # Sort by timestamp
        all_new_trades.sort(key=lambda x: x["timestamp"])

        # Add to existing trades
        existing_trades.extend(all_new_trades)

        # Save to file
        with open(trade_log_path, 'w') as f:
            json.dump(existing_trades, f, indent=2)

        print(f"\nSaved {len(all_new_trades)} new trades to: {trade_log_path}")
        print(f"Total trades in log: {len(existing_trades)}")

        # Print trade details
        print("\n" + "=" * 70)
        print("NEW TRADES LOGGED")
        print("=" * 70)

        wins = 0
        losses = 0

        for trade in all_new_trades:
            pnl = trade["pnl"]
            if pnl > 0:
                wins += 1
                result = "WIN"
            else:
                losses += 1
                result = "LOSS"

            print(f"  {trade['timestamp'][:19]} | {trade['symbol']:10s} | {trade['side']:5s} | "
                  f"Exit: ${trade['exit_price']:,.4f} | PNL: ${pnl:+8.2f} | {result}")

        print()
        print(f"Wins: {wins} | Losses: {losses} | Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "No trades")
        print(f"Total PNL: ${total_pnl:+.2f}")
    else:
        print("\nNo new trades to log")

    # Also show overall stats from the log
    print("\n" + "=" * 70)
    print("OVERALL TRADE LOG STATS")
    print("=" * 70)

    if existing_trades:
        total_wins = sum(1 for t in existing_trades if t.get("pnl", 0) > 0)
        total_losses = sum(1 for t in existing_trades if t.get("pnl", 0) < 0)
        total_pnl_all = sum(t.get("pnl", 0) for t in existing_trades)

        print(f"Total Trades: {len(existing_trades)}")
        print(f"Wins: {total_wins} | Losses: {total_losses}")
        if total_wins + total_losses > 0:
            print(f"Win Rate: {total_wins/(total_wins+total_losses)*100:.1f}%")
        print(f"Total PNL: ${total_pnl_all:+.2f}")

        # By symbol
        print("\nBy Symbol:")
        symbol_stats = {}
        for trade in existing_trades:
            sym = trade.get("symbol", "UNKNOWN")
            if sym not in symbol_stats:
                symbol_stats[sym] = {"trades": 0, "pnl": 0.0, "wins": 0}
            symbol_stats[sym]["trades"] += 1
            symbol_stats[sym]["pnl"] += trade.get("pnl", 0)
            if trade.get("pnl", 0) > 0:
                symbol_stats[sym]["wins"] += 1

        for sym, stats in sorted(symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = (stats["wins"]/stats["trades"]*100) if stats["trades"] > 0 else 0
            print(f"  {sym:10s}: {stats['trades']:3d} trades | PNL: ${stats['pnl']:+8.2f} | WR: {wr:.0f}%")

    print("=" * 70)

    return all_new_trades


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch trade history from Binance")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    parser.add_argument("--live", action="store_true", help="Use mainnet instead of demo")

    args = parser.parse_args()

    fetch_and_log_trades(days_back=args.days, testnet=not args.live)
