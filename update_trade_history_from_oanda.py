"""
Update trade history from OANDA transaction data
Parse OANDA trade history and reconcile with our logs
"""

import json
from datetime import datetime
import pytz
import os

# OANDA trade history data (from user's paste)
oanda_history = [
    {"id": 175, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:50:08 AM"},
    {"id": 174, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:50:08 AM"},
    {"id": 173, "type": "Sell Market Filled", "instrument": "GBP/USD", "units": "10,992", "price": "1.33900", "pnl": "-1.0442", "pips": "-0.66", "commission": "-", "swap": "0.00", "balance": "4910.35", "time": "12/12/2025, 1:50:08 AM"},
    {"id": 172, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:45:26 AM"},
    {"id": 171, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:45:26 AM"},
    {"id": 170, "type": "Sell Market Filled", "instrument": "AUD/USD", "units": "22,098", "price": "0.66639", "pnl": "-1.5469", "pips": "6.41", "commission": "-", "swap": "0.00", "balance": "4911.01", "time": "12/12/2025, 1:45:26 AM"},
    {"id": 169, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:40:22 AM"},
    {"id": 168, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:40:22 AM"},
    {"id": 167, "type": "Buy Market Filled", "instrument": "USD/CAD", "units": "10,681", "price": "1.37757", "pnl": "-0.7366", "pips": "1.77", "commission": "-", "swap": "0.00", "balance": "4904.60", "time": "12/12/2025, 1:40:22 AM"},
    {"id": 166, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:36:54 AM"},
    {"id": 165, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:36:54 AM"},
    {"id": 164, "type": "Sell Market Filled", "instrument": "EUR/USD", "units": "12,545", "price": "1.17351", "pnl": "-1.0663", "pips": "2.51", "commission": "-", "swap": "0.00", "balance": "4902.83", "time": "12/12/2025, 1:36:54 AM"},
    {"id": 163, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:36:13 AM"},
    {"id": 162, "type": "Order Cancelled", "instrument": "-", "units": "-", "price": "-", "pnl": "-", "pips": "-", "commission": "-", "swap": "-", "balance": "-", "time": "12/12/2025, 1:36:13 AM"},
    {"id": 161, "type": "Buy Market Filled", "instrument": "USD/CHF", "units": "18,513", "price": "0.79517", "pnl": "-1.9792", "pips": "-5.85", "commission": "-", "swap": "0.00", "balance": "4900.32", "time": "12/12/2025, 1:36:13 AM"},
    {"id": 157, "type": "Buy Market", "instrument": "AUD/USD", "units": "22,098", "price": "0.66610", "pnl": "-1.3259", "pips": "0.00", "commission": "-", "swap": "0.00", "balance": "4906.17", "time": "12/12/2025, 12:00:36 AM"},
    {"id": 152, "type": "Sell Market", "instrument": "USD/CHF", "units": "18,513", "price": "0.79492", "pnl": "-2.2122", "pips": "0.00", "commission": "-", "swap": "0.00", "balance": "4906.17", "time": "12/12/2025, 12:00:34 AM"},
]

print("=" * 100)
print("UPDATING TRADE HISTORY FROM OANDA")
print("=" * 100)

# Parse trades and match entries/exits
trades = {}

for row in oanda_history:
    if row["type"] in ["Buy Market", "Sell Market"]:
        # Entry
        trade_id = str(row["id"])
        instrument = row["instrument"].replace("/", "_")
        direction = "LONG" if row["type"] == "Buy Market" else "SHORT"
        units = int(row["units"].replace(",", ""))
        entry_price = float(row["price"])
        entry_time = datetime.strptime(row["time"], "%m/%d/%Y, %I:%M:%S %p")
        entry_time = pytz.UTC.localize(entry_time)

        trades[trade_id] = {
            "trade_id": trade_id,
            "instrument": instrument,
            "direction": direction,
            "units": units,
            "entry_price": entry_price,
            "entry_time": entry_time.isoformat(),
            "status": "OPEN"
        }

        print(f"\n[ENTRY] Trade #{trade_id}: {instrument} {direction} {units:,} units @ {entry_price:.5f}")
        print(f"  Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    elif row["type"] in ["Buy Market Filled", "Sell Market Filled"]:
        # Exit
        trade_id = str(row["id"])

        if trade_id in trades:
            instrument = row["instrument"].replace("/", "_")
            exit_price = float(row["price"])
            pnl = float(row["pnl"])
            pips = float(row["pips"]) if row["pips"] != "-" else 0
            exit_time = datetime.strptime(row["time"], "%m/%d/%Y, %I:%M:%S %p")
            exit_time = pytz.UTC.localize(exit_time)

            trades[trade_id]["exit_price"] = exit_price
            trades[trade_id]["exit_time"] = exit_time.isoformat()
            trades[trade_id]["pnl"] = pnl
            trades[trade_id]["pips"] = pips
            trades[trade_id]["status"] = "CLOSED"

            # Determine exit reason (SL/TP/Manual)
            entry = trades[trade_id]
            direction = entry["direction"]
            entry_price = entry["entry_price"]

            # Calculate expected TP/SL (approximate based on pips)
            if "USD_JPY" in instrument:
                pip_value = 0.01
            else:
                pip_value = 0.0001

            # For now, mark as MANUAL since we don't have TP/SL from history
            exit_reason = "MANUAL"

            trades[trade_id]["exit_reason"] = exit_reason

            print(f"\n[EXIT] Trade #{trade_id}: {instrument} closed @ {exit_price:.5f}")
            print(f"  P&L: ${pnl:+.2f} ({pips:+.2f} pips)")
            print(f"  Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  Reason: {exit_reason}")

# Write to log file
log_dir = "forex_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "trades_20251211.jsonl")

print("\n" + "=" * 100)
print(f"Writing {len(trades)} trades to {log_file}")
print("=" * 100)

# Read existing logs to avoid duplicates
existing_trade_ids = set()
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("type") == "ENTRY":
                    existing_trade_ids.add(entry.get("trade_id"))
            except:
                pass

# Append new trades
with open(log_file, 'a') as f:
    for trade_id, trade in trades.items():
        if trade_id not in existing_trade_ids:
            # Write entry
            entry_log = {
                "type": "ENTRY",
                "timestamp": trade["entry_time"],
                "trade_id": trade_id,
                "instrument": trade["instrument"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "units": trade["units"],
                "note": "Imported from OANDA history"
            }
            f.write(json.dumps(entry_log) + "\n")
            print(f"  ✓ Logged ENTRY for trade #{trade_id}")

            # Write exit if closed
            if trade["status"] == "CLOSED":
                exit_log = {
                    "type": "EXIT",
                    "timestamp": trade["exit_time"],
                    "trade_id": trade_id,
                    "instrument": trade["instrument"],
                    "direction": trade["direction"],
                    "entry_price": trade["entry_price"],
                    "exit_price": trade["exit_price"],
                    "units": trade["units"],
                    "pl": trade["pnl"],
                    "pips": trade["pips"],
                    "exit_reason": trade["exit_reason"],
                    "note": "Imported from OANDA history"
                }
                f.write(json.dumps(exit_log) + "\n")
                print(f"  ✓ Logged EXIT for trade #{trade_id}")

# Summary
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

closed_trades = [t for t in trades.values() if t["status"] == "CLOSED"]
open_trades = [t for t in trades.values() if t["status"] == "OPEN"]

print(f"\nTotal Trades: {len(trades)}")
print(f"  Open: {len(open_trades)}")
print(f"  Closed: {len(closed_trades)}")

if closed_trades:
    total_pnl = sum(t["pnl"] for t in closed_trades)
    winners = [t for t in closed_trades if t["pnl"] > 0]
    losers = [t for t in closed_trades if t["pnl"] < 0]

    print(f"\nClosed Trades Performance:")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    print(f"  Winners: {len(winners)}")
    print(f"  Losers: {len(losers)}")
    print(f"  Win Rate: {len(winners)/len(closed_trades)*100:.1f}%")

if open_trades:
    print(f"\nOpen Trades:")
    for trade in open_trades:
        print(f"  #{trade['trade_id']}: {trade['instrument']} {trade['direction']} @ {trade['entry_price']:.5f}")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
