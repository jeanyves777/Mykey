"""
Parse OANDA closed trades from the filled market orders
"""

import json
from datetime import datetime
import pytz
import os

print("=" * 100)
print("PARSING OANDA CLOSED TRADES")
print("=" * 100)

# These are the CLOSED trades (exits)
closed_trades = [
    {
        "id": "173",
        "instrument": "GBP_USD",
        "direction": "SHORT",  # Sell Market Filled = closed a LONG position
        "units": 10992,
        "exit_price": 1.33900,
        "pnl": -1.0442,
        "pips": -0.66,
        "time": "12/12/2025, 1:50:08 AM",
        "exit_reason": "MANUAL"  # Since it shows as "Sell Market Filled" not TP/SL
    },
    {
        "id": "170",
        "instrument": "AUD_USD",
        "direction": "LONG",  # Sell Market Filled for a SHORT position
        "units": 22098,
        "exit_price": 0.66639,
        "pnl": -1.5469,
        "pips": 6.41,  # Wait, positive pips but negative P&L? This was closing a LONG
        "time": "12/12/2025, 1:45:26 AM",
        "exit_reason": "MANUAL"
    },
    {
        "id": "167",
        "instrument": "USD_CAD",
        "direction": "LONG",  # Buy Market Filled = closed a SHORT
        "units": 10681,
        "exit_price": 1.37757,
        "pnl": -0.7366,
        "pips": 1.77,
        "time": "12/12/2025, 1:40:22 AM",
        "exit_reason": "MANUAL"
    },
    {
        "id": "164",
        "instrument": "EUR_USD",
        "direction": "SHORT",  # Sell Market Filled = closed a LONG
        "units": 12545,
        "exit_price": 1.17351,
        "pnl": -1.0663,
        "pips": 2.51,
        "time": "12/12/2025, 1:36:54 AM",
        "exit_reason": "MANUAL"
    },
    {
        "id": "161",
        "instrument": "USD_CHF",
        "direction": "LONG",  # Buy Market Filled = closed a SHORT
        "units": 18513,
        "exit_price": 0.79517,
        "pnl": -1.9792,
        "pips": -5.85,
        "time": "12/12/2025, 1:36:13 AM",
        "exit_reason": "MANUAL"
    }
]

# Current open positions (from earlier in the conversation)
open_positions = [
    {"id": "157", "instrument": "AUD_USD", "direction": "LONG", "units": 22098, "entry_price": 0.66610},
    {"id": "152", "instrument": "USD_CHF", "direction": "SHORT", "units": 18513, "entry_price": 0.79492}
]

print(f"\nFound {len(closed_trades)} closed trades")
print(f"Found {len(open_positions)} open positions\n")

# Write exits to log
log_dir = "forex_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "trades_20251211.jsonl")

with open(log_file, 'a') as f:
    for trade in closed_trades:
        exit_time = datetime.strptime(trade["time"], "%m/%d/%Y, %I:%M:%S %p")
        exit_time = pytz.UTC.localize(exit_time)

        exit_log = {
            "type": "EXIT",
            "timestamp": exit_time.isoformat(),
            "trade_id": trade["id"],
            "instrument": trade["instrument"],
            "direction": trade["direction"],
            "exit_price": trade["exit_price"],
            "units": trade["units"],
            "pl": trade["pnl"],
            "pips": trade["pips"],
            "exit_reason": trade["exit_reason"],
            "note": "Manually closed - imported from OANDA history"
        }

        f.write(json.dumps(exit_log) + "\n")

        print(f"[EXIT] Trade #{trade['id']}: {trade['instrument']} {trade['direction']}")
        print(f"  Exit: {trade['exit_price']:.5f}")
        print(f"  P&L: ${trade['pnl']:+.2f} ({trade['pips']:+.1f} pips)")
        print(f"  Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Reason: {trade['exit_reason']}\n")

# Summary
print("=" * 100)
print("SUMMARY")
print("=" * 100)

total_pnl = sum(t["pnl"] for t in closed_trades)
total_pips = sum(t["pips"] for t in closed_trades)
winners = [t for t in closed_trades if t["pnl"] > 0]
losers = [t for t in closed_trades if t["pnl"] < 0]

print(f"\nClosed Trades: {len(closed_trades)}")
print(f"  Winners: {len(winners)}")
print(f"  Losers: {len(losers)}")
print(f"  Win Rate: {len(winners)/len(closed_trades)*100:.1f}%" if closed_trades else "  Win Rate: N/A")
print(f"\nTotal P&L: ${total_pnl:+.2f}")
print(f"Total Pips: {total_pips:+.1f}")

print(f"\nOpen Positions: {len(open_positions)}")
for pos in open_positions:
    print(f"  #{pos['id']}: {pos['instrument']} {pos['direction']} @ {pos['entry_price']:.5f}")

print("\n" + "=" * 100)
print(f"Logs updated in: {log_file}")
print("=" * 100)
