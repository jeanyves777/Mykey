"""
Reconstruct missing trade logs from OANDA history
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
import json
from datetime import datetime
import pytz

print("=" * 100)
print("RECONSTRUCTING MISSING TRADE LOGS FROM OANDA")
print("=" * 100)

# Initialize OANDA client
client = OandaClient()

# Get full trade history
print("\n[1/3] Fetching complete trade history from OANDA...")
oanda_history = client.get_trade_history(count=100)
print(f"      Found {len(oanda_history)} trades")

# Read existing local logs
print("\n[2/3] Reading existing local logs...")
log_files = [
    "forex_logs/trades_20251211.jsonl",
    "forex_logs/trades_20251212.jsonl"
]

existing_trade_ids = set()

for log_file in log_files:
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('trade_id'):
                        existing_trade_ids.add(data['trade_id'])
                except:
                    pass

print(f"      Found {len(existing_trade_ids)} trade IDs in local logs")

# Identify missing trades
print("\n[3/3] Identifying and reconstructing missing trades...")

missing_trades = []
for trade in oanda_history:
    trade_id = trade.get('id')
    if trade_id and trade_id not in existing_trade_ids:
        missing_trades.append(trade)

print(f"      Found {len(missing_trades)} missing trades")

if len(missing_trades) == 0:
    print("\n[COMPLETE] All trades are already logged!")
    exit(0)

# Create reconstruction log file
reconstruction_file = "forex_logs/reconstructed_trades.jsonl"

print(f"\n      Writing reconstructed trades to: {reconstruction_file}")

with open(reconstruction_file, 'w') as f:
    for trade in missing_trades:
        trade_id = trade.get('id')
        instrument = trade.get('instrument', 'UNKNOWN')
        units = trade.get('initial_units', 0)
        entry_price = trade.get('price', 0)
        realized_pl = float(trade.get('realized_pl', 0))

        # Determine direction
        direction = "BUY" if units > 0 else "SELL"

        # Parse time - ensure it's a string
        open_time_raw = trade.get('open_time')
        close_time_raw = trade.get('close_time')

        if isinstance(open_time_raw, str):
            open_time = open_time_raw
        else:
            open_time = datetime.now(pytz.UTC).isoformat()

        if isinstance(close_time_raw, str):
            close_time = close_time_raw
        else:
            close_time = datetime.now(pytz.UTC).isoformat()

        # Calculate exit price from P&L
        # P&L = (exit_price - entry_price) * units (for BUY)
        # P&L = (entry_price - exit_price) * abs(units) (for SELL)
        if abs(units) > 0:
            if units > 0:  # BUY
                exit_price = entry_price + (realized_pl / units)
            else:  # SELL
                exit_price = entry_price - (realized_pl / abs(units))
        else:
            exit_price = entry_price

        # Determine exit reason
        exit_reason = "UNKNOWN"
        if realized_pl > 0:
            exit_reason = "TP"
        elif realized_pl < 0:
            exit_reason = "SL"
        else:
            exit_reason = "BE"

        # Calculate pips
        pip_multiplier = 100 if 'JPY' in instrument else 10000
        pips = abs(exit_price - entry_price) * pip_multiplier

        # Create ENTRY log
        entry_log = {
            "type": "ENTRY",
            "timestamp": open_time,
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "units": abs(units),
            "trade_id": trade_id,
            "reconstructed": True,
            "note": "Reconstructed from OANDA history"
        }

        # Create EXIT log
        exit_log = {
            "type": "EXIT",
            "timestamp": close_time,
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": open_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "trade_id": trade_id,
            "pnl": realized_pl,
            "pips": pips,
            "won": realized_pl > 0,
            "reconstructed": True,
            "note": "Reconstructed from OANDA history"
        }

        # Write both to file
        f.write(json.dumps(entry_log) + '\n')
        f.write(json.dumps(exit_log) + '\n')

print(f"\n[SUCCESS] Reconstructed {len(missing_trades)} missing trades")

# Print summary of reconstructed trades
print("\n" + "=" * 100)
print("RECONSTRUCTED TRADES SUMMARY")
print("=" * 100)

total_pl = sum(float(t.get('realized_pl', 0)) for t in missing_trades)
winners = len([t for t in missing_trades if float(t.get('realized_pl', 0)) > 0])
losers = len([t for t in missing_trades if float(t.get('realized_pl', 0)) < 0])

print(f"\nTotal Reconstructed: {len(missing_trades)} trades")
print(f"Winners: {winners}")
print(f"Losers: {losers}")
print(f"Total P&L: ${total_pl:+.2f}")

print("\nReconstructed Trades:")
for i, trade in enumerate(missing_trades, 1):
    trade_id = trade.get('id')
    instrument = trade.get('instrument')
    pl = float(trade.get('realized_pl', 0))
    status = "WIN" if pl > 0 else "LOSS" if pl < 0 else "BE"
    print(f"  {i}. Trade #{trade_id}: {instrument} ${pl:+.2f} [{status}]")

print("\n" + "=" * 100)
print("NEXT STEPS:")
print("=" * 100)
print(f"\n1. Review reconstructed trades in: {reconstruction_file}")
print(f"2. All future trades will be logged automatically by the live trading system")
print(f"3. Run verify_and_sync_logs.py again to confirm all trades are now logged")

print("\n" + "=" * 100)
