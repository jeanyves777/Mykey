#!/usr/bin/env python3
"""
Verify Trade History - Check if reported profits are real
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trading_system", "Binance_Futures_Trading"))

from trading_system.Binance_Futures_Trading.engine.binance_client import BinanceClient
from datetime import datetime, timedelta

print("="*100)
print("VERIFYING BINANCE TRADE HISTORY - Last 24 Hours")
print("="*100)

client = BinanceClient(testnet=False, use_demo=False)

# Get income history for all symbols
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

total_realized = 0.0
trade_count = 0

for symbol in symbols:
    print(f"\n{'='*100}")
    print(f"SYMBOL: {symbol}")
    print(f"{'='*100}")
    
    try:
        # Get realized PNL records (last 50)
        income = client.get_income_history(symbol, income_type="REALIZED_PNL", limit=50)
        
        if not income:
            print(f"  No realized PNL records found")
            continue
        
        # Filter last 24 hours
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_24h_ms = int(last_24h.timestamp() * 1000)
        
        symbol_pnl = 0.0
        symbol_trades = 0
        
        print(f"\n  {'Time':<20} | {'PNL':>12} | {'Trade ID':<20}")
        print(f"  {'-'*60}")
        
        for record in income:
            record_time = int(record.get("time", 0))
            
            if record_time < last_24h_ms:
                continue  # Skip older than 24h
            
            pnl = float(record.get("income", 0))
            trade_id = record.get("tranId", "")
            timestamp = datetime.fromtimestamp(record_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
            
            symbol_pnl += pnl
            symbol_trades += 1
            total_realized += pnl
            trade_count += 1
            
            status = "✅ WIN" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ BE"
            print(f"  {timestamp:<20} | ${pnl:>10.4f} | {trade_id:<20} | {status}")
        
        print(f"  {'-'*60}")
        print(f"  SYMBOL TOTAL: ${symbol_pnl:+.4f} ({symbol_trades} trades)")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*100}")
print(f"SUMMARY - Last 24 Hours")
print(f"{'='*100}")
print(f"  Total Trades:     {trade_count}")
print(f"  Total Realized:   ${total_realized:+.4f}")
print(f"  Win/Loss Ratio:   {sum(1 for s in symbols for r in client.get_income_history(s, 'REALIZED_PNL', 50) if float(r.get('income',0)) > 0)}/{sum(1 for s in symbols for r in client.get_income_history(s, 'REALIZED_PNL', 50) if float(r.get('income',0)) < 0)}")
print(f"{'='*100}")

# Compare with balance
try:
    account = client.get_account_info()
    balance = float(account.get("totalWalletBalance", 0))
    unrealized = float(account.get("totalUnrealizedProfit", 0))
    equity = balance + unrealized
    
    print(f"\nCURRENT ACCOUNT STATE:")
    print(f"  Wallet Balance:    ${balance:.2f}")
    print(f"  Unrealized P&L:    ${unrealized:+.2f}")
    print(f"  Total Equity:      ${equity:.2f}")
    print(f"{'='*100}")
    
except Exception as e:
    print(f"\nERROR getting account: {e}")

print("\n⚡ CHECK: Does the system's reported Daily P&L match Binance's actual realized PNL?")
print("="*100)
