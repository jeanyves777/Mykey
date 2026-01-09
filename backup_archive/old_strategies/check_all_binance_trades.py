#!/usr/bin/env python3
"""
Check all recent trades directly from Binance to find any missing from stats.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system/Binance_Futures_Trading'))

from engine.binance_client import BinanceClient

def main():
    print("\n" + "="*70)
    print("CHECKING ALL BINANCE TRADES (Last 24 Hours)")
    print("="*70)
    
    # Initialize client
    client = BinanceClient(testnet=False, use_demo=False)
    
    # Get account info
    account_info = client.get_account_info()
    balance = float(account_info.get("totalWalletBalance", 0))
    print(f"\n💰 Current Balance: ${balance:.2f} USDT")
    
    # All potential symbols
    all_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOTUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
    
    print(f"\n📊 Checking trades for: {', '.join(all_symbols)}\n")
    
    total_pnl = 0
    total_trades = 0
    
    for symbol in all_symbols:
        try:
            # Get income history (realized PnL)
            income_history = client.get_income_history(
                symbol=symbol,
                income_type="REALIZED_PNL",
                limit=50
            )
            
            if income_history:
                # Filter last 24 hours
                now = datetime.now()
                yesterday = now - timedelta(hours=24)
                yesterday_ts = int(yesterday.timestamp() * 1000)
                
                recent_trades = [
                    inc for inc in income_history 
                    if inc.get("time", 0) >= yesterday_ts
                ]
                
                if recent_trades:
                    symbol_pnl = sum(float(t.get("income", 0)) for t in recent_trades)
                    total_pnl += symbol_pnl
                    total_trades += len(recent_trades)
                    
                    print(f"✅ {symbol}:")
                    print(f"   Trades: {len(recent_trades)}")
                    print(f"   Total PnL: ${symbol_pnl:+.2f}")
                    
                    # Show each trade
                    for i, trade in enumerate(recent_trades, 1):
                        pnl = float(trade.get("income", 0))
                        timestamp = trade.get("time", 0)
                        dt = datetime.fromtimestamp(timestamp / 1000)
                        print(f"     {i}. {dt.strftime('%Y-%m-%d %H:%M:%S')} | PnL: ${pnl:+.2f}")
                    print()
                    
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    print("="*70)
    print(f"📈 TOTAL: {total_trades} trades | Total PnL: ${total_pnl:+.2f}")
    print("="*70)
    
    # Now check what's in session stats
    print("\n" + "="*70)
    print("COMPARING WITH SESSION STATS")
    print("="*70)
    
    import json
    stats_file = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/session_stats.json"
    if not os.path.exists(stats_file):
        stats_file = "/workspaces/Mykey/trading_system/Binance_Futures_Trading/engine/session_stats.json"
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n📋 Session Stats:")
        print(f"   Total Trades: {stats.get('trades_today', 0)}")
        print(f"   Total PnL: ${stats.get('pnl_today', 0):.2f}")
        print(f"\n   By Symbol:")
        
        for symbol in all_symbols:
            if symbol in stats.get('symbol_stats', {}):
                sym_stats = stats['symbol_stats'][symbol]
                wins = sym_stats.get('wins', 0)
                losses = sym_stats.get('losses', 0)
                pnl = sym_stats.get('pnl', 0)
                print(f"   {symbol}: {wins}W/{losses}L | PnL: ${pnl:+.2f}")
        
        # Check for discrepancy
        stats_total_pnl = stats.get('pnl_today', 0)
        if abs(total_pnl - stats_total_pnl) > 0.01:
            print(f"\n⚠️  DISCREPANCY FOUND!")
            print(f"   Binance PnL: ${total_pnl:+.2f}")
            print(f"   Stats PnL:   ${stats_total_pnl:+.2f}")
            print(f"   Difference:  ${total_pnl - stats_total_pnl:+.2f}")
    else:
        print(f"\n❌ Stats file not found: {stats_file}")

if __name__ == "__main__":
    main()
