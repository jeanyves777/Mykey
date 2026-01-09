#!/usr/bin/env python3
"""
Fix missing trades in session stats based on actual Binance data.
"""

import json
import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system/Binance_Futures_Trading'))

from engine.binance_client import BinanceClient
from datetime import datetime, timedelta

def main():
    print("\n" + "="*70)
    print("FIXING MISSING TRADES IN SESSION STATS")
    print("="*70)
    
    # Path to stats file
    stats_file = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/session_stats.json"
    if not os.path.exists(stats_file):
        stats_file = "/workspaces/Mykey/trading_system/Binance_Futures_Trading/engine/session_stats.json"
    
    if not os.path.exists(stats_file):
        print(f"❌ Stats file not found")
        return
    
    # Load current stats
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("\n📋 Current Stats:")
    print(f"   Total Trades: {stats.get('trades_today', 0)}")
    print(f"   Total PnL: ${stats.get('pnl_today', 0):.2f}")
    
    # Initialize client
    client = BinanceClient(testnet=False, use_demo=False)
    
    # Get actual PnL from Binance for all symbols
    all_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOTUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
    
    print("\n🔍 Getting actual PnL from Binance...")
    
    actual_data = {}
    total_binance_pnl = 0
    total_binance_trades = 0
    
    for symbol in all_symbols:
        try:
            # Get income history
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
                    wins = sum(1 for t in recent_trades if float(t.get("income", 0)) > 0)
                    losses = len(recent_trades) - wins
                    
                    actual_data[symbol] = {
                        "pnl": symbol_pnl,
                        "wins": wins,
                        "losses": losses,
                        "trades": len(recent_trades)
                    }
                    
                    total_binance_pnl += symbol_pnl
                    total_binance_trades += len(recent_trades)
                    
                    print(f"   {symbol}: {wins}W/{losses}L | PnL: ${symbol_pnl:+.2f}")
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    # Update stats with actual Binance data
    print("\n✅ Updating stats with actual Binance data...")
    
    # Ensure all symbols exist in symbol_stats
    if "symbol_stats" not in stats:
        stats["symbol_stats"] = {}
    
    for symbol in all_symbols:
        if symbol not in stats["symbol_stats"]:
            stats["symbol_stats"][symbol] = {"wins": 0, "losses": 0, "pnl": 0}
        
        # Update with actual data if available
        if symbol in actual_data:
            stats["symbol_stats"][symbol] = actual_data[symbol]
    
    # Update totals
    stats["trades_today"] = total_binance_trades
    stats["pnl_today"] = total_binance_pnl
    stats["wins_today"] = sum(s.get("wins", 0) for s in stats["symbol_stats"].values())
    stats["losses_today"] = sum(s.get("losses", 0) for s in stats["symbol_stats"].values())
    
    # Save updated stats
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ Stats Updated!")
    print(f"   Total Trades: {stats['trades_today']}")
    print(f"   Wins/Losses: {stats['wins_today']}W/{stats['losses_today']}L")
    print(f"   Win Rate: {stats['wins_today'] / stats['trades_today'] * 100:.1f}%")
    print(f"   Total PnL: ${stats['pnl_today']:+.2f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
