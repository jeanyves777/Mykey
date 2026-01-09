#!/usr/bin/env python3
"""
Fix XRPUSDT stats - manually update the PnL to the correct value.
The position was closed with actual loss of $1.23 but stats show $-0.07
"""

import json
import os
import sys

# Path to stats file - check both local and VPS paths
if os.path.exists("/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/session_stats.json"):
    stats_file = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/session_stats.json"
else:
    stats_file = "/workspaces/Mykey/trading_system/Binance_Futures_Trading/engine/session_stats.json"

# Load current stats
if os.path.exists(stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("Current XRPUSDT stats:")
    print(f"  Wins: {stats['symbol_stats']['XRPUSDT']['wins']}")
    print(f"  Losses: {stats['symbol_stats']['XRPUSDT']['losses']}")
    print(f"  PnL: ${stats['symbol_stats']['XRPUSDT']['pnl']:.2f}")
    
    # Fix the PnL
    old_pnl = stats['symbol_stats']['XRPUSDT']['pnl']
    actual_loss = -1.23  # Actual loss from balance change
    
    # Update XRPUSDT stats
    stats['symbol_stats']['XRPUSDT']['pnl'] = actual_loss
    
    # Update today's total PnL (remove old, add correct)
    stats['pnl_today'] = stats['pnl_today'] - old_pnl + actual_loss
    
    # Save updated stats
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ Updated XRPUSDT stats:")
    print(f"  Wins: {stats['symbol_stats']['XRPUSDT']['wins']}")
    print(f"  Losses: {stats['symbol_stats']['XRPUSDT']['losses']}")
    print(f"  PnL: ${stats['symbol_stats']['XRPUSDT']['pnl']:.2f}")
    print(f"\n  Total PnL Today: ${stats['pnl_today']:.2f}")
    
else:
    print(f"❌ Stats file not found: {stats_file}")
    print("   Make sure the engine has been run at least once.")
