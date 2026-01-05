#!/usr/bin/env python3
"""
Live Trading Monitor - Extract key metrics from logs
"""
import re
import sys
from datetime import datetime
from collections import defaultdict

def parse_log_file(log_path):
    """Parse log file and extract key metrics"""
    
    metrics = {
        "balance": [],
        "total_pnl": [],
        "positions": [],
        "boost_activations": [],
        "tp_hits": [],
        "sl_hits": [],
        "trades_closed": [],
        "trailing_activations": [],
        "half_closes": [],
        "session_stats": {},
        "symbol_stats": defaultdict(dict),
        "critical_alerts": []
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # Balance updates
                if "Balance:" in line and "$" in line:
                    balance_match = re.search(r'Balance: \$([0-9,.]+)', line)
                    if balance_match:
                        balance = float(balance_match.group(1).replace(',', ''))
                        metrics["balance"].append((timestamp, balance))
                
                # Total Unrealized P&L
                if "TOTAL Unrealized P&L:" in line:
                    pnl_match = re.search(r'TOTAL Unrealized P&L: \$([+-]?[0-9,.]+)', line)
                    if pnl_match:
                        pnl = float(pnl_match.group(1).replace(',', ''))
                        metrics["total_pnl"].append((timestamp, pnl))
                
                # Boost activations
                if "[BOOST] {symbol}: ACTIVATED!" in line or "BOOST ROI" in line or "DUAL BOOST" in line:
                    metrics["boost_activations"].append((timestamp, line.strip()))
                
                # TP hits
                if "TP HIT" in line or "hit TP" in line:
                    metrics["tp_hits"].append((timestamp, line.strip()))
                
                # SL hits
                if "SL HIT" in line or "STOP_MARKET" in line and "filled" in line:
                    metrics["sl_hits"].append((timestamp, line.strip()))
                
                # Trade closures
                if "Position" in line and "CLOSED" in line:
                    metrics["trades_closed"].append((timestamp, line.strip()))
                
                # Trailing TP
                if "TRAILING TP" in line:
                    metrics["trailing_activations"].append((timestamp, line.strip()))
                
                # Half-closes
                if "BOOST HALF-CLOSE" in line or "Closing 50%" in line:
                    metrics["half_closes"].append((timestamp, line.strip()))
                
                # Session stats
                if "Trades Today:" in line:
                    stats_match = re.search(r'Trades Today: (\d+) \(W:(\d+) / L:(\d+)\)', line)
                    if stats_match:
                        metrics["session_stats"]["trades"] = int(stats_match.group(1))
                        metrics["session_stats"]["wins"] = int(stats_match.group(2))
                        metrics["session_stats"]["losses"] = int(stats_match.group(3))
                
                if "Daily P&L:" in line:
                    pnl_match = re.search(r'Daily P&L: \$([+-]?[0-9,.]+)', line)
                    if pnl_match:
                        metrics["session_stats"]["daily_pnl"] = float(pnl_match.group(1).replace(',', ''))
                
                # Per-symbol stats
                if re.search(r'(BTCUSDT|ETHUSDT|BNBUSDT): W:\d+', line):
                    symbol_match = re.search(r'(BTCUSDT|ETHUSDT|BNBUSDT): W:(\d+)/L:(\d+) \(([0-9.]+)%\).*P&L: \$([+-]?[0-9,.]+)', line)
                    if symbol_match:
                        symbol = symbol_match.group(1)
                        metrics["symbol_stats"][symbol] = {
                            "wins": int(symbol_match.group(2)),
                            "losses": int(symbol_match.group(3)),
                            "win_rate": float(symbol_match.group(4)),
                            "pnl": float(symbol_match.group(5).replace(',', ''))
                        }
                
                # Critical alerts
                if any(keyword in line for keyword in ["ERROR", "CRITICAL", "liquidation", "margin call"]):
                    if "ERROR" in line and "No need to change" not in line:
                        metrics["critical_alerts"].append((timestamp, line.strip()))
                
                # Position ROI danger
                if "ROI" in line and "%" in line:
                    roi_match = re.search(r'([A-Z]+USDT).*ROI.*?([+-]?\d+\.\d+)%', line)
                    if roi_match:
                        roi = float(roi_match.group(2))
                        if roi < -50:  # Danger zone
                            metrics["critical_alerts"].append((timestamp, f"⚠️  {roi_match.group(1)} at {roi}% ROI - DANGER!"))
    
    except Exception as e:
        print(f"Error reading log: {e}")
    
    return metrics

def display_metrics(metrics):
    """Display formatted metrics"""
    
    print("\n" + "="*80)
    print("📊 LIVE TRADING METRICS DASHBOARD")
    print("="*80)
    
    # Latest Balance
    if metrics["balance"]:
        latest_balance = metrics["balance"][-1]
        print(f"\n💰 CURRENT BALANCE: ${latest_balance[1]:.2f} (as of {latest_balance[0]})")
    
    # Latest P&L
    if metrics["total_pnl"]:
        latest_pnl = metrics["total_pnl"][-1]
        pnl_color = "🟢" if latest_pnl[1] >= 0 else "🔴"
        print(f"{pnl_color} UNREALIZED P&L: ${latest_pnl[1]:+.2f}")
    
    # Session Stats
    if metrics["session_stats"]:
        stats = metrics["session_stats"]
        win_rate = (stats.get("wins", 0) / stats.get("trades", 1) * 100) if stats.get("trades", 0) > 0 else 0
        print(f"\n📈 SESSION STATS:")
        print(f"   Trades: {stats.get('trades', 0)} (W:{stats.get('wins', 0)} / L:{stats.get('losses', 0)})")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Daily P&L: ${stats.get('daily_pnl', 0):+.2f}")
    
    # Per-Symbol Stats
    if metrics["symbol_stats"]:
        print(f"\n📊 PER-SYMBOL PERFORMANCE:")
        for symbol, stats in metrics["symbol_stats"].items():
            pnl_color = "🟢" if stats["pnl"] >= 0 else "🔴"
            print(f"   {symbol}: W:{stats['wins']}/L:{stats['losses']} ({stats['win_rate']:.0f}%) | {pnl_color} P&L: ${stats['pnl']:+.2f}")
    
    # Boost Activations
    if metrics["boost_activations"]:
        print(f"\n🚀 BOOST MODE ACTIVATIONS ({len(metrics['boost_activations'])} total):")
        for timestamp, event in metrics["boost_activations"][-5:]:  # Last 5
            print(f"   [{timestamp}] {event}")
    
    # Trailing TP
    if metrics["trailing_activations"]:
        print(f"\n📉 TRAILING TP EVENTS ({len(metrics['trailing_activations'])} total):")
        for timestamp, event in metrics["trailing_activations"][-5:]:
            print(f"   [{timestamp}] {event}")
    
    # Half-closes
    if metrics["half_closes"]:
        print(f"\n✂️  HALF-CLOSE CYCLES ({len(metrics['half_closes'])} total):")
        for timestamp, event in metrics["half_closes"][-5:]:
            print(f"   [{timestamp}] {event}")
    
    # Trade Closures
    if metrics["trades_closed"]:
        print(f"\n🔒 POSITIONS CLOSED ({len(metrics['trades_closed'])} total):")
        for timestamp, event in metrics["trades_closed"][-5:]:
            print(f"   [{timestamp}] {event}")
    
    # TP/SL Hits
    tp_count = len(metrics["tp_hits"])
    sl_count = len(metrics["sl_hits"])
    print(f"\n🎯 TP HITS: {tp_count} | ⛔ SL HITS: {sl_count}")
    
    # Critical Alerts
    if metrics["critical_alerts"]:
        print(f"\n⚠️  CRITICAL ALERTS ({len(metrics['critical_alerts'])} total):")
        for timestamp, alert in metrics["critical_alerts"][-10:]:  # Last 10
            print(f"   [{timestamp}] {alert}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/root/live_trading.log"
    
    print(f"📡 Monitoring: {log_path}")
    print("=" * 80)
    
    metrics = parse_log_file(log_path)
    display_metrics(metrics)
    
    print(f"\n✅ Monitoring complete. Run again to see updates.")
    print(f"💡 Tip: Use 'watch -n 60 python3 monitor_live_trading.py' for auto-refresh")
