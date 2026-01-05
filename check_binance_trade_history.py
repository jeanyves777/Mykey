#!/usr/bin/env python3
"""
Verify Trade History from Binance - Check if reported profits are real
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trading_system", "Binance_Futures_Trading"))

from trading_system.Binance_Futures_Trading.engine.binance_client import BinanceClient
from datetime import datetime, timedelta

def check_real_trades():
    print("="*100)
    print("CHECKING BINANCE TRADE HISTORY - Verify Reported Profits")
    print("="*100)
    print()
    
    client = BinanceClient(testnet=False, use_demo=False)
    
    # Get trades from last 24 hours
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    for symbol in symbols:
        print(f"\n{'='*100}")
        print(f"{symbol} - RECENT TRADES")
        print(f"{'='*100}")
        
        try:
            # Get recent trades
            trades = client.client.futures_account_trades(symbol=symbol, limit=20)
            
            if not trades:
                print(f"  No recent trades for {symbol}")
                continue
            
            print(f"  Found {len(trades)} recent trades:")
            print()
            
            total_realized_pnl = 0
            
            for i, trade in enumerate(trades[-10:], 1):  # Last 10 trades
                trade_time = datetime.fromtimestamp(trade['time'] / 1000)
                side = trade['side']
                qty = float(trade['qty'])
                price = float(trade['price'])
                realized_pnl = float(trade['realizedPnl'])
                commission = float(trade['commission'])
                
                total_realized_pnl += realized_pnl
                
                pnl_symbol = "✅" if realized_pnl > 0 else "❌" if realized_pnl < 0 else "⚪"
                
                print(f"  {i}. {trade_time.strftime('%m-%d %H:%M:%S')} | {side:5} | Qty: {qty:10.6f} | Price: ${price:,.2f}")
                print(f"     Realized P&L: {pnl_symbol} ${realized_pnl:+.4f} | Commission: ${commission:.4f}")
                print()
            
            print(f"  Total Realized P&L (last 10 trades): ${total_realized_pnl:+.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Get current balance
    print(f"\n{'='*100}")
    print("CURRENT ACCOUNT STATUS")
    print(f"{'='*100}")
    
    try:
        account = client.client.futures_account()
        total_balance = float(account['totalWalletBalance'])
        available_balance = float(account['availableBalance'])
        total_unrealized_pnl = float(account['totalUnrealizedProfit'])
        
        print(f"  Wallet Balance:     ${total_balance:.2f}")
        print(f"  Available:          ${available_balance:.2f}")
        print(f"  Unrealized P&L:     ${total_unrealized_pnl:.2f}")
        print(f"  Total Equity:       ${total_balance + total_unrealized_pnl:.2f}")
        print()
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("="*100)

if __name__ == "__main__":
    check_real_trades()
