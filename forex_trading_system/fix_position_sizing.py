#!/usr/bin/env python3
"""
Fix position sizing to implement proper 2% risk management
"""
import sys
import time
import requests
sys.path.append(".")
from engine.oanda_client import OANDAClient

def calculate_proper_position_size(instrument, balance, stop_loss_pips=10):
    """Calculate proper position size for 2% risk"""
    risk_amount = balance * 0.02  # 2% of balance
    
    if "JPY" in instrument:
        # For JPY pairs: 1 pip = 0.01
        risk_per_pip = 0.01
    else:
        # For major pairs: 1 pip = 0.0001  
        risk_per_pip = 0.0001
    
    # Calculate units for given stop loss
    risk_per_unit = risk_per_pip * stop_loss_pips
    units = int(risk_amount / risk_per_unit)
    
    return min(units, 5000)  # Cap at 5000 units max

def close_trade_by_id(client, trade_id):
    """Close a specific trade by ID"""
    try:
        url = f"{client.base_url}/accounts/{client.account_id}/trades/{trade_id}/close"
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.put(url, headers=headers)
        return response.status_code == 200
    except Exception as e:
        print(f"Error closing trade {trade_id}: {e}")
        return False

def main():
    print("=== POSITION SIZING FIX ===")
    
    client = OANDAClient()
    
    # Get account info
    account = client.get_account_summary()
    balance = float(account.get("balance", 0))
    margin_used = float(account.get("marginUsed", 0))
    
    print(f"Balance: ${balance:.2f}")
    print(f"Current margin used: ${margin_used:.2f}")
    print(f"Target 2% risk per trade: ${balance * 0.02:.2f}")
    
    # Get current trades
    trades = client.get_open_trades()
    print(f"\nCurrent trades: {len(trades)}")
    
    trades_to_close = []
    trades_to_keep = []
    
    for trade in trades:
        tid = trade["id"]
        instrument = trade["instrument"]
        units = trade["units"]
        pl = trade["unrealized_pl"]
        
        # Calculate proper size
        proper_size = calculate_proper_position_size(instrument, balance)
        
        print(f"Trade {tid}: {instrument} {units:,.0f} units, PL: ${pl:.2f}")
        print(f"  Proper size: {proper_size:,.0f} units")
        
        # If position is grossly oversized (>10x proper), close it
        if abs(units) > proper_size * 10:
            print(f"  -> CLOSE (oversized)")
            trades_to_close.append(trade)
        # If position is way too small (USD_JPY with 100 units), close it too
        elif instrument == "USD_JPY" and abs(units) < proper_size * 0.1:
            print(f"  -> CLOSE (too small)")
            trades_to_close.append(trade)
        else:
            print(f"  -> KEEP (reasonable size)")
            trades_to_keep.append(trade)
    
    # Close oversized/undersized trades
    print(f"\nClosing {len(trades_to_close)} problematic trades...")
    for trade in trades_to_close:
        tid = trade["id"]
        instrument = trade["instrument"]
        units = trade["units"]
        
        print(f"Closing trade {tid}: {instrument} {units:,.0f} units")
        success = close_trade_by_id(client, tid)
        print(f"  Result: {'SUCCESS' if success else 'FAILED'}")
        time.sleep(1)
    
    # Wait for trades to close
    print("\nWaiting for trades to close...")
    time.sleep(5)
    
    # Check new margin usage
    account = client.get_account_summary()
    new_margin = float(account.get("marginUsed", 0))
    print(f"New margin used: ${new_margin:.2f}")
    
    # Open new properly sized positions for key pairs
    symbols_to_trade = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
    
    print(f"\nOpening new properly sized positions...")
    for symbol in symbols_to_trade:
        proper_size = calculate_proper_position_size(symbol, balance)
        
        # Get current price to set stops and targets
        try:
            current_price = client.get_current_price(symbol)
            if not current_price:
                continue
                
            price = current_price["ask"]  # Use ask for long positions
            
            # Set stop loss and take profit (10 pips each)
            if "JPY" in symbol:
                sl_distance = 0.10  # 10 pips for JPY
                tp_distance = 0.12  # 12 pips for JPY
            else:
                sl_distance = 0.0010  # 10 pips for majors
                tp_distance = 0.0012  # 12 pips for majors
            
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
            
            print(f"Opening {symbol}: {proper_size:,.0f} units")
            print(f"  Price: {price}, SL: {stop_loss}, TP: {take_profit}")
            
            # Place the order
            result = client.place_market_order(
                instrument=symbol,
                units=proper_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result:
                print(f"  SUCCESS: Opened {symbol}")
            else:
                print(f"  FAILED: Could not open {symbol}")
                
            time.sleep(2)
            
        except Exception as e:
            print(f"Error opening {symbol}: {e}")
    
    # Final account check
    print("\n=== FINAL STATUS ===")
    account = client.get_account_summary()
    final_balance = float(account.get("balance", 0))
    final_margin = float(account.get("marginUsed", 0))
    final_nav = float(account.get("nav", 0))
    
    print(f"Balance: ${final_balance:.2f}")
    print(f"Margin used: ${final_margin:.2f}")
    print(f"NAV: ${final_nav:.2f}")
    print(f"Margin utilization: {(final_margin/final_balance)*100:.1f}%")
    
    # Show new positions
    new_positions = client.get_open_positions()
    print(f"\nNew positions: {len(new_positions)}")
    for pos in new_positions:
        print(f"  {pos['instrument']}: {pos['side']} {pos['units']:,.0f} units")

if __name__ == "__main__":
    main()