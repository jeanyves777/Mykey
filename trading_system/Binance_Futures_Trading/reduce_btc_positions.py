"""
Reduce BTCUSDT position sizes to correct $5 margin each
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient

def main():
    # Initialize client (LIVE mode - not testnet)
    client = BinanceClient(testnet=False)
    
    symbol = "BTCUSDT"
    
    # Get current positions (using the method that works)
    positions = client.get_positions()
    
    print(f"\n{'='*60}")
    print(f"Current {symbol} Positions:")
    print(f"{'='*60}")
    
    long_pos = None
    short_pos = None
    
    for pos in positions:
        if pos.get('symbol') == symbol:
            qty = pos.get('quantity', 0)
            side = pos.get('side', '')
            entry = pos.get('entry_price', 0)
            margin = pos.get('isolated_wallet', 0)
            
            if side == "LONG":
                long_pos = pos
            else:
                short_pos = pos
                
            print(f"\n{side}:")
            print(f"  Current Qty: {qty}")
            print(f"  Entry Price: ${entry:,.2f}")
            print(f"  Position Value: ${qty * entry:,.2f}")
            print(f"  Margin Used: ${margin:.2f}")
    
    if not long_pos and not short_pos:
        print(f"\nNo {symbol} positions found!")
        return
    
    # Get current price
    price_data = client.get_current_price(symbol)
    current_price = price_data['price']
    
    print(f"\nCurrent Price: ${current_price:,.2f}")
    
    # Calculate target quantities for $5 margin each (with 20x leverage)
    # $5 margin * 20x = $100 position value
    # Important: Binance requires $100 minimum notional for BTCUSDT
    target_position_value = 5.0 * 20  # $100
    target_qty = target_position_value / current_price
    target_qty = round(target_qty, 3)  # Round to 3 decimals for BTC
    
    # Verify minimum notional is met
    min_notional = 100.0
    if target_qty * current_price < min_notional:
        target_qty = min_notional / current_price
        target_qty = round(target_qty, 3)
    
    print(f"\n{'='*60}")
    print(f"Target Position Size (for $5 margin each):")
    print(f"{'='*60}")
    print(f"Target Qty: {target_qty} BTC")
    print(f"Target Position Value: ${target_qty * current_price:.2f}")
    print(f"Target Margin (20x): ${(target_qty * current_price) / 20:.2f}")
    print(f"Min Notional Required: ${min_notional:.2f} ✓")
    
    # Reduce positions
    print(f"\n{'='*60}")
    print(f"Reducing Positions:")
    print(f"{'='*60}")
    
    if long_pos:
        current_long_qty = long_pos.get('quantity', 0)
        if current_long_qty > target_qty:
            reduce_qty = current_long_qty - target_qty
            reduce_qty = round(reduce_qty, 3)
            print(f"\nLONG: Reducing by {reduce_qty} BTC (from {current_long_qty} to {target_qty})")
            
            # Place market SELL order to reduce LONG position
            result = client.place_market_order(
                symbol=symbol,
                side="SELL",
                quantity=reduce_qty,
                position_side="LONG"
            )
            print(f"  ✓ Order placed: {result}")
        else:
            print(f"\nLONG: Already at or below target ({current_long_qty} <= {target_qty})")
    
    if short_pos:
        current_short_qty = short_pos.get('quantity', 0)
        if current_short_qty > target_qty:
            reduce_qty = current_short_qty - target_qty
            reduce_qty = round(reduce_qty, 3)
            print(f"\nSHORT: Reducing by {reduce_qty} BTC (from {current_short_qty} to {target_qty})")
            
            # Place market BUY order to reduce SHORT position
            result = client.place_market_order(
                symbol=symbol,
                side="BUY",
                quantity=reduce_qty,
                position_side="SHORT"
            )
            print(f"  ✓ Order placed: {result}")
        else:
            print(f"\nSHORT: Already at or below target ({current_short_qty} <= {target_qty})")
    
    print(f"\n{'='*60}")
    print(f"Position Reduction Complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
