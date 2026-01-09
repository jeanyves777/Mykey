"""Fix doubled positions by selling excess to get to correct $66.60 per symbol."""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load .env file
load_dotenv(r'C:\Users\Jean-Yves\thevolumeainative\.env')

api_key = os.getenv('ALPACA_CRYPTO_KEY') or os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_CRYPTO_SECRET') or os.getenv('ALPACA_SECRET_KEY')

if not api_key or not api_secret:
    print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    exit(1)

client = TradingClient(api_key, api_secret, paper=True)
positions = client.get_all_positions()

TARGET_VALUE = 66.60  # Target $66.60 per symbol (Entry position only)

print("=" * 60)
print("FIXING DOUBLED POSITIONS")
print("=" * 60)

for p in positions:
    symbol = p.symbol
    current_qty = float(p.qty)
    current_price = float(p.current_price)
    current_value = float(p.market_value)

    # Calculate target quantity
    target_qty = TARGET_VALUE / current_price
    excess_qty = current_qty - target_qty

    print(f"\n{symbol}:")
    print(f"  Current: {current_qty:.6f} = ${current_value:,.2f}")
    print(f"  Target:  {target_qty:.6f} = ${TARGET_VALUE:,.2f}")
    print(f"  Excess:  {excess_qty:.6f} = ${excess_qty * current_price:,.2f}")

    if excess_qty > 0:
        # Round to appropriate precision
        if symbol == 'BTCUSD':
            sell_qty = round(excess_qty, 6)
        elif symbol == 'ETHUSD':
            sell_qty = round(excess_qty, 5)
        else:
            sell_qty = round(excess_qty, 4)

        print(f"  Selling: {sell_qty:.6f}")

        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=sell_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            result = client.submit_order(order)
            print(f"  [OK] Order submitted: {result.id}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print(f"  Already at correct size")

print("\n" + "=" * 60)
print("Waiting for orders to fill...")
print("=" * 60)

import time
time.sleep(2)

# Check final positions
positions = client.get_all_positions()
total = sum(float(p.market_value) for p in positions)
print("\nFINAL POSITIONS:")
for p in positions:
    print(f"  {p.symbol}: {float(p.qty):.6f} = ${float(p.market_value):,.2f}")
print(f"\nTOTAL: ${total:,.2f} (target: ~$200)")
