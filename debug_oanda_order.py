"""Debug OANDA order placement errors"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
import json

# Connect
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

print("=" * 80)
print("DEBUGGING OANDA ORDER PLACEMENT")
print("=" * 80)

# Get current price for USD_CHF (one that failed)
print("\n1. Testing USD_CHF order (the one that failed)...")
price_data = client.get_current_price("USD_CHF")
print(f"   Current price: {price_data['ask']:.5f}")

# Try to place order with same parameters
entry_price = price_data['ask']
units = 24000
stop_loss = entry_price * (1 - 0.00006)  # 6 pips SL
take_profit = entry_price * (1 + 0.00010)  # 10 pips TP
trailing_distance = entry_price * 0.00003  # 3 pips trailing

print(f"\n2. Order Parameters:")
print(f"   Entry: {entry_price:.5f}")
print(f"   Units: {units:,}")
print(f"   Stop Loss: {stop_loss:.5f} ({(entry_price - stop_loss) * 10000:.1f} pips)")
print(f"   Take Profit: {take_profit:.5f} ({(take_profit - entry_price) * 10000:.1f} pips)")
print(f"   Trailing Distance: {trailing_distance:.5f}")

print(f"\n3. Attempting to place order...")

# Build order data manually to see what we're sending
order_data = {
    "order": {
        "type": "MARKET",
        "instrument": "USD_CHF",
        "units": str(units),
        "timeInForce": "FOK",
        "positionFill": "DEFAULT",
        "stopLossOnFill": {
            "price": str(round(stop_loss, 5))
        },
        "takeProfitOnFill": {
            "price": str(round(take_profit, 5))
        },
        "trailingStopLossOnFill": {
            "distance": str(round(trailing_distance, 5))
        }
    }
}

print(f"\n4. Order JSON:")
print(json.dumps(order_data, indent=2))

# Try to place the order
result = client.place_market_order(
    "USD_CHF",
    units,
    stop_loss,
    take_profit,
    trailing_distance
)

print(f"\n5. Result:")
print(json.dumps(result, indent=2, default=str))

if not result.get("success"):
    print(f"\n6. FAILED! Error: {result.get('error')}")

    # Try without trailing stop
    print(f"\n7. Trying without trailing stop...")
    result2 = client.place_market_order(
        "USD_CHF",
        units,
        stop_loss,
        take_profit,
        None  # No trailing
    )
    print(json.dumps(result2, indent=2, default=str))

    if not result2.get("success"):
        # Try with minimal order
        print(f"\n8. Trying minimal order (just units)...")
        result3 = client.place_market_order(
            "USD_CHF",
            units,
            None,  # No SL
            None,  # No TP
            None   # No trailing
        )
        print(json.dumps(result3, indent=2, default=str))

print("\n" + "=" * 80)
