"""Test USD_JPY position sizing specifically"""
import sys, os
sys.path.insert(0, '.')
from trading_system.Forex_Trading.engine.oanda_client import OandaClient

os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

balance = client.get_balance()
print(f"\nAccount Balance: ${balance:,.2f}")

# Get USD_JPY price
price_data = client.get_current_price('USD_JPY')
price = (price_data['bid'] + price_data['ask']) / 2

print(f"USD_JPY Price: {price:.5f}")

# Calculate 15% position
target_15pct = balance * 0.15
units = int(target_15pct / price)

print(f"\nTarget Position: 15% = ${target_15pct:.2f}")
print(f"Calculated Units: {units}")

# Check what happens with different unit amounts
print(f"\nTesting different unit sizes:")
for test_units in [1, 2, 3, 4, 5, 10]:
    pos_value = test_units * price
    pct = (pos_value / balance) * 100
    status = "OK" if pct <= 20 else "TOO HIGH"
    print(f"  {test_units} units = ${pos_value:,.2f} = {pct:.2f}% [{status}]")

# Now place a REAL test order with correct sizing
print(f"\nPlacing TEST order with {units} units...")
result = client.place_market_order(
    instrument='USD_JPY',
    units=units,
    stop_loss=None,
    take_profit=None,
    trailing_stop_distance=None
)

if result.get('success'):
    filled_price = result.get('filled_price')
    filled_units = result.get('filled_units')
    actual_value = filled_units * filled_price
    actual_pct = (actual_value / balance) * 100

    print(f"\n[SUCCESS]")
    print(f"  Filled: {filled_units} units @ {filled_price:.5f}")
    print(f"  Position Value: ${actual_value:,.2f}")
    print(f"  Percentage: {actual_pct:.2f}%")

    # Close immediately
    trade_id = result.get('trade_id')
    if trade_id:
        print(f"\nClosing test trade...")
        close_result = client.close_trade(trade_id)
        if close_result.get('success'):
            pnl = close_result.get('pl', 0)
            print(f"  Closed! P&L: ${pnl:+.2f}")
else:
    print(f"\n[FAILED]: {result.get('error')}")

print("\nTest complete!")
