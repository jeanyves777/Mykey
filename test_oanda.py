import os
os.environ.clear()
from dotenv import load_dotenv
load_dotenv(override=True)

from trading_system.Forex_Trading.engine.oanda_client import OandaClient

print("Testing OANDA Connection...")
print("="*60)

client = OandaClient('practice')

print(f"\nAccount Balance: ${client.get_balance():,.2f}")
print(f"NAV: ${client.get_nav():,.2f}")
print(f"Margin Available: ${client.get_margin_available():,.2f}")

print("\nGetting EUR/USD price...")
price = client.get_current_price('EUR_USD')
if 'bid' in price:
    print(f"EUR/USD: Bid={price['bid']}, Ask={price['ask']}, Spread={price['spread']:.5f}")
else:
    print(f"Error: {price}")

print("\nGetting 10 recent 1-min candles...")
candles = client.get_candles('EUR_USD', 'M1', 10)
print(f"Retrieved {len(candles)} candles")
if candles:
    print(f"Latest candle: {candles[-1]}")

print("\n✅ OANDA API is working!")
