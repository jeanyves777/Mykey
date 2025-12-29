"""Quick test of OANDA connection for live trading"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Force correct account
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'

from trading_system.Forex_Trading.engine.oanda_client import OandaClient

print("=" * 60)
print("TESTING OANDA CONNECTION")
print("=" * 60)

# Connect
client = OandaClient('practice')

# Get account info
print("\nFetching account info...")
account_info = client.get_account_info()

if account_info:
    print("\nCONNECTION SUCCESSFUL!")
    print(f"\nAccount Details:")
    print(f"  Balance: ${client.get_balance():,.2f}")
    print(f"  NAV: ${client.get_nav():,.2f}")
    print(f"  Unrealized P&L: ${client.get_unrealized_pl():,.2f}")
    print(f"  Margin Available: ${client.get_margin_available():,.2f}")

    # Test getting current price
    print(f"\nFetching EUR_USD price...")
    price_data = client.get_current_price("EUR_USD")
    if price_data:
        print(f"  Bid: {price_data['bid']:.5f}")
        print(f"  Ask: {price_data['ask']:.5f}")
        print(f"  Spread: {price_data['spread']*10000:.1f} pips")

    print("\n" + "=" * 60)
    print("READY FOR LIVE TRADING!")
    print("=" * 60)
else:
    print("\nCONNECTION FAILED")
    print("Check your credentials and account ID")
