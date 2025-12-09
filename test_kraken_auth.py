"""Test Kraken Futures API with authentication."""

from trading_system.clients.kraken_futures_client import KrakenFuturesClient

# Demo API credentials
DEMO_API_KEY = "A/D8Ua1cQEFnHZVZUvZ757FxYqG9O+6T6gQsnTOzLLLxjNilxDqpCNZR"
DEMO_API_SECRET = "Ge5Ad1NNwum2RtqJDP8Eguso/ZaIzVVSXnr2mrUR8M8zE6ksD7W3OmnKVfhupyJfUjVx5ALev36CXw9CHYoWb43L"

print("=" * 60)
print("TESTING KRAKEN FUTURES DEMO API")
print("=" * 60)

# Create client with demo credentials
client = KrakenFuturesClient(
    api_key=DEMO_API_KEY,
    api_secret=DEMO_API_SECRET,
    demo=True
)

# Test public endpoint
print("\n1. Testing Public API (Tickers)...")
if client.test_connection():
    print("   Public API: OK")
else:
    print("   Public API: FAILED")

# Test authenticated endpoint
print("\n2. Testing Authenticated API (Account)...")
if client.test_auth():
    print("   Authenticated API: OK")
else:
    print("   Authenticated API: FAILED")

# Get account summary
print("\n3. Account Summary:")
summary = client.get_account_summary()
if summary:
    print(f"   Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}")
    print(f"   Available Margin: ${summary.get('available_margin', 0):,.2f}")
    print(f"   Used Margin: ${summary.get('used_margin', 0):,.2f}")
    print(f"   Balance: ${summary.get('balance', 0):,.2f}")
else:
    print("   Could not get account summary")

# Get ETH quote
print("\n4. ETH Perpetual Quote:")
quote = client.get_quote("PI_ETHUSD")
if quote:
    print(f"   Bid: ${quote.bid:,.2f}")
    print(f"   Ask: ${quote.ask:,.2f}")
    print(f"   Mid: ${quote.mid:,.2f}")
else:
    print("   Could not get quote")

# Check for existing positions
print("\n5. Open Positions:")
positions = client.get_positions()
if positions:
    for pos in positions:
        print(f"   {pos.symbol}: {pos.side} {pos.size} @ ${pos.entry_price:.2f}")
else:
    print("   No open positions")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
