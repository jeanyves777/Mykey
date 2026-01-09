"""Test Kraken Futures using official SDK."""

from kraken.futures import User, Market

# LIVE API credentials
API_KEY = "WdkZJs/OEQEvW85E5pji6chtRDPf35NsDy8L/VBgx7lpHd54yXfJGB9v"
API_SECRET = "jnj0eETBm9wtNwo37pay8Mgg6QhQC5+3riMBY/6UBEa1DWklQF+D0s58dvddZxGXtA75o25bwd2MUpox88wA/g=="

print("=" * 60)
print("TESTING KRAKEN FUTURES USING OFFICIAL SDK")
print("=" * 60)

# Test public market data first
print("\n1. Testing Public Market Data...")
try:
    market = Market()
    tickers = market.get_tickers()
    print(f"   Public API: OK - Found {len(tickers.get('tickers', []))} symbols")

    # Find ETH
    for ticker in tickers.get('tickers', []):
        if ticker.get('symbol') == 'PI_ETHUSD':
            print(f"   ETH Perpetual: Last=${ticker.get('last', 0)}")
            break
except Exception as e:
    print(f"   Public API Error: {e}")

# Test authenticated endpoint
print("\n2. Testing Authenticated API (LIVE)...")
try:
    user = User(key=API_KEY, secret=API_SECRET)
    accounts = user.get_wallets()
    print(f"   Authenticated API: OK")
    print(f"   Response: {accounts}")
except Exception as e:
    print(f"   Authenticated API Error: {e}")

print("\n" + "=" * 60)
