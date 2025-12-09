"""Test new Kraken Futures API credentials."""

from kraken.futures import User, Market

# NEW LIVE API credentials
API_KEY = "eGjebbTZb9xexGZF0/KqVVD4cPO8X4IpNtwlP33b0hTPO1rdxnmgsZTQ"
API_SECRET = "GeOKTYIGecTsivBBrXDV95WpxXVqt5BEudTmv0PlsXoBVAvaceIFOp0K1EglsoUOrvfLtBiwWC1IF2WdkRYjrQ=="

print("=" * 60)
print("TESTING NEW KRAKEN FUTURES API CREDENTIALS")
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
    wallets = user.get_wallets()
    print(f"   Authenticated API: OK")
    print(f"   Wallets: {wallets}")
except Exception as e:
    print(f"   Authenticated API Error: {e}")

# Get account info
print("\n3. Getting Account Info...")
try:
    user = User(key=API_KEY, secret=API_SECRET)
    accounts = user.get_accounts()
    print(f"   Accounts: {accounts}")
except Exception as e:
    print(f"   Account Error: {e}")

print("\n" + "=" * 60)
