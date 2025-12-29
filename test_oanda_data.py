import requests

token = '077a323e34fdce0c3ed3804f1578c688-434054a5ecc079f56a1fbd6e5f180f1e'
account = '101-001-8364309-001'

headers = {'Authorization': f'Bearer {token}'}

print("Testing OANDA historical data...")
print("="*60)

# Test getting recent candles
r = requests.get(
    'https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles',
    headers=headers,
    params={'granularity': 'M1', 'count': 100}
)

print(f"Status: {r.status_code}")

if r.status_code == 200:
    candles = r.json().get('candles', [])
    print(f"Candles retrieved: {len(candles)}")

    if candles:
        latest = candles[-1]
        print(f"\nLatest candle:")
        print(f"  Time: {latest['time']}")
        print(f"  Open: {latest['mid']['o']}")
        print(f"  High: {latest['mid']['h']}")
        print(f"  Low: {latest['mid']['l']}")
        print(f"  Close: {latest['mid']['c']}")
        print(f"  Volume: {latest['volume']}")

        print("\nOANDA historical data is working!")
else:
    print(f"Error: {r.text}")
