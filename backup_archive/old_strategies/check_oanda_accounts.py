import requests

token = '077a323e34fdce0c3ed3804f1578c688-434054a5ecc079f56a1fbd6e5f180f1e'
headers = {'Authorization': f'Bearer {token}'}

print("Checking All OANDA Accounts...")
print("="*60)

r = requests.get('https://api-fxpractice.oanda.com/v3/accounts', headers=headers)
accounts = r.json()['accounts']

print(f"\nFound {len(accounts)} accounts\n")

for acc in accounts:
    acc_id = acc['id']
    print(f"Account: {acc_id}")
    print(f"  Tags: {acc.get('tags', [])}")

    # Try to get details
    r2 = requests.get(f'https://api-fxpractice.oanda.com/v3/accounts/{acc_id}/summary', headers=headers)

    if r2.status_code == 200:
        data = r2.json()['account']
        print(f"  Alias: {data.get('alias', 'N/A')}")
        print(f"  Balance: ${float(data['balance']):,.2f}")
        print(f"  NAV: ${float(data['NAV']):,.2f}")
        print(f"  Currency: {data['currency']}")
        print(f"  Status: ACCESSIBLE")
    else:
        print(f"  Status: NOT ACCESSIBLE ({r2.status_code})")

    print()

print("="*60)
print("\nRecommendation: Use the account with highest balance")
