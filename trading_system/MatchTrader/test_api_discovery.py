"""
Match-Trader API Discovery
==========================
Test different API endpoint patterns to find working authentication.
"""

import requests
import json

# Config
BASE_URLS = [
    "https://demo.match-trader.com",
    "https://demo.match-trader.com/api",
    "https://demo.match-trader.com/mtr-api",
    "https://api.demo.match-trader.com",
]

LOGIN_ENDPOINTS = [
    "/manager/mtr-login",
    "/manager/login",
    "/api/login",
    "/api/auth/login",
    "/auth/login",
    "/v1/auth/login",
    "/mtr-api/login",
    "/trading/login",
    "/user/login",
]

EMAIL = "kouakoukoffi71@gmail.com"
PASSWORD = "Michellebb7$"
BROKER_ID = "1052747"

def test_endpoint(base_url: str, endpoint: str, data: dict) -> dict:
    """Test an endpoint with different payload formats"""
    url = f"{base_url}{endpoint}"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        return {
            'url': url,
            'status': response.status_code,
            'response': response.text[:500] if response.text else None
        }
    except requests.exceptions.RequestException as e:
        return {
            'url': url,
            'status': 'error',
            'response': str(e)
        }

def main():
    print("\n" + "=" * 70)
    print("  MATCH-TRADER API ENDPOINT DISCOVERY")
    print("=" * 70)

    # Different payload formats to try
    payloads = [
        {"email": EMAIL, "password": PASSWORD, "brokerId": BROKER_ID},
        {"email": EMAIL, "password": PASSWORD, "broker_id": BROKER_ID},
        {"login": EMAIL, "password": PASSWORD, "brokerId": BROKER_ID},
        {"username": EMAIL, "password": PASSWORD, "brokerId": BROKER_ID},
        {"email": EMAIL, "password": PASSWORD},
    ]

    results = []

    for base_url in BASE_URLS:
        print(f"\nTesting base URL: {base_url}")
        print("-" * 50)

        for endpoint in LOGIN_ENDPOINTS:
            for payload in payloads:
                result = test_endpoint(base_url, endpoint, payload)

                status = result['status']
                # Highlight potentially successful responses
                if status == 200:
                    print(f"  SUCCESS! {endpoint}")
                    print(f"    Payload: {list(payload.keys())}")
                    print(f"    Response: {result['response'][:200]}")
                    results.append(result)
                elif status == 401:
                    print(f"  [401] {endpoint} - Auth required (endpoint exists!)")
                    results.append(result)
                elif status == 400:
                    print(f"  [400] {endpoint} - Bad request (endpoint exists!)")
                    results.append(result)
                elif status == 404:
                    pass  # Skip not found
                elif status == 403:
                    print(f"  [403] {endpoint} - Forbidden")
                elif status != 'error':
                    print(f"  [{status}] {endpoint}")

    # Also try to get any public info
    print("\n" + "=" * 70)
    print("  TESTING PUBLIC ENDPOINTS")
    print("=" * 70)

    public_endpoints = [
        "/api/version",
        "/api/health",
        "/api/status",
        "/manager/platform-details",
        "/api/platform-details",
        "/api/info",
    ]

    for base_url in BASE_URLS[:2]:
        for endpoint in public_endpoints:
            url = f"{base_url}{endpoint}"
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"  [200] {url}")
                    print(f"    Response: {response.text[:300]}")
            except:
                pass

    print("\n" + "=" * 70)
    print("  DISCOVERY COMPLETE")
    print("=" * 70)

    # Summary
    if results:
        print("\nPotential working endpoints found:")
        for r in results:
            if r['status'] in [200, 400, 401]:
                print(f"  - {r['url']} ({r['status']})")

if __name__ == "__main__":
    main()
