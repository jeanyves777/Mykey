"""Test Kraken Futures LIVE API with authentication."""

import hashlib
import hmac
import base64
import time
import requests

# LIVE API credentials (for market data testing only)
API_KEY = "WdkZJs/OEQEvW85E5pji6chtRDPf35NsDy8L/VBgx7lpHd54yXfJGB9v"
API_SECRET = "jnj0eETBm9wtNwo37pay8Mgg6QhQC5+3riMBY/6UBEa1DWklQF+D0s58dvddZxGXtA75o25bwd2MUpox88wA/g=="

# Use LIVE API
BASE_URL = "https://futures.kraken.com"
API_PATH = "/derivatives/api/v3"

def sign_request(endpoint, post_data="", nonce=None):
    """Sign a request using Kraken Futures method."""
    if nonce is None:
        nonce = str(int(time.time() * 1000))

    # Method from Kraken docs: SHA256(postData + nonce + endpointPath)
    message = post_data + nonce + endpoint
    sha256_hash = hashlib.sha256(message.encode('utf-8')).digest()

    # HMAC-SHA512 with base64-decoded secret
    secret_decoded = base64.b64decode(API_SECRET)
    signature = hmac.new(secret_decoded, sha256_hash, hashlib.sha512).digest()
    signature_b64 = base64.b64encode(signature).decode('utf-8')

    return nonce, signature_b64

print("=" * 60)
print("TESTING KRAKEN FUTURES LIVE API")
print("=" * 60)
print(f"API Key: {API_KEY[:20]}...")
print(f"Base URL: {BASE_URL}")

# Test accounts endpoint
endpoint = f"{API_PATH}/accounts"
nonce, signature = sign_request(endpoint)

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'APIKey': API_KEY,
    'Nonce': nonce,
    'Authent': signature,
}

print(f"\nTesting authenticated endpoint: {endpoint}")

try:
    response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, timeout=30)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body: {response.text[:1000]}")
except Exception as e:
    print(f"Error: {e}")
