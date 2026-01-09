"""Debug Kraken Futures API authentication."""

import hashlib
import hmac
import base64
import time
import requests

# Demo API credentials
API_KEY = "A/D8Ua1cQEFnHZVZUvZ757FxYqG9O+6T6gQsnTOzLLLxjNilxDqpCNZR"
API_SECRET = "Ge5Ad1NNwum2RtqJDP8Eguso/ZaIzVVSXnr2mrUR8M8zE6ksD7W3OmnKVfhupyJfUjVx5ALev36CXw9CHYoWb43L"

BASE_URL = "https://demo-futures.kraken.com"
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

print("Testing Kraken Futures Demo Authentication...")
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

print(f"\nRequest:")
print(f"  URL: {BASE_URL}{endpoint}")
print(f"  Nonce: {nonce}")
print(f"  Signature: {signature[:40]}...")

try:
    response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, timeout=30)
    print(f"\nResponse:")
    print(f"  Status: {response.status_code}")
    print(f"  Body: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
