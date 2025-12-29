"""
Test NinjaTrader Bridge Connection

This script sends a test signal to the bridge to verify it's working.
"""

import socket
import json
import time

def test_bridge():
    print("=" * 80)
    print("NINJATRADER BRIDGE - CONNECTION TEST")
    print("=" * 80)
    print()

    # Test signal
    signal = {
        'Action': 'ENTRY',
        'Symbol': 'M6E',
        'Side': 'BUY',
        'Quantity': 1,
        'EntryPrice': 1.05000,
        'StopLoss': 1.04800,
        'TakeProfit': 1.05200,
        'Timestamp': '2025-12-13T18:00:00'
    }

    print(f"[TEST] Sending signal to NinjaTrader Bridge...")
    print(f"  Symbol: {signal['Symbol']}")
    print(f"  Side: {signal['Side']}")
    print(f"  Entry: {signal['EntryPrice']}")
    print(f"  Stop Loss: {signal['StopLoss']}")
    print(f"  Take Profit: {signal['TakeProfit']}")
    print()

    try:
        # Connect to bridge
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)

        print("[TEST] Connecting to localhost:8888...")
        sock.connect(('localhost', 8888))
        print("[TEST] ✓ Connected!")

        # Send signal
        message = json.dumps(signal)
        sock.sendall(message.encode('utf-8'))
        print("[TEST] ✓ Signal sent!")

        # Wait for response
        response = sock.recv(1024).decode('utf-8')
        sock.close()

        print(f"[TEST] Response from bridge: {response}")
        print()

        if response == "OK":
            print("=" * 80)
            print("✓ SUCCESS! Bridge is working!")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Check NinjaTrader for the order")
            print("2. Verify it placed BUY 1 M6E @ Market")
            print("3. Check for Stop Loss and Take Profit orders")
            print()
            return True
        else:
            print("=" * 80)
            print(f"⚠ WARNING: Unexpected response: {response}")
            print("=" * 80)
            return False

    except ConnectionRefusedError:
        print()
        print("=" * 80)
        print("❌ ERROR: Could not connect to bridge")
        print("=" * 80)
        print()
        print("Make sure:")
        print("1. NinjaTrader 8 is running")
        print("2. NinjaTraderBridge.exe is running")
        print()
        return False

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    test_bridge()
