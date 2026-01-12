"""
Test OANDA Connection
====================
Quick test to verify OANDA API connection and fetch sample data.

Usage:
    python test_connection.py
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.oanda_client import OANDAClient
from config.trading_config import OANDA_CONFIG

def test_connection():
    """Test OANDA API connection."""
    print("\n" + "=" * 60)
    print("OANDA API CONNECTION TEST")
    print("=" * 60)
    
    # Initialize client
    client = OANDAClient(
        practice=OANDA_CONFIG["practice"]
    )
    
    print(f"Mode: {'PRACTICE' if OANDA_CONFIG['practice'] else 'LIVE'}")
    print(f"Account ID: {OANDA_CONFIG['account_id']}")
    print("-" * 60)
    
    # Test 1: Connection
    print("\n1. Testing connection...")
    if client.test_connection():
        print("   ✓ Connection successful")
    else:
        print("   ✗ Connection failed")
        return False
    
    # Test 2: Account summary
    print("\n2. Fetching account summary...")
    account_info = client.get_account_summary()
    if account_info:
        print(f"   ✓ Balance: ${float(account_info.get('balance', 0)):,.2f}")
        print(f"   ✓ NAV: ${float(account_info.get('NAV', 0)):,.2f}")
        print(f"   ✓ Unrealized P&L: ${float(account_info.get('unrealizedPL', 0)):+,.2f}")
        print(f"   ✓ Margin Available: ${float(account_info.get('marginAvailable', 0)):,.2f}")
    else:
        print("   ✗ Failed to fetch account summary")
        return False
    
    # Test 3: Fetch candles
    print("\n3. Fetching EUR_USD candles (M15, last 10)...")
    candles = client.get_candles("EUR_USD", "M15", 10)
    if candles is not None and not candles.empty:
        print(f"   ✓ Fetched {len(candles)} candles")
        if len(candles) > 0:
            latest = candles.iloc[-1]
            print(f"   Latest: O={latest['open']:.5f} H={latest['high']:.5f} L={latest['low']:.5f} C={latest['close']:.5f}")
    else:
        print("   ✗ Failed to fetch candles")
        return False
    
    # Test 4: Current pricing
    print("\n4. Fetching current pricing for EUR_USD...")
    pricing = client.get_current_price("EUR_USD")
    if pricing:
        print(f"   ✓ Bid: {pricing.get('bid', 0):.5f}")
        print(f"   ✓ Ask: {pricing.get('ask', 0):.5f}")
        print(f"   ✓ Mid: {pricing.get('mid', 0):.5f}")
        print(f"   ✓ Spread: {pricing.get('spread', 0):.5f}")
    else:
        print("   ✗ Failed to fetch pricing")
        return False
    
    # Test 5: Open positions
    print("\n5. Checking open positions...")
    positions = client.get_open_positions()
    if positions is not None:
        print(f"   ✓ Open positions: {len(positions)}")
        if len(positions) > 0:
            for pos in positions:
                instrument = pos.get('instrument', 'UNKNOWN')
                units = pos.get('units', 0)
                unrealized_pl = pos.get('unrealizedPL', 0)
                print(f"     - {instrument}: {units} units, P&L: ${float(unrealized_pl):+.2f}")
    else:
        print("   ✗ Failed to check positions")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nYou can now run the trading system:")
    print("  python main.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
