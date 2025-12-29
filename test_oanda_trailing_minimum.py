"""Test OANDA trailing stop minimum"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient

os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

print("Testing OANDA trailing stop minimums...")
print("=" * 60)

# Test different trailing distances
test_distances = [0.00010, 0.00020, 0.00050, 0.00100]  # 10, 20, 50, 100 pips

for distance in test_distances:
    pips = distance * 10000
    print(f"\nTesting {pips:.0f} pips trailing distance ({distance})...")

    result = client.place_market_order(
        "USD_CHF",
        1000,  # Small position
        None,  # No SL
        None,  # No TP
        distance  # Just trailing
    )

    if result.get("success"):
        print(f"  ✅ SUCCESS! {pips:.0f} pips works")
        # Close the trade
        if result.get("trade_id"):
            client.close_trade(result["trade_id"])
        break
    else:
        error = result.get("error", "Unknown error")
        if "MINIMUM_NOT_MET" in str(error):
            print(f"  ❌ TOO SMALL - Minimum not met")
        else:
            print(f"  ❌ FAILED - {error}")

print("\n" + "=" * 60)
