"""Test 15% position sizing on all pairs to verify ~$600 target"""

import sys, os
sys.path.insert(0, '.')
from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
import time

print("=" * 80)
print("15% POSITION SIZING TEST - TARGET ~$600 PER PAIR")
print("=" * 80)

# Initialize OANDA
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

# Get account balance
balance = client.get_balance()
target_dollars = balance * 0.15

print(f"\nAccount Balance: ${balance:,.2f}")
print(f"15% Target: ${target_dollars:,.2f}\n")

print(f"{'Pair':<12} {'Price':<10} {'Units':<10} {'Position $':<12} {'% of Acct':<12} {'Diff from $600'}")
print("-" * 85)

test_results = []

for instrument in MAJOR_PAIRS:
    try:
        # Get current price
        price_data = client.get_current_price(instrument)
        if not price_data:
            print(f"{instrument:<12} ERROR: Could not get price")
            continue

        current_price = (price_data['bid'] + price_data['ask']) / 2

        # Calculate position size - 15% of balance
        target_position_dollars = balance * 0.15
        units = int(target_position_dollars / current_price)

        # Minimum 1 unit
        if units < 1:
            units = 1

        # Calculate actual position value
        actual_position_value = units * current_price
        actual_pct = (actual_position_value / balance) * 100
        diff_from_600 = actual_position_value - 600

        print(f"{instrument:<12} {current_price:<10.5f} {units:<10,} ${actual_position_value:<11,.2f} {actual_pct:<11.2f}% ${diff_from_600:+.2f}")

        test_results.append({
            'instrument': instrument,
            'units': units,
            'actual_dollars': actual_position_value,
            'actual_pct': actual_pct,
            'diff_from_600': diff_from_600
        })

        # Small delay
        time.sleep(0.2)

    except Exception as e:
        print(f"{instrument:<12} ERROR: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if test_results:
    avg_dollars = sum(r['actual_dollars'] for r in test_results) / len(test_results)
    avg_pct = sum(r['actual_pct'] for r in test_results) / len(test_results)

    print(f"\nTested {len(test_results)}/{len(MAJOR_PAIRS)} pairs")
    print(f"Target: ${target_dollars:,.2f} per pair (15% of ${balance:,.2f})")
    print(f"\nAverage position: ${avg_dollars:,.2f} ({avg_pct:.2f}%)")

    # Check if all within acceptable range
    all_good = all(r['actual_pct'] >= 12.0 and r['actual_pct'] <= 16.0 for r in test_results)

    if all_good:
        print("\n✓ ALL PAIRS WITHIN ACCEPTABLE RANGE (12-16%)")
        print("  Position sizing is working correctly!")
    else:
        print("\n⚠ SOME PAIRS OUTSIDE ACCEPTABLE RANGE")
        for r in test_results:
            if r['actual_pct'] < 12.0 or r['actual_pct'] > 16.0:
                print(f"  {r['instrument']}: {r['actual_pct']:.2f}%")

else:
    print("\nNo successful tests")

print("\n" + "=" * 80)
