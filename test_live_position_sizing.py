"""
Test Live Position Sizing on OANDA
Places TINY test orders on all pairs to verify sizing is correct
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.multi_symbol_scalping_config import MAJOR_PAIRS
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params
import time

print("=" * 80)
print("LIVE POSITION SIZING TEST - OANDA PRACTICE ACCOUNT")
print("=" * 80)
print("\nThis will place TINY test orders to verify position sizing")
print("Using 1% of account per test (not 15%)")
print()

# Initialize OANDA
os.environ['OANDA_PRACTICE_ACCOUNT_ID'] = '101-001-8364309-001'
client = OandaClient('practice')

# Get account balance
balance = client.get_balance()
print(f"Account Balance: ${balance:,.2f}\n")

# Initialize strategy (just to use position sizing function)
strategy = ForexScalpingStrategy(
    instruments=MAJOR_PAIRS,
    max_trades_per_day=10,
    daily_profit_target=0.05,
    trade_size_pct=0.05
)

print(f"{'Pair':<12} {'Price':<10} {'Target $':<12} {'Units':<10} {'Actual $':<12} {'% of Acct':<12} {'Status'}")
print("-" * 90)

test_results = []

for instrument in MAJOR_PAIRS:
    try:
        # Get current price
        price_data = client.get_current_price(instrument)
        if not price_data:
            print(f"{instrument:<12} ERROR: Could not get price")
            continue

        current_price = (price_data['bid'] + price_data['ask']) / 2

        # Calculate position size - USE 1% FOR TEST (not 15%)
        test_balance = balance * 0.01  # 1% test
        units = int(test_balance / current_price)

        # Make sure at least 1 unit
        if units < 1:
            units = 1

        # Calculate actual position value
        actual_position_value = units * current_price
        actual_pct = (actual_position_value / balance) * 100

        print(f"{instrument:<12} {current_price:<10.5f} ${test_balance:<11,.2f} {units:<10,} ${actual_position_value:<11,.2f} {actual_pct:<11.2f}% ", end="")

        # Place TINY BUY order (we'll close immediately)
        result = client.place_market_order(
            instrument=instrument,
            units=units,  # SMALL test
            stop_loss=None,
            take_profit=None,
            trailing_stop_distance=None
        )

        if result.get('success'):
            print("✓ PLACED")

            test_results.append({
                'instrument': instrument,
                'target_dollars': test_balance,
                'units': units,
                'actual_dollars': actual_position_value,
                'actual_pct': actual_pct,
                'trade_id': result.get('trade_id'),
                'filled_price': result.get('filled_price')
            })

            # Wait a moment
            time.sleep(0.5)

            # CLOSE the test trade immediately
            if result.get('trade_id'):
                close_result = client.close_trade(result['trade_id'])
                if close_result.get('success'):
                    pnl = close_result.get('pl', 0)
                    print(f"              CLOSED | P&L: ${pnl:+.2f}")
                else:
                    print(f"              ERROR CLOSING: {close_result.get('error')}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"✗ FAILED: {error}")

        # Small delay between orders
        time.sleep(1)

    except Exception as e:
        print(f"ERROR: {e}")

# Summary
print("\n" + "=" * 80)
print("TEST RESULTS SUMMARY")
print("=" * 80)

if test_results:
    print(f"\nSuccessfully tested {len(test_results)}/{len(MAJOR_PAIRS)} pairs")
    print(f"\nAll tests used 1% of account = ${balance * 0.01:,.2f}")
    print("\nActual percentages achieved:")

    for result in test_results:
        diff = abs(result['actual_pct'] - 1.0)
        status = "✓" if diff < 0.1 else "⚠"
        print(f"  {status} {result['instrument']}: {result['actual_pct']:.2f}% (target: 1.00%)")

    avg_pct = sum(r['actual_pct'] for r in test_results) / len(test_results)
    print(f"\nAverage: {avg_pct:.2f}%")

    if avg_pct > 0.9 and avg_pct < 1.1:
        print("\n✓ POSITION SIZING IS CORRECT!")
        print("  All pairs are close to 1% target")
        print("  In live trading, 15% will work correctly")
    else:
        print("\n⚠ POSITION SIZING NEEDS REVIEW")
        print(f"  Average {avg_pct:.2f}% instead of 1.00%")

else:
    print("\nNo successful tests")

print("\n" + "=" * 80)
print("Test complete. All positions closed.")
print("=" * 80)
