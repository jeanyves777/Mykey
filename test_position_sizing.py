"""Test position sizing calculation"""

account_balance = 979.87
trade_size_pct = 0.05  # 5%
trade_value = account_balance * trade_size_pct

print("=" * 60)
print("POSITION SIZING TEST")
print("=" * 60)
print(f"\nAccount Balance: ${account_balance:.2f}")
print(f"Trade Size (5%): ${trade_value:.2f}")
print()

# Calculate target dollars per pip
target_dollars_per_pip = trade_value / 20  # Risk 20 pips
print(f"Target Risk per Pip: ${target_dollars_per_pip:.2f}")

# Calculate units needed
units = int(target_dollars_per_pip * 10000)
units = (units // 1000) * 1000
units = max(units, 5000)

print(f"\nCalculated Units: {units:,}")
print()

print("Position Sizing Reference:")
print("  5,000 units = $0.50/pip")
print("  10,000 units = $1.00/pip")
print("  20,000 units = $2.00/pip")
print(f"  {units:,} units = ${units/10000:.2f}/pip")
print()

print(f"With {units:,} units on GBP/USD:")
print(f"  10 pips profit = ${units/10000 * 10:.2f}")
print(f"  20 pips SL = ${units/10000 * 20:.2f} loss")
print(f"  30 pips TP = ${units/10000 * 30:.2f} profit")
print()

print("=" * 60)
print("MUCH BETTER! Now making $2.40/pip instead of $0.10/pip")
print("=" * 60)
