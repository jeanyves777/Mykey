"""Test the NEW position sizing calculation (with 100-unit rounding)"""

balance = 4906.97

pairs = [
    ("EUR_USD", 1.17),
    ("GBP_USD", 1.34),
    ("AUD_USD", 0.667),
    ("USD_CAD", 1.37),
]

print("NEW POSITION SIZING (100-unit rounding)")
print("=" * 70)
print(f"Account Balance: ${balance:,.2f}")
print(f"Target: 15% leverage per position")
print()
print(f"{'Pair':<12} {'Price':<10} {'Units':<12} {'Position $':<14} {'Leverage %':<12}")
print("-" * 70)

for pair, price in pairs:
    # NEW calculation
    max_position_value = balance * 0.15
    exact_units = max_position_value / price
    units = int(round(exact_units / 100) * 100)  # Round to 100
    units = max(units, 1000)  # Min 1000

    pos_value = units * price
    leverage_pct = (pos_value / balance) * 100

    print(f"{pair:<12} {price:<10.5f} {units:<12,} ${pos_value:<13,.2f} {leverage_pct:<11.2f}%")

print()
print("This should show ~15% for all pairs (or close to it)")
