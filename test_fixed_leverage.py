"""Test FIXED position sizing - should be 15% for ALL pairs"""

balance = 4906.97

pairs = [
    ("EUR_USD", 1.17),
    ("GBP_USD", 1.34),
    ("USD_JPY", 155.50),
    ("USD_CHF", 0.795),
    ("AUD_USD", 0.667),
    ("USD_CAD", 1.37),
    ("NZD_USD", 0.625),
]

print("FIXED POSITION SIZING - STRICT 15% LEVERAGE")
print("=" * 80)
print(f"Account Balance: ${balance:,.2f}")
print(f"Target: EXACTLY 15% leverage per position\n")
print(f"{'Pair':<12} {'Price':<10} {'Units':<12} {'Position $':<14} {'Leverage %':<12} {'$/pip'}")
print("-" * 80)

for pair, price in pairs:
    # FIXED calculation
    target_pos_value = balance * 0.15
    exact_units = target_pos_value / price
    units = int(round(exact_units / 100) * 100)
    if units < 100:
        units = 100

    pos_value = units * price
    leverage_pct = (pos_value / balance) * 100

    # Calculate $/pip
    if "JPY" in pair:
        dollars_per_pip = (units / 100) / 100
    else:
        dollars_per_pip = units / 10000

    status = "✓" if 14 <= leverage_pct <= 16 else "✗"

    print(f"{pair:<12} {price:<10.5f} {units:<12,} ${pos_value:<13,.2f} {leverage_pct:<11.2f}% ${dollars_per_pip:.2f}")

print()
print("All pairs should show 14-16% leverage (close to 15% target)")
print("EUR_USD should NOT be 23% anymore!")
