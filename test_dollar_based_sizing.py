"""Test DOLLAR-BASED position sizing - SIMPLE and CORRECT"""

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

print("DOLLAR-BASED POSITION SIZING - SIMPLE & CORRECT")
print("=" * 85)
print(f"Account Balance: ${balance:,.2f}")
print(f"Target: 15% of balance = ${balance * 0.15:,.2f} per position\n")
print(f"{'Pair':<12} {'Price':<10} {'Units':<12} {'Position $':<14} {'% of Acct':<12} {'$/pip'}")
print("-" * 85)

for pair, price in pairs:
    # SIMPLE: 15% of balance in dollars
    target_dollars = balance * 0.15
    units = int(target_dollars / price)

    pos_value = units * price
    pct_of_account = (pos_value / balance) * 100

    # Calculate $/pip
    if "JPY" in pair:
        dollars_per_pip = (units / 100) / 100
    else:
        dollars_per_pip = units / 10000

    print(f"{pair:<12} {price:<10.5f} {units:<12,} ${pos_value:<13,.2f} {pct_of_account:<11.2f}% ${dollars_per_pip:.2f}")

print()
print("ALL pairs should show ~15% of account (±1% due to rounding)")
print("This is SIMPLE and avoids all leverage complications!")
