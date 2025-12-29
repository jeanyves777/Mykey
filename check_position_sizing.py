"""Check position sizing with 15% leverage cap"""
balance = 981.06
leverage_cap = 0.15  # 15% per position

print("=" * 70)
print("NEW POSITION SIZING - 15% LEVERAGE CAP")
print("=" * 70)
print(f"\nAccount Balance: ${balance:,.2f}")
print(f"Leverage Cap per Trade: {leverage_cap*100}%")
print(f"Max Position Value: ${balance * leverage_cap:,.2f}")
print(f"Max 5 Positions Total: {leverage_cap * 5 * 100}% = ${balance * leverage_cap * 5:,.2f}")

print(f"\n{'Pair':<10} {'Price':<8} {'Units':<10} {'Position $':<12} {'$/pip':<8} {'% of Acct':<10}")
print("-" * 70)

pairs = [
    ("EUR/USD", 1.17),
    ("GBP/USD", 1.34),
    ("USD/JPY", 155.50),
    ("USD/CHF", 0.795),
    ("AUD/USD", 0.665),
    ("USD/CAD", 1.37),
    ("NZD/USD", 0.625),
]

for pair, price in pairs:
    max_position_value = balance * leverage_cap
    units = int(max_position_value / price)
    units = (units // 1000) * 1000
    units = max(units, 1000)

    position_value = units * price
    pct_of_account = (position_value / balance) * 100

    # Calculate $/pip
    if "JPY" in pair:
        dollars_per_pip = (units / 100) / 100  # JPY pairs
    else:
        dollars_per_pip = units / 10000

    print(f"{pair:<10} {price:<8.5f} {units:<10,} ${position_value:<11,.2f} ${dollars_per_pip:<7.2f} {pct_of_account:<10.1f}%")

print("\n" + "=" * 70)
print("RISK PER TRADE (with new position sizes):")
print("=" * 70)
print(f"\n{'Pair':<10} {'TP Pips':<8} {'SL Pips':<8} {'TP $':<10} {'SL $':<10}")
print("-" * 70)

targets = {
    "EUR/USD": (12, 8),
    "GBP/USD": (20, 12),
    "USD/JPY": (12, 8),
    "USD/CHF": (10, 6),
    "AUD/USD": (12, 8),
    "USD/CAD": (12, 8),
    "NZD/USD": (12, 8),
}

for pair, price in pairs:
    max_position_value = balance * leverage_cap
    units = int(max_position_value / price)
    units = (units // 1000) * 1000
    units = max(units, 1000)

    if "JPY" in pair:
        dollars_per_pip = (units / 100) / 100
    else:
        dollars_per_pip = units / 10000

    tp_pips, sl_pips = targets.get(pair, (12, 8))
    tp_dollars = tp_pips * dollars_per_pip
    sl_dollars = sl_pips * dollars_per_pip

    print(f"{pair:<10} {tp_pips:<8} {sl_pips:<8} ${tp_dollars:<9.2f} ${sl_dollars:<9.2f}")

print("\n" + "=" * 70)
