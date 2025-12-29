"""Test MARGIN-BASED position sizing with OANDA leverage"""

balance = 4906.97
leverage = 20  # OANDA leverage for major pairs

pairs = [
    ("EUR_USD", 1.17),
    ("GBP_USD", 1.34),
    ("USD_JPY", 155.50),
    ("USD_CHF", 0.795),
    ("AUD_USD", 0.667),
    ("USD_CAD", 1.37),
    ("NZD_USD", 0.625),
]

print("MARGIN-BASED POSITION SIZING - ACCOUNTING FOR OANDA LEVERAGE")
print("=" * 95)
print(f"Account Balance: ${balance:,.2f}")
print(f"OANDA Leverage: {leverage}:1 (5% margin requirement)")
print(f"Target Margin: 15% of balance = ${balance * 0.15:,.2f} per position\n")

print(f"{'Pair':<12} {'Price':<10} {'Units':<12} {'Notional $':<14} {'Margin $':<12} {'% of Acct':<12}")
print("-" * 95)

for pair, price in pairs:
    # NEW: Calculate based on MARGIN usage
    target_margin = balance * 0.15
    target_notional = target_margin * leverage
    units = int(target_notional / price)

    # Actual values
    notional_value = units * price
    actual_margin = notional_value / leverage
    margin_pct = (actual_margin / balance) * 100

    print(f"{pair:<12} {price:<10.5f} {units:<12,} ${notional_value:<13,.2f} ${actual_margin:<11,.2f} {margin_pct:<11.2f}%")

print("\n" + "=" * 95)
print("COMPARISON:")
print("=" * 95)

print("\nOLD CALCULATION (No Leverage):")
print(f"  EUR_USD: 626 units = $732 = 1 margin used = WRONG ❌")
print(f"  GBP_USD: 549 units = $735 = $37 margin used = WRONG ❌")

print("\nNEW CALCULATION (With 20:1 Leverage):")
old_units_eur = int((balance * 0.15) / 1.17)
old_units_gbp = int((balance * 0.15) / 1.34)

new_units_eur = int((balance * 0.15 * leverage) / 1.17)
new_units_gbp = int((balance * 0.15 * leverage) / 1.34)

print(f"  EUR_USD: {new_units_eur:,} units = ${new_units_eur * 1.17:,.2f} notional = ${(new_units_eur * 1.17)/leverage:,.2f} margin ✓")
print(f"  GBP_USD: {new_units_gbp:,} units = ${new_units_gbp * 1.34:,.2f} notional = ${(new_units_gbp * 1.34)/leverage:,.2f} margin ✓")

print("\n" + "=" * 95)
print("All pairs now use ~15% of account as MARGIN (not notional value)")
print("With 5 positions max = 75% total margin usage")
print("=" * 95)
