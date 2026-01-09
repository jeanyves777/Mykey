"""Check what's wrong with AUD_USD leverage"""

balance = 4906.97  # Current balance
price = 0.66703    # AUD/USD price
units = 1000       # What was actually used

pos_value = units * price
leverage_pct = (pos_value / balance) * 100

print(f"CURRENT SITUATION:")
print(f"  Balance: ${balance:.2f}")
print(f"  Price: {price}")
print(f"  Units: {units:,}")
print(f"  Position Value: ${pos_value:.2f}")
print(f"  Leverage Used: {leverage_pct:.2f}%")
print()

print(f"TARGET (15% leverage):")
target_pos_value = balance * 0.15
target_units = int(target_pos_value / price)
print(f"  Should be: ${target_pos_value:.2f} position value")
print(f"  Which needs: {target_units:,} units")
print(f"  Actual leverage: {(target_units * price / balance) * 100:.2f}%")
print()

print(f"DIFFERENCE:")
print(f"  Using {units:,} units instead of {target_units:,} units")
print(f"  That's {((target_units - units) / target_units) * 100:.0f}% less than target!")
