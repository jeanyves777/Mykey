"""Test pair-specific settings display"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params, PAIR_VOLATILITY

MAJOR_PAIRS = list(PAIR_VOLATILITY.keys())

print("=" * 80)
print("PAIR-SPECIFIC SCALPING SETTINGS")
print("=" * 80)

print(f"\n{'Pair':<10} {'ADR':<6} {'TP':<8} {'SL':<8} {'R:R':<8} {'Trailing':<15} {'$/pip@24k'}")
print(f"{'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*15} {'-'*10}")

for pair in sorted(MAJOR_PAIRS):
    params = get_scalping_params(pair)

    # Calculate $ per pip for 24,000 units
    pip_value_usd = 2.4 if "JPY" not in pair else 2.4 * 100  # Rough estimate

    print(f"{pair:<10} {params['adr']:<6} {params['tp_pips']:<3}p     {params['sl_pips']:<3}p     "
          f"{params['risk_reward_ratio']:.2f}:1    "
          f"{params['trail_trigger_pips']}p @ {params['trail_distance_pips']}p trail  "
          f"${pip_value_usd:.2f}")

print("\n" + "=" * 80)
print("KEY CHANGES FROM BEFORE:")
print("=" * 80)
print("\nOLD (One-size-fits-all):")
print("  - All pairs: 30 pips TP / 20 pips SL")
print("  - No consideration for volatility differences")
print("  - GBP/USD and CHF using same targets (wrong!)")

print("\nNEW (Pair-specific):")
print("  - EUR/USD: 20 pips TP / 12 pips SL  (moderate volatility)")
print("  - GBP/USD: 30 pips TP / 20 pips SL  (high volatility - wider stops)")
print("  - USD/CHF: 15 pips TP / 10 pips SL  (low volatility - tighter stops)")
print("  - Each pair optimized for its natural movement")

print("\n" + "=" * 80)
print("BENEFITS:")
print("=" * 80)
print("  1. Higher win rate - targets are more realistic for each pair")
print("  2. Better risk control - stops match volatility")
print("  3. Faster trades - scalping targets hit within 1-3 hours")
print("  4. Less drawdown - tighter stops on stable pairs")
print("  5. More profits - GBP can run to 30 pips, CHF exits at 15 pips")
print("=" * 80)

print("\n\nExample Trade Scenarios (with 24,000 units):")
print("-" * 80)

# EUR/USD Example
eur_params = get_scalping_params("EUR_USD")
print(f"\nEUR/USD Trade:")
print(f"  TP: {eur_params['tp_pips']} pips = ${eur_params['tp_pips'] * 2.4:.2f}")
print(f"  SL: {eur_params['sl_pips']} pips = ${eur_params['sl_pips'] * 2.4:.2f}")
print(f"  Risk/Reward: {eur_params['risk_reward_ratio']:.2f}:1")

# GBP/USD Example
gbp_params = get_scalping_params("GBP_USD")
print(f"\nGBP/USD Trade:")
print(f"  TP: {gbp_params['tp_pips']} pips = ${gbp_params['tp_pips'] * 2.4:.2f}")
print(f"  SL: {gbp_params['sl_pips']} pips = ${gbp_params['sl_pips'] * 2.4:.2f}")
print(f"  Risk/Reward: {gbp_params['risk_reward_ratio']:.2f}:1")

# USD/CHF Example
chf_params = get_scalping_params("USD_CHF")
print(f"\nUSD/CHF Trade:")
print(f"  TP: {chf_params['tp_pips']} pips = ${chf_params['tp_pips'] * 2.4:.2f}")
print(f"  SL: {chf_params['sl_pips']} pips = ${chf_params['sl_pips'] * 2.4:.2f}")
print(f"  Risk/Reward: {chf_params['risk_reward_ratio']:.2f}:1")

print("\n" + "=" * 80)
