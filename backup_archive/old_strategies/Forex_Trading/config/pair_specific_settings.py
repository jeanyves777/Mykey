"""
Pair-Specific Trading Settings
Each forex pair has different volatility and typical daily ranges
Optimized for SCALPING with realistic pip targets
"""

# Average Daily Range (ADR) for major pairs (in pips)
# Based on typical London/NY session volatility
# HIGH_VOLATILITY pairs need extra validation before entry
PAIR_VOLATILITY = {
    "EUR_USD": {
        "adr": 80,           # Average daily range: 80 pips
        "description": "Most liquid, moderate volatility",
        "high_volatility": False
    },
    "GBP_USD": {
        "adr": 120,          # Cable moves more - higher volatility
        "description": "High volatility, larger moves",
        "high_volatility": True  # VOLATILE - needs extra validation
    },
    "USD_JPY": {
        "adr": 90,           # JPY pairs can spike on news
        "description": "High volatility, news-sensitive",
        "high_volatility": True  # VOLATILE - needs extra validation
    },
    "USD_CHF": {
        "adr": 60,           # Swiss franc - lower volatility
        "description": "Lower volatility, safe haven",
        "high_volatility": False
    },
    "AUD_USD": {
        "adr": 85,           # Aussie - commodity currency, can spike
        "description": "High volatility, commodity-linked",
        "high_volatility": True  # VOLATILE - needs extra validation
    },
    "USD_CAD": {
        "adr": 70,           # Loonie - oil-linked
        "description": "Moderate volatility, oil-linked",
        "high_volatility": False
    },
    "NZD_USD": {
        "adr": 85,           # Kiwi - can move fast
        "description": "High volatility, commodity-linked",
        "high_volatility": True  # VOLATILE - needs extra validation
    }
}

def get_scalping_params(pair: str, account_balance: float = 1000) -> dict:
    """
    Get optimized scalping parameters for each pair

    Scalping philosophy:
    - TP should be 25-35% of ADR (achievable within hours)
    - SL should be 15-20% of ADR (tight risk control)
    - Trailing should activate at 75-80% of TP (let trades develop!)
    - Trail distance should be 40-50% of TP (give room to breathe)
    - Risk/Reward ratio: 1.5:1 to 2:1

    Args:
        pair: Currency pair (e.g., "EUR_USD")
        account_balance: Account size for position sizing

    Returns:
        Dictionary with TP, SL, trailing settings in pips and percentages
    """

    # Default to EUR_USD if pair not found
    if pair not in PAIR_VOLATILITY:
        pair = "EUR_USD"

    adr = PAIR_VOLATILITY[pair]["adr"]

    # REVISED SCALPING TARGETS - WIDER STOPS
    # TP = 20-30% of ADR (realistic exits, 1-3hr holds)
    # SL = 20-25% of ADR (wider stops, let good signals develop)
    # Goal: Avoid premature stop-outs, let winners run

    # =================================================================
    # HIGH VOLATILITY PAIRS - GBP, JPY, AUD, NZD
    # These pairs need MUCH wider trailing stops to avoid premature exits
    # Trail trigger at 90% of TP, trail distance = 50% of TP
    # =================================================================

    if pair == "GBP_USD":
        # GBP/USD - HIGHEST volatility, needs widest stops
        tp_pips = 35      # 29% of 120 pip ADR (increased from 30)
        sl_pips = 28      # 23% of 120 pip ADR (increased from 25)
        trail_trigger = 32  # Activate at 91% of TP (almost at TP before trailing)
        trail_distance = 18  # 50% of TP - very wide trail (was 12)

    elif pair == "USD_JPY":
        # USD/JPY - Can spike hard on news, needs room
        tp_pips = 25      # 28% of 90 pip ADR (increased from 18)
        sl_pips = 20      # 22% of 90 pip ADR (increased from 15)
        trail_trigger = 22  # Activate at 88% of TP
        trail_distance = 12  # 48% of TP - wide trail (was 8)

    elif pair == "AUD_USD":
        # AUD/USD - Commodity currency, spiky
        tp_pips = 25      # 29% of 85 pip ADR (increased from 20)
        sl_pips = 20      # 24% of 85 pip ADR (increased from 16)
        trail_trigger = 22  # Activate at 88% of TP
        trail_distance = 12  # 48% of TP - wide trail (was 8)

    elif pair == "NZD_USD":
        # NZD/USD - Similar to AUD, very spiky
        tp_pips = 25      # 29% of 85 pip ADR (increased from 20)
        sl_pips = 20      # 24% of 85 pip ADR (increased from 16)
        trail_trigger = 22  # Activate at 88% of TP
        trail_distance = 12  # 48% of TP - wide trail (was 8)

    # =================================================================
    # MODERATE VOLATILITY PAIRS - EUR, CHF, CAD
    # Standard settings with 80% trail trigger
    # =================================================================

    elif pair == "EUR_USD":
        # EUR/USD - Most liquid, predictable
        tp_pips = 20      # 25% of 80 pip ADR
        sl_pips = 16      # 20% of 80 pip ADR
        trail_trigger = 16  # Activate at 80% of TP
        trail_distance = 8   # 40% of TP

    elif pair == "USD_CHF":
        # USD/CHF - Lower volatility, safe haven
        tp_pips = 15      # 25% of 60 pip ADR
        sl_pips = 12      # 20% of 60 pip ADR
        trail_trigger = 12  # Activate at 80% of TP
        trail_distance = 6   # 40% of TP

    elif pair == "USD_CAD":
        # USD/CAD - Oil-linked but more stable than AUD/NZD
        tp_pips = 18      # 26% of 70 pip ADR
        sl_pips = 14      # 20% of 70 pip ADR
        trail_trigger = 14  # Activate at 78% of TP
        trail_distance = 7   # 39% of TP

    else:
        # Default - moderate settings
        tp_pips = 20
        sl_pips = 16
        trail_trigger = 16  # Activate at 80% of TP
        trail_distance = 8   # 40% of TP

    # Convert pips to absolute price movement (NOT percentage - that requires price)
    # For non-JPY pairs: 1 pip = 0.0001
    # For JPY pairs: 1 pip = 0.01

    is_jpy_pair = "JPY" in pair
    pip_value = 0.01 if is_jpy_pair else 0.0001

    # These are ABSOLUTE PRICE MOVEMENTS, not percentages
    # Example: EUR/USD 20 pips = 0.0020, USD/JPY 18 pips = 0.18
    tp_absolute = (tp_pips * pip_value)
    sl_absolute = (sl_pips * pip_value)
    trail_trigger_absolute = (trail_trigger * pip_value)
    trail_distance_absolute = (trail_distance * pip_value)

    # Calculate Risk/Reward ratio
    risk_reward = tp_pips / sl_pips

    # Get high_volatility flag
    is_high_volatility = PAIR_VOLATILITY[pair].get("high_volatility", False)

    return {
        "pair": pair,
        "adr": adr,
        "high_volatility": is_high_volatility,  # Flag for extra validation

        # Pip values (for display)
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
        "trail_trigger_pips": trail_trigger,
        "trail_distance_pips": trail_distance,

        # Absolute price movements (for trading - add/subtract from price)
        "take_profit_pct": tp_absolute,
        "stop_loss_pct": sl_absolute,
        "trailing_stop_trigger": trail_trigger_absolute,
        "trailing_stop_distance": trail_distance_absolute,

        # Risk management
        "risk_reward_ratio": risk_reward,
        "position_size_pct": 0.05,  # 5% per trade

        # Expected values (if win rate = 55%)
        "expected_win": tp_pips * 0.55,
        "expected_loss": sl_pips * 0.45,
        "expected_pips_per_trade": (tp_pips * 0.55) - (sl_pips * 0.45)
    }


def get_all_pair_settings():
    """Get settings for all pairs for comparison"""
    settings = {}
    for pair in PAIR_VOLATILITY.keys():
        settings[pair] = get_scalping_params(pair)
    return settings


def print_pair_comparison():
    """Print comparison table of all pair settings"""
    print("=" * 100)
    print("SCALPING SETTINGS BY PAIR (Optimized for realistic targets)")
    print("=" * 100)
    print(f"\n{'Pair':<10} {'ADR':<6} {'TP':<8} {'SL':<8} {'R:R':<6} {'Trail':<8} {'Expected':<12}")
    print(f"{'':10} {'(pips)':<6} {'(pips)':<8} {'(pips)':<8} {'Ratio':<6} {'(pips)':<8} {'(pips/trade)':<12}")
    print("-" * 100)

    for pair in sorted(PAIR_VOLATILITY.keys()):
        params = get_scalping_params(pair)
        print(f"{pair:<10} {params['adr']:<6} {params['tp_pips']:<8} {params['sl_pips']:<8} "
              f"{params['risk_reward_ratio']:.2f}{'':4} {params['trail_trigger_pips']:<8} "
              f"{params['expected_pips_per_trade']:+.1f}")

    print("=" * 100)
    print("\nNotes:")
    print("  - TP/SL optimized for each pair's volatility")
    print("  - Tighter stops for lower volatility pairs (CHF)")
    print("  - Wider stops for higher volatility pairs (GBP)")
    print("  - Risk/Reward ratio maintained between 1.5:1 and 2.0:1")
    print("  - Expected pips assumes 55% win rate")
    print("=" * 100)


if __name__ == "__main__":
    # Test the settings
    print_pair_comparison()

    print("\n\nExample: EUR/USD Settings Detail:")
    params = get_scalping_params("EUR_USD")
    print(f"  Take Profit: {params['tp_pips']} pips ({params['take_profit_pct']:.5f})")
    print(f"  Stop Loss: {params['sl_pips']} pips ({params['stop_loss_pct']:.5f})")
    print(f"  Trailing Trigger: {params['trail_trigger_pips']} pips")
    print(f"  Trailing Distance: {params['trail_distance_pips']} pips")
    print(f"  Risk/Reward: {params['risk_reward_ratio']:.2f}:1")
