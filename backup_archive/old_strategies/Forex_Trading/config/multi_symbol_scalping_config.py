"""
Multi-Symbol Forex Scalping Configuration
Trades all major forex pairs simultaneously
"""

# Major forex pairs to trade
MAJOR_PAIRS = [
    "EUR_USD",  # Euro / US Dollar
    "GBP_USD",  # British Pound / US Dollar
    "USD_JPY",  # US Dollar / Japanese Yen
    "USD_CHF",  # US Dollar / Swiss Franc
    "AUD_USD",  # Australian Dollar / US Dollar
    "USD_CAD",  # US Dollar / Canadian Dollar
    "NZD_USD",  # New Zealand Dollar / US Dollar
]

# Multi-symbol strategy parameters
MULTI_SYMBOL_STRATEGY_CONFIG = {
    "instruments": MAJOR_PAIRS,
    "max_trades_per_day_per_symbol": 3,  # 3 trades per symbol = 21 trades/day total
    "max_concurrent_positions": 5,  # Max 5 positions open at once
    "daily_profit_target": 0.05,  # 5% daily profit target (across all pairs)
    "trade_size_pct": 0.05,  # 5% per trade (since we have multiple positions)
    "take_profit_pct": 0.003,  # 30 pips TP
    "stop_loss_pct": 0.002,  # 20 pips SL
    "trailing_stop_trigger": 0.002,  # Start trailing at 20 pips
    "trailing_stop_distance": 0.001,  # Trail 10 pips behind

    # Filters
    "require_htf_strict": True,  # HTF must align for quality
    "pullback_required": False,  # Don't wait for pullback
    "min_consensus_score": 1,  # Only need 1 signal
}

# Risk management
MULTI_SYMBOL_RISK_CONFIG = {
    "max_position_size_pct": 0.10,  # Max 10% per position
    "max_daily_loss_pct": 0.05,  # Stop at -5% daily loss
    "max_total_exposure_pct": 0.25,  # Max 25% total exposure across all positions
    "max_correlated_positions": 2,  # Max 2 correlated pairs (EUR_USD + GBP_USD)
}

# Pair correlations (for risk management)
PAIR_CORRELATIONS = {
    "EUR_USD": ["GBP_USD", "AUD_USD", "NZD_USD"],  # Positively correlated
    "GBP_USD": ["EUR_USD", "AUD_USD", "NZD_USD"],
    "USD_JPY": ["USD_CHF", "USD_CAD"],  # USD pairs
    "USD_CHF": ["USD_JPY", "USD_CAD"],
    "USD_CAD": ["USD_JPY", "USD_CHF"],
    "AUD_USD": ["EUR_USD", "GBP_USD", "NZD_USD"],  # Commodity currencies
    "NZD_USD": ["EUR_USD", "GBP_USD", "AUD_USD"],
}
