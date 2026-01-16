"""
Asian Session Portfolio Scalper Configuration
=============================================
Trades 15 currency pairs simultaneously during Asian session
Portfolio-level profit target with individual pair management
Uses OANDA Account: 101-001-8364309-002
"""

import os

# =============================================================================
# OANDA API CONFIGURATION - DEDICATED ACCOUNT
# =============================================================================
OANDA_CONFIG = {
    # API Key from OANDA
    "api_key": os.getenv("OANDA_API_KEY", "cf70f63218d3886203a14d1d80bdf54a-5d6a2e9817818954b37b2c253aa9d685"),

    # DEDICATED ACCOUNT for Asian Scalper
    "account_id": "101-001-8364309-002",

    # Environment
    "practice": True,

    # API URLs
    "practice_url": "https://api-fxpractice.oanda.com",
    "live_url": "https://api-fxtrade.oanda.com",

    # Stream URLs
    "practice_stream_url": "https://stream-fxpractice.oanda.com",
    "live_stream_url": "https://stream-fxtrade.oanda.com",
}

# =============================================================================
# 15 CURRENCY PAIRS FOR PORTFOLIO SCALPING
# =============================================================================
SCALP_PAIRS = [
    # Major Pairs (4) - removed USD_JPY, USD_CAD, USD_CHF (big losers)
    "EUR_USD",
    "GBP_USD",
    "AUD_USD",
    "NZD_USD",

    # Cross Pairs (7) - removed GBP_AUD (big loser)
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
    "EUR_AUD",
    "AUD_JPY",
    "EUR_CAD",
    "GBP_CAD",
]

# =============================================================================
# PAIR-SPECIFIC SETTINGS (pip location for price calculations)
# =============================================================================
PAIR_SETTINGS = {
    # Standard pairs (4 decimal places, 0.0001 = 1 pip)
    "EUR_USD": {"pip_location": -4, "min_units": 1000},
    "GBP_USD": {"pip_location": -4, "min_units": 1000},
    "AUD_USD": {"pip_location": -4, "min_units": 1000},
    "USD_CAD": {"pip_location": -4, "min_units": 1000},
    "NZD_USD": {"pip_location": -4, "min_units": 1000},
    "USD_CHF": {"pip_location": -4, "min_units": 1000},
    "EUR_GBP": {"pip_location": -4, "min_units": 1000},
    "EUR_AUD": {"pip_location": -4, "min_units": 1000},
    "GBP_AUD": {"pip_location": -4, "min_units": 1000},
    "EUR_CAD": {"pip_location": -4, "min_units": 1000},
    "GBP_CAD": {"pip_location": -4, "min_units": 1000},

    # JPY pairs (2 decimal places, 0.01 = 1 pip)
    "USD_JPY": {"pip_location": -2, "min_units": 1000},
    "EUR_JPY": {"pip_location": -2, "min_units": 1000},
    "GBP_JPY": {"pip_location": -2, "min_units": 1000},
    "AUD_JPY": {"pip_location": -2, "min_units": 1000},
}

# =============================================================================
# PORTFOLIO SCALPING STRATEGY SETTINGS
# =============================================================================
SCALP_CONFIG = {
    # Portfolio target - Close ALL positions when total unrealized P/L reaches this
    "portfolio_target": 30.0,  # $30 total profit target

    # Individual pair settings
    "individual_tp_pips": 6,   # 6 pips take profit per pair (increased from 3 to account for spread)
    "individual_sl_pips": 30,  # 30 pips stop loss (wide SL)

    # Position sizing
    "units_per_pair": 5000,    # 5000 units per pair (0.05 lots)

    # Entry timing
    "entry_mode": "ALL_AT_ONCE",  # Open all 15 pairs simultaneously

    # Spread filter - skip pairs with spread > this value in pips
    "max_spread_pips": 2.0,    # Only trade pairs with spread <= 2 pips

    # Direction determination - NOW PER-PAIR (smarter direction)
    "use_per_pair_direction": True,  # Each pair gets its own direction based on EMA
    "ema_fast": 5,    # 5-period EMA
    "ema_slow": 10,   # 10-period EMA
    "timeframe": "M5",  # 5-minute chart for direction

    # Force entry even without clear trend
    "force_entry": True,  # Trade even if direction unclear
    "fallback_direction": "LONG",  # Default to LONG if no clear signal

    # DCA (Dollar Cost Averaging) for recovery
    "dca_enabled": True,
    "dca_trigger_pl": -15.0,    # Trigger DCA when portfolio unrealized P/L hits -$15
    "dca_units": 5000,          # Same as entry size
    "dca_max_per_pair": 1,      # Max 1 DCA per pair

    # Recovery exit - close when recovered 50% from DCA point
    "recovery_exit_enabled": True,
    "recovery_exit_pl": -15.0,  # Close all when recovered to -$15 (50% of -$30)
}

# =============================================================================
# SESSION TIMING (UTC)
# =============================================================================
# Forex market opens Sunday 5pm EST (winter) / 6pm EST (summer)
# 5pm EST = 22:00 UTC (winter, no DST)
# 6pm EST = 22:00 UTC (summer, with DST)
SESSION_CONFIG = {
    "asian_start_hour": 22,  # 22:00 UTC (5pm EST winter / 6pm EST summer)
    "asian_end_hour": 8,     # 08:00 UTC (London open)

    # Extended for Tokyo/London overlap
    "london_overlap_start": 7,  # 07:00 UTC
    "london_overlap_end": 9,    # 09:00 UTC

    # Trade window - when to open positions
    "trade_window_start": 22,   # Start trading at 22:00 UTC (5pm EST)
    "trade_window_end": 6,      # Stop new entries by 06:00 UTC

    # Check interval
    "check_interval_seconds": 10,  # Check portfolio P/L every 10 seconds
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Get the directory where this config file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

LOGGING_CONFIG = {
    "log_dir": os.path.join(_CONFIG_DIR, "logs"),
    "data_dir": os.path.join(_CONFIG_DIR, "data"),
    "trade_log_file": "asian_scalper_trades.json",
    "session_log_file": "asian_scalper_sessions.json",
    "console_output": True,
    "file_output": True,
}


def get_pair_pip_value(pair: str) -> float:
    """Get pip value multiplier for a pair."""
    settings = PAIR_SETTINGS.get(pair, {"pip_location": -4})
    return 10 ** settings["pip_location"]


def get_pip_location(pair: str) -> int:
    """Get pip location for a pair."""
    return PAIR_SETTINGS.get(pair, {"pip_location": -4})["pip_location"]
