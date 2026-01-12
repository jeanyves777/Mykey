"""
OANDA Forex Trading Configuration
==================================
HTF Confluence Strategy for Forex
Based on proven Binance Futures strategy
"""

import os

# =============================================================================
# 1. OANDA API CONFIGURATION
# =============================================================================
OANDA_CONFIG = {
    # API Key from OANDA
    "api_key": os.getenv("OANDA_API_KEY", "cf70f63218d3886203a14d1d80bdf54a-5d6a2e9817818954b37b2c253aa9d685"),
    
    # Account ID (MT4_USD_Practice)
    "account_id": os.getenv("OANDA_ACCOUNT_ID", "101-001-8364309-001"),
    
    # Environment
    "practice": True,  # True = demo account, False = live account
    
    # API URLs
    "practice_url": "https://api-fxpractice.oanda.com",
    "live_url": "https://api-fxtrade.oanda.com",
    
    # Stream URLs
    "practice_stream_url": "https://stream-fxpractice.oanda.com",
    "live_stream_url": "https://stream-fxtrade.oanda.com",
}

# =============================================================================
# 2. FOREX SYMBOLS - MAJOR PAIRS (Best liquidity & spreads)
# =============================================================================
# Tier 1: Major Pairs (Tightest spreads, 24/5 liquidity)
FOREX_SYMBOLS_TIER1 = [
    "EUR_USD",   # Euro/US Dollar - Most liquid pair, ~0.1 pip spread
    "GBP_USD",   # British Pound/US Dollar - Cable, volatile
    "USD_JPY",   # US Dollar/Japanese Yen
]

# Tier 2: Major Crosses (Good liquidity)
FOREX_SYMBOLS_TIER2 = [
    "AUD_USD",   # Australian Dollar/US Dollar - Commodity currency
    "USD_CAD",   # US Dollar/Canadian Dollar
    "NZD_USD",   # New Zealand Dollar/US Dollar
]

# Tier 3: Cross Pairs (More volatile)
FOREX_SYMBOLS_TIER3 = [
    # "EUR_GBP",   # DISABLED (optimization: -$26,950, 51.8% WR, 20 consecutive losses)
    # "EUR_JPY",   # DISABLED (optimization: -$13,585, 40.5% WR)
    # "GBP_JPY",   # DISABLED (optimization: -$24,870, 46.7% WR)
]

# Active trading symbols - ALL SYMBOLS ENABLED
FOREX_SYMBOLS_LIVE = FOREX_SYMBOLS_TIER1 + FOREX_SYMBOLS_TIER2 + FOREX_SYMBOLS_TIER3

# All symbols for demo/testing
FOREX_SYMBOLS_DEMO = FOREX_SYMBOLS_TIER1 + FOREX_SYMBOLS_TIER2 + FOREX_SYMBOLS_TIER3

# Default symbols - ALL PAIRS
FOREX_SYMBOLS = FOREX_SYMBOLS_LIVE

# =============================================================================
# 3. SESSION TRADING CONTROL
# =============================================================================
# Session Performance Analysis Results:
# - ASIAN (00:00-08:00 UTC): +$70,127 profit, 63.9% WR ✅ EXCELLENT  
# - LONDON (08:00-17:00 UTC): -$42,634 loss, 49.9% WR ❌ UNPROFITABLE
# - NY (13:00-22:00 UTC): High volatility, re-enabled for testing

SESSION_TRADING = {
    "enabled": True,
    "allowed_sessions": ["ASIAN", "LONDON", "NY"],  # Trade all sessions
    "session_hours": {
        "ASIAN": (0, 8),     # 00:00-08:00 UTC
        "LONDON": (8, 17),   # 08:00-17:00 UTC
        "NY": (13, 22),      # 13:00-22:00 UTC
    }
}

# =============================================================================
# 4. SYMBOL-SPECIFIC SETTINGS
# =============================================================================
# Symbol-specific settings - SCALPING OPTIMIZED (Asset-specific TP/SL based on volatility)
SYMBOL_SETTINGS = {
    # === TIER 1: MAJOR PAIRS (Tightest spreads, best for scalping) ===
    
    "EUR_USD": {
        "pip_location": -4,        # 0.0001 = 1 pip
        "display_precision": 5,    # Show 5 decimals
        "min_trade_size": 1000,    # 1 micro lot
        "tp_pips": 15,            # 15 pips TP (Low volatility, tight spreads)
        "sl_pips": 12,            # 12 pips SL (1.25:1 R:R - favorable for scalping)
        "min_confluence_score": 4, # 5/8 confluence for more signals
        "avg_daily_range": 60,    # Average daily range in pips
        "typical_spread": 0.1,    # Typical spread in pips
    },
    
    "GBP_USD": {
        "pip_location": -4,
        "display_precision": 5,
        "min_trade_size": 1000,
        "tp_pips": 20,            # 20 pips TP (More volatile than EUR)
        "sl_pips": 15,            # 15 pips SL (1.33:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 100,   # More volatile
        "typical_spread": 0.3,
    },
    
    "USD_JPY": {
        "pip_location": -2,        # 0.01 = 1 pip for JPY pairs
        "display_precision": 3,
        "min_trade_size": 1000,
        "tp_pips": 18,            # 18 pips TP (Moderate volatility)
        "sl_pips": 14,            # 14 pips SL (1.29:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 70,
        "typical_spread": 0.2,
    },
    
    # === TIER 2: MAJOR CROSSES (Good liquidity, slightly wider spreads) ===
    
    "AUD_USD": {
        "pip_location": -4,
        "display_precision": 5,
        "min_trade_size": 1000,
        "tp_pips": 18,            # 18 pips TP (Commodity currency, moderate moves)
        "sl_pips": 14,            # 14 pips SL (1.29:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 75,
        "typical_spread": 0.4,
    },
    
    "USD_CAD": {
        "pip_location": -4,
        "display_precision": 5,
        "min_trade_size": 1000,
        "tp_pips": 16,            # 16 pips TP (Oil correlation, steady moves)
        "sl_pips": 13,            # 13 pips SL (1.23:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 65,
        "typical_spread": 0.5,
    },
    
    "NZD_USD": {
        "pip_location": -4,
        "display_precision": 5,
        "min_trade_size": 1000,
        "tp_pips": 17,            # 17 pips TP (Risk-on currency)
        "sl_pips": 13,            # 13 pips SL (1.31:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 70,
        "typical_spread": 0.6,
    },
    
    # === TIER 3: CROSS PAIRS (Higher volatility, wider spreads) ===
    
    "EUR_GBP": {
        "pip_location": -4,
        "display_precision": 5,
        "min_trade_size": 1000,
        "tp_pips": 12,            # 12 pips TP (Low volatility cross)
        "sl_pips": 10,            # 10 pips SL (1.2:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 50,    # Tight range
        "typical_spread": 0.8,
    },
    
    "EUR_JPY": {
        "pip_location": -2,
        "display_precision": 3,
        "min_trade_size": 1000,
        "tp_pips": 25,            # 25 pips TP (Carry trade, more volatile)
        "sl_pips": 18,            # 18 pips SL (1.39:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 90,
        "typical_spread": 1.0,
    },
    
    "GBP_JPY": {
        "pip_location": -2,
        "display_precision": 3,
        "min_trade_size": 1000,
        "tp_pips": 30,            # 30 pips TP (VERY volatile - "The Beast")
        "sl_pips": 22,            # 22 pips SL (1.36:1 R:R)
        "min_confluence_score": 4,
        "avg_daily_range": 120,   # Highest volatility
        "typical_spread": 1.5,
    },
}

# =============================================================================
# 3. STRATEGY CONFIGURATION (Same as Binance)
# =============================================================================
STRATEGY_CONFIG = {
    # Trade management
    "max_trades_per_day": 50,
    "daily_profit_target": 0.03,  # 3% daily target
    
    # Leverage (Forex typically allows up to 50x)
    "leverage": 50,               # 50x leverage (standard for forex)
    "margin_type": "ISOLATED",    # OANDA handles this automatically
    
    # Risk per trade
    "risk_per_trade": 0.02,       # 2% risk per trade
}

# =============================================================================
# 4. HTF CONFLUENCE SETTINGS (Identical to Binance)
# =============================================================================
HTF_CONFLUENCE_CONFIG = {
    # HTF trend detection (1H timeframe)
    "htf_ema_fast": 21,
    "htf_ema_slow": 50,
    
    # LTF entry signals (15m timeframe)
    "ltf_ema_fast": 9,
    "ltf_ema_slow": 21,
    
    # RSI settings (widened for trending markets)
    "rsi_period": 14,
    "rsi_long_min": 35,
    "rsi_long_max": 75,
    "rsi_short_min": 25,
    "rsi_short_max": 65,
    
    # MACD settings
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # ADX trend strength
    "adx_period": 14,
    "adx_threshold": 20,
    
    # Volume confirmation
    "volume_min_ratio": 1.2,
    
    # Confluence requirement (5/8 for scalping, more signals)
    "min_confluence_score": 4,  # 5/8 confluence for more quality signals
}

# =============================================================================
# 5. SMART ENTRY FILTERS (Same as Binance)
# =============================================================================
SMART_FILTERS_CONFIG = {
    # 1m pullback filter
    "use_1m_pullback": True,
    "pullback_tolerance": 0.003,  # 0.3% from 21 EMA
    
    # Candle confirmation
    "use_candle_confirm": True,
    "min_candle_body_ratio": 0.3,
    
    # Volatility filter
    "use_volatility_filter": True,
    "atr_pct_max": 0.35,
    "atr_spike_threshold": 1.5,
    
    # Volume filter
    "use_volume_filter": True,
    "volume_min_ratio": 1.5,
    
    # RSI zone filter
    "use_rsi_zone_filter": True,
    "rsi_entry_min": 30,
    "rsi_entry_max": 70,
}

# =============================================================================
# 6. SMART EXIT SYSTEM - SCALPING OPTIMIZED (Pip-based thresholds)
# =============================================================================
SMART_EXIT_CONFIG = {
    # Trailing profit lock (pip-based for scalping)
    "use_trailing_profit_lock": True,
    "trailing_lock_activation_pips": 8,   # Activate after 8 pips profit
    "trailing_lock_distance_pips": 4,     # Trail 4 pips behind peak
    "trailing_lock_min_floor_pips": 5,    # Never give back more than 5 pips
    
    # Profit lock (early close on trend reversal) - pip-based
    "use_profit_lock": True,
    "profit_lock_min_pips": 8,            # Close if >= 8 pips profit + HTF reverses
    
    # Fakeout protection - scalping optimized
    "use_fakeout_protection": True,
    "reversal_cycle_threshold": 3,        # Faster response for scalping (3 cycles)
    "breakeven_pips_threshold": 6,        # Move to breakeven at 6 pips profit
    "small_profit_exit_pips": 3,          # Exit at 3 pips if weak signals
    "damage_control_pips": -12,           # Close at -12 pips if reversed (less aggressive)
}

# =============================================================================
# 7. TIMEFRAMES (OANDA format)
# =============================================================================
TIMEFRAMES = {
    "entry": "M1",      # 1-minute
    "ltf": "M15",       # 15-minute (entry signals)
    "htf_short": "H1",  # 1-hour (trend confirmation)
    "htf_long": "H4",   # 4-hour (main trend)
    "confirmation": "M5", # 5-minute (confirmation)
}

# Candle counts per timeframe
DATA_CONFIG = {
    "candles_m1": 500,
    "candles_m5": 100,
    "candles_m15": 150,
    "candles_h1": 150,
    "candles_h4": 150,
}

# =============================================================================
# 8. LOGGING CONFIGURATION
# =============================================================================
LOGGING_CONFIG = {
    "log_dir": "logs",
    "trade_log_file": "oanda_trades.json",
    "performance_log_file": "oanda_performance.json",
    "console_output": True,
    "file_output": True,
}

# =============================================================================
# 9. ML DATA LOGGING
# =============================================================================
ML_LOGGING_CONFIG = {
    "enabled": True,
    "log_dir": "ml_logs",
    "signals_file": "signals_log.csv",
    "trades_file": "trades_log.csv",
    "market_snapshots_file": "market_snapshots.csv",
}

# =============================================================================
# 10. SESSION MANAGEMENT
# =============================================================================
SESSION_CONFIG = {
    "check_interval": 60,           # Check every 60 seconds
    "cooldown_minutes": 0,          # No cooldown between trades
    "stats_file": "session_stats.json",
    "positions_file": "tracked_positions.json",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_symbol_config(symbol: str) -> dict:
    """
    Get complete configuration for a specific symbol.
    
    Args:
        symbol: Forex pair (e.g., "EUR_USD")
        
    Returns:
        Dictionary with symbol-specific settings merged with strategy config
    """
    symbol_settings = SYMBOL_SETTINGS.get(symbol, {})
    
    if not symbol_settings:
        # Default settings for unknown symbols
        symbol_settings = {
            "pip_location": -4,
            "display_precision": 5,
            "min_trade_size": 1000,
            "tp_pips": 15,
            "sl_pips": 12,
            "min_confluence_score": 8,
            "avg_daily_range": 60,
            "typical_spread": 0.5,
        }
    
    # Merge with strategy config
    return {
        **symbol_settings,
        "leverage": STRATEGY_CONFIG["leverage"],
        "risk_per_trade": STRATEGY_CONFIG["risk_per_trade"],
    }


def get_all_symbols():
    """Get list of all configured forex symbols."""
    return FOREX_SYMBOLS


def get_symbol_info(symbol: str) -> str:
    """
    Get human-readable info for a symbol.
    
    Args:
        symbol: Forex pair
        
    Returns:
        Formatted string with symbol details
    """
    config = get_symbol_config(symbol)
    pip_loc = config["pip_location"]
    tp = config["tp_pips"]
    sl = config["sl_pips"]
    rr = tp / sl
    spread = config.get("typical_spread", 0)
    daily_range = config.get("avg_daily_range", 0)
    
    return (
        f"{symbol}: TP {tp} pips / SL {sl} pips ({rr:.2f}:1 R:R) | "
        f"Spread ~{spread} pips | Daily range ~{daily_range} pips"
    )


# Convenience exports for common configs
SYMBOL_CONFIG = SYMBOL_SETTINGS  # Alias for backward compatibility
