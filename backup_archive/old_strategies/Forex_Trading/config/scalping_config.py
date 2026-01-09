"""
SCALPING STRATEGY - PAPER TRADING CONFIG
=========================================

5-15 pip scalping with multi-indicator validation
NO OVERTRADING - strict entry requirements

Strategy: Multi-indicator validation (5+ checks)
- Trend alignment (EMA9>EMA21>EMA50)
- Momentum confirmation (RSI 40-60)
- MACD trigger (crossover + histogram)
- Candle confirmation (strong body)
- Volatility filter (ATR range)

Target: 75%+ win rate with quick scalps
"""

from pathlib import Path

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================
ACCOUNT_TYPE = 'practice'  # OANDA practice account
LOG_DIR = 'forex_scalping_logs'  # Log directory

# =============================================================================
# TRADING PARAMETERS - CONSERVATIVE FOR SCALPING
# =============================================================================
MAX_CONCURRENT_POSITIONS = 5  # REDUCED to avoid overtrading
TIMEFRAME = 'M1'  # 1-minute candles
CANDLE_COUNT = 200  # More data for better indicator calculation

# STRICT COOLDOWN to prevent overtrading
COOLDOWN_MINUTES = 30  # Wait 30 minutes after each trade

# =============================================================================
# SESSION FILTERS - LONDON & NY OVERLAP (HIGH VOLUME)
# Only trade during high-volume hours to avoid low-liquidity whipsaws
# =============================================================================
TRADING_SESSIONS = {
    # All pairs use same high-volume session filter
    'EUR_USD': {'allowed_hours': list(range(12, 20)), 'session_filter': True},  # 12:00-20:00 UTC
    'GBP_USD': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'USD_JPY': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'USD_CHF': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'USD_CAD': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'AUD_USD': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'NZD_USD': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'EUR_GBP': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'EUR_JPY': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
    'GBP_JPY': {'allowed_hours': list(range(12, 20)), 'session_filter': True},
}

# =============================================================================
# PAIR-SPECIFIC SETTINGS - SCALPING OPTIMIZED
# Start with 5 pip TP, adjust based on backtest results
# =============================================================================
PAIR_SETTINGS = {
    'EUR_USD': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 5,   # Ultra-tight scalping
        'sl_pips': 15,  # 1:3 risk/reward
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'GBP_USD': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 8,   # Slightly wider (more volatile)
        'sl_pips': 20,  # 1:2.5 risk/reward
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'USD_JPY': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 5,
        'sl_pips': 15,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'USD_CHF': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 6,
        'sl_pips': 18,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'USD_CAD': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 6,
        'sl_pips': 18,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'AUD_USD': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 5,
        'sl_pips': 15,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'NZD_USD': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 5,
        'sl_pips': 15,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'EUR_GBP': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 4,   # Lower volatility pair
        'sl_pips': 12,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'EUR_JPY': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 8,   # Higher volatility
        'sl_pips': 20,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
    'GBP_JPY': {
        'strategy': 'MULTI_INDICATOR_SCALP',
        'tp_pips': 10,  # Highest volatility
        'sl_pips': 25,
        'cooldown_minutes': 30,
        'expected_wr': 75.0,
    },
}

# List of pairs to trade
SCALPING_PAIRS = list(PAIR_SETTINGS.keys())

# =============================================================================
# RISK MANAGEMENT - CONSERVATIVE
# =============================================================================
POSITION_SIZE_UNITS = 1000  # $1 per pip with 100:1 leverage
MAX_DAILY_LOSS_PIPS = 50    # Stop trading if down 50 pips in a day
MAX_DAILY_TRADES = 15       # Maximum trades per day (prevent overtrading)

# =============================================================================
# ANTI-OVERTRADING FILTERS
# =============================================================================
MIN_TIME_BETWEEN_TRADES = 30  # Minimum 30 minutes between trades (same as cooldown)
MAX_CONSECUTIVE_LOSSES = 3    # Stop trading after 3 consecutive losses
REQUIRE_ALL_INDICATORS = True  # Must pass ALL 5 indicator checks

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_pair_settings(instrument: str) -> dict:
    """Get settings for a specific pair."""
    return PAIR_SETTINGS.get(instrument, PAIR_SETTINGS['EUR_USD'])


def is_allowed_hour(instrument: str, hour_utc: int) -> bool:
    """Check if trading is allowed at this hour for the given instrument."""
    session = TRADING_SESSIONS.get(instrument, {})
    
    if not session.get('session_filter', False):
        return True  # No filter, always allowed
    
    allowed_hours = session.get('allowed_hours', list(range(0, 24)))
    return hour_utc in allowed_hours


def print_config_info():
    """Print configuration information."""
    print("=" * 70)
    print("SCALPING STRATEGY CONFIG - MULTI-INDICATOR VALIDATION")
    print("=" * 70)
    print(f"\nAccount: {ACCOUNT_TYPE}")
    print(f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"Cooldown: {COOLDOWN_MINUTES} minutes")
    print(f"Max Daily Trades: {MAX_DAILY_TRADES}")
    print(f"Max Daily Loss: {MAX_DAILY_LOSS_PIPS} pips")
    print(f"Session Filter: 12:00-20:00 UTC (London/NY overlap)")
    print()
    print("Strategy Requirements (ALL must pass):")
    print("  1. Trend Alignment: EMA9 > EMA21 > EMA50")
    print("  2. Momentum: RSI in 40-60 range")
    print("  3. MACD Trigger: Crossover + histogram momentum")
    print("  4. Candle: Strong body (>40% of range)")
    print("  5. Volatility: ATR in acceptable range")
    print()
    print(f"\n{'PAIR':<10} {'TP':<6} {'SL':<6} {'RISK:REWARD':<12} {'COOLDOWN'}")
    print("-" * 50)
    for pair in SCALPING_PAIRS:
        s = PAIR_SETTINGS[pair]
        rr = f"1:{s['sl_pips']/s['tp_pips']:.1f}"
        print(f"{pair:<10} {s['tp_pips']:<6} {s['sl_pips']:<6} {rr:<12} {s['cooldown_minutes']}m")
    print("-" * 50)
    print("=" * 70)


if __name__ == '__main__':
    print_config_info()
