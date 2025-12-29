"""
OPTIMIZED FOREX STRATEGY - LIVE TRADING CONFIG V2
==================================================
Multi-Stage Optimization Results (100K candles, 14 strategies)
- Total: +10,289 pips across 10 pairs
- Walk-Forward Validated: 9/10 pairs
- Overall Win Rate: 86.0%

Optimization Date: Dec 18, 2025
Data: 100,000 candles per pair (~3 months)

!!! REAL MONEY TRADING - USE WITH CAUTION !!!

This config is used by:
- run_optimized_forex_live.py
"""

from pathlib import Path

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================
ACCOUNT_TYPE = 'live'           # OANDA live account - REAL MONEY
LOG_DIR = 'forex_logs_live'     # Separate log directory for live trading

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
MAX_CONCURRENT_POSITIONS = 10   # One per optimized pair
TIMEFRAME = 'M1'                # 1-minute candles
CANDLE_COUNT = 100              # Number of candles to fetch for analysis
COOLDOWN_MINUTES = 0            # No cooldown - trade when signals appear

# =============================================================================
# SESSION DEFINITIONS (UTC hours)
# =============================================================================
SESSIONS = {
    'ASIAN': list(range(0, 8)),      # 00:00-08:00 UTC
    'LONDON': list(range(8, 16)),    # 08:00-16:00 UTC
    'NEW_YORK': list(range(13, 21)), # 13:00-21:00 UTC
    'ALL': list(range(0, 24)),       # All hours
}

# =============================================================================
# TRADING SESSIONS - OPTIMIZED PER PAIR (from V2 optimization)
# JPY pairs removed - problematic performance
# =============================================================================
TRADING_SESSIONS = {
    'EUR_USD': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'GBP_USD': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'USD_CHF': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'USD_CAD': {'allowed_hours': SESSIONS['ASIAN'], 'session_filter': True},
    'NZD_USD': {'allowed_hours': SESSIONS['ALL'], 'session_filter': False},
    'AUD_CHF': {'allowed_hours': SESSIONS['ALL'], 'session_filter': False},
    'EUR_GBP': {'allowed_hours': SESSIONS['ALL'], 'session_filter': False},
}

# =============================================================================
# PAIR-SPECIFIC SETTINGS - V2 OPTIMIZATION RESULTS
# Results from optimize_14_strategies_v2.py (Dec 18, 2025)
# 7 pairs | JPY pairs removed (problematic)
# !!! REAL MONEY TRADING - USE WITH CAUTION !!!
# =============================================================================
PAIR_SETTINGS = {
    # TOP PERFORMERS (Walk-Forward Validated)
    'EUR_USD': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ASIAN',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 90.0,
        'backtest_pips': 1313,
    },
    'GBP_USD': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ASIAN',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 86.0,
        'backtest_pips': 1021,
    },
    'USD_CHF': {
        'strategy': 'MACD_CROSS',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ASIAN',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 92.7,
        'backtest_pips': 1015,
    },
    'USD_CAD': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ASIAN',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 96.4,
        'backtest_pips': 1279,
    },
    'NZD_USD': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 94.0,
        'backtest_pips': 3429,
    },
    # OPTIMIZED (Dec 18, 2025 - 100K candles)
    'AUD_CHF': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 84.4,
        'backtest_pips': 1004,
    },
    'EUR_GBP': {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'cooldown_minutes': 0,
        'expected_wr': 92.5,
        'backtest_pips': 2009,
    },
}

# List of optimized pairs to trade
OPTIMIZED_PAIRS = list(PAIR_SETTINGS.keys())

# =============================================================================
# PIP MULTIPLIERS (for price calculations)
# =============================================================================
PIP_MULTIPLIERS = {
    'EUR_USD': 10000,
    'GBP_USD': 10000,
    'USD_CHF': 10000,
    'USD_CAD': 10000,
    'NZD_USD': 10000,
    'AUD_JPY': 100,
    'USD_JPY': 100,
    'GBP_JPY': 100,
    'EUR_GBP': 10000,
    'AUD_CHF': 10000,
    'EUR_CAD': 10000,
}

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data")

# CSV file prefixes for each pair
CSV_PREFIXES = {
    'EUR_USD': 'EURUSD',
    'GBP_USD': 'GBPUSD',
    'USD_JPY': 'USDJPY',
    'USD_CHF': 'USDCHF',
    'USD_CAD': 'USDCAD',
    'NZD_USD': 'NZDUSD',
    'AUD_JPY': 'AUDJPY',
    'GBP_JPY': 'GBPJPY',
    'EUR_GBP': 'EURGBP',
    'AUD_CHF': 'AUDCHF',
    'EUR_CAD': 'EURCAD',
}

# =============================================================================
# POSITION SIZING
# =============================================================================
def calculate_position_size(instrument: str, current_price: float) -> int:
    """
    Calculate position size for $1 per pip.

    Args:
        instrument: Currency pair
        current_price: Current market price

    Returns:
        Number of units to trade
    """
    if instrument in ['EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD']:
        return 10000
    elif instrument == 'USD_JPY':
        return int(current_price * 100)  # ~15,300 at 153.00
    elif instrument in ['AUD_JPY', 'EUR_JPY', 'GBP_JPY']:
        return 10000
    elif instrument == 'USD_CHF':
        return int(10000 / current_price)  # ~11,000 at 0.90
    elif instrument == 'USD_CAD':
        return int(10000 * current_price)  # ~14,350 at 1.4350 for $1/pip
    elif instrument == 'EUR_GBP':
        return 12700
    elif instrument == 'AUD_CHF':
        return 11000
    elif instrument == 'EUR_CAD':
        return 14300
    else:
        return 10000


def get_pair_settings(instrument: str) -> dict:
    """Get settings for a specific pair."""
    return PAIR_SETTINGS.get(instrument, {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 8,
        'sl_pips': 20,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'expected_wr': 80.0
    })


def is_allowed_hour(instrument: str, hour_utc: int) -> bool:
    """
    Check if current hour is allowed for trading this pair.
    Based on optimization - some pairs trade specific sessions.

    Args:
        instrument: Currency pair
        hour_utc: Current hour in UTC (0-23)

    Returns:
        True if trading is allowed, False otherwise
    """
    session = TRADING_SESSIONS.get(instrument, {'allowed_hours': list(range(0, 24)), 'session_filter': False})
    # If session_filter is False, always allow trading
    if not session.get('session_filter', False):
        return True
    return hour_utc in session['allowed_hours']


def get_cooldown_minutes(instrument: str) -> int:
    """
    Get cooldown minutes for a specific pair.

    Args:
        instrument: Currency pair

    Returns:
        Cooldown in minutes (0 based on optimization)
    """
    settings = get_pair_settings(instrument)
    return settings.get('cooldown_minutes', COOLDOWN_MINUTES)


def calculate_tp_sl(instrument: str, entry_price: float, direction: str) -> tuple:
    """
    Calculate Take Profit and Stop Loss prices.

    Args:
        instrument: Currency pair
        entry_price: Entry price
        direction: 'BUY' or 'SELL'

    Returns:
        Tuple of (take_profit_price, stop_loss_price)
    """
    settings = get_pair_settings(instrument)
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']

    pip_value = 0.01 if 'JPY' in instrument else 0.0001

    if direction == 'BUY':
        take_profit = entry_price + (tp_pips * pip_value)
        stop_loss = entry_price - (sl_pips * pip_value)
    else:  # SELL
        take_profit = entry_price - (tp_pips * pip_value)
        stop_loss = entry_price + (sl_pips * pip_value)

    return take_profit, stop_loss


def print_config_info():
    """Print configuration info."""
    print("=" * 80)
    print("OPTIMIZED FOREX STRATEGY V2 - LIVE TRADING")
    print("!!! REAL MONEY - 14-STRATEGY MULTI-STAGE OPTIMIZATION !!!")
    print("=" * 80)
    print(f"\nAccount Type: {ACCOUNT_TYPE.upper()} (REAL MONEY)")
    print(f"Timeframe: {TIMEFRAME} (1-minute)")
    print(f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"\nTrading {len(OPTIMIZED_PAIRS)} Optimized Pairs:")
    print(f"{'PAIR':<10} {'STRATEGY':<18} {'TP':<4} {'SL':<4} {'SESSION':<10} {'WR':<8} {'PIPS':<8}")
    print("-" * 75)
    total_pips = 0
    for pair in OPTIMIZED_PAIRS:
        s = PAIR_SETTINGS[pair]
        pips = s.get('backtest_pips', 0)
        total_pips += pips
        print(f"{pair:<10} {s['strategy']:<18} {s['tp_pips']:<4} {s['sl_pips']:<4} {s['session']:<10} {s['expected_wr']:.1f}%   +{pips}p")
    print("-" * 75)
    print(f"Total: +{total_pips} pips (100K candle backtest) | 86% Avg WR | Walk-Forward Validated")
    print("=" * 80)
