"""
IMPROVED FOREX TRADING CONFIG
Based on backtest analysis showing:
1. 5-pip TP is too small - losses wipe out too many wins
2. US session (12:00-20:00 UTC) has best win rates
3. Lower volatility = better win rates
4. NZD_USD is by far the best performer
"""

from pathlib import Path

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================
ACCOUNT_TYPE = 'practice'  # OANDA practice account
LOG_DIR = 'forex_logs'     # Log directory for paper trading

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
MAX_CONCURRENT_POSITIONS = 5  # Reduced to focus on quality
COOLDOWN_MINUTES = 30
TIMEFRAME = 'M1'  # 1-minute candles
CANDLE_COUNT = 100  # Number of candles to fetch for analysis

# =============================================================================
# SESSION FILTERS (UTC Hours)
# Only trade during high win-rate hours
# =============================================================================
TRADING_SESSIONS = {
    # Best hours per pair based on backtest analysis
    'EUR_USD': {'allowed_hours': list(range(12, 21))},  # 12:00-20:00 UTC (85%+ WR)
    'GBP_USD': {'allowed_hours': list(range(12, 21))},  # 12:00-20:00 UTC (83%+ WR)
    'USD_JPY': {'allowed_hours': list(range(0, 24))},   # All hours similar (~73%)
    'USD_CHF': {'allowed_hours': [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23]},  # Avoid 01-03, 08
    'USD_CAD': {'allowed_hours': list(range(14, 22))},  # 14:00-21:00 UTC (80%+ WR)
    'NZD_USD': {'allowed_hours': list(range(0, 24))},   # All hours good (82%+ WR)
    'AUD_JPY': {'allowed_hours': list(range(10, 17))},  # 10:00-16:00 UTC (76%+ WR)
}

# =============================================================================
# PAIR-SPECIFIC SETTINGS - ADJUSTED FOR PROFITABILITY
# Key change: Increased TP to improve risk/reward
# =============================================================================
PAIR_SETTINGS = {
    # TOP TIER - Best performers, keep trading
    'NZD_USD': {
        # BEST: +46,350 pips, 82.6% win rate
        'strategy': 'RSI_30_70',
        'tp_pips': 10,  # Keep as is
        'sl_pips': 30,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'expected_wr': 82.6,
        'tier': 'TOP',
    },

    # SECOND TIER - Profitable but can improve
    'GBP_USD': {
        # +7,960 pips, 76% win rate
        # Increase TP from 10 to 12 for better RR
        'strategy': 'RSI_30_70',
        'tp_pips': 12,  # Increased from 10
        'sl_pips': 30,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'expected_wr': 76.0,
        'tier': 'SECOND',
    },
    'USD_CHF': {
        # +4,896 pips, 76.1% win rate
        'strategy': 'STRONG_TREND',
        'tp_pips': 10,  # Increased from 8
        'sl_pips': 24,
        'expected_wr': 76.1,
        'tier': 'SECOND',
    },

    # PROBLEMATIC - Need wider TP or consider removing
    'EUR_USD': {
        # -95 pips despite 80% WR! Problem is 5-pip TP
        # Solution: Widen TP to 8 pips
        'strategy': 'RSI_30_70',
        'tp_pips': 8,   # INCREASED from 5 - critical fix
        'sl_pips': 20,
        'expected_wr': 80.0,
        'tier': 'PROBLEM',
    },
    'USD_CAD': {
        # -9,695 pips, 77.8% WR - 5-pip TP too small
        # Solution: Widen TP to 8 pips
        'strategy': 'RSI_30_70',
        'tp_pips': 8,   # INCREASED from 5 - critical fix
        'sl_pips': 20,
        'expected_wr': 77.8,
        'tier': 'PROBLEM',
    },

    # CONSIDER REMOVING - Low WR with bad RR
    'USD_JPY': {
        # -15,410 pips, 73.3% WR - MACD not working well
        # Try RSI_30_70 instead of MACD_CROSS
        'strategy': 'RSI_30_70',
        'tp_pips': 12,  # Wider TP
        'sl_pips': 30,
        'expected_wr': 73.3,
        'tier': 'TEST',
    },
    'AUD_JPY': {
        # -11,990 pips, 73.7% WR - MACD not working
        # Try RSI_30_70 instead of MACD_CROSS
        'strategy': 'RSI_30_70',
        'tp_pips': 12,  # Wider TP
        'sl_pips': 30,
        'expected_wr': 73.7,
        'tier': 'TEST',
    },
}

# RECOMMENDED PAIRS (profitable in backtest)
RECOMMENDED_PAIRS = ['NZD_USD', 'GBP_USD', 'USD_CHF']

# ALL PAIRS TO TRADE (including problem ones with new settings)
OPTIMIZED_PAIRS = list(PAIR_SETTINGS.keys())

# =============================================================================
# VOLATILITY FILTERS
# Based on analysis: Wins happen in lower volatility
# =============================================================================
VOLATILITY_FILTERS = {
    # Maximum ATR% to allow entry (filter out high volatility)
    'max_atr_pct': {
        'EUR_USD': 0.015,  # Avg win: 0.0113, Avg loss: 0.0124
        'GBP_USD': 0.016,  # Avg win: 0.0130, Avg loss: 0.0138
        'USD_JPY': 0.022,  # Avg win: 0.0187, Avg loss: 0.0197
        'USD_CHF': 0.018,  # Avg win: 0.0142, Avg loss: 0.0156
        'USD_CAD': 0.012,  # Avg win: 0.0095, Avg loss: 0.0099
        'NZD_USD': 0.022,  # Avg win: 0.0182, Avg loss: 0.0196
        'AUD_JPY': 0.028,  # Avg win: 0.0224, Avg loss: 0.0238
    },
    # Maximum Bollinger Band width to allow entry
    'max_bb_width': {
        'EUR_USD': 0.09,   # Avg win: 0.0704, Avg loss: 0.0774
        'GBP_USD': 0.085,  # Avg win: 0.0683, Avg loss: 0.0726
        'USD_JPY': 0.10,   # Avg win: 0.0837, Avg loss: 0.0888
        'USD_CHF': 0.07,   # Avg win: 0.0545, Avg loss: 0.0611
        'USD_CAD': 0.07,   # Avg win: 0.0590, Avg loss: 0.0615
        'NZD_USD': 0.12,   # Avg win: 0.0979, Avg loss: 0.1050
        'AUD_JPY': 0.12,   # Avg win: 0.0947, Avg loss: 0.1016
    },
}

# =============================================================================
# PIP MULTIPLIERS (for price calculations)
# =============================================================================
PIP_MULTIPLIERS = {
    'EUR_USD': 10000,
    'GBP_USD': 10000,
    'USD_JPY': 100,
    'USD_CHF': 10000,
    'USD_CAD': 10000,
    'NZD_USD': 10000,
    'AUD_JPY': 100,
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
}

# =============================================================================
# POSITION SIZING
# =============================================================================
def calculate_position_size(instrument: str, current_price: float) -> int:
    """
    Calculate position size for $1 per pip.
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
    else:
        return 10000


def get_pair_settings(instrument: str) -> dict:
    """Get settings for a specific pair."""
    return PAIR_SETTINGS.get(instrument, {
        'strategy': 'RSI_30_70',
        'tp_pips': 10,
        'sl_pips': 30,
        'expected_wr': 70.0,
        'tier': 'DEFAULT',
    })


def is_allowed_hour(instrument: str, hour_utc: int) -> bool:
    """Check if current hour is allowed for trading this pair."""
    session = TRADING_SESSIONS.get(instrument, {'allowed_hours': list(range(0, 24))})
    return hour_utc in session['allowed_hours']


def check_volatility_filter(instrument: str, atr_pct: float, bb_width: float) -> bool:
    """
    Check if current volatility is acceptable for entry.
    Returns True if volatility is LOW ENOUGH to trade.
    """
    max_atr = VOLATILITY_FILTERS['max_atr_pct'].get(instrument, 0.03)
    max_bb = VOLATILITY_FILTERS['max_bb_width'].get(instrument, 0.15)

    return atr_pct <= max_atr and bb_width <= max_bb


# =============================================================================
# PRINT CONFIG
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("IMPROVED FOREX TRADING CONFIG")
    print("=" * 70)
    print(f"\nAccount Type: {ACCOUNT_TYPE.upper()}")
    print(f"Timeframe: {TIMEFRAME} (1-minute)")
    print(f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"Cooldown: {COOLDOWN_MINUTES} minutes")

    print(f"\n{'=' * 70}")
    print("PAIR SETTINGS (sorted by tier)")
    print(f"{'=' * 70}")

    for tier in ['TOP', 'SECOND', 'PROBLEM', 'TEST']:
        tier_pairs = [p for p, s in PAIR_SETTINGS.items() if s.get('tier') == tier]
        if tier_pairs:
            print(f"\n{tier} TIER:")
            for pair in tier_pairs:
                s = PAIR_SETTINGS[pair]
                hours = TRADING_SESSIONS.get(pair, {}).get('allowed_hours', [])
                hour_range = f"{min(hours):02d}:00-{max(hours):02d}:00" if hours else "All"
                print(f"  {pair}: {s['strategy']} | TP: {s['tp_pips']}p | SL: {s['sl_pips']}p | Hours: {hour_range}")

    print(f"\n{'=' * 70}")
    print("RECOMMENDED ACTION:")
    print("- Trade primarily: NZD_USD, GBP_USD, USD_CHF")
    print("- EUR_USD & USD_CAD: Test with new 8-pip TP")
    print("- USD_JPY & AUD_JPY: Consider removing if still losing")
    print(f"{'=' * 70}")
