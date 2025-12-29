"""
CRYPTO MARGIN TRADING - LIVE TRADING CONFIG
============================================
Configuration for LIVE crypto margin trading on Kraken.

WARNING: This configuration is for REAL MONEY trading.
Only use after thorough backtesting and paper trading validation.

Settings should be copied from optimized paper config after validation.
"""

from pathlib import Path

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================
ACCOUNT_TYPE = 'live'  # LIVE trading mode - REAL MONEY
LOG_DIR = 'crypto_logs_live'  # Separate log directory for live trades

# =============================================================================
# TRADING PARAMETERS - CONSERVATIVE FOR LIVE
# =============================================================================
MAX_CONCURRENT_POSITIONS = 3  # Lower for live trading
DEFAULT_LEVERAGE = 2  # Conservative leverage for live
TIMEFRAME = '1m'
CANDLE_COUNT = 100
COOLDOWN_MINUTES = 5  # 5 min cooldown between trades for safety

# Risk per trade (percentage of account) - LOWER FOR LIVE
RISK_PER_TRADE_PCT = 1.0  # 1% risk per trade (conservative)

# Maximum daily loss before stopping (percentage)
MAX_DAILY_LOSS_PCT = 5.0  # Stop trading after 5% daily loss

# =============================================================================
# SESSION DEFINITIONS (UTC hours)
# =============================================================================
SESSIONS = {
    'ASIA': list(range(0, 8)),
    'EUROPE': list(range(8, 16)),
    'US': list(range(14, 22)),
    'ALL': list(range(0, 24)),
}

# =============================================================================
# TRADING PAIRS - Binance symbols (for data) + Kraken mapping
# =============================================================================
BINANCE_TO_KRAKEN = {
    'BTCUSDT': 'XXBTZUSD',
    'ETHUSDT': 'XETHZUSD',
    'SOLUSDT': 'SOLUSD',
    'XRPUSDT': 'XXRPZUSD',
    'DOGEUSDT': 'XDGUSD',
    'LTCUSDT': 'XLTCZUSD',
    'ADAUSDT': 'ADAUSD',
    'LINKUSDT': 'LINKUSD',
    'AVAXUSDT': 'AVAXUSD',
    'DOTUSDT': 'DOTUSD',
}

KRAKEN_TO_BINANCE = {v: k for k, v in BINANCE_TO_KRAKEN.items()}

# =============================================================================
# PAIR-SPECIFIC SETTINGS - LIVE (Conservative)
# Copy optimized settings from paper config after validation
# =============================================================================
PAIR_SETTINGS = {
    # BTC - Most liquid, safest for live
    'BTCUSDT': {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.4,
        'sl_pct': 1.0,
        'leverage': 2,  # Conservative leverage
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,  # Enable volatility filter for safety
        'max_atr_pct': 1.5,  # Lower ATR threshold
        'cooldown_minutes': 5,
        'enabled': True,
    },
    # ETH - Second choice for live
    'ETHUSDT': {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.0,
        'cooldown_minutes': 5,
        'enabled': True,
    },
    # SOL - Disabled for initial live trading (higher volatility)
    'SOLUSDT': {
        'strategy': 'MACD_CROSS',
        'tp_pct': 0.8,
        'sl_pct': 2.0,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': True,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 10,
        'enabled': False,  # Disabled for initial live
    },
    # XRP
    'XRPUSDT': {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 5,
        'enabled': False,  # Enable after paper validation
    },
    # DOGE - Disabled (too volatile for initial live)
    'DOGEUSDT': {
        'strategy': 'RSI_EXTREME',
        'tp_pct': 1.0,
        'sl_pct': 2.5,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': True,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 15,
        'enabled': False,  # Disabled - too volatile
    },
    # LTC
    'LTCUSDT': {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.0,
        'cooldown_minutes': 5,
        'enabled': False,  # Enable after paper validation
    },
    # ADA
    'ADAUSDT': {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 5,
        'enabled': False,
    },
    # LINK
    'LINKUSDT': {
        'strategy': 'MACD_CROSS',
        'tp_pct': 0.6,
        'sl_pct': 1.5,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 5,
        'enabled': False,
    },
    # AVAX
    'AVAXUSDT': {
        'strategy': 'RSI_MACD_COMBO',
        'tp_pct': 0.7,
        'sl_pct': 1.8,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': True,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 10,
        'enabled': False,
    },
    # DOT
    'DOTUSDT': {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 5,
        'enabled': False,
    },
}

# List of enabled pairs (only BTC and ETH for initial live trading)
TRADING_PAIRS = [pair for pair, settings in PAIR_SETTINGS.items() if settings.get('enabled', False)]

# =============================================================================
# DATA SETTINGS
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "Crypto_Data_from_Binance"

# =============================================================================
# HELPER FUNCTIONS (same as paper config)
# =============================================================================

def get_pair_settings(pair: str) -> dict:
    """Get settings for a specific pair."""
    return PAIR_SETTINGS.get(pair, {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': True,
        'trend_filter': False,
        'volatility_filter': True,
        'max_atr_pct': 2.0,
        'cooldown_minutes': 5,
        'enabled': False,
    })


def get_kraken_symbol(binance_symbol: str) -> str:
    """Convert Binance symbol to Kraken symbol."""
    return BINANCE_TO_KRAKEN.get(binance_symbol, binance_symbol)


def get_binance_symbol(kraken_symbol: str) -> str:
    """Convert Kraken symbol to Binance symbol."""
    return KRAKEN_TO_BINANCE.get(kraken_symbol, kraken_symbol)


def is_allowed_hour(pair: str, hour_utc: int) -> bool:
    """Check if current hour is allowed for trading this pair."""
    settings = get_pair_settings(pair)
    session = settings.get('session', 'ALL')

    if session == 'ALL':
        return True

    allowed_hours = SESSIONS.get(session, list(range(0, 24)))
    return hour_utc in allowed_hours


def get_cooldown_minutes(pair: str) -> int:
    """Get cooldown minutes for a specific pair."""
    settings = get_pair_settings(pair)
    return settings.get('cooldown_minutes', COOLDOWN_MINUTES)


def calculate_position_size(
    account_balance: float,
    entry_price: float,
    sl_pct: float,
    leverage: int = 1,
    risk_pct: float = None
) -> float:
    """
    Calculate position size based on risk management.

    Args:
        account_balance: Current account balance
        entry_price: Entry price
        sl_pct: Stop loss percentage
        leverage: Leverage multiplier
        risk_pct: Risk percentage (default: RISK_PER_TRADE_PCT)

    Returns:
        Position size in base currency
    """
    if risk_pct is None:
        risk_pct = RISK_PER_TRADE_PCT

    risk_amount = account_balance * (risk_pct / 100)
    position_value = risk_amount / (sl_pct / 100)
    max_position = account_balance * leverage
    position_value = min(position_value, max_position)
    position_size = position_value / entry_price

    return position_size


def calculate_tp_sl(pair: str, entry_price: float, direction: str) -> tuple:
    """Calculate Take Profit and Stop Loss prices."""
    settings = get_pair_settings(pair)
    tp_pct = settings['tp_pct']
    sl_pct = settings['sl_pct']

    if direction == 'BUY':
        take_profit = entry_price * (1 + tp_pct / 100)
        stop_loss = entry_price * (1 - sl_pct / 100)
    else:
        take_profit = entry_price * (1 - tp_pct / 100)
        stop_loss = entry_price * (1 + sl_pct / 100)

    return take_profit, stop_loss


def print_config_info():
    """Print configuration info."""
    print("=" * 80)
    print("CRYPTO MARGIN TRADING - LIVE MODE (REAL MONEY)")
    print("=" * 80)
    print("\n*** WARNING: LIVE TRADING - REAL MONEY AT RISK ***\n")
    print(f"Account Type: {ACCOUNT_TYPE.upper()}")
    print(f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"Default Leverage: {DEFAULT_LEVERAGE}x (Conservative)")
    print(f"Risk Per Trade: {RISK_PER_TRADE_PCT}%")
    print(f"Max Daily Loss: {MAX_DAILY_LOSS_PCT}%")
    print(f"\nTrading {len(TRADING_PAIRS)} Pairs (Conservative Selection):")
    print(f"{'PAIR':<12} {'STRATEGY':<16} {'TP%':<6} {'SL%':<6} {'LEV':<4} {'FILTERS':<15}")
    print("-" * 75)

    for pair in TRADING_PAIRS:
        s = PAIR_SETTINGS[pair]
        filters = []
        if s.get('volume_filter'):
            filters.append('Vol')
        if s.get('trend_filter'):
            filters.append('Trend')
        if s.get('volatility_filter'):
            filters.append('ATR')
        filter_str = '+'.join(filters) if filters else 'None'

        print(f"{pair:<12} {s['strategy']:<16} {s['tp_pct']:<6.1f} {s['sl_pct']:<6.1f} {s['leverage']}x   {filter_str:<15}")

    print("=" * 80)


if __name__ == "__main__":
    print_config_info()
