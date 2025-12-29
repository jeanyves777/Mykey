"""
CRYPTO MARGIN TRADING - PAPER TRADING CONFIG
=============================================
Configuration for paper/simulated crypto margin trading on Kraken.

Initial settings - to be optimized through backtesting.

Note: These percentages are starting points. Run backtesting
with optimize_crypto_strategy.py to find optimal values.
"""

from pathlib import Path

# =============================================================================
# ACCOUNT SETTINGS
# =============================================================================
ACCOUNT_TYPE = 'paper'  # Paper trading mode
LOG_DIR = 'crypto_logs'  # Log directory
INITIAL_BALANCE = 10000.0  # Starting paper balance (USD)

# =============================================================================
# TRADING PARAMETERS
# =============================================================================
MAX_CONCURRENT_POSITIONS = 5  # Max open positions at once
DEFAULT_LEVERAGE = 3  # Default leverage (1-5)
TIMEFRAME = '1m'  # 1-minute candles
CANDLE_COUNT = 100  # Number of candles for analysis
COOLDOWN_MINUTES = 0  # No cooldown between trades (0 = trade on every signal)

# Risk per trade (percentage of account)
RISK_PER_TRADE_PCT = 2.0  # 2% risk per trade

# =============================================================================
# SESSION DEFINITIONS (UTC hours) - Optional for crypto (24/7 market)
# =============================================================================
SESSIONS = {
    'ASIA': list(range(0, 8)),       # 00:00-08:00 UTC (Asia)
    'EUROPE': list(range(8, 16)),    # 08:00-16:00 UTC (Europe)
    'US': list(range(14, 22)),       # 14:00-22:00 UTC (US)
    'ALL': list(range(0, 24)),       # All hours (24/7)
}

# =============================================================================
# TRADING PAIRS - Binance symbols (for data) + Kraken mapping
# Updated Dec 19, 2025 with actual Kraken leverage limits
# =============================================================================
# Binance symbol -> Kraken symbol mapping
BINANCE_TO_KRAKEN = {
    # 10x leverage pairs
    'BTCUSDT': 'XXBTZUSD',
    'ETHUSDT': 'XETHZUSD',
    'SOLUSDT': 'SOLUSD',
    'XRPUSDT': 'XXRPZUSD',
    'DOGEUSDT': 'XDGUSD',
    'LTCUSDT': 'XLTCZUSD',
    'ADAUSDT': 'ADAUSD',
    'LINKUSDT': 'LINKUSD',
    'AVAXUSDT': 'AVAXUSD',
    'SUIUSDT': 'SUIUSD',
    # 3x leverage pairs
    'DOTUSDT': 'DOTUSD',
    'ZECUSDT': 'XZECZUSD',
    'BCHUSDT': 'BCHUSD',
    'PEPEUSDT': 'PEPEUSD',
    'UNIUSDT': 'UNIUSD',
    # 2x leverage pairs
    'XLMUSDT': 'XXLMZUSD',
    'XMRUSDT': 'XXMRZUSD',
}

KRAKEN_TO_BINANCE = {v: k for k, v in BINANCE_TO_KRAKEN.items()}

# Kraken max leverage per pair (actual limits Dec 2025)
KRAKEN_MAX_LEVERAGE = {
    'BTCUSDT': 10, 'ETHUSDT': 10, 'SOLUSDT': 10, 'XRPUSDT': 10,
    'DOGEUSDT': 10, 'LTCUSDT': 10, 'ADAUSDT': 10, 'LINKUSDT': 10,
    'AVAXUSDT': 10, 'SUIUSDT': 10,
    'DOTUSDT': 3, 'ZECUSDT': 3, 'BCHUSDT': 3, 'PEPEUSDT': 3, 'UNIUSDT': 3,
    'XLMUSDT': 2, 'XMRUSDT': 2,
}

# =============================================================================
# PAIR-SPECIFIC SETTINGS - REALISTIC OPTIMIZATION Dec 19, 2025 (v2)
# WITH KRAKEN FEES (0.52% round-trip) + REALISTIC SPREADS
# 8/17 pairs VALIDATED with walk-forward testing!
# =============================================================================
PAIR_SETTINGS = {
    # =========================================================================
    # VALIDATED PAIRS (8) - ENABLED
    # These passed 70/30 walk-forward validation with realistic fees
    # =========================================================================

    # XRP - RSI_DIVERGENCE: 71.7% WR, 1.21 PF, VALIDATED
    'XRPUSDT': {
        'strategy': 'RSI_DIVERGENCE',
        'tp_pct': 2.0,
        'sl_pct': 2.5,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # LTC - SWING_REVERSAL: 80.1% WR, 1.30 PF, VALIDATED
    'LTCUSDT': {
        'strategy': 'SWING_REVERSAL',
        'tp_pct': 2.0,
        'sl_pct': 4.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # ADA - RSI_TREND_FOLLOW: 69.0% WR, 1.40 PF, VALIDATED
    'ADAUSDT': {
        'strategy': 'RSI_TREND_FOLLOW',
        'tp_pct': 3.5,
        'sl_pct': 4.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # ZEC - RSI_DIVERGENCE: 60.0% WR, 1.71 PF, VALIDATED
    'ZECUSDT': {
        'strategy': 'RSI_DIVERGENCE',
        'tp_pct': 4.0,
        'sl_pct': 2.5,
        'leverage': 3,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 4.0,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # BCH - EMA_CROSSOVER: 61.7% WR, 1.54 PF, VALIDATED
    'BCHUSDT': {
        'strategy': 'EMA_CROSSOVER',
        'tp_pct': 5.0,
        'sl_pct': 4.0,
        'leverage': 3,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # UNI - TRIPLE_CONFIRM: 70.6% WR, 2.27 PF!! BEST PERFORMER, VALIDATED
    'UNIUSDT': {
        'strategy': 'TRIPLE_CONFIRM',
        'tp_pct': 2.5,
        'sl_pct': 1.5,
        'leverage': 3,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.5,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # XLM - RSI_EXTREME: 62.9% WR, 1.27 PF, VALIDATED
    'XLMUSDT': {
        'strategy': 'RSI_EXTREME',
        'tp_pct': 4.0,
        'sl_pct': 4.0,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # XMR - MACD_ZERO: 91.7% WR, 5.93 PF!! EXCELLENT, VALIDATED
    'XMRUSDT': {
        'strategy': 'MACD_ZERO',
        'tp_pct': 3.0,
        'sl_pct': 4.0,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': True,
    },

    # =========================================================================
    # PROFITABLE BUT NOT VALIDATED (3) - DISABLED (high risk)
    # Good in-sample but failed walk-forward validation
    # =========================================================================

    # BTC - VOLUME_BREAKOUT: 66.7% WR, 1.50 PF (best validated option)
    'BTCUSDT': {
        'strategy': 'VOLUME_BREAKOUT',
        'tp_pct': 4.0,
        'sl_pct': 4.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 2.0,
        'cooldown_minutes': 0,
        'enabled': True,  # Keep enabled - passed validation
    },

    # ETH - RSI_DIVERGENCE: 63.3% WR, 1.63 PF (NOT validated)
    'ETHUSDT': {
        'strategy': 'RSI_DIVERGENCE',
        'tp_pct': 5.0,
        'sl_pct': 4.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 2.5,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - failed walk-forward validation
    },

    # DOGE - RSI_DIVERGENCE: 77.3% WR, 1.93 PF (NOT validated)
    'DOGEUSDT': {
        'strategy': 'RSI_DIVERGENCE',
        'tp_pct': 2.0,
        'sl_pct': 2.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 4.0,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - failed walk-forward validation
    },

    # =========================================================================
    # UNPROFITABLE PAIRS (6) - DISABLED
    # No strategy found profitable after fees
    # =========================================================================

    # SOL - NO PROFITABLE STRATEGY AFTER FEES
    'SOLUSDT': {
        'strategy': 'EMA_PULLBACK',
        'tp_pct': 3.0,
        'sl_pct': 4.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - not profitable after fees
    },

    # LINK - NO PROFITABLE STRATEGY AFTER FEES
    'LINKUSDT': {
        'strategy': 'MOMENTUM_BREAKOUT',
        'tp_pct': 5.0,
        'sl_pct': 4.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - not profitable after fees
    },

    # AVAX - NO PROFITABLE STRATEGY AFTER FEES
    'AVAXUSDT': {
        'strategy': 'SWING_REVERSAL',
        'tp_pct': 7.0,
        'sl_pct': 5.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.5,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - not profitable after fees
    },

    # SUI - NO PROFITABLE STRATEGY AFTER FEES
    'SUIUSDT': {
        'strategy': 'MOMENTUM_BREAKOUT',
        'tp_pct': 6.0,
        'sl_pct': 5.0,
        'leverage': 5,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.5,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - not profitable after fees
    },

    # DOT - NO PROFITABLE STRATEGY AFTER FEES
    'DOTUSDT': {
        'strategy': 'RSI_TREND_FOLLOW',
        'tp_pct': 4.0,
        'sl_pct': 5.0,
        'leverage': 3,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - not profitable after fees
    },

    # PEPE - NO PROFITABLE STRATEGY AFTER FEES
    'PEPEUSDT': {
        'strategy': 'TRIPLE_CONFIRM',
        'tp_pct': 3.0,
        'sl_pct': 4.0,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 5.0,
        'cooldown_minutes': 0,
        'enabled': False,  # DISABLED - not profitable after fees
    },
}

# List of enabled pairs to trade
TRADING_PAIRS = [pair for pair, settings in PAIR_SETTINGS.items() if settings.get('enabled', True)]

# =============================================================================
# DATA SETTINGS
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "Crypto_Data_from_Binance"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pair_settings(pair: str) -> dict:
    """Get settings for a specific pair."""
    return PAIR_SETTINGS.get(pair, {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'leverage': 2,
        'session': 'ALL',
        'volume_filter': False,
        'trend_filter': False,
        'volatility_filter': False,
        'max_atr_pct': 3.0,
        'cooldown_minutes': 0,
        'enabled': False,
    })


def get_kraken_symbol(binance_symbol: str) -> str:
    """Convert Binance symbol to Kraken symbol."""
    return BINANCE_TO_KRAKEN.get(binance_symbol, binance_symbol)


def get_binance_symbol(kraken_symbol: str) -> str:
    """Convert Kraken symbol to Binance symbol."""
    return KRAKEN_TO_BINANCE.get(kraken_symbol, kraken_symbol)


def is_allowed_hour(pair: str, hour_utc: int) -> bool:
    """
    Check if current hour is allowed for trading this pair.

    Args:
        pair: Trading pair
        hour_utc: Current hour in UTC (0-23)

    Returns:
        True if trading is allowed
    """
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
        Position size in base currency (e.g., BTC amount)
    """
    if risk_pct is None:
        risk_pct = RISK_PER_TRADE_PCT

    # Risk amount in USD
    risk_amount = account_balance * (risk_pct / 100)

    # Position value to risk this amount at stop loss
    # If SL is 1%, and we risk $200, position = $200 / 0.01 = $20,000
    position_value = risk_amount / (sl_pct / 100)

    # Apply leverage limit
    max_position = account_balance * leverage
    position_value = min(position_value, max_position)

    # Convert to base currency amount
    position_size = position_value / entry_price

    return position_size


def calculate_tp_sl(
    pair: str,
    entry_price: float,
    direction: str
) -> tuple:
    """
    Calculate Take Profit and Stop Loss prices.

    Args:
        pair: Trading pair
        entry_price: Entry price
        direction: 'BUY' or 'SELL'

    Returns:
        Tuple of (take_profit_price, stop_loss_price)
    """
    settings = get_pair_settings(pair)
    tp_pct = settings['tp_pct']
    sl_pct = settings['sl_pct']

    if direction == 'BUY':
        take_profit = entry_price * (1 + tp_pct / 100)
        stop_loss = entry_price * (1 - sl_pct / 100)
    else:  # SELL
        take_profit = entry_price * (1 - tp_pct / 100)
        stop_loss = entry_price * (1 + sl_pct / 100)

    return take_profit, stop_loss


def print_config_info():
    """Print configuration info."""
    print("=" * 80)
    print("CRYPTO MARGIN TRADING - PAPER MODE")
    print("=" * 80)
    print(f"\nAccount Type: {ACCOUNT_TYPE.upper()}")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"Default Leverage: {DEFAULT_LEVERAGE}x")
    print(f"Risk Per Trade: {RISK_PER_TRADE_PCT}%")
    print(f"\nTrading {len(TRADING_PAIRS)} Pairs:")
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
