"""
Forex Futures Symbol Settings for Tradovate

Based on CME Micro Forex Futures specifications
"""

# ==================== TRADOVATE SYMBOL MAPPING ====================

# Tradovate uses specific symbol formats
# Format: [Product][Month][Year]
# Example: M6EU2 = Micro Euro, December 2025

SYMBOL_MAP = {
    # Format: 'Friendly Name': 'Tradovate Symbol'
    'EUR_USD': 'M6EU2',  # Micro Euro (Dec 2025)
    'GBP_USD': 'M6BU2',  # Micro British Pound (Dec 2025)
    'USD_JPY': 'MJYU2',  # Micro Japanese Yen (Dec 2025)
    'USD_CAD': 'MCDU2',  # Micro Canadian Dollar (Dec 2025)
    'USD_CHF': 'MSFU2',  # Micro Swiss Franc (Dec 2025)
    'AUD_USD': 'M6AU2',  # Micro Australian Dollar (Dec 2025)
}

# Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
#              N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

# ==================== PAIR-SPECIFIC SETTINGS ====================

PAIR_SETTINGS = {
    'M6EU2': {  # Micro Euro (EUR/USD)
        'name': 'EUR/USD',
        'tp_pips': 20,
        'sl_pips': 16,
        'tp_ticks': 40,      # 20 pips × 2
        'sl_ticks': 32,      # 16 pips × 2
        'tick_size': 0.00005,
        'tick_value': 6.25,
        'pip_multiplier': 2.0,
        'contract_size': 12500,  # €12,500
    },
    'M6BU2': {  # Micro British Pound (GBP/USD)
        'name': 'GBP/USD',
        'tp_pips': 30,
        'sl_pips': 25,
        'tp_ticks': 30,      # 30 pips × 1
        'sl_ticks': 25,      # 25 pips × 1
        'tick_size': 0.0001,
        'tick_value': 6.25,
        'pip_multiplier': 1.0,
        'contract_size': 6250,   # £6,250
    },
    'MJYU2': {  # Micro Japanese Yen (USD/JPY)
        'name': 'USD/JPY',
        'tp_pips': 18,
        'sl_pips': 15,
        'tp_ticks': 180,     # 18 pips × 10
        'sl_ticks': 150,     # 15 pips × 10
        'tick_size': 0.000001,
        'tick_value': 1.25,
        'pip_multiplier': 10.0,
        'contract_size': 1250000,  # ¥1,250,000
    },
    'MCDU2': {  # Micro Canadian Dollar (USD/CAD)
        'name': 'USD/CAD',
        'tp_pips': 20,
        'sl_pips': 16,
        'tp_ticks': 40,      # 20 pips × 2
        'sl_ticks': 32,      # 16 pips × 2
        'tick_size': 0.00005,
        'tick_value': 5.00,
        'pip_multiplier': 2.0,
        'contract_size': 10000,    # C$10,000
    },
    'MSFU2': {  # Micro Swiss Franc (USD/CHF)
        'name': 'USD/CHF',
        'tp_pips': 15,
        'sl_pips': 12,
        'tp_ticks': 30,      # 15 pips × 2
        'sl_ticks': 24,      # 12 pips × 2
        'tick_size': 0.00005,
        'tick_value': 6.25,
        'pip_multiplier': 2.0,
        'contract_size': 12500,    # CHF 12,500
    },
    'M6AU2': {  # Micro Australian Dollar (AUD/USD)
        'name': 'AUD/USD',
        'tp_pips': 20,
        'sl_pips': 16,
        'tp_ticks': 40,      # 20 pips × 2
        'sl_ticks': 32,      # 16 pips × 2
        'tick_size': 0.00005,
        'tick_value': 5.00,
        'pip_multiplier': 2.0,
        'contract_size': 10000,    # A$10,000
    },
}

# ==================== ACTIVE TRADING PAIRS ====================

# Pairs to trade (based on data availability)
ACTIVE_PAIRS = [
    'M6EU2',  # EUR/USD - Best performer
    'M6BU2',  # GBP/USD
    'MSFU2',  # USD/CHF
    # Add more as available:
    # 'MJYU2',  # USD/JPY
    # 'MCDU2',  # USD/CAD
    # 'M6AU2',  # AUD/USD
]

# ==================== INDICATOR SETTINGS ====================

INDICATOR_SETTINGS = {
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'bb_period': 20,
    'bb_std': 2.0,
    'stoch_period': 14,
    'stoch_k': 3,
    'stoch_d': 3,
    'stoch_oversold': 30,
    'stoch_overbought': 70,
}

# ==================== HELPER FUNCTIONS ====================

def get_tradovate_symbol(oanda_symbol: str) -> str:
    """Convert OANDA symbol to Tradovate symbol"""
    return SYMBOL_MAP.get(oanda_symbol, oanda_symbol)


def get_oanda_symbol(tradovate_symbol: str) -> str:
    """Convert Tradovate symbol to OANDA symbol"""
    reverse_map = {v: k for k, v in SYMBOL_MAP.items()}
    return reverse_map.get(tradovate_symbol, tradovate_symbol)


def get_pair_settings(symbol: str) -> dict:
    """Get settings for a symbol"""
    return PAIR_SETTINGS.get(symbol, {})


def calculate_stop_loss_price(entry_price: float, side: str, symbol: str) -> float:
    """Calculate stop loss price"""
    settings = get_pair_settings(symbol)
    sl_ticks = settings.get('sl_ticks', 32)
    tick_size = settings.get('tick_size', 0.00005)

    if side.lower() == 'buy':
        return entry_price - (sl_ticks * tick_size)
    else:
        return entry_price + (sl_ticks * tick_size)


def calculate_take_profit_price(entry_price: float, side: str, symbol: str) -> float:
    """Calculate take profit price"""
    settings = get_pair_settings(symbol)
    tp_ticks = settings.get('tp_ticks', 40)
    tick_size = settings.get('tick_size', 0.00005)

    if side.lower() == 'buy':
        return entry_price + (tp_ticks * tick_size)
    else:
        return entry_price - (tp_ticks * tick_size)


def calculate_risk_per_contract(symbol: str) -> float:
    """Calculate risk per contract in USD"""
    settings = get_pair_settings(symbol)
    sl_ticks = settings.get('sl_ticks', 32)
    tick_value = settings.get('tick_value', 6.25)

    return sl_ticks * tick_value
