"""
OANDA Forex Trading Configuration
Major currency pairs with multi-timeframe momentum strategy
"""

# Major forex pairs to trade (ONLY symbols with NinjaTrader Micro E-mini FX futures mappings)
FOREX_INSTRUMENTS = [
    "EUR_USD",  # Euro / US Dollar → M6E (Micro EUR/USD)
    "GBP_USD",  # British Pound / US Dollar → M6B (Micro GBP/USD)
    "USD_JPY",  # US Dollar / Japanese Yen → MJY (Micro USD/JPY)
    "USD_CHF",  # US Dollar / Swiss Franc → MSF (Micro USD/CHF)
    "USD_CAD",  # US Dollar / Canadian Dollar → MCD (Micro USD/CAD)
    # NOTE: AUD_USD and NZD_USD removed - no NinjaTrader micro futures mapping
]

# Strategy parameters
STRATEGY_CONFIG = {
    "max_trades_per_day": 3,  # Maximum trades per day
    "daily_profit_target": 0.02,  # 2% daily profit target
    "trade_size_pct": 0.10,  # 10% of account per trade
    "take_profit_pct": 0.015,  # 1.5% TP (150 pips on standard lot)
    "stop_loss_pct": 0.01,  # 1% SL (100 pips on standard lot)
    "trailing_stop_trigger": 0.006,  # Start trailing at 0.6% (60 pips)
    "trailing_stop_distance": 0.004,  # Trail 0.4% behind (40 pips)
}

# Risk management
RISK_CONFIG = {
    "max_position_size_pct": 0.20,  # Max 20% of account in one position
    "max_daily_loss_pct": 0.03,  # Stop trading if down 3% in a day
    "max_total_positions": 3,  # Maximum concurrent positions across all pairs
}

# Timeframes for analysis
TIMEFRAMES = {
    "entry": "M1",  # 1-minute for entry signals
    "price_action": "M5",  # 5-minute for price action
    "htf_short": "M30",  # 30-minute for trend filter
    "htf_long": "H1",  # 1-hour for trend filter
}

# Data requirements
DATA_CONFIG = {
    "candles_1min": 500,  # Number of 1-min candles to fetch
    "candles_5min": 100,  # Number of 5-min candles to fetch
    "candles_30min": 50,  # Number of 30-min candles to fetch
    "candles_1hour": 50,  # Number of 1-hour candles to fetch
}

# Trading hours (UTC)
TRADING_HOURS = {
    "london_open": "08:00",  # London open
    "ny_open": "13:00",  # NY open (8 AM EST)
    "ny_close": "21:00",  # NY close (4 PM EST)
    "best_session_start": "13:00",  # London/NY overlap
    "best_session_end": "17:00",  # End of overlap
}

# Paper trading simulation
PAPER_TRADING_CONFIG = {
    "initial_balance": 10000,  # Starting balance for paper trading
    "commission_per_trade": 0.00002,  # 0.2 pips commission per trade
    "slippage_pips": 0.5,  # Average slippage in pips
}

# Backtest configuration
BACKTEST_CONFIG = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000,
    "commission": 0.00002,  # 0.2 pips per trade
    "slippage": 0.5,  # Average slippage in pips
}

# Logging
LOGGING_CONFIG = {
    "log_trades": True,
    "log_signals": True,
    "log_to_file": True,
    "log_directory": "trading_system/Forex_Trading/logs",
}
