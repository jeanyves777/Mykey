"""
BTCC Futures Trading Configuration
==================================
Configuration settings for BTCC Futures trading system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BTCCConfig:
    """BTCC Futures Trading Configuration"""

    # ==================== API Credentials ====================
    API_KEY: str = "2cc78c86-cf78-47a4-ad38-ca1c0e9b8517"
    SECRET_KEY: str = "6697e175-4ec6-4748-99cf-0ce4ac92790b"

    # Login credentials (required for authentication)
    USER_NAME: str = ""  # Your email or phone number
    PASSWORD: str = ""   # Your account password

    COMPANY_ID: int = 1  # Default company ID

    # ==================== Trading Settings ====================
    # Mode: 'paper' for simulation, 'live' for real trading
    TRADING_MODE: str = "paper"

    # Default leverage (1x to 150x depending on symbol)
    DEFAULT_LEVERAGE: int = 20

    # Position sizing
    POSITION_SIZE_PCT: float = 2.0   # % of account per trade
    MAX_POSITIONS: int = 5           # Maximum concurrent positions
    MAX_POSITION_VALUE: float = 1000  # Maximum position value in USDT

    # Risk management
    DEFAULT_STOP_LOSS_PCT: float = 2.0    # Default SL percentage
    DEFAULT_TAKE_PROFIT_PCT: float = 4.0  # Default TP percentage
    MAX_DAILY_LOSS_PCT: float = 5.0       # Max daily loss before stopping
    TRAILING_STOP_PCT: float = 1.0        # Trailing stop percentage

    # ==================== Symbol Settings ====================
    # Symbols to trade with their specific settings
    SYMBOLS: Dict[str, Dict] = field(default_factory=lambda: {
        'BTCUSDT150x': {
            'enabled': True,
            'leverage': 20,
            'min_volume': 0.01,
            'max_volume': 10.0,
            'volume_step': 0.01,
            'digits': 2,
            'strategy': 'SCALPING_MOMENTUM',
            'tp_pct': 3.0,
            'sl_pct': 1.5,
        },
        'ETHUSDT100x': {
            'enabled': True,
            'leverage': 15,
            'min_volume': 0.01,
            'max_volume': 50.0,
            'volume_step': 0.01,
            'digits': 2,
            'strategy': 'SCALPING_MOMENTUM',
            'tp_pct': 3.5,
            'sl_pct': 1.5,
        },
        'SOLUSDT50x': {
            'enabled': True,
            'leverage': 10,
            'min_volume': 0.1,
            'max_volume': 100.0,
            'volume_step': 0.1,
            'digits': 3,
            'strategy': 'RSI_REVERSAL',
            'tp_pct': 4.0,
            'sl_pct': 2.0,
        },
        'XRPUSDT50x': {
            'enabled': False,
            'leverage': 10,
            'min_volume': 1.0,
            'max_volume': 10000.0,
            'volume_step': 1.0,
            'digits': 4,
            'strategy': 'RSI_REVERSAL',
            'tp_pct': 3.5,
            'sl_pct': 2.0,
        },
        'DOGEUSDT50x': {
            'enabled': False,
            'leverage': 10,
            'min_volume': 10.0,
            'max_volume': 100000.0,
            'volume_step': 10.0,
            'digits': 5,
            'strategy': 'RSI_REVERSAL',
            'tp_pct': 4.0,
            'sl_pct': 2.0,
        },
    })

    # ==================== Strategy Settings ====================
    STRATEGY_PARAMS: Dict[str, Dict] = field(default_factory=lambda: {
        'SCALPING_MOMENTUM': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ema_fast': 9,
            'ema_slow': 21,
            'volume_threshold': 1.5,  # Volume multiplier for entry
            'min_atr_mult': 1.0,      # Minimum ATR for entry
        },
        'RSI_REVERSAL': {
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'confirmation_bars': 2,
            'min_reversal_pct': 0.5,
        },
        'BREAKOUT': {
            'lookback_period': 20,
            'breakout_threshold': 1.5,  # ATR multiplier
            'volume_confirmation': True,
            'retest_entry': False,
        },
        'EMA_CROSSOVER': {
            'ema_fast': 9,
            'ema_slow': 21,
            'ema_trend': 50,
            'min_separation': 0.1,  # Minimum % separation
        },
        'BOLLINGER_SQUEEZE': {
            'bb_period': 20,
            'bb_std': 2.0,
            'kc_period': 20,
            'kc_mult': 1.5,
            'squeeze_threshold': 0.05,
        },
    })

    # ==================== Timing Settings ====================
    # Trading hours (24h format, UTC)
    TRADING_START_HOUR: int = 0   # Start trading at 00:00 UTC
    TRADING_END_HOUR: int = 24    # End trading at 24:00 UTC (24/7)

    # Update intervals
    PRICE_UPDATE_INTERVAL: float = 1.0    # Seconds between price checks
    POSITION_CHECK_INTERVAL: float = 5.0  # Seconds between position checks
    STRATEGY_EVAL_INTERVAL: float = 60.0  # Seconds between strategy evaluation

    # ==================== Logging & Monitoring ====================
    LOG_LEVEL: str = "INFO"
    LOG_TRADES: bool = True
    LOG_FILE: str = "btcc_trading.log"

    # Discord/Telegram notifications (optional)
    ENABLE_NOTIFICATIONS: bool = False
    NOTIFICATION_WEBHOOK: str = ""

    # ==================== Paper Trading Settings ====================
    PAPER_INITIAL_BALANCE: float = 10000.0  # Starting balance for paper trading
    PAPER_COMMISSION_PCT: float = 0.06      # Commission per side (0.06%)

    # ==================== Helper Methods ====================

    def get_enabled_symbols(self) -> List[str]:
        """Get list of enabled symbols."""
        return [s for s, cfg in self.SYMBOLS.items() if cfg.get('enabled', False)]

    def get_symbol_config(self, symbol: str) -> Optional[Dict]:
        """Get configuration for specific symbol."""
        return self.SYMBOLS.get(symbol)

    def get_strategy_params(self, strategy: str) -> Dict:
        """Get parameters for specific strategy."""
        return self.STRATEGY_PARAMS.get(strategy, {})

    def is_paper_mode(self) -> bool:
        """Check if running in paper trading mode."""
        return self.TRADING_MODE.lower() == 'paper'


# Pre-configured instances
BTCC_PAPER_CONFIG = BTCCConfig(
    TRADING_MODE="paper",
    PAPER_INITIAL_BALANCE=10000.0,
)

BTCC_LIVE_CONFIG = BTCCConfig(
    TRADING_MODE="live",
)
