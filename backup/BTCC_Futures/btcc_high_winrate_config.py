"""
BTCC High Win Rate Trading Configuration
==========================================
Optimized configurations based on backtest results from find_high_winrate.py

TOP 3 HIGH WIN RATE STRATEGIES:
1. ETHUSDT SCALP_MOMENTUM 20x TP:1.0% SL:3.0% - 84.8% win rate, +51.5% return
2. BTCUSDT SCALP_MOMENTUM 20x TP:1.0% SL:3.0% - 83.7% win rate, +26.3% return
3. XRPUSDT BOLLINGER_MEAN 20x TP:1.0% SL:3.0% - 80.4% win rate, +17.2% return

Key insight: Small TP (1%) + Wide SL (3%) = High win rate with inverted risk/reward
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BTCCHighWinRateConfig:
    """BTCC High Win Rate Trading Configuration"""

    # ==================== API Credentials ====================
    API_KEY: str = "2cc78c86-cf78-47a4-ad38-ca1c0e9b8517"
    SECRET_KEY: str = "6697e175-4ec6-4748-99cf-0ce4ac92790b"
    USER_NAME: str = ""
    PASSWORD: str = ""
    COMPANY_ID: int = 1

    # ==================== Trading Settings ====================
    TRADING_MODE: str = "paper"  # 'paper' or 'live'

    # Fixed 20x leverage for all high win rate strategies
    DEFAULT_LEVERAGE: int = 20

    # Position sizing (conservative due to 3:1 SL/TP ratio)
    POSITION_SIZE_PCT: float = 5.0   # % of account per trade
    MAX_POSITIONS: int = 3           # Maximum concurrent positions
    MAX_POSITION_VALUE: float = 2000  # Maximum position value in USDT

    # Risk management - CRITICAL: High win rate uses inverted risk/reward
    DEFAULT_STOP_LOSS_PCT: float = 3.0    # Wide SL for high win rate
    DEFAULT_TAKE_PROFIT_PCT: float = 1.0  # Small TP for quick wins
    MAX_DAILY_LOSS_PCT: float = 10.0      # Max daily loss before stopping
    TRAILING_STOP_PCT: float = 0.5        # Trailing stop percentage

    # ==================== HIGH WIN RATE SYMBOL CONFIGS ====================
    # Based on find_high_winrate.py backtest results (30-day test)
    SYMBOLS: Dict[str, Dict] = field(default_factory=lambda: {
        # #1 BEST: ETHUSDT - 84.8% win rate, +51.5% return, 8.6% max DD
        'ETHUSDT': {
            'enabled': True,
            'leverage': 20,
            'min_volume': 0.01,
            'max_volume': 50.0,
            'volume_step': 0.01,
            'digits': 2,
            'strategy': 'SCALP_MOMENTUM',
            'tp_pct': 1.0,   # Small TP for quick wins
            'sl_pct': 3.0,   # Wide SL for high win rate
            'expected_win_rate': 84.8,
            'expected_return': 51.5,
            'expected_max_dd': 8.6,
        },
        # #2: BTCUSDT - 83.7% win rate, +26.3% return, 7.7% max DD
        'BTCUSDT': {
            'enabled': True,
            'leverage': 20,
            'min_volume': 0.001,
            'max_volume': 10.0,
            'volume_step': 0.001,
            'digits': 2,
            'strategy': 'SCALP_MOMENTUM',
            'tp_pct': 1.0,
            'sl_pct': 3.0,
            'expected_win_rate': 83.7,
            'expected_return': 26.3,
            'expected_max_dd': 7.7,
        },
        # #3: XRPUSDT - 80.4% win rate, +17.2% return, 11.0% max DD
        'XRPUSDT': {
            'enabled': True,
            'leverage': 20,
            'min_volume': 1.0,
            'max_volume': 10000.0,
            'volume_step': 1.0,
            'digits': 4,
            'strategy': 'BOLLINGER_MEAN',
            'tp_pct': 1.0,
            'sl_pct': 3.0,
            'expected_win_rate': 80.4,
            'expected_return': 17.2,
            'expected_max_dd': 11.0,
        },
    })

    # ==================== STRATEGY PARAMS ====================
    STRATEGY_PARAMS: Dict[str, Dict] = field(default_factory=lambda: {
        'SCALP_MOMENTUM': {
            'ema_fast': 5,
            'ema_slow': 13,
            'rsi_period': 7,
            'rsi_long_min': 50,
            'rsi_long_max': 70,
            'rsi_short_min': 30,
            'rsi_short_max': 50,
        },
        'BOLLINGER_MEAN': {
            'bb_period': 20,
            'bb_std': 2.0,
            'touch_threshold': 0.01,
        },
    })

    # ==================== Timing Settings ====================
    TRADING_START_HOUR: int = 0
    TRADING_END_HOUR: int = 24
    PRICE_UPDATE_INTERVAL: float = 1.0
    POSITION_CHECK_INTERVAL: float = 5.0
    STRATEGY_EVAL_INTERVAL: float = 60.0  # 1 minute candles

    # ==================== Logging ====================
    LOG_LEVEL: str = "INFO"
    LOG_TRADES: bool = True
    LOG_FILE: str = "btcc_high_winrate.log"
    ENABLE_NOTIFICATIONS: bool = False
    NOTIFICATION_WEBHOOK: str = ""

    # ==================== Paper Trading ====================
    PAPER_INITIAL_BALANCE: float = 10000.0
    PAPER_COMMISSION_PCT: float = 0.045  # 0.045% taker fee

    # ==================== Liquidation Settings ====================
    # At 20x leverage, liquidation occurs at ~5% move against position
    MAINTENANCE_MARGIN_RATE: float = 0.005  # 0.5%

    def get_liquidation_price(self, entry_price: float, direction: int, leverage: int = None) -> float:
        """Calculate liquidation price for a position."""
        lev = leverage or self.DEFAULT_LEVERAGE
        if direction == 1:  # Long
            return entry_price * (1 - (1 / lev) + self.MAINTENANCE_MARGIN_RATE)
        else:  # Short
            return entry_price * (1 + (1 / lev) - self.MAINTENANCE_MARGIN_RATE)

    # ==================== Helper Methods ====================
    def get_enabled_symbols(self) -> List[str]:
        return [s for s, cfg in self.SYMBOLS.items() if cfg.get('enabled', False)]

    def get_symbol_config(self, symbol: str) -> Optional[Dict]:
        return self.SYMBOLS.get(symbol)

    def get_strategy_params(self, strategy: str) -> Dict:
        return self.STRATEGY_PARAMS.get(strategy, {})

    def is_paper_mode(self) -> bool:
        return self.TRADING_MODE.lower() == 'paper'


# Pre-configured instances
HIGH_WINRATE_PAPER_CONFIG = BTCCHighWinRateConfig(
    TRADING_MODE="paper",
    PAPER_INITIAL_BALANCE=10000.0,
)

HIGH_WINRATE_LIVE_CONFIG = BTCCHighWinRateConfig(
    TRADING_MODE="live",
)
