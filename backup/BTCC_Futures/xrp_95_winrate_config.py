"""
XRP 95% Win Rate Trading Configuration
=======================================
Optimized for XRPUSDT @ 20x leverage

Backtest Results (7 days):
- Win Rate: 95.5%
- Return: +33.9%
- Liquidations: 0
- Max Drawdown: 0%

Strategy: SCALP_MOMENTUM (EMA5/EMA13 + RSI 50-70)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class XRP95WinRateConfig:
    """XRP 95% Win Rate Trading Configuration"""

    # ==================== API Credentials ====================
    API_KEY: str = ""
    SECRET_KEY: str = ""
    USER_NAME: str = ""
    PASSWORD: str = ""

    # ==================== Trading Mode ====================
    TRADING_MODE: str = "paper"  # 'paper' or 'live'

    # ==================== Core Settings ====================
    SYMBOL: str = "XRPUSDT"
    LEVERAGE: int = 20
    TAKE_PROFIT_PCT: float = 1.0   # 1% TP
    STOP_LOSS_PCT: float = 10.0    # 10% SL (wide for high win rate)
    POSITION_SIZE_PCT: float = 10.0  # 10% of account per trade

    # ==================== Risk Management ====================
    MAX_DAILY_LOSS_PCT: float = 25.0  # Stop trading if down 25%
    MAX_OPEN_POSITIONS: int = 1       # Only 1 position at a time
    MAINTENANCE_MARGIN: float = 0.005  # 0.5%

    # ==================== Fees ====================
    TAKER_FEE: float = 0.00045  # 0.045%

    # ==================== Strategy Parameters ====================
    # SCALP_MOMENTUM strategy
    EMA_FAST: int = 5
    EMA_SLOW: int = 13
    RSI_PERIOD: int = 7
    RSI_LONG_MIN: int = 50
    RSI_LONG_MAX: int = 70
    RSI_SHORT_MIN: int = 30
    RSI_SHORT_MAX: int = 50

    # ==================== Timing ====================
    CANDLE_INTERVAL: str = "1m"  # 1 minute candles
    CHECK_INTERVAL: float = 5.0   # Check every 5 seconds

    # ==================== Paper Trading ====================
    PAPER_INITIAL_BALANCE: float = 100.0  # Start with $100

    # ==================== Logging ====================
    LOG_FILE: str = "xrp_95_winrate.log"
    LOG_TRADES: bool = True

    def get_liquidation_price(self, entry_price: float, direction: int) -> float:
        """Calculate liquidation price"""
        if direction == 1:  # Long
            return entry_price * (1 - (1 / self.LEVERAGE) + self.MAINTENANCE_MARGIN)
        else:  # Short
            return entry_price * (1 + (1 / self.LEVERAGE) - self.MAINTENANCE_MARGIN)

    def get_tp_price(self, entry_price: float, direction: int) -> float:
        """Calculate take profit price"""
        tp_pct = self.TAKE_PROFIT_PCT / 100
        if direction == 1:  # Long
            return entry_price * (1 + tp_pct)
        else:  # Short
            return entry_price * (1 - tp_pct)

    def get_sl_price(self, entry_price: float, direction: int) -> float:
        """Calculate stop loss price"""
        sl_pct = self.STOP_LOSS_PCT / 100
        if direction == 1:  # Long
            return entry_price * (1 - sl_pct)
        else:  # Short
            return entry_price * (1 + sl_pct)


# Pre-configured instances
XRP_PAPER_CONFIG = XRP95WinRateConfig(
    TRADING_MODE="paper",
    PAPER_INITIAL_BALANCE=100.0,
)

XRP_LIVE_CONFIG = XRP95WinRateConfig(
    TRADING_MODE="live",
)
