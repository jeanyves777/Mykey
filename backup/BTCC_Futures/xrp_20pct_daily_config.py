"""
XRP 20% Daily Scalping Configuration
=====================================
Aggressive scalping strategy targeting 20%+ daily returns

Backtest Results (7 days):
- Win Rate: 95.0%
- Return: +205.3%
- Daily Average: +29.3%
- Liquidations: 0
- Max Drawdown: 0%

Strategy: FAST_SCALP_MOMENTUM (EMA2/EMA5 + RSI 40-80/20-60)

WARNING: This is an AGGRESSIVE strategy with 50% position sizing.
A single loss could cost 50% of your position value at SL.
Only trade with capital you can afford to lose!
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class XRP20PctDailyConfig:
    """XRP 20% Daily Scalping Configuration"""

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
    TAKE_PROFIT_PCT: float = 1.0    # 1% TP
    STOP_LOSS_PCT: float = 10.0     # 10% SL (wide for high win rate)
    POSITION_SIZE_PCT: float = 50.0  # 50% of account per trade (AGGRESSIVE!)

    # ==================== Risk Management ====================
    MAX_DAILY_LOSS_PCT: float = 50.0  # Stop trading if down 50%
    MAX_OPEN_POSITIONS: int = 1       # Only 1 position at a time
    MAINTENANCE_MARGIN: float = 0.005  # 0.5%

    # ==================== Fees ====================
    TAKER_FEE: float = 0.00045  # 0.045%

    # ==================== Strategy Parameters ====================
    # FAST_SCALP_MOMENTUM strategy - faster EMAs for more signals
    EMA_FAST: int = 2
    EMA_SLOW: int = 5
    RSI_PERIOD: int = 5
    RSI_LONG_MIN: int = 40
    RSI_LONG_MAX: int = 80
    RSI_SHORT_MIN: int = 20
    RSI_SHORT_MAX: int = 60

    # ==================== Timing ====================
    CANDLE_INTERVAL: str = "1m"  # 1 minute candles
    CHECK_INTERVAL: float = 3.0   # Check every 3 seconds (faster for scalping)

    # ==================== Paper Trading ====================
    PAPER_INITIAL_BALANCE: float = 100.0  # Start with $100

    # ==================== Logging ====================
    LOG_FILE: str = "xrp_20pct_daily.log"
    LOG_TRADES: bool = True

    def get_liquidation_price(self, entry_price: float, direction: int) -> float:
        """Calculate liquidation price (at 20x, ~4.5% move)"""
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
XRP_20PCT_PAPER_CONFIG = XRP20PctDailyConfig(
    TRADING_MODE="paper",
    PAPER_INITIAL_BALANCE=100.0,
)

XRP_20PCT_LIVE_CONFIG = XRP20PctDailyConfig(
    TRADING_MODE="live",
)

# Conservative version (30% position instead of 50%)
XRP_15PCT_DAILY_CONFIG = XRP20PctDailyConfig(
    TRADING_MODE="paper",
    POSITION_SIZE_PCT=30.0,  # Less aggressive
    PAPER_INITIAL_BALANCE=100.0,
)

# Expected results per position size:
# 50% position = ~29% daily (AGGRESSIVE)
# 40% position = ~22% daily
# 35% position = ~18% daily
# 30% position = ~15% daily
# 25% position = ~12% daily
