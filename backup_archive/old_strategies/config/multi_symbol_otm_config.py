"""
Multi-Symbol OTM Day Trading Configuration

Weekly OTM Day Trading Plan — $200 Account
Symbols: SPY, QQQ, AMD, IWM, PLTR, BAC

Day-specific parameters for DTE, strike distance, budget, target, and stop loss.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import time


@dataclass
class DayConfig:
    """Configuration for a specific day of the week."""
    dte: int  # Days to expiration
    strike_otm: int  # How many strikes OTM (1 = 1 OTM, 2 = 2 OTM)
    strike_otm_max: int  # Max OTM strikes (for range like 1-2)
    budget: float  # Daily budget in dollars
    target_pct: float  # Target profit percentage
    stop_loss_pct: float  # Stop loss percentage


@dataclass
class MultiSymbolOTMConfig:
    """Configuration for Multi-Symbol OTM Day Trading Strategy."""

    # Account settings
    total_account: float = 200.0

    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "AMD", "IWM", "PLTR", "BAC"
    ])

    # Day-specific configurations (0=Monday, 4=Friday)
    day_configs: Dict[int, DayConfig] = field(default_factory=lambda: {
        0: DayConfig(dte=4, strike_otm=1, strike_otm_max=1, budget=50.0, target_pct=30.0, stop_loss_pct=30.0),  # Monday
        1: DayConfig(dte=3, strike_otm=1, strike_otm_max=1, budget=50.0, target_pct=35.0, stop_loss_pct=35.0),  # Tuesday
        2: DayConfig(dte=2, strike_otm=1, strike_otm_max=2, budget=50.0, target_pct=40.0, stop_loss_pct=40.0),  # Wednesday
        3: DayConfig(dte=1, strike_otm=1, strike_otm_max=2, budget=50.0, target_pct=50.0, stop_loss_pct=45.0),  # Thursday
        4: DayConfig(dte=0, strike_otm=1, strike_otm_max=2, budget=60.0, target_pct=75.0, stop_loss_pct=50.0),  # Friday
    })

    # Trading hours
    entry_time_start: time = field(default_factory=lambda: time(9, 35))  # 9:35 AM - after open volatility
    entry_time_end: time = field(default_factory=lambda: time(15, 30))  # 3:30 PM - before close
    force_exit_time: time = field(default_factory=lambda: time(15, 50))  # 3:50 PM - force exit

    # Position management
    max_positions: int = 2  # Max simultaneous positions
    max_positions_per_symbol: int = 1  # Max positions per symbol

    # Signal thresholds (using V4 pullback detection)
    min_pullback_score: int = 3
    min_recovery_score: int = 4

    # EMA settings for trend detection
    ema_period: int = 20

    # Minimum option price
    min_option_price: float = 0.10  # Minimum $0.10 per contract
    max_option_price: float = 1.00  # Maximum $1.00 per contract (for $200 account)

    # API settings (loaded from environment or config file)
    api_key: str = ""
    api_secret: str = ""
    use_paper: bool = True

    def get_day_config(self, weekday: int) -> DayConfig:
        """Get configuration for a specific weekday (0=Monday, 4=Friday)."""
        return self.day_configs.get(weekday, self.day_configs[2])  # Default to Wednesday

    def get_daily_budget(self, weekday: int) -> float:
        """Get budget for a specific day."""
        return self.get_day_config(weekday).budget

    def get_target_pct(self, weekday: int) -> float:
        """Get target profit percentage for a specific day."""
        return self.get_day_config(weekday).target_pct

    def get_stop_loss_pct(self, weekday: int) -> float:
        """Get stop loss percentage for a specific day."""
        return self.get_day_config(weekday).stop_loss_pct

    def get_dte(self, weekday: int) -> int:
        """Get DTE for a specific day."""
        return self.get_day_config(weekday).dte

    def get_strike_otm_range(self, weekday: int) -> tuple:
        """Get strike OTM range for a specific day."""
        config = self.get_day_config(weekday)
        return (config.strike_otm, config.strike_otm_max)


# Default configuration instance
DEFAULT_CONFIG = MultiSymbolOTMConfig()


def load_config_from_file(config_path: str) -> MultiSymbolOTMConfig:
    """Load configuration from a JSON file."""
    import json

    config = MultiSymbolOTMConfig()

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Load API credentials
        config.api_key = data.get('api_key', '')
        config.api_secret = data.get('api_secret', '')
        config.use_paper = data.get('use_paper', True)

        # Load account settings
        config.total_account = data.get('total_account', 200.0)

        # Load symbols if provided
        if 'symbols' in data:
            config.symbols = data['symbols']

        # Load day configs if provided
        if 'day_configs' in data:
            for day_str, day_data in data['day_configs'].items():
                day_num = int(day_str)
                config.day_configs[day_num] = DayConfig(
                    dte=day_data.get('dte', 2),
                    strike_otm=day_data.get('strike_otm', 1),
                    strike_otm_max=day_data.get('strike_otm_max', 1),
                    budget=day_data.get('budget', 50.0),
                    target_pct=day_data.get('target_pct', 40.0),
                    stop_loss_pct=day_data.get('stop_loss_pct', 40.0)
                )

        # Load position settings
        config.max_positions = data.get('max_positions', 2)
        config.max_positions_per_symbol = data.get('max_positions_per_symbol', 1)

        # Load option price limits
        config.min_option_price = data.get('min_option_price', 0.10)
        config.max_option_price = data.get('max_option_price', 1.00)

    except Exception as e:
        print(f"[WARNING] Could not load config from {config_path}: {e}")
        print("[INFO] Using default configuration")

    return config
