"""
Application Configuration Manager
Handles persistent settings for the Trading Dashboard
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class OandaConfig:
    """OANDA API Configuration"""
    practice_api_key: str = ""
    practice_account_id: str = ""
    live_api_key: str = ""
    live_account_id: str = ""
    use_live: bool = False  # False = Practice, True = Live


@dataclass
class RiskConfig:
    """Risk Management Configuration"""
    initial_balance: float = 25000.0
    daily_loss_limit: float = 500.0
    profit_target: float = 1250.0
    max_drawdown: float = 1000.0
    max_trades_per_day: int = 50
    consistency_rule_enabled: bool = False


@dataclass
class PathConfig:
    """Application Paths Configuration"""
    ninjatrader_path: str = "C:\\Program Files\\NinjaTrader 8\\bin\\NinjaTrader.exe"
    logs_directory: str = ""
    bridge_path: str = ""


@dataclass
class AutoStartConfig:
    """Auto-start Options"""
    auto_launch_ninjatrader: bool = False
    auto_launch_bridge: bool = False
    auto_start_trading: bool = False


@dataclass
class TradingParams:
    """Trading Parameters per Symbol"""
    # These match pair_specific_settings.py
    symbols: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = {
                'M6E': {'tp_pips': 20, 'sl_pips': 16, 'ts_trigger': 12, 'ts_distance': 6, 'name': 'EUR/USD'},
                'M6B': {'tp_pips': 30, 'sl_pips': 25, 'ts_trigger': 18, 'ts_distance': 8, 'name': 'GBP/USD'},
                'MJY': {'tp_pips': 18, 'sl_pips': 15, 'ts_trigger': 12, 'ts_distance': 6, 'name': 'USD/JPY'},
                'MCD': {'tp_pips': 20, 'sl_pips': 16, 'ts_trigger': 12, 'ts_distance': 6, 'name': 'USD/CAD'},
                'MSF': {'tp_pips': 15, 'sl_pips': 12, 'ts_trigger': 10, 'ts_distance': 5, 'name': 'USD/CHF'}
            }


class AppConfig:
    """
    Main Application Configuration Manager
    Handles loading, saving, and accessing all settings
    """

    CONFIG_FILENAME = "trading_config.json"

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize config manager

        Args:
            config_dir: Directory to store config file. Defaults to app directory.
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to same directory as this script
            self.config_dir = Path(__file__).parent

        self.config_file = self.config_dir / self.CONFIG_FILENAME

        # Initialize config sections
        self.oanda = OandaConfig()
        self.risk = RiskConfig()
        self.paths = PathConfig()
        self.auto_start = AutoStartConfig()
        self.trading_params = TradingParams()

        # Set default paths
        self._set_default_paths()

        # Load existing config if available
        self.load()

    def _set_default_paths(self):
        """Set default paths based on current installation"""
        # Bridge path
        bridge_path = self.config_dir / "NinjaTraderBridge.exe"
        if bridge_path.exists():
            self.paths.bridge_path = str(bridge_path)

        # Logs directory
        logs_dir = self.config_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        self.paths.logs_directory = str(logs_dir)

        # NinjaTrader path - check common locations
        nt_paths = [
            "C:\\Program Files\\NinjaTrader 8\\bin\\NinjaTrader.exe",
            "C:\\Program Files (x86)\\NinjaTrader 8\\bin\\NinjaTrader.exe"
        ]
        for path in nt_paths:
            if os.path.exists(path):
                self.paths.ninjatrader_path = path
                break

    def load(self) -> bool:
        """
        Load configuration from file

        Returns:
            True if loaded successfully, False if using defaults
        """
        if not self.config_file.exists():
            print(f"[CONFIG] No config file found at {self.config_file}, using defaults")
            return False

        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)

            # Load OANDA config
            if 'oanda' in data:
                self.oanda = OandaConfig(**data['oanda'])

            # Load Risk config
            if 'risk' in data:
                self.risk = RiskConfig(**data['risk'])

            # Load Paths config
            if 'paths' in data:
                self.paths = PathConfig(**data['paths'])

            # Load Auto-start config
            if 'auto_start' in data:
                self.auto_start = AutoStartConfig(**data['auto_start'])

            # Load Trading params
            if 'trading_params' in data:
                self.trading_params = TradingParams(symbols=data['trading_params'].get('symbols'))

            print(f"[CONFIG] Loaded configuration from {self.config_file}")
            return True

        except Exception as e:
            print(f"[CONFIG] Error loading config: {e}")
            return False

    def save(self) -> bool:
        """
        Save configuration to file

        Returns:
            True if saved successfully
        """
        try:
            data = {
                'oanda': asdict(self.oanda),
                'risk': asdict(self.risk),
                'paths': asdict(self.paths),
                'auto_start': asdict(self.auto_start),
                'trading_params': {
                    'symbols': self.trading_params.symbols
                },
                'last_updated': datetime.now().isoformat()
            }

            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[CONFIG] Saved configuration to {self.config_file}")
            return True

        except Exception as e:
            print(f"[CONFIG] Error saving config: {e}")
            return False

    def get_active_oanda_credentials(self) -> tuple:
        """
        Get the active OANDA credentials based on live/practice mode

        Returns:
            (api_key, account_id)
        """
        if self.oanda.use_live:
            return self.oanda.live_api_key, self.oanda.live_account_id
        else:
            return self.oanda.practice_api_key, self.oanda.practice_account_id

    def is_configured(self) -> bool:
        """
        Check if minimum configuration is set (OANDA credentials)

        Returns:
            True if OANDA API key is configured
        """
        api_key, account_id = self.get_active_oanda_credentials()
        return bool(api_key and account_id)

    def get_mode_string(self) -> str:
        """Get current mode as string"""
        return "LIVE" if self.oanda.use_live else "PRACTICE"

    def validate(self) -> list:
        """
        Validate configuration and return list of issues

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check OANDA credentials
        api_key, account_id = self.get_active_oanda_credentials()
        if not api_key:
            issues.append(f"OANDA {self.get_mode_string()} API Key is not set")
        if not account_id:
            issues.append(f"OANDA {self.get_mode_string()} Account ID is not set")

        # Check paths
        if not os.path.exists(self.paths.ninjatrader_path):
            issues.append(f"NinjaTrader not found at: {self.paths.ninjatrader_path}")

        if self.paths.bridge_path and not os.path.exists(self.paths.bridge_path):
            issues.append(f"Bridge not found at: {self.paths.bridge_path}")

        # Check risk settings
        if self.risk.daily_loss_limit <= 0:
            issues.append("Daily loss limit must be positive")
        if self.risk.profit_target <= 0:
            issues.append("Profit target must be positive")

        return issues


# Global config instance
_config_instance = None

def get_config() -> AppConfig:
    """Get the global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def reload_config() -> AppConfig:
    """Reload configuration from file"""
    global _config_instance
    _config_instance = AppConfig()
    return _config_instance
