"""
Crypto Paper Trading Configuration Module

Handles configuration for crypto paper trading including:
- API credentials (Alpaca)
- Position sizing
- Risk parameters
- Symbol selection
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from pathlib import Path


# Configuration file path
CONFIG_DIR = Path.home() / ".thevolumeai"
CRYPTO_CONFIG_FILE = CONFIG_DIR / "crypto_paper_trading_config.json"


# Default crypto symbols to trade (V18: 5 symbols optimized with ML ensemble)
# These are the symbols the ML models were trained on for highest accuracy
DEFAULT_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"
]


@dataclass
class CryptoPaperTradingConfig:
    """Configuration for crypto paper trading."""

    # Alpaca API credentials
    api_key: str = ""
    api_secret: str = ""

    # Use paper trading endpoint (always True for paper trading)
    use_paper: bool = True

    # Trading parameters - position size MUST be set by user
    fixed_position_value: float = 0.0  # Fixed $ amount per trade (user must set)

    # Symbols to trade (5 ML-optimized symbols)
    symbols: List[str] = field(default_factory=lambda: DEFAULT_CRYPTO_SYMBOLS.copy())

    # Risk parameters - V18 OPTIMAL: From ML ensemble backtest optimization
    # V18 Results: 93.1% WR (27W/2L), 29 trades, +$1.61, PF=11.69, Sharpe=75.46
    # Key: Tight SL/TP with 5/5 model agreement for highest confidence
    target_profit_pct: float = 0.8    # V18 OPTIMAL: 0.8% TP (quick profits)
    stop_loss_pct: float = 0.6        # V18 OPTIMAL: 0.6% SL (tight stops)
    trailing_stop_pct: float = 0.0    # V18: Disabled - use fixed TP/SL only
    trailing_stop_activation: float = 999.0  # V18: Never activate
    use_trailing_stop: bool = False   # V18: Disabled

    # Trading hours (UTC) - crypto trades 24/7 but these are peak hours
    # Peak: 13:00-21:00 UTC (US trading hours), 00:00-08:00 UTC (Asian hours)
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(
        default_factory=lambda: list(range(0, 9)) + list(range(13, 22))
    )

    # Position management - max 5 (one per symbol for all 5 ML-optimized cryptos)
    max_concurrent_positions: int = 5
    max_trades_per_hour: int = 20  # Increased for more data collection

    # Hold time limits - V17: No max hold - wait for TP or SL
    max_hold_minutes: int = 0  # V17: 0 = disabled, wait for TP or SL
    min_hold_minutes: int = 1   # V17: Minimum 1 min before SL exit

    # Entry requirements - V17 DATA COLLECTION: Lower score for more trades
    min_entry_score: int = 5  # Lowered to 5 for more frequent trades & data collection

    # Alpaca crypto fee (0.25% taker)
    taker_fee_pct: float = 0.25

    def is_configured(self) -> bool:
        """Check if API credentials and position size are configured."""
        return bool(self.api_key and self.api_secret and self.fixed_position_value > 0)

    def save(self) -> None:
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CRYPTO_CONFIG_FILE, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"\nConfiguration saved to: {CRYPTO_CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'CryptoPaperTradingConfig':
        """Load configuration from file."""
        if CRYPTO_CONFIG_FILE.exists():
            try:
                with open(CRYPTO_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load config file: {e}")
                return cls()
        return cls()

    @classmethod
    def exists(cls) -> bool:
        """Check if configuration file exists."""
        return CRYPTO_CONFIG_FILE.exists()

    def get_active_symbols(self) -> List[str]:
        """Get list of symbols to actively trade."""
        return self.symbols if self.symbols else DEFAULT_CRYPTO_SYMBOLS


def mask_api_key(key: str) -> str:
    """Mask API key for display (show first 4 and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]
