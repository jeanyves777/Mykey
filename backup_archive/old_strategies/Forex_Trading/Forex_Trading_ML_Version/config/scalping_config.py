"""
M1 Scalping Configuration for OANDA
=====================================

Optimized pairs and sessions based on 30-day backtest results.
Each pair trades ONLY in its best-performing session.

BACKTEST RESULTS (30 days, $5/pip):
- Portfolio Return: +70.8%
- Win Rate: 58.3%
- 8 out of 10 pairs profitable

TOP 7 WINNING PAIRS:
1. USD_JPY @ LONDON  -> +240.6%, 66.3% win rate
2. USD_CAD @ ASIAN   -> +135.5%, 65.1% win rate
3. USD_CHF @ ASIAN   -> +115.1%, 67.3% win rate
4. EUR_JPY @ ASIAN   -> +86.5%, 56.3% win rate
5. EUR_GBP @ LONDON  -> +61.8%, 67.5% win rate
6. GBP_JPY @ ASIAN   -> +51.3%, 53.5% win rate
7. EUR_CAD @ LONDON  -> +50.2%, 56.6% win rate

AVOID: EUR_USD, GBP_USD, NZD_USD (losing or breakeven)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PairConfig:
    """Configuration for a single trading pair."""
    symbol: str
    session: str                    # ASIAN, LONDON, NY, OVERLAP
    session_hours: Tuple[int, int]  # (start_hour, end_hour) UTC
    tp_pips: float = 15.0          # Take profit
    sl_pips: float = 10.0          # Stop loss
    trailing_activation: float = 8.0  # Activate trailing at +8 pips
    trailing_distance: float = 3.0    # Trail 3 pips behind
    pip_value: float = 0.0001      # Pip value (0.01 for JPY pairs)
    enabled: bool = True           # Enable/disable this pair


@dataclass
class ScalpingConfig:
    """
    M1 Scalping Configuration for OANDA Live Trading.

    Based on 30-day backtest optimization.
    Each pair trades only during its winning session.
    """

    # OANDA Settings
    environment: str = 'practice'  # 'practice' or 'live'

    # Position Sizing
    dollars_per_pip: float = 5.0   # $5 per pip (adjust based on account size)
    max_positions: int = 3         # Max concurrent positions
    max_daily_losses: int = 3      # Stop trading after 3 losses in a day

    # Signal Filters (from backtest optimization)
    adx_threshold: float = 25.0    # ADX > 25 for trend
    atr_expansion_mult: float = 1.2  # ATR > 1.2x average
    di_difference_min: float = 5.0   # |+DI - -DI| > 5 for strong signal

    # Higher Timeframe Confirmation
    htf_ema_fast: int = 50         # EMA 50 on M1 (simulates H1 trend)
    htf_ema_slow: int = 200        # EMA 200 on M1

    # TOP 7 WINNING PAIRS with their optimal sessions
    pairs: Dict[str, PairConfig] = field(default_factory=lambda: {
        # ================================================================
        # TIER 1: BEST PERFORMERS (>100% return in backtest)
        # ================================================================
        'USD_JPY': PairConfig(
            symbol='USD_JPY',
            session='LONDON',
            session_hours=(8, 16),  # 08:00-16:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.01,  # JPY pair
            enabled=True
        ),
        'USD_CAD': PairConfig(
            symbol='USD_CAD',
            session='ASIAN',
            session_hours=(0, 8),  # 00:00-08:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.0001,
            enabled=True
        ),
        'USD_CHF': PairConfig(
            symbol='USD_CHF',
            session='ASIAN',
            session_hours=(0, 8),  # 00:00-08:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.0001,
            enabled=True
        ),

        # ================================================================
        # TIER 2: GOOD PERFORMERS (50-100% return in backtest)
        # ================================================================
        'EUR_JPY': PairConfig(
            symbol='EUR_JPY',
            session='ASIAN',
            session_hours=(0, 8),  # 00:00-08:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.01,  # JPY pair
            enabled=True
        ),
        'EUR_GBP': PairConfig(
            symbol='EUR_GBP',
            session='LONDON',
            session_hours=(8, 16),  # 08:00-16:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.0001,
            enabled=True
        ),
        'GBP_JPY': PairConfig(
            symbol='GBP_JPY',
            session='ASIAN',
            session_hours=(0, 8),  # 00:00-08:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.01,  # JPY pair
            enabled=True
        ),
        'EUR_CAD': PairConfig(
            symbol='EUR_CAD',
            session='LONDON',
            session_hours=(8, 16),  # 08:00-16:00 UTC
            tp_pips=15.0,
            sl_pips=10.0,
            trailing_activation=8.0,
            trailing_distance=3.0,
            pip_value=0.0001,
            enabled=True
        ),

        # ================================================================
        # DISABLED PAIRS (losing in backtest - DO NOT TRADE)
        # ================================================================
        'EUR_USD': PairConfig(
            symbol='EUR_USD',
            session='LONDON',
            session_hours=(8, 16),
            pip_value=0.0001,
            enabled=False  # -9.7% in backtest
        ),
        'GBP_USD': PairConfig(
            symbol='GBP_USD',
            session='ASIAN',
            session_hours=(0, 8),
            pip_value=0.0001,
            enabled=False  # -23.4% in backtest
        ),
        'NZD_USD': PairConfig(
            symbol='NZD_USD',
            session='LONDON',
            session_hours=(8, 16),
            pip_value=0.0001,
            enabled=False  # 0% in backtest (breakeven)
        ),
    })

    @property
    def api_key(self) -> str:
        """Get OANDA API key from environment."""
        if self.environment == 'practice':
            return os.getenv('OANDA_PRACTICE_API_KEY', os.getenv('OANDA_API_KEY', ''))
        return os.getenv('OANDA_LIVE_API_KEY', os.getenv('OANDA_API_KEY', ''))

    @property
    def account_id(self) -> str:
        """Get OANDA account ID from environment."""
        if self.environment == 'practice':
            return os.getenv('OANDA_PRACTICE_ACCOUNT_ID', os.getenv('OANDA_ACCOUNT_ID', ''))
        return os.getenv('OANDA_LIVE_ACCOUNT_ID', os.getenv('OANDA_ACCOUNT_ID', ''))

    @property
    def base_url(self) -> str:
        """Get OANDA API base URL."""
        if self.environment == 'practice':
            return 'https://api-fxpractice.oanda.com'
        return 'https://api-fxtrade.oanda.com'

    def get_enabled_pairs(self) -> List[PairConfig]:
        """Get list of enabled pairs for trading."""
        return [p for p in self.pairs.values() if p.enabled]

    def get_asian_pairs(self) -> List[PairConfig]:
        """Get pairs that trade in Asian session."""
        return [p for p in self.pairs.values() if p.enabled and p.session == 'ASIAN']

    def get_london_pairs(self) -> List[PairConfig]:
        """Get pairs that trade in London session."""
        return [p for p in self.pairs.values() if p.enabled and p.session == 'LONDON']

    def is_pair_in_session(self, symbol: str, current_hour: int) -> bool:
        """Check if a pair should be trading at the current hour (UTC)."""
        if symbol not in self.pairs:
            return False

        pair = self.pairs[symbol]
        if not pair.enabled:
            return False

        start, end = pair.session_hours
        if start < end:
            return start <= current_hour < end
        else:
            # Handles overnight sessions (not used currently)
            return current_hour >= start or current_hour < end

    def get_pairs_for_hour(self, current_hour: int) -> List[PairConfig]:
        """Get all pairs that should be trading at the current hour."""
        return [p for p in self.pairs.values()
                if p.enabled and self.is_pair_in_session(p.symbol, current_hour)]


def load_scalping_config() -> ScalpingConfig:
    """Load scalping configuration."""
    return ScalpingConfig()


def print_scalping_config(config: ScalpingConfig):
    """Print scalping configuration summary."""
    print("=" * 80)
    print("M1 SCALPING CONFIGURATION - OANDA LIVE TRADING")
    print("=" * 80)
    print(f"\nEnvironment: {config.environment.upper()}")
    print(f"Position Size: ${config.dollars_per_pip}/pip")
    print(f"Max Positions: {config.max_positions}")
    print(f"Max Daily Losses: {config.max_daily_losses}")

    print(f"\nSignal Filters:")
    print(f"  ADX Threshold: > {config.adx_threshold}")
    print(f"  ATR Expansion: > {config.atr_expansion_mult}x avg")
    print(f"  DI Difference: > {config.di_difference_min}")
    print(f"  HTF Trend: EMA{config.htf_ema_fast} vs EMA{config.htf_ema_slow}")

    print(f"\n{'='*80}")
    print("ENABLED PAIRS (by session)")
    print(f"{'='*80}")

    # Asian session pairs
    asian_pairs = config.get_asian_pairs()
    if asian_pairs:
        print(f"\nASIAN SESSION (00:00-08:00 UTC):")
        for p in asian_pairs:
            print(f"  {p.symbol}: TP {p.tp_pips}p / SL {p.sl_pips}p / Trail +{p.trailing_activation}p")

    # London session pairs
    london_pairs = config.get_london_pairs()
    if london_pairs:
        print(f"\nLONDON SESSION (08:00-16:00 UTC):")
        for p in london_pairs:
            print(f"  {p.symbol}: TP {p.tp_pips}p / SL {p.sl_pips}p / Trail +{p.trailing_activation}p")

    # Disabled pairs
    disabled = [p for p in config.pairs.values() if not p.enabled]
    if disabled:
        print(f"\nDISABLED PAIRS (losing in backtest):")
        for p in disabled:
            print(f"  {p.symbol} - DO NOT TRADE")

    print(f"\n{'='*80}")
    print(f"Total Enabled Pairs: {len(config.get_enabled_pairs())}")
    print(f"{'='*80}")


if __name__ == "__main__":
    config = load_scalping_config()
    print_scalping_config(config)
