"""
Trading Configuration for Forex ML System
==========================================

Centralized configuration using dataclasses for type safety.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OandaConfig:
    """OANDA API Configuration."""
    environment: str = 'practice'  # 'practice' for paper, 'live' for real

    # API endpoints
    practice_url: str = 'https://api-fxpractice.oanda.com'
    live_url: str = 'https://api-fxtrade.oanda.com'

    def __post_init__(self):
        # Load credentials based on environment
        # Uses same env vars as existing forex trading system
        if self.environment == 'practice':
            self._api_key = os.getenv('OANDA_PRACTICE_API_KEY', os.getenv('OANDA_API_KEY', ''))
            self._account_id = os.getenv('OANDA_PRACTICE_ACCOUNT_ID', os.getenv('OANDA_ACCOUNT_ID', ''))
        else:
            self._api_key = os.getenv('OANDA_LIVE_API_KEY', os.getenv('OANDA_API_KEY', ''))
            self._account_id = os.getenv('OANDA_LIVE_ACCOUNT_ID', os.getenv('OANDA_ACCOUNT_ID', ''))

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def base_url(self) -> str:
        return self.practice_url if self.environment == 'practice' else self.live_url

    def is_valid(self) -> bool:
        return bool(self._api_key and self._account_id)


@dataclass
class MLConfig:
    """ML Model Configuration."""
    # Ensemble settings
    confidence_threshold: float = 0.60  # 60% minimum confidence
    min_model_agreement: int = 5  # All 5 models must agree (5/5)
    total_models_for_display: int = 5  # Show agreement as X/5
    voting_method: str = 'soft'  # 'soft' or 'hard'
    use_dynamic_weights: bool = True

    # Model weights (must sum to 1.0)
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'random_forest': 0.20,
        'xgboost': 0.25,
        'lightgbm': 0.20,
        'catboost': 0.20,
        'neural_network': 0.15
    })

    # Random Forest hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 5

    # XGBoost hyperparameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # LightGBM hyperparameters
    lgb_n_estimators: int = 300
    lgb_max_depth: int = 10
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31

    # CatBoost hyperparameters
    cat_iterations: int = 300
    cat_depth: int = 8
    cat_learning_rate: float = 0.05

    # Neural Network hyperparameters
    nn_hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    nn_dropout_rate: float = 0.3
    nn_learning_rate: float = 0.001
    nn_epochs: int = 100
    nn_batch_size: int = 64

    # Training settings
    train_test_split: float = 0.8
    validation_split: float = 0.2
    walk_forward_windows: int = 5
    prediction_horizon: int = 5  # Predict 5 bars ahead


@dataclass
class DCALevel:
    """Single DCA level configuration."""
    trigger_pct: float  # Price drop % to trigger this DCA level
    multiplier: float   # Position size multiplier for this level


@dataclass
class DCAConfig:
    """Dollar Cost Averaging Configuration with TREND FILTER.

    NEW STRATEGY - 6 DCA LEVELS with 4-HOUR MAX HOLD

    Entry: Only trade WITH the trend (H1/H4 EMA50 vs EMA200)
    DCA: 6 levels with 8-pip spacing for maximum recovery potential
    Exit: 4 hours max hold - if no recovery, take loss and start over

    DCA Levels (8 pip intervals):
    - DCA Level 1: 8 pips loss -> add 1.25x
    - DCA Level 2: 16 pips loss -> add 1.50x
    - DCA Level 3: 24 pips loss -> add 1.75x
    - DCA Level 4: 32 pips loss -> add 2.00x
    - DCA Level 5: 40 pips loss -> add 2.25x
    - DCA Level 6: 48 pips loss -> add 2.50x
    - Initial SL: 55 pips (beyond all DCAs)
    - After DCA: 58 pips SL from ORIGINAL entry

    Trend Filter:
    - Uses ADX to measure trend strength
    - Uses EMA crossover to detect reversals
    - Won't DCA into strong trends against us
    - Waits for reversal confirmation before adding
    """
    enabled: bool = True

    # Use pip-based triggers instead of percentage
    use_pip_based_dca: bool = True

    # DISABLE pre-placed limit orders - use market orders with trend check
    use_pending_orders: bool = False

    # PIP-BASED TP/SL - STOP LOSS FAR ENOUGH FOR ALL 6 DCAs
    take_profit_pips: float = 5.0       # 5 pips take profit (FASTER WINS)
    trailing_stop_pips: float = 4.0     # 4 pips trailing stop (after in profit)
    initial_sl_pips: float = 40.0       # 40 pips initial SL (beyond DCA6 at 30p + buffer)

    # DCA trigger levels in PIPS - 6 LEVELS (TIGHT for 4H recovery)
    # L1-L4: Execute IMMEDIATELY when level reached (no confirmation)
    # L5-L6: Require reversal confirmation before executing
    dca_trigger_pips_1: float = 5.0     # 5 pips loss -> DCA 1 (IMMEDIATE)
    dca_trigger_pips_2: float = 10.0    # 10 pips loss -> DCA 2 (IMMEDIATE)
    dca_trigger_pips_3: float = 15.0    # 15 pips loss -> DCA 3 (IMMEDIATE)
    dca_trigger_pips_4: float = 20.0    # 20 pips loss -> DCA 4 (IMMEDIATE)
    dca_trigger_pips_5: float = 25.0    # 25 pips loss -> DCA 5 (REVERSAL REQUIRED)
    dca_trigger_pips_6: float = 30.0    # 30 pips loss -> DCA 6 (REVERSAL REQUIRED)

    # DCA size multipliers - 6 LEVELS for gradual recovery
    # Entry = 1.0x, then each DCA progressively larger
    # Total if all DCAs trigger: 1.0 + 1.25 + 1.5 + 1.75 + 2.0 + 2.25 + 2.5 = 12.25x
    dca_multiplier_1: float = 1.25      # DCA1: 125% of initial
    dca_multiplier_2: float = 1.50      # DCA2: 150% of initial
    dca_multiplier_3: float = 1.75      # DCA3: 175% of initial
    dca_multiplier_4: float = 2.00      # DCA4: 200% of initial
    dca_multiplier_5: float = 2.25      # DCA5: 225% of initial
    dca_multiplier_6: float = 2.50      # DCA6: 250% of initial

    # After any DCA entry (in pips) - SL stays FAR to allow all DCAs
    dca_profit_target_pips: float = 10.0  # 10 pips profit target after DCA (from avg)
    sl_after_dca_pips: float = 45.0       # 45 pips stop loss after DCA (beyond DCA6 at 30p)

    # TREND FILTER SETTINGS - ONLY FOR DCA L5-L6
    # L1-L4: Execute IMMEDIATELY (no trend check)
    # L5-L6: Require reversal confirmation
    use_trend_filter: bool = True         # Enable trend-based DCA filtering for L5-L6
    dca_immediate_levels: int = 4         # L1-L4 execute immediately, L5+ need reversal
    adx_strong_trend: float = 30.0        # ADX above this = strong trend, don't DCA
    adx_weak_trend: float = 20.0          # ADX below this = weak trend, safer to DCA
    ema_fast_period: int = 8              # Fast EMA for reversal detection
    ema_slow_period: int = 21             # Slow EMA for trend direction
    require_reversal_candle: bool = True  # Require bullish/bearish reversal candle (for L5-L6)
    min_reversal_strength: float = 0.3    # Minimum candle body ratio for reversal

    # SMART DCA TRAILING - Dynamic TP for 6 DCA levels
    # Take profit REDUCES as DCA level increases (exit faster with larger positions)
    # OANDA minimum is 5 pips - for high DCA levels we use MANUAL monitoring
    tp_pips_dca0: float = 5.0     # Initial entry: 5 pips TP (FASTER WINS)
    tp_pips_dca1: float = 5.0     # After DCA1: 5 pips TP
    tp_pips_dca2: float = 5.0     # After DCA2: 5 pips TP
    tp_pips_dca3: float = 5.0     # After DCA3: 5 pips TP (OANDA minimum)
    tp_pips_dca4: float = 5.0     # After DCA4: 5 pips (manual exit at 3p)
    tp_pips_dca5: float = 5.0     # After DCA5: 5 pips (manual exit at 2p)
    tp_pips_dca6: float = 5.0     # After DCA6: 5 pips (manual exit at 1p)

    # MANUAL TP for DCA4-6 (below OANDA 5 pip minimum)
    # We set 5 pips on OANDA as backup, but manually close at these levels
    manual_tp_dca3_pips: float = 4.0   # Close DCA3 at +4 pips
    manual_tp_dca4_pips: float = 3.0   # Close DCA4 at +3 pips
    manual_tp_dca5_pips: float = 2.0   # Close DCA5 at +2 pips
    manual_tp_dca6_pips: float = 1.0   # Close DCA6 at +1 pip (just cover spread)

    # BREAKEVEN PROTECTION for DCA 3-4 (large positions)
    use_breakeven_protection: bool = True
    breakeven_activation_pips: float = 3.0   # Lock breakeven when +3 pips from avg entry
    breakeven_buffer_pips: float = 1.0       # SL at breakeven + 1 pip buffer

    # MANUAL TRAILING for DCA positions (moves ALL trades together)
    use_manual_dca_trailing: bool = True
    manual_trailing_activation_pips: float = 5.0  # Activate after 5 pips profit from avg
    manual_trailing_distance_pips: float = 8.0    # Trail 8 pips behind price

    # EMERGENCY EXIT for DCA 4 (maximum exposure - get out quick)
    dca4_emergency_exit: bool = True
    dca4_exit_profit_pips: float = 3.0  # Close ALL at +3 pips profit (don't wait for TP)

    # ================================================================
    # SMART RECOVERY EXIT - Exit after DCA recovers to breakeven
    # ================================================================
    # After DCA, if we recover to breakeven/small profit, EXIT immediately
    # Don't wait for full TP - risk of another reversal is too high
    use_smart_recovery_exit: bool = True

    # DCA1-2: Exit at breakeven + small profit
    recovery_exit_dca1_pips: float = 2.0   # Exit at +2 pips after DCA1 recovery
    recovery_exit_dca2_pips: float = 1.5   # Exit at +1.5 pips after DCA2 recovery

    # DCA3-4: Exit at breakeven + buffer
    recovery_exit_dca3_pips: float = 1.0   # Exit at +1 pip after DCA3 recovery
    recovery_exit_dca4_pips: float = 0.5   # Exit at +0.5 pip after DCA4 recovery

    # DCA5-6: Exit at breakeven (just cover fees, preserve capital)
    recovery_exit_dca5_pips: float = 0.0   # Exit at breakeven after DCA5 recovery
    recovery_exit_dca6_pips: float = 0.0   # Exit at breakeven after DCA6 recovery

    # TIME-BASED RECOVERY EXIT - For positions held too long
    # If held > 8 hours with DCA, exit at small loss to preserve capital
    use_time_based_recovery: bool = True
    max_hold_hours: float = 8.0            # Close if held > 8 hours - GIVE MORE TIME FOR RECOVERY
    time_recovery_loss_pips: float = -5.0  # Accept up to -5 pips loss after 8h hold

    # Minimum time with DCA before recovery exit (avoid premature exits)
    min_dca_hold_minutes: float = 10.0     # Wait 10 min after DCA before recovery exit

    # Legacy percentage-based (kept for compatibility, not used when use_pip_based_dca=True)
    take_profit_pct: float = 0.03
    trailing_stop_pct: float = 0.014
    initial_sl_pct: float = 0.08
    dca_level_1: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.008, multiplier=1.25))
    dca_level_2: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.008, multiplier=1.50))
    dca_level_3: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.008, multiplier=1.75))
    dca_level_4: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.008, multiplier=2.00))
    dca_level_5: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.008, multiplier=2.25))
    dca_level_6: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.008, multiplier=2.50))
    dca_profit_target_pct: float = 0.03
    sl_after_dca_pct: float = 0.012

    # Max DCA entries per position - NOW 6 LEVELS
    max_dca_levels: int = 6

    # Initial position size divisor for DCA
    # Total exposure with all 6 DCAs = 1 + 1.25 + 1.5 + 1.75 + 2.0 + 2.25 + 2.5 = 12.25x
    # Divisor of 5 means initial = 20%, max total = ~2.45x normal
    initial_size_divisor: float = 5.0

    def get_dca_levels(self) -> List[DCALevel]:
        """Get all DCA levels as a list."""
        return [self.dca_level_1, self.dca_level_2, self.dca_level_3,
                self.dca_level_4, self.dca_level_5, self.dca_level_6]

    def get_total_dca_multiplier(self) -> float:
        """Get total multiplier if all DCA levels trigger."""
        if self.use_pip_based_dca:
            return (1.0 + self.dca_multiplier_1 + self.dca_multiplier_2 +
                    self.dca_multiplier_3 + self.dca_multiplier_4 +
                    self.dca_multiplier_5 + self.dca_multiplier_6)
        return 1.0 + sum(level.multiplier for level in self.get_dca_levels())

    def get_dca_trigger_pips(self, level: int) -> float:
        """Get DCA trigger in pips for a specific level (1-6)."""
        triggers = [self.dca_trigger_pips_1, self.dca_trigger_pips_2,
                    self.dca_trigger_pips_3, self.dca_trigger_pips_4,
                    self.dca_trigger_pips_5, self.dca_trigger_pips_6]
        if 1 <= level <= 6:
            return triggers[level - 1]
        return 0.0

    def get_dca_multiplier(self, level: int) -> float:
        """Get DCA multiplier for a specific level (1-6)."""
        multipliers = [self.dca_multiplier_1, self.dca_multiplier_2,
                       self.dca_multiplier_3, self.dca_multiplier_4,
                       self.dca_multiplier_5, self.dca_multiplier_6]
        if 1 <= level <= 6:
            return multipliers[level - 1]
        return 0.0

    def get_tp_for_dca_level(self, dca_level: int) -> float:
        """
        Get dynamic take profit based on DCA level (now 6 levels).

        TP REDUCES as DCA increases for faster exit:
        - Level 0 (initial): 10 pips
        - Level 1: 8 pips
        - Level 2: 6 pips
        - Level 3: 5 pips (OANDA minimum)
        - Level 4-6: 5 pips on OANDA, but MANUAL exit at lower levels

        Args:
            dca_level: Current DCA level (0-6)

        Returns:
            Take profit in pips (for OANDA order, manual TP is lower for high DCA)
        """
        tp_by_level = [
            self.tp_pips_dca0,  # 10 pips (initial)
            self.tp_pips_dca1,  # 8 pips
            self.tp_pips_dca2,  # 6 pips
            self.tp_pips_dca3,  # 5 pips (OANDA min)
            self.tp_pips_dca4,  # 5 pips (manual exit at 3p)
            self.tp_pips_dca5,  # 5 pips (manual exit at 2p)
            self.tp_pips_dca6,  # 5 pips (manual exit at 1p)
        ]
        if 0 <= dca_level <= 6:
            return tp_by_level[dca_level]
        return self.tp_pips_dca6  # Default to smallest for safety


@dataclass
class MasterMomentumConfig:
    """Master Momentum Entry Configuration.

    FAST ENTRY when trend and momentum ALIGN.
    No heavy ML consensus needed - simple, fast signals for scalping with DCA.

    Entry Logic:
    1. Momentum spike detected (>momentum_threshold in momentum_period bars)
    2. Trend aligns (EMA8 vs EMA21 matches momentum direction)
    3. RSI not extreme (30-70 range)
    4. ADX shows trend exists (>min_adx)

    Exit: Simple fixed rules from DCA config (no trend-aware TP adjustment)
    """
    enabled: bool = True  # Enable Master Momentum entry (bypass ML for aligned signals)

    # Momentum detection
    momentum_period: int = 3          # Bars to measure momentum (2-3 for fast detection)
    momentum_threshold: float = 0.08  # Min % move to trigger (0.08% = 8 pips in 3 bars - real momentum)

    # Trend alignment (EMA crossover)
    ema_fast_period: int = 8          # Fast EMA
    ema_slow_period: int = 21         # Slow EMA

    # RSI filter (avoid extremes)
    rsi_period: int = 14
    rsi_max_for_buy: float = 70.0     # Don't BUY if RSI > 70 (overbought)
    rsi_min_for_sell: float = 30.0    # Don't SELL if RSI < 30 (oversold)

    # ADX filter (trend must exist)
    adx_period: int = 14
    min_adx: float = 15.0             # Minimum ADX to enter (trend exists)

    # Signal cooldown
    cooldown_bars: int = 5            # Min bars between signals on same pair

    # Fallback to ML if momentum not aligned
    use_ml_fallback: bool = True      # If no momentum signal, check ML


@dataclass
class RiskConfig:
    """Risk Management Configuration."""
    # Position sizing
    position_size_pct: float = 0.02  # 2% of equity per trade
    max_position_value: float = 10000  # Max $10,000 per position
    max_positions: int = 8  # Max concurrent positions (one per symbol)

    # Stop loss / Take profit (in pips) - used when DCA disabled
    default_stop_loss_pips: float = 20
    default_take_profit_pips: float = 8
    use_atr_stops: bool = True  # Use ATR-based dynamic stops
    atr_stop_multiplier: float = 1.5
    atr_tp_multiplier: float = 1.0

    # Risk limits
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    max_trades_per_day: int = 50  # Increased for trend following
    cooldown_seconds: int = 0  # NO COOLDOWN - open immediately when position closes

    # Trailing stop (pip-based, used when DCA disabled)
    use_trailing_stop: bool = True
    trailing_stop_pips: float = 10
    trailing_activation_pips: float = 5  # Activate after 5 pips profit


@dataclass
class FeatureConfig:
    """Feature Engineering Configuration."""
    # Lookback periods for features
    lookback_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])

    # Moving average periods
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # RSI periods
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR period
    atr_period: int = 14

    # ADX period
    adx_period: int = 14

    # Stochastic settings
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3

    # Feature normalization
    use_z_score: bool = True
    z_score_window: int = 100


@dataclass
class BacktestConfig:
    """Backtesting Configuration."""
    initial_capital: float = 100000
    commission_rate: float = 0.00002  # 0.2 pips spread
    slippage_pips: float = 0.5

    # Walk-forward settings
    train_window_days: int = 60  # 60 days training
    test_window_days: int = 20   # 20 days testing
    min_train_samples: int = 5000

    # Data settings
    timeframe: str = '5m'  # 5-minute bars
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class TradingConfig:
    """Master Trading Configuration."""
    # Sub-configurations
    oanda: OandaConfig = field(default_factory=OandaConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    dca: DCAConfig = field(default_factory=DCAConfig)
    momentum: MasterMomentumConfig = field(default_factory=MasterMomentumConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Trading pairs - REDUCED TO 3 BEST LIQUID PAIRS
    # Focus on fewer pairs = better management and trend following
    symbols: List[str] = field(default_factory=lambda: [
        'EUR_USD', 'GBP_USD', 'USD_JPY'
    ])

    # Pair-specific settings (TP/SL in pips)
    pair_settings: Dict[str, Dict] = field(default_factory=lambda: {
        'EUR_USD': {'tp_pips': 8, 'sl_pips': 20, 'pip_value': 0.0001},
        'GBP_USD': {'tp_pips': 10, 'sl_pips': 25, 'pip_value': 0.0001},
        'USD_JPY': {'tp_pips': 8, 'sl_pips': 20, 'pip_value': 0.01},
        'USD_CHF': {'tp_pips': 8, 'sl_pips': 20, 'pip_value': 0.0001},
        'AUD_USD': {'tp_pips': 8, 'sl_pips': 20, 'pip_value': 0.0001},
        'USD_CAD': {'tp_pips': 8, 'sl_pips': 20, 'pip_value': 0.0001},
        'NZD_USD': {'tp_pips': 8, 'sl_pips': 20, 'pip_value': 0.0001},
        'EUR_GBP': {'tp_pips': 6, 'sl_pips': 15, 'pip_value': 0.0001},
    })

    # Timeframe
    timeframe: str = '5m'  # Default 5-minute bars
    data_buffer_size: int = 500  # Keep 500 bars in memory

    # Trading hours (UTC)
    trading_start_hour: int = 8   # 8 AM UTC
    trading_end_hour: int = 20    # 8 PM UTC

    # Logging
    log_dir: str = 'logs'
    save_trades: bool = True
    verbose: bool = True

    def get_pair_settings(self, symbol: str) -> Dict:
        """Get settings for a specific pair."""
        return self.pair_settings.get(symbol, {
            'tp_pips': 8,
            'sl_pips': 20,
            'pip_value': 0.0001
        })

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol."""
        settings = self.get_pair_settings(symbol)
        return settings.get('pip_value', 0.0001)


def load_config() -> TradingConfig:
    """Load trading configuration."""
    return TradingConfig()


def print_config_info(config: TradingConfig):
    """Print configuration summary."""
    print("=" * 80)
    print("FOREX ML TRADING SYSTEM - Configuration")
    print("=" * 80)
    print(f"\nSymbols: {', '.join(config.symbols)}")
    print(f"Timeframe: {config.timeframe}")
    print(f"\nML Settings:")
    print(f"  Confidence Threshold: {config.ml.confidence_threshold:.0%}")
    total_display = getattr(config.ml, 'total_models_for_display', 3)
    print(f"  Min Model Agreement: {config.ml.min_model_agreement}/{total_display}")
    print(f"  Voting Method: {config.ml.voting_method}")
    print(f"\nRisk Settings:")
    print(f"  Position Size: {config.risk.position_size_pct:.1%} of equity")
    print(f"  Max Daily Loss: {config.risk.max_daily_loss_pct:.1%}")
    print(f"  Max Trades/Day: {config.risk.max_trades_per_day}")
    print(f"  Cooldown: {config.risk.cooldown_seconds}s")
    print(f"\nDCA Settings: {'ENABLED' if config.dca.enabled else 'DISABLED'}")
    if config.dca.enabled:
        print(f"  Take Profit: {config.dca.take_profit_pct:.1%}")
        initial_sl = getattr(config.dca, 'initial_sl_pct', 0.08)
        print(f"  Initial SL: {initial_sl:.1%} (wide, allows all DCAs)")
        print(f"  Trailing Stop: {config.dca.trailing_stop_pct:.1%} (after in profit)")
        print(f"  Initial Size: {100/config.dca.initial_size_divisor:.0f}% of normal (divisor={config.dca.initial_size_divisor:.0f})")
        print(f"  DCA Levels:")
        for i, level in enumerate(config.dca.get_dca_levels(), 1):
            print(f"    Level {i}: Trigger {level.trigger_pct:.1%}, Mult {level.multiplier:.2f}x")
        total_mult = config.dca.get_total_dca_multiplier()
        max_exposure = total_mult / config.dca.initial_size_divisor
        print(f"  Max Exposure: {max_exposure:.2f}x normal (if all DCAs trigger)")
        print(f"  SL after DCA: {config.dca.sl_after_dca_pct:.1%} (tighter, from avg entry)")

    # Master Momentum Settings
    print(f"\nMaster Momentum: {'ENABLED' if config.momentum.enabled else 'DISABLED'}")
    if config.momentum.enabled:
        print(f"  >>> FAST ENTRY when Trend + Momentum ALIGN <<<")
        print(f"  Momentum Period: {config.momentum.momentum_period} bars")
        print(f"  Momentum Threshold: {config.momentum.momentum_threshold}% move")
        print(f"  Trend EMAs: EMA{config.momentum.ema_fast_period} vs EMA{config.momentum.ema_slow_period}")
        print(f"  RSI Period: {config.momentum.rsi_period}")
        print(f"  RSI Filters: BUY if RSI < {config.momentum.rsi_max_for_buy}, SELL if RSI > {config.momentum.rsi_min_for_sell}")
        print(f"  Min ADX: {config.momentum.min_adx} (trend strength required)")
        print(f"  Signal Cooldown: {config.momentum.cooldown_bars} bars")
        print(f"  ML Fallback: {'YES - use ML if no momentum signal' if config.momentum.use_ml_fallback else 'NO - momentum only'}")
        print(f"  Entry Logic:")
        print(f"    1. Momentum spike > {config.momentum.momentum_threshold}% in {config.momentum.momentum_period} bars")
        print(f"    2. Trend aligns (EMA{config.momentum.ema_fast_period} vs EMA{config.momentum.ema_slow_period} matches direction)")
        print(f"    3. RSI not extreme (30-70 range)")
        print(f"    4. ADX > {config.momentum.min_adx} (trend exists)")
        print(f"    >>> When ALL 4 align = ENTER IMMEDIATELY <<<")
    print("=" * 80)
