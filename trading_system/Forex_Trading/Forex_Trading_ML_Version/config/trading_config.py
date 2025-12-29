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

    SMART DCA - Only add to positions when market shows reversal signs.
    NO pre-placed limit orders. Check trend at each level before entering.

    WIDER DCA spacing (10 pip intervals):
    - DCA Level 1: 10 pips loss -> add 0.50x (if trend weakening)
    - DCA Level 2: 20 pips loss -> add 1.00x (if reversal forming)
    - DCA Level 3: 30 pips loss -> add 1.50x (if strong reversal signal)
    - DCA Level 4: 40 pips loss -> add 2.00x (only if confirmed reversal)
    - Initial SL: 50 pips (well beyond all DCAs at 40 pips)
    - After DCA: 55 pips SL from ORIGINAL entry

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
    use_pending_orders: bool = False  # Changed from True to False

    # PIP-BASED TP/SL (WIDER STOPS)
    take_profit_pips: float = 12.0      # 12 pips take profit (wider)
    trailing_stop_pips: float = 8.0     # 8 pips trailing stop (after in profit)
    initial_sl_pips: float = 50.0       # 50 pips initial SL (well beyond all DCAs)

    # DCA trigger levels in PIPS - WIDER 10 PIP INTERVALS
    dca_trigger_pips_1: float = 10.0    # 10 pips loss -> DCA 1
    dca_trigger_pips_2: float = 20.0    # 20 pips loss -> DCA 2
    dca_trigger_pips_3: float = 30.0    # 30 pips loss -> DCA 3
    dca_trigger_pips_4: float = 40.0    # 40 pips loss -> DCA 4

    # DCA size multipliers - EACH DCA BIGGER THAN PREVIOUS FOR FAST RECOVERY
    # Entry = 1.0x (base), then each DCA is LARGER to pull average closer to current price
    # Total if all DCAs trigger: 1.0 + 1.5 + 2.0 + 2.5 + 3.0 = 10.0x
    # Use initial_size_divisor to scale down entry so total stays within risk
    dca_multiplier_1: float = 1.50      # DCA1: 150% of initial - pulls avg ~40% toward current
    dca_multiplier_2: float = 2.00      # DCA2: 200% of initial - pulls avg ~45% toward current
    dca_multiplier_3: float = 2.50      # DCA3: 250% of initial - pulls avg ~50% toward current
    dca_multiplier_4: float = 3.00      # DCA4: 300% of initial - pulls avg ~55% toward current

    # After any DCA entry (in pips)
    dca_profit_target_pips: float = 12.0  # 12 pips profit target after DCA (from avg)
    sl_after_dca_pips: float = 55.0       # 55 pips stop loss after DCA (from ORIGINAL)

    # TREND FILTER SETTINGS
    use_trend_filter: bool = True         # Enable trend-based DCA filtering
    adx_strong_trend: float = 30.0        # ADX above this = strong trend, don't DCA
    adx_weak_trend: float = 20.0          # ADX below this = weak trend, safer to DCA
    ema_fast_period: int = 8              # Fast EMA for reversal detection
    ema_slow_period: int = 21             # Slow EMA for trend direction
    require_reversal_candle: bool = True  # Require bullish/bearish reversal candle
    min_reversal_strength: float = 0.3    # Minimum candle body ratio for reversal

    # SMART DCA TRAILING - Dynamic TP and protection for DCA positions
    # Take profit REDUCES AGGRESSIVELY as DCA level increases (exit faster with larger positions)
    # OANDA minimum is 5 pips - for DCA3-4 we use MANUAL monitoring and market close
    tp_pips_dca0: float = 8.0     # Initial entry: 8 pips TP
    tp_pips_dca1: float = 6.0     # After DCA1: 6 pips TP
    tp_pips_dca2: float = 5.0     # After DCA2: 5 pips TP (OANDA minimum)
    tp_pips_dca3: float = 5.0     # After DCA3: Set 5p on OANDA, but MANUAL EXIT at 3p
    tp_pips_dca4: float = 5.0     # After DCA4: Set 5p on OANDA, but MANUAL EXIT at 2p

    # MANUAL TP for DCA3-4 (below OANDA 5 pip minimum)
    # We set 5 pips on OANDA as backup, but manually close at these levels
    manual_tp_dca3_pips: float = 3.0   # Close DCA3 at +3 pips (manual market order)
    manual_tp_dca4_pips: float = 2.0   # Close DCA4 at +2 pips (manual market order)

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

    # DCA1-2: Exit at breakeven + 1 pip (small recovery)
    recovery_exit_dca1_pips: float = 1.0   # Exit at +1 pip after DCA1 recovery
    recovery_exit_dca2_pips: float = 1.0   # Exit at +1 pip after DCA2 recovery

    # DCA3-4: Exit at breakeven (just cover fees, preserve capital)
    recovery_exit_dca3_pips: float = 0.5   # Exit at +0.5 pip after DCA3 recovery
    recovery_exit_dca4_pips: float = 0.0   # Exit at breakeven after DCA4 recovery

    # TIME-BASED RECOVERY EXIT - For positions held too long
    # If held > 4 hours with DCA, exit at small loss to preserve capital
    use_time_based_recovery: bool = True
    max_hold_hours: float = 4.0            # Close if held > 4 hours
    time_recovery_loss_pips: float = -2.0  # Accept up to -2 pips loss after 4h hold

    # Minimum time with DCA before recovery exit (avoid premature exits)
    min_dca_hold_minutes: float = 15.0     # Wait 15 min after DCA before recovery exit

    # Legacy percentage-based (kept for compatibility, not used when use_pip_based_dca=True)
    take_profit_pct: float = 0.03
    trailing_stop_pct: float = 0.014
    initial_sl_pct: float = 0.08
    dca_level_1: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.014, multiplier=1.75))
    dca_level_2: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.015, multiplier=1.25))
    dca_level_3: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.025, multiplier=1.50))
    dca_level_4: DCALevel = field(default_factory=lambda: DCALevel(trigger_pct=0.020, multiplier=0.75))
    dca_profit_target_pct: float = 0.03
    sl_after_dca_pct: float = 0.012

    # Max DCA entries per position
    max_dca_levels: int = 4

    # Initial position size divisor for DCA - AGGRESSIVE SCALING
    # Total exposure with all DCAs = 1 + 1.5 + 2.0 + 2.5 + 3.0 = 10.0x
    # Divisor of 4 means initial = 25% (~2,500 units), max total = 2.5x normal
    # This ensures each DCA pulls average MUCH closer to current price for fast recovery
    initial_size_divisor: float = 4.0

    def get_dca_levels(self) -> List[DCALevel]:
        """Get all DCA levels as a list."""
        return [self.dca_level_1, self.dca_level_2, self.dca_level_3, self.dca_level_4]

    def get_total_dca_multiplier(self) -> float:
        """Get total multiplier if all DCA levels trigger."""
        if self.use_pip_based_dca:
            return 1.0 + self.dca_multiplier_1 + self.dca_multiplier_2 + self.dca_multiplier_3 + self.dca_multiplier_4
        return 1.0 + sum(level.multiplier for level in self.get_dca_levels())

    def get_dca_trigger_pips(self, level: int) -> float:
        """Get DCA trigger in pips for a specific level (1-4)."""
        triggers = [self.dca_trigger_pips_1, self.dca_trigger_pips_2,
                    self.dca_trigger_pips_3, self.dca_trigger_pips_4]
        if 1 <= level <= 4:
            return triggers[level - 1]
        return 0.0

    def get_dca_multiplier(self, level: int) -> float:
        """Get DCA multiplier for a specific level (1-4)."""
        multipliers = [self.dca_multiplier_1, self.dca_multiplier_2,
                       self.dca_multiplier_3, self.dca_multiplier_4]
        if 1 <= level <= 4:
            return multipliers[level - 1]
        return 0.0

    def get_tp_for_dca_level(self, dca_level: int) -> float:
        """
        Get dynamic take profit based on DCA level.

        TP REDUCES as DCA increases for faster exit:
        - Level 0 (initial): 12 pips
        - Level 1: 10 pips
        - Level 2: 5 pips (OANDA minimum)
        - Level 3: 5 pips on OANDA, but MANUAL exit at 3 pips
        - Level 4: 5 pips on OANDA, but MANUAL exit at 2 pips

        Args:
            dca_level: Current DCA level (0-4)

        Returns:
            Take profit in pips (for OANDA order, manual TP is lower for DCA3-4)
        """
        tp_by_level = [
            self.tp_pips_dca0,  # 8 pips (initial)
            self.tp_pips_dca1,  # 6 pips
            self.tp_pips_dca2,  # 5 pips (OANDA min)
            self.tp_pips_dca3,  # 5 pips (OANDA min, manual exit at 3p)
            self.tp_pips_dca4,  # 5 pips (OANDA min, manual exit at 2p)
        ]
        if 0 <= dca_level <= 4:
            return tp_by_level[dca_level]
        return self.tp_pips_dca4  # Default to smallest for safety


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
    max_trades_per_day: int = 20
    cooldown_seconds: int = 300  # 5 minutes between trades

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

    # Trading pairs
    symbols: List[str] = field(default_factory=lambda: [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF',
        'AUD_USD', 'USD_CAD', 'NZD_USD', 'EUR_GBP'
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
