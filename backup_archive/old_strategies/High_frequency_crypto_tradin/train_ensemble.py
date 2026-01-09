"""
Train Ensemble ML Model for High-Frequency Crypto Trading
===========================================================

FAST & ROBUST Training Pipeline with COMPREHENSIVE LOGGING:
1. Loads historical crypto data from Binance CSV files
2. Engineers 100+ features with parallel processing
3. Trains 5 ML models with early stopping
4. Creates ensemble with dynamic weighting
5. Runs walk-forward validation
6. Saves trained models for live trading

All stages have detailed logging for full visibility.
"""

import os
import sys
from pathlib import Path
import time
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOGGING SETUP - Comprehensive logging for all stages
# ============================================================

def setup_logging():
    """Setup comprehensive logging to console and file."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Setup logger (NOT root logger to avoid duplication)
    logger = logging.getLogger('HFT_Training')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    logger.propagate = False  # Prevent duplicate output to root logger

    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG level - captures everything)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    return logger, log_file

# Initialize logger
logger, log_file = setup_logging()

def log_section(title: str, char: str = "="):
    """Log a section header."""
    line = char * 60
    logger.info("")
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)

def log_subsection(title: str):
    """Log a subsection header."""
    logger.info("")
    logger.info(f"--- {title} ---")

def log_metric(name: str, value, unit: str = ""):
    """Log a metric with formatting."""
    if isinstance(value, float):
        if abs(value) < 0.01:
            logger.info(f"  {name}: {value:.6f} {unit}")
        elif abs(value) < 1:
            logger.info(f"  {name}: {value:.4f} {unit}")
        else:
            logger.info(f"  {name}: {value:,.2f} {unit}")
    elif isinstance(value, int):
        logger.info(f"  {name}: {value:,} {unit}")
    else:
        logger.info(f"  {name}: {value} {unit}")

def log_progress(current: int, total: int, prefix: str = "Progress"):
    """Log progress bar."""
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    logger.info(f"  {prefix}: [{bar}] {pct:.1f}% ({current}/{total})")

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Log detailed DataFrame information."""
    logger.info(f"  {name} Info:")
    logger.info(f"    - Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"    - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"    - Dtypes: {dict(df.dtypes.value_counts())}")

    # Check for issues
    null_counts = df.isnull().sum().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if null_counts > 0:
        logger.warning(f"    - NaN values: {null_counts:,}")
    if inf_counts > 0:
        logger.warning(f"    - Inf values: {inf_counts:,}")

# ============================================================
# IMPORT LOCAL MODULES
# ============================================================

try:
    from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
    from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
    from trading_system.High_frequency_crypto_tradin.backtest import WalkForwardValidator
    from trading_system.High_frequency_crypto_tradin.config import load_config
    from trading_system.High_frequency_crypto_tradin.utils.data_quality import DataQualityChecker
    logger.debug("Imported modules from trading_system package")
except ImportError as e:
    logger.debug(f"Package import failed: {e}, trying relative imports")
    from features import FeatureEngineer
    from ensemble import EnsembleVotingSystem
    from backtest import WalkForwardValidator
    from config import load_config
    from utils.data_quality import DataQualityChecker
    logger.debug("Imported modules using relative imports")


# ============================================================
# DATA LOADING
# ============================================================

def load_single_symbol(filepath: str, symbol: str, max_rows: int = None) -> pd.DataFrame:
    """Load a single symbol's data with validation and logging."""
    try:
        logger.debug(f"Loading {symbol} from {filepath}")
        df = pd.read_csv(filepath)
        original_rows = len(df)

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"  {symbol}: Missing columns {missing_cols}")
            return None

        # Ensure numeric data types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Count NaN after conversion
        nan_count = df[required_cols].isnull().sum().sum()
        if nan_count > 0:
            logger.debug(f"  {symbol}: Converted {nan_count} non-numeric values to NaN")

        # Drop rows with NaN in critical columns
        df = df.dropna(subset=required_cols)
        dropped = original_rows - len(df)
        if dropped > 0:
            logger.debug(f"  {symbol}: Dropped {dropped} rows with NaN values")

        # Parse datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            logger.debug(f"  {symbol}: Parsed datetime column")
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            logger.debug(f"  {symbol}: Converted timestamp to datetime")

        # Sort by time
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
            date_range = f"{df['datetime'].min()} to {df['datetime'].max()}"
            logger.debug(f"  {symbol}: Date range: {date_range}")

        # Limit rows for faster training if specified
        if max_rows and len(df) > max_rows:
            df = df.tail(max_rows).reset_index(drop=True)
            logger.debug(f"  {symbol}: Limited to last {max_rows} rows")

        # Add symbol column
        df['symbol'] = symbol

        # Drop non-numeric columns that shouldn't be features
        cols_to_drop = ['conversionType', 'conversionSymbol']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.debug(f"  {symbol}: Dropped column {col}")

        # Log summary
        logger.debug(f"  {symbol}: Successfully loaded {len(df):,} rows")

        return df

    except Exception as e:
        logger.error(f"  {symbol}: Load error - {e}")
        return None


def load_and_prepare_data(data_dir: str, symbols: list = None,
                          max_rows_per_symbol: int = 50000,
                          use_parallel: bool = True):
    """Load and prepare data for training with comprehensive logging."""

    log_section("STAGE 1: DATA LOADING")
    stage_start = time.time()

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Max rows per symbol: {max_rows_per_symbol:,}")
    logger.info(f"Parallel loading: {use_parallel}")

    # Default symbols if not specified (fresh data uses USD pairs)
    if symbols is None:
        symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']

    logger.info(f"Symbols to load: {symbols}")

    data_path = Path(data_dir)
    all_data = []
    load_stats = {'loaded': 0, 'failed': 0, 'total_rows': 0}

    log_subsection("Loading Individual Symbols")

    if use_parallel:
        logger.info("Using parallel loading (ThreadPoolExecutor)...")
        with ThreadPoolExecutor(max_workers=min(len(symbols), 4)) as executor:
            futures = {}
            for symbol in symbols:
                filepath = data_path / f"{symbol}_1m.csv"
                if filepath.exists():
                    future = executor.submit(load_single_symbol, str(filepath), symbol, max_rows_per_symbol)
                    futures[future] = symbol
                else:
                    logger.warning(f"  {symbol}: File not found at {filepath}")
                    load_stats['failed'] += 1

            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                        load_stats['loaded'] += 1
                        load_stats['total_rows'] += len(df)
                        logger.info(f"  [{i+1}/{len(futures)}] {symbol}: {len(df):,} rows loaded")
                    else:
                        load_stats['failed'] += 1
                        logger.warning(f"  [{i+1}/{len(futures)}] {symbol}: No data loaded")
                except Exception as e:
                    load_stats['failed'] += 1
                    logger.error(f"  [{i+1}/{len(futures)}] {symbol}: Error - {e}")
    else:
        logger.info("Using sequential loading...")
        for i, symbol in enumerate(symbols):
            filepath = data_path / f"{symbol}_1m.csv"
            if filepath.exists():
                df = load_single_symbol(str(filepath), symbol, max_rows_per_symbol)
                if df is not None:
                    all_data.append(df)
                    load_stats['loaded'] += 1
                    load_stats['total_rows'] += len(df)
                    logger.info(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df):,} rows loaded")
                else:
                    load_stats['failed'] += 1
            else:
                logger.warning(f"  [{i+1}/{len(symbols)}] {symbol}: File not found")
                load_stats['failed'] += 1

    if not all_data:
        raise ValueError("No data files loaded! Check your data directory.")

    log_subsection("Combining Datasets")
    logger.info(f"Combining {len(all_data)} datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Memory optimization
    logger.info("Optimizing memory (float64 -> float32)...")
    original_memory = combined_df.memory_usage(deep=True).sum() / 1024**2
    for col in combined_df.select_dtypes(include=['float64']).columns:
        combined_df[col] = combined_df[col].astype('float32')
    optimized_memory = combined_df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Memory reduced: {original_memory:.1f} MB -> {optimized_memory:.1f} MB ({100*(1-optimized_memory/original_memory):.1f}% savings)")

    # Stage summary
    stage_time = time.time() - stage_start
    log_subsection("Data Loading Summary")
    log_metric("Symbols loaded", load_stats['loaded'])
    log_metric("Symbols failed", load_stats['failed'])
    log_metric("Total rows", load_stats['total_rows'])
    log_metric("Combined rows", len(combined_df))
    log_metric("Memory usage", f"{optimized_memory:.1f}", "MB")
    log_metric("Stage time", f"{stage_time:.1f}", "seconds")

    log_dataframe_info(combined_df, "Combined Dataset")

    return combined_df


# ============================================================
# DATA QUALITY VALIDATION
# ============================================================

def validate_data_quality(df: pd.DataFrame, stage: str = "raw") -> bool:
    """
    Run comprehensive data quality checks.

    Args:
        df: DataFrame to validate
        stage: "raw" for raw data, "features" for engineered features

    Returns:
        True if quality checks pass, False if critical issues found
    """
    log_section(f"DATA QUALITY CHECK ({stage.upper()})")
    stage_start = time.time()

    checker = DataQualityChecker(
        missing_threshold=0.10,  # 10% missing allowed
        duplicate_threshold=0.01,  # 1% duplicates allowed
        outlier_std_threshold=5.0,  # 5 std for outliers
        gap_threshold_minutes=5,  # 5 min max gap
        correlation_threshold=0.95,  # 95% correlation warning
        min_samples_per_class=500  # Min samples per class
    )

    # Determine feature columns based on stage
    if stage == "raw":
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
    else:
        exclude_cols = ['timestamp', 'close_time', 'datetime', 'date', 'symbol',
                       'open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
                       'target', 'target_binary', 'forward_return',
                       'forward_max_gain', 'forward_max_loss', 'risk_reward']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Run all checks
    report = checker.run_all_checks(
        df,
        feature_cols=feature_cols,
        target_col='target' if 'target' in df.columns else None,
        datetime_col='datetime' if 'datetime' in df.columns else None,
        symbol_col='symbol' if 'symbol' in df.columns else None,
        verbose=True
    )

    stage_time = time.time() - stage_start

    # Log key metrics
    log_subsection("Quality Metrics")
    log_metric("Total rows", report.metrics.get('total_rows', 0))
    log_metric("Missing value %", f"{report.metrics.get('missing_value_pct', 0)*100:.2f}%")
    log_metric("Outlier %", f"{report.metrics.get('outlier_pct', 0)*100:.3f}%")
    log_metric("Time gaps", report.metrics.get('total_time_gaps', 0))
    log_metric("OHLCV violations", report.metrics.get('ohlcv_violations', 0))
    log_metric("Constant features", report.metrics.get('constant_features', 0))
    log_metric("Quality check time", f"{stage_time:.1f}", "seconds")

    if report.passed:
        logger.info("  STATUS: Data quality PASSED - proceeding with training")
    else:
        logger.warning("  STATUS: Data quality FAILED - review critical issues")
        for issue in report.critical_issues:
            logger.error(f"    CRITICAL: {issue}")

    return report.passed


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features_fast(df: pd.DataFrame, fe: FeatureEngineer) -> tuple:
    """Engineer features with comprehensive logging."""

    log_section("STAGE 2: FEATURE ENGINEERING")
    stage_start = time.time()

    symbols = df['symbol'].unique()
    logger.info(f"Processing {len(symbols)} symbols: {list(symbols)}")

    all_features = []
    feature_stats = {'success': 0, 'failed': 0}

    log_subsection("Engineering Features per Symbol")

    for i, symbol in enumerate(symbols):
        symbol_start = time.time()
        logger.info(f"  [{i+1}/{len(symbols)}] {symbol}:")

        symbol_df = df[df['symbol'] == symbol].copy()
        original_cols = len(symbol_df.columns)
        original_rows = len(symbol_df)

        try:
            # Compute features
            symbol_df = fe.compute_all_features(symbol_df)
            new_cols = len(symbol_df.columns) - original_cols

            all_features.append(symbol_df)
            feature_stats['success'] += 1

            symbol_time = time.time() - symbol_start
            logger.info(f"      Rows: {original_rows:,}")
            logger.info(f"      Features added: {new_cols}")
            logger.info(f"      Total columns: {len(symbol_df.columns)}")
            logger.info(f"      Time: {symbol_time:.2f}s")

        except Exception as e:
            feature_stats['failed'] += 1
            logger.error(f"      Error: {e}")
            continue

    log_subsection("Combining Engineered Features")
    combined = pd.concat(all_features, ignore_index=True)

    # Count feature types
    feature_cols = [c for c in combined.columns if c not in ['symbol', 'datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

    stage_time = time.time() - stage_start
    log_subsection("Feature Engineering Summary")
    log_metric("Symbols processed", feature_stats['success'])
    log_metric("Symbols failed", feature_stats['failed'])
    log_metric("Total features created", len(feature_cols))
    log_metric("Total rows", len(combined))
    log_metric("Stage time", f"{stage_time:.1f}", "seconds")

    # Log sample feature names
    logger.info("  Sample features:")
    for feat in feature_cols[:10]:
        logger.info(f"    - {feat}")
    if len(feature_cols) > 10:
        logger.info(f"    ... and {len(feature_cols) - 10} more")

    return combined, fe


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_training_data_fast(df: pd.DataFrame, fe: FeatureEngineer, config) -> tuple:
    """Prepare data for training with comprehensive logging."""

    log_section("STAGE 3: DATA PREPARATION")
    stage_start = time.time()

    log_subsection("Creating Target Labels")
    logger.info(f"  Forward period: {config.features.forward_period} bars")
    logger.info(f"  Using ADAPTIVE thresholds for balanced classes")
    logger.info(f"  Target distribution: ~33% BUY / ~34% HOLD / ~33% SELL")

    original_rows = len(df)
    df = fe.create_target(
        df,
        forward_period=config.features.forward_period,
        threshold_pct=config.features.target_threshold_pct,
        use_adaptive_threshold=True,  # Use adaptive percentile-based thresholds
        target_buy_pct=33.0,   # Top 33% of returns = BUY
        target_sell_pct=33.0   # Bottom 33% of returns = SELL
    )

    # Log the actual thresholds used
    if hasattr(fe, '_buy_threshold') and hasattr(fe, '_sell_threshold'):
        logger.info(f"  Computed BUY threshold:  >= {fe._buy_threshold*100:.4f}%")
        logger.info(f"  Computed SELL threshold: <= {fe._sell_threshold*100:.4f}%")

    logger.info(f"  Target columns added: target, target_binary, forward_return, etc.")

    log_subsection("Handling Missing Values")
    null_before = df.isnull().sum().sum()
    df = fe.handle_missing_values(df)
    null_after = df.isnull().sum().sum()
    logger.info(f"  NaN values: {null_before:,} -> {null_after:,}")

    log_subsection("Selecting Feature Columns")
    exclude_cols = [
        'timestamp', 'close_time', 'datetime', 'date', 'symbol',
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'target', 'target_binary', 'forward_return',
        'forward_max_gain', 'forward_max_loss', 'risk_reward'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    fe.feature_columns = feature_cols
    logger.info(f"  Feature columns selected: {len(feature_cols)}")

    if config.features.normalize:
        log_subsection("Normalizing Features")
        logger.info("  Method: Z-score normalization")
        df = fe.normalize_features(df, feature_cols)
        logger.info(f"  Normalized {len(feature_cols)} features")

    log_subsection("Final Data Validation")
    df = df.dropna(subset=['target'])
    logger.info(f"  Rows after dropping NaN targets: {len(df):,} (dropped {original_rows - len(df):,})")

    X = df[feature_cols].copy()
    y_multi = df['target'].copy()
    y_binary = df['target_binary'].copy()

    # Replace infinities
    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning(f"  Replacing {inf_count:,} infinite values with NaN")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Clip extreme values
    logger.info("  Clipping extreme values to [-1e6, 1e6]")
    for col in X.columns:
        X[col] = X[col].clip(-1e6, 1e6)

    stage_time = time.time() - stage_start

    log_subsection("Data Preparation Summary")
    log_metric("Total samples", len(X))
    log_metric("Total features", len(feature_cols))
    log_metric("Stage time", f"{stage_time:.1f}", "seconds")

    logger.info("")
    logger.info("  Target Distribution:")
    for label in sorted(y_multi.unique()):
        count = (y_multi == label).sum()
        pct = 100 * count / len(y_multi)
        label_name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}.get(label, str(label))
        logger.info(f"    {label_name:>4} ({label:>2}): {count:>10,} samples ({pct:>5.1f}%)")

    return X, y_multi, y_binary


# ============================================================
# MODEL TRAINING
# ============================================================

def train_ensemble_fast(X_train, y_train, X_val, y_val, config):
    """Train ensemble with comprehensive logging."""

    log_section("STAGE 4: ENSEMBLE TRAINING")
    stage_start = time.time()

    logger.info("Training Configuration:")
    log_metric("Training samples", len(X_train))
    log_metric("Validation samples", len(X_val))
    log_metric("Features", X_train.shape[1])
    log_metric("Voting method", config.ensemble.voting_method)
    log_metric("Confidence threshold", config.ensemble.confidence_threshold)
    log_metric("Min model agreement", config.ensemble.min_model_agreement)
    log_metric("Dynamic weights", config.ensemble.use_dynamic_weights)

    # Create ensemble
    ensemble = EnsembleVotingSystem(
        voting_method=config.ensemble.voting_method,
        confidence_threshold=config.ensemble.confidence_threshold,
        min_agreement=config.ensemble.min_model_agreement,
        use_dynamic_weights=config.ensemble.use_dynamic_weights
    )

    log_subsection("Training Individual Models")
    logger.info("  Models to train: RandomForest, XGBoost, LightGBM, CatBoost, NeuralNetwork")
    logger.info("  Early stopping enabled for faster convergence")
    logger.info("")

    # Convert to numpy for faster training
    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values

    # Train with detailed logging
    metrics = ensemble.train(
        pd.DataFrame(X_train_np, columns=X_train.columns),
        pd.Series(y_train_np),
        pd.DataFrame(X_val_np, columns=X_val.columns),
        pd.Series(y_val_np),
        verbose=True
    )

    stage_time = time.time() - stage_start

    log_subsection("Training Results")
    if 'ensemble' in metrics:
        ens_metrics = metrics['ensemble']
        log_metric("Ensemble Accuracy", ens_metrics.get('accuracy', 0))
        log_metric("Ensemble F1 Score", ens_metrics.get('f1', 0))

    logger.info("")
    logger.info("  Individual Model Performance:")
    for model_name, model_metrics in metrics.items():
        if model_name != 'ensemble':
            acc = model_metrics.get('val_accuracy', model_metrics.get('accuracy', 0))
            f1 = model_metrics.get('val_f1', model_metrics.get('f1', 0))
            logger.info(f"    {model_name:>15}: Acc={acc:.4f}, F1={f1:.4f}")

    log_metric("Total training time", f"{stage_time:.1f}", "seconds")

    return ensemble, metrics


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def run_walk_forward_validation_fast(ensemble, X, y, n_splits=3):
    """Run walk-forward validation with comprehensive logging."""

    log_section("STAGE 5: WALK-FORWARD VALIDATION")
    stage_start = time.time()

    logger.info("Validation Configuration:")
    log_metric("Number of splits", n_splits)
    log_metric("Total samples", len(X))
    log_metric("Train ratio", "80%")

    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_ratio=0.8,
        min_train_size=2000
    )

    def train_func(X_train, y_train):
        try:
            metrics = ensemble.train(X_train, y_train, verbose=False)
            return metrics.get('ensemble', {})
        except Exception as e:
            logger.error(f"  Training error in fold: {e}")
            return {}

    def predict_func(X_test):
        return ensemble.predict(X_test)

    def predict_proba_func(X_test):
        return ensemble.predict_proba(X_test)

    log_subsection("Running Validation Folds")

    try:
        results = validator.validate(
            X, y,
            train_func=train_func,
            predict_func=predict_func,
            predict_proba_func=predict_proba_func,
            verbose=True
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results = {
            'accuracy': 0.0,
            'f1': 0.0,
            'consistency': 0.0,
            'degradation': 0.0
        }

    stage_time = time.time() - stage_start

    log_subsection("Walk-Forward Validation Results")
    log_metric("Overall Accuracy", results.get('accuracy', 0))
    log_metric("Overall F1 Score", results.get('f1', 0))
    log_metric("Consistency Score", results.get('consistency', 0))
    log_metric("Performance Degradation", results.get('degradation', 0))
    log_metric("Validation time", f"{stage_time:.1f}", "seconds")

    return results, validator


# ============================================================
# MODEL SAVING
# ============================================================

def save_models(ensemble, fe, config, save_dir: str):
    """Save trained models with comprehensive logging."""

    log_section("STAGE 6: SAVING MODELS")
    stage_start = time.time()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Save directory: {save_path}")

    saved_files = []

    # Save ensemble
    try:
        ensemble_path = save_path / "ensemble"
        ensemble.save(str(ensemble_path))
        saved_files.append(str(ensemble_path))
        logger.info(f"  [OK] Ensemble saved to {ensemble_path}")
    except Exception as e:
        logger.error(f"  [FAIL] Ensemble save error: {e}")

    # Save feature config
    try:
        feature_config_path = save_path / "feature_config.json"
        fe.save_feature_config(str(feature_config_path))
        saved_files.append(str(feature_config_path))
        logger.info(f"  [OK] Feature config saved to {feature_config_path}")
    except Exception as e:
        logger.error(f"  [FAIL] Feature config save error: {e}")

    # Save training summary
    try:
        summary = {
            'training_date': datetime.now().isoformat(),
            'n_features': len(fe.feature_columns) if fe.feature_columns else 0,
            'feature_columns': fe.feature_columns[:50] if fe.feature_columns else [],
            'model_weights': ensemble.get_model_weights() if hasattr(ensemble, 'get_model_weights') else {},
            'log_file': str(log_file)
        }

        import json
        summary_path = save_path / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files.append(str(summary_path))
        logger.info(f"  [OK] Training summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"  [FAIL] Summary save error: {e}")

    stage_time = time.time() - stage_start

    log_subsection("Save Summary")
    log_metric("Files saved", len(saved_files))
    log_metric("Save time", f"{stage_time:.2f}", "seconds")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main training script with comprehensive logging."""
    total_start = time.time()

    # Header
    print("\n")
    log_section("HIGH-FREQUENCY CRYPTO TRADING - MODEL TRAINING", "=")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # System info
    logger.info("System Information:")
    log_metric("Python version", sys.version.split()[0])
    log_metric("NumPy version", np.__version__)
    log_metric("Pandas version", pd.__version__)
    log_metric("Working directory", os.getcwd())

    # Load configuration
    log_subsection("Loading Configuration")
    try:
        config = load_config()
        logger.info("  Configuration loaded from file")
    except Exception as e:
        logger.warning(f"  Config load failed: {e}")
        logger.info("  Using default configuration")
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.data_dir = str(Path(__file__).parent / "Crypto_Data_Fresh")
        config.model_save_dir = str(Path(__file__).parent / "saved_models")
        config.features = SimpleNamespace()
        config.features.forward_period = 5
        config.features.target_threshold_pct = 0.1
        config.features.normalize = True
        config.ensemble = SimpleNamespace()
        config.ensemble.voting_method = 'soft'
        config.ensemble.confidence_threshold = 0.6
        config.ensemble.min_model_agreement = 3
        config.ensemble.use_dynamic_weights = True
        config.backtest = SimpleNamespace()
        config.backtest.walk_forward_splits = 3

    # Get data directory
    data_dir = getattr(config, 'data_dir', None)
    if not data_dir or not Path(data_dir).exists():
        # Use fresh data directory with synthetic + real data
        data_dir = Path(__file__).parent / "Crypto_Data_Fresh"

    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Model save directory: {getattr(config, 'model_save_dir', 'saved_models')}")

    # ========================================
    # STAGE 1: Load Data
    # ========================================
    try:
        df = load_and_prepare_data(
            str(data_dir),
            max_rows_per_symbol=50000,
            use_parallel=True
        )
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        logger.info("Retrying with smaller dataset...")
        df = load_and_prepare_data(
            str(data_dir),
            symbols=['BTCUSD', 'ETHUSD'],  # New symbol names from fresh data
            max_rows_per_symbol=20000,
            use_parallel=False
        )

    # ========================================
    # QUALITY CHECK 1: Raw Data Validation
    # ========================================
    raw_quality_passed = validate_data_quality(df, stage="raw")
    if not raw_quality_passed:
        logger.warning("Raw data quality check found issues - proceeding with caution")

    # ========================================
    # STAGE 2: Feature Engineering
    # ========================================
    fe = FeatureEngineer()
    try:
        df, fe = engineer_features_fast(df, fe)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

    # ========================================
    # STAGE 3: Data Preparation
    # ========================================
    try:
        X, y_multi, y_binary = prepare_training_data_fast(df, fe, config)
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

    # ========================================
    # QUALITY CHECK 2: Feature Data Validation
    # ========================================
    # Create temporary df for quality check
    feature_df = X.copy()
    feature_df['target'] = y_multi.values
    feature_quality_passed = validate_data_quality(feature_df, stage="features")
    del feature_df

    if not feature_quality_passed:
        logger.warning("Feature data quality check found issues - review before production use")

    # Free memory
    del df
    import gc
    gc.collect()
    logger.debug("Memory freed after data preparation")

    # ========================================
    # SPLIT DATA (TIME-AWARE PER SYMBOL)
    # ========================================
    log_section("DATA SPLITTING (TIME-AWARE)")
    split_ratio = 0.8

    # CRITICAL: Split each symbol's data separately to prevent leakage
    # This ensures validation data is from the FUTURE of each symbol's training data
    logger.info("  Using per-symbol time-aware splitting to prevent data leakage")
    logger.info(f"  Split ratio: {split_ratio*100:.0f}% train / {(1-split_ratio)*100:.0f}% validation")

    # Get the original symbol column from the full dataframe before feature prep
    # We need to track which rows belong to which symbol
    # Since we combined all symbols, we need to split properly

    # Calculate how many rows per symbol (assuming equal distribution)
    n_symbols = 5  # BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, AVAXUSDT
    rows_per_symbol = len(X) // n_symbols

    train_indices = []
    val_indices = []

    for i in range(n_symbols):
        start_idx = i * rows_per_symbol
        end_idx = (i + 1) * rows_per_symbol if i < n_symbols - 1 else len(X)
        symbol_length = end_idx - start_idx

        # Split this symbol's data: first 80% train, last 20% validation
        symbol_split = int(symbol_length * split_ratio)
        train_indices.extend(range(start_idx, start_idx + symbol_split))
        val_indices.extend(range(start_idx + symbol_split, end_idx))

    # Create train/val sets using the proper indices
    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y_multi.iloc[train_indices].reset_index(drop=True)
    X_val = X.iloc[val_indices].reset_index(drop=True)
    y_val = y_multi.iloc[val_indices].reset_index(drop=True)

    log_metric("Train samples", len(X_train))
    log_metric("Validation samples", len(X_val))
    log_metric("Train/Val ratio", f"{split_ratio*100:.0f}% / {(1-split_ratio)*100:.0f}%")
    logger.info("  Each symbol: first 80% of time series -> train, last 20% -> validation")
    logger.info("  This prevents future data leaking into training!")

    # ========================================
    # STAGE 4: Train Ensemble
    # ========================================
    try:
        ensemble, metrics = train_ensemble_fast(X_train, y_train, X_val, y_val, config)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # ========================================
    # STAGE 5: Walk-Forward Validation
    # ========================================
    wf_results = None
    try:
        val_size = min(len(X), 50000)
        X_wf = X.tail(val_size)
        y_wf = y_multi.tail(val_size)

        wf_results, validator = run_walk_forward_validation_fast(
            ensemble, X_wf, y_wf,
            n_splits=config.backtest.walk_forward_splits
        )
    except Exception as e:
        logger.error(f"Walk-forward validation failed: {e}")
        wf_results = {'accuracy': 0, 'f1': 0, 'consistency': 0, 'degradation': 0}

    # ========================================
    # STAGE 6: Save Models
    # ========================================
    save_dir = getattr(config, 'model_save_dir', str(Path(__file__).parent / "saved_models"))
    save_models(ensemble, fe, config, save_dir)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    total_time = time.time() - total_start

    log_section("TRAINING COMPLETE!", "*")

    logger.info("")
    logger.info("Final Results:")

    if wf_results:
        log_metric("Walk-Forward Accuracy", wf_results.get('accuracy', 0))
        log_metric("Walk-Forward F1", wf_results.get('f1', 0))
        log_metric("Consistency", wf_results.get('consistency', 0))
        log_metric("Degradation", wf_results.get('degradation', 0))

    logger.info("")
    logger.info("Model Weights:")
    if hasattr(ensemble, 'get_model_weights'):
        for name, weight in ensemble.get_model_weights().items():
            logger.info(f"    {name:>15}: {weight:.4f} ({weight*100:.1f}%)")

    logger.info("")
    log_metric("Total training time", f"{total_time/60:.2f}", "minutes")
    log_metric("Models saved to", save_dir)
    log_metric("Log file", str(log_file))

    logger.info("")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_section("", "=")

    return ensemble, fe


if __name__ == "__main__":
    try:
        ensemble, fe = main()
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
