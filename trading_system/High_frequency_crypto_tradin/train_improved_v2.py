"""
IMPROVED HF SCALPING Training V2 - Focus on FAST, HIGH-PROBABILITY moves
=========================================================================

KEY IMPROVEMENTS over previous training:
1. TARGET: Only label as BUY if TP hit FAST (within 30 bars) and CLEANLY (no SL touch)
2. MOMENTUM: Add strong momentum detection - only enter on confirmed moves
3. VOLATILITY: Filter for appropriate volatility (not too low, not too high)
4. VOLUME: Require above-average volume to confirm moves
5. BETTER FEATURES: Add momentum, order flow proxy, and regime detection

Run: python -m trading_system.High_frequency_crypto_tradin.train_improved_v2
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem


class ImprovedScalpingConfig:
    """Configuration for improved HF scalping model training."""

    # DATA SETTINGS
    timeframe = '1m'
    max_rows_per_symbol = 200000  # More data for better learning

    # TARGET SETTINGS - Focus on FAST wins
    # TIGHTER targets for more BUY signals in crypto (highly volatile)
    take_profit_pct = 0.005    # 0.5% TP
    stop_loss_pct = 0.004      # 0.4% SL (1.25:1 R:R)

    # KEY CHANGE: Allow more time for TP to hit (crypto can be slow)
    max_bars_for_fast_tp = 60  # Must hit TP within 60 bars (1 hour)
    max_bars_total = 180       # Total lookforward for labeling

    # Quality filters for labeling
    min_volatility = 0.0005    # Minimum hourly volatility
    max_volatility = 0.005     # Maximum (avoid crazy moves)
    min_volume_ratio = 1.0     # Minimum volume vs 20-bar average

    # MODEL SETTINGS
    confidence_threshold = 0.40
    min_model_agreement = 2

    # PATHS
    data_dir = str(Path(__file__).parent / "Crypto_Data_Fresh")
    model_save_dir = str(Path(__file__).parent / "saved_models_hf_scalping")

    # SYMBOLS - Focus on most liquid for better data
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD']


config = ImprovedScalpingConfig()


def create_improved_target(df: pd.DataFrame,
                          tp_pct: float = 0.008,
                          sl_pct: float = 0.006,
                          max_bars_fast: int = 30,
                          max_bars_total: int = 120) -> pd.DataFrame:
    """
    IMPROVED target creation focusing on FAST, CLEAN wins.

    Labels as BUY (1) only if:
    1. TP is hit FAST (within max_bars_fast)
    2. SL was NOT hit before TP (clean win)
    3. There was clear momentum in the move

    This filters out:
    - Slow grindy moves that eventually hit TP
    - Volatile choppy moves where both TP and SL get touched
    - Low-momentum setups that don't follow through
    """
    n = len(df)
    targets = np.zeros(n, dtype=np.int32)

    # Track stats
    fast_tp_hits = 0
    slow_tp_hits = 0
    sl_hits = 0
    timeout = 0
    same_bar = 0

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    print(f"\nCreating IMPROVED targets: TP={tp_pct*100:.2f}%, SL={sl_pct*100:.2f}%")
    print(f"  Fast TP requirement: within {max_bars_fast} bars")

    for i in range(n - max_bars_total):
        entry_price = closes[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        # Track when each level is hit
        tp_bar = None
        sl_bar = None

        for j in range(1, max_bars_total):
            idx = i + j
            if idx >= n:
                break

            # Check hits
            if tp_bar is None and highs[idx] >= tp_price:
                tp_bar = j
            if sl_bar is None and lows[idx] <= sl_price:
                sl_bar = j

            # Exit early if both found
            if tp_bar is not None and sl_bar is not None:
                break

        # Determine label
        if tp_bar is not None and sl_bar is not None:
            if tp_bar == sl_bar:
                # Same bar - unknown which hit first, skip (label as 0)
                same_bar += 1
                targets[i] = 0
            elif tp_bar < sl_bar:
                # TP hit first
                if tp_bar <= max_bars_fast:
                    # FAST TP hit - this is what we want!
                    fast_tp_hits += 1
                    targets[i] = 1  # BUY
                else:
                    # Slow TP hit - not ideal, label as 0
                    slow_tp_hits += 1
                    targets[i] = 0
            else:
                # SL hit first
                sl_hits += 1
                targets[i] = 0
        elif tp_bar is not None:
            # Only TP hit
            if tp_bar <= max_bars_fast:
                fast_tp_hits += 1
                targets[i] = 1  # BUY
            else:
                slow_tp_hits += 1
                targets[i] = 0
        elif sl_bar is not None:
            # Only SL hit
            sl_hits += 1
            targets[i] = 0
        else:
            # Neither hit (timeout)
            timeout += 1
            targets[i] = 0

    df = df.copy()
    df['target'] = targets

    # Stats
    n_valid = n - max_bars_total
    n_buy = (targets[:n_valid] == 1).sum()
    pct_buy = 100 * n_buy / n_valid if n_valid > 0 else 0

    print(f"\n  Target Distribution:")
    print(f"    FAST TP (BUY=1): {fast_tp_hits:,} ({100*fast_tp_hits/n_valid:.1f}%)")
    print(f"    Slow TP (NO_TRADE): {slow_tp_hits:,} ({100*slow_tp_hits/n_valid:.1f}%)")
    print(f"    SL Hit (NO_TRADE): {sl_hits:,} ({100*sl_hits/n_valid:.1f}%)")
    print(f"    Same Bar (NO_TRADE): {same_bar:,} ({100*same_bar/n_valid:.1f}%)")
    print(f"    Timeout (NO_TRADE): {timeout:,} ({100*timeout/n_valid:.1f}%)")
    print(f"\n  Final: {pct_buy:.1f}% BUY signals")

    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ENHANCED features focused on momentum and regime detection.
    All features properly shifted to prevent look-ahead bias.
    """
    df = df.copy()

    # ================================================================
    # MOMENTUM FEATURES - Key for identifying fast moves
    # ================================================================

    # Short-term momentum (1-5 bars)
    df['momentum_1'] = df['close'].pct_change(1).shift(1)
    df['momentum_3'] = df['close'].pct_change(3).shift(1)
    df['momentum_5'] = df['close'].pct_change(5).shift(1)

    # Momentum acceleration
    df['momentum_accel'] = (df['momentum_1'] - df['momentum_1'].shift(1)).shift(1)

    # Higher highs / Higher lows (trend strength)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int).shift(1)
    df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int).shift(1)
    df['hh_hl_streak'] = (df['higher_high'].rolling(5).sum() + df['higher_low'].rolling(5).sum()).shift(1)

    # ================================================================
    # TREND STRENGTH FEATURES
    # ================================================================

    # Multiple SMAs for trend detection
    df['sma_10'] = df['close'].rolling(10).mean().shift(1)
    df['sma_20'] = df['close'].rolling(20).mean().shift(1)
    df['sma_50'] = df['close'].rolling(50).mean().shift(1)

    prev_close = df['close'].shift(1)

    # Price above SMAs (trend alignment)
    df['above_sma10'] = (prev_close > df['sma_10']).astype(int)
    df['above_sma20'] = (prev_close > df['sma_20']).astype(int)
    df['above_sma50'] = (prev_close > df['sma_50']).astype(int)
    df['trend_alignment'] = df['above_sma10'] + df['above_sma20'] + df['above_sma50']

    # SMA slope (trend direction)
    df['sma20_slope'] = df['sma_20'].pct_change(5)
    df['sma50_slope'] = df['sma_50'].pct_change(10)

    # ================================================================
    # VOLATILITY REGIME FEATURES
    # ================================================================

    # ATR (Average True Range)
    high = df['high'].shift(1)
    low = df['low'].shift(1)
    close_prev = df['close'].shift(2)

    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['atr_5'] = df['true_range'].rolling(5).mean()

    # Volatility ratio (current vs average)
    df['volatility_ratio'] = df['atr_5'] / (df['atr_14'] + 1e-10)

    # Price volatility
    df['price_volatility'] = df['close'].pct_change().rolling(20).std().shift(1)

    # ================================================================
    # VOLUME FEATURES (Order Flow Proxy)
    # ================================================================

    prev_volume = df['volume'].shift(1)
    df['volume_sma_20'] = df['volume'].rolling(20).mean().shift(1)
    df['volume_ratio'] = prev_volume / (df['volume_sma_20'] + 1e-10)

    # Volume spike detection
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)

    # Volume trend
    df['volume_trend'] = df['volume'].rolling(5).mean().shift(1) / (df['volume'].rolling(20).mean().shift(1) + 1e-10)

    # On-Balance Volume (simplified)
    direction = np.sign(df['close'].diff().shift(1))
    df['obv_change'] = direction * prev_volume
    df['obv_ma'] = df['obv_change'].rolling(10).mean()

    # ================================================================
    # RSI WITH MULTIPLE TIMEFRAMES
    # ================================================================

    delta = df['close'].diff()

    for period in [7, 14, 21]:
        gain = (delta.where(delta > 0, 0)).rolling(period).mean().shift(1)
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean().shift(1)
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # RSI momentum
    df['rsi_momentum'] = df['rsi_14'] - df['rsi_14'].shift(5)

    # ================================================================
    # MACD WITH SIGNAL
    # ================================================================

    exp12 = df['close'].ewm(span=12, adjust=False).mean().shift(1)
    exp26 = df['close'].ewm(span=26, adjust=False).mean().shift(1)
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_change'] = df['macd_hist'].diff()

    # MACD crossover signal
    df['macd_bullish'] = ((df['macd'] > df['macd_signal']) &
                          (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)

    # ================================================================
    # BOLLINGER BANDS
    # ================================================================

    rolling_std = df['close'].rolling(20).std().shift(1)
    df['bb_upper'] = df['sma_20'] + 2 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2 * rolling_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-10)
    df['bb_position'] = (prev_close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # ================================================================
    # PRICE ACTION PATTERNS
    # ================================================================

    prev_open = df['open'].shift(1)
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)

    candle_range = prev_high - prev_low + 1e-10
    df['body_size'] = abs(prev_close - prev_open) / candle_range
    df['upper_wick'] = (prev_high - pd.concat([prev_open, prev_close], axis=1).max(axis=1)) / candle_range
    df['lower_wick'] = (pd.concat([prev_open, prev_close], axis=1).min(axis=1) - prev_low) / candle_range

    # Candle patterns
    df['is_bullish'] = (prev_close > prev_open).astype(int)
    df['is_doji'] = (df['body_size'] < 0.1).astype(int)
    df['is_hammer'] = ((df['lower_wick'] > 2 * df['body_size']) & (df['upper_wick'] < df['body_size'])).astype(int)

    # Consecutive candles
    df['bullish_streak'] = df['is_bullish'].rolling(5).sum()
    df['bearish_streak'] = (1 - df['is_bullish']).rolling(5).sum()

    # ================================================================
    # SUPPORT/RESISTANCE PROXIMITY
    # ================================================================

    # Recent highs/lows as S/R
    df['recent_high_20'] = df['high'].rolling(20).max().shift(1)
    df['recent_low_20'] = df['low'].rolling(20).min().shift(1)
    df['range_20'] = df['recent_high_20'] - df['recent_low_20']
    df['sr_position'] = (prev_close - df['recent_low_20']) / (df['range_20'] + 1e-10)

    # Distance from recent high (breakout potential)
    df['dist_from_high'] = (df['recent_high_20'] - prev_close) / (prev_close + 1e-10)

    # ================================================================
    # TIME-BASED FEATURES
    # ================================================================

    if 'timestamp' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek

            # Session indicators (UTC time)
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
            df['is_overlap'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)  # London/NY overlap
        except:
            pass

    return df


def load_and_prepare_data():
    """Load all symbol data."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    all_data = []

    for symbol in config.symbols:
        filepath = Path(config.data_dir) / f"{symbol}_1m.csv"
        if not filepath.exists():
            print(f"  {symbol}: NOT FOUND")
            continue

        df = pd.read_csv(filepath)

        if len(df) > config.max_rows_per_symbol:
            df = df.tail(config.max_rows_per_symbol).reset_index(drop=True)

        df['symbol'] = symbol
        all_data.append(df)
        print(f"  {symbol}: {len(df):,} rows")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Total: {len(combined):,} rows")

    return combined


def engineer_all_features(df: pd.DataFrame):
    """Engineer all features with proper look-ahead bias prevention."""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    fe = FeatureEngineer()
    symbols = df['symbol'].unique()
    all_features = []

    no_shift_cols = [
        'timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'target', 'conversionType', 'conversionSymbol',
        'hour', 'day_of_week', 'is_asian_session', 'is_london_session',
        'is_ny_session', 'is_overlap'
    ]

    for symbol in symbols:
        print(f"\n  Processing {symbol}...")
        symbol_df = df[df['symbol'] == symbol].copy()

        # Drop string columns
        for col in ['conversionType', 'conversionSymbol']:
            if col in symbol_df.columns:
                symbol_df = symbol_df.drop(columns=[col])

        # Step 1: Standard features
        symbol_df = fe.compute_all_features(symbol_df)

        # Step 2: Create improved target
        symbol_df = create_improved_target(
            symbol_df,
            tp_pct=config.take_profit_pct,
            sl_pct=config.stop_loss_pct,
            max_bars_fast=config.max_bars_for_fast_tp,
            max_bars_total=config.max_bars_total
        )

        # Step 3: Shift standard features
        feature_cols_to_shift = [col for col in symbol_df.columns if col not in no_shift_cols]
        for col in feature_cols_to_shift:
            if col in symbol_df.columns:
                symbol_df[col] = symbol_df[col].shift(1)

        # Step 4: Add enhanced features (already shifted internally)
        symbol_df = add_enhanced_features(symbol_df)

        all_features.append(symbol_df)

    combined = pd.concat(all_features, ignore_index=True)

    # Get feature columns
    feature_cols = [col for col in combined.columns if col not in no_shift_cols and col != 'target']
    fe.feature_columns = feature_cols

    print(f"\n  Total features: {len(feature_cols)}")

    return combined, fe, feature_cols


def prepare_training_data(df: pd.DataFrame, feature_cols: list):
    """Prepare X and y for training."""
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    # Drop rows with NaN target
    df = df.dropna(subset=['target'])

    # Keep only necessary columns
    keep_cols = [col for col in feature_cols if col in df.columns] + ['target', 'symbol']
    df = df[[col for col in keep_cols if col in df.columns]].copy()

    # Update feature_cols to only include columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns and col != 'target' and col != 'symbol']

    # Handle missing values
    for col in feature_cols:
        df[col] = df[col].ffill().bfill()
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    X = df[feature_cols]
    y = df['target']

    # Normalize features
    print("  Normalizing features...")
    for col in feature_cols:
        mean = X[col].mean()
        std = X[col].std()
        if std > 0:
            X[col] = (X[col] - mean) / std
        else:
            X[col] = 0

    X = X.clip(-10, 10)
    X = X.fillna(0)

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  BUY signals: {(y == 1).sum():,} ({100*(y==1).mean():.1f}%)")
    print(f"  NO TRADE: {(y == 0).sum():,} ({100*(y==0).mean():.1f}%)")

    return X, y, df, feature_cols


def train_ensemble(X_train, y_train, X_val, y_val):
    """Train ensemble with improved settings."""
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE")
    print("=" * 60)

    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val: {len(X_val):,} samples")

    n_no_trade = (y_train == 0).sum()
    n_buy = (y_train == 1).sum()
    imbalance_ratio = n_no_trade / n_buy if n_buy > 0 else 1
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

    ensemble = EnsembleVotingSystem(
        voting_method='soft',
        confidence_threshold=config.confidence_threshold,
        min_agreement=config.min_model_agreement,
        use_dynamic_weights=True
    )

    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.values
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values

    metrics = ensemble.train(
        pd.DataFrame(X_train_np, columns=X_train.columns),
        pd.Series(y_train_np),
        pd.DataFrame(X_val_np, columns=X_val.columns),
        pd.Series(y_val_np),
        verbose=True
    )

    return ensemble, metrics


def evaluate_on_test(ensemble, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values

    test_preds = ensemble.predict(pd.DataFrame(X_test_np, columns=X_test.columns))
    test_probas = ensemble.predict_proba(pd.DataFrame(X_test_np, columns=X_test.columns))

    test_accuracy = accuracy_score(y_test_np, test_preds)
    test_precision = precision_score(y_test_np, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_np, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_np, test_preds, average='weighted', zero_division=0)

    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")

    cm = confusion_matrix(y_test_np, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 NO_TRADE  BUY")
    print(f"  Actual NO_TRADE  {cm[0,0]:6}  {cm[0,1]:6}")
    if cm.shape[0] > 1:
        print(f"  Actual BUY       {cm[1,0]:6}  {cm[1,1]:6}")

    buy_preds = test_preds == 1
    actual_buys = y_test_np == 1

    if buy_preds.sum() > 0:
        buy_precision = (buy_preds & actual_buys).sum() / buy_preds.sum()
        print(f"\n  BUY Signal Analysis:")
        print(f"    Predicted BUYs: {buy_preds.sum()}")
        print(f"    Correct BUYs:   {(buy_preds & actual_buys).sum()}")
        print(f"    BUY Precision:  {buy_precision:.1%}")

    # Probability distribution
    buy_proba = test_probas[:, 1]
    print(f"\n  BUY Probability Distribution:")
    print(f"    Min: {buy_proba.min():.3f}")
    print(f"    Max: {buy_proba.max():.3f}")
    print(f"    Mean: {buy_proba.mean():.3f}")
    print(f"    Median: {np.median(buy_proba):.3f}")

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'buy_precision': buy_precision if buy_preds.sum() > 0 else 0
    }


def main():
    """Main training pipeline."""
    total_start = time.time()

    print("\n")
    print("=" * 60)
    print("IMPROVED HF SCALPING MODEL TRAINING V2")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"\nConfiguration:")
    print(f"  Take Profit: {config.take_profit_pct*100:.2f}%")
    print(f"  Stop Loss: {config.stop_loss_pct*100:.2f}%")
    print(f"  R:R Ratio: {config.take_profit_pct/config.stop_loss_pct:.2f}")
    print(f"  Fast TP bars: {config.max_bars_for_fast_tp}")
    print(f"  Symbols: {config.symbols}")

    # Load data
    df = load_and_prepare_data()

    # Engineer features
    df, fe, feature_cols = engineer_all_features(df)

    # Prepare training data
    X, y, df, feature_cols = prepare_training_data(df, feature_cols)

    # Train/val/test split
    print("\n" + "=" * 60)
    print("TRAIN/VAL/TEST SPLIT (70/15/15)")
    print("=" * 60)

    train_indices = []
    val_indices = []
    test_indices = []

    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_indices = np.where(symbol_mask)[0]

        n_samples = len(symbol_indices)
        train_end = int(n_samples * 0.70)
        val_end = int(n_samples * 0.85)

        train_indices.extend(symbol_indices[:train_end])
        val_indices.extend(symbol_indices[train_end:val_end])
        test_indices.extend(symbol_indices[val_end:])

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    X_val = X.iloc[val_indices].reset_index(drop=True)
    y_val = y.iloc[val_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    print(f"  Train: {len(X_train):,} ({100*len(X_train)/(len(X)+1):.0f}%) - BUY: {100*y_train.mean():.1f}%")
    print(f"  Val:   {len(X_val):,} ({100*len(X_val)/(len(X)+1):.0f}%) - BUY: {100*y_val.mean():.1f}%")
    print(f"  Test:  {len(X_test):,} ({100*len(X_test)/(len(X)+1):.0f}%) - BUY: {100*y_test.mean():.1f}%")

    # Train
    ensemble, metrics = train_ensemble(X_train, y_train, X_val, y_val)

    # Evaluate
    test_metrics = evaluate_on_test(ensemble, X_test, y_test)

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    save_path = Path(config.model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ensemble.save(str(save_path / "ensemble"))
    fe.feature_columns = feature_cols
    fe.save_feature_config(str(save_path / "feature_config.json"))

    # Save summary
    import json
    summary = {
        'training_date': datetime.now().isoformat(),
        'version': 'improved_v2',
        'take_profit_pct': config.take_profit_pct,
        'stop_loss_pct': config.stop_loss_pct,
        'rr_ratio': config.take_profit_pct / config.stop_loss_pct,
        'max_bars_fast': config.max_bars_for_fast_tp,
        'n_features': len(feature_cols),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_buy_ratio': float(y_train.mean()),
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_f1': test_metrics['f1'],
        'test_buy_precision': test_metrics['buy_precision'],
        'symbols': config.symbols
    }

    with open(save_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Models saved to: {save_path}")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {total_time/60:.2f} minutes")

    return ensemble, fe


if __name__ == "__main__":
    try:
        ensemble, fe = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
