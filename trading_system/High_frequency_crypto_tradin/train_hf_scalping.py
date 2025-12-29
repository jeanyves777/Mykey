"""
HF SCALPING Training - Train models for 0.4% TP / 0.3% SL scalping
====================================================================

KEY DIFFERENCES from previous training:
1. Uses 1-minute candles (not 5-min) for true HF trading
2. Target: Will price move +0.4% before -0.3%? (actual scalping target)
3. Uses RISK-REWARD labels: BUY if TP hits before SL
4. Works in BOTH uptrending and downtrending markets
5. Trained on all 5 symbols for generalization

Run: python -m trading_system.High_frequency_crypto_tradin.train_hf_scalping
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


# ============================================================
# CONFIGURATION FOR HF SCALPING
# ============================================================

class HFScalpingConfig:
    """Configuration for HF scalping model training."""

    # DATA SETTINGS - Use 1-minute for HF
    timeframe = '1m'
    max_rows_per_symbol = 100000  # ~69 days of 1-min data (reduced to avoid memory issues)

    # SCALPING TARGET SETTINGS
    # These MUST match the backtest config!
    take_profit_pct = 0.004    # 0.4% TP
    stop_loss_pct = 0.003      # 0.3% SL
    max_bars_to_check = 60     # Look ahead max 60 bars (1 hour)

    # MARKET REGIME - Learn patterns in ALL conditions
    learn_uptrend = True
    learn_downtrend = True
    learn_ranging = True

    # MODEL SETTINGS - Lower thresholds to get MORE trades
    # Let the TP/SL do the filtering, not the model confidence
    confidence_threshold = 0.45  # Lower - allow more trades
    min_model_agreement = 2       # Only 2/5 models need to agree

    # PATHS
    data_dir = str(Path(__file__).parent / "Crypto_Data_Fresh")
    model_save_dir = str(Path(__file__).parent / "saved_models_hf_scalping")

    # SYMBOLS
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']


config = HFScalpingConfig()


# ============================================================
# SCALPING TARGET CREATION
# ============================================================

def create_scalping_target(df: pd.DataFrame,
                           tp_pct: float = 0.004,
                           sl_pct: float = 0.003,
                           max_bars: int = 60) -> pd.DataFrame:
    """
    Create target for HF scalping: BUY if TP hits before SL.

    For each bar, we look forward and ask:
    - Will price reach +TP% before reaching -SL%?
    - If yes: BUY signal (1)
    - If no (SL hit first or neither hit): No trade (0)

    This creates labels that EXACTLY match our backtest logic.
    Works in BOTH uptrending and downtrending markets because:
    - In uptrend: More BUY signals (TP hits more often)
    - In downtrend: Fewer BUY signals (SL hits more often)
    - Model learns WHEN to enter in each condition
    """
    n = len(df)
    targets = np.zeros(n, dtype=np.int32)
    tp_prices = np.zeros(n)
    sl_prices = np.zeros(n)
    bars_to_tp = np.zeros(n)
    bars_to_sl = np.zeros(n)

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    print(f"Creating scalping targets: TP={tp_pct*100:.2f}%, SL={sl_pct*100:.2f}%, MaxBars={max_bars}")

    for i in range(n - max_bars):
        entry_price = closes[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        tp_prices[i] = tp_price
        sl_prices[i] = sl_price

        # Look forward to see what hits first
        tp_hit = False
        sl_hit = False
        tp_bar = 0
        sl_bar = 0

        for j in range(1, min(max_bars, n - i)):
            future_high = highs[i + j]
            future_low = lows[i + j]

            # Check if TP hit (high >= tp_price)
            if not tp_hit and future_high >= tp_price:
                tp_hit = True
                tp_bar = j

            # Check if SL hit (low <= sl_price)
            if not sl_hit and future_low <= sl_price:
                sl_hit = True
                sl_bar = j

            # Both checked, can exit early
            if tp_hit and sl_hit:
                break

        bars_to_tp[i] = tp_bar if tp_hit else max_bars
        bars_to_sl[i] = sl_bar if sl_hit else max_bars

        # Determine target
        # IMPORTANT: If both TP and SL hit on SAME bar, we CANNOT know which hit first
        # Conservative assumption: SL hits first (assume worst case for realistic training)
        if tp_hit and (not sl_hit or tp_bar < sl_bar):
            # TP hit STRICTLY before SL
            targets[i] = 1  # BUY
        else:
            # SL hit first, OR same bar (unknown order), OR neither hit
            targets[i] = 0  # NO TRADE

    df = df.copy()
    df['target'] = targets
    df['tp_price'] = tp_prices
    df['sl_price'] = sl_prices
    df['bars_to_tp'] = bars_to_tp
    df['bars_to_sl'] = bars_to_sl

    # Stats
    n_valid = n - max_bars
    n_buy = (targets[:n_valid] == 1).sum()
    pct_buy = 100 * n_buy / n_valid if n_valid > 0 else 0

    print(f"  Total bars: {n_valid:,}")
    print(f"  BUY signals: {n_buy:,} ({pct_buy:.1f}%)")
    print(f"  NO TRADE: {n_valid - n_buy:,} ({100 - pct_buy:.1f}%)")

    avg_bars_tp = bars_to_tp[:n_valid][targets[:n_valid] == 1].mean() if n_buy > 0 else 0
    print(f"  Avg bars to TP (when hit): {avg_bars_tp:.1f}")

    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that help identify market regime.
    This helps model learn when to trade in different conditions.

    CRITICAL: All features are shifted by 1 bar to prevent look-ahead bias!
    When predicting for bar N, we can only use data from bar N-1 and earlier.
    """
    df = df.copy()

    # ================================================================
    # TREND INDICATORS - All shifted by 1 bar
    # ================================================================
    df['sma_20'] = df['close'].rolling(20).mean().shift(1)
    df['sma_50'] = df['close'].rolling(50).mean().shift(1)
    df['sma_200'] = df['close'].rolling(200).mean().shift(1)

    # Trend direction (based on previous bar's close vs previous bar's SMA)
    prev_close = df['close'].shift(1)
    df['trend_20'] = (prev_close > df['sma_20']).astype(int)
    df['trend_50'] = (prev_close > df['sma_50']).astype(int)
    df['trend_200'] = (prev_close > df['sma_200']).astype(int)

    # Trend strength
    df['trend_strength'] = df['trend_20'] + df['trend_50'] + df['trend_200']  # 0-3

    # ================================================================
    # VOLATILITY REGIME - All shifted by 1 bar
    # ================================================================
    df['volatility_20'] = df['close'].pct_change().rolling(20).std().shift(1)
    df['volatility_50'] = df['close'].pct_change().rolling(50).std().shift(1)
    df['vol_regime'] = df['volatility_20'] / (df['volatility_50'] + 1e-10)

    # ================================================================
    # MOMENTUM / RETURNS - All shifted by 1 bar
    # ================================================================
    df['return_5'] = df['close'].pct_change(5).shift(1)
    df['return_10'] = df['close'].pct_change(10).shift(1)
    df['return_20'] = df['close'].pct_change(20).shift(1)

    # ================================================================
    # MEAN REVERSION - All shifted by 1 bar
    # ================================================================
    rolling_std = df['close'].rolling(20).std().shift(1)
    df['zscore_20'] = (prev_close - df['sma_20']) / (rolling_std + 1e-10)

    # Bollinger position
    df['bb_upper'] = df['sma_20'] + 2 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2 * rolling_std
    df['bb_position'] = (prev_close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # ================================================================
    # VOLUME CONTEXT - All shifted by 1 bar
    # ================================================================
    prev_volume = df['volume'].shift(1)
    df['volume_sma'] = df['volume'].rolling(20).mean().shift(1)
    df['volume_ratio'] = prev_volume / (df['volume_sma'] + 1e-10)

    # ================================================================
    # RSI - Shifted by 1 bar
    # ================================================================
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().shift(1)
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().shift(1)
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ================================================================
    # MACD - Shifted by 1 bar
    # ================================================================
    exp1 = df['close'].ewm(span=12, adjust=False).mean().shift(1)
    exp2 = df['close'].ewm(span=26, adjust=False).mean().shift(1)
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # Signal based on already-shifted MACD
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ================================================================
    # PRICE ACTION - Use previous bar's candle data
    # ================================================================
    prev_open = df['open'].shift(1)
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_close = df['close'].shift(1)

    candle_range = prev_high - prev_low + 1e-10
    df['body_size'] = abs(prev_close - prev_open) / candle_range
    df['upper_wick'] = (prev_high - pd.concat([prev_open, prev_close], axis=1).max(axis=1)) / candle_range
    df['lower_wick'] = (pd.concat([prev_open, prev_close], axis=1).min(axis=1) - prev_low) / candle_range

    # Candlestick direction (previous bar)
    df['is_green'] = (prev_close > prev_open).astype(int)

    # Consecutive candles (based on previous bars)
    df['consecutive_green'] = df['is_green'].rolling(5).sum()
    df['consecutive_red'] = (1 - df['is_green']).rolling(5).sum()

    return df


def load_and_prepare_data():
    """Load all symbol data and prepare for training."""
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

        # Limit rows
        if len(df) > config.max_rows_per_symbol:
            df = df.tail(config.max_rows_per_symbol).reset_index(drop=True)

        df['symbol'] = symbol
        all_data.append(df)
        print(f"  {symbol}: {len(df):,} rows")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Total: {len(combined):,} rows")

    return combined


def engineer_features_for_scalping(df: pd.DataFrame):
    """
    Engineer all features for scalping.

    CRITICAL: All features are shifted by 1 bar to prevent look-ahead bias!
    When predicting for bar N, we can only use data from bar N-1 and earlier.
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING (WITH LOOK-AHEAD BIAS PREVENTION)")
    print("=" * 60)

    fe = FeatureEngineer()
    symbols = df['symbol'].unique()
    all_features = []

    # Define columns that should NOT be shifted (raw data, metadata, targets, string cols)
    no_shift_cols = [
        'timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
        'target', 'target_binary', 'forward_return',
        'forward_max_gain', 'forward_max_loss', 'risk_reward',
        'tp_price', 'sl_price', 'bars_to_tp', 'bars_to_sl',
        # String columns that should NOT be features
        'conversionType', 'conversionSymbol'
    ]

    for symbol in symbols:
        print(f"  Processing {symbol}...")
        symbol_df = df[df['symbol'] == symbol].copy()

        # Drop string columns that cause issues
        for col in ['conversionType', 'conversionSymbol']:
            if col in symbol_df.columns:
                symbol_df = symbol_df.drop(columns=[col])

        # Step 1: Compute standard FeatureEngineer features
        symbol_df = fe.compute_all_features(symbol_df)

        # Step 2: Create scalping target BEFORE shifting features
        # Target uses future data (which is correct - we're labeling what WILL happen)
        symbol_df = create_scalping_target(
            symbol_df,
            tp_pct=config.take_profit_pct,
            sl_pct=config.stop_loss_pct,
            max_bars=config.max_bars_to_check
        )

        # Step 3: SHIFT ALL FEATURE COLUMNS BY 1 BAR
        # This is CRITICAL to prevent look-ahead bias!
        # Features computed from bar N data should only be used to predict bar N+1
        feature_cols_to_shift = [col for col in symbol_df.columns if col not in no_shift_cols]

        print(f"    Shifting {len(feature_cols_to_shift)} feature columns by 1 bar...")
        for col in feature_cols_to_shift:
            symbol_df[col] = symbol_df[col].shift(1)

        # Step 4: Add market regime features (already shifted in function)
        symbol_df = add_market_regime_features(symbol_df)

        all_features.append(symbol_df)

    combined = pd.concat(all_features, ignore_index=True)

    # Get feature columns (exclude metadata and targets)
    feature_cols = [col for col in combined.columns if col not in no_shift_cols]
    fe.feature_columns = feature_cols

    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  All features shifted by 1 bar to prevent look-ahead bias [OK]")

    return combined, fe, feature_cols


def prepare_training_data(df: pd.DataFrame, feature_cols: list):
    """Prepare X and y for training."""
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    # Drop rows with NaN target first to reduce memory
    df = df.dropna(subset=['target'])

    # Only work with feature columns + target + symbol
    keep_cols = feature_cols + ['target', 'symbol']
    df = df[keep_cols].copy()

    # Handle missing values and infinities in-place
    for col in feature_cols:
        df[col] = df[col].ffill().bfill()
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    X = df[feature_cols]
    y = df['target']

    # Normalize features in-place
    print("  Normalizing features...")
    for col in feature_cols:
        mean = X[col].mean()
        std = X[col].std()
        if std > 0:
            X[col] = (X[col] - mean) / std
        else:
            X[col] = 0

    # Clip extreme values and fill NaN
    X = X.clip(-10, 10)
    X = X.fillna(0)

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  BUY signals: {(y == 1).sum():,} ({100*(y==1).mean():.1f}%)")
    print(f"  NO TRADE: {(y == 0).sum():,} ({100*(y==0).mean():.1f}%)")

    return X, y, df


def train_ensemble_models(X_train, y_train, X_val, y_val):
    """Train the ensemble on scalping targets with class balancing."""
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE (WITH CLASS BALANCING)")
    print("=" * 60)

    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")

    # Calculate class imbalance ratio
    n_no_trade = (y_train == 0).sum()
    n_buy = (y_train == 1).sum()
    imbalance_ratio = n_no_trade / n_buy if n_buy > 0 else 1
    print(f"  Class imbalance ratio: {imbalance_ratio:.1f}:1 (NO_TRADE:BUY)")
    print(f"  Models will upweight BUY class to learn winning patterns")

    ensemble = EnsembleVotingSystem(
        voting_method='soft',
        confidence_threshold=config.confidence_threshold,
        min_agreement=config.min_model_agreement,
        use_dynamic_weights=True
    )

    # Convert to numpy
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

    print("\n  Training Results:")
    if 'ensemble' in metrics:
        ens = metrics['ensemble']
        print(f"    Accuracy: {ens.get('accuracy', 0):.4f}")
        print(f"    F1 Score: {ens.get('f1', 0):.4f}")

    return ensemble, metrics


def main():
    """Main training pipeline for HF scalping."""
    total_start = time.time()

    print("\n")
    print("=" * 60)
    print("HF SCALPING MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"\nConfiguration:")
    print(f"  Take Profit: {config.take_profit_pct*100:.2f}%")
    print(f"  Stop Loss: {config.stop_loss_pct*100:.2f}%")
    print(f"  Max bars to check: {config.max_bars_to_check}")
    print(f"  Symbols: {config.symbols}")

    # Load data
    df = load_and_prepare_data()

    # Engineer features
    df, fe, feature_cols = engineer_features_for_scalping(df)

    # Prepare training data
    X, y, df = prepare_training_data(df, feature_cols)

    # Time-aware train/val/test split (70/15/15)
    print("\n" + "=" * 60)
    print("TRAIN/VAL/TEST SPLIT (TIME-AWARE 70/15/15)")
    print("=" * 60)

    train_ratio = 0.70
    val_ratio = 0.15
    # test_ratio = 0.15 (remainder)

    train_indices = []
    val_indices = []
    test_indices = []

    for i, symbol in enumerate(df['symbol'].unique()):
        symbol_mask = df['symbol'] == symbol
        symbol_indices = np.where(symbol_mask)[0]

        n_samples = len(symbol_indices)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_indices.extend(symbol_indices[:train_end])
        val_indices.extend(symbol_indices[train_end:val_end])
        test_indices.extend(symbol_indices[val_end:])

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    X_val = X.iloc[val_indices].reset_index(drop=True)
    y_val = y.iloc[val_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    print(f"  Train: {len(X_train):,} samples ({100*len(X_train)/(len(X_train)+len(X_val)+len(X_test)):.0f}%)")
    print(f"  Val:   {len(X_val):,} samples ({100*len(X_val)/(len(X_train)+len(X_val)+len(X_test)):.0f}%)")
    print(f"  Test:  {len(X_test):,} samples ({100*len(X_test)/(len(X_train)+len(X_val)+len(X_test)):.0f}%)")
    print(f"  Train BUY ratio: {100*y_train.mean():.1f}%")
    print(f"  Val BUY ratio: {100*y_val.mean():.1f}%")
    print(f"  Test BUY ratio: {100*y_test.mean():.1f}%")

    # Train ensemble
    ensemble, metrics = train_ensemble_models(X_train, y_train, X_val, y_val)

    # ================================================================
    # TEST SET EVALUATION (UNSEEN DATA)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (UNSEEN DATA)")
    print("=" * 60)

    X_test_np = X_test.values.astype(np.float32)
    y_test_np = y_test.values

    test_preds = ensemble.predict(pd.DataFrame(X_test_np, columns=X_test.columns))
    test_probas = ensemble.predict_proba(pd.DataFrame(X_test_np, columns=X_test.columns))

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    test_accuracy = accuracy_score(y_test_np, test_preds)
    test_precision = precision_score(y_test_np, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_np, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_np, test_preds, average='weighted', zero_division=0)

    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall:    {test_recall:.4f}")
    print(f"  Test F1 Score:  {test_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_np, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 NO_TRADE  BUY")
    print(f"  Actual NO_TRADE  {cm[0,0]:6}  {cm[0,1]:6}")
    if cm.shape[0] > 1:
        print(f"  Actual BUY       {cm[1,0]:6}  {cm[1,1]:6}")

    # Buy signal analysis
    buy_preds = test_preds == 1
    actual_buys = y_test_np == 1

    if buy_preds.sum() > 0:
        buy_precision = (buy_preds & actual_buys).sum() / buy_preds.sum()
        print(f"\n  BUY Signal Analysis:")
        print(f"    Predicted BUYs: {buy_preds.sum()}")
        print(f"    Correct BUYs:   {(buy_preds & actual_buys).sum()}")
        print(f"    BUY Precision:  {buy_precision:.1%}")

    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    save_path = Path(config.model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ensemble.save(str(save_path / "ensemble"))
    fe.save_feature_config(str(save_path / "feature_config.json"))

    # Save training summary
    import json
    summary = {
        'training_date': datetime.now().isoformat(),
        'timeframe': config.timeframe,
        'take_profit_pct': config.take_profit_pct,
        'stop_loss_pct': config.stop_loss_pct,
        'max_bars_to_check': config.max_bars_to_check,
        'n_features': len(feature_cols),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_buy_ratio': float(y_train.mean()),
        'val_buy_ratio': float(y_val.mean()),
        'test_buy_ratio': float(y_test.mean()),
        'symbols': config.symbols,
        'look_ahead_bias_prevented': True,
        'feature_shift': 1,
        # Test set metrics (unseen data - most important!)
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1)
    }

    if 'ensemble' in metrics:
        summary['val_accuracy'] = metrics['ensemble'].get('accuracy', 0)
        summary['val_f1'] = metrics['ensemble'].get('f1', 0)

    with open(save_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Models saved to: {save_path}")

    # Final summary
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Model saved to: {config.model_save_dir}")
    print(f"\nNext steps:")
    print(f"  1. Update run_backtest.py to use 'saved_models_hf_scalping'")
    print(f"  2. Run: python -m trading_system.High_frequency_crypto_tradin.run_backtest")

    return ensemble, fe


if __name__ == "__main__":
    try:
        ensemble, fe = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
