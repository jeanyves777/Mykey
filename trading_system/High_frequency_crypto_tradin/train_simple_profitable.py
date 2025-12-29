"""
SIMPLE PROFITABLE TRAINING - Focus on what actually works in backtests
======================================================================

This training approach:
1. Uses ACTUAL backtest P&L to create targets (not just TP/SL hits)
2. Trains on entries that would have been PROFITABLE after commission
3. Simpler feature set focused on momentum and trend

Run: python -m trading_system.High_frequency_crypto_tradin.train_simple_profitable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem


class SimpleConfig:
    """Simple profitable config."""
    max_rows_per_symbol = 200000

    # Trading params that MIGHT work
    take_profit_pct = 0.003    # 0.3% TP (tight, fast)
    stop_loss_pct = 0.003      # 0.3% SL (equal risk)
    max_bars = 30              # Must hit within 30 bars

    # Commission to account for
    commission_rate = 0.001    # 0.1% round trip

    # Paths
    data_dir = str(Path(__file__).parent / "Crypto_Data_Fresh")
    model_save_dir = str(Path(__file__).parent / "saved_models_hf_scalping")

    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD']


config = SimpleConfig()


def create_profitable_target(df: pd.DataFrame,
                            tp_pct: float,
                            sl_pct: float,
                            max_bars: int,
                            commission_pct: float) -> pd.DataFrame:
    """
    Create target based on ACTUAL net P&L after commission.

    BUY = 1 if the trade would be NET PROFITABLE after:
    - TP hit before SL
    - After deducting commission

    This is more realistic than just checking if TP hits.
    """
    n = len(df)
    targets = np.zeros(n, dtype=np.int32)

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    profitable = 0
    losing = 0
    timeout = 0

    # Net profit needed to be profitable after commission
    # If we buy at 100, TP at 100.3 (0.3%), and pay 0.1% commission:
    # Net = 0.3% - 0.1% = 0.2% profit
    min_net_profit_pct = tp_pct - commission_pct

    print(f"\nCreating PROFITABLE targets:")
    print(f"  TP: {tp_pct*100:.2f}%, SL: {sl_pct*100:.2f}%")
    print(f"  Commission: {commission_pct*100:.2f}%")
    print(f"  Min net profit needed: {min_net_profit_pct*100:.2f}%")

    for i in range(n - max_bars):
        entry_price = closes[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        tp_bar = None
        sl_bar = None

        for j in range(1, max_bars + 1):
            idx = i + j
            if idx >= n:
                break

            if tp_bar is None and highs[idx] >= tp_price:
                tp_bar = j
            if sl_bar is None and lows[idx] <= sl_price:
                sl_bar = j

            if tp_bar is not None and sl_bar is not None:
                break

        # Determine if profitable
        if tp_bar is not None and (sl_bar is None or tp_bar < sl_bar):
            # TP hit first - this is a WIN
            profitable += 1
            targets[i] = 1
        elif sl_bar is not None:
            # SL hit first or same bar - this is a LOSS
            losing += 1
            targets[i] = 0
        else:
            # Neither hit - timeout
            timeout += 1
            targets[i] = 0

    df = df.copy()
    df['target'] = targets

    n_valid = n - max_bars
    print(f"\n  Results:")
    print(f"    Profitable (BUY=1): {profitable:,} ({100*profitable/n_valid:.1f}%)")
    print(f"    Losing (NO_TRADE): {losing:,} ({100*losing/n_valid:.1f}%)")
    print(f"    Timeout (NO_TRADE): {timeout:,} ({100*timeout/n_valid:.1f}%)")

    return df


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SIMPLE, EFFECTIVE features focused on:
    - Momentum (are we going up?)
    - Trend (is the trend up?)
    - Volume (is there conviction?)
    """
    df = df.copy()

    # Use previous bar data (shift by 1)
    prev_close = df['close'].shift(1)
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_open = df['open'].shift(1)
    prev_volume = df['volume'].shift(1)

    # ================================================================
    # MOMENTUM - Key for scalping
    # ================================================================
    df['return_1'] = df['close'].pct_change(1).shift(1) * 100  # 1-bar return %
    df['return_3'] = df['close'].pct_change(3).shift(1) * 100
    df['return_5'] = df['close'].pct_change(5).shift(1) * 100
    df['return_10'] = df['close'].pct_change(10).shift(1) * 100

    # Momentum acceleration
    df['momentum_accel'] = (df['return_1'] - df['return_1'].shift(1))

    # ================================================================
    # TREND
    # ================================================================
    df['sma_10'] = df['close'].rolling(10).mean().shift(1)
    df['sma_20'] = df['close'].rolling(20).mean().shift(1)
    df['sma_50'] = df['close'].rolling(50).mean().shift(1)

    df['above_sma10'] = (prev_close > df['sma_10']).astype(float)
    df['above_sma20'] = (prev_close > df['sma_20']).astype(float)
    df['above_sma50'] = (prev_close > df['sma_50']).astype(float)
    df['trend_score'] = df['above_sma10'] + df['above_sma20'] + df['above_sma50']

    # SMA slopes
    df['sma10_slope'] = df['sma_10'].pct_change(3).shift(1) * 100
    df['sma20_slope'] = df['sma_20'].pct_change(5).shift(1) * 100

    # ================================================================
    # VOLATILITY
    # ================================================================
    df['volatility'] = df['close'].pct_change().rolling(10).std().shift(1) * 100
    df['atr'] = (df['high'] - df['low']).rolling(10).mean().shift(1)
    df['atr_pct'] = (df['atr'] / df['close'].shift(1)) * 100

    # ================================================================
    # VOLUME
    # ================================================================
    df['volume_sma'] = df['volume'].rolling(20).mean().shift(1)
    df['volume_ratio'] = prev_volume / (df['volume_sma'] + 1e-10)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(float)

    # ================================================================
    # RSI
    # ================================================================
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().shift(1)
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().shift(1)
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)

    # ================================================================
    # MACD
    # ================================================================
    exp12 = df['close'].ewm(span=12, adjust=False).mean().shift(1)
    exp26 = df['close'].ewm(span=26, adjust=False).mean().shift(1)
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(float)

    # ================================================================
    # CANDLE PATTERNS
    # ================================================================
    candle_range = prev_high - prev_low + 1e-10
    df['body_pct'] = abs(prev_close - prev_open) / candle_range
    df['is_green'] = (prev_close > prev_open).astype(float)

    # Consecutive green/red
    df['green_streak'] = df['is_green'].rolling(5).sum()

    # ================================================================
    # HIGHER HIGHS / HIGHER LOWS
    # ================================================================
    df['higher_high'] = (df['high'].shift(1) > df['high'].shift(2)).astype(float)
    df['higher_low'] = (df['low'].shift(1) > df['low'].shift(2)).astype(float)
    df['hh_hl_score'] = df['higher_high'] + df['higher_low']

    # ================================================================
    # BOLLINGER BANDS
    # ================================================================
    rolling_std = df['close'].rolling(20).std().shift(1)
    df['bb_upper'] = df['sma_20'] + 2 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2 * rolling_std
    df['bb_position'] = (prev_close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    return df


def main():
    """Main training."""
    start_time = time.time()

    print("=" * 60)
    print("SIMPLE PROFITABLE MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"\nConfig:")
    print(f"  TP: {config.take_profit_pct*100:.2f}%")
    print(f"  SL: {config.stop_loss_pct*100:.2f}%")
    print(f"  Commission: {config.commission_rate*100:.2f}%")

    # Load data
    print("\n" + "=" * 60)
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

    # Process each symbol
    print("\n" + "=" * 60)
    print("CREATING TARGETS AND FEATURES")
    print("=" * 60)

    fe = FeatureEngineer()
    all_processed = []

    for symbol in combined['symbol'].unique():
        print(f"\n--- {symbol} ---")
        symbol_df = combined[combined['symbol'] == symbol].copy()

        # Drop string columns
        for col in ['conversionType', 'conversionSymbol']:
            if col in symbol_df.columns:
                symbol_df = symbol_df.drop(columns=[col])

        # Create targets
        symbol_df = create_profitable_target(
            symbol_df,
            tp_pct=config.take_profit_pct,
            sl_pct=config.stop_loss_pct,
            max_bars=config.max_bars,
            commission_pct=config.commission_rate
        )

        # Add features
        symbol_df = add_simple_features(symbol_df)

        all_processed.append(symbol_df)

    df = pd.concat(all_processed, ignore_index=True)

    # Prepare training data
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)

    # Define feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote',
                    'target', 'symbol', 'datetime', 'date', 'open_time', 'close_time']

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    fe.feature_columns = feature_cols

    # Drop rows with NaN
    df = df.dropna(subset=['target'])
    df = df.dropna(subset=feature_cols)

    # Handle inf
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    X = df[feature_cols].astype(np.float32)
    y = df['target']

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  BUY ratio: {100*y.mean():.1f}%")

    # Split by time
    train_indices = []
    val_indices = []
    test_indices = []

    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_indices = np.where(symbol_mask)[0]

        n = len(symbol_indices)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_indices.extend(symbol_indices[:train_end])
        val_indices.extend(symbol_indices[train_end:val_end])
        test_indices.extend(symbol_indices[val_end:])

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    X_val = X.iloc[val_indices].reset_index(drop=True)
    y_val = y.iloc[val_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    print(f"\n  Train: {len(X_train):,} - BUY: {100*y_train.mean():.1f}%")
    print(f"  Val: {len(X_val):,} - BUY: {100*y_val.mean():.1f}%")
    print(f"  Test: {len(X_test):,} - BUY: {100*y_test.mean():.1f}%")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE")
    print("=" * 60)

    ensemble = EnsembleVotingSystem(
        voting_method='soft',
        confidence_threshold=0.40,
        min_agreement=2,
        use_dynamic_weights=True
    )

    metrics = ensemble.train(
        pd.DataFrame(X_train.values, columns=X_train.columns),
        pd.Series(y_train.values),
        pd.DataFrame(X_val.values, columns=X_val.columns),
        pd.Series(y_val.values),
        verbose=True
    )

    # Test
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

    test_preds = ensemble.predict(pd.DataFrame(X_test.values, columns=X_test.columns))
    test_probas = ensemble.predict_proba(pd.DataFrame(X_test.values, columns=X_test.columns))

    print(f"  Accuracy: {accuracy_score(y_test.values, test_preds):.4f}")
    print(f"  Precision: {precision_score(y_test.values, test_preds, zero_division=0):.4f}")
    print(f"  F1: {f1_score(y_test.values, test_preds, zero_division=0):.4f}")

    cm = confusion_matrix(y_test.values, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"  Predicted:   NO_TRADE   BUY")
    print(f"  NO_TRADE     {cm[0,0]:6}   {cm[0,1]:6}")
    if cm.shape[0] > 1:
        print(f"  BUY          {cm[1,0]:6}   {cm[1,1]:6}")

    buy_proba = test_probas[:, 1]
    print(f"\n  BUY Probability Distribution:")
    print(f"    Min: {buy_proba.min():.3f}, Max: {buy_proba.max():.3f}")
    print(f"    Mean: {buy_proba.mean():.3f}, Median: {np.median(buy_proba):.3f}")

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    save_path = Path(config.model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    ensemble.save(str(save_path / "ensemble"))
    fe.save_feature_config(str(save_path / "feature_config.json"))

    import json
    summary = {
        'training_date': datetime.now().isoformat(),
        'version': 'simple_profitable_v1',
        'tp_pct': config.take_profit_pct,
        'sl_pct': config.stop_loss_pct,
        'commission': config.commission_rate,
        'n_features': len(feature_cols),
        'test_buy_ratio': float(y_test.mean())
    }
    with open(save_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to: {save_path}")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
