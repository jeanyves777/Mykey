"""
RETRAIN ALL 5 ML MODELS - DCA-AWARE HF DAY TRADING
===================================================
Fresh training on all 5 crypto symbols for BUY-only spot trading with DCA.

Target Config (HF Day Trading with DCA):
- Conf >= 60%
- Model Agreement: 5/5 unanimous
- Stop Loss: 2.5% (final, after DCA exhausted)
- Take Profit: 1% (quick exit)
- DCA Trigger: 0.5% drop triggers first DCA
- DCA Spacing: 0.5% between each DCA level
- DCA Profit Target: 0.4% on averaged position
- Max DCA Entries: 3 (4 total entries)
- BUY-ONLY signals (no short selling for spot crypto)

Symbols: BTCUSD, ETHUSD, SOLUSD, DOGEUSD, AVAXUSD
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime


def load_all_data():
    """Load 5-minute data for all 5 symbols."""

    print("\n" + "=" * 80)
    print("STAGE 1: LOADING DATA")
    print("=" * 80)

    data_path = Path(__file__).parent / "Crypto_Data_5m"
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']

    all_data = []

    for symbol in symbols:
        filepath = data_path / f"{symbol}_5m.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['symbol'] = symbol

            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])

            all_data.append(df)

            start = df['datetime'].iloc[0] if 'datetime' in df.columns else 'N/A'
            end = df['datetime'].iloc[-1] if 'datetime' in df.columns else 'N/A'
            print(f"  {symbol}: {len(df):,} bars | {start} to {end}")
        else:
            print(f"  {symbol}: FILE NOT FOUND")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined):,} bars across {len(symbols)} symbols")

    return combined


def engineer_features(df):
    """Engineer all features for training."""

    print("\n" + "=" * 80)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 80)

    from features.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()

    # Process each symbol separately to avoid cross-symbol contamination
    symbols = df['symbol'].unique()
    all_featured = []

    for symbol in symbols:
        print(f"  Processing {symbol}...")
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_featured = fe.compute_all_features(symbol_df)
        all_featured.append(symbol_featured)

    featured_df = pd.concat(all_featured, ignore_index=True)

    # Create asymmetric targets (top/bottom 15% extreme moves)
    print("\n  Creating asymmetric targets (top/bottom 15% moves)...")
    featured_df = fe.create_target(
        featured_df,
        use_asymmetric_targets=True,
        extreme_pct=15.0,  # Top/bottom 15%
        forward_period=6   # 30 min at 5-min bars
    )

    # Get feature columns
    feature_columns = fe.get_feature_columns(featured_df)

    # Remove non-numeric columns
    exclude = ['symbol', 'is_extreme_move', 'open_time', 'timestamp', 'datetime',
               'open', 'high', 'low', 'close', 'volume', 'target', 'forward_return']
    feature_columns = [c for c in feature_columns if c not in exclude]

    # Keep only numeric columns
    numeric_cols = featured_df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = numeric_cols

    print(f"\n  Features: {len(feature_columns)}")
    print(f"  Total rows: {len(featured_df):,}")

    # Target distribution
    if 'target' in featured_df.columns:
        target_counts = featured_df['target'].value_counts()
        print(f"\n  Target distribution:")
        for label, count in target_counts.items():
            name = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}.get(label, str(label))
            print(f"    {name}: {count:,} ({100*count/len(featured_df):.1f}%)")

    return featured_df, feature_columns, fe


def prepare_training_data(df, feature_columns):
    """Prepare training and validation data."""

    print("\n" + "=" * 80)
    print("STAGE 3: PREPARING TRAINING DATA")
    print("=" * 80)

    # Remove NaN
    df = df.dropna(subset=['target'] + feature_columns)

    # Split 80/20 train/validation
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()

    # Prepare features
    X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df['target'].values

    X_val = val_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = val_df['target'].values

    print(f"  Training: {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")

    # For binary classification, filter to extreme moves only and convert to binary
    # -1 (SELL) -> 0, 1 (BUY) -> 1, ignore 0 (HOLD)
    extreme_mask_train = y_train != 0
    X_train_extreme = X_train[extreme_mask_train]
    y_train_extreme = y_train[extreme_mask_train]
    y_train_binary = np.where(y_train_extreme == 1, 1, 0)

    extreme_mask_val = y_val != 0
    X_val_extreme = X_val[extreme_mask_val]
    y_val_extreme = y_val[extreme_mask_val]
    y_val_binary = np.where(y_val_extreme == 1, 1, 0)

    print(f"\n  Training (extreme moves only): {len(X_train_extreme):,}")
    print(f"    BUY: {np.sum(y_train_binary == 1):,}")
    print(f"    SELL: {np.sum(y_train_binary == 0):,}")

    print(f"\n  Validation (extreme moves only): {len(X_val_extreme):,}")
    print(f"    BUY: {np.sum(y_val_binary == 1):,}")
    print(f"    SELL: {np.sum(y_val_binary == 0):,}")

    return X_train_extreme, y_train_binary, X_val_extreme, y_val_binary, val_df


def train_all_models(X_train, y_train, X_val, y_val):
    """Train all 5 ML models."""

    print("\n" + "=" * 80)
    print("STAGE 4: TRAINING ALL 5 ML MODELS")
    print("=" * 80)

    from ml_models.random_forest_model import RandomForestModel
    from ml_models.xgboost_model import XGBoostModel
    from ml_models.lightgbm_model import LightGBMModel
    from ml_models.catboost_model import CatBoostModel
    from ml_models.neural_network_model import NeuralNetworkModel

    model_classes = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'neural_network': NeuralNetworkModel
    }

    models = {}
    metrics = {}

    for name, model_class in model_classes.items():
        print(f"\n  Training {name}...")

        try:
            model = model_class()

            # Train with validation data
            train_metrics = model.train(
                pd.DataFrame(X_train, columns=X_train.columns),
                pd.Series(y_train),
                pd.DataFrame(X_val, columns=X_val.columns),
                pd.Series(y_val)
            )

            models[name] = model

            # Get validation accuracy
            val_preds = model.predict(X_val)
            val_acc = np.mean(val_preds == y_val)

            # Get validation probabilities for confidence calibration
            try:
                val_probas = model.predict_proba(X_val)
                if val_probas.shape[1] > 1:
                    max_conf = np.max(val_probas, axis=1)
                    avg_conf = np.mean(max_conf)
                else:
                    avg_conf = np.mean(val_probas)
            except:
                avg_conf = val_acc

            metrics[name] = {
                'val_accuracy': val_acc,
                'avg_confidence': avg_conf,
                **train_metrics
            }

            print(f"    Accuracy: {val_acc:.1%}")
            print(f"    Avg Confidence: {avg_conf:.3f}")

        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\n  Successfully trained {len(models)}/5 models")

    return models, metrics


def save_models(models, feature_columns, metrics):
    """Save all trained models."""

    print("\n" + "=" * 80)
    print("STAGE 5: SAVING MODELS")
    print("=" * 80)

    save_dir = Path(__file__).parent / "saved_models_improved"
    ensemble_dir = save_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Save each model
    for name, model in models.items():
        print(f"  Saving {name}...")
        try:
            model.save(str(ensemble_dir / name))
            print(f"    OK")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Save feature config
    feature_config = {
        'feature_columns': feature_columns,
        'n_features': len(feature_columns)
    }

    with open(save_dir / "feature_config.json", 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"  Saved feature_config.json ({len(feature_columns)} features)")

    # Save training summary
    training_summary = {
        'training_date': datetime.now().isoformat(),
        'timeframe': '5m',
        'use_asymmetric_targets': True,
        'extreme_pct': 15.0,
        'forward_period': 6,
        'n_features': len(feature_columns),
        'models_trained': list(models.keys()),
        'metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()}
                   for k, v in metrics.items()}
    }

    with open(save_dir / "training_summary.json", 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"  Saved training_summary.json")

    # Save ensemble metadata (DCA-aware HF day trading config)
    ensemble_meta = {
        'voting_method': 'soft',
        'confidence_threshold': 0.60,     # V18 optimal: 60%
        'min_agreement': 5,               # V18 optimal: 5/5 unanimous
        'use_dynamic_weights': True,
        'feature_names': feature_columns,
        'is_trained': True,
        'buy_only_mode': True,            # BUY-only for spot trading
        # DCA Settings (tight for HF day trading)
        'dca_enabled': True,
        'dca_trigger_pct': 0.005,         # 0.5% drop triggers first DCA
        'dca_spacing_pct': 0.005,         # 0.5% between each DCA level
        'max_dca_entries': 3,             # Max 3 DCA adds
        'dca_multiplier': 1.5,            # Each DCA is 1.5x previous
        'dca_profit_target_pct': 0.004,   # 0.4% profit on avg to exit
        'stop_loss_pct': 0.025,           # 2.5% final SL
        'take_profit_pct': 0.01,          # 1% TP
    }

    with open(ensemble_dir / "ensemble_metadata.json", 'w') as f:
        json.dump(ensemble_meta, f, indent=2)
    print(f"  Saved ensemble_metadata.json")

    print(f"\nAll models saved to: {save_dir}")


def run_quick_backtest(models, val_df, feature_columns):
    """Run quick backtest on validation data."""

    print("\n" + "=" * 80)
    print("STAGE 6: QUICK BACKTEST ON VALIDATION DATA")
    print("=" * 80)

    from ultimate_trading_system import UltimateRiskConfig, UltimateTradingEngine

    # Prepare features
    X = val_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get predictions from all models
    all_predictions = {}
    all_probas = {}

    print("\n  Model predictions:")
    for name, model in models.items():
        try:
            preds = model.predict(X)
            all_predictions[name] = preds

            try:
                probas = model.predict_proba(X)
                all_probas[name] = probas
            except:
                all_probas[name] = np.column_stack([1 - preds, preds])

            buy_pct = 100 * np.sum(preds == 1) / len(preds)
            print(f"    {name}: {buy_pct:.1f}% BUY")
        except Exception as e:
            print(f"    {name}: FAILED - {e}")

    if len(all_predictions) == 0:
        print("  No predictions available!")
        return

    # Ensemble voting
    pred_matrix = np.column_stack(list(all_predictions.values()))
    predictions = np.round(np.mean(pred_matrix, axis=1)).astype(int)

    # Average confidence
    proba_list = []
    for name, probas in all_probas.items():
        if probas.shape[1] > 1:
            proba_list.append(probas[:, 1])
        else:
            proba_list.append(probas.flatten())
    confidences = np.mean(proba_list, axis=0)

    # Model votes
    model_names = list(all_predictions.keys())
    model_votes = []
    for i in range(len(predictions)):
        votes = {name: int(all_predictions[name][i]) for name in model_names}
        model_votes.append(votes)

    print(f"\n  Ensemble: {np.sum(predictions == 1)}/{len(predictions)} BUY signals")
    print(f"  Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")

    # Run backtest with DCA-aware HF day trading config
    config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.60,              # V18 optimal: 60%
        min_model_agreement=5,            # 5/5 unanimous
        stop_loss_pct=0.025,              # 2.5% final SL (after DCA)
        take_profit_pct=0.01,             # 1% TP (quick exit)
        min_risk_reward=1.0,              # Lower R:R since DCA helps
        max_trades_per_day=50,            # Higher for HF trading
        require_trend_alignment=False,
        min_volume_ratio=0.5,
        min_volatility=0.0001,
        cooldown_minutes=3,               # 3 min cooldown for HF
    )

    engine = UltimateTradingEngine(config=config)

    results = engine.run_backtest(
        df=val_df.reset_index(drop=True),
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=True
    )

    # Sensitivity analysis
    print("\n" + "-" * 60)
    print("SENSITIVITY ANALYSIS")
    print("-" * 60)
    print(f"{'Conf':>6} {'Agree':>5} {'Trades':>8} {'Win Rate':>10} {'P&L':>10}")
    print("-" * 60)

    for conf in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for agree in [3, 4, 5]:
            config.min_confidence = conf
            config.min_model_agreement = agree

            eng = UltimateTradingEngine(config=config)
            res = eng.run_backtest(
                df=val_df.reset_index(drop=True),
                predictions=predictions,
                confidences=confidences,
                model_votes=model_votes,
                verbose=False
            )

            if res['total_trades'] > 0:
                wr = res['win_rate'] * 100
                pnl = res['total_return']
                trades = res['total_trades']
                marker = " ***" if wr >= 80 else " **" if wr >= 70 else " *" if wr >= 60 else ""
                print(f"{conf:>5.0%} {agree:>5}/5 {trades:>8} {wr:>9.1f}% ${pnl:>9.2f}{marker}")

    return results


def main():
    """Main training pipeline."""

    print("\n" + "=" * 80)
    print("RETRAINING ALL 5 ML MODELS - DCA-AWARE HF DAY TRADING")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print("\nDCA Strategy Config (HF Day Trading):")
    print("  - BUY-ONLY signals (spot trading)")
    print("  - Confidence >= 60% (V18 optimal)")
    print("  - Model Agreement: 5/5 unanimous")
    print("  - Stop Loss: 2.5% (final, after DCA)")
    print("  - Take Profit: 1% (quick exit)")
    print("  - DCA Trigger: 0.5% drop")
    print("  - DCA Spacing: 0.5% between levels")
    print("  - DCA Profit Target: 0.4% on avg")
    print("  - Max DCA Entries: 3")
    print("\nSymbols: BTCUSD, ETHUSD, SOLUSD, DOGEUSD, AVAXUSD")

    # Load data
    df = load_all_data()

    # Engineer features
    featured_df, feature_columns, fe = engineer_features(df)

    # Prepare training data
    X_train, y_train, X_val, y_val, val_df = prepare_training_data(featured_df, feature_columns)

    # Train all models
    models, metrics = train_all_models(X_train, y_train, X_val, y_val)

    # Save models
    save_models(models, feature_columns, metrics)

    # Quick backtest
    run_quick_backtest(models, val_df, feature_columns)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now()}")
    print(f"Models saved to: saved_models_improved/")
    print("\nNext step: Run run_1week_pretrained.py to evaluate on 1-week test data")


if __name__ == "__main__":
    main()
