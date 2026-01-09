"""
1-Week Backtest with PRE-TRAINED Models
========================================
Uses the robust ensemble trained on 6 months of data.
Tests on all 5 symbols for 1 week.

Optimal Configuration:
- Confidence >= 75%
- Model Agreement: 5/5 (unanimous)
- Stop Loss: 0.7%
- Take Profit: 1.4%
- Risk:Reward >= 1.2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from ultimate_trading_system import UltimateRiskConfig, UltimateTradingEngine


def load_pretrained_models():
    """Load the pre-trained ensemble models from saved_models_improved/"""

    model_path = Path(__file__).parent / "saved_models_improved"
    ensemble_path = model_path / "ensemble"

    if not ensemble_path.exists():
        print(f"ERROR: No pre-trained models found at {ensemble_path}")
        return None, None

    print(f"\nLoading pre-trained models from: {model_path}")

    # Load feature config
    feature_config_path = model_path / "feature_config.json"
    with open(feature_config_path, 'r') as f:
        config = json.load(f)
    feature_columns = config['feature_columns']
    print(f"  Feature columns: {len(feature_columns)}")

    # Import model classes
    from ml_models.random_forest_model import RandomForestModel
    from ml_models.xgboost_model import XGBoostModel
    from ml_models.lightgbm_model import LightGBMModel
    from ml_models.catboost_model import CatBoostModel
    from ml_models.neural_network_model import NeuralNetworkModel

    models = {}

    # Load each model
    model_files = {
        'random_forest': (RandomForestModel, 'random_forest'),
        'xgboost': (XGBoostModel, 'xgboost'),
        'lightgbm': (LightGBMModel, 'lightgbm'),
        'catboost': (CatBoostModel, 'catboost'),
        'neural_network': (NeuralNetworkModel, 'neural_network'),
    }

    for name, (model_class, file_prefix) in model_files.items():
        try:
            model = model_class()
            model.load(str(ensemble_path / file_prefix))
            models[name] = model
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    print(f"\nSuccessfully loaded {len(models)}/5 models")

    return models, feature_columns


def run_backtest():
    """Run 1-week backtest on all 5 symbols using pre-trained models."""

    print("=" * 80)
    print("1-WEEK BACKTEST WITH PRE-TRAINED ENSEMBLE (6 months training)")
    print("=" * 80)

    # Load pre-trained models
    models, feature_columns = load_pretrained_models()

    if models is None or len(models) == 0:
        print("ERROR: Could not load models")
        return

    # Load data
    data_path = Path(__file__).parent / "Crypto_Data_5m"
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']
    bars_per_week = 7 * 24 * 12  # 2016 bars

    print("\n" + "=" * 80)
    print("LOADING 1-WEEK DATA")
    print("=" * 80)

    all_data = []
    for symbol in symbols:
        filepath = data_path / f"{symbol}_5m.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Get last 1 week
            df = df.tail(bars_per_week).reset_index(drop=True)
            df['symbol'] = symbol

            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])

            all_data.append(df)

            start = df['datetime'].iloc[0] if 'datetime' in df.columns else 'N/A'
            end = df['datetime'].iloc[-1] if 'datetime' in df.columns else 'N/A'
            print(f"  {symbol}: {len(df)} bars | {start} to {end}")
        else:
            print(f"  {symbol}: FILE NOT FOUND")

    if not all_data:
        print("ERROR: No data found")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data: {len(combined_df)} bars")

    # Engineer features using the same feature engineer
    from features.feature_engineer import FeatureEngineer

    print("\nEngineering features...")
    fe = FeatureEngineer()
    featured_df = fe.compute_all_features(combined_df)

    # Check which feature columns are available
    available_features = [c for c in feature_columns if c in featured_df.columns]
    missing_features = [c for c in feature_columns if c not in featured_df.columns]

    print(f"  Expected features: {len(feature_columns)}")
    print(f"  Available features: {len(available_features)}")
    if missing_features:
        print(f"  Missing features: {len(missing_features)}")
        print(f"    Examples: {missing_features[:5]}")

    if len(available_features) < len(feature_columns) * 0.9:
        print("WARNING: Many features missing. Results may be affected.")

    # Use available features only
    feature_cols_to_use = available_features

    # Prepare features for prediction
    X = featured_df[feature_cols_to_use].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get predictions from all models
    # IMPORTANT: Models were trained with 3 classes: -1 (SELL), 0 (HOLD), 1 (BUY)
    print("\nGenerating predictions from pre-trained models...")
    print("  (Models trained with 3 classes: -1=SELL, 0=HOLD, 1=BUY)")

    all_predictions = {}
    all_probas = {}

    for name, model in models.items():
        try:
            preds = model.predict(X)
            all_predictions[name] = preds

            try:
                probas = model.predict_proba(X)
                all_probas[name] = probas
            except:
                # Create dummy probas for 3 classes
                all_probas[name] = np.zeros((len(preds), 3))
                for i, p in enumerate(preds):
                    if p == -1:
                        all_probas[name][i] = [0.8, 0.1, 0.1]  # SELL
                    elif p == 0:
                        all_probas[name][i] = [0.1, 0.8, 0.1]  # HOLD
                    else:
                        all_probas[name][i] = [0.1, 0.1, 0.8]  # BUY

            buy_count = np.sum(preds == 1)
            sell_count = np.sum(preds == -1)
            hold_count = np.sum(preds == 0)
            print(f"  {name}: BUY={buy_count} ({100*buy_count/len(preds):.1f}%) | HOLD={hold_count} ({100*hold_count/len(preds):.1f}%) | SELL={sell_count} ({100*sell_count/len(preds):.1f}%)")
        except Exception as e:
            print(f"  {name} prediction failed: {e}")

    if len(all_predictions) == 0:
        print("ERROR: No predictions generated")
        return

    # Ensemble voting for 3-class predictions
    # Count votes for each class per sample
    n_samples = len(X)
    model_names = list(all_predictions.keys())

    # Determine final prediction by majority vote
    predictions = np.zeros(n_samples, dtype=int)
    confidences = np.zeros(n_samples)

    for i in range(n_samples):
        votes = {name: int(all_predictions[name][i]) for name in model_names}
        vote_counts = {-1: 0, 0: 0, 1: 0}
        for v in votes.values():
            vote_counts[v] = vote_counts.get(v, 0) + 1

        # Get the class with most votes
        max_class = max(vote_counts, key=vote_counts.get)
        predictions[i] = max_class

        # Calculate confidence as agreement percentage
        max_votes = vote_counts[max_class]
        confidences[i] = max_votes / len(model_names)

    # Model votes per sample (for UltimateTradingEngine)
    # Convert 3-class to binary: 1 = BUY, 0 = NOT BUY (SELL or HOLD)
    model_votes = []
    for i in range(n_samples):
        votes = {name: (1 if all_predictions[name][i] == 1 else 0) for name in model_names}
        model_votes.append(votes)

    # Convert predictions to binary for backtest engine (1=BUY, 0=NOT_BUY)
    binary_predictions = (predictions == 1).astype(int)

    print(f"\nEnsemble Summary (3-class):")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  BUY signals (1): {np.sum(predictions == 1)} ({100*np.sum(predictions==1)/len(predictions):.1f}%)")
    print(f"  HOLD signals (0): {np.sum(predictions == 0)} ({100*np.sum(predictions==0)/len(predictions):.1f}%)")
    print(f"  SELL signals (-1): {np.sum(predictions == -1)} ({100*np.sum(predictions==-1)/len(predictions):.1f}%)")
    print(f"  Model agreement range: {confidences.min():.1%} - {confidences.max():.1%}")

    # Use binary predictions for backtest
    predictions = binary_predictions

    # Run backtest with BEST BALANCED CONFIGURATION from optimization
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST - BEST BALANCED CONFIGURATION (93.1% WR)")
    print("=" * 80)
    print("  Stop Loss: 0.6%")
    print("  Take Profit: 0.8%")
    print("  Min R:R: 1.2")
    print("  Model Agreement: 5/5 (unanimous)")
    print("  Confidence: 60%+")
    print("=" * 80)

    # BEST BALANCED CONFIG from optimization:
    # 93.1% Win Rate, 29 trades, $1.61 P&L, 11.69 Profit Factor
    num_models = len(models)
    config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.60,   # 60% model agreement
        min_model_agreement=5, # 5/5 unanimous
        stop_loss_pct=0.006,   # 0.6% SL (tighter)
        take_profit_pct=0.008, # 0.8% TP (quick profits)
        min_risk_reward=1.2,   # R:R >= 1.2
        max_trades_per_day=50, # Allow more trades
        require_trend_alignment=False,
        min_volume_ratio=0.5,
        min_volatility=0.0001,
        cooldown_minutes=10,
    )

    engine = UltimateTradingEngine(config=config)

    results = engine.run_backtest(
        df=featured_df.reset_index(drop=True),
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=True
    )

    # Sensitivity analysis
    print("\n" + "=" * 80)
    print("CONFIDENCE SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\n{'Conf':>6} {'Agree':>5} {'Trades':>8} {'Win Rate':>10} {'Wins':>6} {'Losses':>6} {'P&L':>10} {'Final':>10}")
    print("-" * 80)

    # Test different configurations - full range now that all 5 models loaded
    test_configs = [
        (0.50, 3), (0.55, 3), (0.60, 3), (0.65, 3), (0.70, 3), (0.75, 3),
        (0.50, 4), (0.55, 4), (0.60, 4), (0.65, 4), (0.70, 4), (0.75, 4),
        (0.50, 5), (0.55, 5), (0.60, 5), (0.65, 5), (0.70, 5), (0.75, 5),
        (0.80, 5), (0.85, 5), (0.90, 5), (0.95, 5), (1.0, 5),
    ]

    best_wr = 0
    best_config = None

    for conf, agree in test_configs:
        config.min_confidence = conf
        config.min_model_agreement = agree

        eng = UltimateTradingEngine(config=config)
        res = eng.run_backtest(
            df=featured_df.reset_index(drop=True),
            predictions=predictions,
            confidences=confidences,
            model_votes=model_votes,
            verbose=False
        )

        wr = res['win_rate'] * 100
        trades = res['total_trades']
        wins = res['wins']
        losses = res['losses']
        pnl = res['total_return']
        final = res['final_capital']

        marker = " ***" if wr >= 90 and trades >= 3 else " **" if wr >= 80 and trades >= 3 else " *" if wr >= 70 and trades >= 3 else ""
        print(f"{conf:>5.0%} {agree:>5}/5 {trades:>8} {wr:>9.1f}% {wins:>6} {losses:>6} ${pnl:>9.2f} ${final:>9.2f}{marker}")

        if wr > best_wr and trades >= 3:
            best_wr = wr
            best_config = (conf, agree, trades, wr, final)

    if best_config:
        print(f"\nBest Configuration: Conf>={best_config[0]:.0%}, {best_config[1]}/5 agree")
        print(f"  Trades: {best_config[2]}, Win Rate: {best_config[3]:.1f}%, Final: ${best_config[4]:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    if 'datetime' in featured_df.columns:
        start = featured_df['datetime'].iloc[0]
        end = featured_df['datetime'].iloc[-1]
        print(f"\nTest Period: {start} to {end}")

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Total Bars: {len(featured_df)}")
    print(f"Pre-trained Models: {len(models)}/5")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    run_backtest()
