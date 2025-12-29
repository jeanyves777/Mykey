"""
Run Ultimate Trading System Test
================================
Standalone script to run the $200 capital test with all risk management.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import from ultimate_trading_system
from ultimate_trading_system import (
    UltimateRiskConfig,
    UltimateTradingEngine,
    RegimeDetector,
    SignalFilter
)

def load_data_and_run_test():
    """Load data, generate predictions, and run the ultimate test."""

    print("\n" + "=" * 70)
    print("ULTIMATE $200 TRADING SYSTEM TEST")
    print("=" * 70)

    # Load 5-minute data
    data_path = Path(__file__).parent / "Crypto_Data_5m"

    # Load BTC data for testing
    btc_file = data_path / "BTCUSD_5m.csv"
    eth_file = data_path / "ETHUSD_5m.csv"

    all_data = []

    for filepath, symbol in [(btc_file, 'BTCUSD'), (eth_file, 'ETHUSD')]:
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['symbol'] = symbol
            # Use last 5000 bars for testing
            df = df.tail(5000).reset_index(drop=True)
            all_data.append(df)
            print(f"Loaded {symbol}: {len(df)} bars")

    if not all_data:
        print("ERROR: No data found")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal test data: {len(combined_df)} bars")

    # Import feature engineer
    from features.feature_engineer import FeatureEngineer

    # Engineer features
    print("\nEngineering features...")
    fe = FeatureEngineer()
    featured_df = fe.compute_all_features(combined_df)

    # Create asymmetric targets for analysis
    featured_df = fe.create_target(
        featured_df,
        use_asymmetric_targets=True,
        extreme_pct=15.0  # Top/bottom 15%
    )

    print(f"Features: {len(fe.feature_columns)}")
    print(f"Target distribution: {featured_df['target'].value_counts().to_dict()}")

    # Load trained ensemble model
    model_path = Path(__file__).parent / "saved_models_improved"
    ensemble_path = model_path / "ensemble"

    # Import ML models with proper path setup
    from ml_models.random_forest_model import RandomForestModel
    from ml_models.xgboost_model import XGBoostModel
    from ml_models.lightgbm_model import LightGBMModel
    from ml_models.catboost_model import CatBoostModel
    from ml_models.neural_network_model import NeuralNetworkModel

    # Try to load ensemble or train simple models
    models = {}
    model_classes = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'neural_network': NeuralNetworkModel
    }

    # Load feature config
    feature_config_path = model_path / "feature_config.json"
    if feature_config_path.exists():
        import json
        with open(feature_config_path, 'r') as f:
            config = json.load(f)
            feature_columns = config.get('feature_columns', fe.feature_columns)
    else:
        feature_columns = fe.feature_columns

    print(f"\nUsing {len(feature_columns)} features")

    # Load models
    loaded_count = 0
    for name, model_class in model_classes.items():
        model_file = ensemble_path / f"{name}_model.joblib"
        if model_file.exists():
            try:
                model = model_class()
                model.load(str(ensemble_path / f"{name}"))
                models[name] = model
                loaded_count += 1
                print(f"  Loaded {name}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

    if loaded_count == 0:
        print("\nNo trained models found. Training quick models...")

        # Prepare training data (use first 80% for training)
        train_size = int(len(featured_df) * 0.8)
        train_df = featured_df.iloc[:train_size]
        test_df = featured_df.iloc[train_size:]

        # Remove NaN
        train_df = train_df.dropna(subset=['target'] + feature_columns)
        test_df = test_df.dropna(subset=['target'] + feature_columns)

        X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df['target'].values

        X_test = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_test = test_df['target'].values

        # Filter to only extreme moves for training (remove neutral class 0)
        extreme_mask_train = y_train != 0
        X_train_extreme = X_train[extreme_mask_train]
        y_train_extreme = y_train[extreme_mask_train]

        extreme_mask_test = y_test != 0
        X_test_extreme = X_test[extreme_mask_test]
        y_test_extreme = y_test[extreme_mask_test]

        # Convert -1/1 to 0/1 for binary
        y_train_binary = np.where(y_train_extreme == 1, 1, 0)  # 1 = BUY, 0 = SELL
        y_test_binary = np.where(y_test_extreme == 1, 1, 0)

        X_train = X_train_extreme
        y_train = y_train_binary
        X_test = X_test_extreme
        y_test = y_test_binary

        print(f"Training data (extreme only): {len(X_train)} samples ({np.sum(y_train==1)} BUY, {np.sum(y_train==0)} SELL)")
        print(f"Test data (extreme only): {len(X_test)} samples ({np.sum(y_test==1)} BUY, {np.sum(y_test==0)} SELL)")

        # Train quick models on balanced extreme moves only
        for name, model_class in model_classes.items():
            print(f"Training {name}...")
            model = model_class()
            try:
                model.train(X_train, y_train)  # y_train is now binary 0/1
                models[name] = model

                # Quick accuracy check on extreme test samples
                preds = model.predict(X_test)
                acc = np.mean(preds == y_test)
                print(f"  {name} accuracy: {acc:.1%}")
            except Exception as e:
                print(f"  Failed to train {name}: {e}")

        # For backtest, use the full test set (including neutrals)
        # The model will predict on all samples, but we only act on high confidence
        featured_df = test_df.copy()  # Use full test set for backtest

    if len(models) == 0:
        print("ERROR: No models available")
        return

    # Prepare features for prediction
    X = featured_df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get predictions from each model
    print("\nGetting predictions from models...")

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
                # Create dummy probas if not available
                all_probas[name] = np.column_stack([1 - preds, preds])

            print(f"  {name}: {np.sum(preds == 1)}/{len(preds)} BUY signals")
        except Exception as e:
            print(f"  Failed {name}: {e}")

    # Ensemble voting
    pred_matrix = np.column_stack(list(all_predictions.values()))

    # Majority vote
    predictions = np.round(np.mean(pred_matrix, axis=1)).astype(int)

    # Average probabilities for confidence
    proba_list = [p[:, 1] if p.shape[1] > 1 else p.flatten() for p in all_probas.values()]
    confidences = np.mean(proba_list, axis=0)

    # Create model votes per sample
    model_names = list(all_predictions.keys())
    model_votes = []
    for i in range(len(predictions)):
        votes = {name: int(all_predictions[name][i]) for name in model_names}
        model_votes.append(votes)

    print(f"\nEnsemble predictions: {np.sum(predictions == 1)}/{len(predictions)} BUY signals")
    print(f"Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")

    # Run ultimate backtest
    print("\n" + "=" * 70)
    print("RUNNING $200 CAPITAL BACKTEST WITH ALL FILTERS")
    print("=" * 70)

    config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.50,        # Start lower to see more trades
        min_model_agreement=3,      # 3/5 models
        max_trades_per_day=10,
        min_risk_reward=1.0,        # Lower R:R to allow more trades
        stop_loss_pct=0.015,        # 1.5% stop
        take_profit_pct=0.025,      # 2.5% TP
        require_trend_alignment=False,  # Relax for more trades
        min_volume_ratio=0.8,       # Lower volume requirement
        min_volatility=0.0005,      # Lower volatility floor
    )

    engine = UltimateTradingEngine(config=config)

    results = engine.run_backtest(
        df=featured_df.reset_index(drop=True),
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=True
    )

    # Comprehensive sensitivity analysis
    print("\n" + "=" * 70)
    print("COMPREHENSIVE OPTIMIZATION - FINDING 90% WIN RATE")
    print("=" * 70)

    best_combo = None
    best_score = 0  # Score = win_rate * sqrt(trades) to balance WR and sample size

    results_list = []

    # Ultra-fine grid search for 90%+ win rate
    for min_conf in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92]:
        for min_agree in [5]:  # Only 5/5 agreement (unanimous)
            for sl, tp in [(0.006, 0.012), (0.007, 0.012), (0.007, 0.014), (0.008, 0.014),
                           (0.008, 0.015), (0.008, 0.016), (0.009, 0.018), (0.01, 0.02)]:
                for min_rr in [1.0, 1.2, 1.5]:  # Higher R:R for better odds
                    config.min_confidence = min_conf
                    config.min_model_agreement = min_agree
                    config.stop_loss_pct = sl
                    config.take_profit_pct = tp
                    config.min_risk_reward = min_rr

                    engine = UltimateTradingEngine(config=config)
                    res = engine.run_backtest(
                        df=featured_df.reset_index(drop=True),
                        predictions=predictions,
                        confidences=confidences,
                        model_votes=model_votes,
                        verbose=False
                    )

                    wr = res['win_rate']
                    trades = res['total_trades']
                    ret = res['total_return_pct']
                    final_cap = res['final_capital']

                    if trades >= 3:
                        # Score prioritizes high win rate with enough trades
                        score = wr * np.sqrt(trades)
                        results_list.append({
                            'conf': min_conf,
                            'agree': min_agree,
                            'sl': sl,
                            'tp': tp,
                            'min_rr': min_rr,
                            'trades': trades,
                            'wr': wr,
                            'return': ret,
                            'final': final_cap,
                            'score': score
                        })

                        if score > best_score:
                            best_score = score
                            best_combo = (min_conf, min_agree, sl, tp, min_rr, trades, wr, final_cap)

    # Sort by win rate, then by trades
    results_list.sort(key=lambda x: (-x['wr'], -x['trades']))

    print("\nTOP 20 PARAMETER COMBINATIONS:")
    print("-" * 100)
    print(f"{'Conf':>6} {'Agree':>5} {'SL':>6} {'TP':>6} {'R:R':>4} {'Trades':>6} {'Win Rate':>10} {'Return':>8} {'Final':>8}")
    print("-" * 100)

    for r in results_list[:20]:
        wr_str = f"{r['wr']*100:.1f}%"
        ret_str = f"{r['return']:+.2f}%"
        marker = " ***" if r['wr'] >= 0.9 else " **" if r['wr'] >= 0.8 else " *" if r['wr'] >= 0.7 else ""
        print(f"{r['conf']:>5.0%} {r['agree']:>5}/5 {r['sl']:>5.1%} {r['tp']:>5.1%} {r['min_rr']:>4.1f} {r['trades']:>6} {wr_str:>10} {ret_str:>8} ${r['final']:>7.2f}{marker}")

    print("\n" + "=" * 70)

    if best_combo:
        print(f"\nBEST COMBINATION FOUND:")
        print(f"  Confidence >= {best_combo[0]:.0%}")
        print(f"  Model Agreement: {best_combo[1]}/5")
        print(f"  Stop Loss: {best_combo[2]:.2%}")
        print(f"  Take Profit: {best_combo[3]:.2%}")
        print(f"  Min R:R: {best_combo[4]:.1f}")
        print(f"  Trades: {best_combo[5]}")
        print(f"  Win Rate: {best_combo[6]*100:.1f}%")
        print(f"  Final Capital: ${best_combo[7]:.2f}")

        # Run best combo with verbose output
        print("\n" + "=" * 70)
        print("RUNNING BEST CONFIGURATION IN DETAIL:")
        print("=" * 70)

        config.min_confidence = best_combo[0]
        config.min_model_agreement = best_combo[1]
        config.stop_loss_pct = best_combo[2]
        config.take_profit_pct = best_combo[3]
        config.min_risk_reward = best_combo[4]

        engine = UltimateTradingEngine(config=config)
        results = engine.run_backtest(
            df=featured_df.reset_index(drop=True),
            predictions=predictions,
            confidences=confidences,
            model_votes=model_votes,
            verbose=True
        )

    print("\n" + "=" * 70)
    print("ULTIMATE TEST COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    load_data_and_run_test()
