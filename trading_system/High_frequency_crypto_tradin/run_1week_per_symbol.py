"""
1-Week Backtest - Per Symbol
============================
Run separate backtests for each symbol to avoid data interleaving issues.

Optimal Configuration:
- Confidence >= 75%
- Model Agreement: 5/5 (unanimous)
- Stop Loss: 0.7%
- Take Profit: 1.4%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ultimate_trading_system import UltimateRiskConfig, UltimateTradingEngine


def run_symbol_backtest(symbol: str, df: pd.DataFrame, feature_columns: list, models: dict):
    """Run backtest for a single symbol."""

    # Prepare features
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get predictions
    all_predictions = {}
    all_probas = {}

    for name, model in models.items():
        preds = model.predict(X)
        all_predictions[name] = preds
        try:
            probas = model.predict_proba(X)
            all_probas[name] = probas
        except:
            all_probas[name] = np.column_stack([1 - preds, preds])

    # Ensemble voting
    pred_matrix = np.column_stack(list(all_predictions.values()))
    predictions = np.round(np.mean(pred_matrix, axis=1)).astype(int)

    # Average confidence
    proba_list = [p[:, 1] if p.shape[1] > 1 else p.flatten() for p in all_probas.values()]
    confidences = np.mean(proba_list, axis=0)

    # Model votes
    model_names = list(all_predictions.keys())
    model_votes = []
    for i in range(len(predictions)):
        votes = {name: int(all_predictions[name][i]) for name in model_names}
        model_votes.append(votes)

    # Configuration - optimized for more trades
    config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.65,         # Lower threshold
        min_model_agreement=4,       # 4/5 agreement
        stop_loss_pct=0.008,         # 0.8% SL
        take_profit_pct=0.015,       # 1.5% TP
        min_risk_reward=0,
        max_trades_per_day=15,
        require_trend_alignment=False,
        min_volume_ratio=0.5,
        min_volatility=0.0001,
        cooldown_minutes=5,          # Faster
    )

    engine = UltimateTradingEngine(config=config)

    results = engine.run_backtest(
        df=df.reset_index(drop=True),
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=False
    )

    return results, predictions, confidences


def main():
    print("\n" + "=" * 80)
    print("1-WEEK BACKTEST - PER SYMBOL ANALYSIS")
    print("=" * 80)

    data_path = Path(__file__).parent / "Crypto_Data_5m"
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']
    bars_per_week = 7 * 24 * 12  # 2016 bars

    from features.feature_engineer import FeatureEngineer
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

    all_results = {}
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")

        filepath = data_path / f"{symbol}_5m.csv"
        if not filepath.exists():
            print(f"  FILE NOT FOUND")
            continue

        # Load data
        df = pd.read_csv(filepath)
        df = df.tail(bars_per_week).reset_index(drop=True)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        print(f"  Data: {len(df)} bars")
        print(f"  Period: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

        # Engineer features
        fe = FeatureEngineer()
        featured_df = fe.compute_all_features(df)
        featured_df = fe.create_target(featured_df, use_asymmetric_targets=True, extreme_pct=15.0)

        feature_columns = fe.get_feature_columns(featured_df)
        exclude = ['symbol', 'is_extreme_move', 'open_time', 'timestamp']
        feature_columns = [c for c in feature_columns if c not in exclude]
        numeric_cols = featured_df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = numeric_cols

        # Split 80/20
        train_size = int(len(featured_df) * 0.8)
        train_df = featured_df.iloc[:train_size].copy()
        test_df = featured_df.iloc[train_size:].copy()

        # Remove NaN
        train_df = train_df.dropna(subset=['target'] + feature_columns)
        test_df = test_df.dropna(subset=['target'] + feature_columns)

        # Training data
        X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df['target'].values

        extreme_mask = y_train != 0
        X_train_extreme = X_train[extreme_mask]
        y_train_extreme = y_train[extreme_mask]
        y_train_binary = np.where(y_train_extreme == 1, 1, 0)

        print(f"  Training: {len(X_train_extreme)} samples (BUY: {np.sum(y_train_binary==1)}, SELL: {np.sum(y_train_binary==0)})")
        print(f"  Test: {len(test_df)} bars")

        # Train models
        models = {}
        for name, model_class in model_classes.items():
            model = model_class()
            try:
                model.train(X_train_extreme, y_train_binary)
                models[name] = model
            except Exception as e:
                print(f"  {name} failed: {e}")

        if len(models) == 0:
            print("  No models trained!")
            continue

        # Run backtest
        results, predictions, confidences = run_symbol_backtest(
            symbol, test_df, feature_columns, models
        )

        all_results[symbol] = results

        trades = results['total_trades']
        wins = results['wins']
        losses = results['losses']
        wr = results['win_rate'] * 100
        pnl = results['total_return']
        final = results['final_capital']

        total_trades += trades
        total_wins += wins
        total_losses += losses
        total_pnl += pnl

        print(f"\n  RESULTS:")
        print(f"    Trades: {trades} | Wins: {wins} | Losses: {losses}")
        print(f"    Win Rate: {wr:.1f}%")
        print(f"    P&L: ${pnl:.2f} | Final: ${final:.2f}")
        print(f"    Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
        print(f"    BUY signals: {np.sum(predictions==1)}/{len(predictions)}")

    # Summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ALL SYMBOLS")
    print("=" * 80)

    print(f"\n{'Symbol':<12} {'Trades':>8} {'Wins':>6} {'Losses':>6} {'Win Rate':>10} {'P&L':>10} {'Final':>10}")
    print("-" * 70)

    for symbol, res in all_results.items():
        wr = res['win_rate'] * 100
        pnl = res['total_return']
        print(f"{symbol:<12} {res['total_trades']:>8} {res['wins']:>6} {res['losses']:>6} {wr:>9.1f}% ${pnl:>9.2f} ${res['final_capital']:>9.2f}")

    print("-" * 70)

    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"{'TOTAL':<12} {total_trades:>8} {total_wins:>6} {total_losses:>6} {overall_wr:>9.1f}% ${total_pnl:>9.2f}")

    # Combined capital (if trading all 5 with $200 each = $1000 total)
    combined_final = sum(r['final_capital'] for r in all_results.values())
    combined_initial = 200.0 * len(all_results)
    combined_return = (combined_final - combined_initial) / combined_initial * 100

    print(f"\n  Combined Capital: ${combined_initial:.2f} -> ${combined_final:.2f} ({combined_return:+.2f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
