"""
FAST RISK MANAGEMENT PARAMETER OPTIMIZATION
============================================
Focused grid search on key parameters.
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
from itertools import product
from datetime import datetime

from ultimate_trading_system import UltimateRiskConfig, UltimateTradingEngine


def load_pretrained_models():
    """Load the pre-trained ensemble models."""
    model_path = Path(__file__).parent / "saved_models_improved"
    ensemble_path = model_path / "ensemble"

    with open(model_path / "feature_config.json", 'r') as f:
        config = json.load(f)
    feature_columns = config['feature_columns']

    from ml_models.random_forest_model import RandomForestModel
    from ml_models.xgboost_model import XGBoostModel
    from ml_models.lightgbm_model import LightGBMModel
    from ml_models.catboost_model import CatBoostModel
    from ml_models.neural_network_model import NeuralNetworkModel

    models = {}
    model_files = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'neural_network': NeuralNetworkModel,
    }

    for name, model_class in model_files.items():
        try:
            model = model_class()
            model.load(str(ensemble_path / name))
            models[name] = model
        except Exception as e:
            pass

    return models, feature_columns


def run_fast_optimization():
    """Run focused parameter optimization."""

    print("\n" + "=" * 100)
    print("FAST RISK MANAGEMENT PARAMETER OPTIMIZATION")
    print("=" * 100)

    # Load models
    print("\nLoading models...")
    models, feature_columns = load_pretrained_models()
    print(f"  Loaded {len(models)}/5 models")

    # Load data
    print("\nLoading 1-week test data...")
    data_path = Path(__file__).parent / "Crypto_Data_5m"
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']
    bars_per_week = 7 * 24 * 12

    all_data = []
    for symbol in symbols:
        filepath = data_path / f"{symbol}_5m.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df = df.tail(bars_per_week).reset_index(drop=True)
            df['symbol'] = symbol
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    print(f"  Total bars: {len(df):,}")

    # Engineer features
    print("\nEngineering features...")
    from features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    featured_df = fe.compute_all_features(df)

    available_features = [c for c in feature_columns if c in featured_df.columns]
    X = featured_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Generate predictions
    print("\nGenerating predictions...")
    all_predictions = {}
    for name, model in models.items():
        try:
            preds = model.predict(X)
            all_predictions[name] = preds
        except:
            pass

    n_samples = len(X)
    model_names = list(all_predictions.keys())

    predictions = np.zeros(n_samples, dtype=int)
    confidences = np.zeros(n_samples)

    for i in range(n_samples):
        votes = {name: int(all_predictions[name][i]) for name in model_names}
        vote_counts = {-1: 0, 0: 0, 1: 0}
        for v in votes.values():
            vote_counts[v] = vote_counts.get(v, 0) + 1
        max_class = max(vote_counts, key=vote_counts.get)
        predictions[i] = max_class
        confidences[i] = vote_counts[max_class] / len(model_names)

    model_votes = []
    for i in range(n_samples):
        votes = {name: (1 if all_predictions[name][i] == 1 else 0) for name in model_names}
        model_votes.append(votes)

    binary_predictions = (predictions == 1).astype(int)
    predictions = binary_predictions

    print(f"  BUY signals: {np.sum(predictions == 1)} ({100*np.sum(predictions==1)/len(predictions):.1f}%)")

    # Focused parameter grid
    print("\n" + "=" * 100)
    print("RUNNING OPTIMIZATION...")
    print("=" * 100)

    # Key parameters to test
    stop_losses = [0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015]
    take_profits = [0.008, 0.010, 0.012, 0.014, 0.016, 0.020, 0.025, 0.030]
    risk_rewards = [0, 0.8, 1.0, 1.2, 1.5, 2.0]
    agreements = [3, 4, 5]

    results = []

    total = len(stop_losses) * len(take_profits) * len(risk_rewards) * len(agreements)
    print(f"\n  Testing {total} combinations...")

    count = 0
    for sl in stop_losses:
        for tp in take_profits:
            if tp <= sl:
                continue
            for rr in risk_rewards:
                for agree in agreements:
                    config = UltimateRiskConfig(
                        initial_capital=200.0,
                        min_confidence=0.60,
                        min_model_agreement=agree,
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        min_risk_reward=rr,
                        cooldown_minutes=10,
                        min_volume_ratio=0.5,
                        max_trades_per_day=50,
                        require_trend_alignment=False,
                        min_volatility=0.0001,
                    )

                    engine = UltimateTradingEngine(config=config)
                    res = engine.run_backtest(
                        df=featured_df.reset_index(drop=True),
                        predictions=predictions,
                        confidences=confidences,
                        model_votes=model_votes,
                        verbose=False
                    )

                    if res['total_trades'] >= 3:
                        results.append({
                            'sl': sl, 'tp': tp, 'rr': rr, 'agree': agree,
                            'trades': res['total_trades'],
                            'wins': res['wins'],
                            'losses': res['losses'],
                            'wr': res['win_rate'],
                            'pnl': res['total_return'],
                            'pf': res['profit_factor'],
                            'sharpe': res['sharpe_ratio'],
                            'final': res['final_capital'],
                        })

                    count += 1

    print(f"  Tested {count} combinations, {len(results)} with trades")

    # Sort and display results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("TOP 30 BY WIN RATE")
    print("=" * 100)
    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Trades':>7} {'WR':>8} {'W':>4} {'L':>4} {'P&L':>9} {'PF':>6} {'Final':>10}")
    print("-" * 100)

    top_wr = results_df.sort_values(['wr', 'trades'], ascending=[False, False]).head(30)
    for _, r in top_wr.iterrows():
        marker = " ***" if r['wr'] >= 0.70 else " **" if r['wr'] >= 0.60 else " *" if r['wr'] >= 0.55 else ""
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['trades']:>7} {r['wr']*100:>7.1f}% {r['wins']:>4} {r['losses']:>4} ${r['pnl']:>8.2f} {r['pf']:>6.2f} ${r['final']:>9.2f}{marker}")

    print("\n" + "=" * 100)
    print("TOP 20 BY P&L")
    print("=" * 100)
    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Trades':>7} {'WR':>8} {'P&L':>9} {'PF':>6} {'Final':>10}")
    print("-" * 100)

    top_pnl = results_df.sort_values(['pnl', 'wr'], ascending=[False, False]).head(20)
    for _, r in top_pnl.iterrows():
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['trades']:>7} {r['wr']*100:>7.1f}% ${r['pnl']:>8.2f} {r['pf']:>6.2f} ${r['final']:>9.2f}")

    print("\n" + "=" * 100)
    print("TOP 20 BY PROFIT FACTOR (min 5 trades)")
    print("=" * 100)
    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Trades':>7} {'WR':>8} {'P&L':>9} {'PF':>6} {'Sharpe':>8}")
    print("-" * 100)

    top_pf = results_df[results_df['trades'] >= 5].sort_values(['pf', 'trades'], ascending=[False, False]).head(20)
    for _, r in top_pf.iterrows():
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['trades']:>7} {r['wr']*100:>7.1f}% ${r['pnl']:>8.2f} {r['pf']:>6.2f} {r['sharpe']:>8.2f}")

    # Analysis by agreement
    print("\n" + "=" * 100)
    print("ANALYSIS BY AGREEMENT LEVEL")
    print("=" * 100)

    for agree in [3, 4, 5]:
        subset = results_df[results_df['agree'] == agree]
        if len(subset) > 0:
            print(f"\n  {agree}/5 Agreement:")
            print(f"    Configs: {len(subset)}")
            print(f"    Avg WR: {subset['wr'].mean()*100:.1f}% | Max: {subset['wr'].max()*100:.1f}%")
            print(f"    Avg Trades: {subset['trades'].mean():.1f}")
            print(f"    Avg P&L: ${subset['pnl'].mean():.2f} | Best: ${subset['pnl'].max():.2f}")

    # Analysis by R:R requirement
    print("\n" + "=" * 100)
    print("ANALYSIS BY RISK:REWARD REQUIREMENT")
    print("=" * 100)

    for rr in [0, 0.8, 1.0, 1.2, 1.5, 2.0]:
        subset = results_df[results_df['rr'] == rr]
        if len(subset) > 0:
            print(f"\n  R:R >= {rr}:")
            print(f"    Configs: {len(subset)}")
            print(f"    Avg WR: {subset['wr'].mean()*100:.1f}% | Max: {subset['wr'].max()*100:.1f}%")
            print(f"    Avg Trades: {subset['trades'].mean():.1f}")
            print(f"    Avg P&L: ${subset['pnl'].mean():.2f} | Best: ${subset['pnl'].max():.2f}")

    # Find best balanced config
    print("\n" + "=" * 100)
    print("BEST BALANCED CONFIGURATION")
    print("=" * 100)

    # Score balancing WR, trades, and P&L
    results_df['score'] = results_df['wr'] * np.log1p(results_df['trades']) * (1 + results_df['pnl'].clip(lower=0) / 5)
    best = results_df.sort_values('score', ascending=False).iloc[0]

    print(f"\n  Stop Loss: {best['sl']:.2%}")
    print(f"  Take Profit: {best['tp']:.2%}")
    print(f"  Min R:R: {best['rr']:.1f}")
    print(f"  Agreement: {best['agree']}/5")
    print(f"  ---")
    print(f"  Trades: {best['trades']}")
    print(f"  Win Rate: {best['wr']*100:.1f}%")
    print(f"  P&L: ${best['pnl']:.2f}")
    print(f"  Profit Factor: {best['pf']:.2f}")
    print(f"  Final: ${best['final']:.2f}")

    # Run detailed backtest
    print("\n" + "=" * 100)
    print("DETAILED BACKTEST WITH BEST CONFIG")
    print("=" * 100)

    config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.60,
        min_model_agreement=int(best['agree']),
        stop_loss_pct=best['sl'],
        take_profit_pct=best['tp'],
        min_risk_reward=best['rr'],
        cooldown_minutes=10,
        min_volume_ratio=0.5,
        max_trades_per_day=50,
        require_trend_alignment=False,
        min_volatility=0.0001,
    )

    engine = UltimateTradingEngine(config=config)
    engine.run_backtest(
        df=featured_df.reset_index(drop=True),
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=True
    )

    # Save results
    results_df.to_csv(Path(__file__).parent / "optimization_results.csv", index=False)
    print(f"\nResults saved to optimization_results.csv")

    print("\n" + "=" * 100)
    print("OPTIMIZATION COMPLETE")
    print("=" * 100)

    return results_df


if __name__ == "__main__":
    run_fast_optimization()
