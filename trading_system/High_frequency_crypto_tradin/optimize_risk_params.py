"""
RISK MANAGEMENT PARAMETER OPTIMIZATION
=======================================
Comprehensive grid search to find optimal risk parameters.

Tests various combinations of:
- Stop Loss %
- Take Profit %
- Risk:Reward ratio
- Model Agreement levels
- Confidence thresholds
- Cooldown periods
- Volume filters

Goal: Find configuration that maximizes win rate while maintaining profitability.
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

    # Load feature config
    with open(model_path / "feature_config.json", 'r') as f:
        config = json.load(f)
    feature_columns = config['feature_columns']

    # Import model classes
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
            print(f"  Failed to load {name}: {e}")

    return models, feature_columns


def load_test_data():
    """Load 1-week test data for all 5 symbols."""

    data_path = Path(__file__).parent / "Crypto_Data_5m"
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']
    bars_per_week = 7 * 24 * 12  # 2016 bars

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

    return pd.concat(all_data, ignore_index=True)


def generate_predictions(models, featured_df, feature_columns):
    """Generate ensemble predictions."""

    X = featured_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

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
                all_probas[name] = np.column_stack([1 - preds, preds])
        except:
            pass

    # Ensemble voting
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

    # Model votes for binary BUY signals
    model_votes = []
    for i in range(n_samples):
        votes = {name: (1 if all_predictions[name][i] == 1 else 0) for name in model_names}
        model_votes.append(votes)

    # Convert to binary
    binary_predictions = (predictions == 1).astype(int)

    return binary_predictions, confidences, model_votes


def run_optimization():
    """Run comprehensive parameter optimization."""

    print("\n" + "=" * 100)
    print("RISK MANAGEMENT PARAMETER OPTIMIZATION")
    print("=" * 100)
    print(f"Started: {datetime.now()}")

    # Load models
    print("\nLoading pre-trained models...")
    models, feature_columns = load_pretrained_models()
    print(f"  Loaded {len(models)}/5 models")

    # Load data
    print("\nLoading 1-week test data...")
    df = load_test_data()
    print(f"  Total bars: {len(df):,}")

    # Engineer features
    print("\nEngineering features...")
    from features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    featured_df = fe.compute_all_features(df)

    # Filter to available features
    available_features = [c for c in feature_columns if c in featured_df.columns]
    print(f"  Features: {len(available_features)}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions, confidences, model_votes = generate_predictions(
        models, featured_df, available_features
    )

    buy_pct = 100 * np.sum(predictions == 1) / len(predictions)
    print(f"  BUY signals: {np.sum(predictions == 1)} ({buy_pct:.1f}%)")
    print(f"  Confidence range: {confidences.min():.1%} - {confidences.max():.1%}")

    # Parameter grid
    print("\n" + "=" * 100)
    print("PARAMETER GRID")
    print("=" * 100)

    # Risk management parameters to test
    stop_loss_pcts = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.015]
    take_profit_pcts = [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030]
    min_risk_rewards = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    min_agreements = [3, 4, 5]
    cooldown_minutes = [5, 10, 15, 20]
    volume_ratios = [0.3, 0.5, 0.7]

    # Calculate total combinations
    total_combinations = (len(stop_loss_pcts) * len(take_profit_pcts) *
                         len(min_risk_rewards) * len(min_agreements) *
                         len(cooldown_minutes) * len(volume_ratios))

    print(f"\n  Stop Loss %: {[f'{x:.1%}' for x in stop_loss_pcts]}")
    print(f"  Take Profit %: {[f'{x:.1%}' for x in take_profit_pcts]}")
    print(f"  Min R:R: {min_risk_rewards}")
    print(f"  Min Agreement: {min_agreements}")
    print(f"  Cooldown (min): {cooldown_minutes}")
    print(f"  Volume Ratio: {volume_ratios}")
    print(f"\n  Total combinations: {total_combinations:,}")

    # Run optimization
    print("\n" + "=" * 100)
    print("RUNNING OPTIMIZATION...")
    print("=" * 100)

    results_list = []
    tested = 0

    # Base config
    base_config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.60,
        require_trend_alignment=False,
        min_volatility=0.0001,
        max_trades_per_day=50,  # Allow many trades
    )

    for sl, tp, rr, agree, cooldown, vol_ratio in product(
        stop_loss_pcts, take_profit_pcts, min_risk_rewards,
        min_agreements, cooldown_minutes, volume_ratios
    ):
        # Skip invalid combinations (TP must be > SL for long trades)
        if tp <= sl:
            continue

        # Update config
        base_config.stop_loss_pct = sl
        base_config.take_profit_pct = tp
        base_config.min_risk_reward = rr
        base_config.min_model_agreement = agree
        base_config.cooldown_minutes = cooldown
        base_config.min_volume_ratio = vol_ratio

        # Run backtest
        engine = UltimateTradingEngine(config=base_config)
        res = engine.run_backtest(
            df=featured_df.reset_index(drop=True),
            predictions=predictions,
            confidences=confidences,
            model_votes=model_votes,
            verbose=False
        )

        trades = res['total_trades']

        if trades >= 3:  # Minimum sample size
            results_list.append({
                'sl': sl,
                'tp': tp,
                'rr': rr,
                'agree': agree,
                'cooldown': cooldown,
                'vol_ratio': vol_ratio,
                'trades': trades,
                'wins': res['wins'],
                'losses': res['losses'],
                'win_rate': res['win_rate'],
                'pnl': res['total_return'],
                'pnl_pct': res['total_return_pct'],
                'final': res['final_capital'],
                'profit_factor': res['profit_factor'],
                'sharpe': res['sharpe_ratio'],
                'max_dd': res['max_drawdown_pct'],
                'avg_win': res['avg_win'],
                'avg_loss': res['avg_loss'],
            })

        tested += 1
        if tested % 500 == 0:
            print(f"  Tested {tested:,}/{total_combinations:,} combinations...")

    print(f"\n  Completed: {tested:,} combinations tested")
    print(f"  Valid results: {len(results_list):,} (with >= 3 trades)")

    if not results_list:
        print("No valid results found!")
        return

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results_list)

    # Sort by different criteria
    print("\n" + "=" * 100)
    print("TOP CONFIGURATIONS BY WIN RATE")
    print("=" * 100)

    by_wr = results_df.sort_values(['win_rate', 'trades'], ascending=[False, False]).head(30)

    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Cool':>5} {'Vol':>5} {'Trades':>7} {'WR':>7} {'W':>4} {'L':>4} {'P&L':>8} {'PF':>6} {'Sharpe':>7}")
    print("-" * 100)

    for _, r in by_wr.iterrows():
        wr_str = f"{r['win_rate']*100:.1f}%"
        marker = " ***" if r['win_rate'] >= 0.70 else " **" if r['win_rate'] >= 0.60 else " *" if r['win_rate'] >= 0.55 else ""
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['cooldown']:>5} {r['vol_ratio']:>5.1f} {r['trades']:>7} {wr_str:>7} {r['wins']:>4} {r['losses']:>4} ${r['pnl']:>7.2f} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f}{marker}")

    # Best by profit factor
    print("\n" + "=" * 100)
    print("TOP CONFIGURATIONS BY PROFIT FACTOR")
    print("=" * 100)

    by_pf = results_df[results_df['trades'] >= 5].sort_values(['profit_factor', 'trades'], ascending=[False, False]).head(20)

    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Cool':>5} {'Vol':>5} {'Trades':>7} {'WR':>7} {'P&L':>8} {'PF':>6} {'Sharpe':>7}")
    print("-" * 100)

    for _, r in by_pf.iterrows():
        wr_str = f"{r['win_rate']*100:.1f}%"
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['cooldown']:>5} {r['vol_ratio']:>5.1f} {r['trades']:>7} {wr_str:>7} ${r['pnl']:>7.2f} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f}")

    # Best by Sharpe ratio
    print("\n" + "=" * 100)
    print("TOP CONFIGURATIONS BY SHARPE RATIO")
    print("=" * 100)

    by_sharpe = results_df[results_df['trades'] >= 5].sort_values(['sharpe', 'trades'], ascending=[False, False]).head(20)

    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Cool':>5} {'Vol':>5} {'Trades':>7} {'WR':>7} {'P&L':>8} {'PF':>6} {'Sharpe':>7}")
    print("-" * 100)

    for _, r in by_sharpe.iterrows():
        wr_str = f"{r['win_rate']*100:.1f}%"
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['cooldown']:>5} {r['vol_ratio']:>5.1f} {r['trades']:>7} {wr_str:>7} ${r['pnl']:>7.2f} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f}")

    # Best by total P&L
    print("\n" + "=" * 100)
    print("TOP CONFIGURATIONS BY TOTAL P&L")
    print("=" * 100)

    by_pnl = results_df.sort_values(['pnl', 'win_rate'], ascending=[False, False]).head(20)

    print(f"\n{'SL':>6} {'TP':>6} {'R:R':>5} {'Agr':>4} {'Cool':>5} {'Vol':>5} {'Trades':>7} {'WR':>7} {'P&L':>8} {'PF':>6} {'Final':>10}")
    print("-" * 100)

    for _, r in by_pnl.iterrows():
        wr_str = f"{r['win_rate']*100:.1f}%"
        print(f"{r['sl']:>5.1%} {r['tp']:>5.1%} {r['rr']:>5.1f} {r['agree']:>4}/5 {r['cooldown']:>5} {r['vol_ratio']:>5.1f} {r['trades']:>7} {wr_str:>7} ${r['pnl']:>7.2f} {r['profit_factor']:>6.2f} ${r['final']:>9.2f}")

    # Analysis by agreement level
    print("\n" + "=" * 100)
    print("ANALYSIS BY MODEL AGREEMENT LEVEL")
    print("=" * 100)

    for agree in [3, 4, 5]:
        subset = results_df[results_df['agree'] == agree]
        if len(subset) > 0:
            print(f"\n  {agree}/5 Agreement:")
            print(f"    Configs tested: {len(subset)}")
            print(f"    Avg Win Rate: {subset['win_rate'].mean()*100:.1f}%")
            print(f"    Max Win Rate: {subset['win_rate'].max()*100:.1f}%")
            print(f"    Avg Trades: {subset['trades'].mean():.1f}")
            print(f"    Avg P&L: ${subset['pnl'].mean():.2f}")
            print(f"    Best P&L: ${subset['pnl'].max():.2f}")

    # Analysis by SL/TP ratio
    print("\n" + "=" * 100)
    print("ANALYSIS BY SL/TP RATIO")
    print("=" * 100)

    results_df['sl_tp_ratio'] = results_df['tp'] / results_df['sl']

    for ratio_range in [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)]:
        subset = results_df[(results_df['sl_tp_ratio'] >= ratio_range[0]) &
                           (results_df['sl_tp_ratio'] < ratio_range[1])]
        if len(subset) > 0:
            print(f"\n  TP/SL Ratio {ratio_range[0]:.1f}-{ratio_range[1]:.1f}x:")
            print(f"    Configs: {len(subset)}")
            print(f"    Avg Win Rate: {subset['win_rate'].mean()*100:.1f}%")
            print(f"    Avg Trades: {subset['trades'].mean():.1f}")
            print(f"    Avg P&L: ${subset['pnl'].mean():.2f}")

    # Find balanced best config
    print("\n" + "=" * 100)
    print("BALANCED BEST CONFIGURATION")
    print("=" * 100)

    # Score: win_rate * log(trades) * (1 + pnl/10)
    results_df['score'] = (results_df['win_rate'] *
                          np.log1p(results_df['trades']) *
                          (1 + results_df['pnl'].clip(lower=0) / 10))

    best = results_df.sort_values('score', ascending=False).head(1).iloc[0]

    print(f"\n  BEST BALANCED CONFIGURATION:")
    print(f"  Stop Loss: {best['sl']:.2%}")
    print(f"  Take Profit: {best['tp']:.2%}")
    print(f"  Min R:R: {best['rr']:.1f}")
    print(f"  Model Agreement: {best['agree']}/5")
    print(f"  Cooldown: {best['cooldown']} min")
    print(f"  Volume Ratio: {best['vol_ratio']:.1f}")
    print(f"  ---")
    print(f"  Trades: {best['trades']}")
    print(f"  Win Rate: {best['win_rate']*100:.1f}%")
    print(f"  P&L: ${best['pnl']:.2f}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Final Capital: ${best['final']:.2f}")

    # Run detailed backtest with best config
    print("\n" + "=" * 100)
    print("DETAILED BACKTEST WITH BEST CONFIG")
    print("=" * 100)

    best_config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.60,
        min_model_agreement=int(best['agree']),
        stop_loss_pct=best['sl'],
        take_profit_pct=best['tp'],
        min_risk_reward=best['rr'],
        cooldown_minutes=int(best['cooldown']),
        min_volume_ratio=best['vol_ratio'],
        max_trades_per_day=50,
        require_trend_alignment=False,
        min_volatility=0.0001,
    )

    engine = UltimateTradingEngine(config=best_config)
    final_results = engine.run_backtest(
        df=featured_df.reset_index(drop=True),
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=True
    )

    # Save results to file
    results_df.to_csv(Path(__file__).parent / "optimization_results.csv", index=False)
    print(f"\nResults saved to optimization_results.csv")

    print("\n" + "=" * 100)
    print("OPTIMIZATION COMPLETE")
    print("=" * 100)
    print(f"Finished: {datetime.now()}")

    return results_df


if __name__ == "__main__":
    run_optimization()
