"""
FAST PARAMETER OPTIMIZATION FOR DYNAMIC DCA STRATEGY
=====================================================

Optimizes DCA parameters using:
1. Parallel processing (multiprocessing)
2. Smart grid search with adaptive refinement
3. Single symbol fast evaluation (BTCUSD only for speed)
4. Early stopping for bad configurations
5. Cached feature computation

Parameters to optimize:
- Take Profit %
- DCA Trigger levels (4 levels)
- DCA Multipliers (4 levels)
- DCA Profit Target %
- SL after last DCA %
- Trailing Stop %

Run: python -m trading_system.High_frequency_crypto_tradin.optimize_dca_params
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.backtest import BacktestEngine, BacktestConfig
from trading_system.High_frequency_crypto_tradin.config import load_config


@dataclass
class OptimizationResult:
    """Store optimization result."""
    params: Dict
    total_pnl: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_win: float
    avg_loss: float
    sharpe: float
    max_drawdown: float


# Global cache for data (shared across processes)
_cached_data = {}
_cached_signals = {}


def load_data_once(symbol: str = "BTCUSD"):
    """Load data and compute features once, cache for reuse."""
    global _cached_data, _cached_signals

    if symbol in _cached_data:
        return _cached_data[symbol], _cached_signals[symbol]

    config = load_config()
    model_dir = Path(config.model_save_dir)
    data_dir = Path(config.data_dir)

    # Load model
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    # Load data
    data_file = data_dir / f"{symbol}_1m.csv"
    df = fe.load_data(str(data_file))

    # Compute features
    df_features = fe.compute_all_features(df)
    df_features = fe.handle_missing_values(df_features)

    # Generate signals
    X = df_features[fe.feature_columns]
    predictions = ensemble.predict(X)
    probas = ensemble.predict_proba(X)
    confidences = np.max(probas, axis=1)

    _cached_data[symbol] = df
    _cached_signals[symbol] = (predictions, confidences)

    return df, (predictions, confidences)


def run_single_backtest(params: Dict, df: pd.DataFrame,
                        signals: np.ndarray, confidences: np.ndarray,
                        symbol: str = "BTCUSD") -> OptimizationResult:
    """Run a single backtest with given parameters."""

    # Create backtest config
    bt_config = BacktestConfig(
        initial_capital=100000.0,
        position_size_pct=0.02,
        max_position_size=10000.0,
        commission_rate=0.001,
        slippage_pct=0.0005,
        # Parameters to optimize
        take_profit_pct=params['take_profit_pct'],
        trailing_stop_pct=params['trailing_stop_pct'],
        dca_enabled=True,
        max_dca_entries=4,
        dca_level_1_trigger_pct=params['dca_level_1_trigger_pct'],
        dca_level_1_multiplier=params['dca_level_1_multiplier'],
        dca_level_2_trigger_pct=params['dca_level_2_trigger_pct'],
        dca_level_2_multiplier=params['dca_level_2_multiplier'],
        dca_level_3_trigger_pct=params['dca_level_3_trigger_pct'],
        dca_level_3_multiplier=params['dca_level_3_multiplier'],
        dca_level_4_trigger_pct=params['dca_level_4_trigger_pct'],
        dca_level_4_multiplier=params['dca_level_4_multiplier'],
        dca_profit_target_pct=params['dca_profit_target_pct'],
        sl_after_last_dca_pct=params['sl_after_last_dca_pct'],
        stop_loss_pct=0.05,
        min_confidence=0.6,
        cooldown_bars=3,
        max_daily_trades=50
    )

    # Run backtest
    engine = BacktestEngine(bt_config, use_centralized_risk=False)
    results = engine.run(df, signals, confidences, symbol=symbol, verbose=False)

    # Get trade stats
    trades_df = engine.get_trades_df()
    if trades_df.empty:
        return OptimizationResult(
            params=params,
            total_pnl=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            avg_win=0,
            avg_loss=0,
            sharpe=0,
            max_drawdown=0
        )

    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] <= 0]

    return OptimizationResult(
        params=params,
        total_pnl=results.get('total_pnl', 0),
        win_rate=results.get('win_rate', 0),
        profit_factor=results.get('profit_factor', 0),
        total_trades=results.get('total_trades', 0),
        avg_win=winning['pnl'].mean() if len(winning) > 0 else 0,
        avg_loss=abs(losing['pnl'].mean()) if len(losing) > 0 else 0,
        sharpe=results.get('sharpe_ratio', 0),
        max_drawdown=results.get('max_drawdown', 0)
    )


def worker_backtest(args):
    """Worker function for parallel execution."""
    params, df_dict, signals, confidences, symbol = args

    # Reconstruct DataFrame
    df = pd.DataFrame(df_dict)

    try:
        result = run_single_backtest(params, df, signals, confidences, symbol)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_param_grid() -> List[Dict]:
    """Generate parameter combinations for grid search."""

    # Define search space - FOCUSED on key parameters
    param_space = {
        # Take Profit (wider range)
        'take_profit_pct': [0.015, 0.02, 0.025, 0.03, 0.035],

        # DCA Level 1 - AGGRESSIVE
        'dca_level_1_trigger_pct': [0.008, 0.01, 0.012],
        'dca_level_1_multiplier': [1.5, 2.0, 2.5],

        # DCA Level 2 - AGGRESSIVE
        'dca_level_2_trigger_pct': [0.012, 0.015, 0.018],
        'dca_level_2_multiplier': [1.5, 2.0, 2.5],

        # DCA Level 3 - MODERATE
        'dca_level_3_trigger_pct': [0.015, 0.02, 0.025],
        'dca_level_3_multiplier': [1.0, 1.5, 2.0],

        # DCA Level 4 - CONSERVATIVE
        'dca_level_4_trigger_pct': [0.02, 0.025, 0.03],
        'dca_level_4_multiplier': [0.5, 1.0, 1.5],

        # DCA Profit Target (CRITICAL)
        'dca_profit_target_pct': [0.01, 0.015, 0.02, 0.025, 0.03],

        # SL after last DCA
        'sl_after_last_dca_pct': [0.01, 0.015, 0.02, 0.025],

        # Trailing Stop
        'trailing_stop_pct': [0.008, 0.01, 0.012, 0.015],
    }

    # Smart sampling - don't do full grid (would be millions)
    # Instead, use Latin Hypercube Sampling or random sampling
    n_samples = 500  # Number of random combinations to test

    param_combinations = []
    np.random.seed(42)  # Reproducibility

    for _ in range(n_samples):
        params = {
            key: np.random.choice(values)
            for key, values in param_space.items()
        }
        param_combinations.append(params)

    # Also add some "smart" combinations based on intuition
    smart_combinations = [
        # Aggressive DCA, tight TP
        {
            'take_profit_pct': 0.02,
            'dca_level_1_trigger_pct': 0.008,
            'dca_level_1_multiplier': 2.5,
            'dca_level_2_trigger_pct': 0.012,
            'dca_level_2_multiplier': 2.0,
            'dca_level_3_trigger_pct': 0.015,
            'dca_level_3_multiplier': 1.5,
            'dca_level_4_trigger_pct': 0.02,
            'dca_level_4_multiplier': 1.0,
            'dca_profit_target_pct': 0.015,
            'sl_after_last_dca_pct': 0.015,
            'trailing_stop_pct': 0.01,
        },
        # Conservative DCA, wider TP
        {
            'take_profit_pct': 0.03,
            'dca_level_1_trigger_pct': 0.012,
            'dca_level_1_multiplier': 1.5,
            'dca_level_2_trigger_pct': 0.018,
            'dca_level_2_multiplier': 1.5,
            'dca_level_3_trigger_pct': 0.025,
            'dca_level_3_multiplier': 1.0,
            'dca_level_4_trigger_pct': 0.03,
            'dca_level_4_multiplier': 0.5,
            'dca_profit_target_pct': 0.025,
            'sl_after_last_dca_pct': 0.02,
            'trailing_stop_pct': 0.012,
        },
        # Balanced
        {
            'take_profit_pct': 0.025,
            'dca_level_1_trigger_pct': 0.01,
            'dca_level_1_multiplier': 2.0,
            'dca_level_2_trigger_pct': 0.015,
            'dca_level_2_multiplier': 2.0,
            'dca_level_3_trigger_pct': 0.02,
            'dca_level_3_multiplier': 1.5,
            'dca_level_4_trigger_pct': 0.025,
            'dca_level_4_multiplier': 1.0,
            'dca_profit_target_pct': 0.02,
            'sl_after_last_dca_pct': 0.015,
            'trailing_stop_pct': 0.01,
        },
        # Quick exits
        {
            'take_profit_pct': 0.015,
            'dca_level_1_trigger_pct': 0.008,
            'dca_level_1_multiplier': 2.0,
            'dca_level_2_trigger_pct': 0.012,
            'dca_level_2_multiplier': 2.0,
            'dca_level_3_trigger_pct': 0.015,
            'dca_level_3_multiplier': 1.5,
            'dca_level_4_trigger_pct': 0.02,
            'dca_level_4_multiplier': 1.0,
            'dca_profit_target_pct': 0.01,
            'sl_after_last_dca_pct': 0.01,
            'trailing_stop_pct': 0.008,
        },
        # Wide stops
        {
            'take_profit_pct': 0.035,
            'dca_level_1_trigger_pct': 0.012,
            'dca_level_1_multiplier': 2.0,
            'dca_level_2_trigger_pct': 0.018,
            'dca_level_2_multiplier': 1.5,
            'dca_level_3_trigger_pct': 0.025,
            'dca_level_3_multiplier': 1.5,
            'dca_level_4_trigger_pct': 0.03,
            'dca_level_4_multiplier': 1.0,
            'dca_profit_target_pct': 0.03,
            'sl_after_last_dca_pct': 0.025,
            'trailing_stop_pct': 0.015,
        },
    ]

    param_combinations.extend(smart_combinations)

    return param_combinations


def run_optimization(symbol: str = "BTCUSD", max_workers: int = 4):
    """Run parallel parameter optimization."""

    print("=" * 70)
    print("FAST PARAMETER OPTIMIZATION - DYNAMIC DCA STRATEGY")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Symbol: {symbol}")
    print(f"Workers: {max_workers}")
    print()

    # Load data once
    print("Loading data and computing features...")
    df, (signals, confidences) = load_data_once(symbol)
    print(f"  Data rows: {len(df)}")
    print(f"  BUY signals: {np.sum(signals == 1)}")
    print()

    # Generate parameter grid
    print("Generating parameter combinations...")
    param_grid = generate_param_grid()
    print(f"  Total combinations: {len(param_grid)}")
    print()

    # Convert DataFrame to dict for pickling (multiprocessing)
    df_dict = df.to_dict('list')

    # Run optimization
    print("Running optimization...")
    print("-" * 70)

    results: List[OptimizationResult] = []
    completed = 0

    # Sequential for debugging (faster startup)
    # For production, use ProcessPoolExecutor

    use_parallel = True

    if use_parallel and max_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for params in param_grid:
                args = (params, df_dict, signals, confidences, symbol)
                future = executor.submit(worker_backtest, args)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

                completed += 1
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(param_grid)} ({100*completed/len(param_grid):.1f}%)")
    else:
        # Sequential execution
        for i, params in enumerate(param_grid):
            result = run_single_backtest(params, df, signals, confidences, symbol)
            results.append(result)

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(param_grid)} ({100*(i+1)/len(param_grid):.1f}%)")

    print()
    print(f"Completed {len(results)} backtests")

    return results


def analyze_results(results: List[OptimizationResult]) -> pd.DataFrame:
    """Analyze optimization results and find best parameters."""

    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    # Convert to DataFrame for analysis
    data = []
    for r in results:
        row = {
            'total_pnl': r.total_pnl,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'total_trades': r.total_trades,
            'avg_win': r.avg_win,
            'avg_loss': r.avg_loss,
            'sharpe': r.sharpe,
            'max_drawdown': r.max_drawdown,
            **r.params
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Filter out configurations with too few trades
    df = df[df['total_trades'] >= 20]

    if df.empty:
        print("No valid configurations found with enough trades!")
        return df

    # Calculate composite score
    # We want: high P&L, high win rate, high profit factor, low drawdown
    df['score'] = (
        df['total_pnl'] / 1000 +  # Normalize P&L
        df['win_rate'] * 100 +     # Win rate as percentage
        df['profit_factor'] * 10 + # Profit factor
        df['sharpe'] * 5 -         # Sharpe ratio
        df['max_drawdown'] * 100   # Penalize drawdown
    )

    # Sort by score
    df = df.sort_values('score', ascending=False)

    # Top 10 configurations
    print("\nTOP 10 CONFIGURATIONS (sorted by composite score):")
    print("-" * 70)

    top10 = df.head(10)

    for i, (idx, row) in enumerate(top10.iterrows()):
        print(f"\n#{i+1} - Score: {row['score']:.2f}")
        print(f"  P&L: ${row['total_pnl']:,.2f} | Win Rate: {row['win_rate']*100:.1f}% | PF: {row['profit_factor']:.2f}")
        print(f"  Trades: {row['total_trades']} | Avg Win: ${row['avg_win']:.2f} | Avg Loss: ${row['avg_loss']:.2f}")
        print(f"  Sharpe: {row['sharpe']:.2f} | Max DD: {row['max_drawdown']*100:.2f}%")
        print(f"  Parameters:")
        print(f"    TP: {row['take_profit_pct']*100:.2f}% | Trail: {row['trailing_stop_pct']*100:.2f}%")
        print(f"    DCA1: trigger={row['dca_level_1_trigger_pct']*100:.2f}%, mult={row['dca_level_1_multiplier']:.1f}x")
        print(f"    DCA2: trigger={row['dca_level_2_trigger_pct']*100:.2f}%, mult={row['dca_level_2_multiplier']:.1f}x")
        print(f"    DCA3: trigger={row['dca_level_3_trigger_pct']*100:.2f}%, mult={row['dca_level_3_multiplier']:.1f}x")
        print(f"    DCA4: trigger={row['dca_level_4_trigger_pct']*100:.2f}%, mult={row['dca_level_4_multiplier']:.1f}x")
        print(f"    DCA Profit Target: {row['dca_profit_target_pct']*100:.2f}%")
        print(f"    SL after DCA: {row['sl_after_last_dca_pct']*100:.2f}%")

    # Best by different metrics
    print("\n" + "=" * 70)
    print("BEST BY SPECIFIC METRICS:")
    print("-" * 70)

    # Best P&L
    best_pnl = df.loc[df['total_pnl'].idxmax()]
    print(f"\nBest P&L: ${best_pnl['total_pnl']:,.2f}")
    print(f"  Win Rate: {best_pnl['win_rate']*100:.1f}% | Trades: {best_pnl['total_trades']}")

    # Best Win Rate
    best_wr = df.loc[df['win_rate'].idxmax()]
    print(f"\nBest Win Rate: {best_wr['win_rate']*100:.1f}%")
    print(f"  P&L: ${best_wr['total_pnl']:,.2f} | Trades: {best_wr['total_trades']}")

    # Best Profit Factor
    best_pf = df.loc[df['profit_factor'].idxmax()]
    print(f"\nBest Profit Factor: {best_pf['profit_factor']:.2f}")
    print(f"  P&L: ${best_pf['total_pnl']:,.2f} | Win Rate: {best_pf['win_rate']*100:.1f}%")

    # Best Sharpe
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    print(f"\nBest Sharpe Ratio: {best_sharpe['sharpe']:.2f}")
    print(f"  P&L: ${best_sharpe['total_pnl']:,.2f}")

    return df


def save_results(df: pd.DataFrame, output_dir: str = None):
    """Save optimization results."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "logs" / "optimization"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results
    csv_file = output_path / f"optimization_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")

    # Save top 10 as JSON
    import json
    top10 = df.head(10).to_dict('records')
    json_file = output_path / f"top10_configs_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(top10, f, indent=2)
    print(f"Top 10 saved to: {json_file}")


def main():
    """Run the optimization."""

    # Detect number of CPU cores
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    max_workers = max(1, n_cores - 1)  # Leave 1 core free

    print(f"System has {n_cores} CPU cores, using {max_workers} workers")

    # Run optimization on BTCUSD (fastest, most liquid)
    results = run_optimization(symbol="BTCUSD", max_workers=max_workers)

    # Analyze results
    df = analyze_results(results)

    # Save results
    if not df.empty:
        save_results(df)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now()}")
    print("\nReview the results above and choose your preferred configuration.")
    print("Then update trading_config.py with the selected parameters.")


if __name__ == "__main__":
    main()
