"""
FOCUSED OPTIMIZATION - Push to Profitability
=============================================

Fine-tune around the Best P&L config (-$23) to find profitable parameters.

Best P&L Config baseline:
- Take Profit: 3.5%
- Trailing Stop: 1.2%
- DCA1 Trigger: 1.2%, Mult: 1.5x
- DCA2 Trigger: 1.5%, Mult: 1.5x
- DCA3 Trigger: 2.5%, Mult: 2.0x
- DCA4 Trigger: 2.5%, Mult: 0.5x
- DCA Profit Target: 2.5%
- SL after DCA: 1.5%

Run: python -m trading_system.High_frequency_crypto_tradin.optimize_focused
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.backtest import BacktestEngine, BacktestConfig
from trading_system.High_frequency_crypto_tradin.config import load_config


def load_data_and_signals(symbol: str = "BTCUSD"):
    """Load data and compute signals once."""
    config = load_config()
    model_dir = Path(config.model_save_dir)
    data_dir = Path(config.data_dir)

    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    data_file = data_dir / f"{symbol}_1m.csv"
    df = fe.load_data(str(data_file))

    df_features = fe.compute_all_features(df)
    df_features = fe.handle_missing_values(df_features)

    X = df_features[fe.feature_columns]
    predictions = ensemble.predict(X)
    probas = ensemble.predict_proba(X)
    confidences = np.max(probas, axis=1)

    return df, predictions, confidences


def run_backtest(params: Dict, df, signals, confidences, symbol: str = "BTCUSD"):
    """Run single backtest."""
    bt_config = BacktestConfig(
        initial_capital=100000.0,
        position_size_pct=0.02,
        max_position_size=10000.0,
        commission_rate=0.001,
        slippage_pct=0.0005,
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

    engine = BacktestEngine(bt_config, use_centralized_risk=False)
    results = engine.run(df, signals, confidences, symbol=symbol, verbose=False)

    trades_df = engine.get_trades_df()
    if trades_df.empty:
        return None

    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] <= 0]

    return {
        'params': params,
        'total_pnl': results.get('total_pnl', 0),
        'win_rate': results.get('win_rate', 0),
        'profit_factor': results.get('profit_factor', 0),
        'total_trades': results.get('total_trades', 0),
        'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
        'avg_loss': abs(losing['pnl'].mean()) if len(losing) > 0 else 0,
        'sharpe': results.get('sharpe_ratio', 0),
    }


def main():
    print("=" * 70)
    print("FOCUSED OPTIMIZATION - Push to Profitability")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print("\nBaseline: Best P&L config (-$23)")
    print()

    # Load data once
    print("Loading data...")
    df, signals, confidences = load_data_and_signals("BTCUSD")
    print(f"  Data: {len(df)} rows")
    print()

    # FOCUSED search space around Best P&L config
    focused_space = {
        'take_profit_pct': [0.030, 0.035, 0.040, 0.045],
        'trailing_stop_pct': [0.010, 0.012, 0.014, 0.016],
        'dca_level_1_trigger_pct': [0.010, 0.012, 0.014],
        'dca_level_1_multiplier': [1.25, 1.5, 1.75],
        'dca_level_2_trigger_pct': [0.012, 0.015, 0.018],
        'dca_level_2_multiplier': [1.25, 1.5, 1.75],
        'dca_level_3_trigger_pct': [0.020, 0.025, 0.030],
        'dca_level_3_multiplier': [1.5, 2.0, 2.5],
        'dca_level_4_trigger_pct': [0.020, 0.025, 0.030],
        'dca_level_4_multiplier': [0.5, 0.75, 1.0],
        'dca_profit_target_pct': [0.020, 0.025, 0.030],
        'sl_after_last_dca_pct': [0.012, 0.015, 0.018],
    }

    # Generate all combinations
    keys = list(focused_space.keys())
    values = [focused_space[k] for k in keys]

    all_combinations = list(product(*values))
    print(f"Total combinations: {len(all_combinations)}")

    # Random sample if too many
    if len(all_combinations) > 1000:
        np.random.seed(42)
        indices = np.random.choice(len(all_combinations), 1000, replace=False)
        sampled = [all_combinations[i] for i in indices]
    else:
        sampled = all_combinations

    print(f"Testing: {len(sampled)} combinations")
    print()

    # Run optimization
    results = []
    profitable_count = 0

    for i, combo in enumerate(sampled):
        params = {keys[j]: combo[j] for j in range(len(keys))}
        result = run_backtest(params, df, signals, confidences)

        if result:
            results.append(result)
            if result['total_pnl'] > 0:
                profitable_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(sampled)} | Profitable: {profitable_count}")

    print()
    print(f"Completed {len(results)} backtests")
    print(f"Profitable configs: {profitable_count}")

    # Sort by P&L
    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Show results
    print()
    print("=" * 70)
    print("TOP 10 CONFIGURATIONS BY P&L")
    print("=" * 70)

    for i, r in enumerate(results[:10]):
        pnl_status = "PROFIT" if r['total_pnl'] > 0 else "LOSS"
        print(f"\n#{i+1} - {pnl_status}: ${r['total_pnl']:,.2f}")
        print(f"  Win Rate: {r['win_rate']*100:.1f}% | PF: {r['profit_factor']:.2f} | Trades: {r['total_trades']}")
        print(f"  Avg Win: ${r['avg_win']:.2f} | Avg Loss: ${r['avg_loss']:.2f}")
        p = r['params']
        print(f"  TP: {p['take_profit_pct']*100:.1f}% | Trail: {p['trailing_stop_pct']*100:.1f}%")
        print(f"  DCA1: trig={p['dca_level_1_trigger_pct']*100:.1f}%, mult={p['dca_level_1_multiplier']:.2f}x")
        print(f"  DCA2: trig={p['dca_level_2_trigger_pct']*100:.1f}%, mult={p['dca_level_2_multiplier']:.2f}x")
        print(f"  DCA3: trig={p['dca_level_3_trigger_pct']*100:.1f}%, mult={p['dca_level_3_multiplier']:.2f}x")
        print(f"  DCA4: trig={p['dca_level_4_trigger_pct']*100:.1f}%, mult={p['dca_level_4_multiplier']:.2f}x")
        print(f"  DCA Target: {p['dca_profit_target_pct']*100:.1f}% | SL: {p['sl_after_last_dca_pct']*100:.1f}%")

    # Save best config
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        if best['total_pnl'] > 0:
            print("*** PROFITABLE CONFIG FOUND! ***")
        print("=" * 70)

        # Save to file
        output_dir = Path(__file__).parent / "logs" / "optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        import json
        with open(output_dir / f"focused_best_{timestamp}.json", 'w') as f:
            json.dump(results[:10], f, indent=2)

        print(f"\nResults saved to: {output_dir / f'focused_best_{timestamp}.json'}")

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
