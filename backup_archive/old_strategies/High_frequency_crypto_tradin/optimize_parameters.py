"""
Parameter Optimization for HF Crypto Scalping
===============================================
Grid search to find optimal TP, SL, and BUY probability threshold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.config import load_config
from trading_system.High_frequency_crypto_tradin.train_hf_scalping import add_market_regime_features


@dataclass
class BacktestResult:
    tp_pct: float
    sl_pct: float
    buy_threshold: float
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe: float


def fast_backtest(df: pd.DataFrame, signals: np.ndarray,
                  tp_pct: float, sl_pct: float,
                  cooldown_bars: int = 10,
                  initial_capital: float = 100000,
                  position_size_pct: float = 0.02,
                  commission_rate: float = 0.001) -> Dict:
    """
    Fast vectorized backtest simulation.
    """
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    trades = []
    capital = initial_capital
    last_trade_bar = -cooldown_bars - 1

    i = 0
    while i < n - 100:  # Leave room for trade simulation
        # Check for BUY signal with cooldown
        if signals[i] == 1 and (i - last_trade_bar) > cooldown_bars:
            entry_price = close[i]
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)

            position_value = capital * position_size_pct
            quantity = position_value / entry_price

            # Simulate trade
            exit_price = None
            exit_bar = None

            for j in range(i + 1, min(i + 500, n)):  # Max 500 bar hold
                # Check if TP hit
                if high[j] >= tp_price:
                    exit_price = tp_price
                    exit_bar = j
                    break
                # Check if SL hit
                if low[j] <= sl_price:
                    exit_price = sl_price
                    exit_bar = j
                    break

            # If no TP/SL hit, exit at last bar
            if exit_price is None:
                exit_price = close[min(i + 500, n - 1)]
                exit_bar = min(i + 500, n - 1)

            # Calculate P&L
            pnl = quantity * (exit_price - entry_price)
            commission = position_value * commission_rate * 2  # Entry + exit
            net_pnl = pnl - commission

            trades.append({
                'entry_bar': i,
                'exit_bar': exit_bar,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'is_win': net_pnl > 0
            })

            capital += net_pnl
            last_trade_bar = exit_bar
            i = exit_bar + 1
        else:
            i += 1

    # Calculate metrics
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe': 0
        }

    wins = [t['pnl'] for t in trades if t['is_win']]
    losses = [t['pnl'] for t in trades if not t['is_win']]

    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0.0001

    returns = [t['pnl'] / initial_capital for t in trades]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24 * 60)  # Annualized

    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'total_pnl': sum(t['pnl'] for t in trades),
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean([abs(l) for l in losses]) if losses else 0,
        'profit_factor': total_win / total_loss,
        'sharpe': sharpe
    }


def main():
    print("=" * 70)
    print("HF CRYPTO SCALPING - PARAMETER OPTIMIZATION")
    print("=" * 70)

    config = load_config()

    # Load model
    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    print("\nLoading model...")
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    # Load BTCUSD data only for speed
    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    print(f"Loading {data_file}...")

    df = fe.load_data(str(data_file))
    # Use subset for faster optimization
    df = df.tail(50000).copy()  # Last ~35 days
    df = df.reset_index(drop=True)
    print(f"Using {len(df)} bars for optimization")

    # Compute features
    print("Computing features...")
    df_features = fe.compute_all_features(df)

    # Shift features
    no_shift_cols = ['timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
                     'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    feature_cols = [col for col in df_features.columns if col not in no_shift_cols]
    for col in feature_cols:
        df_features[col] = df_features[col].shift(1)

    df_features = add_market_regime_features(df_features)
    df_features = fe.handle_missing_values(df_features)

    # Get probabilities once
    print("Getting model probabilities...")
    X = df_features[fe.feature_columns]
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]

    # Parameter grid
    tp_values = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005]  # 0.2% to 0.5%
    sl_values = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008]    # 0.2% to 0.8%
    threshold_values = [0.30, 0.35, 0.38, 0.40, 0.42, 0.45]   # BUY probability threshold

    print(f"\nGrid Search:")
    print(f"  TP values: {[f'{v*100:.2f}%' for v in tp_values]}")
    print(f"  SL values: {[f'{v*100:.2f}%' for v in sl_values]}")
    print(f"  Thresholds: {threshold_values}")
    print(f"  Total combinations: {len(tp_values) * len(sl_values) * len(threshold_values)}")

    results = []
    best_pnl = -float('inf')
    best_params = None

    print("\n" + "-" * 70)
    print("Running optimization...")
    print("-" * 70)

    total = len(tp_values) * len(sl_values) * len(threshold_values)
    count = 0

    for tp_pct, sl_pct, threshold in product(tp_values, sl_values, threshold_values):
        count += 1

        # Generate signals with this threshold
        signals = (buy_proba > threshold).astype(int)

        # Run backtest
        result = fast_backtest(
            df_features, signals,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            cooldown_bars=10
        )

        result['tp_pct'] = tp_pct
        result['sl_pct'] = sl_pct
        result['threshold'] = threshold
        results.append(result)

        # Track best
        if result['total_pnl'] > best_pnl and result['total_trades'] >= 50:
            best_pnl = result['total_pnl']
            best_params = (tp_pct, sl_pct, threshold)

        if count % 36 == 0 or count == total:
            print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by P&L
    results_df = results_df.sort_values('total_pnl', ascending=False)

    print("\n" + "=" * 70)
    print("TOP 20 PARAMETER COMBINATIONS (by P&L)")
    print("=" * 70)
    print(f"{'TP%':<8} {'SL%':<8} {'Thresh':<8} {'Trades':<8} {'WinRate':<10} {'P&L':<12} {'PF':<8}")
    print("-" * 70)

    for _, row in results_df.head(20).iterrows():
        print(f"{row['tp_pct']*100:.2f}%   {row['sl_pct']*100:.2f}%   {row['threshold']:.2f}    "
              f"{int(row['total_trades']):<8} {row['win_rate']*100:.1f}%     "
              f"${row['total_pnl']:>10,.2f}  {row['profit_factor']:.2f}")

    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    if best_params:
        print(f"  Take Profit: {best_params[0]*100:.2f}%")
        print(f"  Stop Loss: {best_params[1]*100:.2f}%")
        print(f"  BUY Threshold: {best_params[2]:.2f}")
        print(f"  Expected P&L: ${best_pnl:,.2f}")

    # Find profitable combinations
    profitable = results_df[results_df['total_pnl'] > 0]
    print(f"\n  Profitable combinations: {len(profitable)} / {len(results_df)}")

    # Find best by win rate with min trades
    with_min_trades = results_df[results_df['total_trades'] >= 50]
    if len(with_min_trades) > 0:
        best_wr = with_min_trades.sort_values('win_rate', ascending=False).iloc[0]
        print(f"\n  Best Win Rate (min 50 trades):")
        print(f"    TP: {best_wr['tp_pct']*100:.2f}%, SL: {best_wr['sl_pct']*100:.2f}%, "
              f"Thresh: {best_wr['threshold']:.2f}")
        print(f"    Win Rate: {best_wr['win_rate']*100:.1f}%, P&L: ${best_wr['total_pnl']:,.2f}")

    # Save results
    output_file = Path(__file__).parent / "logs" / "optimization_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    return results_df


if __name__ == "__main__":
    main()
