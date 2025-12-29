"""
Backtest the simple profitable model with proper threshold tuning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.config import load_config
from trading_system.High_frequency_crypto_tradin.train_simple_profitable import add_simple_features


def backtest_with_stats(df, signals, tp_pct, sl_pct, cooldown_bars=5,
                        initial_capital=100000, position_size_pct=0.02, commission_rate=0.001):
    """Backtest with detailed statistics."""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    tp_count = 0
    sl_count = 0
    timeout_count = 0
    trades = []
    capital = initial_capital
    last_trade_bar = -cooldown_bars - 1

    i = 0
    while i < n - 100:
        if signals[i] == 1 and (i - last_trade_bar) > cooldown_bars:
            entry_price = close[i]
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)

            position_value = capital * position_size_pct
            quantity = position_value / entry_price

            exit_price = None
            exit_bar = None

            for j in range(i + 1, min(i + 100, n)):
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price

                if hit_tp and hit_sl:
                    exit_price = sl_price
                    exit_bar = j
                    sl_count += 1
                    break
                elif hit_tp:
                    exit_price = tp_price
                    exit_bar = j
                    tp_count += 1
                    break
                elif hit_sl:
                    exit_price = sl_price
                    exit_bar = j
                    sl_count += 1
                    break

            if exit_price is None:
                exit_price = close[min(i + 100, n - 1)]
                exit_bar = min(i + 100, n - 1)
                timeout_count += 1

            pnl = quantity * (exit_price - entry_price)
            commission = position_value * commission_rate * 2
            net_pnl = pnl - commission

            trades.append({'pnl': net_pnl, 'is_win': net_pnl > 0})

            capital += net_pnl
            last_trade_bar = exit_bar
            i = exit_bar + 1
        else:
            i += 1

    total = len(trades)
    if total == 0:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'tp_rate': 0, 'sl_rate': 0, 'timeout_rate': 0, 'profit_factor': 0}

    wins = [t['pnl'] for t in trades if t['is_win']]
    losses = [t['pnl'] for t in trades if not t['is_win']]
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0.0001

    return {
        'total_trades': total,
        'win_rate': len(wins) / total,
        'total_pnl': sum(t['pnl'] for t in trades),
        'profit_factor': total_win / total_loss,
        'tp_rate': tp_count / total,
        'sl_rate': sl_count / total,
        'timeout_rate': timeout_count / total
    }


def main():
    print("=" * 80)
    print("BACKTEST SIMPLE PROFITABLE MODEL")
    print("=" * 80)

    config = load_config()

    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    print(f"\nLoading {data_file}...")

    df = pd.read_csv(data_file)
    # Use last portion for testing (simulating unseen data)
    df = df.tail(50000).reset_index(drop=True)
    print(f"Using {len(df):,} rows for backtest")

    # Add features
    print("Adding features...")
    df = add_simple_features(df)

    # Handle missing values
    for col in fe.feature_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.replace([np.inf, -np.inf], 0)

    # Get predictions
    print("Getting model predictions...")
    X = df[fe.feature_columns].astype(np.float32)
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]

    print(f"\nBUY probability distribution:")
    print(f"  Min: {buy_proba.min():.3f}")
    print(f"  Max: {buy_proba.max():.3f}")
    print(f"  Mean: {buy_proba.mean():.3f}")
    print(f"  Median: {np.median(buy_proba):.3f}")

    # Percentiles
    percentiles = [50, 60, 70, 75, 80, 85, 90, 95]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(buy_proba, p)
        print(f"    {p}th: {val:.3f}")

    # Test different thresholds
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    thresholds = [0.30, 0.35, 0.40, 0.42, 0.44, 0.45]

    for threshold in thresholds:
        signals = (buy_proba > threshold).astype(int)
        pct = 100 * signals.sum() / len(signals)
        print(f"  Threshold {threshold:.2f}: {signals.sum():,} signals ({pct:.1f}%)")

    # Test best combo at 0% commission first
    print("\n" + "=" * 80)
    print("BACKTEST AT 0% COMMISSION (to see if model has edge)")
    print("=" * 80)

    best_threshold = 0.40  # Selective
    signals = (buy_proba > best_threshold).astype(int)
    print(f"\nUsing threshold {best_threshold:.2f}: {signals.sum():,} signals")

    tp_sl_combos = [
        (0.003, 0.003),  # 0.3% TP / 0.3% SL (1:1)
        (0.004, 0.003),  # 0.4% TP / 0.3% SL (1.33:1)
        (0.005, 0.004),  # 0.5% TP / 0.4% SL (1.25:1)
        (0.006, 0.004),  # 0.6% TP / 0.4% SL (1.5:1)
        (0.008, 0.005),  # 0.8% TP / 0.5% SL (1.6:1)
        (0.01, 0.006),   # 1.0% TP / 0.6% SL (1.67:1)
    ]

    print(f"\n{'TP%':<7} {'SL%':<7} {'Trades':<8} {'WinRate':<9} {'TP%':<8} {'SL%':<8} {'T/O%':<8} {'P&L'}")
    print("-" * 75)

    for tp, sl in tp_sl_combos:
        result = backtest_with_stats(df, signals, tp_pct=tp, sl_pct=sl, commission_rate=0.0)
        marker = " ***" if result['total_pnl'] > 0 else ""
        print(f"{tp*100:.2f}   {sl*100:.2f}   {result['total_trades']:<8} {result['win_rate']*100:.1f}%     "
              f"{result['tp_rate']*100:.1f}%    {result['sl_rate']*100:.1f}%    {result['timeout_rate']*100:.1f}%    "
              f"${result['total_pnl']:>10,.2f}{marker}")

    # Full grid search at 0% commission
    print("\n" + "=" * 80)
    print("FULL GRID SEARCH (0% COMMISSION)")
    print("=" * 80)

    tp_values = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01]
    sl_values = [0.002, 0.003, 0.004, 0.005, 0.006]
    threshold_values = [0.35, 0.38, 0.40, 0.42, 0.45]

    results = []
    for threshold in threshold_values:
        signals = (buy_proba > threshold).astype(int)
        for tp, sl in product(tp_values, sl_values):
            result = backtest_with_stats(df, signals, tp_pct=tp, sl_pct=sl, commission_rate=0.0)
            result['tp_pct'] = tp
            result['sl_pct'] = sl
            result['threshold'] = threshold
            result['rr_ratio'] = tp / sl
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_pnl', ascending=False)

    # Show best results
    profitable = results_df[results_df['total_pnl'] > 0]
    print(f"\nProfitable combinations (0% commission): {len(profitable)}/{len(results_df)}")

    print(f"\n{'Thresh':<7} {'TP%':<6} {'SL%':<6} {'RR':<5} {'Trades':<7} {'WR%':<7} {'TP%':<6} {'T/O%':<6} {'P&L':<12}")
    print("-" * 75)

    for _, row in results_df.head(15).iterrows():
        marker = " ***" if row['total_pnl'] > 0 else ""
        print(f"{row['threshold']:.2f}   {row['tp_pct']*100:.2f}  {row['sl_pct']*100:.2f}  {row['rr_ratio']:.2f}  "
              f"{int(row['total_trades']):<7} {row['win_rate']*100:.1f}%   {row['tp_rate']*100:.1f}%  "
              f"{row['timeout_rate']*100:.1f}%  ${row['total_pnl']:>10,.2f}{marker}")

    # Test with 0.1% commission
    if len(profitable) > 0:
        print("\n" + "=" * 80)
        print("TEST WITH 0.1% COMMISSION")
        print("=" * 80)

        best = profitable.iloc[0]
        signals = (buy_proba > best['threshold']).astype(int)

        result_with_comm = backtest_with_stats(
            df, signals,
            tp_pct=best['tp_pct'],
            sl_pct=best['sl_pct'],
            commission_rate=0.001
        )

        print(f"\nBest params: Threshold={best['threshold']:.2f}, TP={best['tp_pct']*100:.2f}%, SL={best['sl_pct']*100:.2f}%")
        print(f"  0% commission: ${best['total_pnl']:,.2f}")
        print(f"  0.1% commission: ${result_with_comm['total_pnl']:,.2f}")
        print(f"  Commission impact: ${best['total_pnl'] - result_with_comm['total_pnl']:,.2f}")

    # Compare with random baseline
    print("\n" + "=" * 80)
    print("RANDOM BASELINE COMPARISON")
    print("=" * 80)

    np.random.seed(42)
    random_signals = np.zeros(len(buy_proba))
    # Same number of signals as threshold 0.40
    n_signals = (buy_proba > 0.40).sum()
    random_idx = np.random.choice(len(buy_proba), size=n_signals, replace=False)
    random_signals[random_idx] = 1

    print(f"\nRandom signals: {int(random_signals.sum()):,}")

    for tp, sl in [(0.003, 0.003), (0.004, 0.003), (0.005, 0.004)]:
        ml_result = backtest_with_stats(df, (buy_proba > 0.40).astype(int), tp, sl, commission_rate=0.0)
        rand_result = backtest_with_stats(df, random_signals, tp, sl, commission_rate=0.0)

        print(f"\n  TP={tp*100:.1f}%, SL={sl*100:.1f}%:")
        print(f"    ML P&L: ${ml_result['total_pnl']:>10,.2f} (WR={ml_result['win_rate']*100:.1f}%)")
        print(f"    Random P&L: ${rand_result['total_pnl']:>10,.2f} (WR={rand_result['win_rate']*100:.1f}%)")
        edge = ml_result['total_pnl'] - rand_result['total_pnl']
        print(f"    ML Edge: ${edge:>10,.2f}")


if __name__ == "__main__":
    main()
