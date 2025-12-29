"""
Quick backtest for the improved model.
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
from trading_system.High_frequency_crypto_tradin.train_improved_v2 import add_enhanced_features


def fast_backtest(df, signals, tp_pct, sl_pct, cooldown_bars=10,
                  initial_capital=100000, position_size_pct=0.02, commission_rate=0.001):
    """Fast backtest with detailed stats."""
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
    while i < n - 500:
        if signals[i] == 1 and (i - last_trade_bar) > cooldown_bars:
            entry_price = close[i]
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)

            position_value = capital * position_size_pct
            quantity = position_value / entry_price

            exit_price = None
            exit_bar = None

            for j in range(i + 1, min(i + 500, n)):
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
                exit_price = close[min(i + 500, n - 1)]
                exit_bar = min(i + 500, n - 1)
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
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'tp_rate': 0, 'sl_rate': 0, 'timeout_rate': 0}

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
    print("QUICK BACKTEST - IMPROVED MODEL")
    print("=" * 80)

    config = load_config()

    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    print(f"\nLoading data from {data_file}...")
    df = fe.load_data(str(data_file))
    print(f"Loaded {len(df):,} rows")

    # Compute standard features
    print("Computing features...")
    df_features = fe.compute_all_features(df)

    # Shift standard features
    no_shift_cols = ['timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
                     'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    feature_cols = [col for col in df_features.columns if col not in no_shift_cols]
    for col in feature_cols:
        df_features[col] = df_features[col].shift(1)

    # Add enhanced features (already shifted internally)
    print("Adding enhanced features...")
    df_features = add_enhanced_features(df_features)

    # Handle missing values
    df_features = df_features.fillna(0)
    df_features = df_features.replace([np.inf, -np.inf], 0)

    # Get predictions
    print("Getting model predictions...")

    # Use only the features the model was trained on
    available_cols = [col for col in fe.feature_columns if col in df_features.columns]
    missing_cols = [col for col in fe.feature_columns if col not in df_features.columns]

    if missing_cols:
        print(f"WARNING: {len(missing_cols)} features missing, filling with 0")
        for col in missing_cols:
            df_features[col] = 0

    # Convert all features to numeric
    X = df_features[fe.feature_columns].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]

    print(f"\nBUY probability distribution:")
    print(f"  Min: {buy_proba.min():.3f}")
    print(f"  Max: {buy_proba.max():.3f}")
    print(f"  Mean: {buy_proba.mean():.3f}")
    print(f"  Median: {np.median(buy_proba):.3f}")

    # Test different thresholds
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT THRESHOLDS")
    print("=" * 80)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

    for threshold in thresholds:
        signals = (buy_proba > threshold).astype(int)
        pct = 100 * signals.sum() / len(signals)
        print(f"\nThreshold {threshold:.2f}: {signals.sum():,} signals ({pct:.1f}%)")

    # Test with different TP/SL combos at best threshold
    print("\n" + "=" * 80)
    print("PARAMETER OPTIMIZATION")
    print("=" * 80)

    # Use threshold that gives reasonable number of signals
    best_threshold = 0.18  # Based on distribution
    for t in [0.16, 0.17, 0.18, 0.19, 0.20]:
        test_signals = (buy_proba > t).astype(int)
        pct = 100 * test_signals.sum() / len(test_signals)
        if pct > 10 and pct < 30:
            best_threshold = t
            break

    signals = (buy_proba > best_threshold).astype(int)
    print(f"\nUsing threshold {best_threshold:.2f}: {signals.sum():,} signals ({100*signals.sum()/len(signals):.1f}%)")

    tp_values = [0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02]
    sl_values = [0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.015]

    results = []
    for tp_pct, sl_pct in product(tp_values, sl_values):
        result = fast_backtest(df_features, signals, tp_pct=tp_pct, sl_pct=sl_pct, commission_rate=0.001)
        result['tp_pct'] = tp_pct
        result['sl_pct'] = sl_pct
        result['rr_ratio'] = tp_pct / sl_pct
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_pnl', ascending=False)

    print(f"\n{'TP%':<7} {'SL%':<7} {'RR':<5} {'Trades':<7} {'WinRate':<8} {'TP%':<7} {'SL%':<7} {'T/O%':<7} {'P&L':<12}")
    print("-" * 80)

    for _, row in results_df.head(20).iterrows():
        marker = " ***" if row['total_pnl'] > 0 else ""
        print(f"{row['tp_pct']*100:.2f}   {row['sl_pct']*100:.2f}   {row['rr_ratio']:.2f}  "
              f"{int(row['total_trades']):<7} {row['win_rate']*100:.1f}%   "
              f"{row['tp_rate']*100:.1f}%   {row['sl_rate']*100:.1f}%   {row['timeout_rate']*100:.1f}%   "
              f"${row['total_pnl']:>10,.2f}{marker}")

    # Profitable combos
    profitable = results_df[results_df['total_pnl'] > 0]
    print(f"\n\nProfitable combinations: {len(profitable)}/{len(results_df)}")

    if len(profitable) > 0:
        print("\n*** PROFITABLE COMBOS ***")
        for _, row in profitable.iterrows():
            print(f"  TP={row['tp_pct']*100:.2f}%, SL={row['sl_pct']*100:.2f}%, "
                  f"RR={row['rr_ratio']:.2f}, WR={row['win_rate']*100:.1f}%, "
                  f"TP Rate={row['tp_rate']*100:.1f}%, P&L=${row['total_pnl']:,.2f}")

    # Test with 0 commission
    print("\n" + "=" * 80)
    print("ZERO COMMISSION TEST")
    print("=" * 80)

    results_zero = []
    for tp_pct, sl_pct in product(tp_values, sl_values):
        result = fast_backtest(df_features, signals, tp_pct=tp_pct, sl_pct=sl_pct, commission_rate=0.0)
        result['tp_pct'] = tp_pct
        result['sl_pct'] = sl_pct
        results_zero.append(result)

    results_zero_df = pd.DataFrame(results_zero)
    profitable_zero = results_zero_df[results_zero_df['total_pnl'] > 0]

    print(f"Profitable at 0% commission: {len(profitable_zero)}/{len(results_zero_df)}")

    if len(profitable_zero) > 0:
        best = results_zero_df.sort_values('total_pnl', ascending=False).iloc[0]
        print(f"\nBest at 0% commission:")
        print(f"  TP: {best['tp_pct']*100:.2f}%")
        print(f"  SL: {best['sl_pct']*100:.2f}%")
        print(f"  Win Rate: {best['win_rate']*100:.1f}%")
        print(f"  P&L: ${best['total_pnl']:,.2f}")


if __name__ == "__main__":
    main()
