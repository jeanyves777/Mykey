"""
Parameter Optimization V2 - FULL RANGE: Aggressive, Moderate, Conservative
Tests wide range of TP/SL combinations to find profitable sweet spot.
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
from trading_system.High_frequency_crypto_tradin.train_hf_scalping import add_market_regime_features


def fast_backtest(df, signals, tp_pct, sl_pct, cooldown_bars=10,
                  initial_capital=100000, position_size_pct=0.02, commission_rate=0.001):
    """Fast vectorized backtest."""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

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

            for j in range(i + 1, min(i + 1000, n)):  # Max 1000 bar hold
                if high[j] >= tp_price:
                    exit_price = tp_price
                    exit_bar = j
                    break
                if low[j] <= sl_price:
                    exit_price = sl_price
                    exit_bar = j
                    break

            if exit_price is None:
                exit_price = close[min(i + 1000, n - 1)]
                exit_bar = min(i + 1000, n - 1)

            pnl = quantity * (exit_price - entry_price)
            commission = position_value * commission_rate * 2
            net_pnl = pnl - commission

            trades.append({
                'pnl': net_pnl,
                'is_win': net_pnl > 0
            })

            capital += net_pnl
            last_trade_bar = exit_bar
            i = exit_bar + 1
        else:
            i += 1

    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0}

    wins = [t['pnl'] for t in trades if t['is_win']]
    losses = [t['pnl'] for t in trades if not t['is_win']]
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0.0001

    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'total_pnl': sum(t['pnl'] for t in trades),
        'profit_factor': total_win / total_loss,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean([abs(l) for l in losses]) if losses else 0
    }


def main():
    print("=" * 80)
    print("HF CRYPTO SCALPING - FULL PARAMETER OPTIMIZATION")
    print("Testing AGGRESSIVE, MODERATE, and CONSERVATIVE parameters")
    print("=" * 80)

    config = load_config()

    # Load model
    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    print("\nLoading model...")
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    # Load BTCUSD - full dataset
    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    print(f"Loading {data_file}...")

    df = fe.load_data(str(data_file))
    print(f"Using FULL dataset: {len(df)} bars")

    # Compute features
    print("Computing features...")
    df_features = fe.compute_all_features(df)

    no_shift_cols = ['timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
                     'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    feature_cols = [col for col in df_features.columns if col not in no_shift_cols]
    for col in feature_cols:
        df_features[col] = df_features[col].shift(1)

    df_features = add_market_regime_features(df_features)
    df_features = fe.handle_missing_values(df_features)

    # Get probabilities
    print("Getting model probabilities...")
    X = df_features[fe.feature_columns]
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]

    # ================================================================
    # FULL PARAMETER GRID - Aggressive to Conservative
    # ================================================================

    # Take Profit: 0.5% to 5% (aggressive to conservative)
    tp_values = [
        0.005,   # 0.5% - Very aggressive
        0.0075,  # 0.75%
        0.01,    # 1.0% - Aggressive
        0.015,   # 1.5%
        0.02,    # 2.0% - Moderate
        0.025,   # 2.5%
        0.03,    # 3.0% - Conservative
        0.04,    # 4.0%
        0.05,    # 5.0% - Very conservative
    ]

    # Stop Loss: 0.5% to 5%
    sl_values = [
        0.005,   # 0.5% - Very tight
        0.0075,  # 0.75%
        0.01,    # 1.0% - Tight
        0.015,   # 1.5%
        0.02,    # 2.0% - Moderate
        0.025,   # 2.5%
        0.03,    # 3.0% - Wide
        0.04,    # 4.0%
        0.05,    # 5.0% - Very wide
    ]

    # BUY probability thresholds
    threshold_values = [0.20, 0.25, 0.30, 0.35, 0.38, 0.40]

    total_combos = len(tp_values) * len(sl_values) * len(threshold_values)
    print(f"\nParameter Grid:")
    print(f"  TP values: {[f'{v*100:.1f}%' for v in tp_values]}")
    print(f"  SL values: {[f'{v*100:.1f}%' for v in sl_values]}")
    print(f"  Thresholds: {threshold_values}")
    print(f"  Total combinations: {total_combos}")

    results = []
    best_pnl = -float('inf')
    best_params = None

    print("\n" + "-" * 80)
    print("Running optimization...")
    print("-" * 80)

    count = 0
    for tp_pct, sl_pct, threshold in product(tp_values, sl_values, threshold_values):
        count += 1

        # Generate signals with this threshold
        signals = (buy_proba > threshold).astype(int)

        # Run backtest
        result = fast_backtest(df_features, signals, tp_pct=tp_pct, sl_pct=sl_pct)
        result['tp_pct'] = tp_pct
        result['sl_pct'] = sl_pct
        result['threshold'] = threshold
        result['rr_ratio'] = tp_pct / sl_pct  # Risk-reward ratio
        results.append(result)

        if result['total_pnl'] > best_pnl and result['total_trades'] >= 20:
            best_pnl = result['total_pnl']
            best_params = (tp_pct, sl_pct, threshold)

        if count % 100 == 0 or count == total_combos:
            print(f"  Progress: {count}/{total_combos} ({100*count/total_combos:.0f}%)")

    results_df = pd.DataFrame(results)

    # ================================================================
    # ANALYSIS BY CATEGORY
    # ================================================================

    print("\n" + "=" * 80)
    print("RESULTS BY STRATEGY TYPE")
    print("=" * 80)

    # Aggressive: TP <= 1%, SL <= 1%
    aggressive = results_df[(results_df['tp_pct'] <= 0.01) & (results_df['sl_pct'] <= 0.01)]
    aggressive = aggressive[aggressive['total_trades'] >= 20].sort_values('total_pnl', ascending=False)

    print("\n--- AGGRESSIVE (TP <= 1%, SL <= 1%) ---")
    print(f"{'TP%':<8} {'SL%':<8} {'Thresh':<8} {'Trades':<8} {'WinRate':<10} {'P&L':<14} {'PF':<8}")
    print("-" * 80)
    for _, row in aggressive.head(10).iterrows():
        print(f"{row['tp_pct']*100:.2f}%   {row['sl_pct']*100:.2f}%   {row['threshold']:.2f}    "
              f"{int(row['total_trades']):<8} {row['win_rate']*100:.1f}%     "
              f"${row['total_pnl']:>12,.2f}  {row['profit_factor']:.2f}")

    # Moderate: TP 1-2.5%, SL 1-2.5%
    moderate = results_df[(results_df['tp_pct'] > 0.01) & (results_df['tp_pct'] <= 0.025) &
                          (results_df['sl_pct'] > 0.01) & (results_df['sl_pct'] <= 0.025)]
    moderate = moderate[moderate['total_trades'] >= 20].sort_values('total_pnl', ascending=False)

    print("\n--- MODERATE (TP 1-2.5%, SL 1-2.5%) ---")
    print(f"{'TP%':<8} {'SL%':<8} {'Thresh':<8} {'Trades':<8} {'WinRate':<10} {'P&L':<14} {'PF':<8}")
    print("-" * 80)
    for _, row in moderate.head(10).iterrows():
        print(f"{row['tp_pct']*100:.2f}%   {row['sl_pct']*100:.2f}%   {row['threshold']:.2f}    "
              f"{int(row['total_trades']):<8} {row['win_rate']*100:.1f}%     "
              f"${row['total_pnl']:>12,.2f}  {row['profit_factor']:.2f}")

    # Conservative: TP > 2.5%, SL > 2.5%
    conservative = results_df[(results_df['tp_pct'] > 0.025) & (results_df['sl_pct'] > 0.025)]
    conservative = conservative[conservative['total_trades'] >= 20].sort_values('total_pnl', ascending=False)

    print("\n--- CONSERVATIVE (TP > 2.5%, SL > 2.5%) ---")
    print(f"{'TP%':<8} {'SL%':<8} {'Thresh':<8} {'Trades':<8} {'WinRate':<10} {'P&L':<14} {'PF':<8}")
    print("-" * 80)
    for _, row in conservative.head(10).iterrows():
        print(f"{row['tp_pct']*100:.2f}%   {row['sl_pct']*100:.2f}%   {row['threshold']:.2f}    "
              f"{int(row['total_trades']):<8} {row['win_rate']*100:.1f}%     "
              f"${row['total_pnl']:>12,.2f}  {row['profit_factor']:.2f}")

    # ================================================================
    # OVERALL BEST RESULTS
    # ================================================================

    print("\n" + "=" * 80)
    print("TOP 20 OVERALL RESULTS (by P&L)")
    print("=" * 80)

    results_df = results_df.sort_values('total_pnl', ascending=False)
    top_results = results_df[results_df['total_trades'] >= 20].head(20)

    print(f"{'TP%':<8} {'SL%':<8} {'RR':<6} {'Thresh':<8} {'Trades':<8} {'WinRate':<10} {'P&L':<14} {'PF':<8}")
    print("-" * 80)
    for _, row in top_results.iterrows():
        print(f"{row['tp_pct']*100:.2f}%   {row['sl_pct']*100:.2f}%   {row['rr_ratio']:.2f}   {row['threshold']:.2f}    "
              f"{int(row['total_trades']):<8} {row['win_rate']*100:.1f}%     "
              f"${row['total_pnl']:>12,.2f}  {row['profit_factor']:.2f}")

    # ================================================================
    # SUMMARY STATISTICS
    # ================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    profitable = results_df[results_df['total_pnl'] > 0]
    print(f"Profitable combinations: {len(profitable)} / {len(results_df)} ({100*len(profitable)/len(results_df):.1f}%)")

    if len(profitable) > 0:
        print(f"\nBest profitable combo:")
        best = profitable.iloc[0]
        print(f"  TP: {best['tp_pct']*100:.2f}%")
        print(f"  SL: {best['sl_pct']*100:.2f}%")
        print(f"  Threshold: {best['threshold']:.2f}")
        print(f"  Win Rate: {best['win_rate']*100:.1f}%")
        print(f"  P&L: ${best['total_pnl']:,.2f}")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")

    if best_params:
        print(f"\nBest overall (min 20 trades):")
        print(f"  TP: {best_params[0]*100:.2f}%")
        print(f"  SL: {best_params[1]*100:.2f}%")
        print(f"  Threshold: {best_params[2]:.2f}")
        print(f"  P&L: ${best_pnl:,.2f}")

    # Save results
    output_file = Path(__file__).parent / "logs" / "optimization_results_full.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    return results_df


if __name__ == "__main__":
    main()
