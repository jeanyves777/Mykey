"""
Optimize for TP HIT RATE - find parameters where TP is actually reached.
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


def backtest_with_stats(df, signals, tp_pct, sl_pct, cooldown_bars=10,
                        initial_capital=100000, position_size_pct=0.02, commission_rate=0.001):
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

            for j in range(i + 1, min(i + 500, n)):  # Max 500 bars
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price

                if hit_tp and hit_sl:
                    exit_price = sl_price  # Conservative
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
    print("OPTIMIZE FOR TP HIT RATE")
    print("Finding parameters where TP is actually reached")
    print("=" * 80)

    config = load_config()

    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    df = fe.load_data(str(data_file))

    df_features = fe.compute_all_features(df)

    no_shift_cols = ['timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
                     'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    feature_cols = [col for col in df_features.columns if col not in no_shift_cols]
    for col in feature_cols:
        df_features[col] = df_features[col].shift(1)

    df_features = add_market_regime_features(df_features)
    df_features = fe.handle_missing_values(df_features)

    X = df_features[fe.feature_columns]
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]
    signals = (buy_proba > 0.40).astype(int)

    # Test tighter TPs that are more likely to hit
    tp_values = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015]  # 0.3% to 1.5%
    sl_values = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]  # 0.5% to 3%

    results = []
    for tp_pct, sl_pct in product(tp_values, sl_values):
        result = backtest_with_stats(df_features, signals, tp_pct=tp_pct, sl_pct=sl_pct)
        result['tp_pct'] = tp_pct
        result['sl_pct'] = sl_pct
        result['rr_ratio'] = tp_pct / sl_pct
        results.append(result)

    results_df = pd.DataFrame(results)

    # Sort by P&L
    results_df = results_df.sort_values('total_pnl', ascending=False)

    print("\n" + "=" * 80)
    print("TOP RESULTS BY P&L (showing TP/SL/Timeout rates)")
    print("=" * 80)
    print(f"{'TP%':<7} {'SL%':<7} {'RR':<5} {'Trades':<7} {'WinRate':<8} {'TP%':<7} {'SL%':<7} {'T/O%':<7} {'P&L':<12}")
    print("-" * 80)

    for _, row in results_df.head(20).iterrows():
        marker = " ***" if row['total_pnl'] > 0 else ""
        print(f"{row['tp_pct']*100:.2f}   {row['sl_pct']*100:.2f}   {row['rr_ratio']:.2f}  "
              f"{int(row['total_trades']):<7} {row['win_rate']*100:.1f}%   "
              f"{row['tp_rate']*100:.1f}%   {row['sl_rate']*100:.1f}%   {row['timeout_rate']*100:.1f}%   "
              f"${row['total_pnl']:>10,.2f}{marker}")

    # Sort by highest TP hit rate with min 100 trades
    with_trades = results_df[results_df['total_trades'] >= 100]
    by_tp_rate = with_trades.sort_values('tp_rate', ascending=False)

    print("\n" + "=" * 80)
    print("SORTED BY HIGHEST TP HIT RATE (min 100 trades)")
    print("=" * 80)
    print(f"{'TP%':<7} {'SL%':<7} {'RR':<5} {'Trades':<7} {'WinRate':<8} {'TP%':<7} {'SL%':<7} {'T/O%':<7} {'P&L':<12}")
    print("-" * 80)

    for _, row in by_tp_rate.head(15).iterrows():
        marker = " ***" if row['total_pnl'] > 0 else ""
        print(f"{row['tp_pct']*100:.2f}   {row['sl_pct']*100:.2f}   {row['rr_ratio']:.2f}  "
              f"{int(row['total_trades']):<7} {row['win_rate']*100:.1f}%   "
              f"{row['tp_rate']*100:.1f}%   {row['sl_rate']*100:.1f}%   {row['timeout_rate']*100:.1f}%   "
              f"${row['total_pnl']:>10,.2f}{marker}")

    # Profitable combinations
    profitable = results_df[results_df['total_pnl'] > 0]
    print(f"\n\nProfitable combinations: {len(profitable)}/{len(results_df)}")

    if len(profitable) > 0:
        print("\n*** PROFITABLE COMBOS ***")
        for _, row in profitable.iterrows():
            print(f"  TP={row['tp_pct']*100:.2f}%, SL={row['sl_pct']*100:.2f}%, "
                  f"RR={row['rr_ratio']:.2f}, WR={row['win_rate']*100:.1f}%, "
                  f"TP Rate={row['tp_rate']*100:.1f}%, P&L=${row['total_pnl']:,.2f}")


if __name__ == "__main__":
    main()
