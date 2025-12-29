"""
Test with lower commission rates to find viable parameters.
Tests: 0.1%, 0.05%, 0.02%, and 0% commission
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

            for j in range(i + 1, min(i + 1000, n)):
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

            trades.append({'pnl': net_pnl, 'is_win': net_pnl > 0})

            capital += net_pnl
            last_trade_bar = exit_bar
            i = exit_bar + 1
        else:
            i += 1

    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'profit_factor': 0}

    wins = [t['pnl'] for t in trades if t['is_win']]
    losses = [t['pnl'] for t in trades if not t['is_win']]
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0.0001

    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'total_pnl': sum(t['pnl'] for t in trades),
        'profit_factor': total_win / total_loss
    }


def main():
    print("=" * 80)
    print("COMMISSION SENSITIVITY ANALYSIS")
    print("=" * 80)

    config = load_config()

    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    print("\nLoading model...")
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    df = fe.load_data(str(data_file))
    print(f"Using {len(df)} bars")

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

    print("Getting model probabilities...")
    X = df_features[fe.feature_columns]
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]

    # Best parameters from previous test
    tp_values = [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    sl_values = [0.0075, 0.01, 0.015, 0.02]
    threshold = 0.40

    # Commission rates to test
    commission_rates = [0.001, 0.0005, 0.0002, 0.0001, 0.0]

    signals = (buy_proba > threshold).astype(int)

    for comm_rate in commission_rates:
        print("\n" + "=" * 80)
        print(f"COMMISSION RATE: {comm_rate*100:.3f}%")
        print("=" * 80)

        results = []
        for tp_pct, sl_pct in product(tp_values, sl_values):
            result = fast_backtest(df_features, signals, tp_pct=tp_pct, sl_pct=sl_pct,
                                   commission_rate=comm_rate)
            result['tp_pct'] = tp_pct
            result['sl_pct'] = sl_pct
            result['rr_ratio'] = tp_pct / sl_pct
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_pnl', ascending=False)

        profitable = results_df[results_df['total_pnl'] > 0]
        print(f"\nProfitable: {len(profitable)}/{len(results_df)}")

        print(f"\n{'TP%':<8} {'SL%':<8} {'RR':<6} {'Trades':<8} {'WinRate':<10} {'P&L':<14} {'PF':<8}")
        print("-" * 80)

        for _, row in results_df.head(10).iterrows():
            marker = " ***" if row['total_pnl'] > 0 else ""
            print(f"{row['tp_pct']*100:.2f}%   {row['sl_pct']*100:.2f}%   {row['rr_ratio']:.2f}   "
                  f"{int(row['total_trades']):<8} {row['win_rate']*100:.1f}%     "
                  f"${row['total_pnl']:>12,.2f}  {row['profit_factor']:.2f}{marker}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY: MINIMUM COMMISSION FOR PROFITABILITY")
    print("=" * 80)

    # Test the best combo at different rates
    best_tp = 0.02
    best_sl = 0.01

    print(f"\nBest params (TP={best_tp*100}%, SL={best_sl*100}%):")
    for comm_rate in [0.001, 0.0005, 0.0003, 0.0002, 0.0001, 0.00005, 0.0]:
        result = fast_backtest(df_features, signals, best_tp, best_sl, commission_rate=comm_rate)
        status = "PROFITABLE" if result['total_pnl'] > 0 else "LOSS"
        print(f"  {comm_rate*100:.4f}% commission: P&L ${result['total_pnl']:>10,.2f} - {status}")


if __name__ == "__main__":
    main()
