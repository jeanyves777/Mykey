"""
Analyze market conditions and test with zero commission.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.config import load_config
from trading_system.High_frequency_crypto_tradin.train_hf_scalping import add_market_regime_features


def fast_backtest(df, signals, tp_pct, sl_pct, cooldown_bars=10,
                  initial_capital=100000, position_size_pct=0.02, commission_rate=0.0):
    """Fast backtest with configurable commission."""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

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
            for j in range(i + 1, min(i + 500, n)):
                if high[j] >= tp_price:
                    exit_price = tp_price
                    break
                if low[j] <= sl_price:
                    exit_price = sl_price
                    break

            if exit_price is None:
                exit_price = close[min(i + 500, n - 1)]
                j = min(i + 500, n - 1)

            pnl = quantity * (exit_price - entry_price)
            commission = position_value * commission_rate * 2
            net_pnl = pnl - commission

            trades.append({'pnl': net_pnl, 'is_win': net_pnl > 0})
            capital += net_pnl
            last_trade_bar = j
            i = j + 1
        else:
            i += 1

    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}

    wins = [t['pnl'] for t in trades if t['is_win']]
    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'total_pnl': sum(t['pnl'] for t in trades)
    }


def main():
    print("=" * 70)
    print("MARKET CONDITIONS & ZERO-COMMISSION ANALYSIS")
    print("=" * 70)

    config = load_config()

    # Load model
    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    # Load data
    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    df = fe.load_data(str(data_file))
    print(f"\nData period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")

    # Analyze market conditions
    print("\n" + "=" * 70)
    print("MARKET CONDITIONS ANALYSIS")
    print("=" * 70)

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(60).std()
    df['trend'] = df['close'].pct_change(60)  # 1-hour trend

    # Overall stats
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    avg_volatility = df['volatility'].mean() * 100
    max_drawdown = ((df['close'] / df['close'].cummax()) - 1).min() * 100

    print(f"  Total return over period: {total_return:.2f}%")
    print(f"  Average 1-hour volatility: {avg_volatility:.4f}%")
    print(f"  Max drawdown: {max_drawdown:.2f}%")

    # Count trending vs ranging periods
    trending = (abs(df['trend']) > 0.005).sum()  # >0.5% hourly move
    ranging = len(df) - trending
    print(f"  Trending periods (>0.5% hourly move): {trending} ({100*trending/len(df):.1f}%)")
    print(f"  Ranging periods: {ranging} ({100*ranging/len(df):.1f}%)")

    # Use recent data
    df = df.tail(100000).copy().reset_index(drop=True)

    # Compute features
    print("\nComputing features...")
    df_features = fe.compute_all_features(df)

    no_shift_cols = ['timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
                     'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    feature_cols = [col for col in df_features.columns if col not in no_shift_cols]
    for col in feature_cols:
        df_features[col] = df_features[col].shift(1)

    df_features = add_market_regime_features(df_features)
    df_features = fe.handle_missing_values(df_features)

    # Get ML signals
    X = df_features[fe.feature_columns]
    probas = ensemble.predict_proba(X)
    buy_proba = probas[:, 1]
    ml_signals = (buy_proba > 0.38).astype(int)

    # Test with ZERO commission
    print("\n" + "=" * 70)
    print("ZERO COMMISSION BACKTEST")
    print("=" * 70)

    tp_values = [0.002, 0.003, 0.004, 0.005]
    sl_values = [0.003, 0.004, 0.005, 0.006]

    print(f"\nML Signals with ZERO commission:")
    print(f"{'TP%':<8} {'SL%':<8} {'Trades':<8} {'WinRate':<10} {'P&L':<12}")
    print("-" * 50)

    best_pnl_zero = -float('inf')
    best_params_zero = None

    for tp in tp_values:
        for sl in sl_values:
            result = fast_backtest(df_features, ml_signals, tp, sl, commission_rate=0.0)
            if result['total_trades'] > 0:
                print(f"{tp*100:.2f}%   {sl*100:.2f}%   {result['total_trades']:<8} "
                      f"{result['win_rate']*100:.1f}%     ${result['total_pnl']:>10,.2f}")
                if result['total_pnl'] > best_pnl_zero:
                    best_pnl_zero = result['total_pnl']
                    best_params_zero = (tp, sl)

    if best_params_zero:
        print(f"\nBest with ZERO commission: TP={best_params_zero[0]*100:.2f}%, SL={best_params_zero[1]*100:.2f}%")
        print(f"  P&L: ${best_pnl_zero:,.2f}")

    # Compare with 0.1% commission (typical)
    print(f"\nSame params with 0.1% commission:")
    result_with_comm = fast_backtest(df_features, ml_signals,
                                     best_params_zero[0], best_params_zero[1],
                                     commission_rate=0.001)
    print(f"  P&L: ${result_with_comm['total_pnl']:,.2f}")
    print(f"  Commission impact: ${best_pnl_zero - result_with_comm['total_pnl']:,.2f}")

    # Random baseline with zero commission
    print("\n" + "-" * 50)
    print("Random signals with ZERO commission:")
    np.random.seed(42)
    random_signals = np.zeros(len(ml_signals))
    random_indices = np.random.choice(len(ml_signals), size=ml_signals.sum(), replace=False)
    random_signals[random_indices] = 1

    for tp in [0.003, 0.004]:
        for sl in [0.004, 0.005]:
            result = fast_backtest(df_features, random_signals, tp, sl, commission_rate=0.0)
            if result['total_trades'] > 0:
                print(f"  TP={tp*100:.2f}%, SL={sl*100:.2f}%: "
                      f"Win={result['win_rate']*100:.1f}%, P&L=${result['total_pnl']:,.2f}")

    # Check if ML edge persists without commission
    print("\n" + "=" * 70)
    print("ML EDGE ANALYSIS (Zero Commission)")
    print("=" * 70)

    ml_zero = fast_backtest(df_features, ml_signals, 0.003, 0.004, commission_rate=0.0)
    rand_zero = fast_backtest(df_features, random_signals, 0.003, 0.004, commission_rate=0.0)

    print(f"  ML P&L (zero comm): ${ml_zero['total_pnl']:,.2f}")
    print(f"  Random P&L (zero comm): ${rand_zero['total_pnl']:,.2f}")
    print(f"  ML edge: ${ml_zero['total_pnl'] - rand_zero['total_pnl']:,.2f}")

    if ml_zero['total_pnl'] > 0:
        print("\n  *** PROFITABLE WITHOUT COMMISSION ***")
        print("  Solution: Reduce commission costs (lower fees, market maker status)")
    else:
        print("\n  *** NOT PROFITABLE EVEN WITHOUT COMMISSION ***")
        print("  Problem: ML model doesn't have predictive power in this market")


if __name__ == "__main__":
    main()
