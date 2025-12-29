"""
Debug why strategy loses even with 44% win rate and 2:1 RR at 0% commission.
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


def detailed_backtest(df, signals, tp_pct, sl_pct, cooldown_bars=10,
                      initial_capital=100000, position_size_pct=0.02, commission_rate=0.0):
    """Detailed backtest with full trade logging."""
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
            exit_type = "timeout"

            for j in range(i + 1, min(i + 1000, n)):
                # Check TP and SL on same bar
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price

                if hit_tp and hit_sl:
                    # Both hit on same bar - assume SL first (conservative)
                    exit_price = sl_price
                    exit_bar = j
                    exit_type = "sl_same_bar"
                    break
                elif hit_tp:
                    exit_price = tp_price
                    exit_bar = j
                    exit_type = "tp"
                    break
                elif hit_sl:
                    exit_price = sl_price
                    exit_bar = j
                    exit_type = "sl"
                    break

            if exit_price is None:
                exit_price = close[min(i + 1000, n - 1)]
                exit_bar = min(i + 1000, n - 1)

            pnl = quantity * (exit_price - entry_price)
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                'entry_bar': i,
                'exit_bar': exit_bar,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_type': exit_type,
                'is_win': pnl > 0,
                'holding_bars': exit_bar - i
            })

            capital += pnl
            last_trade_bar = exit_bar
            i = exit_bar + 1
        else:
            i += 1

    return trades


def main():
    print("=" * 80)
    print("DEBUG: WHY STRATEGY LOSES WITH GOOD WIN RATE")
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

    # Run detailed backtest
    print("\nRunning detailed backtest (TP=2%, SL=1%, 0% commission)...")
    trades = detailed_backtest(df_features, signals, tp_pct=0.02, sl_pct=0.01)

    trades_df = pd.DataFrame(trades)

    print(f"\nTotal trades: {len(trades_df)}")

    # Win/Loss breakdown
    wins = trades_df[trades_df['is_win']]
    losses = trades_df[~trades_df['is_win']]

    print(f"\nWins: {len(wins)} ({100*len(wins)/len(trades_df):.1f}%)")
    print(f"Losses: {len(losses)} ({100*len(losses)/len(trades_df):.1f}%)")

    # Exit type breakdown
    print("\n" + "=" * 40)
    print("EXIT TYPE BREAKDOWN")
    print("=" * 40)
    exit_types = trades_df['exit_type'].value_counts()
    for exit_type, count in exit_types.items():
        pct = 100 * count / len(trades_df)
        avg_pnl = trades_df[trades_df['exit_type'] == exit_type]['pnl_pct'].mean()
        print(f"  {exit_type}: {count} ({pct:.1f}%) - Avg P&L: {avg_pnl:.3f}%")

    # P&L analysis
    print("\n" + "=" * 40)
    print("P&L ANALYSIS")
    print("=" * 40)
    print(f"Expected TP P&L: +2.00%")
    print(f"Expected SL P&L: -1.00%")
    print(f"")
    print(f"Actual avg WIN P&L: {wins['pnl_pct'].mean():.4f}%")
    print(f"Actual avg LOSS P&L: {losses['pnl_pct'].mean():.4f}%")
    print(f"")
    print(f"Total P&L: ${trades_df['pnl'].sum():,.2f}")

    # Check if trades are hitting exact TP/SL
    tp_hits = trades_df[trades_df['exit_type'] == 'tp']
    sl_hits = trades_df[trades_df['exit_type'] == 'sl']

    if len(tp_hits) > 0:
        print(f"\nTP hits actual P&L%: min={tp_hits['pnl_pct'].min():.4f}, max={tp_hits['pnl_pct'].max():.4f}, mean={tp_hits['pnl_pct'].mean():.4f}")

    if len(sl_hits) > 0:
        print(f"SL hits actual P&L%: min={sl_hits['pnl_pct'].min():.4f}, max={sl_hits['pnl_pct'].max():.4f}, mean={sl_hits['pnl_pct'].mean():.4f}")

    # Check timeouts
    timeouts = trades_df[trades_df['exit_type'] == 'timeout']
    if len(timeouts) > 0:
        print(f"\nTimeouts: {len(timeouts)} trades")
        print(f"  Avg P&L: {timeouts['pnl_pct'].mean():.4f}%")
        print(f"  Win rate: {100*len(timeouts[timeouts['is_win']])/len(timeouts):.1f}%")

    # Same-bar hits (TP and SL hit on same bar)
    same_bar = trades_df[trades_df['exit_type'] == 'sl_same_bar']
    if len(same_bar) > 0:
        print(f"\nSame-bar TP/SL (assumed SL): {len(same_bar)} trades")
        print(f"  This is {100*len(same_bar)/len(trades_df):.1f}% of all trades!")

    # Breakdown by actual P&L range
    print("\n" + "=" * 40)
    print("P&L DISTRIBUTION")
    print("=" * 40)
    bins = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    trades_df['pnl_bin'] = pd.cut(trades_df['pnl_pct'], bins=bins)
    dist = trades_df['pnl_bin'].value_counts().sort_index()
    for bin_range, count in dist.items():
        print(f"  {bin_range}: {count}")


if __name__ == "__main__":
    main()
