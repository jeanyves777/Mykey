"""
Analyze the BUY probability distribution from the ensemble model.
This helps find the optimal threshold for trading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.config import load_config
from trading_system.High_frequency_crypto_tradin.train_hf_scalping import add_market_regime_features

def main():
    print("Analyzing BUY Probability Distribution")
    print("=" * 60)

    config = load_config()

    # Load model
    model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_dir / "ensemble"))

    fe = FeatureEngineer()
    fe.load_feature_config(str(model_dir / "feature_config.json"))

    # Load one symbol's data
    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    print(f"\nLoading {data_file}...")

    df = fe.load_data(str(data_file))

    # Only use first 10000 rows for speed
    df = df.head(10000)
    print(f"Using {len(df)} rows for analysis")

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

    # Get predictions
    print("\nGetting probability predictions...")
    X = df_features[fe.feature_columns]
    probas = ensemble.predict_proba(X)

    # Analyze BUY probability (class 1)
    buy_proba = probas[:, 1]

    print("\n" + "=" * 60)
    print("BUY PROBABILITY DISTRIBUTION")
    print("=" * 60)
    print(f"Min BUY prob:    {buy_proba.min():.4f}")
    print(f"Max BUY prob:    {buy_proba.max():.4f}")
    print(f"Mean BUY prob:   {buy_proba.mean():.4f}")
    print(f"Median BUY prob: {np.median(buy_proba):.4f}")
    print(f"Std BUY prob:    {buy_proba.std():.4f}")

    # Percentiles
    print("\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(buy_proba, p)
        print(f"  {p}th percentile: {val:.4f}")

    # Count samples at different thresholds
    print("\n" + "-" * 60)
    print("BUY SIGNAL COUNT AT DIFFERENT THRESHOLDS:")
    print("-" * 60)
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    for thresh in thresholds:
        count = (buy_proba > thresh).sum()
        pct = 100 * count / len(buy_proba)
        print(f"  Threshold {thresh:.2f}: {count:,} BUYs ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)

    # For profitable trading with 0.40% TP and 0.30% SL:
    # - Need ~43% win rate minimum
    # - Model should filter to quality setups only
    # - Aim for 5-15% of bars as BUY signals (selective)

    # Find threshold that gives ~10% BUY rate
    target_rate = 0.10
    for thresh in np.arange(0.10, 0.70, 0.01):
        rate = (buy_proba > thresh).mean()
        if rate <= target_rate:
            print(f"For ~10% BUY rate: use threshold {thresh:.2f} (gives {rate*100:.1f}%)")
            break

    # Find threshold that gives ~20% BUY rate
    target_rate = 0.20
    for thresh in np.arange(0.10, 0.70, 0.01):
        rate = (buy_proba > thresh).mean()
        if rate <= target_rate:
            print(f"For ~20% BUY rate: use threshold {thresh:.2f} (gives {rate*100:.1f}%)")
            break

if __name__ == "__main__":
    main()
