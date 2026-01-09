"""
Train Ensemble for Forex ML System
==================================

Script to train the 5-model ensemble on Forex data.

Usage:
    cd C:\\Users\\Jean-Yves\\thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.train_ensemble --days 60
"""

import warnings
import os
import sys
import argparse
from datetime import datetime

# Suppress warnings for cleaner training output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import train_test_split

# Use relative imports when run as module
from .config.trading_config import TradingConfig, load_config
from .data.oanda_client import OandaClient
from .data.data_loader import ForexDataLoader
from .features.feature_engineer import FeatureEngineer
from .ensemble.ensemble_voting import EnsembleVotingSystem


def train_ensemble(config: TradingConfig, days: int = 60, save_path: str = None):
    """
    Train the ensemble on historical Forex data.

    Args:
        config: TradingConfig object
        days: Number of days of history to use
        save_path: Path to save trained models
    """
    print("=" * 60)
    print("FOREX ML ENSEMBLE TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training days: {days}")
    print(f"Symbols: {config.symbols}")
    print("=" * 60)

    # Initialize client
    if not config.oanda.is_valid():
        print("\n[!] OANDA credentials not configured.")
        print("Please set OANDA_API_KEY and OANDA_ACCOUNT_ID environment variables.")
        print("Using sample data generation for demo...")
        use_sample_data = True
    else:
        use_sample_data = False

    # Load data
    print("\n[*] Loading training data...")

    if use_sample_data:
        # Generate sample data for demonstration
        all_data = generate_sample_data(config.symbols, days)
    else:
        client = OandaClient(
            api_key=config.oanda.api_key,
            account_id=config.oanda.account_id,
            environment=config.oanda.environment
        )

        # Test connection
        success, message = client.test_connection()
        if not success:
            print(f"[X] Connection failed: {message}")
            return

        print(f"[+] Connected to OANDA")

        # Load data for all symbols
        data_loader = ForexDataLoader(client, config)
        all_data = data_loader.load_training_data(config.symbols, days=days, granularity='M5')

    if not all_data:
        print("[X] No data loaded")
        return

    # Combine data from all symbols
    print("\n[*] Engineering features...")
    feature_engineer = FeatureEngineer(config)

    all_X = []
    all_y = []

    for symbol, df in all_data.items():
        print(f"  Processing {symbol}...")
        try:
            # Engineer features
            featured_df = feature_engineer.engineer_features(df, fit=True)

            # Create targets
            featured_df = feature_engineer.create_target(featured_df, horizon=5)

            # Drop NaN rows
            featured_df = featured_df.dropna()

            if len(featured_df) > 0:
                X = feature_engineer.get_feature_matrix(featured_df)
                y = featured_df['target'].values

                all_X.append(X)
                all_y.append(y)
                print(f"    [+] {len(X)} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"    [X] Error: {e}")

    if not all_X:
        print("[X] No training data generated")
        return

    # Combine all data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\n[*] Total training data: {len(X)} samples, {X.shape[1]} features")
    print(f"   Class distribution: SELL={sum(y==0)}, HOLD={sum(y==1)}, BUY={sum(y==2)}")

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")

    # Initialize and train ensemble
    print("\n[*] Training ensemble...")
    ensemble = EnsembleVotingSystem(config)

    results = ensemble.train_all(X_train, y_train, X_val, y_val)

    # Print results
    print("\n[*] Training Results:")
    print("-" * 40)
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            train_acc = metrics.get('train_accuracy', 0)
            val_acc = metrics.get('val_accuracy', 0)
            print(f"  {model_name:20s}: Train={train_acc:.4f}, Val={val_acc:.4f}")
        else:
            print(f"  {model_name:20s}: Error - {metrics['error']}")

    # Evaluate ensemble
    print("\n[*] Ensemble Evaluation:")
    predictions, confidences, agreements = ensemble.predict(X_val)
    accuracy = np.mean(predictions == y_val)
    print(f"  Validation Accuracy: {accuracy:.4f}")
    print(f"  Average Confidence: {np.mean(confidences):.4f}")
    print(f"  Average Agreement: {np.mean(agreements):.1f}/5")

    # Filter by thresholds
    mask = (confidences >= config.ml.confidence_threshold) & (agreements >= config.ml.min_model_agreement)
    filtered_acc = np.mean(predictions[mask] == y_val[mask]) if sum(mask) > 0 else 0
    print(f"  Filtered Accuracy: {filtered_acc:.4f} ({sum(mask)}/{len(mask)} samples)")

    # Save models
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'ensemble')

    os.makedirs(save_path, exist_ok=True)

    print(f"\n[*] Saving models to {save_path}...")
    ensemble.save(save_path)

    # Save feature config
    feature_config_path = os.path.join(save_path, 'feature_config.json')
    feature_engineer.save_config(feature_config_path)
    print(f"  Saved feature config to {feature_config_path}")

    print("\n" + "=" * 60)
    print("[+] Training complete!")
    print("=" * 60)

    return ensemble, feature_engineer


def generate_sample_data(symbols: list, days: int) -> dict:
    """Generate sample data for demonstration when OANDA is not configured."""
    import pandas as pd

    print("  Generating sample data for demonstration...")

    data = {}
    n_bars = days * 288  # 5-minute bars per day

    for symbol in symbols:
        # Generate random walk price data
        np.random.seed(hash(symbol) % 2**32)

        if 'JPY' in symbol:
            base_price = 110.0
            volatility = 0.0003
        else:
            base_price = 1.10
            volatility = 0.00008

        # Random walk
        returns = np.random.randn(n_bars) * volatility
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n_bars) * 0.0001),
            'high': prices * (1 + np.abs(np.random.randn(n_bars)) * 0.0002),
            'low': prices * (1 - np.abs(np.random.randn(n_bars)) * 0.0002),
            'close': prices,
            'volume': np.random.randint(100, 1000, n_bars)
        })

        # Fix high/low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        # Add datetime index
        end_time = datetime.now()
        times = pd.date_range(end=end_time, periods=n_bars, freq='5min')
        df.index = times

        data[symbol] = df
        print(f"    Generated {symbol}: {len(df)} bars")

    return data


def main():
    parser = argparse.ArgumentParser(description='Train Forex ML Ensemble')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save models')
    args = parser.parse_args()

    config = load_config()
    train_ensemble(config, days=args.days, save_path=args.save_path)


if __name__ == '__main__':
    main()
