"""
Quick Test Script - Verify System Installation
================================================

Run this first to verify all components are working before training.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test all critical imports."""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)

    errors = []

    # Core libraries
    try:
        import numpy as np
        import pandas as pd
        print(f"  numpy {np.__version__}: OK")
        print(f"  pandas {pd.__version__}: OK")
    except ImportError as e:
        errors.append(f"Core: {e}")

    # ML libraries
    try:
        from sklearn.ensemble import RandomForestClassifier
        import sklearn
        print(f"  scikit-learn {sklearn.__version__}: OK")
    except ImportError as e:
        errors.append(f"sklearn: {e}")

    try:
        import xgboost as xgb
        print(f"  xgboost {xgb.__version__}: OK")
    except ImportError as e:
        errors.append(f"xgboost: {e}")
        print(f"  xgboost: MISSING (pip install xgboost)")

    try:
        import lightgbm as lgb
        print(f"  lightgbm {lgb.__version__}: OK")
    except ImportError as e:
        errors.append(f"lightgbm: {e}")
        print(f"  lightgbm: MISSING (pip install lightgbm)")

    try:
        import catboost
        print(f"  catboost {catboost.__version__}: OK")
    except ImportError as e:
        errors.append(f"catboost: {e}")
        print(f"  catboost: MISSING (pip install catboost)")

    try:
        import tensorflow as tf
        print(f"  tensorflow {tf.__version__}: OK")
    except ImportError as e:
        errors.append(f"tensorflow: {e}")
        print(f"  tensorflow: MISSING (pip install tensorflow)")

    # Alpaca
    try:
        from alpaca.trading.client import TradingClient
        print(f"  alpaca-py: OK")
    except ImportError as e:
        errors.append(f"alpaca-py: {e}")
        print(f"  alpaca-py: MISSING (pip install alpaca-py)")

    return errors


def test_modules():
    """Test our custom modules."""
    print("\n" + "=" * 60)
    print("Testing Custom Modules")
    print("=" * 60)

    errors = []

    try:
        from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
        print("  FeatureEngineer: OK")
    except Exception as e:
        errors.append(f"FeatureEngineer: {e}")
        print(f"  FeatureEngineer: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.features.technical_features import TechnicalFeatures
        print("  TechnicalFeatures: OK")
    except Exception as e:
        errors.append(f"TechnicalFeatures: {e}")
        print(f"  TechnicalFeatures: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.features.microstructure_features import MicrostructureFeatures
        print("  MicrostructureFeatures: OK")
    except Exception as e:
        errors.append(f"MicrostructureFeatures: {e}")
        print(f"  MicrostructureFeatures: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.ml_models import (
            RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel, NeuralNetworkModel
        )
        print("  ML Models (5): OK")
    except Exception as e:
        errors.append(f"ML Models: {e}")
        print(f"  ML Models: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
        print("  EnsembleVotingSystem: OK")
    except Exception as e:
        errors.append(f"EnsembleVotingSystem: {e}")
        print(f"  EnsembleVotingSystem: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.backtest import BacktestEngine, WalkForwardValidator
        print("  BacktestEngine: OK")
        print("  WalkForwardValidator: OK")
    except Exception as e:
        errors.append(f"Backtest: {e}")
        print(f"  Backtest: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.engine import HFTradingStrategy
        print("  HFTradingStrategy: OK")
    except Exception as e:
        errors.append(f"HFTradingStrategy: {e}")
        print(f"  HFTradingStrategy: FAILED - {e}")

    try:
        from trading_system.High_frequency_crypto_tradin.config import load_config
        print("  Config: OK")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"  Config: FAILED - {e}")

    return errors


def test_data():
    """Test data availability."""
    print("\n" + "=" * 60)
    print("Testing Data Availability")
    print("=" * 60)

    data_dir = Path(__file__).parent / "Crypto_Data_from_Binance"

    if not data_dir.exists():
        print(f"  ERROR: Data directory not found: {data_dir}")
        return ["Data directory missing"]

    csv_files = list(data_dir.glob("*.csv"))
    print(f"  Data directory: {data_dir}")
    print(f"  CSV files found: {len(csv_files)}")

    if csv_files:
        # Check first file
        import pandas as pd
        sample_file = csv_files[0]
        df = pd.read_csv(sample_file, nrows=5)
        print(f"\n  Sample file: {sample_file.name}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample rows: {len(df)}")

    return []


def test_feature_engineering():
    """Quick test of feature engineering."""
    print("\n" + "=" * 60)
    print("Testing Feature Engineering")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer

        # Load sample data
        data_dir = Path(__file__).parent / "Crypto_Data_from_Binance"
        sample_file = data_dir / "BTCUSDT_1m.csv"

        if not sample_file.exists():
            print("  Skipping - no data file found")
            return []

        fe = FeatureEngineer()
        df = fe.load_data(str(sample_file))
        print(f"  Loaded {len(df)} rows")

        # Take subset for speed
        df_sample = df.head(1000)

        # Compute features
        df_features = fe.compute_all_features(df_sample)
        n_features = len([c for c in df_features.columns if c not in
                         ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                          'datetime', 'close_time', 'quote_volume', 'trades',
                          'taker_buy_base', 'taker_buy_quote']])

        print(f"  Features computed: {n_features}")
        print(f"  Sample features: {list(df_features.columns[10:15])}")

        return []

    except Exception as e:
        print(f"  FAILED: {e}")
        return [str(e)]


def test_ensemble_quick():
    """Quick ensemble test with synthetic data."""
    print("\n" + "=" * 60)
    print("Testing Ensemble (Synthetic Data)")
    print("=" * 60)

    try:
        import numpy as np
        import pandas as pd
        from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem

        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))

        # Split
        split = int(0.8 * n_samples)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")

        # Create ensemble
        ensemble = EnsembleVotingSystem(
            voting_method='soft',
            confidence_threshold=0.5
        )

        print("\n  Training ensemble (this may take a minute)...")

        # Train
        metrics = ensemble.train(X_train, y_train, X_val, y_val, verbose=False)

        print(f"\n  Training complete!")
        print(f"  Models trained: {len(ensemble.models)}")

        # Test prediction
        predictions = ensemble.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        print(f"  Validation Accuracy: {accuracy:.4f}")

        # Get signal
        signal, confidence, details = ensemble.get_trade_signal(X_val.iloc[:1])
        print(f"  Sample Signal: {signal}, Confidence: {confidence:.4f}")

        return []

    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return [str(e)]


def main():
    """Run all tests."""
    print("=" * 60)
    print("HIGH-FREQUENCY CRYPTO TRADING - SYSTEM TEST")
    print("=" * 60)

    all_errors = []

    # Test imports
    errors = test_imports()
    all_errors.extend(errors)

    # Test modules
    errors = test_modules()
    all_errors.extend(errors)

    # Test data
    errors = test_data()
    all_errors.extend(errors)

    # Test feature engineering
    errors = test_feature_engineering()
    all_errors.extend(errors)

    # Test ensemble (optional - takes time)
    print("\n" + "=" * 60)
    print("Quick Ensemble Test?")
    print("=" * 60)
    print("This will train a quick ensemble on synthetic data.")
    print("It takes about 1-2 minutes.")

    try:
        response = input("\nRun ensemble test? (y/n): ")
        if response.lower() == 'y':
            errors = test_ensemble_quick()
            all_errors.extend(errors)
        else:
            print("Skipping ensemble test.")
    except:
        # Non-interactive mode
        errors = test_ensemble_quick()
        all_errors.extend(errors)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\nErrors found: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")
        print("\nPlease fix these errors before training.")
    else:
        print("\n  ALL TESTS PASSED!")
        print("\n  You can now run: python train_ensemble.py")

    return len(all_errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
