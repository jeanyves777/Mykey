"""
Run Paper Trading for High-Frequency Crypto Trading System
============================================================

This script:
1. Loads trained ensemble model
2. Connects to Alpaca Paper Trading API
3. Runs live trading with ML signals
4. Logs all trades and performance
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.engine import HFTradingStrategy, AlpacaLiveEngine
from trading_system.High_frequency_crypto_tradin.engine.alpaca_live_engine import AlpacaConfig
from trading_system.High_frequency_crypto_tradin.config import load_config


def check_credentials():
    """Check for Alpaca API credentials."""
    api_key = os.getenv('ALPACA_CRYPTO_KEY')
    api_secret = os.getenv('ALPACA_CRYPTO_SECRET')

    if not api_key or not api_secret:
        print("ERROR: Alpaca crypto trading credentials not found!")
        print("\nPlease set the following environment variables:")
        print("  ALPACA_CRYPTO_KEY")
        print("  ALPACA_CRYPTO_SECRET")
        print("\nOr create a .env file with these values.")
        return False

    print(f"Alpaca Crypto API Key: {api_key[:8]}...")
    return True


def load_trained_model(model_dir: str):
    """Load trained ensemble model."""
    print("\nLoading trained model...")

    model_path = Path(model_dir) / "ensemble"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train_ensemble.py first.")
        return None, None

    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_path))

    fe = FeatureEngineer()
    fe.load_feature_config(str(Path(model_dir) / "feature_config.json"))

    print(f"  Model loaded successfully")
    print(f"  Features: {len(fe.feature_columns)}")

    return ensemble, fe


def create_strategy(ensemble, fe, config):
    """Create trading strategy with DCA-aware HF day trading config."""
    strategy_config = {
        'min_confidence': config.ensemble.confidence_threshold,
        'min_agreement': config.ensemble.min_model_agreement,
        'position_size_pct': config.strategy.position_size_pct,
        'max_position_value': config.strategy.max_position_value,
        'stop_loss_pct': config.strategy.stop_loss_pct,
        'take_profit_pct': config.strategy.take_profit_pct,
        'trailing_stop_pct': config.strategy.trailing_stop_pct,
        'max_trades_per_day': config.strategy.max_trades_per_day,
        'cooldown_seconds': config.strategy.cooldown_seconds,
        # DCA settings (HF day trading)
        'buy_only_mode': config.ensemble.buy_only_mode,
        'dca_enabled': config.strategy.dca_enabled,
        'dca_trigger_pct': config.strategy.dca_trigger_pct,
        'dca_spacing_pct': config.strategy.dca_spacing_pct,
        'max_dca_entries': config.strategy.max_dca_entries,
        'dca_multiplier': config.strategy.dca_multiplier,
        'dca_profit_target_pct': config.strategy.dca_profit_target_pct,
    }

    strategy = HFTradingStrategy(
        ensemble=ensemble,
        feature_engineer=fe,
        config=strategy_config
    )

    return strategy


def create_engine(strategy, config):
    """Create live trading engine."""
    alpaca_config = AlpacaConfig(
        api_key=os.getenv('ALPACA_CRYPTO_KEY', ''),
        api_secret=os.getenv('ALPACA_CRYPTO_SECRET', ''),
        base_url=os.getenv('ALPACA_CRYPTO_BASE_URL', 'https://paper-api.alpaca.markets'),
        crypto_symbols=config.symbols
    )

    engine = AlpacaLiveEngine(
        config=alpaca_config,
        strategy=strategy,
        ensemble=strategy.ensemble,
        feature_engineer=strategy.feature_engineer
    )

    return engine


def display_status(engine):
    """Display current trading status."""
    print("\n" + "=" * 60)
    print("TRADING STATUS")
    print("=" * 60)

    # Account info
    account = engine.get_account()
    if account:
        print(f"\nAccount:")
        print(f"  Equity: ${account.get('equity', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")

    # Positions
    positions = engine.get_positions()
    if positions:
        print(f"\nOpen Positions: {len(positions)}")
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos['qty']} @ ${pos['avg_entry_price']:.2f} "
                  f"(P&L: ${pos['unrealized_pl']:.2f})")
    else:
        print("\nNo open positions")

    print("=" * 60)


def main():
    """Main paper trading script."""
    print("=" * 60)
    print("High-Frequency Crypto Trading - Paper Trading")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Check credentials
    if not check_credentials():
        return

    # Load configuration
    config = load_config()

    # Load trained model
    ensemble, fe = load_trained_model(config.model_save_dir)
    if ensemble is None:
        return

    # Create strategy
    print("\nCreating trading strategy...")
    strategy = create_strategy(ensemble, fe, config)

    # Create engine
    print("Initializing Alpaca connection...")
    engine = create_engine(strategy, config)

    if not engine.api:
        print("ERROR: Failed to connect to Alpaca API")
        return

    # Display initial status
    display_status(engine)

    # Trading parameters
    print("\n" + "=" * 60)
    print("TRADING PARAMETERS")
    print("=" * 60)
    print(f"  Symbols: {config.symbols}")
    print(f"  Interval: {config.trading_interval_seconds} seconds")
    print(f"  Max Runtime: {config.max_runtime_hours} hours")
    print(f"  Confidence Threshold: {config.ensemble.confidence_threshold}")
    print(f"  Position Size: {config.strategy.position_size_pct*100}%")
    print(f"  Stop Loss: {config.strategy.stop_loss_pct*100}%")
    print(f"  Take Profit: {config.strategy.take_profit_pct*100}%")

    # Confirmation
    print("\n" + "=" * 60)
    print("Ready to start PAPER trading.")
    print("Press Ctrl+C at any time to stop.")
    print("=" * 60)

    try:
        input("\nPress Enter to start trading (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return

    # Start trading
    print("\n" + "=" * 60)
    print("STARTING PAPER TRADING")
    print("=" * 60)

    try:
        engine.run_live_trading(
            symbols=config.symbols,
            interval_seconds=config.trading_interval_seconds,
            max_runtime_hours=config.max_runtime_hours
        )
    except KeyboardInterrupt:
        print("\n\nTrading stopped by user")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        # Final status
        display_status(engine)

        # Close all positions if desired
        positions = engine.get_positions()
        if positions:
            print("\nYou have open positions. Close them? (y/n)")
            try:
                response = input()
                if response.lower() == 'y':
                    engine.close_all_positions()
                    print("All positions closed.")
            except:
                pass

        print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
