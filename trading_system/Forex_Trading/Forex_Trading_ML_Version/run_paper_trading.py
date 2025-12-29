"""
Run Paper Trading for Forex ML System
=====================================

Main entry point for paper trading with the ML ensemble.

Usage:
    cd C:\\Users\\Jean-Yves\\thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.run_paper_trading
"""

import warnings
import os
import sys
import argparse

# Suppress all warnings for clean output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Use relative imports when run as module
from .config.trading_config import TradingConfig, load_config, print_config_info
from .data.oanda_client import OandaClient
from .features.feature_engineer import FeatureEngineer
from .ensemble.ensemble_voting import EnsembleVotingSystem
from .engine.paper_trading_engine import PaperTradingEngine
from .risk_management.risk_manager import RiskManager


def print_dca_position_sizing_info(config: TradingConfig):
    """Print detailed DCA and position sizing information."""
    print("\n" + "=" * 80)
    print("DCA & POSITION SIZING CONFIGURATION")
    print("=" * 80)

    if not config.dca.enabled:
        print("\n[!] DCA is DISABLED - Using standard pip-based TP/SL")
        print(f"    Default SL: {config.risk.default_stop_loss_pips} pips")
        print(f"    Default TP: {config.risk.default_take_profit_pips} pips")
        return

    dca = config.dca
    risk_manager = RiskManager(config)
    use_pips = getattr(dca, 'use_pip_based_dca', False)

    # DCA Overview
    print("\n[DCA PARAMETERS]")
    if use_pips:
        print(f"  Mode:              PIP-BASED")
        print(f"  Take Profit:       {dca.take_profit_pips:.0f} pips")
        print(f"  Initial SL:        {dca.initial_sl_pips:.0f} pips (allows DCAs at 5, 10, 15, 20 pips)")
        print(f"  Trailing Stop:     {dca.trailing_stop_pips:.0f} pips")
        print(f"  SL after DCA:      {dca.sl_after_dca_pips:.0f} pips (from avg entry)")
        print(f"  DCA Profit Target: {dca.dca_profit_target_pips:.0f} pips")
    else:
        initial_sl = getattr(dca, 'initial_sl_pct', 0.08)
        print(f"  Mode:              PERCENTAGE-BASED")
        print(f"  Take Profit:       {dca.take_profit_pct:.1%}")
        print(f"  Initial SL:        {initial_sl:.1%} (wide, allows all DCAs to trigger)")
        print(f"  Trailing Stop:     {dca.trailing_stop_pct:.1%} (activates after in profit)")
        print(f"  SL after DCA:      {dca.sl_after_dca_pct:.1%} (tighter, from avg entry)")
        print(f"  DCA Profit Target: {dca.dca_profit_target_pct:.1%}")
    print(f"  Max DCA Levels:    {dca.max_dca_levels}")

    # Position Sizing with DCA
    print("\n[POSITION SIZING WITH DCA]")
    print(f"  Initial Size Divisor: {dca.initial_size_divisor:.1f}x")
    print(f"  Initial Position:     {100/dca.initial_size_divisor:.0f}% of normal $1/pip size")

    # DCA Levels Table
    print("\n[DCA LEVEL BREAKDOWN]")
    print("-" * 80)

    if use_pips:
        print(f"{'Level':<8} {'Trigger':<12} {'Multiplier':<12} {'Loss (pips)':<15} {'Size vs Initial':<15}")
        print("-" * 80)
        print(f"{'Initial':<8} {'Entry':<12} {'1.00x':<12} {'0 pips':<15} {'1.00x (Base)':<15}")

        total_multiplier = 1.0
        for i in range(1, dca.max_dca_levels + 1):
            trigger_pips = dca.get_dca_trigger_pips(i)
            multiplier = dca.get_dca_multiplier(i)
            total_multiplier += multiplier
            trigger_str = f"{trigger_pips:.0f} pips"
            mult_str = f"{multiplier:.2f}x"
            drop_str = f"{trigger_pips:.0f} pips loss"
            total_str = f"{total_multiplier:.2f}x total"
            print(f"{'DCA ' + str(i):<8} {trigger_str:<12} {mult_str:<12} {drop_str:<15} {total_str:<15}")
    else:
        print(f"{'Level':<8} {'Trigger':<12} {'Multiplier':<12} {'Cumulative Drop':<18} {'Size vs Initial':<15}")
        print("-" * 80)
        print(f"{'Initial':<8} {'Entry':<12} {'1.00x':<12} {'0.0%':<18} {'1.00x (Base)':<15}")

        cumulative_drop = 0.0
        total_multiplier = 1.0
        dca_levels = dca.get_dca_levels()
        for i, level in enumerate(dca_levels, 1):
            cumulative_drop += level.trigger_pct * 100
            total_multiplier += level.multiplier
            trigger_str = f"{level.trigger_pct:.1%}"
            mult_str = f"{level.multiplier:.2f}x"
            drop_str = f"{cumulative_drop:.1f}% from avg"
            total_str = f"{total_multiplier:.2f}x total"
            print(f"{'DCA ' + str(i):<8} {trigger_str:<12} {mult_str:<12} {drop_str:<18} {total_str:<15}")

    print("-" * 80)

    # Calculate max exposure
    max_total_mult = dca.get_total_dca_multiplier()
    max_exposure_pct = (max_total_mult / dca.initial_size_divisor) * 100

    print(f"\n[EXPOSURE SUMMARY]")
    print(f"  Total DCA Multiplier (all levels): {max_total_mult:.2f}x")
    print(f"  Max Position Size (all DCAs):      {max_exposure_pct:.0f}% of base risk")
    print(f"  Effective Max Exposure:            {max_exposure_pct/100:.2f}x base position")

    # Example calculation using risk-based sizing
    print("\n[EXAMPLE: Risk-Based Position Sizing]")
    sample_equity = 5000.0  # Example account equity
    risk_pct = config.risk.position_size_pct  # Default 2%
    base_risk = sample_equity * risk_pct
    sl_pips = dca.initial_sl_pips  # 25 pips

    print(f"  Account Equity:         ${sample_equity:,.0f}")
    print(f"  Risk per Trade:         {risk_pct:.1%} = ${base_risk:.2f}")
    print(f"  Initial SL:             {sl_pips:.0f} pips")
    print(f"  DCA Divisor:            {dca.initial_size_divisor:.0f}x")

    # Risk per initial entry with DCA
    initial_risk = base_risk / dca.initial_size_divisor
    print(f"  Initial Entry Risk:     ${initial_risk:.2f} ({100/dca.initial_size_divisor:.0f}% of base)")

    # Example units for EUR_USD (pip value = $0.0001/unit)
    pip_value_usd = 0.0001  # EUR_USD
    initial_units = int(initial_risk / (sl_pips * pip_value_usd))
    print(f"  EUR_USD Initial Units:  {initial_units:,} units")

    MAX_TOTAL_UNITS = 100000  # Max units limit

    print("\n  Progression if all DCAs trigger:")
    print(f"  {'Stage':<12} {'Units Added':<15} {'Total Units':<15} {'Risk $':<15}")
    print(f"  {'-'*60}")

    running_units = initial_units
    running_risk = initial_risk
    print(f"  {'Entry':<12} {initial_units:>13,} {running_units:>13,} ${running_risk:>12.2f}")

    if use_pips:
        for i in range(1, dca.max_dca_levels + 1):
            multiplier = dca.get_dca_multiplier(i)
            dca_units = int(initial_units * multiplier)
            dca_risk = initial_risk * multiplier
            running_units += dca_units
            running_risk += dca_risk
            print(f"  {'+ DCA ' + str(i):<12} {dca_units:>13,} {running_units:>13,} ${running_risk:>12.2f}")
    else:
        dca_levels = dca.get_dca_levels()
        for i, level in enumerate(dca_levels, 1):
            dca_units = int(initial_units * level.multiplier)
            dca_risk = initial_risk * level.multiplier
            running_units += dca_units
            running_risk += dca_risk
            print(f"  {'+ DCA ' + str(i):<12} {dca_units:>13,} {running_units:>13,} ${running_risk:>12.2f}")

    print(f"  {'-'*60}")
    print(f"  {'MAX TOTAL':<12} {'':<15} {running_units:>13,} ${running_risk:>12.2f}")

    # Risk per trade at each stage
    print("\n[STOP LOSS RULES]")
    if use_pips:
        print(f"  Initial Entry SL:  {dca.initial_sl_pips:.0f} pips (wide to allow all DCAs)")
        print(f"  After DCA SL:      {dca.sl_after_dca_pips:.0f} pips from ORIGINAL ENTRY (not avg)")
        print(f"  >>> ALL TRADES (original + DCAs) share the SAME SL <<<")
        print(f"  >>> When SL hits, ALL trades close together <<<")
    else:
        print(f"  Initial Entry SL:  {dca.trailing_stop_pct:.1%} (trailing)")
        print(f"  After DCA SL:      {dca.sl_after_dca_pct:.1%} from ORIGINAL ENTRY")

    # Pending Orders Feature
    print("\n[PENDING DCA ORDERS]")
    print(f"  Mode: LIMIT ORDERS (staged upfront)")
    print(f"  >>> When position opens, ALL {dca.max_dca_levels} DCA limit orders are placed immediately <<<")
    print(f"  >>> Orders execute automatically when price reaches trigger level <<<")
    print(f"  >>> Works even if system is offline - OANDA handles execution <<<")
    print(f"  >>> When position closes, all unfilled DCA orders are cancelled <<<")

    print("=" * 80)


def check_credentials(config: TradingConfig) -> bool:
    """Check if OANDA credentials are configured."""
    if not config.oanda.is_valid():
        print("[X] OANDA credentials not configured!")
        print("\nPlease set the following environment variables:")
        print("  OANDA_API_KEY=your_api_key")
        print("  OANDA_ACCOUNT_ID=your_account_id")
        print("\nOr add them to a .env file in the project root.")
        return False
    return True


def load_trained_ensemble(config: TradingConfig, model_path: str = None):
    """Load trained ensemble and feature engineer."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'ensemble')

    if not os.path.exists(model_path):
        print(f"[X] Model directory not found: {model_path}")
        print("\nPlease train the ensemble first:")
        print("  py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.train_ensemble")
        return None, None

    # Load ensemble
    print(f"Loading ensemble from {model_path}...")
    ensemble = EnsembleVotingSystem(config)
    ensemble.load(model_path)

    # Check if models are trained
    status = ensemble.get_model_status()
    trained = sum(status.values())
    if trained == 0:
        print("[X] No trained models found!")
        print("\nPlease train the ensemble first:")
        print("  py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.train_ensemble")
        return None, None

    print(f"  Loaded {trained}/{len(status)} models")

    # Load feature config
    feature_engineer = FeatureEngineer(config)
    feature_config_path = os.path.join(model_path, 'feature_config.json')
    if os.path.exists(feature_config_path):
        feature_engineer.load_config(feature_config_path)
        print(f"  Loaded feature config ({len(feature_engineer.feature_names)} features)")
    else:
        print("  [!] Feature config not found, using defaults")

    return ensemble, feature_engineer


def main():
    parser = argparse.ArgumentParser(description='Forex ML Paper Trading')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained models')
    parser.add_argument('--interval', type=int, default=60, help='Trading interval in seconds')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FOREX ML PAPER TRADING SYSTEM")
    print("=" * 60)

    # Load configuration
    config = load_config()
    print_config_info(config)

    # Print detailed DCA and position sizing info
    print_dca_position_sizing_info(config)

    # Check credentials
    if not check_credentials(config):
        return

    # Load trained ensemble
    ensemble, feature_engineer = load_trained_ensemble(config, args.model_path)
    if ensemble is None:
        return

    # Confirm before starting
    if not args.no_confirm:
        print("\n" + "-" * 60)
        print("Ready to start paper trading.")
        print(f"  Environment: {config.oanda.environment}")
        print(f"  Symbols: {len(config.symbols)}")
        print(f"  Interval: {args.interval}s")
        print("-" * 60)

        response = input("\nStart paper trading? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return

    # Initialize engine
    engine = PaperTradingEngine(config, ensemble, feature_engineer)

    # Initialize
    if not engine.initialize():
        print("[X] Failed to initialize engine")
        return

    # Run trading loop
    try:
        engine.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Offer to close positions
        if engine.position_manager.get_position_count() > 0:
            print(f"\n{engine.position_manager.get_position_count()} positions still open.")
            response = input("Close all positions? [y/N]: ").strip().lower()
            if response == 'y':
                engine.close_all_positions()
                print("All positions closed.")


if __name__ == '__main__':
    main()
