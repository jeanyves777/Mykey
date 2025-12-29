"""
Run Backtest for High-Frequency Crypto Trading System
=======================================================

This script:
1. Loads trained ensemble model
2. Loads historical data
3. Generates ML signals
4. Runs backtest with realistic execution
5. Produces performance report
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.features import FeatureEngineer
from trading_system.High_frequency_crypto_tradin.ensemble import EnsembleVotingSystem
from trading_system.High_frequency_crypto_tradin.backtest import BacktestEngine, BacktestConfig
from trading_system.High_frequency_crypto_tradin.config import load_config
from trading_system.High_frequency_crypto_tradin.train_hf_scalping import add_market_regime_features


def load_trained_model(model_dir: str, config=None):
    """Load trained ensemble model."""
    print("Loading trained model...")

    ensemble = EnsembleVotingSystem()
    ensemble.load(str(Path(model_dir) / "ensemble"))

    # Override threshold from config if provided
    if config is not None:
        buy_prob_thresh = getattr(config.ensemble, 'buy_probability_threshold', 0.38)
        ensemble.buy_probability_threshold = buy_prob_thresh
        print(f"  BUY probability threshold: {buy_prob_thresh:.2f}")

    fe = FeatureEngineer()
    fe.load_feature_config(str(Path(model_dir) / "feature_config.json"))

    print(f"  Model loaded with {len(fe.feature_columns)} features")

    return ensemble, fe


def prepare_backtest_data(data_path: str, fe: FeatureEngineer):
    """Prepare data for backtesting.

    CRITICAL: Features must be shifted by 1 bar to match training!
    This prevents look-ahead bias during inference.
    """
    print(f"\nLoading data from {data_path}...")

    df = fe.load_data(data_path)
    print(f"  Rows: {len(df)}")

    # Drop string columns that cause issues
    for col in ['conversionType', 'conversionSymbol']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Compute features
    print("  Computing features...")
    df_features = fe.compute_all_features(df)

    # SHIFT ALL FEATURES BY 1 BAR to prevent look-ahead bias
    # This must match what was done during training!
    no_shift_cols = [
        'timestamp', 'open_time', 'close_time', 'datetime', 'date', 'symbol',
        'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'
    ]
    feature_cols = [col for col in df_features.columns if col not in no_shift_cols]
    print(f"  Shifting {len(feature_cols)} features by 1 bar...")
    for col in feature_cols:
        df_features[col] = df_features[col].shift(1)

    # Add market regime features (already shifted in the function)
    print("  Adding market regime features...")
    df_features = add_market_regime_features(df_features)

    df_features = fe.handle_missing_values(df_features)

    return df, df_features


def generate_signals(ensemble, df_features: pd.DataFrame, fe: FeatureEngineer):
    """Generate trading signals using the ensemble."""
    print("\nGenerating signals...")

    # Get feature columns
    feature_cols = fe.feature_columns
    X = df_features[feature_cols]

    # Generate predictions
    predictions = ensemble.predict(X)
    probas = ensemble.predict_proba(X)
    confidences = np.max(probas, axis=1)

    print(f"  Total predictions: {len(predictions)}")
    print(f"  Signal distribution (binary: 0=NO_TRADE, 1=BUY):")
    print(f"    BUY (1): {np.sum(predictions == 1)}")
    print(f"    NO_TRADE (0): {np.sum(predictions == 0)}")

    return predictions, confidences


def run_backtest(df: pd.DataFrame, signals: np.ndarray, confidences: np.ndarray,
                 config, symbol: str = "CRYPTO"):
    """Run HF SCALPING backtest - tight TP/SL, quick exits."""
    print("\n" + "=" * 60)
    print("Running Backtest - HF SCALPING Strategy")
    print("=" * 60)
    print(f"TP: {config.strategy.take_profit_pct*100:.2f}% | SL: {config.strategy.stop_loss_pct*100:.2f}% | Trail: {config.strategy.trailing_stop_pct*100:.2f}%")
    buy_prob_thresh = getattr(config.ensemble, 'buy_probability_threshold', 0.22)
    print(f"BUY Prob Threshold: {buy_prob_thresh*100:.0f}% | Cooldown: {config.strategy.cooldown_seconds}s")

    # Create backtest config for HF scalping
    bt_config = BacktestConfig(
        initial_capital=config.backtest.initial_capital,
        position_size_pct=config.strategy.position_size_pct,
        max_position_size=config.strategy.max_position_value,
        commission_rate=config.backtest.commission_rate,
        slippage_pct=config.backtest.slippage_pct,
        # HF Scalping settings
        take_profit_pct=config.strategy.take_profit_pct,
        stop_loss_pct=config.strategy.stop_loss_pct,
        trailing_stop_pct=config.strategy.trailing_stop_pct,
        # DCA disabled
        dca_enabled=False,
        max_dca_entries=0,
        # Other settings
        min_confidence=config.ensemble.confidence_threshold,
        cooldown_bars=max(1, config.strategy.cooldown_seconds // 60),
        max_daily_trades=config.strategy.max_trades_per_day,
        max_daily_loss_pct=config.strategy.max_daily_loss_pct
    )

    # Create engine and run
    engine = BacktestEngine(bt_config, use_centralized_risk=False)
    results = engine.run(df, signals, confidences, symbol=symbol, verbose=True)

    return results, engine


def analyze_results(results: dict, engine):
    """Analyze and display backtest results."""
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)

    # Trade analysis
    trades_df = engine.get_trades_df()
    if not trades_df.empty:
        print("\nTrade Analysis:")
        print(f"  Total Trades: {len(trades_df)}")

        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]

        print(f"  Winning Trades: {len(winning)} ({len(winning)/len(trades_df)*100:.1f}%)")
        print(f"  Losing Trades: {len(losing)} ({len(losing)/len(trades_df)*100:.1f}%)")

        if len(winning) > 0:
            print(f"\n  Avg Win: ${winning['pnl'].mean():,.2f}")
            print(f"  Max Win: ${winning['pnl'].max():,.2f}")

        if len(losing) > 0:
            print(f"\n  Avg Loss: ${abs(losing['pnl'].mean()):,.2f}")
            print(f"  Max Loss: ${abs(losing['pnl'].min()):,.2f}")

        print(f"\n  Avg Holding Time: {trades_df['holding_bars'].mean():.1f} bars")

    # Monthly breakdown
    print("\n" + "-" * 40)
    print("Performance by Trade Side:")

    buy_trades = trades_df[trades_df['side'] == 'BUY']
    sell_trades = trades_df[trades_df['side'] == 'SELL']

    if len(buy_trades) > 0:
        buy_pnl = buy_trades['pnl'].sum()
        buy_wr = len(buy_trades[buy_trades['pnl'] > 0]) / len(buy_trades) * 100
        print(f"  BUY: {len(buy_trades)} trades, P&L: ${buy_pnl:,.2f}, Win Rate: {buy_wr:.1f}%")

    if len(sell_trades) > 0:
        sell_pnl = sell_trades['pnl'].sum()
        sell_wr = len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) * 100
        print(f"  SELL: {len(sell_trades)} trades, P&L: ${sell_pnl:,.2f}, Win Rate: {sell_wr:.1f}%")

    return trades_df


def save_backtest_results(results: dict, trades_df: pd.DataFrame, output_dir: str):
    """Save backtest results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary_file = output_path / f"backtest_summary_{timestamp}.json"
    import json
    with open(summary_file, 'w') as f:
        # Convert numpy types to Python types
        save_results = {k: v for k, v in results.items()
                       if k not in ['equity_curve', 'trades']}
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to {summary_file}")

    # Save trades
    if not trades_df.empty:
        trades_file = output_path / f"trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Trades saved to {trades_file}")


def main():
    """Main backtest script."""
    print("=" * 60)
    print("High-Frequency Crypto Trading - Backtest")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Load configuration
    config = load_config()

    # Check for trained model - Use HF scalping model
    # Priority: saved_models_hf_scalping > config.model_save_dir
    hf_model_dir = Path(__file__).parent / "saved_models_hf_scalping"
    if (hf_model_dir / "ensemble").exists():
        model_dir = hf_model_dir
        print(f"Using HF SCALPING model: {model_dir}")
    else:
        model_dir = Path(config.model_save_dir)

    if not (model_dir / "ensemble").exists():
        print("\nERROR: No trained model found!")
        print("Please run train_hf_scalping.py first.")
        return

    # Load trained model (pass config for threshold override)
    ensemble, fe = load_trained_model(str(model_dir), config)

    # Get data file
    data_dir = Path(config.data_dir)

    # Run backtest on each symbol
    all_results = {}

    # All 5 trained symbols
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'AVAXUSD']

    for symbol in symbols:
        data_file = data_dir / f"{symbol}_1m.csv"

        if not data_file.exists():
            print(f"\nWarning: {data_file} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Backtesting: {symbol}")
        print("=" * 60)

        # Prepare data
        df, df_features = prepare_backtest_data(str(data_file), fe)

        # Generate signals
        signals, confidences = generate_signals(ensemble, df_features, fe)

        # Run backtest
        results, engine = run_backtest(df, signals, confidences, config, symbol)

        # Analyze results
        trades_df = analyze_results(results, engine)

        # Store results
        all_results[symbol] = {
            'results': results,
            'trades_df': trades_df
        }

        # Save results
        output_dir = Path(config.log_dir) / "backtest_results"
        save_backtest_results(results, trades_df, str(output_dir))

    # Summary across all symbols
    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)

    total_pnl = 0
    total_trades = 0

    for symbol, data in all_results.items():
        pnl = data['results']['total_pnl']
        trades = data['results']['total_trades']
        total_pnl += pnl
        total_trades += trades
        print(f"{symbol}: {trades} trades, P&L: ${pnl:,.2f}")

    print(f"\nTotal: {total_trades} trades, Total P&L: ${total_pnl:,.2f}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
