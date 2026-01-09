"""
Run Backtest for Forex ML System with Master Momentum
======================================================

Tests the Master Momentum + ML strategy on historical data.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_system.Forex_Trading.Forex_Trading_ML_Version.config.trading_config import (
    TradingConfig, load_config, print_config_info
)
from trading_system.Forex_Trading.Forex_Trading_ML_Version.engine.momentum_signal import MasterMomentumSignal
from trading_system.Forex_Trading.Forex_Trading_ML_Version.features.feature_engineer import FeatureEngineer
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ensemble.ensemble_voting import EnsembleVotingSystem
from trading_system.Forex_Trading.Forex_Trading_ML_Version.engine.trading_strategy import MLTradingStrategy


def load_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Load historical data from HistData CSV files."""
    # Map OANDA symbol format to HistData format
    symbol_map = {
        'EUR_USD': 'EURUSD',
        'GBP_USD': 'GBPUSD',
        'USD_JPY': 'USDJPY',
        'USD_CHF': 'USDCHF',
        'AUD_USD': 'AUDUSD',
        'USD_CAD': 'USDCAD',
        'NZD_USD': 'NZDUSD',
        'EUR_GBP': 'EURGBP',
        'EUR_JPY': 'EURJPY',
        'GBP_JPY': 'GBPJPY',
    }

    histdata_symbol = symbol_map.get(symbol, symbol.replace('_', ''))

    # Path to HistData 2024 files
    # __file__ is in Forex_Trading_ML_Version, so .parent.parent gets us to Forex_Trading
    data_dir = Path(__file__).parent.parent / "Backtesting_data_histdata" / "2024"
    csv_file = data_dir / f"DAT_MT_{histdata_symbol}_M1_2024.csv"

    if not csv_file.exists():
        print(f"Data file not found: {csv_file}")
        return None

    print(f"Loading {symbol} data from {csv_file.name}...")

    try:
        # Load CSV with proper column names
        df = pd.read_csv(
            csv_file,
            names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates={'datetime': ['date', 'time']},
            date_format='%Y.%m.%d %H:%M'
        )

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Filter to last N days if specified
        if days > 0 and days < 365:
            cutoff = df.index[-1] - timedelta(days=days)
            df = df[df.index > cutoff]

        print(f"Loaded {len(df):,} candles from {df.index[0]} to {df.index[-1]}")
        return df

    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simple_backtest(config: TradingConfig, symbol: str, df: pd.DataFrame):
    """
    Run a simple backtest using Master Momentum + ML strategy.
    """
    if df is None or len(df) < 200:
        print(f"Insufficient data for {symbol}")
        return None

    print(f"\n{'='*60}")
    print(f"BACKTESTING {symbol}")
    print(f"{'='*60}")
    print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # Initialize components
    momentum_signal = MasterMomentumSignal(config)

    # Skip ML for fast backtest - just test momentum
    ml_available = False
    print("  (ML disabled for fast backtest - testing MOMENTUM only)")

    # Backtest parameters
    initial_capital = 10000
    position_size_pct = 0.02
    commission_pct = 0.00003  # OANDA spread ~3 pips

    # Realistic scalping TP/SL for 1-min data
    # 0.1% = 10 pips, 0.05% = 5 pips
    # Better risk:reward ratio - 1:1.5 (need only 40% win rate to profit)
    take_profit_pct = 0.003   # 0.3% = 30 pips TP
    stop_loss_pct = 0.002     # 0.2% = 20 pips SL

    # DCA settings (not used in simple backtest but kept for reference)
    dca_enabled = config.dca.enabled
    dca_levels = [0.003, 0.006, 0.010, 0.015]  # 3, 6, 10, 15 pips
    dca_multipliers = [1.5, 2.0, 2.5, 3.0]

    # Tracking
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]

    momentum_signals = 0
    momentum_ml_align = 0
    ml_fallback_signals = 0

    # Simulation
    lookback = 200
    total_bars = len(df) - lookback - 1
    progress_interval = max(1, total_bars // 10)

    for i in range(lookback, len(df) - 1):
        # Progress update
        if (i - lookback) % progress_interval == 0:
            pct = (i - lookback) / total_bars * 100
            print(f"  Progress: {pct:.0f}% ({i - lookback}/{total_bars} bars)", end='\r')
        current_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]  # For fill prices
        window = df.iloc[i - lookback:i + 1].copy()

        current_price = current_bar['close']

        # Check for exit if in position
        if position is not None:
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            if position['direction'] == 'SELL':
                pnl_pct = -pnl_pct

            # TP/SL check - use local variables for realistic scalping
            tp_pct = take_profit_pct
            sl_pct = stop_loss_pct

            if pnl_pct >= tp_pct:
                # Take profit
                exit_price = current_price
                gross_pnl = position['units'] * pnl_pct * position['entry_price']
                commission = abs(position['units'] * commission_pct * exit_price)
                net_pnl = gross_pnl - commission

                capital += net_pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_bar.name,
                    'symbol': symbol,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'exit_reason': 'TP',
                    'signal_type': position.get('signal_type', 'unknown')
                })
                position = None

            elif pnl_pct <= -sl_pct:
                # Stop loss
                exit_price = current_price
                gross_pnl = position['units'] * pnl_pct * position['entry_price']
                commission = abs(position['units'] * commission_pct * exit_price)
                net_pnl = gross_pnl - commission

                capital += net_pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_bar.name,
                    'symbol': symbol,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'exit_reason': 'SL',
                    'signal_type': position.get('signal_type', 'unknown')
                })
                position = None

        # Generate signal if no position
        if position is None:
            # Master Momentum check
            mom_signal, mom_conf, mom_reason = momentum_signal.generate_signal(symbol, window)

            # Get momentum direction even if signal not triggered
            mom_summary = momentum_signal.get_trend_summary(window)
            raw_momentum = mom_summary.get('momentum', 0.0)
            mom_direction = mom_summary.get('momentum_direction', 'NONE')
            momentum_threshold = config.momentum.momentum_threshold

            signal = None
            signal_type = None
            confidence = 0.0

            if mom_signal:
                # Pure momentum signal
                signal = mom_signal
                signal_type = 'MOMENTUM'
                confidence = mom_conf
                momentum_signals += 1

            elif ml_available:
                # Check ML fallback
                strong_momentum = abs(raw_momentum) >= momentum_threshold

                try:
                    ml_signal, ml_conf, ml_agree, ml_reason = ml_strategy.generate_signal(symbol, window)

                    if ml_signal:
                        # Check if ML aligns with momentum
                        ml_aligns = (
                            (mom_direction == 'UP' and ml_signal == 'BUY') or
                            (mom_direction == 'DOWN' and ml_signal == 'SELL')
                        )

                        if strong_momentum and ml_aligns:
                            # SUPER STRONG: Momentum + ML align
                            signal = ml_signal
                            signal_type = 'MOMENTUM+ML'
                            confidence = min(0.95, ml_conf + 0.10)
                            momentum_ml_align += 1
                        else:
                            # ML fallback only
                            signal = ml_signal
                            signal_type = 'ML_FALLBACK'
                            confidence = ml_conf
                            ml_fallback_signals += 1
                except:
                    pass

            # Execute trade
            if signal:
                entry_price = next_bar['open']  # Realistic fill
                units = (capital * position_size_pct) / entry_price
                commission = abs(units * commission_pct * entry_price)

                position = {
                    'entry_time': current_bar.name,
                    'direction': signal,
                    'entry_price': entry_price,
                    'units': units,
                    'confidence': confidence,
                    'signal_type': signal_type,
                    'dca_level': 0
                }
                capital -= commission

        equity_curve.append(capital)

    # Close any open position at end
    if position is not None:
        exit_price = df.iloc[-1]['close']
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        if position['direction'] == 'SELL':
            pnl_pct = -pnl_pct
        gross_pnl = position['units'] * pnl_pct * position['entry_price']
        commission = abs(position['units'] * commission_pct * exit_price)
        net_pnl = gross_pnl - commission
        capital += net_pnl
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': net_pnl,
            'exit_reason': 'END',
            'signal_type': position.get('signal_type', 'unknown')
        })

    # Calculate results
    total_trades = len(trades)
    if total_trades == 0:
        print("No trades executed")
        return None

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Print results
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS - {symbol}")
    print(f"{'='*60}")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Total Trades: {total_trades}")
    print(f"  - MOMENTUM signals: {momentum_signals}")
    print(f"  - MOMENTUM+ML aligned: {momentum_ml_align}")
    print(f"  - ML fallback: {ml_fallback_signals}")
    print(f"\nPerformance:")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Return: {(capital - initial_capital) / initial_capital * 100:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    print(f"  Max Drawdown: {max_dd * 100:.2f}%")
    print(f"  Final Capital: ${capital:.2f}")

    # Trade breakdown by signal type
    print(f"\nTrades by Signal Type:")
    for sig_type in ['MOMENTUM', 'MOMENTUM+ML', 'ML_FALLBACK']:
        type_trades = [t for t in trades if t.get('signal_type') == sig_type]
        if type_trades:
            type_wins = len([t for t in type_trades if t['pnl'] > 0])
            type_pnl = sum(t['pnl'] for t in type_trades)
            print(f"  {sig_type}: {len(type_trades)} trades, {type_wins}/{len(type_trades)} wins ({type_wins/len(type_trades)*100:.0f}%), P&L: ${type_pnl:.2f}")

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd
    }


def main():
    """Run backtest on multiple symbols."""
    print("="*60)
    print("FOREX ML + MASTER MOMENTUM BACKTEST")
    print("="*60)

    # Load config
    config = load_config()
    print_config_info(config)

    # Test symbols
    symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY']

    all_results = {}

    for symbol in symbols:
        # Load data - use 30 days for better sample size
        df = load_historical_data(symbol, days=30)

        if df is not None and len(df) > 200:
            results = run_simple_backtest(config, symbol, df)
            if results:
                all_results[symbol] = results
        else:
            print(f"Skipping {symbol} - insufficient data")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")

        total_pnl = sum(r['total_pnl'] for r in all_results.values())
        avg_win_rate = np.mean([r['win_rate'] for r in all_results.values()])
        avg_pf = np.mean([r['profit_factor'] for r in all_results.values()])

        print(f"Symbols tested: {len(all_results)}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Avg Win Rate: {avg_win_rate:.1f}%")
        print(f"Avg Profit Factor: {avg_pf:.2f}")


if __name__ == "__main__":
    main()
