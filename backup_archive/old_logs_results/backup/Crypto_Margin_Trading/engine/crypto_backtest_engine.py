"""
Crypto Backtesting Engine
=========================
Backtesting engine for crypto margin trading strategy optimization.
Uses historical data from Binance CSV files.

Features:
- Fast backtesting on 100K+ candles
- Strategy parameter optimization
- Walk-forward validation
- Detailed performance metrics
- Multi-pair optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import itertools
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.crypto_margin_strategy import (
    calculate_indicators,
    check_rsi_reversal_signal,
    check_rsi_30_70_signal,
    check_rsi_extreme_signal,
    check_macd_cross_signal,
    check_macd_zero_signal,
    check_ema_pullback_signal,
    check_rsi_macd_combo_signal,
    check_triple_confirm_signal,
    check_volume_filter,
    check_trend_filter,
    check_volatility_filter,
)


class CryptoBacktestEngine:
    """Backtesting engine for crypto strategies"""

    STRATEGIES = {
        'RSI_REVERSAL': check_rsi_reversal_signal,
        'RSI_30_70': lambda df: check_rsi_30_70_signal(df, 30, 70),
        'RSI_EXTREME': check_rsi_extreme_signal,
        'MACD_CROSS': check_macd_cross_signal,
        'MACD_ZERO': check_macd_zero_signal,
        'EMA_PULLBACK': check_ema_pullback_signal,
        'RSI_MACD_COMBO': check_rsi_macd_combo_signal,
        'TRIPLE_CONFIRM': check_triple_confirm_signal,
    }

    def __init__(self, data_dir: str = None):
        """
        Initialize backtest engine.

        Args:
            data_dir: Directory containing CSV data files
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / "Crypto_Data_from_Binance"

        print(f"[Backtest] Data directory: {self.data_dir}")

    def load_data(self, pair: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """
        Load historical data from CSV.

        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe

        Returns:
            DataFrame with OHLCV data
        """
        filename = f"{pair}_{interval}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            print(f"[Backtest] Data file not found: {filepath}")
            return None

        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])

        print(f"[Backtest] Loaded {len(df):,} candles for {pair}")
        print(f"  Range: {df['datetime'].min()} to {df['datetime'].max()}")

        return df

    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy: str,
        tp_pct: float,
        sl_pct: float,
        leverage: int = 1,
        volume_filter: bool = False,
        trend_filter: bool = False,
        volatility_filter: bool = False,
        max_atr_pct: float = 3.0,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 2.0,
    ) -> Dict:
        """
        Run a single backtest with given parameters.

        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy name
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            leverage: Leverage multiplier
            volume_filter: Enable volume filter
            trend_filter: Enable trend filter
            volatility_filter: Enable volatility filter
            max_atr_pct: Max ATR percentage for volatility filter
            initial_balance: Starting balance
            risk_per_trade: Risk percentage per trade

        Returns:
            Backtest results dictionary
        """
        if strategy not in self.STRATEGIES:
            return {'error': f"Unknown strategy: {strategy}"}

        # Calculate indicators
        df = calculate_indicators(df)

        # Get strategy function
        strategy_func = self.STRATEGIES[strategy]

        # Tracking variables
        balance = initial_balance
        trades = []
        position = None
        equity_curve = []

        # Iterate through candles
        for i in range(50, len(df)):  # Start after indicator warmup
            current = df.iloc[i]
            current_price = current['close']
            current_time = current['datetime']

            # Track equity
            if position:
                if position['direction'] == 'BUY':
                    unrealized = (current_price - position['entry_price']) / position['entry_price'] * 100 * leverage
                else:
                    unrealized = (position['entry_price'] - current_price) / position['entry_price'] * 100 * leverage
                current_equity = balance * (1 + unrealized * position['risk_pct'] / 100)
            else:
                current_equity = balance

            equity_curve.append({'time': current_time, 'equity': current_equity})

            # If in position, check for exit
            if position:
                entry_price = position['entry_price']
                direction = position['direction']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']

                exit_price = None
                exit_reason = None

                # Check high/low for TP/SL (use candle extremes for accuracy)
                high = current['high']
                low = current['low']

                if direction == 'BUY':
                    # Check SL first (assume worst case)
                    if low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'SL'
                    elif high >= take_profit:
                        exit_price = take_profit
                        exit_reason = 'TP'
                else:  # SELL
                    if high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'SL'
                    elif low <= take_profit:
                        exit_price = take_profit
                        exit_reason = 'TP'

                # Close position if exit triggered
                if exit_price:
                    if direction == 'BUY':
                        pnl_pct = (exit_price - entry_price) / entry_price * 100 * leverage
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100 * leverage

                    pnl = balance * (position['risk_pct'] / 100) * (pnl_pct / sl_pct)
                    balance += pnl

                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'balance': balance,
                    })

                    position = None

            # If not in position, check for entry signal
            if position is None:
                # Get signal from strategy
                df_slice = df.iloc[max(0, i-99):i+1].copy()
                signal, reason = strategy_func(df_slice)

                if signal:
                    # Apply filters
                    if volume_filter and not check_volume_filter(df_slice):
                        continue
                    if trend_filter and not check_trend_filter(df_slice, signal):
                        continue
                    if volatility_filter and not check_volatility_filter(df_slice, max_atr_pct):
                        continue

                    # Calculate TP/SL
                    if signal == 'BUY':
                        take_profit = current_price * (1 + tp_pct / 100)
                        stop_loss = current_price * (1 - sl_pct / 100)
                    else:  # SELL
                        take_profit = current_price * (1 - tp_pct / 100)
                        stop_loss = current_price * (1 + sl_pct / 100)

                    position = {
                        'direction': signal,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'risk_pct': risk_per_trade,
                    }

        # Close any remaining position at last price
        if position:
            last_price = df.iloc[-1]['close']
            direction = position['direction']
            entry_price = position['entry_price']

            if direction == 'BUY':
                pnl_pct = (last_price - entry_price) / entry_price * 100 * leverage
            else:
                pnl_pct = (entry_price - last_price) / entry_price * 100 * leverage

            pnl = balance * (position['risk_pct'] / 100) * (pnl_pct / sl_pct)
            balance += pnl

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.iloc[-1]['datetime'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': last_price,
                'exit_reason': 'END',
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'balance': balance,
            })

        # Calculate metrics
        total_trades = len(trades)
        if total_trades == 0:
            return {
                'strategy': strategy,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'return_pct': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
            }

        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = sum(1 for t in trades if t['pnl'] < 0)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = sum(t['pnl'] for t in trades)
        return_pct = ((balance / initial_balance) - 1) * 100

        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity_values = [e['equity'] for e in equity_curve]
        peak = equity_values[0]
        max_dd = 0
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return {
            'strategy': strategy,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'leverage': leverage,
            'volume_filter': volume_filter,
            'trend_filter': trend_filter,
            'volatility_filter': volatility_filter,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round(return_pct, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_dd, 2),
            'avg_trade': round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
            'final_balance': round(balance, 2),
        }

    def optimize(
        self,
        df: pd.DataFrame,
        strategies: List[str] = None,
        tp_range: List[float] = None,
        sl_range: List[float] = None,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Optimize strategy parameters.

        Args:
            df: DataFrame with OHLCV data
            strategies: List of strategies to test
            tp_range: Take profit values to test
            sl_range: Stop loss values to test
            top_n: Number of top results to return

        Returns:
            List of top performing parameter combinations
        """
        if strategies is None:
            strategies = list(self.STRATEGIES.keys())
        if tp_range is None:
            tp_range = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        if sl_range is None:
            sl_range = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]

        results = []
        total_tests = len(strategies) * len(tp_range) * len(sl_range)
        test_num = 0

        print(f"\n[Optimize] Running {total_tests} parameter combinations...")

        for strategy in strategies:
            for tp in tp_range:
                for sl in sl_range:
                    test_num += 1

                    result = self.run_backtest(
                        df=df,
                        strategy=strategy,
                        tp_pct=tp,
                        sl_pct=sl,
                    )

                    # Only include if has trades and positive profit factor
                    if result['total_trades'] >= 10 and result['profit_factor'] > 0:
                        results.append(result)

                    if test_num % 50 == 0:
                        print(f"  Progress: {test_num}/{total_tests} ({test_num/total_tests*100:.1f}%)")

        # Sort by profit factor (or custom metric)
        results.sort(key=lambda x: (x['profit_factor'], x['win_rate']), reverse=True)

        print(f"\n[Optimize] Found {len(results)} valid combinations")

        return results[:top_n]

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        strategy: str,
        tp_pct: float,
        sl_pct: float,
        train_pct: float = 0.7,
        **kwargs
    ) -> Dict:
        """
        Perform walk-forward validation.

        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy name
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            train_pct: Percentage of data for training
            **kwargs: Additional backtest parameters

        Returns:
            Validation results
        """
        # Split data
        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\n[Walk-Forward] Train: {len(train_df):,} candles, Test: {len(test_df):,} candles")

        # Run on training data
        train_result = self.run_backtest(
            df=train_df,
            strategy=strategy,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            **kwargs
        )

        # Run on test data
        test_result = self.run_backtest(
            df=test_df,
            strategy=strategy,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            **kwargs
        )

        # Compare results
        validation_passed = (
            test_result['win_rate'] >= train_result['win_rate'] * 0.8 and  # 80% of training WR
            test_result['profit_factor'] > 1.0  # Still profitable
        )

        return {
            'strategy': strategy,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'train_results': train_result,
            'test_results': test_result,
            'validation_passed': validation_passed,
            'train_wr': train_result['win_rate'],
            'test_wr': test_result['win_rate'],
            'train_pf': train_result['profit_factor'],
            'test_pf': test_result['profit_factor'],
        }

    def optimize_pair(
        self,
        pair: str,
        interval: str = '1m',
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Optimize parameters for a single pair.

        Args:
            pair: Trading pair
            interval: Timeframe
            top_n: Number of top results

        Returns:
            Top optimization results with walk-forward validation
        """
        print(f"\n{'='*70}")
        print(f"OPTIMIZING {pair}")
        print(f"{'='*70}")

        # Load data
        df = self.load_data(pair, interval)
        if df is None or len(df) < 1000:
            print(f"[Error] Insufficient data for {pair}")
            return []

        # Run optimization
        top_results = self.optimize(df, top_n=top_n)

        # Walk-forward validation on top results
        validated_results = []
        for result in top_results:
            validation = self.walk_forward_validation(
                df=df,
                strategy=result['strategy'],
                tp_pct=result['tp_pct'],
                sl_pct=result['sl_pct'],
            )
            result['walk_forward'] = validation
            result['validated'] = validation['validation_passed']
            validated_results.append(result)

        # Print results
        print(f"\n{'TOP RESULTS FOR ' + pair:^70}")
        print(f"{'-'*70}")
        print(f"{'STRATEGY':<16} {'TP%':<6} {'SL%':<6} {'TRADES':<8} {'WIN%':<8} {'PF':<8} {'VALID':<8}")
        print(f"{'-'*70}")

        for r in validated_results:
            valid_str = "YES" if r['validated'] else "NO"
            print(f"{r['strategy']:<16} {r['tp_pct']:<6.1f} {r['sl_pct']:<6.1f} {r['total_trades']:<8} {r['win_rate']:<8.1f} {r['profit_factor']:<8.2f} {valid_str:<8}")

        return validated_results

    def optimize_all_pairs(
        self,
        pairs: List[str],
        interval: str = '1m',
        output_file: str = 'optimization_results.json',
    ) -> Dict:
        """
        Optimize all trading pairs and save results.

        Args:
            pairs: List of trading pairs
            interval: Timeframe
            output_file: Output JSON file

        Returns:
            All optimization results
        """
        all_results = {}

        for pair in pairs:
            results = self.optimize_pair(pair, interval)
            all_results[pair] = results

        # Save results
        output_path = self.data_dir.parent / output_file
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n[Optimize] Results saved to {output_path}")

        # Print summary
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"{'PAIR':<12} {'BEST STRATEGY':<16} {'TP%':<6} {'SL%':<6} {'WIN%':<8} {'VALID':<8}")
        print(f"{'-'*70}")

        for pair, results in all_results.items():
            if results:
                best = results[0]
                valid_str = "YES" if best.get('validated', False) else "NO"
                print(f"{pair:<12} {best['strategy']:<16} {best['tp_pct']:<6.1f} {best['sl_pct']:<6.1f} {best['win_rate']:<8.1f} {valid_str:<8}")
            else:
                print(f"{pair:<12} NO VALID RESULTS")

        print(f"{'='*70}")

        return all_results


def main():
    """Run optimization on all pairs"""
    engine = CryptoBacktestEngine()

    # Pairs to optimize
    pairs = [
        'BTCUSDT',
        'ETHUSDT',
        'SOLUSDT',
        'XRPUSDT',
        'DOGEUSDT',
        'LTCUSDT',
        'ADAUSDT',
        'LINKUSDT',
    ]

    # Run optimization
    engine.optimize_all_pairs(pairs)


if __name__ == "__main__":
    main()
