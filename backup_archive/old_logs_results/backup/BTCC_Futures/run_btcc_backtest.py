#!/usr/bin/env python3
"""
BTCC Futures Comprehensive Backtest
===================================
Backtest all strategies with different leverage levels (up to 500x)
using existing Binance data.

BTCC Leverage Tiers:
- BTCUSDT: up to 500x
- ETHUSDT: up to 500x
- SOLUSDT: up to 500x
- XRPUSDT: up to 500x
- PIPPINUSDT: up to 50x
- LUNAUSDT: up to 50x
- ZECUSDT: up to 50x
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.btcc_strategies import (
    TechnicalIndicators, OHLCV,
    ScalpingMomentumStrategy, RSIReversalStrategy,
    BreakoutStrategy, EMACrossoverStrategy, BollingerSqueezeStrategy
)

# Data path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    symbol: str
    leverage: int
    strategy: str
    tp_pct: float  # Take profit %
    sl_pct: float  # Stop loss %
    position_size: float = 100.0  # Position size in USDT
    commission_pct: float = 0.06  # Commission per side (0.06%)


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1=Long, 2=Short
    entry_price: float
    exit_price: float
    volume: float
    leverage: int
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0  # in minutes


class BTCCBacktester:
    """BTCC Futures Backtester"""

    # BTCC symbol configurations with max leverage
    BTCC_SYMBOLS = {
        'BTCUSDT': {'max_leverage': 500, 'min_volume': 0.001, 'digits': 2},
        'ETHUSDT': {'max_leverage': 500, 'min_volume': 0.01, 'digits': 2},
        'SOLUSDT': {'max_leverage': 500, 'min_volume': 0.1, 'digits': 3},
        'XRPUSDT': {'max_leverage': 500, 'min_volume': 1.0, 'digits': 4},
        'LTCUSDT': {'max_leverage': 200, 'min_volume': 0.01, 'digits': 2},
        'LINKUSDT': {'max_leverage': 100, 'min_volume': 0.1, 'digits': 3},
        'ADAUSDT': {'max_leverage': 100, 'min_volume': 1.0, 'digits': 4},
        'DOGEUSDT': {'max_leverage': 100, 'min_volume': 10.0, 'digits': 5},
        'AVAXUSDT': {'max_leverage': 100, 'min_volume': 0.1, 'digits': 3},
        'DOTUSDT': {'max_leverage': 100, 'min_volume': 0.1, 'digits': 3},
        'UNIUSDT': {'max_leverage': 50, 'min_volume': 0.1, 'digits': 3},
        'BCHUSDT': {'max_leverage': 100, 'min_volume': 0.01, 'digits': 2},
        'XLMUSDT': {'max_leverage': 50, 'min_volume': 1.0, 'digits': 4},
        'ZECUSDT': {'max_leverage': 50, 'min_volume': 0.01, 'digits': 2},
        'XMRUSDT': {'max_leverage': 50, 'min_volume': 0.01, 'digits': 2},
    }

    # Strategies to test
    STRATEGIES = {
        'SCALPING_MOMENTUM': ScalpingMomentumStrategy,
        'RSI_REVERSAL': RSIReversalStrategy,
        'BREAKOUT': BreakoutStrategy,
        'EMA_CROSSOVER': EMACrossoverStrategy,
        'BOLLINGER_SQUEEZE': BollingerSqueezeStrategy,
    }

    # Leverage levels to test
    LEVERAGE_TIERS = [5, 10, 20, 50, 100, 150, 200, 300, 500]

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load candle data for symbol."""
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        file_path = os.path.join(self.data_path, f"{symbol}_1m.csv")
        if not os.path.exists(file_path):
            print(f"Data file not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            # Ensure proper column names
            if 'timestamp' not in df.columns and 'open_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            self.data_cache[symbol] = df
            return df

        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

    def df_to_candles(self, df: pd.DataFrame) -> List[OHLCV]:
        """Convert DataFrame to list of OHLCV objects."""
        candles = []
        for _, row in df.iterrows():
            candles.append(OHLCV(
                timestamp=row.get('timestamp', datetime.now()),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            ))
        return candles

    def run_backtest(self, config: BacktestConfig, df: pd.DataFrame) -> BacktestResult:
        """Run backtest for a single configuration."""
        trades = []
        position = None
        entry_time = None
        entry_price = 0.0
        direction = 0
        stop_loss = 0.0
        take_profit = 0.0

        # Get strategy
        strategy_class = self.STRATEGIES.get(config.strategy)
        if not strategy_class:
            return BacktestResult(config=config, trades=[])

        strategy = strategy_class()

        # Convert to candles for strategy evaluation
        candles = self.df_to_candles(df)

        # Walk through data
        lookback = 100  # Candles needed for indicators

        for i in range(lookback, len(candles)):
            current_candle = candles[i]
            current_price = current_candle.close
            current_high = current_candle.high
            current_low = current_candle.low

            # Check if we have a position
            if position is not None:
                # Check stop loss / take profit
                exit_price = None
                exit_reason = None

                if direction == 1:  # Long
                    if current_low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "Stop Loss"
                    elif current_high >= take_profit:
                        exit_price = take_profit
                        exit_reason = "Take Profit"
                else:  # Short
                    if current_high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = "Stop Loss"
                    elif current_low <= take_profit:
                        exit_price = take_profit
                        exit_reason = "Take Profit"

                if exit_price:
                    # Calculate P/L
                    if direction == 1:
                        pnl_pct = ((exit_price - entry_price) / entry_price) * config.leverage
                    else:
                        pnl_pct = ((entry_price - exit_price) / entry_price) * config.leverage

                    # Subtract commission (both sides)
                    pnl_pct -= (config.commission_pct / 100) * 2

                    pnl = config.position_size * (pnl_pct / 100)

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_candle.timestamp,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        volume=config.position_size / entry_price,
                        leverage=config.leverage,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)

                    position = None
                    continue

            # No position - look for entry signal
            if position is None:
                # Get recent candles for strategy
                recent_candles = candles[i - lookback:i + 1]

                signal = strategy.evaluate(recent_candles, current_price)

                if signal and signal.get('strength', 0) > 0.3:  # Min signal strength
                    direction = signal['direction']
                    entry_price = current_price
                    entry_time = current_candle.timestamp

                    # Calculate SL/TP based on direction
                    if direction == 1:  # Long
                        stop_loss = entry_price * (1 - config.sl_pct / 100)
                        take_profit = entry_price * (1 + config.tp_pct / 100)
                    else:  # Short
                        stop_loss = entry_price * (1 + config.sl_pct / 100)
                        take_profit = entry_price * (1 - config.tp_pct / 100)

                    position = True

        # Calculate results
        result = self._calculate_results(config, trades)
        return result

    def _calculate_results(self, config: BacktestConfig, trades: List[Trade]) -> BacktestResult:
        """Calculate backtest statistics."""
        if not trades:
            return BacktestResult(config=config, trades=trades)

        total_pnl = sum(t.pnl for t in trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0

        # Calculate drawdown
        equity = [config.position_size]
        for t in trades:
            equity.append(equity[-1] + t.pnl)

        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Average trade duration
        durations = []
        for t in trades:
            if isinstance(t.entry_time, datetime) and isinstance(t.exit_time, datetime):
                duration = (t.exit_time - t.entry_time).total_seconds() / 60
                durations.append(duration)

        avg_duration = np.mean(durations) if durations else 0

        return BacktestResult(
            config=config,
            trades=trades,
            total_pnl=total_pnl,
            total_return_pct=(total_pnl / config.position_size) * 100,
            win_rate=(len(winning_trades) / len(trades) * 100) if trades else 0,
            profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float('inf'),
            max_drawdown=max_dd,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=(gross_profit / len(winning_trades)) if winning_trades else 0,
            avg_loss=(gross_loss / len(losing_trades)) if losing_trades else 0,
            largest_win=max(t.pnl for t in trades) if trades else 0,
            largest_loss=min(t.pnl for t in trades) if trades else 0,
            avg_trade_duration=avg_duration,
        )

    def run_full_backtest(self, symbols: List[str] = None,
                          strategies: List[str] = None,
                          leverage_levels: List[int] = None,
                          tp_sl_combos: List[Tuple[float, float]] = None) -> List[BacktestResult]:
        """Run comprehensive backtest across all combinations."""
        symbols = symbols or list(self.BTCC_SYMBOLS.keys())
        strategies = strategies or list(self.STRATEGIES.keys())
        leverage_levels = leverage_levels or [10, 20, 50, 100, 200]
        tp_sl_combos = tp_sl_combos or [
            (2.0, 1.0), (3.0, 1.5), (4.0, 2.0), (5.0, 2.5), (6.0, 3.0)
        ]

        results = []
        total_tests = len(symbols) * len(strategies) * len(leverage_levels) * len(tp_sl_combos)
        test_num = 0

        print(f"\nRunning {total_tests} backtest combinations...")
        print("=" * 80)

        for symbol in symbols:
            # Load data
            df = self.load_data(symbol)
            if df is None or len(df) < 1000:
                print(f"Skipping {symbol} - insufficient data")
                continue

            symbol_config = self.BTCC_SYMBOLS.get(symbol, {'max_leverage': 100})
            max_leverage = symbol_config['max_leverage']

            for strategy in strategies:
                for leverage in leverage_levels:
                    # Skip if leverage exceeds max for this symbol
                    if leverage > max_leverage:
                        continue

                    for tp_pct, sl_pct in tp_sl_combos:
                        test_num += 1

                        config = BacktestConfig(
                            symbol=symbol,
                            leverage=leverage,
                            strategy=strategy,
                            tp_pct=tp_pct,
                            sl_pct=sl_pct,
                        )

                        result = self.run_backtest(config, df)
                        results.append(result)

                        if test_num % 50 == 0:
                            print(f"Progress: {test_num}/{total_tests} tests completed...")

        print(f"\nCompleted {len(results)} backtests")
        return results


def print_results_summary(results: List[BacktestResult]):
    """Print summary of backtest results."""
    if not results:
        print("No results to display")
        return

    # Filter profitable results
    profitable = [r for r in results if r.total_pnl > 0 and r.total_trades >= 10]

    print("\n" + "=" * 100)
    print("BTCC FUTURES BACKTEST RESULTS SUMMARY")
    print("=" * 100)

    print(f"\nTotal Configurations Tested: {len(results)}")
    print(f"Profitable Configurations: {len(profitable)}")
    print(f"Success Rate: {len(profitable)/len(results)*100:.1f}%")

    # Best results by total P/L
    print("\n" + "-" * 100)
    print("TOP 20 CONFIGURATIONS BY TOTAL P/L")
    print("-" * 100)
    print(f"{'SYMBOL':<12} {'STRATEGY':<20} {'LEV':<5} {'TP%':<5} {'SL%':<5} "
          f"{'TRADES':<7} {'WIN%':<7} {'PF':<6} {'P/L':<12} {'RET%':<8} {'MAX DD':<8}")
    print("-" * 100)

    sorted_results = sorted(results, key=lambda x: x.total_pnl, reverse=True)[:20]

    for r in sorted_results:
        print(f"{r.config.symbol:<12} {r.config.strategy:<20} {r.config.leverage:<5} "
              f"{r.config.tp_pct:<5.1f} {r.config.sl_pct:<5.1f} "
              f"{r.total_trades:<7} {r.win_rate:<7.1f} {r.profit_factor:<6.2f} "
              f"${r.total_pnl:<11.2f} {r.total_return_pct:<8.1f} {r.max_drawdown:<8.1f}%")

    # Best by leverage tier
    print("\n" + "-" * 100)
    print("BEST RESULTS BY LEVERAGE TIER")
    print("-" * 100)

    for lev in [10, 20, 50, 100, 200, 500]:
        lev_results = [r for r in results if r.config.leverage == lev and r.total_trades >= 5]
        if lev_results:
            best = max(lev_results, key=lambda x: x.total_pnl)
            print(f"\n{lev}x Leverage - Best: {best.config.symbol} / {best.config.strategy}")
            print(f"    P/L: ${best.total_pnl:.2f} | Win Rate: {best.win_rate:.1f}% | "
                  f"Trades: {best.total_trades} | Max DD: {best.max_drawdown:.1f}%")

    # Best by symbol
    print("\n" + "-" * 100)
    print("BEST RESULTS BY SYMBOL")
    print("-" * 100)

    symbols_seen = set()
    for r in sorted_results:
        if r.config.symbol not in symbols_seen and r.total_trades >= 5:
            symbols_seen.add(r.config.symbol)
            print(f"\n{r.config.symbol}:")
            print(f"    Strategy: {r.config.strategy} | Leverage: {r.config.leverage}x")
            print(f"    TP/SL: {r.config.tp_pct}%/{r.config.sl_pct}%")
            print(f"    P/L: ${r.total_pnl:.2f} | Win Rate: {r.win_rate:.1f}% | PF: {r.profit_factor:.2f}")

    # Risk analysis for high leverage
    print("\n" + "-" * 100)
    print("HIGH LEVERAGE RISK ANALYSIS (100x+)")
    print("-" * 100)

    high_lev = [r for r in results if r.config.leverage >= 100 and r.total_trades >= 5]
    if high_lev:
        avg_dd = np.mean([r.max_drawdown for r in high_lev])
        profitable_high = len([r for r in high_lev if r.total_pnl > 0])
        liquidation_risk = len([r for r in high_lev if r.max_drawdown > 50])

        print(f"Configurations tested: {len(high_lev)}")
        print(f"Profitable: {profitable_high} ({profitable_high/len(high_lev)*100:.1f}%)")
        print(f"Average Max Drawdown: {avg_dd:.1f}%")
        print(f"High Liquidation Risk (DD>50%): {liquidation_risk} ({liquidation_risk/len(high_lev)*100:.1f}%)")

    print("\n" + "=" * 100)


def main():
    """Run comprehensive BTCC backtest."""
    print("\n" + "=" * 80)
    print("BTCC FUTURES COMPREHENSIVE BACKTEST")
    print("Testing all strategies with leverage up to 500x")
    print("=" * 80)

    backtester = BTCCBacktester()

    # Check available data
    print("\nChecking available data...")
    available_symbols = []
    for symbol in backtester.BTCC_SYMBOLS.keys():
        df = backtester.load_data(symbol)
        if df is not None and len(df) > 1000:
            available_symbols.append(symbol)
            print(f"  {symbol}: {len(df):,} candles")

    if not available_symbols:
        print("No data available for backtesting!")
        return

    # Run backtests
    results = backtester.run_full_backtest(
        symbols=available_symbols,
        strategies=['SCALPING_MOMENTUM', 'RSI_REVERSAL', 'BREAKOUT', 'EMA_CROSSOVER', 'BOLLINGER_SQUEEZE'],
        leverage_levels=[10, 20, 50, 100, 200, 500],
        tp_sl_combos=[(2.0, 1.0), (3.0, 1.5), (4.0, 2.0), (5.0, 2.5)]
    )

    # Print results
    print_results_summary(results)

    # Save results to CSV
    results_data = []
    for r in results:
        results_data.append({
            'symbol': r.config.symbol,
            'strategy': r.config.strategy,
            'leverage': r.config.leverage,
            'tp_pct': r.config.tp_pct,
            'sl_pct': r.config.sl_pct,
            'total_trades': r.total_trades,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'total_pnl': r.total_pnl,
            'total_return_pct': r.total_return_pct,
            'max_drawdown': r.max_drawdown,
            'winning_trades': r.winning_trades,
            'losing_trades': r.losing_trades,
            'avg_win': r.avg_win,
            'avg_loss': r.avg_loss,
        })

    results_df = pd.DataFrame(results_data)
    output_file = os.path.join(os.path.dirname(__file__), 'btcc_backtest_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
