#!/usr/bin/env python3
"""
BTCC Futures FAST Backtest Engine
=================================
Optimized vectorized backtesting for BTCC Futures with leverage up to 500x.
Uses numpy for fast calculations.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')


@dataclass
class BTCCBacktestResult:
    """Fast backtest result"""
    symbol: str
    strategy: str
    leverage: int
    tp_pct: float
    sl_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    net_pnl: float
    gross_pnl: float
    total_fees: float
    max_drawdown: float
    return_pct: float
    avg_win: float
    avg_loss: float
    sharpe: float


class FastBTCCBacktester:
    """Fast vectorized BTCC backtester"""

    # BTCC leverage tiers
    BTCC_SYMBOLS = {
        'BTCUSDT': 500, 'ETHUSDT': 500, 'SOLUSDT': 500, 'XRPUSDT': 500,
        'LTCUSDT': 200, 'LINKUSDT': 100, 'ADAUSDT': 100, 'DOGEUSDT': 100,
        'AVAXUSDT': 100, 'DOTUSDT': 100, 'UNIUSDT': 50, 'BCHUSDT': 100,
        'XLMUSDT': 50, 'ZECUSDT': 50, 'XMRUSDT': 50, 'PEPEUSDT': 50,
        'SUIUSDT': 50,
    }

    # BTCC fees: 0.045% maker, 0.045% taker = ~0.09% round trip
    COMMISSION_PCT = 0.045  # Per side

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and cache data."""
        if symbol in self.data_cache:
            return self.data_cache[symbol]

        file_path = os.path.join(self.data_path, f"{symbol}_1m.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        self.data_cache[symbol] = df
        return df

    # ==================== FAST INDICATOR CALCULATIONS ====================

    def calc_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized RSI calculation."""
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gains).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(span=period, adjust=False).mean().values

        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_gain))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Vectorized EMA calculation."""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def calc_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Vectorized SMA calculation."""
        return pd.Series(data).rolling(window=period).mean().values

    def calc_macd(self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized MACD calculation."""
        ema_fast = self.calc_ema(close, fast)
        ema_slow = self.calc_ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calc_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calc_bollinger(self, close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized Bollinger Bands calculation."""
        middle = self.calc_sma(close, period)
        std = pd.Series(close).rolling(window=period).std().values
        upper = middle + std_mult * std
        lower = middle - std_mult * std
        return middle, upper, lower

    def calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized ATR calculation."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        return pd.Series(tr).rolling(window=period).mean().values

    # ==================== FAST SIGNAL GENERATION ====================

    def generate_signals_scalping(self, df: pd.DataFrame) -> np.ndarray:
        """Generate SCALPING_MOMENTUM signals (vectorized)."""
        close = df['close'].values
        volume = df['volume'].values

        rsi = self.calc_rsi(close, 14)
        _, _, macd_hist = self.calc_macd(close, 12, 26, 9)
        ema_fast = self.calc_ema(close, 9)
        ema_slow = self.calc_ema(close, 21)

        # Volume filter
        vol_sma = self.calc_sma(volume, 20)
        vol_ok = volume > vol_sma * 1.2

        signals = np.zeros(len(close))

        # Long signals
        long_cond = (rsi < 35) & (macd_hist > np.roll(macd_hist, 1)) & (ema_fast > ema_slow) & vol_ok
        signals[long_cond] = 1

        # Short signals
        short_cond = (rsi > 65) & (macd_hist < np.roll(macd_hist, 1)) & (ema_fast < ema_slow) & vol_ok
        signals[short_cond] = -1

        return signals

    def generate_signals_rsi(self, df: pd.DataFrame) -> np.ndarray:
        """Generate RSI_REVERSAL signals (vectorized)."""
        close = df['close'].values
        rsi = self.calc_rsi(close, 14)

        signals = np.zeros(len(close))

        # RSI reversal from oversold
        prev_rsi = np.roll(rsi, 1)
        long_cond = (prev_rsi < 30) & (rsi > 30) & (rsi > prev_rsi)
        signals[long_cond] = 1

        # RSI reversal from overbought
        short_cond = (prev_rsi > 70) & (rsi < 70) & (rsi < prev_rsi)
        signals[short_cond] = -1

        return signals

    def generate_signals_breakout(self, df: pd.DataFrame) -> np.ndarray:
        """Generate BREAKOUT signals (vectorized)."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Rolling high/low
        resistance = pd.Series(high).rolling(20).max().values
        support = pd.Series(low).rolling(20).min().values

        # Volume confirmation
        vol_sma = self.calc_sma(volume, 20)
        vol_ok = volume > vol_sma * 1.3

        signals = np.zeros(len(close))

        # Breakout above resistance
        prev_close = np.roll(close, 1)
        long_cond = (close > resistance) & (prev_close <= np.roll(resistance, 1)) & vol_ok
        signals[long_cond] = 1

        # Breakdown below support
        short_cond = (close < support) & (prev_close >= np.roll(support, 1)) & vol_ok
        signals[short_cond] = -1

        return signals

    def generate_signals_ema(self, df: pd.DataFrame) -> np.ndarray:
        """Generate EMA_CROSSOVER signals (vectorized)."""
        close = df['close'].values

        ema_fast = self.calc_ema(close, 9)
        ema_slow = self.calc_ema(close, 21)
        ema_trend = self.calc_ema(close, 50)

        prev_fast = np.roll(ema_fast, 1)
        prev_slow = np.roll(ema_slow, 1)

        signals = np.zeros(len(close))

        # Bullish crossover
        long_cond = (prev_fast <= prev_slow) & (ema_fast > ema_slow) & (close > ema_trend)
        signals[long_cond] = 1

        # Bearish crossover
        short_cond = (prev_fast >= prev_slow) & (ema_fast < ema_slow) & (close < ema_trend)
        signals[short_cond] = -1

        return signals

    def generate_signals_bollinger(self, df: pd.DataFrame) -> np.ndarray:
        """Generate BOLLINGER_SQUEEZE signals (vectorized)."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        _, bb_upper, bb_lower = self.calc_bollinger(close, 20, 2.0)
        bb_width = bb_upper - bb_lower

        # Squeeze detection (low volatility)
        avg_width = self.calc_sma(bb_width, 20)
        is_squeeze = bb_width < avg_width * 0.8

        prev_squeeze = np.roll(is_squeeze, 1)
        momentum = close - np.roll(close, 5)

        signals = np.zeros(len(close))

        # Squeeze release bullish
        long_cond = prev_squeeze & ~is_squeeze & (momentum > 0)
        signals[long_cond] = 1

        # Squeeze release bearish
        short_cond = prev_squeeze & ~is_squeeze & (momentum < 0)
        signals[short_cond] = -1

        return signals

    # ==================== FAST BACKTEST ENGINE ====================

    def backtest_vectorized(self, df: pd.DataFrame, signals: np.ndarray,
                           leverage: int, tp_pct: float, sl_pct: float,
                           initial_capital: float = 100.0) -> Dict:
        """
        Ultra-fast vectorized backtesting.
        Returns trade statistics.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        n = len(close)

        trades = []
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_idx = 0

        for i in range(100, n):
            if position == 0:
                # Look for entry
                if signals[i] == 1:  # Long
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                elif signals[i] == -1:  # Short
                    position = -1
                    entry_price = close[i]
                    entry_idx = i

            elif position == 1:  # Long position
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)

                if low[i] <= sl_price:
                    # Stop loss hit
                    pnl_pct = -sl_pct * leverage
                    trades.append(pnl_pct)
                    position = 0
                elif high[i] >= tp_price:
                    # Take profit hit
                    pnl_pct = tp_pct * leverage
                    trades.append(pnl_pct)
                    position = 0

            elif position == -1:  # Short position
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)

                if high[i] >= sl_price:
                    # Stop loss hit
                    pnl_pct = -sl_pct * leverage
                    trades.append(pnl_pct)
                    position = 0
                elif low[i] <= tp_price:
                    # Take profit hit
                    pnl_pct = tp_pct * leverage
                    trades.append(pnl_pct)
                    position = 0

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'gross_pnl': 0,
                'net_pnl': 0,
                'total_fees': 0,
                'max_dd': 0,
                'return_pct': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'sharpe': 0,
            }

        trades = np.array(trades)

        # Calculate fees (both entry and exit)
        fee_per_trade = self.COMMISSION_PCT * 2 * leverage  # % per trade
        total_fees = len(trades) * initial_capital * fee_per_trade / 100

        # Statistics
        wins = trades[trades > 0]
        losses = trades[trades <= 0]

        gross_pnl = np.sum(trades) / 100 * initial_capital
        net_pnl = gross_pnl - total_fees

        # Max drawdown
        cumulative = np.cumsum(trades)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative)
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio
        if len(trades) > 1 and np.std(trades) > 0:
            sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100,
            'profit_factor': pf,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_fees': total_fees,
            'max_dd': max_dd,
            'return_pct': net_pnl / initial_capital * 100,
            'avg_win': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'sharpe': sharpe,
        }

    def run_optimization(self, symbol: str, strategies: List[str] = None,
                         leverage_levels: List[int] = None,
                         tp_range: List[float] = None,
                         sl_range: List[float] = None) -> List[BTCCBacktestResult]:
        """Run optimization for a single symbol."""
        df = self.load_data(symbol)
        if df is None or len(df) < 1000:
            return []

        max_leverage = self.BTCC_SYMBOLS.get(symbol, 100)

        strategies = strategies or ['SCALPING', 'RSI', 'BREAKOUT', 'EMA', 'BOLLINGER']
        leverage_levels = leverage_levels or [10, 20, 50, 100, 200, 500]
        tp_range = tp_range or [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        sl_range = sl_range or [0.5, 1.0, 1.5, 2.0, 2.5]

        # Generate signals for all strategies
        signal_map = {
            'SCALPING': self.generate_signals_scalping(df),
            'RSI': self.generate_signals_rsi(df),
            'BREAKOUT': self.generate_signals_breakout(df),
            'EMA': self.generate_signals_ema(df),
            'BOLLINGER': self.generate_signals_bollinger(df),
        }

        results = []

        for strategy in strategies:
            signals = signal_map.get(strategy)
            if signals is None:
                continue

            for leverage in leverage_levels:
                if leverage > max_leverage:
                    continue

                for tp in tp_range:
                    for sl in sl_range:
                        stats = self.backtest_vectorized(df, signals, leverage, tp, sl)

                        if stats['total_trades'] >= 5:
                            results.append(BTCCBacktestResult(
                                symbol=symbol,
                                strategy=strategy,
                                leverage=leverage,
                                tp_pct=tp,
                                sl_pct=sl,
                                total_trades=stats['total_trades'],
                                win_rate=stats['win_rate'],
                                profit_factor=stats['profit_factor'],
                                net_pnl=stats['net_pnl'],
                                gross_pnl=stats['gross_pnl'],
                                total_fees=stats['total_fees'],
                                max_drawdown=stats['max_dd'],
                                return_pct=stats['return_pct'],
                                avg_win=stats['avg_win'],
                                avg_loss=stats['avg_loss'],
                                sharpe=stats['sharpe'],
                            ))

        return results


def main():
    """Run fast BTCC backtest optimization."""
    print("\n" + "=" * 90)
    print("BTCC FUTURES FAST BACKTEST - ALL LEVERAGE LEVELS UP TO 500x")
    print("=" * 90)

    backtester = FastBTCCBacktester()

    # Check available data
    print("\nLoading data...")
    available_symbols = []
    for symbol, max_lev in backtester.BTCC_SYMBOLS.items():
        df = backtester.load_data(symbol)
        if df is not None and len(df) > 1000:
            available_symbols.append(symbol)
            print(f"  {symbol}: {len(df):,} candles (max leverage: {max_lev}x)")

    if not available_symbols:
        print("No data available!")
        return

    all_results = []

    print(f"\nRunning optimization on {len(available_symbols)} symbols...")
    print("-" * 90)

    for symbol in available_symbols:
        print(f"\nOptimizing {symbol}...", end=" ", flush=True)

        results = backtester.run_optimization(
            symbol=symbol,
            strategies=['SCALPING', 'RSI', 'BREAKOUT', 'EMA', 'BOLLINGER'],
            leverage_levels=[5, 10, 20, 50, 100, 150, 200, 300, 500],
            tp_range=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            sl_range=[0.5, 1.0, 1.5, 2.0, 2.5],
        )

        all_results.extend(results)
        profitable = len([r for r in results if r.net_pnl > 0])
        print(f"{len(results)} configs tested, {profitable} profitable")

    # Sort by net P/L
    all_results.sort(key=lambda x: x.net_pnl, reverse=True)

    # Print summary
    print("\n" + "=" * 90)
    print("TOP 30 CONFIGURATIONS BY NET P/L")
    print("=" * 90)
    print(f"{'SYMBOL':<10} {'STRATEGY':<12} {'LEV':<5} {'TP%':<5} {'SL%':<5} "
          f"{'TRADES':<7} {'WIN%':<7} {'PF':<6} {'NET P/L':<10} {'RET%':<8} {'MAX DD':<8}")
    print("-" * 90)

    for r in all_results[:30]:
        print(f"{r.symbol:<10} {r.strategy:<12} {r.leverage:<5} "
              f"{r.tp_pct:<5.1f} {r.sl_pct:<5.1f} {r.total_trades:<7} "
              f"{r.win_rate:<7.1f} {r.profit_factor:<6.2f} "
              f"${r.net_pnl:<9.2f} {r.return_pct:<8.1f} {r.max_drawdown:<8.1f}%")

    # Best by leverage tier
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION PER LEVERAGE TIER")
    print("=" * 90)

    for lev in [10, 20, 50, 100, 200, 500]:
        lev_results = [r for r in all_results if r.leverage == lev and r.total_trades >= 10]
        if lev_results:
            best = max(lev_results, key=lambda x: x.net_pnl)
            risk_warning = " [HIGH RISK]" if lev >= 100 else ""
            print(f"\n{lev}x Leverage{risk_warning}:")
            print(f"  {best.symbol} / {best.strategy} | TP:{best.tp_pct}% SL:{best.sl_pct}%")
            print(f"  Net P/L: ${best.net_pnl:.2f} | Win Rate: {best.win_rate:.1f}% | "
                  f"Trades: {best.total_trades} | Max DD: {best.max_drawdown:.1f}%")

    # High leverage analysis
    print("\n" + "=" * 90)
    print("HIGH LEVERAGE RISK ANALYSIS (100x - 500x)")
    print("=" * 90)

    for lev in [100, 200, 300, 500]:
        lev_results = [r for r in all_results if r.leverage == lev and r.total_trades >= 10]
        if lev_results:
            profitable = [r for r in lev_results if r.net_pnl > 0]
            avg_dd = np.mean([r.max_drawdown for r in lev_results])
            liquidation_risk = len([r for r in lev_results if r.max_drawdown > 80])

            print(f"\n{lev}x Leverage:")
            print(f"  Configs: {len(lev_results)} | Profitable: {len(profitable)} ({len(profitable)/len(lev_results)*100:.0f}%)")
            print(f"  Avg Max DD: {avg_dd:.1f}% | Liquidation Risk (DD>80%): {liquidation_risk}")

    # Symbol summary
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION PER SYMBOL")
    print("=" * 90)

    for symbol in available_symbols:
        sym_results = [r for r in all_results if r.symbol == symbol and r.total_trades >= 10]
        if sym_results:
            best = max(sym_results, key=lambda x: x.net_pnl)
            max_lev = backtester.BTCC_SYMBOLS.get(symbol, 100)
            print(f"\n{symbol} (max {max_lev}x):")
            print(f"  Best: {best.strategy} @ {best.leverage}x | TP:{best.tp_pct}% SL:{best.sl_pct}%")
            print(f"  Net P/L: ${best.net_pnl:.2f} | Win: {best.win_rate:.1f}% | PF: {best.profit_factor:.2f}")

    # Save results
    results_data = [{
        'symbol': r.symbol, 'strategy': r.strategy, 'leverage': r.leverage,
        'tp_pct': r.tp_pct, 'sl_pct': r.sl_pct, 'total_trades': r.total_trades,
        'win_rate': r.win_rate, 'profit_factor': r.profit_factor,
        'net_pnl': r.net_pnl, 'gross_pnl': r.gross_pnl, 'total_fees': r.total_fees,
        'max_drawdown': r.max_drawdown, 'return_pct': r.return_pct,
        'sharpe': r.sharpe,
    } for r in all_results]

    df_results = pd.DataFrame(results_data)
    output_file = os.path.join(os.path.dirname(__file__), 'btcc_backtest_results.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    print(f"Total configurations tested: {len(all_results)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
