#!/usr/bin/env python3
"""
BTCC Long-Term Profitable Strategy Finder
==========================================
Find strategies that WIN LONG-TERM with 20x-50x leverage and max 50% drawdown.

Requirements:
- 20x to 50x leverage
- Max Drawdown <= 50%
- Positive returns over 30+ days
- Proper liquidation simulation

Usage:
    python find_longterm_strategy.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from itertools import product

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

# BTCC fees
MAKER_FEE = 0.00045
TAKER_FEE = 0.00045
MAINTENANCE_MARGIN_RATE = 0.005


@dataclass
class Position:
    symbol: str
    direction: int
    entry_price: float
    volume: float
    leverage: int
    tp_price: float
    sl_price: float
    liquidation_price: float
    margin: float
    entry_idx: int


@dataclass
class TradeResult:
    pnl: float
    pnl_pct: float
    exit_reason: str
    duration: int


class StrategySimulator:
    """Fast strategy simulator with liquidation"""

    def __init__(self, symbol: str, leverage: int, tp_pct: float, sl_pct: float,
                 strategy: str, position_size_pct: float = 3.0,
                 initial_balance: float = 10000.0):
        self.symbol = symbol
        self.leverage = leverage
        self.tp_pct = tp_pct / 100
        self.sl_pct = sl_pct / 100
        self.strategy = strategy
        self.position_size_pct = position_size_pct
        self.initial_balance = initial_balance
        self.balance = initial_balance

        self.position: Optional[Position] = None
        self.trades: List[TradeResult] = []
        self.max_balance = initial_balance
        self.max_drawdown = 0.0
        self.liquidations = 0

    def calculate_liquidation_price(self, entry_price: float, direction: int) -> float:
        if direction == 1:  # Long
            return entry_price * (1 - (1 / self.leverage) + MAINTENANCE_MARGIN_RATE)
        else:  # Short
            return entry_price * (1 + (1 / self.leverage) - MAINTENANCE_MARGIN_RATE)

    def run(self, df: pd.DataFrame) -> Dict:
        """Run simulation on dataframe"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        # Pre-calculate signals
        signals = self._calculate_signals(close, high, low, volume)

        # Simulation
        for i in range(50, len(df)):
            # Update drawdown
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            dd = (self.max_balance - self.balance) / self.max_balance * 100
            if dd > self.max_drawdown:
                self.max_drawdown = dd

            # Stop if wiped out
            if self.balance <= 0:
                break

            # Process position
            if self.position:
                result = self._check_exit(i, high[i], low[i], close[i])
                if result:
                    self.trades.append(result)
                    self.position = None

            # Open new position
            if self.position is None and signals[i] != 0:
                self._open_position(i, int(signals[i]), close[i])

        # Close remaining position
        if self.position:
            result = self._force_close(len(df) - 1, close[-1])
            self.trades.append(result)

        return self._get_results()

    def _calculate_signals(self, close, high, low, volume) -> np.ndarray:
        """Calculate signals based on strategy"""
        signals = np.zeros(len(close))

        if self.strategy == 'EMA_CROSS':
            ema9 = self._ema(close, 9)
            ema21 = self._ema(close, 21)
            for i in range(25, len(close)):
                if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
                    signals[i] = 1
                elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]:
                    signals[i] = -1

        elif self.strategy == 'EMA_TREND':
            # Only trade with trend (EMA50)
            ema9 = self._ema(close, 9)
            ema21 = self._ema(close, 21)
            ema50 = self._ema(close, 50)
            for i in range(55, len(close)):
                if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and close[i] > ema50[i]:
                    signals[i] = 1
                elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and close[i] < ema50[i]:
                    signals[i] = -1

        elif self.strategy == 'BOLLINGER':
            sma = self._sma(close, 20)
            std = self._rolling_std(close, 20)
            upper = sma + 2 * std
            lower = sma - 2 * std
            for i in range(25, len(close)):
                if close[i-1] <= lower[i-1] and close[i] > lower[i]:
                    signals[i] = 1
                elif close[i-1] >= upper[i-1] and close[i] < upper[i]:
                    signals[i] = -1

        elif self.strategy == 'BOLLINGER_SQUEEZE':
            # Bollinger + volume confirmation
            sma = self._sma(close, 20)
            std = self._rolling_std(close, 20)
            upper = sma + 2 * std
            lower = sma - 2 * std
            vol_ma = self._sma(volume, 20)
            for i in range(25, len(close)):
                vol_confirm = volume[i] > vol_ma[i] * 1.2
                if close[i-1] <= lower[i-1] and close[i] > lower[i] and vol_confirm:
                    signals[i] = 1
                elif close[i-1] >= upper[i-1] and close[i] < upper[i] and vol_confirm:
                    signals[i] = -1

        elif self.strategy == 'RSI_EXTREME':
            rsi = self._rsi(close, 14)
            for i in range(20, len(close)):
                if rsi[i-1] < 25 and rsi[i] >= 25:
                    signals[i] = 1
                elif rsi[i-1] > 75 and rsi[i] <= 75:
                    signals[i] = -1

        elif self.strategy == 'RSI_MOMENTUM':
            # RSI with momentum confirmation
            rsi = self._rsi(close, 14)
            ema9 = self._ema(close, 9)
            ema21 = self._ema(close, 21)
            for i in range(25, len(close)):
                if rsi[i] < 40 and ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
                    signals[i] = 1
                elif rsi[i] > 60 and ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]:
                    signals[i] = -1

        elif self.strategy == 'MACD':
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd = ema12 - ema26
            signal_line = self._ema(macd, 9)
            for i in range(30, len(close)):
                if macd[i] > signal_line[i] and macd[i-1] <= signal_line[i-1]:
                    signals[i] = 1
                elif macd[i] < signal_line[i] and macd[i-1] >= signal_line[i-1]:
                    signals[i] = -1

        elif self.strategy == 'MACD_ZERO':
            # MACD crossing zero line
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd = ema12 - ema26
            for i in range(30, len(close)):
                if macd[i] > 0 and macd[i-1] <= 0:
                    signals[i] = 1
                elif macd[i] < 0 and macd[i-1] >= 0:
                    signals[i] = -1

        elif self.strategy == 'BREAKOUT':
            # Price breakout of 20-period range
            for i in range(25, len(close)):
                high_20 = np.max(high[i-20:i])
                low_20 = np.min(low[i-20:i])
                if close[i] > high_20 and close[i-1] <= high_20:
                    signals[i] = 1
                elif close[i] < low_20 and close[i-1] >= low_20:
                    signals[i] = -1

        elif self.strategy == 'SUPPORT_RESISTANCE':
            # Simple support/resistance with volume
            vol_ma = self._sma(volume, 20)
            for i in range(30, len(close)):
                low_10 = np.min(low[i-10:i])
                high_10 = np.max(high[i-10:i])
                vol_spike = volume[i] > vol_ma[i] * 1.3

                # Bounce from support
                if low[i] <= low_10 * 1.002 and close[i] > low[i] and vol_spike:
                    signals[i] = 1
                # Rejection from resistance
                elif high[i] >= high_10 * 0.998 and close[i] < high[i] and vol_spike:
                    signals[i] = -1

        elif self.strategy == 'TREND_FOLLOW':
            # Strong trend following
            ema20 = self._ema(close, 20)
            ema50 = self._ema(close, 50)
            atr = self._atr(high, low, close, 14)
            for i in range(55, len(close)):
                trend_up = ema20[i] > ema50[i] and ema20[i-5] < ema50[i-5]
                trend_down = ema20[i] < ema50[i] and ema20[i-5] > ema50[i-5]
                if trend_up:
                    signals[i] = 1
                elif trend_down:
                    signals[i] = -1

        return signals

    def _open_position(self, idx: int, direction: int, price: float):
        margin = self.balance * (self.position_size_pct / 100)
        if margin < 1:
            return

        position_value = margin * self.leverage

        # Entry fee
        self.balance -= position_value * TAKER_FEE

        if direction == 1:
            tp_price = price * (1 + self.tp_pct)
            sl_price = price * (1 - self.sl_pct)
        else:
            tp_price = price * (1 - self.tp_pct)
            sl_price = price * (1 + self.sl_pct)

        liq_price = self.calculate_liquidation_price(price, direction)

        self.position = Position(
            symbol=self.symbol,
            direction=direction,
            entry_price=price,
            volume=position_value,
            leverage=self.leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            liquidation_price=liq_price,
            margin=margin,
            entry_idx=idx
        )

    def _check_exit(self, idx: int, high: float, low: float, close: float) -> Optional[TradeResult]:
        pos = self.position
        if not pos:
            return None

        exit_price = None
        exit_reason = None

        if pos.direction == 1:  # Long
            # Check liquidation first
            if low <= pos.liquidation_price:
                exit_price = pos.liquidation_price
                exit_reason = 'LIQUIDATION'
                self.liquidations += 1
            elif high >= pos.tp_price:
                exit_price = pos.tp_price
                exit_reason = 'TP'
            elif low <= pos.sl_price:
                exit_price = pos.sl_price
                exit_reason = 'SL'
        else:  # Short
            if high >= pos.liquidation_price:
                exit_price = pos.liquidation_price
                exit_reason = 'LIQUIDATION'
                self.liquidations += 1
            elif low <= pos.tp_price:
                exit_price = pos.tp_price
                exit_reason = 'TP'
            elif high >= pos.sl_price:
                exit_price = pos.sl_price
                exit_reason = 'SL'

        if exit_price:
            return self._close_position(idx, exit_price, exit_reason)
        return None

    def _close_position(self, idx: int, exit_price: float, reason: str) -> TradeResult:
        pos = self.position

        if reason == 'LIQUIDATION':
            pnl = -pos.margin
        else:
            if pos.direction == 1:
                price_change = (exit_price - pos.entry_price) / pos.entry_price
            else:
                price_change = (pos.entry_price - exit_price) / pos.entry_price
            pnl = price_change * pos.volume
            pnl -= pos.volume * TAKER_FEE

        self.balance += pnl

        return TradeResult(
            pnl=pnl,
            pnl_pct=(pnl / pos.margin) * 100 if pos.margin > 0 else 0,
            exit_reason=reason,
            duration=idx - pos.entry_idx
        )

    def _force_close(self, idx: int, price: float) -> TradeResult:
        pos = self.position
        if pos.direction == 1:
            price_change = (price - pos.entry_price) / pos.entry_price
        else:
            price_change = (pos.entry_price - price) / pos.entry_price
        pnl = price_change * pos.volume
        pnl -= pos.volume * TAKER_FEE
        self.balance += pnl

        return TradeResult(
            pnl=pnl,
            pnl_pct=(pnl / pos.margin) * 100 if pos.margin > 0 else 0,
            exit_reason='MARKET',
            duration=idx - pos.entry_idx
        )

    def _get_results(self) -> Dict:
        if not self.trades:
            return {'profitable': False, 'total_trades': 0}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        tp_exits = len([t for t in self.trades if t.exit_reason == 'TP'])
        sl_exits = len([t for t in self.trades if t.exit_reason == 'SL'])

        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        return {
            'profitable': total_pnl > 0,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(self.trades)) * 100,
            'total_pnl': total_pnl,
            'return_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'max_drawdown': self.max_drawdown,
            'liquidations': self.liquidations,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'avg_trade_duration': np.mean([t.duration for t in self.trades]),
            'final_balance': self.balance,
        }

    def _ema(self, arr, period):
        ema = np.zeros_like(arr)
        alpha = 2 / (period + 1)
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
        return ema

    def _sma(self, arr, period):
        sma = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            sma[i] = np.mean(arr[i-period+1:i+1])
        return sma

    def _rolling_std(self, arr, period):
        std = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            std[i] = np.std(arr[i-period+1:i+1])
        return std

    def _rsi(self, arr, period):
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = np.zeros(len(arr))
        if len(gains) < period:
            return rsi
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(arr) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi[i + 1] = 100
            else:
                rsi[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi

    def _atr(self, high, low, close, period):
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        return self._sma(tr, period)


def load_data(symbol: str, candles: int = None) -> Optional[pd.DataFrame]:
    """Load data for symbol"""
    file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if candles:
        df = df.tail(candles).reset_index(drop=True)
    return df


def run_optimization():
    """Run comprehensive optimization to find long-term profitable strategies"""

    print("\n" + "=" * 100)
    print("BTCC LONG-TERM PROFITABLE STRATEGY FINDER")
    print("=" * 100)
    print("\nObjective: Find strategy with 20x-50x leverage, max 50% DD, POSITIVE returns over 30+ days")
    print("=" * 100)

    # Symbols to test (focus on best performers from earlier backtest)
    symbols = ['AVAXUSDT', 'LINKUSDT', 'XRPUSDT', 'BTCUSDT']

    # Strategies (most promising ones)
    strategies = [
        'EMA_CROSS', 'EMA_TREND', 'BOLLINGER', 'BOLLINGER_SQUEEZE',
        'RSI_MOMENTUM', 'MACD', 'TREND_FOLLOW'
    ]

    # Parameters to test - reduced for speed
    leverages = [20, 30, 50]
    tp_pcts = [1.5, 2.0, 3.0, 4.0]
    sl_pcts = [0.8, 1.0, 1.5, 2.0]
    position_sizes = [3.0, 5.0]

    # Test periods (in 1m candles) - 30 days
    test_period = 60 * 24 * 30  # 30 days = 43200 candles

    all_results = []
    total_configs = len(symbols) * len(strategies) * len(leverages) * len(tp_pcts) * len(sl_pcts) * len(position_sizes)

    print(f"\nTesting {total_configs:,} configurations across {len(symbols)} symbols...")
    print("-" * 100)

    config_count = 0
    for symbol in symbols:
        print(f"\nLoading {symbol}...")
        df = load_data(symbol, test_period)
        if df is None:
            print(f"  SKIP: No data for {symbol}")
            continue

        print(f"  Loaded {len(df):,} candles ({len(df)/1440:.1f} days)")

        for strategy in strategies:
            for leverage in leverages:
                for tp_pct in tp_pcts:
                    for sl_pct in sl_pcts:
                        # Skip if TP/SL ratio is bad
                        if tp_pct < sl_pct:
                            continue

                        for pos_size in position_sizes:
                            config_count += 1

                            sim = StrategySimulator(
                                symbol=symbol,
                                leverage=leverage,
                                tp_pct=tp_pct,
                                sl_pct=sl_pct,
                                strategy=strategy,
                                position_size_pct=pos_size
                            )

                            results = sim.run(df)

                            # Filter: Must be profitable, max 50% DD, no liquidations
                            if (results['profitable'] and
                                results['max_drawdown'] <= 50 and
                                results['liquidations'] == 0 and
                                results['total_trades'] >= 10):

                                results['symbol'] = symbol
                                results['strategy'] = strategy
                                results['leverage'] = leverage
                                results['tp_pct'] = tp_pct
                                results['sl_pct'] = sl_pct
                                results['pos_size'] = pos_size
                                all_results.append(results)

            # Progress update
            if config_count % 1000 == 0:
                print(f"  Progress: {config_count:,}/{total_configs:,} | Found {len(all_results)} viable configs")

    print(f"\n{'=' * 100}")
    print(f"OPTIMIZATION COMPLETE - Found {len(all_results)} viable configurations")
    print("=" * 100)

    if not all_results:
        print("\nNO PROFITABLE CONFIGURATIONS FOUND with the given constraints!")
        print("Try relaxing the max drawdown or testing different periods.")
        return

    # Sort by return
    all_results.sort(key=lambda x: x['return_pct'], reverse=True)

    # Print top 30
    print(f"\n{'TOP 30 LONG-TERM PROFITABLE CONFIGURATIONS':^100}")
    print("-" * 100)
    print(f"{'Symbol':<10} {'Strategy':<18} {'Lev':>4} {'TP%':>5} {'SL%':>5} {'Size':>5} {'Trades':>7} {'Win%':>6} {'Return':>9} {'MaxDD':>7} {'PF':>6}")
    print("-" * 100)

    for r in all_results[:30]:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "INF"
        print(f"{r['symbol']:<10} {r['strategy']:<18} {r['leverage']:>4}x {r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} {r['pos_size']:>5.1f} {r['total_trades']:>7} {r['win_rate']:>5.1f}% {r['return_pct']:>+8.1f}% {r['max_drawdown']:>6.1f}% {pf_str:>6}")

    print("=" * 100)

    # Best per symbol
    print(f"\n{'BEST CONFIGURATION PER SYMBOL':^100}")
    print("-" * 100)

    symbols_found = set(r['symbol'] for r in all_results)
    for symbol in symbols_found:
        sym_results = [r for r in all_results if r['symbol'] == symbol]
        if sym_results:
            best = max(sym_results, key=lambda x: x['return_pct'])
            print(f"\n{symbol}:")
            print(f"  Strategy: {best['strategy']} | Leverage: {best['leverage']}x")
            print(f"  TP: {best['tp_pct']:.1f}% | SL: {best['sl_pct']:.1f}% | Position Size: {best['pos_size']:.1f}%")
            print(f"  Return: {best['return_pct']:+.1f}% | Win Rate: {best['win_rate']:.1f}% | Max DD: {best['max_drawdown']:.1f}%")
            print(f"  Trades: {best['total_trades']} | Profit Factor: {best['profit_factor']:.2f}")

    print("\n" + "=" * 100)

    # Overall best
    best = all_results[0]
    print(f"\n{'RECOMMENDED CONFIGURATION FOR LIVE TRADING':^100}")
    print("=" * 100)
    print(f"\nSymbol: {best['symbol']}")
    print(f"Strategy: {best['strategy']}")
    print(f"Leverage: {best['leverage']}x")
    print(f"Take Profit: {best['tp_pct']}%")
    print(f"Stop Loss: {best['sl_pct']}%")
    print(f"Position Size: {best['pos_size']}% of balance")
    print(f"\nExpected Performance (30 days):")
    print(f"  Return: {best['return_pct']:+.1f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"  Trades: {best['total_trades']}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Liquidations: {best['liquidations']}")
    print("=" * 100)

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'longterm_strategy_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    run_optimization()
