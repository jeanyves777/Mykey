#!/usr/bin/env python3
"""
BTCC High Win Rate Trading System
==================================
Runs the top 3 high win rate strategies from backtest:

1. ETHUSDT SCALP_MOMENTUM 20x - 84.8% win rate, +51.5% return
2. BTCUSDT SCALP_MOMENTUM 20x - 83.7% win rate, +26.3% return
3. XRPUSDT BOLLINGER_MEAN 20x - 80.4% win rate, +17.2% return

Key: Small TP (1%) + Wide SL (3%) = High win rate

Usage:
    python run_high_winrate_trading.py          # Paper trading (default)
    python run_high_winrate_trading.py --live   # Live trading
    python run_high_winrate_trading.py --test   # Quick test mode (1 hour)
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.btcc_high_winrate_config import (
    HIGH_WINRATE_PAPER_CONFIG,
    HIGH_WINRATE_LIVE_CONFIG,
    BTCCHighWinRateConfig
)
from BTCC_Futures.btcc_strategies import get_strategy, OHLCV

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')
TAKER_FEE = 0.00045  # 0.045%


class HighWinRateTradingEngine:
    """High Win Rate Trading Engine with proper liquidation simulation."""

    def __init__(self, config: BTCCHighWinRateConfig):
        self.config = config
        self.balance = config.PAPER_INITIAL_BALANCE
        self.initial_balance = self.balance
        self.max_balance = self.balance
        self.positions = {}  # symbol -> position dict
        self.trades = []
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'liquidations': 0,
            'max_dd': 0,
            'pnl': 0,
        }

    def load_data(self, symbol: str, bars: int = 10000) -> pd.DataFrame:
        """Load historical data for a symbol."""
        # Map symbol to file name (handle both with and without suffix)
        file_names = [
            f"{symbol}_1m.csv",
            f"{symbol}USDT_1m.csv" if not symbol.endswith('USDT') else None,
        ]

        for fn in file_names:
            if fn:
                file_path = os.path.join(DATA_DIR, fn)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    return df.tail(bars).reset_index(drop=True)

        print(f"WARNING: No data found for {symbol}")
        return None

    def get_candles(self, df: pd.DataFrame, idx: int, lookback: int = 100) -> list:
        """Get OHLCV candles for strategy evaluation."""
        start_idx = max(0, idx - lookback)
        candles = []
        for i in range(start_idx, idx + 1):
            row = df.iloc[i]
            candles.append(OHLCV(
                timestamp=datetime.now(),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume', 1.0)
            ))
        return candles

    def calculate_liquidation_price(self, entry: float, direction: int, leverage: int) -> float:
        """Calculate liquidation price."""
        maint = self.config.MAINTENANCE_MARGIN_RATE
        if direction == 1:  # Long
            return entry * (1 - (1 / leverage) + maint)
        else:  # Short
            return entry * (1 + (1 / leverage) - maint)

    def open_position(self, symbol: str, direction: int, entry_price: float,
                      config: dict) -> dict:
        """Open a new position."""
        leverage = config['leverage']
        margin = self.balance * (self.config.POSITION_SIZE_PCT / 100)

        if margin < 10:  # Minimum margin
            return None

        # Calculate levels
        tp_pct = config['tp_pct'] / 100
        sl_pct = config['sl_pct'] / 100

        if direction == 1:  # Long
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:  # Short
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        liq_price = self.calculate_liquidation_price(entry_price, direction, leverage)

        # Deduct entry fee
        position_size = margin * leverage
        entry_fee = position_size * TAKER_FEE
        self.balance -= entry_fee

        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'margin': margin,
            'leverage': leverage,
            'position_size': position_size,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'liq_price': liq_price,
            'entry_time': datetime.now(),
            'entry_fee': entry_fee,
        }

        self.positions[symbol] = position
        return position

    def close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        """Close a position and calculate PnL."""
        if symbol not in self.positions:
            return 0

        pos = self.positions[symbol]
        direction = pos['direction']
        entry = pos['entry_price']
        margin = pos['margin']
        leverage = pos['leverage']

        # Calculate PnL
        if direction == 1:  # Long
            pnl_pct = (exit_price - entry) / entry
        else:  # Short
            pnl_pct = (entry - exit_price) / entry

        gross_pnl = pnl_pct * margin * leverage
        exit_fee = pos['position_size'] * TAKER_FEE
        net_pnl = gross_pnl - exit_fee

        self.balance += net_pnl
        self.stats['pnl'] += net_pnl

        # Track trade
        trade = {
            'symbol': symbol,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry': entry,
            'exit': exit_price,
            'pnl': net_pnl,
            'reason': reason,
            'leverage': leverage,
        }
        self.trades.append(trade)
        self.stats['total_trades'] += 1

        if net_pnl > 0:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1

        if reason == 'LIQUIDATION':
            self.stats['liquidations'] += 1

        del self.positions[symbol]
        return net_pnl

    def check_position(self, symbol: str, high: float, low: float, close: float) -> str:
        """Check if position should be closed."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        direction = pos['direction']

        if direction == 1:  # Long
            # Check liquidation first
            if low <= pos['liq_price']:
                self.close_position(symbol, pos['liq_price'], 'LIQUIDATION')
                self.balance -= pos['margin']  # Lose margin on liquidation
                return 'LIQUIDATION'
            # Check TP
            if high >= pos['tp_price']:
                self.close_position(symbol, pos['tp_price'], 'TP')
                return 'TP'
            # Check SL
            if low <= pos['sl_price']:
                self.close_position(symbol, pos['sl_price'], 'SL')
                return 'SL'
        else:  # Short
            # Check liquidation first
            if high >= pos['liq_price']:
                self.close_position(symbol, pos['liq_price'], 'LIQUIDATION')
                self.balance -= pos['margin']
                return 'LIQUIDATION'
            # Check TP
            if low <= pos['tp_price']:
                self.close_position(symbol, pos['tp_price'], 'TP')
                return 'TP'
            # Check SL
            if high >= pos['sl_price']:
                self.close_position(symbol, pos['sl_price'], 'SL')
                return 'SL'

        return None

    def update_drawdown(self):
        """Update max drawdown tracking."""
        if self.balance > self.max_balance:
            self.max_balance = self.balance

        dd = (self.max_balance - self.balance) / self.max_balance * 100
        if dd > self.stats['max_dd']:
            self.stats['max_dd'] = dd

    def run_simulation(self, duration_bars: int = 43200):
        """
        Run paper trading simulation.

        Args:
            duration_bars: Number of 1-minute bars to simulate (default: 30 days)
        """
        print("=" * 80)
        print("HIGH WIN RATE TRADING SIMULATION")
        print("=" * 80)
        print(f"Initial Balance: ${self.balance:,.2f}")
        print(f"Mode: {'PAPER' if self.config.is_paper_mode() else 'LIVE'}")
        print(f"Duration: {duration_bars:,} minutes ({duration_bars/1440:.1f} days)")
        print()

        # Load data for all enabled symbols
        symbols = self.config.get_enabled_symbols()
        data = {}
        strategies = {}

        for symbol in symbols:
            cfg = self.config.get_symbol_config(symbol)
            df = self.load_data(symbol, duration_bars + 200)

            if df is not None and len(df) > 100:
                data[symbol] = df
                strategy_name = cfg['strategy']
                strategy_params = self.config.get_strategy_params(strategy_name)
                strategies[symbol] = get_strategy(strategy_name, strategy_params)

                print(f"Loaded {symbol}: {len(df):,} bars | Strategy: {strategy_name}")
                print(f"  Config: {cfg['leverage']}x, TP:{cfg['tp_pct']}%, SL:{cfg['sl_pct']}%")
                print(f"  Expected: {cfg['expected_win_rate']}% win rate, +{cfg['expected_return']}% return")

        print()
        print("-" * 80)
        print("Starting simulation...")
        print("-" * 80, flush=True)

        # Simulation loop
        warmup = 100
        trade_count = 0

        for bar_idx in range(warmup, min(duration_bars, min(len(df) for df in data.values()))):
            # Update drawdown
            self.update_drawdown()

            if self.balance <= 0:
                print(f"\nACCOUNT BLOWN at bar {bar_idx}")
                break

            # Check each symbol
            for symbol in symbols:
                if symbol not in data:
                    continue

                df = data[symbol]
                if bar_idx >= len(df):
                    continue

                row = df.iloc[bar_idx]
                high = row['high']
                low = row['low']
                close = row['close']

                cfg = self.config.get_symbol_config(symbol)

                # Check existing position
                result = self.check_position(symbol, high, low, close)
                if result:
                    trade_count += 1
                    dir_str = self.trades[-1]['direction']
                    pnl = self.trades[-1]['pnl']
                    print(f"  [{bar_idx}] {symbol} {dir_str} closed: {result} | PnL: ${pnl:+.2f} | Balance: ${self.balance:.2f}")

                # Check for new signal if no position
                if symbol not in self.positions and len(self.positions) < self.config.MAX_POSITIONS:
                    candles = self.get_candles(df, bar_idx)
                    strategy = strategies[symbol]

                    signal = strategy.evaluate(candles, close)

                    if signal:
                        pos = self.open_position(symbol, signal['direction'], close, cfg)
                        if pos:
                            dir_str = "LONG" if signal['direction'] == 1 else "SHORT"
                            print(f"  [{bar_idx}] {symbol} {dir_str} @ {close:.4f} | TP: {pos['tp_price']:.4f} | SL: {pos['sl_price']:.4f}")

            # Progress update
            if bar_idx % 5000 == 0 and bar_idx > warmup:
                print(f"\n--- Progress: {bar_idx:,} bars | Balance: ${self.balance:,.2f} | Trades: {self.stats['total_trades']} ---\n", flush=True)

        # Close remaining positions
        for symbol in list(self.positions.keys()):
            df = data.get(symbol)
            if df is not None:
                close = df.iloc[-1]['close']
                self.close_position(symbol, close, 'END')
                print(f"  Closed remaining {symbol} @ {close:.4f}")

        # Final report
        self.print_report()

    def print_report(self):
        """Print final trading report."""
        print()
        print("=" * 80)
        print("SIMULATION RESULTS")
        print("=" * 80)
        print()

        # Performance metrics
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (self.stats['wins'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0

        print(f"{'PERFORMANCE SUMMARY':^80}")
        print("-" * 80)
        print(f"Initial Balance:    ${self.initial_balance:,.2f}")
        print(f"Final Balance:      ${self.balance:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"Max Drawdown:       {self.stats['max_dd']:.2f}%")
        print()
        print(f"Total Trades:       {self.stats['total_trades']}")
        print(f"Wins:               {self.stats['wins']}")
        print(f"Losses:             {self.stats['losses']}")
        print(f"Win Rate:           {win_rate:.1f}%")
        print(f"Liquidations:       {self.stats['liquidations']}")
        print()

        # Trade breakdown by symbol
        if self.trades:
            print(f"{'BREAKDOWN BY SYMBOL':^80}")
            print("-" * 80)

            symbols = set(t['symbol'] for t in self.trades)
            for symbol in symbols:
                sym_trades = [t for t in self.trades if t['symbol'] == symbol]
                sym_wins = len([t for t in sym_trades if t['pnl'] > 0])
                sym_pnl = sum(t['pnl'] for t in sym_trades)
                sym_wr = (sym_wins / len(sym_trades) * 100) if sym_trades else 0

                print(f"{symbol:<12} Trades: {len(sym_trades):>4} | Wins: {sym_wins:>4} | WR: {sym_wr:>5.1f}% | PnL: ${sym_pnl:>+10.2f}")

        print()
        print("=" * 80)

        # Comparison with expected
        print(f"{'EXPECTED vs ACTUAL':^80}")
        print("-" * 80)
        for symbol in self.config.get_enabled_symbols():
            cfg = self.config.get_symbol_config(symbol)
            sym_trades = [t for t in self.trades if t['symbol'] == symbol]
            if sym_trades:
                actual_wr = len([t for t in sym_trades if t['pnl'] > 0]) / len(sym_trades) * 100
                print(f"{symbol}: Expected WR: {cfg['expected_win_rate']}% | Actual WR: {actual_wr:.1f}%")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='BTCC High Win Rate Trading')
    parser.add_argument('--live', action='store_true', help='Run in live mode')
    parser.add_argument('--test', action='store_true', help='Quick test mode (1 hour)')
    parser.add_argument('--days', type=float, default=30, help='Simulation duration in days')
    args = parser.parse_args()

    # Select config
    if args.live:
        config = HIGH_WINRATE_LIVE_CONFIG
        print("\n*** LIVE MODE - REAL MONEY AT RISK ***\n")
    else:
        config = HIGH_WINRATE_PAPER_CONFIG

    # Calculate duration
    if args.test:
        duration = 60  # 1 hour
    else:
        duration = int(args.days * 1440)  # Convert days to minutes

    # Create and run engine
    engine = HighWinRateTradingEngine(config)
    engine.run_simulation(duration)


if __name__ == '__main__':
    main()
