#!/usr/bin/env python3
"""
BTCC Futures Paper Trading Simulation
======================================
Run paper trading simulation using historical data with backtest-optimized settings.

Usage:
    python run_btcc_paper_simulation.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

# Best configurations from backtest - SAFER LEVERAGE (reduced from 500x to avoid liquidation)
# Backtest showed 500x has 12000%+ max DD = instant liquidation in real trading
BEST_CONFIGS = {
    # Reduced leverage versions (still profitable per backtest, much safer):
    'XRPUSDT': {'strategy': 'EMA', 'leverage': 50, 'tp_pct': 3.0, 'sl_pct': 2.5},        # Reduced from 500x
    'ETHUSDT': {'strategy': 'BOLLINGER', 'leverage': 50, 'tp_pct': 5.0, 'sl_pct': 2.5},  # Reduced from 500x
    'BTCUSDT': {'strategy': 'EMA', 'leverage': 50, 'tp_pct': 4.0, 'sl_pct': 2.5},        # Reduced from 500x
    'AVAXUSDT': {'strategy': 'BOLLINGER', 'leverage': 100, 'tp_pct': 3.0, 'sl_pct': 0.5}, # $8,734 profit (best risk-adjusted)
    'ZECUSDT': {'strategy': 'SCALPING', 'leverage': 50, 'tp_pct': 4.0, 'sl_pct': 2.0},   # $1,432, 66.7% win, PF 4.0
}

# BTCC fees
MAKER_FEE = 0.00045
TAKER_FEE = 0.00045


class Position:
    """Trading position"""
    def __init__(self, symbol, direction, entry_price, volume, leverage, tp_price, sl_price):
        self.symbol = symbol
        self.direction = direction  # 1=long, -1=short
        self.entry_price = entry_price
        self.volume = volume
        self.leverage = leverage
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.entry_time = None
        self.pnl = 0.0


class BTCCPaperSimulator:
    """Paper trading simulator using historical data"""

    def __init__(self, initial_balance=10000.0, position_size_pct=2.0, max_positions=3):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.positions = []
        self.closed_trades = []
        self.data = {}

    def load_data(self):
        """Load historical data for symbols"""
        print("\nLoading historical data...")

        for symbol in BEST_CONFIGS.keys():
            file_path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Use last 20,000 candles for simulation (about 2 weeks of 1m data)
                self.data[symbol] = df.tail(20000).reset_index(drop=True)
                print(f"  {symbol}: {len(self.data[symbol]):,} candles")
            else:
                print(f"  {symbol}: NOT FOUND")

        return len(self.data) > 0

    def calculate_signals(self, df, strategy, lookback=20):
        """Calculate trading signals"""
        signals = np.zeros(len(df))
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        if strategy == 'EMA':
            ema_short = self._ema(close, 9)
            ema_long = self._ema(close, 21)

            for i in range(lookback, len(df)):
                if ema_short[i] > ema_long[i] and ema_short[i-1] <= ema_long[i-1]:
                    signals[i] = 1  # Long
                elif ema_short[i] < ema_long[i] and ema_short[i-1] >= ema_long[i-1]:
                    signals[i] = -1  # Short

        elif strategy == 'BOLLINGER':
            sma = self._sma(close, 20)
            std = self._rolling_std(close, 20)
            upper = sma + 2 * std
            lower = sma - 2 * std

            for i in range(lookback, len(df)):
                if close[i-1] <= lower[i-1] and close[i] > lower[i]:
                    signals[i] = 1  # Long
                elif close[i-1] >= upper[i-1] and close[i] < upper[i]:
                    signals[i] = -1  # Short

        elif strategy == 'RSI':
            rsi = self._rsi(close, 14)

            for i in range(lookback, len(df)):
                if rsi[i-1] < 30 and rsi[i] >= 30:
                    signals[i] = 1  # Long
                elif rsi[i-1] > 70 and rsi[i] <= 70:
                    signals[i] = -1  # Short

        elif strategy == 'SCALPING':
            # Scalping: RSI + Volume spike + Price momentum
            rsi = self._rsi(close, 7)  # Faster RSI
            vol_ma = self._sma(volume, 20)
            ema_fast = self._ema(close, 5)
            ema_slow = self._ema(close, 13)

            for i in range(lookback, len(df)):
                vol_spike = volume[i] > vol_ma[i] * 1.5
                momentum_up = ema_fast[i] > ema_slow[i]
                momentum_down = ema_fast[i] < ema_slow[i]

                if rsi[i] < 35 and vol_spike and momentum_up:
                    signals[i] = 1  # Long
                elif rsi[i] > 65 and vol_spike and momentum_down:
                    signals[i] = -1  # Short

        return signals

    def _ema(self, arr, period):
        """Exponential moving average"""
        ema = np.zeros_like(arr)
        alpha = 2 / (period + 1)
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
        return ema

    def _sma(self, arr, period):
        """Simple moving average"""
        sma = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            sma[i] = np.mean(arr[i-period+1:i+1])
        return sma

    def _rolling_std(self, arr, period):
        """Rolling standard deviation"""
        std = np.zeros_like(arr)
        for i in range(period - 1, len(arr)):
            std[i] = np.std(arr[i-period+1:i+1])
        return std

    def _rsi(self, arr, period):
        """Relative Strength Index"""
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        rsi = np.zeros(len(arr))
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(arr) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))

        return rsi

    def run_simulation(self):
        """Run the paper trading simulation"""
        print("\n" + "=" * 80)
        print("BTCC FUTURES PAPER TRADING SIMULATION")
        print("=" * 80)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Position Size: {self.position_size_pct}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Symbols: {', '.join(BEST_CONFIGS.keys())}")
        print("-" * 80)

        # Pre-calculate signals for all symbols
        all_signals = {}
        for symbol, config in BEST_CONFIGS.items():
            if symbol in self.data:
                df = self.data[symbol]
                signals = self.calculate_signals(df, config['strategy'])
                all_signals[symbol] = signals

        # Find common time range
        min_len = min(len(self.data[s]) for s in self.data)

        # Simulation loop
        trade_count = 0
        wins = 0
        losses = 0
        total_pnl = 0.0
        max_balance = self.initial_balance
        max_drawdown = 0.0

        print("\nRunning simulation...")

        for i in range(100, min_len):
            # Check existing positions for TP/SL
            positions_to_close = []
            for pos in self.positions:
                if pos.symbol not in self.data:
                    continue

                current_price = self.data[pos.symbol]['close'].iloc[i]

                # Check TP/SL
                if pos.direction == 1:  # Long
                    if current_price >= pos.tp_price:
                        pnl = (current_price - pos.entry_price) / pos.entry_price * pos.leverage * pos.volume
                        pnl -= pos.volume * (MAKER_FEE + TAKER_FEE)  # Fees
                        positions_to_close.append((pos, pnl, 'TP'))
                    elif current_price <= pos.sl_price:
                        pnl = (current_price - pos.entry_price) / pos.entry_price * pos.leverage * pos.volume
                        pnl -= pos.volume * (MAKER_FEE + TAKER_FEE)
                        positions_to_close.append((pos, pnl, 'SL'))
                else:  # Short
                    if current_price <= pos.tp_price:
                        pnl = (pos.entry_price - current_price) / pos.entry_price * pos.leverage * pos.volume
                        pnl -= pos.volume * (MAKER_FEE + TAKER_FEE)
                        positions_to_close.append((pos, pnl, 'TP'))
                    elif current_price >= pos.sl_price:
                        pnl = (pos.entry_price - current_price) / pos.entry_price * pos.leverage * pos.volume
                        pnl -= pos.volume * (MAKER_FEE + TAKER_FEE)
                        positions_to_close.append((pos, pnl, 'SL'))

            # Close positions
            for pos, pnl, reason in positions_to_close:
                self.positions.remove(pos)
                self.balance += pnl
                total_pnl += pnl
                trade_count += 1

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                self.closed_trades.append({
                    'symbol': pos.symbol,
                    'direction': 'LONG' if pos.direction == 1 else 'SHORT',
                    'entry_price': pos.entry_price,
                    'leverage': pos.leverage,
                    'pnl': pnl,
                    'reason': reason
                })

            # Update max drawdown
            if self.balance > max_balance:
                max_balance = self.balance
            dd = (max_balance - self.balance) / max_balance * 100
            if dd > max_drawdown:
                max_drawdown = dd

            # Check for new signals
            if len(self.positions) < self.max_positions:
                for symbol, config in BEST_CONFIGS.items():
                    if symbol not in all_signals:
                        continue

                    # Skip if already have position in this symbol
                    if any(p.symbol == symbol for p in self.positions):
                        continue

                    signal = all_signals[symbol][i]
                    if signal != 0:
                        current_price = self.data[symbol]['close'].iloc[i]

                        # Calculate position size
                        pos_value = self.balance * (self.position_size_pct / 100)
                        leverage = config['leverage']
                        tp_pct = config['tp_pct'] / 100
                        sl_pct = config['sl_pct'] / 100

                        if signal == 1:  # Long
                            tp_price = current_price * (1 + tp_pct)
                            sl_price = current_price * (1 - sl_pct)
                        else:  # Short
                            tp_price = current_price * (1 - tp_pct)
                            sl_price = current_price * (1 + sl_pct)

                        pos = Position(
                            symbol=symbol,
                            direction=int(signal),
                            entry_price=current_price,
                            volume=pos_value,
                            leverage=leverage,
                            tp_price=tp_price,
                            sl_price=sl_price
                        )
                        self.positions.append(pos)

                        # Entry fee
                        self.balance -= pos_value * TAKER_FEE

            # Progress update every 5000 candles
            if i % 5000 == 0:
                print(f"  Progress: {i:,}/{min_len:,} candles | Balance: ${self.balance:,.2f} | Trades: {trade_count}")

        # Close remaining positions at market
        for pos in self.positions:
            if pos.symbol in self.data:
                current_price = self.data[pos.symbol]['close'].iloc[-1]
                if pos.direction == 1:
                    pnl = (current_price - pos.entry_price) / pos.entry_price * pos.leverage * pos.volume
                else:
                    pnl = (pos.entry_price - current_price) / pos.entry_price * pos.leverage * pos.volume
                pnl -= pos.volume * (MAKER_FEE + TAKER_FEE)
                self.balance += pnl
                total_pnl += pnl
                trade_count += 1
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        # Print results
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0

        print("\n" + "=" * 80)
        print("PAPER TRADING SIMULATION RESULTS")
        print("=" * 80)
        print(f"\n{'PERFORMANCE SUMMARY':^80}")
        print("-" * 80)
        print(f"Initial Balance:      ${self.initial_balance:>12,.2f}")
        print(f"Final Balance:        ${self.balance:>12,.2f}")
        print(f"Total P/L:            ${total_pnl:>+12,.2f} ({total_pnl/self.initial_balance*100:+.1f}%)")
        print(f"Max Drawdown:         {max_drawdown:>12.1f}%")
        print("-" * 80)
        print(f"\n{'TRADE STATISTICS':^80}")
        print("-" * 80)
        print(f"Total Trades:         {trade_count:>12}")
        print(f"Winning Trades:       {wins:>12}")
        print(f"Losing Trades:        {losses:>12}")
        print(f"Win Rate:             {win_rate:>12.1f}%")

        # Per-symbol breakdown
        print("-" * 80)
        print(f"\n{'PER-SYMBOL BREAKDOWN':^80}")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'Win%':>8} {'Net P/L':>12}")
        print("-" * 80)

        for symbol in BEST_CONFIGS.keys():
            sym_trades = [t for t in self.closed_trades if t['symbol'] == symbol]
            sym_wins = len([t for t in sym_trades if t['pnl'] > 0])
            sym_losses = len([t for t in sym_trades if t['pnl'] <= 0])
            sym_pnl = sum(t['pnl'] for t in sym_trades)
            sym_winrate = (sym_wins / len(sym_trades) * 100) if sym_trades else 0

            print(f"{symbol:<12} {len(sym_trades):>8} {sym_wins:>8} {sym_losses:>8} {sym_winrate:>7.1f}% ${sym_pnl:>+11,.2f}")

        print("=" * 80)

        # Configuration used
        print(f"\n{'CONFIGURATION USED':^80}")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Strategy':<12} {'Leverage':>10} {'TP%':>8} {'SL%':>8}")
        print("-" * 80)
        for symbol, config in BEST_CONFIGS.items():
            print(f"{symbol:<12} {config['strategy']:<12} {config['leverage']:>10}x {config['tp_pct']:>7.1f}% {config['sl_pct']:>7.1f}%")
        print("=" * 80)

        return {
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }


def main():
    """Main entry point"""
    simulator = BTCCPaperSimulator(
        initial_balance=10000.0,
        position_size_pct=2.0,
        max_positions=3
    )

    if not simulator.load_data():
        print("ERROR: No data available for simulation")
        return

    results = simulator.run_simulation()

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nFinal Return: {results['total_pnl']/100:.1f}% on ${10000:,} initial balance")
    print(f"To run live trading, edit run_btcc_live_trading.py with your credentials")
    print("=" * 80)


if __name__ == "__main__":
    main()
