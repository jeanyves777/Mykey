#!/usr/bin/env python3
"""
AVAXUSDT Scalping Test with Proper Liquidation
===============================================
Test AVAXUSDT scalping with realistic liquidation simulation.

BTCC Liquidation Rules:
- Isolated margin: Liquidation when margin ratio <= maintenance margin
- At 100x leverage: ~1% move against you = liquidation
- At 50x leverage: ~2% move against you = liquidation
- Maintenance margin rate varies by tier (typically 0.5%-1%)

Usage:
    python run_avax_scalping_test.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'Crypto_Margin_Trading', 'Crypto_Data_from_Binance')

# BTCC fees
MAKER_FEE = 0.00045  # 0.045%
TAKER_FEE = 0.00045  # 0.045%

# Maintenance margin rate (for liquidation calculation)
MAINTENANCE_MARGIN_RATE = 0.005  # 0.5%


@dataclass
class Position:
    """Trading position with liquidation tracking"""
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    volume: float  # Position value in USD
    leverage: int
    tp_price: float
    sl_price: float
    liquidation_price: float
    margin: float  # Initial margin (collateral)
    entry_time: int  # Candle index
    pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    leverage: int
    volume: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # TP, SL, LIQUIDATION, MARKET
    duration_candles: int
    entry_time: int
    exit_time: int


class AVAXScalpingSimulator:
    """AVAXUSDT Scalping simulator with proper liquidation"""

    def __init__(self,
                 initial_balance: float = 10000.0,
                 leverage: int = 100,
                 position_size_pct: float = 5.0,
                 tp_pct: float = 3.0,
                 sl_pct: float = 0.5,
                 strategy: str = 'BOLLINGER'):

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.tp_pct = tp_pct / 100
        self.sl_pct = sl_pct / 100
        self.strategy = strategy

        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.data: Optional[pd.DataFrame] = None

        # Statistics
        self.max_balance = initial_balance
        self.min_balance = initial_balance
        self.max_drawdown = 0.0
        self.liquidations = 0
        self.balance_history = []

    def load_data(self, candles: int = None):
        """Load AVAXUSDT data"""
        file_path = os.path.join(DATA_DIR, "AVAXUSDT_1m.csv")
        if not os.path.exists(file_path):
            print(f"ERROR: {file_path} not found")
            return False

        df = pd.read_csv(file_path)
        if candles:
            df = df.tail(candles).reset_index(drop=True)
        self.data = df
        return True

    def calculate_liquidation_price(self, entry_price: float, direction: int, leverage: int) -> float:
        """
        Calculate liquidation price for isolated margin.

        For Long: Liq Price = Entry * (1 - 1/leverage + maintenance_margin)
        For Short: Liq Price = Entry * (1 + 1/leverage - maintenance_margin)
        """
        if direction == 1:  # Long
            liq_price = entry_price * (1 - (1 / leverage) + MAINTENANCE_MARGIN_RATE)
        else:  # Short
            liq_price = entry_price * (1 + (1 / leverage) - MAINTENANCE_MARGIN_RATE)
        return liq_price

    def check_liquidation(self, position: Position, high: float, low: float) -> bool:
        """Check if position would be liquidated in this candle"""
        if position.direction == 1:  # Long
            # Liquidated if price drops to or below liquidation price
            return low <= position.liquidation_price
        else:  # Short
            # Liquidated if price rises to or above liquidation price
            return high >= position.liquidation_price

    def calculate_signals(self):
        """Calculate trading signals based on strategy"""
        df = self.data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        signals = np.zeros(len(df))

        if self.strategy == 'BOLLINGER':
            # Bollinger Band strategy
            sma = self._sma(close, 20)
            std = self._rolling_std(close, 20)
            upper = sma + 2 * std
            lower = sma - 2 * std

            for i in range(25, len(df)):
                # Long: Price touches lower band and bounces
                if close[i-1] <= lower[i-1] and close[i] > lower[i]:
                    signals[i] = 1
                # Short: Price touches upper band and drops
                elif close[i-1] >= upper[i-1] and close[i] < upper[i]:
                    signals[i] = -1

        elif self.strategy == 'SCALPING':
            # Aggressive scalping: RSI + EMA + Volume
            rsi = self._rsi(close, 7)
            ema_fast = self._ema(close, 5)
            ema_slow = self._ema(close, 13)
            vol_ma = self._sma(volume, 20)

            for i in range(25, len(df)):
                vol_spike = volume[i] > vol_ma[i] * 1.3

                if rsi[i] < 35 and ema_fast[i] > ema_slow[i] and vol_spike:
                    signals[i] = 1
                elif rsi[i] > 65 and ema_fast[i] < ema_slow[i] and vol_spike:
                    signals[i] = -1

        elif self.strategy == 'EMA':
            # EMA crossover
            ema_short = self._ema(close, 9)
            ema_long = self._ema(close, 21)

            for i in range(25, len(df)):
                if ema_short[i] > ema_long[i] and ema_short[i-1] <= ema_long[i-1]:
                    signals[i] = 1
                elif ema_short[i] < ema_long[i] and ema_short[i-1] >= ema_long[i-1]:
                    signals[i] = -1

        return signals

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
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))

        return rsi

    def open_position(self, i: int, direction: int, price: float):
        """Open a new position"""
        # Calculate position size
        margin = self.balance * (self.position_size_pct / 100)
        position_value = margin * self.leverage

        # Entry fee
        entry_fee = position_value * TAKER_FEE
        self.balance -= entry_fee

        # Calculate TP/SL prices
        if direction == 1:  # Long
            tp_price = price * (1 + self.tp_pct)
            sl_price = price * (1 - self.sl_pct)
        else:  # Short
            tp_price = price * (1 - self.tp_pct)
            sl_price = price * (1 + self.sl_pct)

        # Calculate liquidation price
        liq_price = self.calculate_liquidation_price(price, direction, self.leverage)

        self.position = Position(
            symbol='AVAXUSDT',
            direction=direction,
            entry_price=price,
            volume=position_value,
            leverage=self.leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            liquidation_price=liq_price,
            margin=margin,
            entry_time=i
        )

    def close_position(self, i: int, exit_price: float, reason: str):
        """Close current position"""
        pos = self.position
        if not pos:
            return

        # Calculate P/L
        if pos.direction == 1:  # Long
            price_change_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:  # Short
            price_change_pct = (pos.entry_price - exit_price) / pos.entry_price

        # P/L with leverage
        pnl = price_change_pct * pos.volume

        # Exit fee
        exit_fee = pos.volume * TAKER_FEE
        pnl -= exit_fee

        # Handle liquidation - lose entire margin
        if reason == 'LIQUIDATION':
            pnl = -pos.margin
            self.liquidations += 1

        # Update balance
        self.balance += pnl

        # Record trade
        trade = Trade(
            symbol=pos.symbol,
            direction='LONG' if pos.direction == 1 else 'SHORT',
            entry_price=pos.entry_price,
            exit_price=exit_price,
            leverage=pos.leverage,
            volume=pos.volume,
            pnl=pnl,
            pnl_pct=(pnl / pos.margin) * 100 if pos.margin > 0 else 0,
            exit_reason=reason,
            duration_candles=i - pos.entry_time,
            entry_time=pos.entry_time,
            exit_time=i
        )
        self.trades.append(trade)

        # Clear position
        self.position = None

    def run_simulation(self, start_idx: int = 0, end_idx: int = None):
        """Run the scalping simulation"""
        if self.data is None:
            print("ERROR: No data loaded")
            return None

        if end_idx is None:
            end_idx = len(self.data)

        # Pre-calculate signals
        signals = self.calculate_signals()

        close = self.data['close'].values
        high = self.data['high'].values
        low = self.data['low'].values

        # Simulation loop
        for i in range(max(start_idx, 30), end_idx):
            # Track balance history
            self.balance_history.append(self.balance)

            # Update max/min balance and drawdown
            if self.balance > self.max_balance:
                self.max_balance = self.balance
            if self.balance < self.min_balance:
                self.min_balance = self.balance

            dd = (self.max_balance - self.balance) / self.max_balance * 100
            if dd > self.max_drawdown:
                self.max_drawdown = dd

            # Check if we're wiped out
            if self.balance <= 0:
                print(f"  ACCOUNT WIPED OUT at candle {i}")
                break

            # Process existing position
            if self.position:
                pos = self.position
                current_high = high[i]
                current_low = low[i]
                current_close = close[i]

                # Check liquidation first (highest priority)
                if self.check_liquidation(pos, current_high, current_low):
                    self.close_position(i, pos.liquidation_price, 'LIQUIDATION')
                    continue

                # Check TP/SL
                if pos.direction == 1:  # Long
                    if current_high >= pos.tp_price:
                        self.close_position(i, pos.tp_price, 'TP')
                    elif current_low <= pos.sl_price:
                        self.close_position(i, pos.sl_price, 'SL')
                else:  # Short
                    if current_low <= pos.tp_price:
                        self.close_position(i, pos.tp_price, 'TP')
                    elif current_high >= pos.sl_price:
                        self.close_position(i, pos.sl_price, 'SL')

            # Open new position if no current position
            if self.position is None and signals[i] != 0:
                # Check if we have enough balance
                required_margin = self.balance * (self.position_size_pct / 100)
                if required_margin >= 1.0:  # Minimum $1 margin
                    self.open_position(i, int(signals[i]), close[i])

        # Close any remaining position at market
        if self.position:
            self.close_position(end_idx - 1, close[end_idx - 1], 'MARKET')

        return self.get_results()

    def get_results(self) -> Dict:
        """Get simulation results"""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_balance': self.balance,
                'total_pnl': self.balance - self.initial_balance,
                'total_return_pct': 0,
                'win_rate': 0,
                'liquidations': 0,
                'max_drawdown': 0
            }

        wins = len([t for t in self.trades if t.pnl > 0])
        losses = len([t for t in self.trades if t.pnl <= 0])

        tp_exits = len([t for t in self.trades if t.exit_reason == 'TP'])
        sl_exits = len([t for t in self.trades if t.exit_reason == 'SL'])
        liq_exits = len([t for t in self.trades if t.exit_reason == 'LIQUIDATION'])

        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl <= 0]) if losses > 0 else 0

        avg_duration = np.mean([t.duration_candles for t in self.trades])

        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_trades': len(self.trades),
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': (wins / len(self.trades)) * 100,
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'liquidations': liq_exits,
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'total_return_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'max_drawdown': self.max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_duration,
        }

    def print_results(self, title: str = ""):
        """Print formatted results"""
        results = self.get_results()

        print("\n" + "=" * 80)
        print(f"AVAXUSDT SCALPING TEST{' - ' + title if title else ''}")
        print("=" * 80)
        print(f"Strategy: {self.strategy} | Leverage: {self.leverage}x | TP: {self.tp_pct*100:.1f}% | SL: {self.sl_pct*100:.1f}%")
        print(f"Position Size: {self.position_size_pct}% of balance")
        print("-" * 80)

        print(f"\n{'PERFORMANCE':^80}")
        print("-" * 80)
        print(f"Initial Balance:     ${results['initial_balance']:>12,.2f}")
        print(f"Final Balance:       ${results['final_balance']:>12,.2f}")
        print(f"Total P/L:           ${results['total_pnl']:>+12,.2f} ({results['total_return_pct']:+.1f}%)")
        print(f"Max Drawdown:        {results['max_drawdown']:>12.1f}%")
        print(f"Profit Factor:       {results['profit_factor']:>12.2f}")

        print(f"\n{'TRADE STATISTICS':^80}")
        print("-" * 80)
        print(f"Total Trades:        {results['total_trades']:>12}")
        print(f"Winning Trades:      {results['winning_trades']:>12}")
        print(f"Losing Trades:       {results['losing_trades']:>12}")
        print(f"Win Rate:            {results['win_rate']:>12.1f}%")
        print(f"Avg Win:             ${results['avg_win']:>+12,.2f}")
        print(f"Avg Loss:            ${results['avg_loss']:>+12,.2f}")
        print(f"Avg Trade Duration:  {results['avg_trade_duration']:>12.1f} candles")

        print(f"\n{'EXIT BREAKDOWN':^80}")
        print("-" * 80)
        print(f"Take Profit (TP):    {results['tp_exits']:>12}")
        print(f"Stop Loss (SL):      {results['sl_exits']:>12}")
        print(f"LIQUIDATIONS:        {results['liquidations']:>12} {'*** WARNING ***' if results['liquidations'] > 0 else ''}")

        print("=" * 80)

        return results


def run_test(candles: int, title: str, leverage: int = 100, tp_pct: float = 3.0, sl_pct: float = 0.5):
    """Run a single test"""
    sim = AVAXScalpingSimulator(
        initial_balance=10000.0,
        leverage=leverage,
        position_size_pct=5.0,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        strategy='BOLLINGER'
    )

    if not sim.load_data(candles):
        return None

    print(f"\nLoading {candles:,} candles ({candles/60:.1f} hours / {candles/1440:.1f} days)")

    sim.run_simulation()
    return sim.print_results(title)


def main():
    print("\n" + "=" * 80)
    print("AVAXUSDT SCALPING TEST WITH PROPER LIQUIDATION")
    print("=" * 80)
    print("\nBTCC Liquidation Rules:")
    print("  - At 100x: ~1% adverse move = LIQUIDATION")
    print("  - At 50x: ~2% adverse move = LIQUIDATION")
    print("  - At 20x: ~5% adverse move = LIQUIDATION")
    print("  - Maintenance margin: 0.5%")
    print("=" * 80)

    all_results = []

    # Test configurations - comparing 10x vs 20x leverage
    configs = [
        # 1 DAY - best config
        (1440, "1 DAY | 20x | TP:1.5% SL:0.6%", 20, 1.5, 0.6),
        (1440, "1 DAY | 10x | TP:1.5% SL:0.6%", 10, 1.5, 0.6),
        (1440, "1 DAY | 10x | TP:2% SL:1%", 10, 2.0, 1.0),
        # 7 DAY tests
        (10080, "7 DAYS | 10x | TP:1.5% SL:0.6%", 10, 1.5, 0.6),
        (10080, "7 DAYS | 10x | TP:2% SL:1%", 10, 2.0, 1.0),
        (10080, "7 DAYS | 10x | TP:3% SL:1.5%", 10, 3.0, 1.5),
        # 30 DAY tests with 10x (safest)
        (43200, "30 DAYS | 10x | TP:1.5% SL:0.6%", 10, 1.5, 0.6),
        (43200, "30 DAYS | 10x | TP:2% SL:1%", 10, 2.0, 1.0),
        (43200, "30 DAYS | 10x | TP:3% SL:1.5%", 10, 3.0, 1.5),
        # Bonus: try 10x with tighter TP for faster scalping
        (43200, "30 DAYS | 10x | TP:1% SL:0.5%", 10, 1.0, 0.5),
    ]

    for i, (candles, title, leverage, tp_pct, sl_pct) in enumerate(configs):
        print(f"\n{'#' * 80}")
        print(f"TEST {i+1}: {title}")
        print("#" * 80)
        result = run_test(candles, title, leverage, tp_pct, sl_pct)
        if result:
            result['config'] = title
            all_results.append(result)

    # Summary comparison
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON - ALL CONFIGURATIONS")
    print("=" * 100)
    print(f"{'Configuration':<35} {'Trades':>7} {'Win%':>7} {'Return%':>10} {'Max DD':>9} {'Liqs':>6} {'PF':>7}")
    print("-" * 100)

    for r in all_results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "INF"
        liq_warn = " ***" if r['liquidations'] > 0 else ""
        print(f"{r['config']:<35} {r['total_trades']:>7} {r['win_rate']:>6.1f}% {r['total_return_pct']:>+9.1f}% {r['max_drawdown']:>8.1f}% {r['liquidations']:>5}{liq_warn} {pf_str:>7}")

    print("=" * 100)

    # Find best config (profitable, no liquidations)
    profitable_safe = [r for r in all_results if r['total_pnl'] > 0 and r['liquidations'] == 0]
    if profitable_safe:
        best = max(profitable_safe, key=lambda x: x['total_return_pct'])
        print(f"\nBEST SAFE CONFIG: {best['config']}")
        print(f"  Return: {best['total_return_pct']:+.1f}% | Win Rate: {best['win_rate']:.1f}% | Max DD: {best['max_drawdown']:.1f}%")
    else:
        # Find least bad
        no_liq = [r for r in all_results if r['liquidations'] == 0]
        if no_liq:
            best = max(no_liq, key=lambda x: x['total_return_pct'])
            print(f"\nLEAST LOSS (NO LIQUIDATION): {best['config']}")
            print(f"  Return: {best['total_return_pct']:+.1f}% | Win Rate: {best['win_rate']:.1f}%")

    print("=" * 100)


if __name__ == "__main__":
    main()
