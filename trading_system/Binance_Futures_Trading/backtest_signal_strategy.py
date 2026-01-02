#!/usr/bin/env python3
"""
Backtest Signal-Based Strategy v1
==================================
Signal-based entries using ADX + Volatility on 1m timeframe
Confirmed by higher timeframe (15m or 1h) trend direction

Strategy:
- Wait for entry signal (no always-in hedge mode)
- Entry conditions:
  1. ADX > threshold on 1m (strong trend)
  2. Volatility spike (ATR expansion)
  3. Higher timeframe confirms direction
- Single direction trade (LONG or SHORT based on signal)
- TP/SL based on ATR
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import STRATEGY_CONFIG


class SignalBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Position sizing - 10% per trade
        self.position_pct = 0.10  # 10% of balance per trade

        # TP settings - NO SL (isolated margin = margin is the SL)
        # With 20x leverage: 15% ROI = 0.75% price move
        self.tp_roi = 0.15  # 15% ROI target
        self.use_sl = False  # NO SL - isolated margin handles it

        # ADX settings
        self.adx_period = 14
        self.adx_threshold = 25  # ADX > 25 = trending market
        self.adx_strong = 40     # ADX > 40 = strong trend

        # Volatility settings (ATR)
        self.atr_period = 14
        self.atr_expansion_mult = 1.5  # ATR must be 1.5x its 20-period average

        # Higher timeframe settings
        self.htf_period = 15  # 15 candles of 1m = 15m equivalent
        self.htf_ema_fast = 9
        self.htf_ema_slow = 21

        # Position tracking
        self.position = None  # Current position (LONG or SHORT)

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance

        # Signal stats
        self.signals_generated = 0
        self.signals_taken = 0
        self.long_signals = 0
        self.short_signals = 0

        print(f"[{symbol}] Signal Strategy | ADX > {self.adx_threshold} | TP: {self.tp_roi*100:.0f}% ROI | SL: NONE (isolated margin)")

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """
        Calculate ADX (Average Directional Index)
        Returns: (adx, plus_di, minus_di)
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # When +DM > -DM, -DM = 0 and vice versa
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0

        # Smoothed TR, +DM, -DM using Wilder's smoothing
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()

        return adx, plus_di, minus_di

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def get_htf_trend(self, df: pd.DataFrame, idx: int) -> str:
        """
        Get higher timeframe trend direction using EMA crossover
        on aggregated candles (simulates 15m from 1m data)
        Returns: "UP", "DOWN", or "NEUTRAL"
        """
        if idx < self.htf_period * self.htf_ema_slow:
            return "NEUTRAL"

        # Get last htf_period candles and aggregate
        start_idx = max(0, idx - self.htf_period + 1)
        htf_close = df['close'].iloc[start_idx:idx+1].mean()

        # Use rolling EMAs on close prices
        close_series = df['close'].iloc[:idx+1]
        ema_fast = self.calculate_ema(close_series, self.htf_ema_fast * self.htf_period)
        ema_slow = self.calculate_ema(close_series, self.htf_ema_slow * self.htf_period)

        if pd.isna(ema_fast.iloc[-1]) or pd.isna(ema_slow.iloc[-1]):
            return "NEUTRAL"

        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]

        # Determine trend
        diff_pct = (fast_val - slow_val) / slow_val * 100

        if diff_pct > 0.1:  # Fast EMA above slow by 0.1%
            return "UP"
        elif diff_pct < -0.1:  # Fast EMA below slow by 0.1%
            return "DOWN"
        else:
            return "NEUTRAL"

    def check_entry_signal(self, idx: int, df: pd.DataFrame,
                           adx: float, plus_di: float, minus_di: float,
                           atr: float, atr_avg: float) -> tuple:
        """
        Check for entry signal
        Returns: (signal_type, direction) where signal_type is "ENTRY" or None
        """
        # Condition 1: ADX above threshold (trending market)
        if adx < self.adx_threshold:
            return None, None

        # Condition 2: Volatility expansion (ATR > 1.5x average)
        if atr < atr_avg * self.atr_expansion_mult:
            return None, None

        # Condition 3: Get higher timeframe trend
        htf_trend = self.get_htf_trend(df, idx)
        if htf_trend == "NEUTRAL":
            return None, None

        # Condition 4: DI confirms direction matches HTF
        if htf_trend == "UP" and plus_di > minus_di:
            return "ENTRY", "LONG"
        elif htf_trend == "DOWN" and minus_di > plus_di:
            return "ENTRY", "SHORT"

        return None, None

    def open_position(self, side: str, price: float, atr: float, timestamp) -> dict:
        """Open new position with ROI-based TP, NO SL (isolated margin)"""
        margin = self.balance * self.position_pct
        quantity = (margin * self.leverage) / price

        # Calculate TP based on ROI target (15% ROI = 0.75% price move with 20x)
        price_move_pct = self.tp_roi / self.leverage  # 0.15 / 20 = 0.0075 = 0.75%
        if side == "LONG":
            tp_price = price * (1 + price_move_pct)
            # Liquidation at ~5% price drop (100% loss of margin)
            liq_price = price * (1 - 1/self.leverage)  # ~95% of entry
        else:
            tp_price = price * (1 - price_move_pct)
            # Liquidation at ~5% price rise
            liq_price = price * (1 + 1/self.leverage)  # ~105% of entry

        position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": margin,
            "tp_price": tp_price,
            "liq_price": liq_price,  # Liquidation price (isolated margin SL)
            "atr_at_entry": atr,
            "entry_time": timestamp
        }

        self.signals_taken += 1
        if side == "LONG":
            self.long_signals += 1
        else:
            self.short_signals += 1

        print(f"[{timestamp}] OPEN {side} @ ${price:.4f} | TP: ${tp_price:.4f} (+{self.tp_roi*100:.0f}% ROI) | LIQ: ${liq_price:.4f}")

        return position

    def close_position(self, exit_price: float, exit_type: str, timestamp) -> float:
        """Close position and calculate PnL"""
        if self.position is None:
            return 0

        price_change_pct = (exit_price - self.position["entry_price"]) / self.position["entry_price"]

        if self.position["side"] == "LONG":
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        pnl = self.position["margin"] * roi

        # Cap loss at margin
        if pnl < -self.position["margin"]:
            pnl = -self.position["margin"]

        self.balance += pnl
        self.total_pnl += pnl

        if self.balance <= 0:
            self.balance = 0

        if pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1

        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.trades.append({
            "timestamp": timestamp,
            "side": self.position["side"],
            "entry_price": self.position["entry_price"],
            "exit_price": exit_price,
            "quantity": self.position["quantity"],
            "margin": self.position["margin"],
            "pnl": pnl,
            "roi": roi * 100,
            "exit_type": exit_type,
            "balance": self.balance,
            "atr": self.position["atr_at_entry"]
        })

        print(f"[{timestamp}] CLOSE {self.position['side']} @ ${exit_price:.4f} | {exit_type} | P&L: ${pnl:+.2f} | ROI: {roi*100:+.1f}%")

        self.position = None
        return pnl

    def get_historical_data(self, days: int = 30, interval: str = "1m"):
        """Fetch historical klines from Binance MAINNET"""
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")
        print("Using Binance MAINNET for real historical data...")

        client = BinanceClient(testnet=False, use_demo=False)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Paginate to get all data
        all_data = []
        current_start = start_time

        # For 1m data, limit is still 1000
        interval_minutes = 1 if interval == "1m" else (5 if interval == "5m" else 60)

        while current_start < end_time:
            df_chunk = client.get_klines(
                self.symbol,
                interval=interval,
                start_time=int(current_start.timestamp() * 1000),
                limit=1000
            )

            if df_chunk is None or len(df_chunk) == 0:
                break

            all_data.append(df_chunk)

            if len(df_chunk) < 1000:
                break

            last_time = df_chunk.index[-1]
            current_start = last_time + timedelta(minutes=interval_minutes)

        if not all_data:
            print("No data returned!")
            return None

        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        print(f"Price change over period: {price_change:+.2f}%")

        return df

    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest with signal-based entries"""
        print("\n" + "="*70)
        print("RUNNING SIGNAL-BASED STRATEGY BACKTEST")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"Position Size: {self.position_pct*100:.0f}% of balance")
        print(f"ADX Threshold: {self.adx_threshold}")
        print(f"ATR Expansion: {self.atr_expansion_mult}x average")
        print(f"TP: {self.tp_roi*100:.0f}% ROI | SL: NONE (Isolated Margin = Liquidation)")
        print("="*70)

        # Calculate indicators
        print("Calculating indicators...")
        adx_series, plus_di_series, minus_di_series = self.calculate_adx(df, self.adx_period)
        atr_series = self.calculate_atr(df, self.atr_period)
        atr_avg_series = atr_series.rolling(window=20).mean()

        # Warmup period
        warmup = max(self.adx_period * 2, self.htf_period * self.htf_ema_slow, 50)

        print(f"Starting from candle {warmup} (warmup period)")

        # Iterate through candles
        for i in range(warmup, len(df)):
            timestamp = df.index[i]
            row = df.iloc[i]

            if self.balance <= 0:
                print(f"[{timestamp}] ACCOUNT BLOWN - Balance depleted!")
                break

            high = row['high']
            low = row['low']
            close = row['close']

            adx = adx_series.iloc[i]
            plus_di = plus_di_series.iloc[i]
            minus_di = minus_di_series.iloc[i]
            atr = atr_series.iloc[i]
            atr_avg = atr_avg_series.iloc[i]

            if pd.isna(adx) or pd.isna(atr) or pd.isna(atr_avg):
                continue

            # Check existing position
            if self.position is not None:
                # Check TP and Liquidation (NO SL - isolated margin handles it)
                if self.position["side"] == "LONG":
                    if high >= self.position["tp_price"]:
                        self.close_position(self.position["tp_price"], "TP", timestamp)
                        continue
                    elif low <= self.position["liq_price"]:
                        # Liquidation - lose entire margin
                        self.close_position(self.position["liq_price"], "LIQUIDATION", timestamp)
                        continue
                else:  # SHORT
                    if low <= self.position["tp_price"]:
                        self.close_position(self.position["tp_price"], "TP", timestamp)
                        continue
                    elif high >= self.position["liq_price"]:
                        # Liquidation - lose entire margin
                        self.close_position(self.position["liq_price"], "LIQUIDATION", timestamp)
                        continue

            # Check for entry signal (only if no position)
            if self.position is None:
                signal_type, direction = self.check_entry_signal(
                    i, df, adx, plus_di, minus_di, atr, atr_avg
                )

                if signal_type == "ENTRY" and direction:
                    self.signals_generated += 1
                    self.position = self.open_position(direction, close, atr, timestamp)

        self.print_results(df)
        self.export_trade_journal()

        return {
            "balance": self.balance,
            "return_pct": (self.balance - self.start_balance) / self.start_balance * 100,
            "win_rate": (self.total_wins / len(self.trades) * 100) if self.trades else 0,
            "max_drawdown": self.max_drawdown,
            "total_trades": len(self.trades),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "signals_generated": self.signals_generated,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals
        }

    def export_trade_journal(self, filename: str = None):
        """Export all trades to CSV"""
        if not self.trades:
            print("No trades to export")
            return

        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signal_journal_{self.symbol}_{timestamp_str}.csv"

        fieldnames = [
            "timestamp", "symbol", "side", "entry_price", "exit_price",
            "quantity", "margin", "pnl", "roi", "exit_type",
            "balance_after", "atr"
        ]

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for trade in self.trades:
                writer.writerow({
                    "timestamp": trade["timestamp"],
                    "symbol": self.symbol,
                    "side": trade["side"],
                    "entry_price": f"{trade['entry_price']:.6f}",
                    "exit_price": f"{trade['exit_price']:.6f}",
                    "quantity": f"{trade['quantity']:.6f}",
                    "margin": f"{trade['margin']:.2f}",
                    "pnl": f"{trade['pnl']:.2f}",
                    "roi": f"{trade['roi']:.2f}",
                    "exit_type": trade["exit_type"],
                    "balance_after": f"{trade['balance']:.2f}",
                    "atr": f"{trade['atr']:.6f}"
                })

        print(f"\n>>> Trade journal exported to: {filename}")

    def print_results(self, df: pd.DataFrame):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - SIGNAL-BASED STRATEGY")
        print("="*70)

        price_start = df['close'].iloc[0]
        price_end = df['close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100

        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\nMarket Performance:")
        print(f"  Price: ${price_start:.4f} -> ${price_end:.4f} ({price_change:+.2f}%)")
        print(f"  Period: {df.index[0]} to {df.index[-1]}")

        print(f"\nStrategy Performance:")
        print(f"  Starting Balance: ${self.start_balance:.2f}")
        print(f"  Ending Balance:   ${self.balance:.2f}")
        print(f"  Total P&L:        ${self.total_pnl:+.2f}")
        print(f"  Total Return:     {((self.balance - self.start_balance) / self.start_balance * 100):+.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Wins:             {self.total_wins}")
        print(f"  Losses:           {self.total_losses}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Max Drawdown:     {self.max_drawdown:.2f}%")

        print(f"\nSignal Statistics:")
        print(f"  Signals Generated: {self.signals_generated}")
        print(f"  Long Signals:      {self.long_signals}")
        print(f"  Short Signals:     {self.short_signals}")

        if total_trades > 0:
            avg_pnl = self.total_pnl / total_trades
            winning_trades = [t for t in self.trades if t["pnl"] > 0]
            losing_trades = [t for t in self.trades if t["pnl"] <= 0]
            avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0

            print(f"\nP&L Analysis:")
            print(f"  Avg P&L per Trade: ${avg_pnl:+.2f}")
            print(f"  Avg Win:           ${avg_win:+.2f}")
            print(f"  Avg Loss:          ${avg_loss:+.2f}")

            if avg_loss != 0:
                profit_factor = abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades))
                print(f"  Profit Factor:     {profit_factor:.2f}x")

        print("\n" + "="*70)


def run_signal_test():
    """Test signal-based strategy"""
    # Configuration
    BACKTEST_DAYS = 7  # 7 days of 1m data (good amount for testing)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print("="*80)
    print(f"SIGNAL-BASED STRATEGY BACKTEST - {BACKTEST_DAYS} DAYS (1m candles)")
    print("="*80)
    print("Strategy: ADX + Volatility signals with HTF confirmation")
    print("  - Entry: ADX > 25, ATR expansion, HTF trend alignment")
    print("  - Exit: 15% ROI TP | NO SL (Isolated Margin = auto liquidation)")
    print("="*80)

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        backtester = SignalBacktester(symbol, start_balance=100.0)
        df = backtester.get_historical_data(days=BACKTEST_DAYS, interval="1m")

        if df is not None and len(df) > 0:
            result = backtester.run_backtest(df)
            result["symbol"] = symbol
            all_results.append(result)
        else:
            print(f"ERROR: No data for {symbol}")

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY - SIGNAL-BASED STRATEGY")
    print("="*100)
    print(f"{'Symbol':<12} {'Start':>10} {'End':>10} {'P&L':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>10}")
    print("-"*90)

    for r in all_results:
        pnl = r['balance'] - 100
        print(f"{r['symbol']:<12} ${100:>9.2f} ${r['balance']:>9.2f} ${pnl:>+9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>9.1f}%")

    print("-"*90)
    total_start = len(all_results) * 100
    total_end = sum(r['balance'] for r in all_results)
    total_pnl = total_end - total_start
    total_return = (total_end / total_start - 1) * 100 if total_start > 0 else 0
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_winrate = sum(r['win_rate'] for r in all_results) / len(all_results) if all_results else 0
    max_dd = max(r['max_drawdown'] for r in all_results) if all_results else 0

    print(f"{'TOTAL':<12} ${total_start:>9.2f} ${total_end:>9.2f} ${total_pnl:>+9.2f} {total_return:>+9.1f}% {total_trades:>7} {avg_winrate:>7.1f}% {max_dd:>9.1f}%")

    return all_results


if __name__ == "__main__":
    run_signal_test()
