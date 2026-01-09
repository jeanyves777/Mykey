#!/usr/bin/env python3
"""
HTF Confluence Backtest with Trailing Stop
===========================================
Compare fixed TP/SL vs trailing stop strategies.

Trailing Stop Options:
1. Simple Trailing: Move SL up as price moves in favor
2. Activation Trailing: Only start trailing after X% profit
3. Step Trailing: Move SL in steps (e.g., every 0.5% gain)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from strategies.htf_confluence_strategy import (
    HTFConfluenceStrategy,
    MODERATE_CONFIG,
    TrendDirection,
    SignalStrength
)


class TrailingStopBacktester:
    """Backtester with trailing stop support"""

    def __init__(self, symbol: str, start_balance: float = 100.0, config: dict = None,
                 trailing_mode: str = "none", trailing_activation: float = 0.10,
                 trailing_distance: float = 0.05):
        """
        Args:
            symbol: Trading pair
            start_balance: Starting capital
            config: Strategy config (leverage, tp_roi, sl_roi)
            trailing_mode: "none", "simple", "activation", "step"
            trailing_activation: ROI % to activate trailing (for activation mode)
            trailing_distance: Distance to trail behind peak (as ROI %)
        """
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.peak_balance = start_balance
        self.max_drawdown = 0

        # Config
        config = config or MODERATE_CONFIG
        self.leverage = config["leverage"]
        self.tp_roi = config["tp_roi"]
        self.sl_roi = config["sl_roi"]

        # Trailing stop settings
        self.trailing_mode = trailing_mode
        self.trailing_activation = trailing_activation  # ROI to activate trailing
        self.trailing_distance = trailing_distance      # Trail distance as ROI %

        # Strategy
        self.strategy = HTFConfluenceStrategy(**config)

        # Trading costs
        self.maker_fee = 0.00006
        self.slippage = 0.0001
        self.risk_per_trade = 0.02

        # Position tracking
        self.position = None

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.total_fees_paid = 0

        # Trailing stats
        self.trailing_activated_count = 0
        self.trailing_improved_exits = 0

    def get_historical_data(self, days: int = 60, ltf_interval: str = "15m"):
        """Fetch historical data using Yahoo Finance"""
        yf_ticker_map = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "BNBUSDT": "BNB-USD",
            "SOLUSDT": "SOL-USD",
            "DOTUSDT": "DOT-USD",
        }

        yf_symbol = yf_ticker_map.get(self.symbol, self.symbol.replace("USDT", "-USD"))

        try:
            ticker = yf.Ticker(yf_symbol)

            ltf_days = min(days, 60) if ltf_interval == "15m" else days
            ltf_df = ticker.history(period=f"{ltf_days}d", interval=ltf_interval)

            if ltf_df is None or len(ltf_df) == 0:
                return None, None

            ltf_df.columns = [c.lower() for c in ltf_df.columns]
            ltf_df = ltf_df[['open', 'high', 'low', 'close', 'volume']]

            htf_days = min(days + 50, 730)
            htf_raw = ticker.history(period=f"{htf_days}d", interval="1h")

            if htf_raw is None or len(htf_raw) == 0:
                return None, None

            htf_raw.columns = [c.lower() for c in htf_raw.columns]
            htf_raw = htf_raw[['open', 'high', 'low', 'close', 'volume']]

            htf_df = htf_raw.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            return ltf_df, htf_df

        except Exception as e:
            print(f"[ERROR] Data fetch failed: {e}")
            return None, None

    def open_position(self, side: str, price: float, timestamp, signal) -> dict:
        """Open new position"""
        if side == "LONG":
            actual_entry = price * (1 + self.slippage)
        else:
            actual_entry = price * (1 - self.slippage)

        sl_pct = self.sl_roi / self.leverage
        risk_amount = self.balance * self.risk_per_trade
        position_value = risk_amount / sl_pct
        margin = position_value / self.leverage
        quantity = position_value / actual_entry

        entry_fee = position_value * self.maker_fee
        self.balance -= entry_fee
        self.total_fees_paid += entry_fee

        stop_loss, take_profit = self.strategy.calculate_exit_levels(actual_entry, side)

        position = {
            "side": side,
            "entry_price": actual_entry,
            "quantity": quantity,
            "margin": margin,
            "tp_price": take_profit,
            "sl_price": stop_loss,
            "original_sl": stop_loss,
            "entry_time": timestamp,
            "peak_price": actual_entry,  # Track best price for trailing
            "trailing_active": False,
            "entry_fee": entry_fee
        }

        return position

    def update_trailing_stop(self, current_high: float, current_low: float) -> bool:
        """
        Update trailing stop based on current price.

        Returns:
            True if trailing stop was updated
        """
        if self.position is None or self.trailing_mode == "none":
            return False

        is_long = self.position["side"] in ["LONG", "BUY"]
        entry = self.position["entry_price"]

        # Calculate current ROI
        if is_long:
            current_best = current_high
            price_move = (current_best - entry) / entry
        else:
            current_best = current_low
            price_move = (entry - current_best) / entry

        current_roi = price_move * self.leverage

        # Update peak price
        if is_long and current_high > self.position["peak_price"]:
            self.position["peak_price"] = current_high
        elif not is_long and current_low < self.position["peak_price"]:
            self.position["peak_price"] = current_low

        # Check if trailing should activate
        if self.trailing_mode == "simple":
            # Simple: Always trail from entry
            self.position["trailing_active"] = True

        elif self.trailing_mode == "activation":
            # Activation: Only trail after reaching activation threshold
            if current_roi >= self.trailing_activation and not self.position["trailing_active"]:
                self.position["trailing_active"] = True
                self.trailing_activated_count += 1

        elif self.trailing_mode == "step":
            # Step: Trail in discrete steps
            if current_roi >= self.trailing_activation:
                self.position["trailing_active"] = True

        # Update trailing stop if active
        if self.position["trailing_active"]:
            peak = self.position["peak_price"]
            trail_distance_price = peak * (self.trailing_distance / self.leverage)

            if is_long:
                new_sl = peak - trail_distance_price
                if new_sl > self.position["sl_price"]:
                    self.position["sl_price"] = new_sl
                    return True
            else:
                new_sl = peak + trail_distance_price
                if new_sl < self.position["sl_price"]:
                    self.position["sl_price"] = new_sl
                    return True

        return False

    def close_position(self, exit_price: float, exit_type: str, timestamp) -> float:
        """Close position and calculate P&L"""
        if self.position is None:
            return 0

        is_long = self.position["side"] in ["LONG", "BUY"]
        if is_long:
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_exit = exit_price * (1 + self.slippage)

        position_value = self.position["quantity"] * actual_exit
        exit_fee = position_value * self.maker_fee
        self.total_fees_paid += exit_fee

        price_change_pct = (actual_exit - self.position["entry_price"]) / self.position["entry_price"]

        if is_long:
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        gross_pnl = self.position["margin"] * roi
        pnl = gross_pnl - exit_fee

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

        # Track if trailing improved the exit
        if self.position["trailing_active"] and exit_type == "TRAILING_SL":
            original_sl_roi = self.sl_roi * 100
            actual_roi = roi * 100
            if actual_roi > -original_sl_roi:
                self.trailing_improved_exits += 1

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
            "exit_price": actual_exit,
            "margin": self.position["margin"],
            "pnl": pnl,
            "roi": roi * 100,
            "exit_type": exit_type,
            "trailing_active": self.position["trailing_active"],
            "sl_moved": self.position["sl_price"] != self.position["original_sl"]
        })

        self.position = None
        return pnl

    def run_backtest(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """Run backtest with trailing stop"""
        warmup = max(30, self.strategy.ema_slow * 2)

        for i in range(warmup, len(ltf_df)):
            timestamp = ltf_df.index[i]
            row = ltf_df.iloc[i]

            if self.balance <= 0:
                break

            high = row['high']
            low = row['low']
            close = row['close']

            htf_available = htf_df[htf_df.index <= timestamp]

            if len(htf_available) < self.strategy.htf_ema_period:
                continue

            # Check existing position
            if self.position is not None:
                is_long = self.position["side"] in ["LONG", "BUY"]

                # Update trailing stop first
                self.update_trailing_stop(high, low)

                # Check exits
                if is_long:
                    # Check TP first
                    if high >= self.position["tp_price"]:
                        self.close_position(self.position["tp_price"], "TP", timestamp)
                        continue
                    # Check SL (could be trailing or original)
                    elif low <= self.position["sl_price"]:
                        exit_type = "TRAILING_SL" if self.position["trailing_active"] else "SL"
                        self.close_position(self.position["sl_price"], exit_type, timestamp)
                        continue
                else:
                    if low <= self.position["tp_price"]:
                        self.close_position(self.position["tp_price"], "TP", timestamp)
                        continue
                    elif high >= self.position["sl_price"]:
                        exit_type = "TRAILING_SL" if self.position["trailing_active"] else "SL"
                        self.close_position(self.position["sl_price"], exit_type, timestamp)
                        continue

            # Check for entry signal
            if self.position is None:
                ltf_available = ltf_df.iloc[:i+1]
                signal = self.strategy.should_enter(ltf_available, htf_available, current_bar=i)

                if signal.action:
                    self.position = self.open_position(signal.action, close, timestamp, signal)

        # Close any open position
        if self.position is not None:
            self.close_position(ltf_df['close'].iloc[-1], "END", ltf_df.index[-1])

        return self.get_results()

    def get_results(self) -> dict:
        """Get backtest results"""
        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0

        # Count exit types
        tp_exits = sum(1 for t in self.trades if t["exit_type"] == "TP")
        sl_exits = sum(1 for t in self.trades if t["exit_type"] == "SL")
        trail_exits = sum(1 for t in self.trades if t["exit_type"] == "TRAILING_SL")

        return {
            "symbol": self.symbol,
            "trailing_mode": self.trailing_mode,
            "balance": self.balance,
            "return_pct": (self.balance - self.start_balance) / self.start_balance * 100,
            "total_trades": total_trades,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "tp_exits": tp_exits,
            "sl_exits": sl_exits,
            "trail_exits": trail_exits,
            "trailing_activated": self.trailing_activated_count,
            "trailing_improved": self.trailing_improved_exits,
            "total_fees": self.total_fees_paid
        }


def compare_trailing_strategies(symbols: list = None, days: int = 60):
    """Compare different trailing stop strategies"""
    if symbols is None:
        symbols = ["DOTUSDT", "BNBUSDT"]

    print("=" * 100)
    print("TRAILING STOP COMPARISON TEST")
    print("=" * 100)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {days} days")
    print(f"Config: MODERATE (20x leverage, 30% TP ROI, 10% SL ROI)")
    print("=" * 100)

    # Test configurations
    configs = [
        {"name": "NO TRAILING (Baseline)", "mode": "none", "activation": 0, "distance": 0},
        {"name": "Simple Trail (5% ROI)", "mode": "simple", "activation": 0, "distance": 0.05},
        {"name": "Activation 10%, Trail 5%", "mode": "activation", "activation": 0.10, "distance": 0.05},
        {"name": "Activation 15%, Trail 5%", "mode": "activation", "activation": 0.15, "distance": 0.05},
        {"name": "Activation 10%, Trail 3%", "mode": "activation", "activation": 0.10, "distance": 0.03},
        {"name": "Activation 20%, Trail 8%", "mode": "activation", "activation": 0.20, "distance": 0.08},
    ]

    all_results = []

    for cfg in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*80}")

        config_results = []

        for symbol in symbols:
            print(f"\n  {symbol}...", end=" ", flush=True)

            backtester = TrailingStopBacktester(
                symbol=symbol,
                start_balance=20.0,
                config=MODERATE_CONFIG,
                trailing_mode=cfg["mode"],
                trailing_activation=cfg["activation"],
                trailing_distance=cfg["distance"]
            )

            ltf_df, htf_df = backtester.get_historical_data(days=days)

            if ltf_df is not None and htf_df is not None:
                result = backtester.run_backtest(ltf_df, htf_df)
                config_results.append(result)
                print(f"${result['balance']:.2f} ({result['return_pct']:+.1f}%) | {result['total_trades']} trades | WR: {result['win_rate']:.0f}%")
            else:
                print("FAILED")

        if config_results:
            # Aggregate results for this config
            total_start = len(config_results) * 20.0
            total_end = sum(r['balance'] for r in config_results)
            total_return = (total_end / total_start - 1) * 100
            total_trades = sum(r['total_trades'] for r in config_results)
            avg_winrate = sum(r['win_rate'] for r in config_results) / len(config_results)
            max_dd = max(r['max_drawdown'] for r in config_results)
            tp_exits = sum(r['tp_exits'] for r in config_results)
            sl_exits = sum(r['sl_exits'] for r in config_results)
            trail_exits = sum(r['trail_exits'] for r in config_results)

            all_results.append({
                "name": cfg["name"],
                "total_end": total_end,
                "return_pct": total_return,
                "total_trades": total_trades,
                "win_rate": avg_winrate,
                "max_drawdown": max_dd,
                "tp_exits": tp_exits,
                "sl_exits": sl_exits,
                "trail_exits": trail_exits
            })

    # Print comparison
    print("\n" + "=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)
    print(f"{'Strategy':<30} {'End $':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>8} {'TP':>6} {'SL':>6} {'Trail':>6}")
    print("-" * 120)

    for r in sorted(all_results, key=lambda x: x['return_pct'], reverse=True):
        print(f"{r['name']:<30} ${r['total_end']:>9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>7.1f}% {r['tp_exits']:>5} {r['sl_exits']:>5} {r['trail_exits']:>5}")

    print("=" * 120)

    # Best strategy
    best = max(all_results, key=lambda x: x['return_pct'])
    baseline = next((r for r in all_results if "Baseline" in r['name']), all_results[0])

    print(f"\nBEST STRATEGY: {best['name']}")
    print(f"  Return: {best['return_pct']:+.1f}% vs Baseline: {baseline['return_pct']:+.1f}%")
    print(f"  Improvement: {best['return_pct'] - baseline['return_pct']:+.1f}%")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest with Trailing Stop")
    parser.add_argument("--symbols", nargs="+", default=["DOTUSDT", "BNBUSDT"],
                        help="Symbols to backtest")
    parser.add_argument("--days", type=int, default=60, help="Backtest period")

    args = parser.parse_args()

    compare_trailing_strategies(symbols=args.symbols, days=args.days)
