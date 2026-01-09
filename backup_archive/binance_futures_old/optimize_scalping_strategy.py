#!/usr/bin/env python3
"""
Scalping Strategy Optimization for DOTUSDT
==========================================
Tests different TP levels and hedge mode (simultaneous long/short) strategies.
Keeps DCA system intact, just optimizes scalping parameters.

Strategies to test:
1. Different TP levels (0.5% to 5% ROI)
2. Hedge Mode: Hold both LONG and SHORT simultaneously, exit whichever profits
3. Quick scalping with tight SL/TP
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient
from engine.momentum_signal import MasterMomentumSignal
from config.trading_config import STRATEGY_CONFIG, RISK_CONFIG, BACKTEST_CONFIG


@dataclass
class ScalpPosition:
    """Position for scalping backtest"""
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    margin_used: float
    dca_count: int = 0
    avg_entry_price: float = 0.0


@dataclass
class HedgePosition:
    """Hedge position (both LONG and SHORT at same time)"""
    symbol: str
    long_entry: float
    short_entry: float
    quantity: float
    entry_time: datetime
    margin_used: float
    long_tp: float
    long_sl: float
    short_tp: float
    short_sl: float


@dataclass
class OptimizationResult:
    """Result for one parameter combination"""
    strategy: str
    tp_pct: float
    sl_pct: float
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_trade_pnl: float
    profit_factor: float
    max_drawdown: float
    avg_hold_time_minutes: float
    extra_params: Dict = field(default_factory=dict)


class ScalpingOptimizer:
    """
    Optimize scalping strategy parameters for DOTUSDT
    """

    def __init__(self, symbol: str = "DOTUSDT"):
        self.symbol = symbol
        self.client = BinanceClient(testnet=True)
        self.signal_generator = MasterMomentumSignal()

        # Base config
        self.initial_balance = BACKTEST_CONFIG["initial_balance"]
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x
        self.commission = BACKTEST_CONFIG["commission_per_trade"]
        self.slippage = BACKTEST_CONFIG["slippage_pct"]

    def fetch_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical 1-minute data"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        print(f"Fetching {self.symbol} data for {days} days...")
        df = self.client.get_historical_klines(self.symbol, "1m", start_date, end_date)
        print(f"Got {len(df)} candles")
        return df

    def run_standard_scalping_test(self, df: pd.DataFrame, tp_roi: float, sl_roi: float) -> OptimizationResult:
        """
        Test standard scalping with specific TP/SL ROI levels

        Args:
            df: Historical data
            tp_roi: Take profit ROI (e.g., 0.02 = 2% ROI, with 20x leverage = 0.1% price move)
            sl_roi: Stop loss ROI (e.g., 0.10 = 10% ROI)
        """
        balance = self.initial_balance
        position: Optional[ScalpPosition] = None
        trades = []
        equity_curve = [balance]
        peak_balance = balance
        max_drawdown = 0.0

        # Convert ROI to price movement (ROI = price_move * leverage)
        tp_price_pct = tp_roi / self.leverage
        sl_price_pct = sl_roi / self.leverage

        # Entry sizing: 20% of balance as margin
        entry_margin_pct = 0.20

        for i in range(100, len(df)):
            current_bar = df.iloc[i]
            lookback = df.iloc[i-100:i+1]
            current_price = current_bar["close"]
            bar_time = df.index[i]

            # If in position, check exit
            if position is not None:
                high = current_bar["high"]
                low = current_bar["low"]

                exit_price = None
                exit_reason = None

                if position.side == "LONG":
                    if low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    elif high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"
                else:  # SHORT
                    if high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    elif low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"

                if exit_price:
                    # Apply slippage
                    if position.side == "LONG":
                        exit_price *= (1 - self.slippage)
                        pnl = (exit_price - position.entry_price) * position.quantity
                    else:
                        exit_price *= (1 + self.slippage)
                        pnl = (position.entry_price - exit_price) * position.quantity

                    # Commission
                    pnl -= exit_price * position.quantity * self.commission

                    # Update balance
                    balance += position.margin_used + pnl

                    # Track trade
                    hold_time = (bar_time - position.entry_time).total_seconds() / 60
                    trades.append({
                        "entry_time": position.entry_time,
                        "exit_time": bar_time,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl / position.margin_used,
                        "exit_reason": exit_reason,
                        "hold_time_min": hold_time
                    })

                    position = None

                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

            # If no position, check entry signal
            elif position is None:
                signal = self.signal_generator.generate_signal_no_cooldown(self.symbol, lookback)

                if signal.signal in ["BUY", "SELL"]:
                    side = "LONG" if signal.signal == "BUY" else "SHORT"

                    # Calculate position
                    margin = balance * entry_margin_pct
                    if margin <= 0:
                        continue

                    if side == "LONG":
                        entry_price = current_price * (1 + self.slippage)
                        sl = entry_price * (1 - sl_price_pct)
                        tp = entry_price * (1 + tp_price_pct)
                    else:
                        entry_price = current_price * (1 - self.slippage)
                        sl = entry_price * (1 + sl_price_pct)
                        tp = entry_price * (1 - tp_price_pct)

                    position_value = margin * self.leverage
                    quantity = position_value / entry_price

                    # Commission on entry
                    balance -= margin
                    balance -= entry_price * quantity * self.commission

                    position = ScalpPosition(
                        symbol=self.symbol,
                        side=side,
                        entry_price=entry_price,
                        quantity=quantity,
                        entry_time=bar_time,
                        stop_loss=sl,
                        take_profit=tp,
                        margin_used=margin,
                        avg_entry_price=entry_price
                    )

            equity_curve.append(balance)

        # Calculate results
        if not trades:
            return OptimizationResult(
                strategy="standard_scalping",
                tp_pct=tp_roi,
                sl_pct=sl_roi,
                total_trades=0,
                winning_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                avg_trade_pnl=0,
                profit_factor=0,
                max_drawdown=0,
                avg_hold_time_minutes=0
            )

        winners = [t for t in trades if t["pnl"] > 0]
        losers = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        gross_profit = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0

        return OptimizationResult(
            strategy="standard_scalping",
            tp_pct=tp_roi,
            sl_pct=sl_roi,
            total_trades=len(trades),
            winning_trades=len(winners),
            win_rate=len(winners) / len(trades),
            total_pnl=total_pnl,
            total_pnl_pct=(balance - self.initial_balance) / self.initial_balance,
            avg_trade_pnl=total_pnl / len(trades),
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            max_drawdown=max_drawdown,
            avg_hold_time_minutes=sum(t["hold_time_min"] for t in trades) / len(trades)
        )

    def run_hedge_mode_test(self, df: pd.DataFrame, tp_roi: float, sl_roi: float) -> OptimizationResult:
        """
        Test hedge mode: Open BOTH long and short at the same time.
        Exit whichever hits TP first, then hold the other or close at SL.

        This exploits volatility without needing to predict direction.
        """
        balance = self.initial_balance
        hedge: Optional[HedgePosition] = None
        trades = []
        equity_curve = [balance]
        peak_balance = balance
        max_drawdown = 0.0

        # Convert ROI to price movement
        tp_price_pct = tp_roi / self.leverage
        sl_price_pct = sl_roi / self.leverage

        # Entry sizing: 10% margin per side (20% total for hedge)
        entry_margin_pct = 0.10

        # Cooldown after exit (bars)
        cooldown = 0
        cooldown_bars = 30  # 30 minutes cooldown

        for i in range(100, len(df)):
            current_bar = df.iloc[i]
            current_price = current_bar["close"]
            high = current_bar["high"]
            low = current_bar["low"]
            bar_time = df.index[i]

            # Decrease cooldown
            if cooldown > 0:
                cooldown -= 1

            # If in hedge position, check exits
            if hedge is not None:
                long_exit = None
                short_exit = None
                long_reason = None
                short_reason = None

                # Check LONG side
                if low <= hedge.long_sl:
                    long_exit = hedge.long_sl
                    long_reason = "SL"
                elif high >= hedge.long_tp:
                    long_exit = hedge.long_tp
                    long_reason = "TP"

                # Check SHORT side
                if high >= hedge.short_sl:
                    short_exit = hedge.short_sl
                    short_reason = "SL"
                elif low <= hedge.short_tp:
                    short_exit = hedge.short_tp
                    short_reason = "TP"

                # Process exits (we close BOTH when either hits)
                if long_exit or short_exit:
                    # Calculate P&L for both sides
                    long_pnl = 0
                    short_pnl = 0

                    # Long side
                    l_exit = long_exit if long_exit else current_price
                    l_exit *= (1 - self.slippage)
                    long_pnl = (l_exit - hedge.long_entry) * hedge.quantity
                    long_pnl -= l_exit * hedge.quantity * self.commission

                    # Short side
                    s_exit = short_exit if short_exit else current_price
                    s_exit *= (1 + self.slippage)
                    short_pnl = (hedge.short_entry - s_exit) * hedge.quantity
                    short_pnl -= s_exit * hedge.quantity * self.commission

                    total_pnl = long_pnl + short_pnl

                    # Update balance (return both margins + combined P&L)
                    balance += (hedge.margin_used * 2) + total_pnl

                    hold_time = (bar_time - hedge.entry_time).total_seconds() / 60

                    trades.append({
                        "entry_time": hedge.entry_time,
                        "exit_time": bar_time,
                        "side": "HEDGE",
                        "long_pnl": long_pnl,
                        "short_pnl": short_pnl,
                        "pnl": total_pnl,
                        "pnl_pct": total_pnl / (hedge.margin_used * 2),
                        "long_reason": long_reason or "FOLLOW",
                        "short_reason": short_reason or "FOLLOW",
                        "hold_time_min": hold_time
                    })

                    hedge = None
                    cooldown = cooldown_bars

                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

            # If no hedge position and not in cooldown, enter new hedge
            elif hedge is None and cooldown == 0:
                # Use momentum to determine IF we should enter (optional - can be removed)
                # For pure hedge, we enter regardless of signal

                margin_per_side = balance * entry_margin_pct
                if margin_per_side * 2 > balance * 0.9:  # Keep 10% buffer
                    continue

                # Entry prices with slippage
                long_entry = current_price * (1 + self.slippage)
                short_entry = current_price * (1 - self.slippage)

                # TP/SL for each side
                long_tp = long_entry * (1 + tp_price_pct)
                long_sl = long_entry * (1 - sl_price_pct)
                short_tp = short_entry * (1 - tp_price_pct)
                short_sl = short_entry * (1 + sl_price_pct)

                position_value = margin_per_side * self.leverage
                quantity = position_value / current_price

                # Deduct margin for both sides
                balance -= margin_per_side * 2

                # Commission on entry (both sides)
                balance -= long_entry * quantity * self.commission
                balance -= short_entry * quantity * self.commission

                hedge = HedgePosition(
                    symbol=self.symbol,
                    long_entry=long_entry,
                    short_entry=short_entry,
                    quantity=quantity,
                    entry_time=bar_time,
                    margin_used=margin_per_side,
                    long_tp=long_tp,
                    long_sl=long_sl,
                    short_tp=short_tp,
                    short_sl=short_sl
                )

            equity_curve.append(balance)

        # Calculate results
        if not trades:
            return OptimizationResult(
                strategy="hedge_mode",
                tp_pct=tp_roi,
                sl_pct=sl_roi,
                total_trades=0,
                winning_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                avg_trade_pnl=0,
                profit_factor=0,
                max_drawdown=0,
                avg_hold_time_minutes=0
            )

        winners = [t for t in trades if t["pnl"] > 0]
        losers = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        gross_profit = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0

        return OptimizationResult(
            strategy="hedge_mode",
            tp_pct=tp_roi,
            sl_pct=sl_roi,
            total_trades=len(trades),
            winning_trades=len(winners),
            win_rate=len(winners) / len(trades),
            total_pnl=total_pnl,
            total_pnl_pct=(balance - self.initial_balance) / self.initial_balance,
            avg_trade_pnl=total_pnl / len(trades),
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            max_drawdown=max_drawdown,
            avg_hold_time_minutes=sum(t["hold_time_min"] for t in trades) / len(trades)
        )

    def run_quick_scalping_test(self, df: pd.DataFrame, tp_roi: float) -> OptimizationResult:
        """
        Quick scalping: Very tight TP with minimal SL.
        High frequency, small gains strategy.
        SL = 2x TP (risk:reward 2:1 but high win rate target)
        """
        sl_roi = tp_roi * 2  # 2:1 risk

        balance = self.initial_balance
        position: Optional[ScalpPosition] = None
        trades = []
        peak_balance = balance
        max_drawdown = 0.0

        tp_price_pct = tp_roi / self.leverage
        sl_price_pct = sl_roi / self.leverage

        # Smaller position for quick scalps (10% margin)
        entry_margin_pct = 0.10

        for i in range(50, len(df)):
            current_bar = df.iloc[i]
            lookback = df.iloc[i-50:i+1]
            current_price = current_bar["close"]
            bar_time = df.index[i]

            if position is not None:
                high = current_bar["high"]
                low = current_bar["low"]

                exit_price = None
                exit_reason = None

                if position.side == "LONG":
                    if low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    elif high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"
                else:
                    if high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    elif low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"

                if exit_price:
                    if position.side == "LONG":
                        exit_price *= (1 - self.slippage)
                        pnl = (exit_price - position.entry_price) * position.quantity
                    else:
                        exit_price *= (1 + self.slippage)
                        pnl = (position.entry_price - exit_price) * position.quantity

                    pnl -= exit_price * position.quantity * self.commission
                    balance += position.margin_used + pnl

                    hold_time = (bar_time - position.entry_time).total_seconds() / 60
                    trades.append({
                        "pnl": pnl,
                        "pnl_pct": pnl / position.margin_used,
                        "exit_reason": exit_reason,
                        "hold_time_min": hold_time
                    })

                    position = None

                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

            elif position is None:
                # Simple momentum check for quick scalp
                if len(lookback) >= 20:
                    rsi = self._calc_rsi(lookback["close"], 14)

                    if rsi < 35:  # Oversold - buy
                        side = "LONG"
                    elif rsi > 65:  # Overbought - sell
                        side = "SHORT"
                    else:
                        continue

                    margin = balance * entry_margin_pct
                    if margin <= 0:
                        continue

                    if side == "LONG":
                        entry_price = current_price * (1 + self.slippage)
                        sl = entry_price * (1 - sl_price_pct)
                        tp = entry_price * (1 + tp_price_pct)
                    else:
                        entry_price = current_price * (1 - self.slippage)
                        sl = entry_price * (1 + sl_price_pct)
                        tp = entry_price * (1 - tp_price_pct)

                    position_value = margin * self.leverage
                    quantity = position_value / entry_price

                    balance -= margin
                    balance -= entry_price * quantity * self.commission

                    position = ScalpPosition(
                        symbol=self.symbol,
                        side=side,
                        entry_price=entry_price,
                        quantity=quantity,
                        entry_time=bar_time,
                        stop_loss=sl,
                        take_profit=tp,
                        margin_used=margin,
                        avg_entry_price=entry_price
                    )

        if not trades:
            return OptimizationResult(
                strategy="quick_scalping",
                tp_pct=tp_roi,
                sl_pct=sl_roi,
                total_trades=0,
                winning_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                avg_trade_pnl=0,
                profit_factor=0,
                max_drawdown=0,
                avg_hold_time_minutes=0
            )

        winners = [t for t in trades if t["pnl"] > 0]
        losers = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        gross_profit = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0

        return OptimizationResult(
            strategy="quick_scalping",
            tp_pct=tp_roi,
            sl_pct=sl_roi,
            total_trades=len(trades),
            winning_trades=len(winners),
            win_rate=len(winners) / len(trades),
            total_pnl=total_pnl,
            total_pnl_pct=(balance - self.initial_balance) / self.initial_balance,
            avg_trade_pnl=total_pnl / len(trades),
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            max_drawdown=max_drawdown,
            avg_hold_time_minutes=sum(t["hold_time_min"] for t in trades) / len(trades)
        )

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def run_optimization(self, days: int = 30) -> List[OptimizationResult]:
        """Run full optimization across all strategies and parameters"""

        print("=" * 70)
        print("SCALPING STRATEGY OPTIMIZATION - DOTUSDT")
        print("=" * 70)

        # Fetch data
        df = self.fetch_data(days)

        if df.empty:
            print("No data available!")
            return []

        results = []

        # ===========================================
        # 1. STANDARD SCALPING - Different TP levels
        # ===========================================
        print("\n" + "-" * 50)
        print("1. STANDARD SCALPING - Testing TP levels")
        print("-" * 50)

        # TP levels to test (ROI %)
        tp_levels = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]  # 0.5% to 5% ROI
        sl_levels = [0.05, 0.10, 0.15]  # 5%, 10%, 15% ROI SL

        for tp in tp_levels:
            for sl in sl_levels:
                print(f"  Testing TP={tp*100:.1f}% ROI, SL={sl*100:.0f}% ROI...", end=" ")
                result = self.run_standard_scalping_test(df, tp, sl)
                results.append(result)
                print(f"Trades: {result.total_trades}, Win: {result.win_rate:.1%}, PnL: ${result.total_pnl:+,.2f}")

        # ===========================================
        # 2. HEDGE MODE - Both directions at once
        # ===========================================
        print("\n" + "-" * 50)
        print("2. HEDGE MODE - Simultaneous Long/Short")
        print("-" * 50)

        hedge_tp_levels = [0.01, 0.02, 0.03, 0.05]  # 1% to 5% ROI
        hedge_sl_levels = [0.05, 0.10]  # 5%, 10% ROI SL

        for tp in hedge_tp_levels:
            for sl in hedge_sl_levels:
                print(f"  Testing Hedge TP={tp*100:.1f}% ROI, SL={sl*100:.0f}% ROI...", end=" ")
                result = self.run_hedge_mode_test(df, tp, sl)
                results.append(result)
                print(f"Trades: {result.total_trades}, Win: {result.win_rate:.1%}, PnL: ${result.total_pnl:+,.2f}")

        # ===========================================
        # 3. QUICK SCALPING - High frequency
        # ===========================================
        print("\n" + "-" * 50)
        print("3. QUICK SCALPING - High frequency, tight TP")
        print("-" * 50)

        quick_tp_levels = [0.003, 0.005, 0.008, 0.01]  # 0.3% to 1% ROI

        for tp in quick_tp_levels:
            print(f"  Testing Quick Scalp TP={tp*100:.2f}% ROI (SL=2x)...", end=" ")
            result = self.run_quick_scalping_test(df, tp)
            results.append(result)
            print(f"Trades: {result.total_trades}, Win: {result.win_rate:.1%}, PnL: ${result.total_pnl:+,.2f}")

        # ===========================================
        # SUMMARY - Best results
        # ===========================================
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 70)

        # Sort by total PnL
        results.sort(key=lambda x: x.total_pnl, reverse=True)

        print("\nTOP 10 STRATEGIES BY P&L:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Strategy':<18} {'TP%':<6} {'SL%':<6} {'Trades':<7} {'Win%':<7} {'PnL':<12} {'Drawdown':<10}")
        print("-" * 70)

        for i, r in enumerate(results[:10], 1):
            print(f"{i:<5} {r.strategy:<18} {r.tp_pct*100:<6.1f} {r.sl_pct*100:<6.0f} {r.total_trades:<7} "
                  f"{r.win_rate*100:<7.1f} ${r.total_pnl:<11,.2f} {r.max_drawdown*100:<10.1f}%")

        # Best by win rate (min 10 trades)
        valid_results = [r for r in results if r.total_trades >= 10]
        if valid_results:
            valid_results.sort(key=lambda x: x.win_rate, reverse=True)

            print("\nTOP 5 BY WIN RATE (min 10 trades):")
            print("-" * 70)
            for i, r in enumerate(valid_results[:5], 1):
                print(f"{i:<5} {r.strategy:<18} {r.tp_pct*100:<6.1f} {r.sl_pct*100:<6.0f} {r.total_trades:<7} "
                      f"{r.win_rate*100:<7.1f} ${r.total_pnl:<11,.2f} Avg: ${r.avg_trade_pnl:.2f}")

        # Best profit factor (min 10 trades)
        pf_results = [r for r in results if r.total_trades >= 10 and r.profit_factor < float("inf")]
        if pf_results:
            pf_results.sort(key=lambda x: x.profit_factor, reverse=True)

            print("\nTOP 5 BY PROFIT FACTOR (min 10 trades):")
            print("-" * 70)
            for i, r in enumerate(pf_results[:5], 1):
                print(f"{i:<5} {r.strategy:<18} {r.tp_pct*100:<6.1f} {r.sl_pct*100:<6.0f} "
                      f"PF: {r.profit_factor:<6.2f} Win: {r.win_rate*100:.1f}% PnL: ${r.total_pnl:+,.2f}")

        print("\n" + "=" * 70)

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scalping Strategy Optimization for DOTUSDT")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data (default: 30)")
    parser.add_argument("--symbol", type=str, default="DOTUSDT", help="Symbol to optimize (default: DOTUSDT)")

    args = parser.parse_args()

    optimizer = ScalpingOptimizer(symbol=args.symbol)
    results = optimizer.run_optimization(days=args.days)

    # Save results
    if results:
        output_file = f"scalping_optimization_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", output_file)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump([{
                "strategy": r.strategy,
                "tp_pct": r.tp_pct,
                "sl_pct": r.sl_pct,
                "total_trades": r.total_trades,
                "winning_trades": r.winning_trades,
                "win_rate": r.win_rate,
                "total_pnl": r.total_pnl,
                "total_pnl_pct": r.total_pnl_pct,
                "avg_trade_pnl": r.avg_trade_pnl,
                "profit_factor": r.profit_factor if r.profit_factor < float("inf") else 999,
                "max_drawdown": r.max_drawdown,
                "avg_hold_time_minutes": r.avg_hold_time_minutes
            } for r in results], f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
