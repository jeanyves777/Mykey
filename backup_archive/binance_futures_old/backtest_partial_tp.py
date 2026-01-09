#!/usr/bin/env python3
"""
HTF Confluence Backtest with Partial Take Profit Strategy
==========================================================
COPY of realistic backtest (backtest_htf_confluence.py) with partial TP added.

Strategy to test:
1. Close 50% at 15% ROI (TP1)
2. When remaining hits 30% ROI -> move SL to lock 20% profit
3. Final TP at 50% ROI for remaining 50%

Compare vs baseline fixed TP/SL strategy (30% TP / 10% SL).
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


class PartialTPBacktester:
    """
    Backtester with partial take profit support.

    COPY of HTFConfluenceBacktester from backtest_htf_confluence.py
    with partial TP feature added.
    """

    def __init__(
        self,
        symbol: str,
        start_balance: float = 100.0,
        config: dict = None,
        use_partial_tp: bool = True,
        tp1_roi: float = 0.15,        # First TP at 15% ROI
        tp1_close_pct: float = 0.50,  # Close 50% at TP1
        lock_trigger_roi: float = 0.30,   # When to lock profit
        lock_profit_roi: float = 0.20,    # Lock at 20% ROI
        tp2_roi: float = 0.50         # Final TP at 50% ROI
    ):
        """
        Initialize backtester.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_balance: Starting balance in USDT
            config: Strategy config (default: MODERATE_CONFIG)
            use_partial_tp: Enable partial TP strategy (False = baseline)
            tp1_roi: First TP ROI (default 15%)
            tp1_close_pct: Percentage to close at TP1 (default 50%)
            lock_trigger_roi: ROI to trigger profit lock (default 30%)
            lock_profit_roi: ROI to lock when triggered (default 20%)
            tp2_roi: Final TP ROI (default 50%)
        """
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance

        # Use moderate config by default
        if config is None:
            config = MODERATE_CONFIG

        self.leverage = config["leverage"]
        self.tp_roi = config["tp_roi"]  # Base TP (30% for MODERATE)
        self.sl_roi = config["sl_roi"]  # Base SL (10% for MODERATE)

        # Partial TP settings
        self.use_partial_tp = use_partial_tp
        self.tp1_roi = tp1_roi
        self.tp1_close_pct = tp1_close_pct
        self.lock_trigger_roi = lock_trigger_roi
        self.lock_profit_roi = lock_profit_roi
        self.tp2_roi = tp2_roi

        # Strategy
        self.strategy = HTFConfluenceStrategy(**config)

        # Position sizing - risk 2% of balance per trade
        self.risk_per_trade = 0.02

        # REALISTIC TRADING COSTS (Binance Futures)
        # Based on user's actual Binance data: $0.07 fee on $1,200 position = 0.006%
        self.taker_fee = 0.0003      # 0.03% taker fee (with BNB discount)
        self.maker_fee = 0.00006     # 0.006% maker fee (actual from screenshot)
        self.slippage = 0.0001       # 0.01% slippage (liquid pairs)
        self.use_maker = True

        # Position tracking
        self.position = None

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance

        # Signal stats
        self.signals_generated = 0
        self.long_signals = 0
        self.short_signals = 0

        # Cost tracking
        self.total_fees_paid = 0
        self.total_slippage_cost = 0

        # Partial TP stats
        self.tp1_hits = 0
        self.tp2_hits = 0
        self.lock_triggers = 0
        self.locked_sl_exits = 0

    def get_historical_data(self, days: int = 60, ltf_interval: str = "15m"):
        """Fetch historical data using Yahoo Finance (no geo restrictions)."""
        yf_ticker_map = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "BNBUSDT": "BNB-USD",
            "SOLUSDT": "SOL-USD",
            "ADAUSDT": "ADA-USD",
            "DOTUSDT": "DOT-USD",
            "XRPUSDT": "XRP-USD",
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

    def calculate_price_from_roi(self, entry_price: float, target_roi: float, is_long: bool) -> float:
        """Calculate price level for a given ROI target."""
        price_move = target_roi / self.leverage
        if is_long:
            return entry_price * (1 + price_move)
        else:
            return entry_price * (1 - price_move)

    def open_position(self, side: str, price: float, timestamp, signal) -> dict:
        """Open new position with realistic costs."""
        is_long = side in ["LONG", "BUY"]

        # Apply slippage to entry price (worse fill)
        if is_long:
            actual_entry = price * (1 + self.slippage)
        else:
            actual_entry = price * (1 - self.slippage)

        # Calculate position size based on risk
        sl_pct = self.sl_roi / self.leverage
        risk_amount = self.balance * self.risk_per_trade
        position_value = risk_amount / sl_pct
        margin = position_value / self.leverage
        quantity = position_value / actual_entry

        # Calculate and deduct entry fee
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        entry_fee = position_value * fee_rate
        self.balance -= entry_fee
        self.total_fees_paid += entry_fee

        # Track slippage cost
        slippage_cost = abs(actual_entry - price) * quantity
        self.total_slippage_cost += slippage_cost

        # Calculate exit levels
        if self.use_partial_tp:
            # Partial TP strategy levels
            tp1_price = self.calculate_price_from_roi(actual_entry, self.tp1_roi, is_long)
            tp2_price = self.calculate_price_from_roi(actual_entry, self.tp2_roi, is_long)
            lock_trigger_price = self.calculate_price_from_roi(actual_entry, self.lock_trigger_roi, is_long)
            locked_sl_price = self.calculate_price_from_roi(actual_entry, self.lock_profit_roi, is_long)
            # Original SL
            sl_price = self.calculate_price_from_roi(actual_entry, -self.sl_roi, is_long)
        else:
            # Baseline: use standard TP/SL from strategy
            sl_price, tp_price = self.strategy.calculate_exit_levels(actual_entry, side)
            tp1_price = tp_price
            tp2_price = tp_price
            lock_trigger_price = 0
            locked_sl_price = 0

        if is_long:
            self.long_signals += 1
        else:
            self.short_signals += 1

        self.signals_generated += 1

        position = {
            "side": side,
            "entry_price": actual_entry,
            "signal_price": price,
            "quantity": quantity,
            "original_quantity": quantity,
            "margin": margin,
            "original_margin": margin,
            "sl_price": sl_price,
            "original_sl_price": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "lock_trigger_price": lock_trigger_price,
            "locked_sl_price": locked_sl_price,
            "entry_time": timestamp,
            "confluence_score": signal.confluence_score,
            "strength": signal.strength.value,
            "entry_fee": entry_fee,
            "tp1_hit": False,
            "lock_triggered": False,
            "partial_pnl": 0.0,
        }

        print(f"[{timestamp}] OPEN {side} @ ${actual_entry:.2f} | Fee: ${entry_fee:.4f}")

        return position

    def close_partial(self, exit_price: float, close_pct: float, exit_type: str, timestamp) -> float:
        """Close partial position with realistic costs."""
        if self.position is None:
            return 0

        is_long = self.position["side"] in ["LONG", "BUY"]

        # Apply slippage
        if is_long:
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_exit = exit_price * (1 + self.slippage)

        # Calculate partial quantities
        close_qty = self.position["quantity"] * close_pct
        close_margin = self.position["margin"] * close_pct

        # Calculate exit fee for partial
        position_value = close_qty * actual_exit
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        exit_fee = position_value * fee_rate
        self.total_fees_paid += exit_fee

        # Track slippage
        slippage_cost = abs(actual_exit - exit_price) * close_qty
        self.total_slippage_cost += slippage_cost

        # Calculate P&L for partial
        price_change_pct = (actual_exit - self.position["entry_price"]) / self.position["entry_price"]
        if is_long:
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        gross_pnl = close_margin * roi
        pnl = gross_pnl - exit_fee

        # Update balance
        self.balance += pnl
        self.total_pnl += pnl
        self.position["partial_pnl"] += pnl

        # Update position (reduce quantity and margin)
        self.position["quantity"] *= (1 - close_pct)
        self.position["margin"] *= (1 - close_pct)

        print(f"  [{timestamp}] PARTIAL {exit_type}: {close_pct*100:.0f}% @ ${actual_exit:.2f} | ROI: {roi*100:+.1f}% | PnL: ${pnl:+.2f}")

        return pnl

    def close_position(self, exit_price: float, exit_type: str, timestamp) -> float:
        """Close remaining position with realistic costs."""
        if self.position is None:
            return 0

        is_long = self.position["side"] in ["LONG", "BUY"]

        # Apply slippage
        if is_long:
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_exit = exit_price * (1 + self.slippage)

        # Calculate exit fee
        position_value = self.position["quantity"] * actual_exit
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        exit_fee = position_value * fee_rate
        self.total_fees_paid += exit_fee

        # Track slippage
        slippage_cost = abs(actual_exit - exit_price) * self.position["quantity"]
        self.total_slippage_cost += slippage_cost

        # Calculate P&L
        price_change_pct = (actual_exit - self.position["entry_price"]) / self.position["entry_price"]

        if is_long:
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        gross_pnl = self.position["margin"] * roi
        pnl = gross_pnl - exit_fee

        # Cap loss at remaining margin
        if pnl < -self.position["margin"]:
            pnl = -self.position["margin"]

        self.balance += pnl
        self.total_pnl += pnl

        if self.balance <= 0:
            self.balance = 0

        # Total PnL for this trade (including partials)
        total_trade_pnl = self.position["partial_pnl"] + pnl

        if total_trade_pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1

        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Calculate overall ROI for the trade
        overall_roi = total_trade_pnl / self.position["original_margin"] * 100

        total_fees = self.position.get("entry_fee", 0) + exit_fee

        self.trades.append({
            "timestamp": timestamp,
            "side": self.position["side"],
            "entry_price": self.position["entry_price"],
            "exit_price": actual_exit,
            "original_margin": self.position["original_margin"],
            "fees": total_fees,
            "partial_pnl": self.position["partial_pnl"],
            "final_pnl": pnl,
            "total_pnl": total_trade_pnl,
            "overall_roi": overall_roi,
            "exit_type": exit_type,
            "tp1_hit": self.position["tp1_hit"],
            "lock_triggered": self.position["lock_triggered"],
            "balance": self.balance,
            "confluence_score": self.position["confluence_score"],
        })

        print(f"  [{timestamp}] CLOSE {exit_type}: @ ${actual_exit:.2f} | ROI: {roi*100:+.1f}% | PnL: ${pnl:+.2f} | Total: ${total_trade_pnl:+.2f}")

        self.position = None
        return pnl

    def run_backtest(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """Run backtest."""
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

                if self.use_partial_tp:
                    # === PARTIAL TP STRATEGY ===

                    # Step 1: Check TP1 (15% ROI - close 50%)
                    if not self.position["tp1_hit"]:
                        if (is_long and high >= self.position["tp1_price"]) or \
                           (not is_long and low <= self.position["tp1_price"]):
                            self.close_partial(self.position["tp1_price"], self.tp1_close_pct, "TP1", timestamp)
                            self.position["tp1_hit"] = True
                            self.tp1_hits += 1
                            continue

                    # Step 2: Check lock trigger (30% ROI -> lock at 20%)
                    if self.position["tp1_hit"] and not self.position["lock_triggered"]:
                        if (is_long and high >= self.position["lock_trigger_price"]) or \
                           (not is_long and low <= self.position["lock_trigger_price"]):
                            self.position["sl_price"] = self.position["locked_sl_price"]
                            self.position["lock_triggered"] = True
                            self.lock_triggers += 1
                            print(f"  [{timestamp}] LOCK: SL -> {self.lock_profit_roi*100:.0f}% ROI (${self.position['sl_price']:.4f})")

                    # Step 3: Check TP2 (50% ROI - close remaining)
                    if self.position["tp1_hit"]:
                        if (is_long and high >= self.position["tp2_price"]) or \
                           (not is_long and low <= self.position["tp2_price"]):
                            self.close_position(self.position["tp2_price"], "TP2", timestamp)
                            self.tp2_hits += 1
                            continue

                    # Step 4: Check SL (original or locked)
                    if (is_long and low <= self.position["sl_price"]) or \
                       (not is_long and high >= self.position["sl_price"]):
                        exit_type = "LOCKED_SL" if self.position["lock_triggered"] else "SL"
                        if self.position["lock_triggered"]:
                            self.locked_sl_exits += 1
                        self.close_position(self.position["sl_price"], exit_type, timestamp)
                        continue

                else:
                    # === BASELINE STRATEGY (Fixed TP/SL from MODERATE_CONFIG: 30%/10%) ===
                    if is_long:
                        if high >= self.position["tp1_price"]:
                            self.close_position(self.position["tp1_price"], "TP", timestamp)
                            continue
                        elif low <= self.position["sl_price"]:
                            self.close_position(self.position["sl_price"], "SL", timestamp)
                            continue
                    else:
                        if low <= self.position["tp1_price"]:
                            self.close_position(self.position["tp1_price"], "TP", timestamp)
                            continue
                        elif high >= self.position["sl_price"]:
                            self.close_position(self.position["sl_price"], "SL", timestamp)
                            continue

            # Check for entry signal
            if self.position is None:
                ltf_available = ltf_df.iloc[:i+1]
                signal = self.strategy.should_enter(ltf_available, htf_available, current_bar=i)

                if signal.action:
                    self.position = self.open_position(signal.action, close, timestamp, signal)

        # Close any open position at end
        if self.position is not None:
            self.close_position(ltf_df['close'].iloc[-1], "END", ltf_df.index[-1])

        return self.get_results()

    def get_results(self) -> dict:
        """Get backtest results."""
        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0

        return {
            "symbol": self.symbol,
            "strategy": "PARTIAL_TP" if self.use_partial_tp else "BASELINE",
            "balance": self.balance,
            "return_pct": (self.balance - self.start_balance) / self.start_balance * 100,
            "total_trades": total_trades,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "total_fees": self.total_fees_paid,
            "total_slippage": self.total_slippage_cost,
            "tp1_hits": self.tp1_hits,
            "tp2_hits": self.tp2_hits,
            "lock_triggers": self.lock_triggers,
            "locked_sl_exits": self.locked_sl_exits,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
        }


def run_partial_tp_comparison(symbols: list = None, days: int = 60):
    """Compare Partial TP strategy vs Baseline (both with realistic costs)."""
    if symbols is None:
        symbols = ["DOTUSDT", "BNBUSDT"]

    print("=" * 100)
    print("PARTIAL TAKE PROFIT STRATEGY BACKTEST (REALISTIC COSTS)")
    print("=" * 100)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {days} days")
    print(f"Leverage: {MODERATE_CONFIG['leverage']}x")
    print(f"Fees: Maker 0.006% | Slippage: 0.01%")
    print()
    print("BASELINE: 30% ROI TP | 10% ROI SL")
    print()
    print("PARTIAL TP STRATEGY:")
    print("  1. Close 50% at 15% ROI (TP1)")
    print("  2. When remaining hits 30% ROI -> Lock SL at 20% ROI")
    print("  3. Close remaining at 50% ROI (TP2)")
    print("  4. Original SL: 10% ROI")
    print("=" * 100)

    baseline_results = []
    partial_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        # Fetch data once
        baseline = PartialTPBacktester(
            symbol=symbol,
            start_balance=20.0,
            config=MODERATE_CONFIG,
            use_partial_tp=False
        )
        ltf_df, htf_df = baseline.get_historical_data(days=days)

        if ltf_df is None or htf_df is None:
            print(f"ERROR: No data for {symbol}")
            continue

        print(f"  Data: {len(ltf_df)} LTF candles | {len(htf_df)} HTF candles")

        # Test BASELINE
        print(f"\n--- BASELINE (30% TP / 10% SL) ---")
        result = baseline.run_backtest(ltf_df.copy(), htf_df.copy())
        baseline_results.append(result)
        print(f"\nBASELINE: ${result['balance']:.2f} ({result['return_pct']:+.1f}%) | {result['total_trades']} trades | WR: {result['win_rate']:.0f}% | Fees: ${result['total_fees']:.2f}")

        # Test PARTIAL TP
        print(f"\n--- PARTIAL TP (15%/50%/lock@30%->20%) ---")
        partial = PartialTPBacktester(
            symbol=symbol,
            start_balance=20.0,
            config=MODERATE_CONFIG,
            use_partial_tp=True,
            tp1_roi=0.15,
            tp1_close_pct=0.50,
            lock_trigger_roi=0.30,
            lock_profit_roi=0.20,
            tp2_roi=0.50
        )
        result = partial.run_backtest(ltf_df.copy(), htf_df.copy())
        partial_results.append(result)
        print(f"\nPARTIAL: ${result['balance']:.2f} ({result['return_pct']:+.1f}%) | {result['total_trades']} trades | WR: {result['win_rate']:.0f}% | Fees: ${result['total_fees']:.2f}")
        print(f"  TP1: {result['tp1_hits']} | TP2: {result['tp2_hits']} | Locks: {result['lock_triggers']} | Locked SL: {result['locked_sl_exits']}")

    # Print comparison summary
    if baseline_results and partial_results:
        print("\n" + "=" * 120)
        print("COMPARISON SUMMARY (REALISTIC COSTS INCLUDED)")
        print("=" * 120)

        print(f"\n{'Symbol':<12} {'Strategy':<15} {'End $':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>8} {'Fees':>10}")
        print("-" * 90)

        for i, symbol in enumerate(symbols):
            if i < len(baseline_results) and i < len(partial_results):
                b = baseline_results[i]
                p = partial_results[i]
                print(f"{symbol:<12} {'BASELINE':<15} ${b['balance']:>9.2f} {b['return_pct']:>+9.1f}% {b['total_trades']:>7} {b['win_rate']:>7.1f}% {b['max_drawdown']:>7.1f}% ${b['total_fees']:>8.2f}")
                print(f"{'':<12} {'PARTIAL TP':<15} ${p['balance']:>9.2f} {p['return_pct']:>+9.1f}% {p['total_trades']:>7} {p['win_rate']:>7.1f}% {p['max_drawdown']:>7.1f}% ${p['total_fees']:>8.2f}")
                diff = p['return_pct'] - b['return_pct']
                print(f"{'':<12} {'DIFFERENCE':<15} {'':<10} {diff:>+9.1f}%")
                print("-" * 90)

        # Total comparison
        print("\nTOTAL COMPARISON:")
        print("-" * 90)

        baseline_total_start = len(baseline_results) * 20.0
        baseline_total_end = sum(r['balance'] for r in baseline_results)
        baseline_total_return = (baseline_total_end / baseline_total_start - 1) * 100
        baseline_total_fees = sum(r['total_fees'] for r in baseline_results)

        partial_total_start = len(partial_results) * 20.0
        partial_total_end = sum(r['balance'] for r in partial_results)
        partial_total_return = (partial_total_end / partial_total_start - 1) * 100
        partial_total_fees = sum(r['total_fees'] for r in partial_results)

        print(f"{'BASELINE':<15} ${baseline_total_end:>9.2f} ({baseline_total_return:>+.1f}%) | Total Fees: ${baseline_total_fees:.2f}")
        print(f"{'PARTIAL TP':<15} ${partial_total_end:>9.2f} ({partial_total_return:>+.1f}%) | Total Fees: ${partial_total_fees:.2f}")
        print(f"{'DIFFERENCE':<15} ${partial_total_end - baseline_total_end:>+9.2f} ({partial_total_return - baseline_total_return:>+.1f}%)")

        # Partial TP stats
        total_tp1 = sum(r['tp1_hits'] for r in partial_results)
        total_tp2 = sum(r['tp2_hits'] for r in partial_results)
        total_locks = sum(r['lock_triggers'] for r in partial_results)
        total_locked_exits = sum(r['locked_sl_exits'] for r in partial_results)

        print("\nPARTIAL TP STATS:")
        print(f"  TP1 Hits (15% ROI): {total_tp1}")
        print(f"  TP2 Hits (50% ROI): {total_tp2}")
        print(f"  Lock Triggers (30% -> 20%): {total_locks}")
        print(f"  Locked SL Exits: {total_locked_exits}")

        # Winner
        print("\n" + "=" * 80)
        if partial_total_return > baseline_total_return:
            print(f"WINNER: PARTIAL TP STRATEGY (+{partial_total_return - baseline_total_return:.1f}% better)")
        elif baseline_total_return > partial_total_return:
            print(f"WINNER: BASELINE STRATEGY (+{baseline_total_return - partial_total_return:.1f}% better)")
        else:
            print("RESULT: TIE")
        print("=" * 80)

    return baseline_results, partial_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Partial TP Strategy")
    parser.add_argument("--symbols", nargs="+", default=["DOTUSDT", "BNBUSDT"],
                        help="Symbols to backtest")
    parser.add_argument("--days", type=int, default=60, help="Backtest period")

    args = parser.parse_args()

    run_partial_tp_comparison(symbols=args.symbols, days=args.days)
