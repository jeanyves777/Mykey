#!/usr/bin/env python3
"""
Backtest HTF Trend + Confluence Strategy
=========================================
Backtests the HTF Confluence Strategy using real Binance historical data.

Strategy:
- HTF (4H): 200 EMA trend filter
- LTF (15m/1H): MACD + RSI + EMA (9/21) confluence
- Single direction trading (follow the trend)
- TP: 2% ROI, SL: 1% ROI (2:1 R:R)
- Leverage: 5-10x
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import csv
import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
from strategies.htf_confluence_strategy import (
    HTFConfluenceStrategy,
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    AGGRESSIVE_CONFIG,
    OPTIMIZED_CONFIG,
    SWING_CONFIG,
    TrendDirection,
    SignalStrength
)


class HTFConfluenceBacktester:
    """Backtester for HTF Confluence Strategy"""

    def __init__(
        self,
        symbol: str,
        start_balance: float = 100.0,
        config: dict = None
    ):
        """
        Initialize backtester.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_balance: Starting balance in USDT
            config: Strategy config (CONSERVATIVE, MODERATE, or AGGRESSIVE)
        """
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance

        # Use moderate config by default
        if config is None:
            config = MODERATE_CONFIG

        self.leverage = config["leverage"]
        self.tp_roi = config["tp_roi"]
        self.sl_roi = config["sl_roi"]

        # Strategy
        self.strategy = HTFConfluenceStrategy(**config)

        # Position sizing - risk 2% of balance per trade
        self.risk_per_trade = 0.02

        # REALISTIC TRADING COSTS (Binance Futures)
        # Based on user's actual Binance data: $0.07 fee on $1,200 position = 0.006%
        # This is likely maker fee with BNB discount
        self.taker_fee = 0.0003      # 0.03% taker fee (with BNB discount)
        self.maker_fee = 0.00006     # 0.006% maker fee (actual from screenshot)
        self.slippage = 0.0001       # 0.01% slippage (liquid pairs like BTC/ETH)
        self.funding_rate = 0.0      # 0% - funding is often 0 or negligible for short trades
        # Use maker fee assumption since limit orders are common
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
        self.signals_skipped_neutral = 0
        self.signals_skipped_weak = 0

        # Cost tracking
        self.total_fees_paid = 0
        self.total_slippage_cost = 0
        self.total_funding_paid = 0

        print(f"[{symbol}] HTF Confluence Strategy (REALISTIC MODE)")
        print(f"  Leverage: {self.leverage}x")
        print(f"  TP: {self.tp_roi*100:.1f}% ROI | SL: {self.sl_roi*100:.1f}% ROI")
        print(f"  Risk per trade: {self.risk_per_trade*100:.0f}% of balance")

    def get_historical_data(self, days: int = 90, ltf_interval: str = "15m"):
        """
        Fetch historical data using Yahoo Finance (no geo restrictions).

        Args:
            days: Number of days of history
            ltf_interval: LTF interval (15m or 1h)

        Returns:
            (ltf_df, htf_df) DataFrames
        """
        print(f"\nFetching {days} days of data for {self.symbol}...")
        print("Using Yahoo Finance for historical data...")

        # Map symbol to Yahoo Finance ticker
        yf_ticker_map = {
            "BTCUSDT": "BTC-USD",
            "ETHUSDT": "ETH-USD",
            "BNBUSDT": "BNB-USD",
            "SOLUSDT": "SOL-USD",
            "ADAUSDT": "ADA-USD",
            "DOTUSDT": "DOT-USD",
            "XRPUSDT": "XRP-USD",
            "LTCUSDT": "LTC-USD",
            "AVAXUSDT": "AVAX-USD",
        }

        yf_symbol = yf_ticker_map.get(self.symbol, self.symbol.replace("USDT", "-USD"))

        # Yahoo Finance interval mapping
        yf_interval_map = {
            "15m": "15m",
            "1h": "1h",
            "4h": "1h",  # Yahoo doesn't have 4h, we'll use 1h and resample
        }

        try:
            ticker = yf.Ticker(yf_symbol)

            # For LTF, use 15m or 1h
            # Yahoo limits 15m data to 60 days, 1h to 730 days
            ltf_days = min(days, 60) if ltf_interval == "15m" else days
            print(f"  Fetching {ltf_interval} data (last {ltf_days} days)...")

            ltf_df = ticker.history(period=f"{ltf_days}d", interval=yf_interval_map.get(ltf_interval, "15m"))

            if ltf_df is None or len(ltf_df) == 0:
                print(f"ERROR: No LTF data returned for {yf_symbol}")
                return None, None

            # Rename columns to lowercase
            ltf_df.columns = [c.lower() for c in ltf_df.columns]
            ltf_df = ltf_df[['open', 'high', 'low', 'close', 'volume']]

            # For HTF, fetch 1h data and resample to 4h
            htf_days = min(days + 50, 730)  # Extra days for 200 EMA warmup
            print(f"  Fetching 1h data (last {htf_days} days) and resampling to 4h...")

            htf_raw = ticker.history(period=f"{htf_days}d", interval="1h")

            if htf_raw is None or len(htf_raw) == 0:
                print(f"ERROR: No HTF data returned for {yf_symbol}")
                return None, None

            htf_raw.columns = [c.lower() for c in htf_raw.columns]
            htf_raw = htf_raw[['open', 'high', 'low', 'close', 'volume']]

            # Resample 1h to 4h
            htf_df = htf_raw.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            print(f"  LTF ({ltf_interval}): {len(ltf_df)} candles")
            print(f"  HTF (4h): {len(htf_df)} candles")

            return ltf_df, htf_df

        except Exception as e:
            print(f"[ERROR] Yahoo Finance failed: {e}")
            return None, None

    def open_position(self, side: str, price: float, timestamp, signal) -> dict:
        """Open new position with realistic costs"""
        # Apply slippage to entry price (worse fill)
        if side == "LONG":
            actual_entry = price * (1 + self.slippage)  # Pay more for longs
        else:
            actual_entry = price * (1 - self.slippage)  # Get less for shorts

        # Calculate position size based on risk
        sl_pct = self.sl_roi / self.leverage
        risk_amount = self.balance * self.risk_per_trade
        position_value = risk_amount / sl_pct
        margin = position_value / self.leverage
        quantity = position_value / actual_entry

        # Calculate and deduct entry fee (use maker if enabled)
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        entry_fee = position_value * fee_rate
        self.balance -= entry_fee
        self.total_fees_paid += entry_fee

        # Track slippage cost
        slippage_cost = abs(actual_entry - price) * quantity
        self.total_slippage_cost += slippage_cost

        # Calculate TP and SL prices based on actual entry
        stop_loss, take_profit = self.strategy.calculate_exit_levels(actual_entry, side)

        position = {
            "side": side,
            "entry_price": actual_entry,
            "signal_price": price,
            "quantity": quantity,
            "margin": margin,
            "tp_price": take_profit,
            "sl_price": stop_loss,
            "entry_time": timestamp,
            "confluence_score": signal.confluence_score,
            "strength": signal.strength.value,
            "entry_fee": entry_fee
        }

        if side == "LONG" or side == "BUY":
            self.long_signals += 1
        else:
            self.short_signals += 1

        self.signals_generated += 1

        print(f"[{timestamp}] OPEN {side} @ ${actual_entry:.2f} (slip: ${price:.2f}) | Fee: ${entry_fee:.2f}")

        return position

    def close_position(self, exit_price: float, exit_type: str, timestamp) -> float:
        """Close position and calculate P&L with realistic costs"""
        if self.position is None:
            return 0

        # Apply slippage to exit price (worse fill)
        is_long_exit = self.position["side"] in ["LONG", "BUY"]
        if is_long_exit:
            actual_exit = exit_price * (1 - self.slippage)  # Get less when selling
        else:
            actual_exit = exit_price * (1 + self.slippage)  # Pay more when covering

        # Calculate exit fee (use maker if enabled)
        position_value = self.position["quantity"] * actual_exit
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        exit_fee = position_value * fee_rate
        self.total_fees_paid += exit_fee

        # Track slippage cost
        slippage_cost = abs(actual_exit - exit_price) * self.position["quantity"]
        self.total_slippage_cost += slippage_cost

        # Calculate P&L
        price_change_pct = (actual_exit - self.position["entry_price"]) / self.position["entry_price"]

        is_long = self.position["side"] in ["LONG", "BUY"]
        if is_long:
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        # Gross P&L before exit fee
        gross_pnl = self.position["margin"] * roi

        # Net P&L after exit fee
        pnl = gross_pnl - exit_fee

        # Cap loss at margin (isolated margin)
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

        total_fees = self.position.get("entry_fee", 0) + exit_fee

        self.trades.append({
            "timestamp": timestamp,
            "side": self.position["side"],
            "entry_price": self.position["entry_price"],
            "exit_price": actual_exit,
            "quantity": self.position["quantity"],
            "margin": self.position["margin"],
            "gross_pnl": gross_pnl,
            "fees": total_fees,
            "pnl": pnl,
            "roi": roi * 100,
            "exit_type": exit_type,
            "balance": self.balance,
            "confluence_score": self.position["confluence_score"],
            "strength": self.position["strength"]
        })

        print(f"[{timestamp}] CLOSE {self.position['side']} @ ${actual_exit:.2f} | {exit_type} | P&L: ${pnl:+.2f} (fees: ${total_fees:.2f})")

        self.position = None
        return pnl

    def get_htf_data_at_time(self, htf_df: pd.DataFrame, current_time) -> pd.DataFrame:
        """Get HTF data available at a specific time (avoid lookahead bias)"""
        return htf_df[htf_df.index <= current_time]

    def run_backtest(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """Run the backtest"""
        print("\n" + "="*70)
        print("RUNNING HTF CONFLUENCE STRATEGY BACKTEST")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"TP: {self.tp_roi*100:.1f}% ROI | SL: {self.sl_roi*100:.1f}% ROI")
        print(f"LTF Period: {ltf_df.index[0]} to {ltf_df.index[-1]}")
        print("="*70)

        # Warmup period - need enough data for indicators
        warmup = max(30, self.strategy.ema_slow * 2)

        print(f"Starting from candle {warmup} (warmup period)")

        # Iterate through LTF candles
        for i in range(warmup, len(ltf_df)):
            timestamp = ltf_df.index[i]
            row = ltf_df.iloc[i]

            if self.balance <= 0:
                print(f"[{timestamp}] ACCOUNT BLOWN - Balance depleted!")
                break

            high = row['high']
            low = row['low']
            close = row['close']

            # Get HTF data available at this time (avoid lookahead)
            htf_available = self.get_htf_data_at_time(htf_df, timestamp)

            if len(htf_available) < self.strategy.htf_ema_period:
                continue

            # Check existing position
            if self.position is not None:
                # Check TP and SL (handle both BUY/LONG and SELL/SHORT)
                is_long = self.position["side"] in ["LONG", "BUY"]
                if is_long:
                    if high >= self.position["tp_price"]:
                        self.close_position(self.position["tp_price"], "TP", timestamp)
                        continue
                    elif low <= self.position["sl_price"]:
                        self.close_position(self.position["sl_price"], "SL", timestamp)
                        continue
                else:  # SHORT
                    if low <= self.position["tp_price"]:
                        self.close_position(self.position["tp_price"], "TP", timestamp)
                        continue
                    elif high >= self.position["sl_price"]:
                        self.close_position(self.position["sl_price"], "SL", timestamp)
                        continue

            # Check for entry signal (only if no position)
            if self.position is None:
                # Get LTF data up to current candle
                ltf_available = ltf_df.iloc[:i+1]

                # Get signal
                signal = self.strategy.should_enter(ltf_available, htf_available, current_bar=i)

                if signal.action:
                    self.position = self.open_position(signal.action, close, timestamp, signal)
                elif signal.trend == TrendDirection.NEUTRAL:
                    self.signals_skipped_neutral += 1
                elif signal.confluence_score < 3:
                    self.signals_skipped_weak += 1

        # Close any open position at end
        if self.position is not None:
            final_price = ltf_df['close'].iloc[-1]
            self.close_position(final_price, "END_OF_BACKTEST", ltf_df.index[-1])

        self.print_results(ltf_df, htf_df)
        self.export_trade_journal()

        return {
            "symbol": self.symbol,
            "balance": self.balance,
            "return_pct": (self.balance - self.start_balance) / self.start_balance * 100,
            "win_rate": (self.total_wins / len(self.trades) * 100) if self.trades else 0,
            "max_drawdown": self.max_drawdown,
            "total_trades": len(self.trades),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "total_pnl": self.total_pnl,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
        }

    def export_trade_journal(self, filename: str = None):
        """Export all trades to CSV"""
        if not self.trades:
            print("No trades to export")
            return

        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"htf_confluence_journal_{self.symbol}_{timestamp_str}.csv"

        fieldnames = [
            "timestamp", "symbol", "side", "entry_price", "exit_price",
            "quantity", "margin", "pnl", "roi", "exit_type",
            "balance_after", "confluence_score", "strength"
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
                    "confluence_score": trade["confluence_score"],
                    "strength": trade["strength"]
                })

        print(f"\n>>> Trade journal exported to: {filename}")

    def print_results(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - HTF CONFLUENCE STRATEGY")
        print("="*70)

        price_start = ltf_df['close'].iloc[0]
        price_end = ltf_df['close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100

        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\nMarket Performance:")
        print(f"  Price: ${price_start:.2f} -> ${price_end:.2f} ({price_change:+.2f}%)")
        print(f"  LTF Period: {ltf_df.index[0]} to {ltf_df.index[-1]}")

        print(f"\nStrategy Performance:")
        print(f"  Starting Balance: ${self.start_balance:.2f}")
        print(f"  Ending Balance:   ${self.balance:.2f}")
        print(f"  Total P&L:        ${self.total_pnl:+.2f}")
        print(f"  Total Return:     {((self.balance - self.start_balance) / self.start_balance * 100):+.2f}%")

        print(f"\nREALISTIC COSTS (deducted from P&L):")
        print(f"  Total Fees Paid:     ${self.total_fees_paid:.2f}")
        print(f"  Total Slippage Cost: ${self.total_slippage_cost:.2f}")
        print(f"  Total Costs:         ${self.total_fees_paid + self.total_slippage_cost:.2f}")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Wins:             {self.total_wins}")
        print(f"  Losses:           {self.total_losses}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Max Drawdown:     {self.max_drawdown:.2f}%")

        print(f"\nSignal Statistics:")
        print(f"  Total Signals:    {self.signals_generated}")
        print(f"  Long Signals:     {self.long_signals}")
        print(f"  Short Signals:    {self.short_signals}")
        print(f"  Skipped (Neutral): {self.signals_skipped_neutral}")
        print(f"  Skipped (Weak):    {self.signals_skipped_weak}")

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

            if losing_trades and sum(t["pnl"] for t in losing_trades) != 0:
                profit_factor = abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades))
                print(f"  Profit Factor:     {profit_factor:.2f}x")

            # Analyze by confluence score
            print(f"\nPerformance by Confluence Score:")
            for score in [3, 4]:
                score_trades = [t for t in self.trades if t["confluence_score"] == score]
                if score_trades:
                    score_wins = len([t for t in score_trades if t["pnl"] > 0])
                    score_wr = score_wins / len(score_trades) * 100
                    score_pnl = sum(t["pnl"] for t in score_trades)
                    print(f"    Score {score}/4: {len(score_trades)} trades, {score_wr:.1f}% WR, ${score_pnl:+.2f} P&L")

        print("\n" + "="*70)


def run_htf_confluence_test(
    symbols: list = None,
    days: int = 90,
    config: dict = None,
    ltf_interval: str = "15m",
    start_balance: float = 100.0,
    shared_capital: bool = False
):
    """
    Run HTF Confluence Strategy backtest.

    Args:
        symbols: List of symbols to test (default: BTC, ETH, BNB, DOT)
        days: Backtest period in days
        config: Strategy config (default: MODERATE)
        ltf_interval: LTF timeframe (15m or 1h)
        start_balance: Starting balance per symbol (or total if shared_capital=True)
        shared_capital: If True, split capital equally among symbols
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOTUSDT"]

    if config is None:
        config = MODERATE_CONFIG

    # Calculate per-symbol balance
    if shared_capital:
        per_symbol_balance = start_balance / len(symbols)
    else:
        per_symbol_balance = start_balance

    print("="*80)
    print(f"HTF CONFLUENCE STRATEGY BACKTEST - {days} DAYS")
    print("="*80)
    print("Strategy: HTF (4H) 200 EMA trend + LTF MACD/RSI/EMA confluence")
    print(f"  - HTF: 4H timeframe with 200 EMA trend filter")
    print(f"  - LTF: {ltf_interval} timeframe for entry signals")
    print(f"  - Entry: 3/4 confluence (HTF trend + EMA cross + RSI range + MACD)")
    print(f"  - Exit: {config['tp_roi']*100:.1f}% ROI TP | {config['sl_roi']*100:.1f}% ROI SL")
    print(f"  - Leverage: {config['leverage']}x")
    if shared_capital:
        print(f"  - SHARED CAPITAL: ${start_balance:.2f} split among {len(symbols)} symbols (${per_symbol_balance:.2f} each)")
    else:
        print(f"  - Balance per symbol: ${per_symbol_balance:.2f}")
    print("="*80)

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        backtester = HTFConfluenceBacktester(symbol, start_balance=per_symbol_balance, config=config)
        ltf_df, htf_df = backtester.get_historical_data(days=days, ltf_interval=ltf_interval)

        if ltf_df is not None and htf_df is not None and len(ltf_df) > 0 and len(htf_df) > 0:
            result = backtester.run_backtest(ltf_df, htf_df)
            all_results.append(result)
        else:
            print(f"ERROR: No data for {symbol}")

    # Print summary
    if all_results:
        print("\n" + "="*100)
        print("SUMMARY - HTF CONFLUENCE STRATEGY")
        if shared_capital:
            print(f"SHARED CAPITAL: ${start_balance:.2f} split among {len(symbols)} symbols")
        print("="*100)
        print(f"{'Symbol':<12} {'Start':>10} {'End':>10} {'P&L':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>10}")
        print("-"*90)

        for r in all_results:
            pnl = r['balance'] - per_symbol_balance
            print(f"{r['symbol']:<12} ${per_symbol_balance:>9.2f} ${r['balance']:>9.2f} ${pnl:>+9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>9.1f}%")

        print("-"*90)
        total_start = len(all_results) * per_symbol_balance
        total_end = sum(r['balance'] for r in all_results)
        total_pnl = total_end - total_start
        total_return = (total_end / total_start - 1) * 100 if total_start > 0 else 0
        total_trades = sum(r['total_trades'] for r in all_results)
        avg_winrate = sum(r['win_rate'] for r in all_results) / len(all_results) if all_results else 0
        max_dd = max(r['max_drawdown'] for r in all_results) if all_results else 0

        print(f"{'TOTAL':<12} ${total_start:>9.2f} ${total_end:>9.2f} ${total_pnl:>+9.2f} {total_return:>+9.1f}% {total_trades:>7} {avg_winrate:>7.1f}% {max_dd:>9.1f}%")
        print("="*100)

        # Save summary
        summary_file = f"htf_confluence_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, "w") as f:
            f.write("="*100 + "\n")
            f.write(f"HTF CONFLUENCE STRATEGY BACKTEST - {days} DAYS\n")
            f.write("="*100 + "\n")
            f.write(f"Config: Leverage={config['leverage']}x, TP={config['tp_roi']*100:.1f}% ROI, SL={config['sl_roi']*100:.1f}% ROI\n")
            f.write(f"LTF Interval: {ltf_interval}\n")
            if shared_capital:
                f.write(f"SHARED CAPITAL: ${start_balance:.2f} split among {len(symbols)} symbols (${per_symbol_balance:.2f} each)\n")
            f.write("="*100 + "\n\n")

            f.write(f"{'Symbol':<12} {'Start':>10} {'End':>10} {'P&L':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>10} {'Longs':>6} {'Shorts':>6}\n")
            f.write("-"*100 + "\n")

            for r in all_results:
                pnl = r['balance'] - per_symbol_balance
                f.write(f"{r['symbol']:<12} ${per_symbol_balance:>9.2f} ${r['balance']:>9.2f} ${pnl:>+9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>9.1f}% {r['long_signals']:>5} {r['short_signals']:>5}\n")

            f.write("-"*100 + "\n")
            f.write(f"{'TOTAL':<12} ${total_start:>9.2f} ${total_end:>9.2f} ${total_pnl:>+9.2f} {total_return:>+9.1f}% {total_trades:>7} {avg_winrate:>7.1f}% {max_dd:>9.1f}%\n")
            f.write("="*100 + "\n")

        print(f"\n>>> Summary saved to: {summary_file}")

    return all_results


def compare_configs():
    """Compare different risk profiles"""
    print("\n" + "="*100)
    print("COMPARING DIFFERENT RISK PROFILES")
    print("="*100)

    configs = {
        "CONSERVATIVE": CONSERVATIVE_CONFIG,
        "MODERATE": MODERATE_CONFIG,
        "AGGRESSIVE": AGGRESSIVE_CONFIG,
    }

    all_results = {}

    for name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name} (Leverage: {config['leverage']}x, TP: {config['tp_roi']*100}%, SL: {config['sl_roi']*100}%)")
        print(f"{'='*80}")

        results = run_htf_confluence_test(
            symbols=["BTCUSDT"],  # Just BTC for comparison
            days=90,
            config=config
        )
        all_results[name] = results[0] if results else None

    # Print comparison
    print("\n" + "="*100)
    print("CONFIG COMPARISON - BTCUSDT 90 DAYS")
    print("="*100)
    print(f"{'Config':<15} {'Leverage':>8} {'TP ROI':>8} {'SL ROI':>8} {'Return':>10} {'Win%':>8} {'Trades':>8} {'MaxDD':>10}")
    print("-"*90)

    for name, config in configs.items():
        r = all_results.get(name)
        if r:
            print(f"{name:<15} {config['leverage']:>7}x {config['tp_roi']*100:>7.1f}% {config['sl_roi']*100:>7.1f}% {r['return_pct']:>+9.1f}% {r['win_rate']:>7.1f}% {r['total_trades']:>7} {r['max_drawdown']:>9.1f}%")

    print("="*100)


def optimize_asset_settings(symbols: list = None, days: int = 60, leverage: int = 20):
    """
    Find optimal SL/TP settings for each asset.
    Tests multiple combinations and finds the best performing one per symbol.
    """
    if symbols is None:
        symbols = ["DOTUSDT", "BNBUSDT", "AVAXUSDT"]

    # SL/TP combinations to test (ROI values)
    # Format: (tp_roi, sl_roi, name)
    configs_to_test = [
        # Tight SL (current - too tight for volatile coins)
        (0.20, 0.10, "20TP/10SL"),  # 2:1 R:R
        (0.30, 0.10, "30TP/10SL"),  # 3:1 R:R (current MODERATE)
        (0.40, 0.10, "40TP/10SL"),  # 4:1 R:R

        # Medium SL (better for volatile coins like DOT)
        (0.20, 0.15, "20TP/15SL"),  # 1.3:1 R:R - higher win rate
        (0.30, 0.15, "30TP/15SL"),  # 2:1 R:R
        (0.40, 0.15, "40TP/15SL"),  # 2.7:1 R:R
        (0.50, 0.15, "50TP/15SL"),  # 3.3:1 R:R

        # Wide SL (for very volatile coins)
        (0.30, 0.20, "30TP/20SL"),  # 1.5:1 R:R - highest win rate
        (0.40, 0.20, "40TP/20SL"),  # 2:1 R:R
        (0.50, 0.20, "50TP/20SL"),  # 2.5:1 R:R
        (0.60, 0.20, "60TP/20SL"),  # 3:1 R:R

        # Scalping (tight TP, wider SL)
        (0.15, 0.15, "15TP/15SL"),  # 1:1 R:R - scalping
        (0.20, 0.20, "20TP/20SL"),  # 1:1 R:R - balanced
    ]

    print("="*120)
    print(f"ASSET-SPECIFIC SL/TP OPTIMIZATION - {days} DAYS")
    print("="*120)
    print(f"Testing {len(configs_to_test)} different SL/TP combinations per asset")
    print(f"Leverage: {leverage}x")
    print("="*120)

    # Store best config per symbol
    best_configs = {}
    all_optimization_results = {}

    for symbol in symbols:
        print(f"\n{'='*100}")
        print(f"OPTIMIZING: {symbol}")
        print(f"{'='*100}")

        symbol_results = []

        # First, fetch data once for this symbol
        temp_config = {"leverage": leverage, "tp_roi": 0.30, "sl_roi": 0.10}
        temp_backtester = HTFConfluenceBacktester(symbol, start_balance=100.0, config=temp_config)
        ltf_df, htf_df = temp_backtester.get_historical_data(days=days, ltf_interval="15m")

        if ltf_df is None or htf_df is None:
            print(f"ERROR: Could not fetch data for {symbol}")
            continue

        # Test each config
        for tp_roi, sl_roi, config_name in configs_to_test:
            config = {"leverage": leverage, "tp_roi": tp_roi, "sl_roi": sl_roi}

            # Create backtester with this config
            backtester = HTFConfluenceBacktester(symbol, start_balance=100.0, config=config)

            # Silence individual trade output for optimization
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                result = backtester.run_backtest(ltf_df.copy(), htf_df.copy())
            finally:
                sys.stdout = old_stdout

            # Store result
            symbol_results.append({
                "config_name": config_name,
                "tp_roi": tp_roi,
                "sl_roi": sl_roi,
                "return_pct": result["return_pct"],
                "win_rate": result["win_rate"],
                "total_trades": result["total_trades"],
                "max_drawdown": result["max_drawdown"],
                "pnl": result["total_pnl"],
                "profit_per_trade": result["total_pnl"] / result["total_trades"] if result["total_trades"] > 0 else 0
            })

        # Sort by return (primary) and win rate (secondary)
        symbol_results.sort(key=lambda x: (x["return_pct"], x["win_rate"]), reverse=True)

        all_optimization_results[symbol] = symbol_results

        # Print results for this symbol
        print(f"\n{'Config':<15} {'TP%':>6} {'SL%':>6} {'Return':>10} {'Win%':>8} {'Trades':>8} {'MaxDD':>8} {'$/Trade':>10}")
        print("-"*85)

        for r in symbol_results:
            # Highlight best config
            marker = " <<<" if r == symbol_results[0] else ""
            print(f"{r['config_name']:<15} {r['tp_roi']*100:>5.0f}% {r['sl_roi']*100:>5.0f}% {r['return_pct']:>+9.1f}% {r['win_rate']:>7.1f}% {r['total_trades']:>7} {r['max_drawdown']:>7.1f}% ${r['profit_per_trade']:>+8.2f}{marker}")

        # Store best config
        best = symbol_results[0]
        best_configs[symbol] = {
            "tp_roi": best["tp_roi"],
            "sl_roi": best["sl_roi"],
            "expected_return": best["return_pct"],
            "expected_winrate": best["win_rate"]
        }

    # Print final summary with recommended settings
    print("\n" + "="*120)
    print("OPTIMIZATION SUMMARY - RECOMMENDED ASSET-SPECIFIC SETTINGS")
    print("="*120)
    print(f"{'Symbol':<12} {'Best Config':<15} {'TP ROI':>8} {'SL ROI':>8} {'Price Move':>12} {'Return':>10} {'Win%':>8}")
    print("-"*85)

    for symbol, config in best_configs.items():
        tp_price_move = config["tp_roi"] / leverage * 100
        sl_price_move = config["sl_roi"] / leverage * 100
        config_name = f"{int(config['tp_roi']*100)}TP/{int(config['sl_roi']*100)}SL"
        print(f"{symbol:<12} {config_name:<15} {config['tp_roi']*100:>7.0f}% {config['sl_roi']*100:>7.0f}% TP:{tp_price_move:.2f}%/SL:{sl_price_move:.2f}% {config['expected_return']:>+9.1f}% {config['expected_winrate']:>7.1f}%")

    print("\n" + "="*120)
    print("PYTHON CONFIG TO USE:")
    print("="*120)
    print("ASSET_SPECIFIC_CONFIG = {")
    for symbol, config in best_configs.items():
        print(f'    "{symbol}": {{"leverage": {leverage}, "tp_roi": {config["tp_roi"]}, "sl_roi": {config["sl_roi"]}}},')
    print("}")
    print("="*120)

    return best_configs, all_optimization_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest HTF Confluence Strategy")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOTUSDT"],
                        help="Symbols to backtest")
    parser.add_argument("--days", type=int, default=60, help="Backtest period in days")
    parser.add_argument("--config", choices=["conservative", "moderate", "aggressive"],
                        default="moderate", help="Risk profile")
    parser.add_argument("--ltf", choices=["15m", "1h"], default="15m",
                        help="LTF interval for entry signals")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all risk profiles")
    parser.add_argument("--optimize", action="store_true",
                        help="Find optimal SL/TP settings for each asset")
    parser.add_argument("--balance", type=float, default=100.0,
                        help="Starting balance (total if --shared is used)")
    parser.add_argument("--shared", action="store_true",
                        help="Split balance equally among symbols (shared capital)")
    parser.add_argument("--leverage", type=int, default=20,
                        help="Leverage to use (default: 20)")

    args = parser.parse_args()

    if args.compare:
        compare_configs()
    elif args.optimize:
        optimize_asset_settings(
            symbols=args.symbols,
            days=args.days,
            leverage=args.leverage
        )
    else:
        config_map = {
            "conservative": CONSERVATIVE_CONFIG,
            "moderate": MODERATE_CONFIG,
            "aggressive": AGGRESSIVE_CONFIG
        }
        run_htf_confluence_test(
            symbols=args.symbols,
            days=args.days,
            config=config_map[args.config],
            ltf_interval=args.ltf,
            start_balance=args.balance,
            shared_capital=args.shared
        )
