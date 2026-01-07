#!/usr/bin/env python3
"""
Backtest HTF Confluence Strategy V2 with 15m/5m Timeframes
==========================================================
Tests the V2 strategy (ATR-based SL/TP, pullback entries, volatility filter)
using 15m HTF and 5m LTF for scalping.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import csv
import requests
from datetime import datetime, timedelta

# Import V2 strategy
from strategies.htf_strategy_v2 import (
    HTFConfluenceStrategyV2,
    V2_CONSERVATIVE,
    V2_MODERATE,
    V2_AGGRESSIVE,
    TrendDirection,
    SignalStrength,
    MarketCondition
)


class HTFConfluenceV2Backtester:
    """Backtester for HTF Confluence Strategy V2 with ATR-based exits"""

    def __init__(
        self,
        symbol: str,
        start_balance: float = 100.0,
        config: dict = None,
        require_pullback: bool = False,  # Can toggle pullback requirement
        require_candle: bool = False      # Can toggle candle confirmation
    ):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance

        # Use moderate config by default
        if config is None:
            config = V2_MODERATE

        self.leverage = config.get("leverage", 20)
        self.require_pullback = require_pullback
        self.require_candle = require_candle

        # Strategy V2
        self.strategy = HTFConfluenceStrategyV2(**config)

        # Position sizing - risk 2% of balance per trade
        self.risk_per_trade = 0.02

        # Trading costs (realistic Binance Futures)
        self.taker_fee = 0.0003      # 0.03% taker fee
        self.maker_fee = 0.00006     # 0.006% maker fee
        self.slippage = 0.0001       # 0.01% slippage
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
        self.signals_skipped_volatile = 0
        self.signals_skipped_ranging = 0
        self.signals_skipped_pullback = 0
        self.signals_skipped_candle = 0

        # Cost tracking
        self.total_fees_paid = 0
        self.total_slippage_cost = 0

        print(f"[{symbol}] HTF Confluence Strategy V2 (ATR-based)")
        print(f"  Leverage: {self.leverage}x")
        print(f"  ATR SL Multiplier: {self.strategy.atr_sl_multiplier}x")
        print(f"  ATR TP Multiplier: {self.strategy.atr_tp_multiplier}x")
        print(f"  Require Pullback: {require_pullback}")
        print(f"  Require Candle Confirm: {require_candle}")

    def get_historical_data(self, days: int = 30):
        """
        Fetch historical data - 5m LTF and 15m HTF for scalping.
        """
        print(f"Fetching {days} days of data for {self.symbol}...")

        def fetch_binance_futures_klines(symbol, interval, num_days):
            base_url = "https://fapi.binance.com"
            endpoint = "/fapi/v1/klines"

            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=num_days)).timestamp() * 1000)

            all_klines = []
            limit = 1500

            while start_time < end_time:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "limit": limit
                }

                response = requests.get(base_url + endpoint, params=params)
                data = response.json()

                if not data or isinstance(data, dict):
                    break

                all_klines.extend(data)
                start_time = data[-1][0] + 1

                if len(data) < limit:
                    break

            if not all_klines:
                return None

            df = pd.DataFrame(all_klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        try:
            # Original timeframes: LTF = 15m, HTF = 4H (better for swing trading)
            ltf_interval = "15m"
            htf_interval = "4h"

            ltf_days = days  # 15m data can go further
            print(f"  Fetching {ltf_interval} data (last {ltf_days} days)...")
            ltf_df = fetch_binance_futures_klines(self.symbol, ltf_interval, ltf_days)

            if ltf_df is None or len(ltf_df) == 0:
                print(f"ERROR: No LTF data returned for {self.symbol}")
                return None, None

            htf_days = days + 10
            print(f"  Fetching {htf_interval} data (last {htf_days} days)...")
            htf_df = fetch_binance_futures_klines(self.symbol, htf_interval, htf_days)

            if htf_df is None or len(htf_df) == 0:
                print(f"ERROR: No HTF data returned for {self.symbol}")
                return None, None

            print(f"  LTF ({ltf_interval}): {len(ltf_df)} candles")
            print(f"  HTF ({htf_interval}): {len(htf_df)} candles")

            return ltf_df, htf_df

        except Exception as e:
            print(f"[ERROR] Binance Futures API failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def open_position(self, side: str, price: float, timestamp, signal) -> dict:
        """Open new position with ATR-based SL/TP"""
        # Apply slippage
        if side in ["LONG", "BUY"]:
            actual_entry = price * (1 + self.slippage)
        else:
            actual_entry = price * (1 - self.slippage)

        # Calculate position size based on ATR-based SL
        # SL distance is already calculated by strategy
        sl_distance_pct = abs(signal.stop_loss - actual_entry) / actual_entry
        risk_amount = self.balance * self.risk_per_trade
        position_value = risk_amount / sl_distance_pct
        margin = position_value / self.leverage
        quantity = position_value / actual_entry

        # Cap margin at available balance
        if margin > self.balance:
            margin = self.balance * 0.5  # Use max 50% of balance
            position_value = margin * self.leverage
            quantity = position_value / actual_entry

        # Entry fee
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        entry_fee = position_value * fee_rate
        self.balance -= entry_fee
        self.total_fees_paid += entry_fee

        # Track slippage
        slippage_cost = abs(actual_entry - price) * quantity
        self.total_slippage_cost += slippage_cost

        # Recalculate SL/TP based on actual entry
        stop_loss, take_profit = self.strategy.calculate_atr_based_exits(
            actual_entry, signal.atr_value, side
        )

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
            "atr_value": signal.atr_value,
            "market_condition": signal.market_condition.value,
            "entry_fee": entry_fee
        }

        if side in ["LONG", "BUY"]:
            self.long_signals += 1
        else:
            self.short_signals += 1

        self.signals_generated += 1

        sl_pct = (stop_loss / actual_entry - 1) * 100
        tp_pct = (take_profit / actual_entry - 1) * 100

        print(f"[{timestamp}] OPEN {side} @ ${actual_entry:.2f}")
        print(f"    ATR: ${signal.atr_value:.4f} | SL: ${stop_loss:.2f} ({sl_pct:+.2f}%) | TP: ${take_profit:.2f} ({tp_pct:+.2f}%)")

        return position

    def close_position(self, exit_price: float, exit_type: str, timestamp) -> float:
        """Close position and calculate P&L"""
        if self.position is None:
            return 0

        # Apply slippage
        is_long = self.position["side"] in ["LONG", "BUY"]
        if is_long:
            actual_exit = exit_price * (1 - self.slippage)
        else:
            actual_exit = exit_price * (1 + self.slippage)

        # Exit fee
        position_value = self.position["quantity"] * actual_exit
        fee_rate = self.maker_fee if self.use_maker else self.taker_fee
        exit_fee = position_value * fee_rate
        self.total_fees_paid += exit_fee

        # Slippage cost
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
            "strength": self.position["strength"],
            "atr_value": self.position["atr_value"],
            "market_condition": self.position["market_condition"]
        })

        print(f"[{timestamp}] CLOSE {self.position['side']} @ ${actual_exit:.2f} | {exit_type} | P&L: ${pnl:+.2f}")

        self.position = None
        return pnl

    def get_htf_data_at_time(self, htf_df: pd.DataFrame, current_time) -> pd.DataFrame:
        """Get HTF data available at a specific time (avoid lookahead)"""
        return htf_df[htf_df.index <= current_time]

    def run_backtest(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """Run the backtest"""
        print("\n" + "="*70)
        print("RUNNING HTF CONFLUENCE STRATEGY V2 BACKTEST")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Timeframes: 5m (LTF) / 15m (HTF) - SCALPING MODE")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"ATR SL: {self.strategy.atr_sl_multiplier}x | ATR TP: {self.strategy.atr_tp_multiplier}x")
        print(f"LTF Period: {ltf_df.index[0]} to {ltf_df.index[-1]}")
        print("="*70)

        # Warmup period
        warmup = max(50, self.strategy.htf_ema_period * 2)

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

            # Get HTF data (avoid lookahead)
            htf_available = self.get_htf_data_at_time(htf_df, timestamp)

            if len(htf_available) < self.strategy.htf_ema_period:
                continue

            # Check existing position
            if self.position is not None:
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
                ltf_available = ltf_df.iloc[:i+1]

                # Get signal from V2 strategy
                signal = self.strategy.should_enter(
                    ltf_available,
                    htf_available,
                    current_bar=i,
                    require_pullback=self.require_pullback,
                    require_candle=self.require_candle
                )

                if signal.action:
                    self.position = self.open_position(signal.action, close, timestamp, signal)
                elif signal.market_condition == MarketCondition.VOLATILE:
                    self.signals_skipped_volatile += 1
                elif signal.market_condition == MarketCondition.RANGING:
                    self.signals_skipped_ranging += 1
                elif "pullback" in signal.reason.lower():
                    self.signals_skipped_pullback += 1
                elif "candle" in signal.reason.lower():
                    self.signals_skipped_candle += 1

        # Close any open position at end
        if self.position is not None:
            final_price = ltf_df['close'].iloc[-1]
            self.close_position(final_price, "END_OF_BACKTEST", ltf_df.index[-1])

        self.print_results(ltf_df, htf_df)

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

    def print_results(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - HTF CONFLUENCE STRATEGY V2")
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

        print(f"\nCosts:")
        print(f"  Total Fees:     ${self.total_fees_paid:.2f}")
        print(f"  Total Slippage: ${self.total_slippage_cost:.2f}")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:   {total_trades}")
        print(f"  Wins:           {self.total_wins}")
        print(f"  Losses:         {self.total_losses}")
        print(f"  Win Rate:       {win_rate:.1f}%")
        print(f"  Max Drawdown:   {self.max_drawdown:.2f}%")

        print(f"\nSignal Statistics:")
        print(f"  Total Signals:       {self.signals_generated}")
        print(f"  Long Signals:        {self.long_signals}")
        print(f"  Short Signals:       {self.short_signals}")
        print(f"  Skipped (Volatile):  {self.signals_skipped_volatile}")
        print(f"  Skipped (Ranging):   {self.signals_skipped_ranging}")
        print(f"  Skipped (Pullback):  {self.signals_skipped_pullback}")
        print(f"  Skipped (Candle):    {self.signals_skipped_candle}")

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

            # Analyze by market condition
            print(f"\nPerformance by Market Condition:")
            for condition in ["TRENDING", "RANGING"]:
                cond_trades = [t for t in self.trades if t.get("market_condition") == condition]
                if cond_trades:
                    cond_wins = len([t for t in cond_trades if t["pnl"] > 0])
                    cond_wr = cond_wins / len(cond_trades) * 100
                    cond_pnl = sum(t["pnl"] for t in cond_trades)
                    print(f"    {condition}: {len(cond_trades)} trades, {cond_wr:.1f}% WR, ${cond_pnl:+.2f} P&L")

        print("\n" + "="*70)


def run_v2_backtest(
    symbols: list = None,
    days: int = 30,
    config: dict = None,
    require_pullback: bool = False,
    require_candle: bool = False,
    start_balance: float = 100.0
):
    """Run V2 strategy backtest"""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOTUSDT"]

    if config is None:
        config = V2_MODERATE

    print("="*80)
    print(f"HTF CONFLUENCE STRATEGY V2 BACKTEST - {days} DAYS")
    print("="*80)
    print("Strategy V2: ATR-based SL/TP + Volatility Filter")
    print(f"  - HTF: 15m timeframe (trend filter)")
    print(f"  - LTF: 5m timeframe (entry signals)")
    print(f"  - ATR SL: {config.get('atr_sl_multiplier', 1.5)}x | ATR TP: {config.get('atr_tp_multiplier', 3.0)}x")
    print(f"  - Volatility filter: Skip if ATR > {config.get('max_atr_percent', 2.0)}%")
    print(f"  - Require Pullback: {require_pullback}")
    print(f"  - Require Candle: {require_candle}")
    print("="*80)

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        backtester = HTFConfluenceV2Backtester(
            symbol,
            start_balance=start_balance,
            config=config,
            require_pullback=require_pullback,
            require_candle=require_candle
        )
        ltf_df, htf_df = backtester.get_historical_data(days=days)

        if ltf_df is not None and htf_df is not None:
            result = backtester.run_backtest(ltf_df, htf_df)
            all_results.append(result)
        else:
            print(f"ERROR: No data for {symbol}")

    # Print summary
    if all_results:
        print("\n" + "="*100)
        print("SUMMARY - HTF CONFLUENCE STRATEGY V2 (5m/15m SCALPING)")
        print("="*100)
        print(f"{'Symbol':<12} {'Start':>10} {'End':>10} {'P&L':>10} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>10}")
        print("-"*90)

        for r in all_results:
            pnl = r['balance'] - start_balance
            print(f"{r['symbol']:<12} ${start_balance:>9.2f} ${r['balance']:>9.2f} ${pnl:>+9.2f} {r['return_pct']:>+9.1f}% {r['total_trades']:>7} {r['win_rate']:>7.1f}% {r['max_drawdown']:>9.1f}%")

        print("-"*90)
        total_start = len(all_results) * start_balance
        total_end = sum(r['balance'] for r in all_results)
        total_pnl = total_end - total_start
        total_return = (total_end / total_start - 1) * 100 if total_start > 0 else 0
        total_trades = sum(r['total_trades'] for r in all_results)
        avg_winrate = sum(r['win_rate'] for r in all_results) / len(all_results) if all_results else 0
        max_dd = max(r['max_drawdown'] for r in all_results) if all_results else 0

        print(f"{'TOTAL':<12} ${total_start:>9.2f} ${total_end:>9.2f} ${total_pnl:>+9.2f} {total_return:>+9.1f}% {total_trades:>7} {avg_winrate:>7.1f}% {max_dd:>9.1f}%")
        print("="*100)

    return all_results


def compare_v2_configs():
    """Compare V2 configurations"""
    print("\n" + "="*100)
    print("COMPARING V2 CONFIGURATIONS")
    print("="*100)

    configs = {
        "V2_CONSERVATIVE": V2_CONSERVATIVE,
        "V2_MODERATE": V2_MODERATE,
        "V2_AGGRESSIVE": V2_AGGRESSIVE,
    }

    all_results = {}

    for name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"  ATR SL: {config['atr_sl_multiplier']}x | ATR TP: {config['atr_tp_multiplier']}x")
        print(f"{'='*80}")

        results = run_v2_backtest(
            symbols=["BTCUSDT"],
            days=30,
            config=config
        )
        all_results[name] = results[0] if results else None

    # Print comparison
    print("\n" + "="*100)
    print("V2 CONFIG COMPARISON - BTCUSDT 30 DAYS (5m/15m)")
    print("="*100)
    print(f"{'Config':<18} {'ATR SL':>8} {'ATR TP':>8} {'Return':>10} {'Win%':>8} {'Trades':>8} {'MaxDD':>10}")
    print("-"*80)

    for name, config in configs.items():
        r = all_results.get(name)
        if r:
            print(f"{name:<18} {config['atr_sl_multiplier']:>7}x {config['atr_tp_multiplier']:>7}x {r['return_pct']:>+9.1f}% {r['win_rate']:>7.1f}% {r['total_trades']:>7} {r['max_drawdown']:>9.1f}%")

    print("="*100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest HTF Confluence Strategy V2")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOTUSDT"],
                        help="Symbols to backtest")
    parser.add_argument("--days", type=int, default=30, help="Backtest period in days (max 30 for 5m data)")
    parser.add_argument("--config", choices=["conservative", "moderate", "aggressive"],
                        default="moderate", help="V2 config profile")
    parser.add_argument("--pullback", action="store_true", help="Require pullback entry")
    parser.add_argument("--candle", action="store_true", help="Require candle confirmation")
    parser.add_argument("--compare", action="store_true", help="Compare V2 configs")
    parser.add_argument("--balance", type=float, default=100.0, help="Starting balance")

    args = parser.parse_args()

    if args.compare:
        compare_v2_configs()
    else:
        config_map = {
            "conservative": V2_CONSERVATIVE,
            "moderate": V2_MODERATE,
            "aggressive": V2_AGGRESSIVE
        }
        run_v2_backtest(
            symbols=args.symbols,
            days=args.days,
            config=config_map[args.config],
            require_pullback=args.pullback,
            require_candle=args.candle,
            start_balance=args.balance
        )
