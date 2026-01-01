#!/usr/bin/env python3
"""
Crash vs Normal Market Backtest
===============================
Compare DCA 3 Safe Boost performance in:
1. Market Crash period (last 30 days)
2. Normal/Bull market period (60-90 days ago)

Test across: DOT, AVAX, ETH, BTC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG


class CrashVsNormalBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]

        self.tp_roi = DCA_CONFIG["take_profit_roi"]
        self.sl_roi = DCA_CONFIG["stop_loss_roi"]
        self.dca_levels = DCA_CONFIG["levels"]
        self.budget_split = DCA_CONFIG["hedge_mode"]["budget_split"]

        self.dca_pcts = [
            DCA_CONFIG.get("initial_entry_pct", 0.10),
            DCA_CONFIG.get("dca1_pct", 0.15),
            DCA_CONFIG.get("dca2_pct", 0.20),
            DCA_CONFIG.get("dca3_pct", 0.25),
            DCA_CONFIG.get("dca4_pct", 0.30),
        ]

        # SAFE BOOST PARAMETERS - DCA 3 trigger
        self.boost_multiplier = 1.5
        self.boost_trigger_dca_level = 3
        self.trailing_activation_roi = 0.02
        self.trailing_distance_roi = 0.03

        self.long_position = None
        self.short_position = None
        self.boost_mode_active = False
        self.boosted_side = None
        self.boost_trigger_side = None
        self.boosted_peak_roi = 0
        self.trailing_active = False

        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance
        self.boost_activations = 0
        self.boost_profits = 0
        self.trailing_closes = 0

    def reset(self):
        """Reset state for new backtest"""
        self.balance = self.start_balance
        self.long_position = None
        self.short_position = None
        self.boost_mode_active = False
        self.boosted_side = None
        self.boost_trigger_side = None
        self.boosted_peak_roi = 0
        self.trailing_active = False
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = self.start_balance
        self.boost_activations = 0
        self.boost_profits = 0
        self.trailing_closes = 0

    def get_historical_data(self, days_ago_start: int, days_ago_end: int, interval: str = "1h"):
        """Fetch historical data for a specific period"""
        client = BinanceClient(testnet=False, use_demo=False)

        end_time = datetime.now() - timedelta(days=days_ago_end)
        start_time = datetime.now() - timedelta(days=days_ago_start)

        df = client.get_klines(
            self.symbol,
            interval=interval,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
            limit=1000
        )

        if df is None or (hasattr(df, 'empty') and df.empty) or len(df) == 0:
            return None

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    def calculate_roi(self, entry_price: float, current_price: float, side: str) -> float:
        if side == "LONG":
            price_pct = (current_price - entry_price) / entry_price
        else:
            price_pct = (entry_price - current_price) / entry_price
        return price_pct * self.leverage

    def get_dca_level(self, position: dict) -> int:
        return position.get("dca_level", 0)

    def get_tp_price(self, entry_price: float, side: str, dca_level: int) -> float:
        if dca_level > 0 and dca_level <= len(self.dca_levels):
            tp_roi = self.dca_levels[dca_level - 1].get("tp_roi", self.tp_roi)
        else:
            tp_roi = self.tp_roi
        tp_pct = tp_roi / self.leverage
        if side == "LONG":
            return entry_price * (1 + tp_pct)
        else:
            return entry_price * (1 - tp_pct)

    def get_sl_price(self, entry_price: float, side: str) -> float:
        sl_pct = self.sl_roi / self.leverage
        if side == "LONG":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def check_dca_trigger(self, position: dict, current_price: float) -> bool:
        dca_level = self.get_dca_level(position)
        if dca_level >= len(self.dca_levels):
            return False
        trigger_roi = self.dca_levels[dca_level]["trigger_roi"]
        current_roi = self.calculate_roi(position["entry_price"], current_price, position["side"])
        return current_roi <= trigger_roi

    def execute_dca(self, position: dict, current_price: float) -> dict:
        dca_level = self.get_dca_level(position)
        add_pct = self.dca_pcts[dca_level + 1] if dca_level + 1 < len(self.dca_pcts) else 0.15

        old_qty = position["quantity"]
        old_margin = position["margin"]
        add_margin = (self.start_balance * self.budget_split) * add_pct
        add_qty = (add_margin * self.leverage) / current_price

        new_qty = old_qty + add_qty
        new_margin = old_margin + add_margin
        old_value = old_qty * position["entry_price"]
        add_value = add_qty * current_price
        new_entry = (old_value + add_value) / new_qty

        position["quantity"] = new_qty
        position["margin"] = new_margin
        position["entry_price"] = new_entry
        position["dca_level"] = dca_level + 1
        position["tp_price"] = self.get_tp_price(new_entry, position["side"], dca_level + 1)
        position["sl_price"] = self.get_sl_price(new_entry, position["side"])
        return position

    def open_position(self, side: str, price: float, boost: bool = False) -> dict:
        budget = self.start_balance * self.budget_split
        initial_margin = budget * self.dca_pcts[0]
        if boost:
            initial_margin *= self.boost_multiplier
        quantity = (initial_margin * self.leverage) / price
        return {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0),
            "sl_price": self.get_sl_price(price, side),
            "is_boosted": boost
        }

    def close_position(self, position: dict, exit_price: float, exit_type: str):
        price_change_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        if position["side"] == "LONG":
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        pnl = position["margin"] * roi
        if pnl < -position["margin"]:
            pnl = -position["margin"]

        self.balance += pnl
        self.total_pnl += pnl

        if self.balance <= 0:
            self.balance = 0

        if pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.trades.append({
            "pnl": pnl,
            "exit_type": exit_type,
            "is_boosted": position.get("is_boosted", False),
            "dca_level": position["dca_level"]
        })
        return pnl

    def activate_boost_mode(self, trigger_side: str):
        self.boost_mode_active = True
        self.boosted_side = "SHORT" if trigger_side == "LONG" else "LONG"
        self.boost_trigger_side = trigger_side
        self.boost_activations += 1
        self.boosted_peak_roi = 0
        self.trailing_active = False

    def deactivate_boost_mode(self):
        if not self.boost_mode_active:
            return
        self.boost_mode_active = False
        self.boosted_side = None
        self.boost_trigger_side = None
        self.boosted_peak_roi = 0
        self.trailing_active = False

    def check_trailing_stop(self, position: dict, current_price: float) -> tuple:
        if not position.get("is_boosted", False):
            return False, None

        current_roi = self.calculate_roi(position["entry_price"], current_price, position["side"])

        if current_roi > self.boosted_peak_roi:
            self.boosted_peak_roi = current_roi

        if not self.trailing_active and current_roi >= self.trailing_activation_roi:
            self.trailing_active = True

        if self.trailing_active:
            trailing_trigger_roi = self.boosted_peak_roi - self.trailing_distance_roi
            if current_roi <= trailing_trigger_roi and trailing_trigger_roi > 0:
                if position["side"] == "LONG":
                    trailing_price = position["entry_price"] * (1 + trailing_trigger_roi / self.leverage)
                else:
                    trailing_price = position["entry_price"] * (1 - trailing_trigger_roi / self.leverage)
                return True, trailing_price

        return False, None

    def run_backtest(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return None

        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]
        price_change = (last_price - first_price) / first_price * 100

        self.long_position = self.open_position("LONG", first_price)
        self.short_position = self.open_position("SHORT", first_price)

        for i, (timestamp, row) in enumerate(df.iterrows()):
            if self.balance <= 0:
                break

            high = row['high']
            low = row['low']
            close = row['close']

            # LONG position logic
            if self.long_position:
                is_boosted = self.long_position.get("is_boosted", False)

                if is_boosted and self.boost_mode_active and self.boosted_side == "LONG":
                    should_trail_close, trail_price = self.check_trailing_stop(self.long_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.long_position, trail_price, "TRAILING")
                        self.boost_profits += pnl
                        self.trailing_closes += 1
                        boost = self.boost_mode_active and self.boosted_side == "LONG"
                        self.long_position = self.open_position("LONG", close, boost=boost)
                        self.boosted_peak_roi = 0
                        self.trailing_active = False
                        continue

                if high >= self.long_position["tp_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["tp_price"], "TP")
                    if is_boosted:
                        self.boost_profits += pnl
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode()
                    boost = self.boost_mode_active and self.boosted_side == "LONG"
                    self.long_position = self.open_position("LONG", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                elif low <= self.long_position["sl_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["sl_price"], "SL")
                    if is_boosted:
                        self.boost_profits += pnl
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode()
                    boost = self.boost_mode_active and self.boosted_side == "LONG"
                    self.long_position = self.open_position("LONG", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                elif self.check_dca_trigger(self.long_position, close):
                    if self.boost_mode_active and self.boosted_side == "LONG":
                        pass
                    else:
                        self.long_position = self.execute_dca(self.long_position, close)
                        new_level = self.long_position["dca_level"]
                        if new_level == self.boost_trigger_dca_level and not self.boost_mode_active:
                            self.activate_boost_mode("LONG")
                            if self.short_position:
                                self.short_position["quantity"] *= self.boost_multiplier
                                self.short_position["margin"] *= self.boost_multiplier
                                self.short_position["is_boosted"] = True

            # SHORT position logic
            if self.short_position:
                is_boosted = self.short_position.get("is_boosted", False)

                if is_boosted and self.boost_mode_active and self.boosted_side == "SHORT":
                    should_trail_close, trail_price = self.check_trailing_stop(self.short_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.short_position, trail_price, "TRAILING")
                        self.boost_profits += pnl
                        self.trailing_closes += 1
                        boost = self.boost_mode_active and self.boosted_side == "SHORT"
                        self.short_position = self.open_position("SHORT", close, boost=boost)
                        self.boosted_peak_roi = 0
                        self.trailing_active = False
                        continue

                if low <= self.short_position["tp_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["tp_price"], "TP")
                    if is_boosted:
                        self.boost_profits += pnl
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode()
                    boost = self.boost_mode_active and self.boosted_side == "SHORT"
                    self.short_position = self.open_position("SHORT", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                elif high >= self.short_position["sl_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["sl_price"], "SL")
                    if is_boosted:
                        self.boost_profits += pnl
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode()
                    boost = self.boost_mode_active and self.boosted_side == "SHORT"
                    self.short_position = self.open_position("SHORT", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                elif self.check_dca_trigger(self.short_position, close):
                    if self.boost_mode_active and self.boosted_side == "SHORT":
                        pass
                    else:
                        self.short_position = self.execute_dca(self.short_position, close)
                        new_level = self.short_position["dca_level"]
                        if new_level == self.boost_trigger_dca_level and not self.boost_mode_active:
                            self.activate_boost_mode("SHORT")
                            if self.long_position:
                                self.long_position["quantity"] *= self.boost_multiplier
                                self.long_position["margin"] *= self.boost_multiplier
                                self.long_position["is_boosted"] = True

        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0
        total_return = ((self.balance - self.start_balance) / self.start_balance * 100)

        return {
            "symbol": self.symbol,
            "price_change": price_change,
            "ending_balance": self.balance,
            "total_return": total_return,
            "total_trades": total_trades,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "boost_activations": self.boost_activations,
            "boost_pnl": self.boost_profits,
            "liquidated": self.balance <= 0
        }


def main():
    print("="*90)
    print("CRASH vs NORMAL MARKET - DCA 3 SAFE BOOST BACKTEST")
    print("="*90)
    print("Testing: DOT, AVAX, ETH, BTC")
    print("Periods: Last 30 days (Crash) vs 60-90 days ago (Normal)")
    print("Strategy: Safe Boost with DCA 3 trigger, 1.5x, No DCA on boost, Trailing")
    print("="*90)

    symbols = ["DOTUSDT", "AVAXUSDT", "ETHUSDT", "BTCUSDT"]

    # Period definitions
    periods = {
        "CRASH (0-30 days)": {"start": 30, "end": 0},
        "NORMAL (60-90 days)": {"start": 90, "end": 60}
    }

    all_results = []

    for period_name, period_range in periods.items():
        print(f"\n{'='*90}")
        print(f"TESTING: {period_name}")
        print(f"{'='*90}")

        period_results = []

        for symbol in symbols:
            bt = CrashVsNormalBacktester(symbol, start_balance=100.0)
            df = bt.get_historical_data(
                days_ago_start=period_range["start"],
                days_ago_end=period_range["end"],
                interval="1h"
            )

            if df is None or len(df) == 0:
                print(f"  {symbol}: No data available")
                continue

            result = bt.run_backtest(df)
            if result:
                result["period"] = period_name
                period_results.append(result)
                all_results.append(result)

                status = "LIQUIDATED!" if result["liquidated"] else ""
                print(f"  {symbol}: Price {result['price_change']:+.2f}% | Balance ${result['ending_balance']:.2f} ({result['total_return']:+.2f}%) | WR {result['win_rate']:.1f}% | DD {result['max_drawdown']:.1f}% {status}")

        # Period summary
        if period_results:
            total_balance = sum(r['ending_balance'] for r in period_results)
            total_trades = sum(r['total_trades'] for r in period_results)
            total_wins = sum(r['wins'] for r in period_results)
            total_losses = sum(r['losses'] for r in period_results)
            avg_return = sum(r['total_return'] for r in period_results) / len(period_results)
            combined_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            liquidations = sum(1 for r in period_results if r['liquidated'])

            print(f"\n  {period_name} SUMMARY:")
            print(f"    Combined Balance: ${total_balance:.2f} (from $400 starting)")
            print(f"    Combined Profit:  ${total_balance - 400:.2f}")
            print(f"    Average Return:   {avg_return:+.2f}%")
            print(f"    Total Trades:     {total_trades}")
            print(f"    Wins/Losses:      {total_wins}/{total_losses}")
            print(f"    Combined Win Rate:{combined_win_rate:.2f}%")
            print(f"    Liquidations:     {liquidations}")

    # Final comparison
    print("\n")
    print("="*90)
    print("FINAL COMPARISON: CRASH vs NORMAL MARKET")
    print("="*90)

    crash_results = [r for r in all_results if "CRASH" in r["period"]]
    normal_results = [r for r in all_results if "NORMAL" in r["period"]]

    print(f"\n{'Metric':<25} {'CRASH Period':<20} {'NORMAL Period':<20}")
    print("-"*65)

    if crash_results:
        crash_balance = sum(r['ending_balance'] for r in crash_results)
        crash_profit = crash_balance - 400
        crash_trades = sum(r['total_trades'] for r in crash_results)
        crash_wins = sum(r['wins'] for r in crash_results)
        crash_wr = (crash_wins / crash_trades * 100) if crash_trades > 0 else 0
        crash_liq = sum(1 for r in crash_results if r['liquidated'])
    else:
        crash_balance = crash_profit = crash_trades = crash_wins = crash_wr = crash_liq = 0

    if normal_results:
        normal_balance = sum(r['ending_balance'] for r in normal_results)
        normal_profit = normal_balance - 400
        normal_trades = sum(r['total_trades'] for r in normal_results)
        normal_wins = sum(r['wins'] for r in normal_results)
        normal_wr = (normal_wins / normal_trades * 100) if normal_trades > 0 else 0
        normal_liq = sum(1 for r in normal_results if r['liquidated'])
    else:
        normal_balance = normal_profit = normal_trades = normal_wins = normal_wr = normal_liq = 0

    print(f"{'Starting Balance':<25} $400.00              $400.00")
    print(f"{'Ending Balance':<25} ${crash_balance:>7.2f}             ${normal_balance:>7.2f}")
    print(f"{'Total Profit':<25} ${crash_profit:>+7.2f}             ${normal_profit:>+7.2f}")
    print(f"{'Total Trades':<25} {crash_trades:>7}               {normal_trades:>7}")
    print(f"{'Total Wins':<25} {crash_wins:>7}               {normal_wins:>7}")
    print(f"{'Combined Win Rate':<25} {crash_wr:>6.2f}%              {normal_wr:>6.2f}%")
    print(f"{'Liquidations':<25} {crash_liq:>7}               {normal_liq:>7}")

    # Per-symbol comparison
    print("\n")
    print("="*90)
    print("PER-SYMBOL COMPARISON")
    print("="*90)
    print(f"\n{'Symbol':<12} {'CRASH Balance':<15} {'NORMAL Balance':<15} {'CRASH %':<12} {'NORMAL %':<12}")
    print("-"*70)

    for symbol in symbols:
        crash_r = next((r for r in crash_results if r['symbol'] == symbol), None)
        normal_r = next((r for r in normal_results if r['symbol'] == symbol), None)

        crash_bal = f"${crash_r['ending_balance']:.2f}" if crash_r else "N/A"
        normal_bal = f"${normal_r['ending_balance']:.2f}" if normal_r else "N/A"
        crash_ret = f"{crash_r['total_return']:+.2f}%" if crash_r else "N/A"
        normal_ret = f"{normal_r['total_return']:+.2f}%" if normal_r else "N/A"

        print(f"{symbol:<12} {crash_bal:<15} {normal_bal:<15} {crash_ret:<12} {normal_ret:<12}")

    # Grand total
    print("\n")
    print("="*90)
    print("GRAND TOTAL (ALL PERIODS COMBINED)")
    print("="*90)

    grand_balance = crash_balance + normal_balance
    grand_profit = grand_balance - 800  # $400 x 2 periods
    grand_trades = crash_trades + normal_trades
    grand_wins = crash_wins + normal_wins
    grand_wr = (grand_wins / grand_trades * 100) if grand_trades > 0 else 0
    grand_liq = crash_liq + normal_liq

    print(f"\n  Starting Capital: $800.00 ($400 per period)")
    print(f"  Ending Balance:   ${grand_balance:.2f}")
    print(f"  Total Profit:     ${grand_profit:+.2f}")
    print(f"  Total Return:     {(grand_profit/800)*100:+.2f}%")
    print(f"  Total Trades:     {grand_trades}")
    print(f"  Total Wins:       {grand_wins}")
    print(f"  Total Losses:     {grand_trades - grand_wins}")
    print(f"  COMBINED WIN RATE: {grand_wr:.2f}%")
    print(f"  Liquidations:     {grand_liq}")

    print("\n" + "="*90)


if __name__ == "__main__":
    main()
