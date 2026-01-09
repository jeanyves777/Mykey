#!/usr/bin/env python3
"""
Multi-Symbol Safe Boost Backtest
================================
Compare DCA 2 vs DCA 3 triggers across DOT, AVAX, ETH, BTC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG


class MultiSymbolBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0, boost_trigger_level: int = 2):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Strategy params
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

        # SAFE BOOST PARAMETERS
        self.boost_multiplier = 1.5
        self.boost_trigger_dca_level = boost_trigger_level  # 2 or 3
        self.trailing_activation_roi = 0.02
        self.trailing_distance_roi = 0.03

        # Positions
        self.long_position = None
        self.short_position = None

        # Boost mode tracking
        self.boost_mode_active = False
        self.boosted_side = None
        self.boost_trigger_side = None
        self.boosted_peak_roi = 0
        self.trailing_active = False

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance
        self.boost_activations = 0
        self.boost_profits = 0
        self.trailing_closes = 0

    def get_historical_data(self, days: int = 30, interval: str = "1h"):
        client = BinanceClient(testnet=False, use_demo=False)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

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

            # Check LONG position
            if self.long_position:
                is_boosted = self.long_position.get("is_boosted", False)

                # Check trailing stop
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

                # Check TP
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

                # Check SL
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

                # Check DCA
                elif self.check_dca_trigger(self.long_position, close):
                    if self.boost_mode_active and self.boosted_side == "LONG":
                        pass  # No DCA on boosted side
                    else:
                        self.long_position = self.execute_dca(self.long_position, close)
                        new_level = self.long_position["dca_level"]
                        if new_level == self.boost_trigger_dca_level and not self.boost_mode_active:
                            self.activate_boost_mode("LONG")
                            if self.short_position:
                                self.short_position["quantity"] *= self.boost_multiplier
                                self.short_position["margin"] *= self.boost_multiplier
                                self.short_position["is_boosted"] = True

            # Check SHORT position
            if self.short_position:
                is_boosted = self.short_position.get("is_boosted", False)

                # Check trailing stop
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

                # Check TP
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

                # Check SL
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

                # Check DCA
                elif self.check_dca_trigger(self.short_position, close):
                    if self.boost_mode_active and self.boosted_side == "SHORT":
                        pass  # No DCA on boosted side
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

        boosted_trades = [t for t in self.trades if t.get("is_boosted", False)]
        boosted_pnl = sum(t["pnl"] for t in boosted_trades) if boosted_trades else 0

        return {
            "symbol": self.symbol,
            "trigger_level": self.boost_trigger_dca_level,
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
            "boosted_trades": len(boosted_trades),
            "trailing_closes": self.trailing_closes
        }


def main():
    print("="*80)
    print("MULTI-SYMBOL SAFE BOOST BACKTEST - DCA 2 vs DCA 3 TRIGGER")
    print("="*80)
    print("Testing: DOT, AVAX, ETH, BTC")
    print("Comparing: DCA 2 trigger vs DCA 3 trigger")
    print("="*80)

    symbols = ["DOTUSDT", "AVAXUSDT", "ETHUSDT", "BTCUSDT"]
    trigger_levels = [2, 3]

    results = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}...")
        print(f"{'='*60}")

        # Get data once for both tests
        test_bt = MultiSymbolBacktester(symbol, start_balance=100.0, boost_trigger_level=2)
        df = test_bt.get_historical_data(days=30, interval="1h")

        if df is None or len(df) == 0:
            print(f"  ERROR: No data for {symbol}")
            continue

        price_start = df['close'].iloc[0]
        price_end = df['close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100
        print(f"  Price: ${price_start:.2f} -> ${price_end:.2f} ({price_change:+.2f}%)")
        print(f"  Candles: {len(df)}")

        for trigger_level in trigger_levels:
            bt = MultiSymbolBacktester(symbol, start_balance=100.0, boost_trigger_level=trigger_level)
            result = bt.run_backtest(df.copy())
            results.append(result)

            print(f"\n  DCA {trigger_level} Trigger:")
            print(f"    Balance: ${result['ending_balance']:.2f} ({result['total_return']:+.2f}%)")
            print(f"    Trades: {result['total_trades']} | Wins: {result['wins']} | Losses: {result['losses']}")
            print(f"    Win Rate: {result['win_rate']:.1f}% | Max DD: {result['max_drawdown']:.1f}%")
            print(f"    Boost: {result['boost_activations']} activations | P&L: ${result['boost_pnl']:+.2f}")

    # Print summary table
    print("\n")
    print("="*100)
    print("SUMMARY TABLE - DCA 2 vs DCA 3 TRIGGER")
    print("="*100)
    print(f"{'Symbol':<10} {'Trigger':<8} {'Price%':<10} {'Balance':<12} {'Return%':<10} {'MaxDD%':<10} {'Boost#':<8} {'BoostPnL':<12}")
    print("-"*100)

    for r in results:
        print(f"{r['symbol']:<10} DCA {r['trigger_level']:<5} {r['price_change']:>+7.2f}%  ${r['ending_balance']:>8.2f}   {r['total_return']:>+7.2f}%   {r['max_drawdown']:>7.2f}%   {r['boost_activations']:>5}    ${r['boost_pnl']:>+8.2f}")

    # Compare DCA 2 vs DCA 3 for each symbol
    print("\n")
    print("="*80)
    print("COMPARISON: DCA 2 vs DCA 3")
    print("="*80)

    for symbol in symbols:
        dca2 = next((r for r in results if r['symbol'] == symbol and r['trigger_level'] == 2), None)
        dca3 = next((r for r in results if r['symbol'] == symbol and r['trigger_level'] == 3), None)

        if dca2 and dca3:
            diff = dca2['ending_balance'] - dca3['ending_balance']
            winner = "DCA 2" if diff > 0 else "DCA 3" if diff < 0 else "TIE"
            print(f"{symbol}: DCA2=${dca2['ending_balance']:.2f} vs DCA3=${dca3['ending_balance']:.2f} -> {winner} wins by ${abs(diff):.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
