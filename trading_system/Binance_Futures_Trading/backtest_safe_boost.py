#!/usr/bin/env python3
"""
Backtest Hedge + DCA + SAFE BOOST MODE Strategy
================================================
Safe Boost Mode with:
1. Partial Boost (1.5x instead of 2x)
2. No DCA on boosted side during boost
3. Trailing stop on boosted side to lock profits
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG


class SafeBoostBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Strategy params from config
        self.tp_roi = DCA_CONFIG["take_profit_roi"]  # 8%
        self.sl_roi = DCA_CONFIG["stop_loss_roi"]    # 90%
        self.dca_levels = DCA_CONFIG["levels"]
        self.budget_split = DCA_CONFIG["hedge_mode"]["budget_split"]  # 50%

        # DCA budget allocation percentages
        self.dca_pcts = [
            DCA_CONFIG.get("initial_entry_pct", 0.10),  # 10%
            DCA_CONFIG.get("dca1_pct", 0.15),           # 15%
            DCA_CONFIG.get("dca2_pct", 0.20),           # 20%
            DCA_CONFIG.get("dca3_pct", 0.25),           # 25%
            DCA_CONFIG.get("dca4_pct", 0.30),           # 30%
        ]

        # SAFE BOOST PARAMETERS
        self.boost_multiplier = 1.5  # 1.5x instead of 2x
        self.boost_trigger_dca_level = 3  # Activate boost at DCA 3 (not DCA 2)
        self.trailing_activation_roi = 0.02  # Start trailing after 2% ROI profit
        self.trailing_distance_roi = 0.03    # Trail 3% behind peak

        # Positions
        self.long_position = None
        self.short_position = None

        # Boost mode tracking
        self.boost_mode_active = False
        self.boosted_side = None  # "LONG" or "SHORT"
        self.boost_trigger_side = None  # The side that triggered boost (at DCA 2)

        # Trailing stop tracking for boosted position
        self.boosted_peak_roi = 0  # Track peak ROI for trailing
        self.trailing_active = False

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance

        # Boost mode stats
        self.boost_activations = 0
        self.boost_profits = 0
        self.trailing_closes = 0

    def get_historical_data(self, days: int = 30, interval: str = "1h"):
        """Fetch historical klines from Binance MAINNET (real data)"""
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")
        print("Using Binance MAINNET for real historical data...")

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
            print("No data returned!")
            return None

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        print(f"Price change over period: {price_change:+.2f}%")

        return df

    def calculate_roi(self, entry_price: float, current_price: float, side: str) -> float:
        """Calculate ROI based on position side"""
        if side == "LONG":
            price_pct = (current_price - entry_price) / entry_price
        else:
            price_pct = (entry_price - current_price) / entry_price
        return price_pct * self.leverage

    def get_dca_level(self, position: dict) -> int:
        """Get current DCA level from position"""
        return position.get("dca_level", 0)

    def get_tp_price(self, entry_price: float, side: str, dca_level: int) -> float:
        """Calculate TP price based on DCA level"""
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
        """Calculate SL price (always 90% ROI)"""
        sl_pct = self.sl_roi / self.leverage

        if side == "LONG":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def check_dca_trigger(self, position: dict, current_price: float) -> bool:
        """Check if DCA should trigger"""
        dca_level = self.get_dca_level(position)
        if dca_level >= len(self.dca_levels):
            return False

        trigger_roi = self.dca_levels[dca_level]["trigger_roi"]
        current_roi = self.calculate_roi(position["entry_price"], current_price, position["side"])

        return current_roi <= trigger_roi

    def execute_dca(self, position: dict, current_price: float) -> dict:
        """Execute DCA - add to position"""
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
        """Open new position"""
        budget = self.start_balance * self.budget_split
        initial_margin = budget * self.dca_pcts[0]

        # Apply 1.5x boost if in boost mode
        if boost:
            initial_margin *= self.boost_multiplier

        quantity = (initial_margin * self.leverage) / price

        position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0),
            "sl_price": self.get_sl_price(price, side),
            "is_boosted": boost
        }
        return position

    def close_position(self, position: dict, exit_price: float, exit_type: str, timestamp):
        """Close position and record trade"""
        price_change_pct = (exit_price - position["entry_price"]) / position["entry_price"]

        if position["side"] == "LONG":
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        pnl = position["margin"] * roi

        # Cap loss at margin (isolated mode)
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

        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.trades.append({
            "timestamp": timestamp,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "quantity": position["quantity"],
            "margin": position["margin"],
            "dca_level": position["dca_level"],
            "pnl": pnl,
            "exit_type": exit_type,
            "balance": self.balance,
            "is_boosted": position.get("is_boosted", False)
        })

        return pnl

    def activate_boost_mode(self, trigger_side: str, timestamp):
        """Activate safe boost mode when one side hits DCA 3"""
        self.boost_mode_active = True
        self.boosted_side = "SHORT" if trigger_side == "LONG" else "LONG"
        self.boost_trigger_side = trigger_side
        self.boost_activations += 1
        self.boosted_peak_roi = 0
        self.trailing_active = False
        print(f"[{timestamp}] >>> SAFE BOOST ACTIVATED! {trigger_side} at DCA {self.boost_trigger_dca_level} -> {self.boosted_side} boosted 1.5x (no DCA, trailing enabled)")

    def deactivate_boost_mode(self, timestamp, reason: str):
        """Deactivate boost mode"""
        if not self.boost_mode_active:
            return

        self.boost_mode_active = False
        self.boosted_side = None
        self.boost_trigger_side = None
        self.boosted_peak_roi = 0
        self.trailing_active = False
        print(f"[{timestamp}] >>> BOOST MODE DEACTIVATED - {reason}")

    def check_trailing_stop(self, position: dict, current_price: float) -> tuple:
        """
        Check if trailing stop should close the boosted position
        Returns: (should_close, trailing_price)
        """
        if not position.get("is_boosted", False):
            return False, None

        current_roi = self.calculate_roi(position["entry_price"], current_price, position["side"])

        # Update peak ROI if current is higher
        if current_roi > self.boosted_peak_roi:
            self.boosted_peak_roi = current_roi

        # Activate trailing once we hit activation threshold
        if not self.trailing_active and current_roi >= self.trailing_activation_roi:
            self.trailing_active = True
            print(f"    [TRAILING] Activated at {current_roi*100:.1f}% ROI")

        # Check if trailing stop triggered
        if self.trailing_active:
            trailing_trigger_roi = self.boosted_peak_roi - self.trailing_distance_roi
            if current_roi <= trailing_trigger_roi and trailing_trigger_roi > 0:
                # Calculate trailing exit price
                if position["side"] == "LONG":
                    trailing_price = position["entry_price"] * (1 + trailing_trigger_roi / self.leverage)
                else:
                    trailing_price = position["entry_price"] * (1 - trailing_trigger_roi / self.leverage)
                return True, trailing_price

        return False, None

    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest with safe boost mode"""
        print("\n" + "="*70)
        print("RUNNING HEDGE + DCA + SAFE BOOST MODE BACKTEST")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"TP ROI: {self.tp_roi*100:.0f}% | SL ROI: {self.sl_roi*100:.0f}%")
        print(f"SAFE BOOST: Trigger at DCA {self.boost_trigger_dca_level}, 1.5x size, NO DCA on boosted side, Trailing stop")
        print(f"  - Trailing activates at: {self.trailing_activation_roi*100:.0f}% ROI")
        print(f"  - Trailing distance: {self.trailing_distance_roi*100:.0f}% ROI")
        print("="*70)

        # Open initial positions
        first_price = df['close'].iloc[0]
        self.long_position = self.open_position("LONG", first_price)
        self.short_position = self.open_position("SHORT", first_price)

        print(f"\nOpened LONG @ ${first_price:.4f}")
        print(f"Opened SHORT @ ${first_price:.4f}")

        # Iterate through candles
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if self.balance <= 0:
                print(f"[{timestamp}] LIQUIDATED - Balance depleted!")
                break

            high = row['high']
            low = row['low']
            close = row['close']

            # Check LONG position
            if self.long_position:
                is_boosted = self.long_position.get("is_boosted", False)

                # Check trailing stop first (only for boosted positions)
                if is_boosted and self.boost_mode_active and self.boosted_side == "LONG":
                    should_trail_close, trail_price = self.check_trailing_stop(self.long_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.long_position, trail_price, "TRAILING", timestamp)
                        self.boost_profits += pnl
                        self.trailing_closes += 1
                        print(f"[{timestamp}] BOOST LONG TRAILING CLOSE @ ${trail_price:.4f} | P&L: ${pnl:+.2f} | Peak ROI: {self.boosted_peak_roi*100:.1f}%")
                        # Re-enter with normal size (boost still active until trigger side resolves)
                        boost = self.boost_mode_active and self.boosted_side == "LONG"
                        self.long_position = self.open_position("LONG", close, boost=boost)
                        self.boosted_peak_roi = 0
                        self.trailing_active = False
                        continue

                # Check TP hit
                if high >= self.long_position["tp_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["tp_price"], "TP", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] LONG TP @ ${self.long_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode(timestamp, "LONG recovered (TP)")

                    # Re-enter (boost if in boost mode and this side is boosted)
                    boost = self.boost_mode_active and self.boosted_side == "LONG"
                    self.long_position = self.open_position("LONG", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                # Check SL hit
                elif low <= self.long_position["sl_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["sl_price"], "SL", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] LONG SL @ ${self.long_position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode(timestamp, "LONG hit SL")

                    # Re-enter
                    boost = self.boost_mode_active and self.boosted_side == "LONG"
                    self.long_position = self.open_position("LONG", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                # Check DCA - BUT NOT if this side is boosted!
                elif self.check_dca_trigger(self.long_position, close):
                    # Skip DCA if this side is boosted
                    if self.boost_mode_active and self.boosted_side == "LONG":
                        pass  # No DCA on boosted side!
                    else:
                        old_level = self.long_position["dca_level"]
                        old_entry = self.long_position["entry_price"]
                        self.long_position = self.execute_dca(self.long_position, close)
                        new_level = self.long_position["dca_level"]
                        print(f"[{timestamp}] LONG DCA {new_level} @ ${close:.4f} | Avg: ${old_entry:.4f} -> ${self.long_position['entry_price']:.4f}")

                        # Check if DCA 3 triggered - activate boost mode
                        if new_level == self.boost_trigger_dca_level and not self.boost_mode_active:
                            self.activate_boost_mode("LONG", timestamp)
                            # Boost the SHORT position (1.5x)
                            if self.short_position:
                                old_margin = self.short_position["margin"]
                                self.short_position["quantity"] *= self.boost_multiplier
                                self.short_position["margin"] *= self.boost_multiplier
                                self.short_position["is_boosted"] = True
                                print(f"[{timestamp}] >>> SHORT boosted: margin ${old_margin:.2f} -> ${self.short_position['margin']:.2f}")

            # Check SHORT position
            if self.short_position:
                is_boosted = self.short_position.get("is_boosted", False)

                # Check trailing stop first (only for boosted positions)
                if is_boosted and self.boost_mode_active and self.boosted_side == "SHORT":
                    should_trail_close, trail_price = self.check_trailing_stop(self.short_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.short_position, trail_price, "TRAILING", timestamp)
                        self.boost_profits += pnl
                        self.trailing_closes += 1
                        print(f"[{timestamp}] BOOST SHORT TRAILING CLOSE @ ${trail_price:.4f} | P&L: ${pnl:+.2f} | Peak ROI: {self.boosted_peak_roi*100:.1f}%")
                        # Re-enter with boost size
                        boost = self.boost_mode_active and self.boosted_side == "SHORT"
                        self.short_position = self.open_position("SHORT", close, boost=boost)
                        self.boosted_peak_roi = 0
                        self.trailing_active = False
                        continue

                # Check TP hit
                if low <= self.short_position["tp_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["tp_price"], "TP", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] SHORT TP @ ${self.short_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode(timestamp, "SHORT recovered (TP)")

                    # Re-enter (boost if in boost mode and this side is boosted)
                    boost = self.boost_mode_active and self.boosted_side == "SHORT"
                    self.short_position = self.open_position("SHORT", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                # Check SL hit
                elif high >= self.short_position["sl_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["sl_price"], "SL", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] SHORT SL @ ${self.short_position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode(timestamp, "SHORT hit SL")

                    # Re-enter
                    boost = self.boost_mode_active and self.boosted_side == "SHORT"
                    self.short_position = self.open_position("SHORT", close, boost=boost)
                    if boost:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                # Check DCA - BUT NOT if this side is boosted!
                elif self.check_dca_trigger(self.short_position, close):
                    # Skip DCA if this side is boosted
                    if self.boost_mode_active and self.boosted_side == "SHORT":
                        pass  # No DCA on boosted side!
                    else:
                        old_level = self.short_position["dca_level"]
                        old_entry = self.short_position["entry_price"]
                        self.short_position = self.execute_dca(self.short_position, close)
                        new_level = self.short_position["dca_level"]
                        print(f"[{timestamp}] SHORT DCA {new_level} @ ${close:.4f} | Avg: ${old_entry:.4f} -> ${self.short_position['entry_price']:.4f}")

                        # Check if DCA 3 triggered - activate boost mode
                        if new_level == self.boost_trigger_dca_level and not self.boost_mode_active:
                            self.activate_boost_mode("SHORT", timestamp)
                            # Boost the LONG position (1.5x)
                            if self.long_position:
                                old_margin = self.long_position["margin"]
                                self.long_position["quantity"] *= self.boost_multiplier
                                self.long_position["margin"] *= self.boost_multiplier
                                self.long_position["is_boosted"] = True
                                print(f"[{timestamp}] >>> LONG boosted: margin ${old_margin:.2f} -> ${self.long_position['margin']:.2f}")

        # Calculate final unrealized PNL
        final_price = df['close'].iloc[-1]
        unrealized_long = 0
        unrealized_short = 0

        if self.long_position:
            price_change = (final_price - self.long_position["entry_price"]) / self.long_position["entry_price"]
            unrealized_long = self.long_position["margin"] * price_change * self.leverage
        if self.short_position:
            price_change = (self.short_position["entry_price"] - final_price) / self.short_position["entry_price"]
            unrealized_short = self.short_position["margin"] * price_change * self.leverage

        total_unrealized = unrealized_long + unrealized_short

        self.print_results(df, total_unrealized)

    def print_results(self, df: pd.DataFrame, unrealized_pnl: float):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - SAFE BOOST MODE")
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
        print(f"  Realized P&L:     ${self.total_pnl:+.2f}")
        print(f"  Unrealized P&L:   ${unrealized_pnl:+.2f}")
        print(f"  Total Return:     {((self.balance - self.start_balance) / self.start_balance * 100):+.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Wins:             {self.total_wins}")
        print(f"  Losses:           {self.total_losses}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Max Drawdown:     {self.max_drawdown:.2f}%")

        # Safe Boost Mode Analysis
        print(f"\n>>> SAFE BOOST MODE Analysis:")
        print(f"  Boost Activations:   {self.boost_activations}")
        print(f"  Boost P&L:           ${self.boost_profits:+.2f}")
        print(f"  Trailing Closes:     {self.trailing_closes}")

        # Boosted trades analysis
        boosted_trades = [t for t in self.trades if t.get("is_boosted", False)]
        if boosted_trades:
            avg_boost_pnl = sum(t["pnl"] for t in boosted_trades) / len(boosted_trades)
            print(f"\nBoosted Trades Analysis:")
            print(f"  Boosted Trades:   {len(boosted_trades)}")
            print(f"  Avg P&L (Boost):  ${avg_boost_pnl:+.2f}")
            print(f"  Total Boost P&L:  ${sum(t['pnl'] for t in boosted_trades):+.2f}")

        # DCA Analysis
        dca_trades = [t for t in self.trades if t["dca_level"] > 0]
        if dca_trades:
            avg_dca_pnl = sum(t["pnl"] for t in dca_trades) / len(dca_trades)
            print(f"\nDCA Analysis:")
            print(f"  Trades with DCA:  {len(dca_trades)}")
            print(f"  Avg P&L (DCA):    ${avg_dca_pnl:+.2f}")

        print("\n" + "="*70)


def main():
    print("="*70)
    print("HEDGE + DCA + SAFE BOOST MODE STRATEGY BACKTEST")
    print("="*70)
    print("Safe Boost Features:")
    print("  1. Partial Boost: 1.5x (not 2x)")
    print("  2. No DCA on boosted side during boost")
    print("  3. Trailing stop to lock in profits")
    print("="*70)

    symbol = "DOTUSDT"

    backtester = SafeBoostBacktester(symbol, start_balance=100.0)

    # Get REAL data from Binance MAINNET
    df = backtester.get_historical_data(days=30, interval="1h")

    if df is not None and len(df) > 0:
        backtester.run_backtest(df)
    else:
        print("ERROR: Failed to get real data from Binance!")
        print("Make sure VPN is active and try again.")


if __name__ == "__main__":
    main()
