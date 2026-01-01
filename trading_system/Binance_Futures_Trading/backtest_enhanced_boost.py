#!/usr/bin/env python3
"""
Backtest Hedge + DCA + ENHANCED BOOST MODE Strategy
====================================================
Enhanced Boost Mode with:
1. At DCA 3 trigger -> Boost opposite side 1.5x
2. When boosted side hits TP: Close HALF, lock profit, add back 0.5x
3. Trailing stop activates AFTER each half-close cycle
4. Continue until losing side recovers (TP) or hits SL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG, SYMBOL_SETTINGS


class EnhancedBoostBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Get symbol-specific settings (for ETH/BTC tighter params)
        symbol_config = SYMBOL_SETTINGS.get(symbol, {})

        # Strategy params - USE SYMBOL-SPECIFIC if available
        self.tp_roi = symbol_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])
        self.sl_roi = DCA_CONFIG["stop_loss_roi"]    # 90% (same for all)
        self.dca_levels = symbol_config.get("dca_levels", DCA_CONFIG["levels"])
        self.budget_split = DCA_CONFIG["hedge_mode"]["budget_split"]  # 50%

        # DCA budget allocation percentages
        self.dca_pcts = [
            DCA_CONFIG.get("initial_entry_pct", 0.10),  # 10%
            DCA_CONFIG.get("dca1_pct", 0.15),           # 15%
            DCA_CONFIG.get("dca2_pct", 0.20),           # 20%
            DCA_CONFIG.get("dca3_pct", 0.25),           # 25%
            DCA_CONFIG.get("dca4_pct", 0.30),           # 30%
        ]

        # ENHANCED BOOST PARAMETERS - USE SYMBOL-SPECIFIC if available
        self.boost_multiplier = 1.5  # 1.5x instead of 2x
        self.boost_trigger_dca_level = symbol_config.get("boost_trigger_dca", 3)  # DCA 2 for BTC/ETH, DCA 3 for others
        self.trailing_activation_roi = 0.02  # Start trailing after 2% ROI profit
        self.trailing_distance_roi = 0.03    # Trail 3% behind peak

        # Log symbol-specific settings
        print(f"[{symbol}] TP ROI: {self.tp_roi*100:.0f}% | Boost Trigger: DCA {self.boost_trigger_dca_level}")

        # Positions
        self.long_position = None
        self.short_position = None

        # Boost mode tracking
        self.boost_mode_active = False
        self.boosted_side = None  # "LONG" or "SHORT"
        self.boost_trigger_side = None  # The side that triggered boost (at DCA 3)

        # Enhanced boost tracking - half-close cycles
        self.boost_cycle_count = 0  # Number of half-close + re-add cycles
        self.boost_locked_profit = 0  # Profit locked from half-closes
        self.trailing_active = False
        self.boosted_peak_roi = 0

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
        self.half_close_count = 0
        self.trailing_closes = 0

    def get_historical_data(self, days: int = 30, interval: str = "1h", days_ago_start: int = 0):
        """Fetch historical klines from Binance MAINNET (real data)"""
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")
        print("Using Binance MAINNET for real historical data...")

        client = BinanceClient(testnet=False, use_demo=False)

        end_time = datetime.now() - timedelta(days=days_ago_start)
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

    def open_position(self, side: str, price: float, boost_multiplier: float = 1.0) -> dict:
        """Open new position with optional boost multiplier"""
        budget = self.start_balance * self.budget_split
        initial_margin = budget * self.dca_pcts[0]

        # Apply boost multiplier
        initial_margin *= boost_multiplier

        quantity = (initial_margin * self.leverage) / price

        position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0),
            "sl_price": self.get_sl_price(price, side),
            "is_boosted": boost_multiplier > 1.0,
            "boost_multiplier": boost_multiplier
        }
        return position

    def close_position(self, position: dict, exit_price: float, exit_type: str, timestamp, close_pct: float = 1.0):
        """Close position (full or partial) and record trade"""
        price_change_pct = (exit_price - position["entry_price"]) / position["entry_price"]

        if position["side"] == "LONG":
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        # Calculate PNL for the closed portion
        closed_margin = position["margin"] * close_pct
        pnl = closed_margin * roi

        # Cap loss at margin (isolated mode)
        if pnl < -closed_margin:
            pnl = -closed_margin

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
            "quantity": position["quantity"] * close_pct,
            "margin": closed_margin,
            "dca_level": position["dca_level"],
            "pnl": pnl,
            "exit_type": exit_type,
            "balance": self.balance,
            "is_boosted": position.get("is_boosted", False),
            "close_pct": close_pct
        })

        return pnl

    def half_close_and_readd(self, position: dict, exit_price: float, timestamp):
        """
        ENHANCED BOOST: Close HALF at TP, lock profit, add back 0.5x
        Returns the modified position
        """
        # Close HALF the position
        half_pnl = self.close_position(position, exit_price, "HALF_TP", timestamp, close_pct=0.5)
        self.boost_locked_profit += half_pnl
        self.half_close_count += 1
        self.boost_cycle_count += 1

        # Reduce position by half
        position["quantity"] *= 0.5
        position["margin"] *= 0.5

        # Add back 0.5x at current price
        add_margin = (self.start_balance * self.budget_split) * self.dca_pcts[0] * 0.5
        add_qty = (add_margin * self.leverage) / exit_price

        # Update position with new average entry
        old_value = position["quantity"] * position["entry_price"]
        add_value = add_qty * exit_price
        new_qty = position["quantity"] + add_qty
        new_entry = (old_value + add_value) / new_qty

        position["quantity"] = new_qty
        position["margin"] += add_margin
        position["entry_price"] = new_entry
        position["tp_price"] = self.get_tp_price(new_entry, position["side"], 0)
        position["sl_price"] = self.get_sl_price(new_entry, position["side"])

        # Activate trailing after half-close cycle
        self.trailing_active = True
        self.boosted_peak_roi = 0

        print(f"[{timestamp}] >>> HALF CLOSE @ ${exit_price:.4f} | Locked: ${half_pnl:+.2f} | Added 0.5x back | Trailing NOW ACTIVE")

        return position

    def activate_boost_mode(self, trigger_side: str, timestamp):
        """Activate enhanced boost mode when one side hits DCA 3"""
        self.boost_mode_active = True
        self.boosted_side = "SHORT" if trigger_side == "LONG" else "LONG"
        self.boost_trigger_side = trigger_side
        self.boost_activations += 1
        self.boost_cycle_count = 0
        self.boost_locked_profit = 0
        self.boosted_peak_roi = 0
        self.trailing_active = False  # Trailing starts AFTER first half-close
        print(f"[{timestamp}] >>> ENHANCED BOOST ACTIVATED! {trigger_side} at DCA {self.boost_trigger_dca_level} -> {self.boosted_side} boosted 1.5x")
        print(f"    [BOOST LOGIC] At TP: Close HALF, lock profit, add 0.5x, then trailing starts")

    def deactivate_boost_mode(self, timestamp, reason: str):
        """Deactivate boost mode"""
        if not self.boost_mode_active:
            return

        print(f"[{timestamp}] >>> BOOST MODE ENDED - {reason}")
        print(f"    [BOOST SUMMARY] Cycles: {self.boost_cycle_count} | Locked Profit: ${self.boost_locked_profit:+.2f}")

        self.boost_mode_active = False
        self.boosted_side = None
        self.boost_trigger_side = None
        self.boost_cycle_count = 0
        self.boosted_peak_roi = 0
        self.trailing_active = False

    def check_trailing_stop(self, position: dict, current_price: float) -> tuple:
        """
        Check if trailing stop should close the boosted position
        Returns: (should_close, trailing_price)
        """
        if not position.get("is_boosted", False) or not self.trailing_active:
            return False, None

        current_roi = self.calculate_roi(position["entry_price"], current_price, position["side"])

        # Update peak ROI if current is higher
        if current_roi > self.boosted_peak_roi:
            self.boosted_peak_roi = current_roi

        # Check if trailing stop triggered (only if we have profit)
        if self.boosted_peak_roi >= self.trailing_activation_roi:
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
        """Run the backtest with enhanced boost mode"""
        print("\n" + "="*70)
        print("RUNNING HEDGE + DCA + ENHANCED BOOST MODE BACKTEST")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"TP ROI: {self.tp_roi*100:.0f}% | SL ROI: {self.sl_roi*100:.0f}%")
        print(f"ENHANCED BOOST: Trigger at DCA {self.boost_trigger_dca_level}")
        print(f"  - Initial boost: 1.5x")
        print(f"  - At TP: Close HALF, lock profit, add 0.5x back")
        print(f"  - Trailing activates AFTER first half-close")
        print(f"  - Continue until losing side recovers or SL")
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

                # Check trailing stop first (only for boosted positions AFTER half-close)
                if is_boosted and self.boost_mode_active and self.boosted_side == "LONG" and self.trailing_active:
                    should_trail_close, trail_price = self.check_trailing_stop(self.long_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.long_position, trail_price, "TRAILING", timestamp)
                        self.boost_profits += pnl
                        self.trailing_closes += 1
                        print(f"[{timestamp}] BOOST LONG TRAILING CLOSE @ ${trail_price:.4f} | P&L: ${pnl:+.2f} | Peak ROI: {self.boosted_peak_roi*100:.1f}%")
                        # Re-enter with boost size (boost still active)
                        self.long_position = self.open_position("LONG", close, boost_multiplier=self.boost_multiplier)
                        self.boosted_peak_roi = 0
                        self.trailing_active = False  # Reset, will activate after next half-close
                        continue

                # Check TP hit for boosted position - do HALF close logic
                if is_boosted and self.boost_mode_active and self.boosted_side == "LONG":
                    if high >= self.long_position["tp_price"]:
                        # ENHANCED: Half close, lock profit, add back 0.5x
                        self.long_position = self.half_close_and_readd(
                            self.long_position,
                            self.long_position["tp_price"],
                            timestamp
                        )
                        self.boost_profits += self.boost_locked_profit
                        continue

                # Check TP hit (non-boosted or full close)
                elif high >= self.long_position["tp_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["tp_price"], "TP", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] LONG TP @ ${self.long_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode(timestamp, "LONG recovered (TP)")

                    # Re-enter (boost if in boost mode and this side is boosted)
                    boost_mult = self.boost_multiplier if (self.boost_mode_active and self.boosted_side == "LONG") else 1.0
                    self.long_position = self.open_position("LONG", close, boost_multiplier=boost_mult)
                    if boost_mult > 1.0:
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
                        self.deactivate_boost_mode(timestamp, "LONG hit SL (lost)")

                    # Re-enter
                    boost_mult = self.boost_multiplier if (self.boost_mode_active and self.boosted_side == "LONG") else 1.0
                    self.long_position = self.open_position("LONG", close, boost_multiplier=boost_mult)
                    if boost_mult > 1.0:
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
                                self.short_position["boost_multiplier"] = self.boost_multiplier
                                print(f"[{timestamp}] >>> SHORT boosted: margin ${old_margin:.2f} -> ${self.short_position['margin']:.2f}")

            # Check SHORT position
            if self.short_position:
                is_boosted = self.short_position.get("is_boosted", False)

                # Check trailing stop first (only for boosted positions AFTER half-close)
                if is_boosted and self.boost_mode_active and self.boosted_side == "SHORT" and self.trailing_active:
                    should_trail_close, trail_price = self.check_trailing_stop(self.short_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.short_position, trail_price, "TRAILING", timestamp)
                        self.boost_profits += pnl
                        self.trailing_closes += 1
                        print(f"[{timestamp}] BOOST SHORT TRAILING CLOSE @ ${trail_price:.4f} | P&L: ${pnl:+.2f} | Peak ROI: {self.boosted_peak_roi*100:.1f}%")
                        # Re-enter with boost size
                        self.short_position = self.open_position("SHORT", close, boost_multiplier=self.boost_multiplier)
                        self.boosted_peak_roi = 0
                        self.trailing_active = False
                        continue

                # Check TP hit for boosted position - do HALF close logic
                if is_boosted and self.boost_mode_active and self.boosted_side == "SHORT":
                    if low <= self.short_position["tp_price"]:
                        # ENHANCED: Half close, lock profit, add back 0.5x
                        self.short_position = self.half_close_and_readd(
                            self.short_position,
                            self.short_position["tp_price"],
                            timestamp
                        )
                        self.boost_profits += self.boost_locked_profit
                        continue

                # Check TP hit (non-boosted or full close)
                elif low <= self.short_position["tp_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["tp_price"], "TP", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] SHORT TP @ ${self.short_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode(timestamp, "SHORT recovered (TP)")

                    # Re-enter (boost if in boost mode and this side is boosted)
                    boost_mult = self.boost_multiplier if (self.boost_mode_active and self.boosted_side == "SHORT") else 1.0
                    self.short_position = self.open_position("SHORT", close, boost_multiplier=boost_mult)
                    if boost_mult > 1.0:
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
                        self.deactivate_boost_mode(timestamp, "SHORT hit SL (lost)")

                    # Re-enter
                    boost_mult = self.boost_multiplier if (self.boost_mode_active and self.boosted_side == "SHORT") else 1.0
                    self.short_position = self.open_position("SHORT", close, boost_multiplier=boost_mult)
                    if boost_mult > 1.0:
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
                                self.long_position["boost_multiplier"] = self.boost_multiplier
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

        # Calculate best/worst trades
        trade_pnls = [t["pnl"] for t in self.trades] if self.trades else [0]
        best_trade = max(trade_pnls)
        worst_trade = min(trade_pnls)

        # Calculate total win $ and total loss $
        total_win_dollars = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        total_loss_dollars = sum(t["pnl"] for t in self.trades if t["pnl"] < 0)

        # Max DD in dollars (from percentage)
        max_dd_dollars = self.start_balance * (self.max_drawdown / 100)

        return {
            "balance": self.balance,
            "return_pct": (self.balance - self.start_balance) / self.start_balance * 100,
            "win_rate": (self.total_wins / len(self.trades) * 100) if self.trades else 0,
            "max_drawdown": self.max_drawdown,
            "max_dd_dollars": max_dd_dollars,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "total_win_dollars": total_win_dollars,
            "total_loss_dollars": total_loss_dollars,
            "total_trades": len(self.trades),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "boost_activations": self.boost_activations,
            "half_closes": self.half_close_count,
            "trailing_closes": self.trailing_closes,
            "liquidated": self.balance <= 0
        }

    def print_results(self, df: pd.DataFrame, unrealized_pnl: float):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - ENHANCED BOOST MODE")
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

        # Enhanced Boost Mode Analysis
        print(f"\n>>> ENHANCED BOOST MODE Analysis:")
        print(f"  Boost Activations:   {self.boost_activations}")
        print(f"  Half-Close Cycles:   {self.half_close_count}")
        print(f"  Trailing Closes:     {self.trailing_closes}")
        print(f"  Total Boost P&L:     ${self.boost_profits:+.2f}")

        # Boosted trades analysis
        boosted_trades = [t for t in self.trades if t.get("is_boosted", False)]
        if boosted_trades:
            avg_boost_pnl = sum(t["pnl"] for t in boosted_trades) / len(boosted_trades)
            print(f"\nBoosted Trades Analysis:")
            print(f"  Boosted Trades:   {len(boosted_trades)}")
            print(f"  Avg P&L (Boost):  ${avg_boost_pnl:+.2f}")
            print(f"  Total Boost P&L:  ${sum(t['pnl'] for t in boosted_trades):+.2f}")

        # Half-close trades
        half_close_trades = [t for t in self.trades if t.get("exit_type") == "HALF_TP"]
        if half_close_trades:
            print(f"\nHalf-Close Analysis:")
            print(f"  Half-Close Trades:  {len(half_close_trades)}")
            print(f"  Locked Profit:      ${sum(t['pnl'] for t in half_close_trades):+.2f}")

        print("\n" + "="*70)


def run_90_day_test():
    """Test Enhanced Boost over 90 days with all 4 pairs"""
    print("="*80)
    print("ENHANCED BOOST MODE - 90 DAY BACKTEST")
    print("="*80)
    print("Strategy: At DCA 3 -> Boost opposite 1.5x")
    print("          At TP: Close HALF, lock profit, add 0.5x back")
    print("          Trailing starts AFTER each half-close")
    print("          Continue until losing side recovers or SL")
    print("="*80)

    # Test DOT, BTC, AVAX and SOL
    symbols = ["BTCUSDT", "AVAXUSDT", "DOTUSDT", "SOLUSDT", "BNBUSDT"]

    all_results = []
    total_starting = 0
    total_ending = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_half_closes = 0
    liquidations = 0

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        backtester = EnhancedBoostBacktester(symbol, start_balance=100.0)
        df = backtester.get_historical_data(days=90, interval="1h")

        if df is not None and len(df) > 0:
            result = backtester.run_backtest(df)
            result["symbol"] = symbol
            result["price_change"] = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            all_results.append(result)

            total_starting += 100
            total_ending += result["balance"]
            total_trades += result["total_trades"]
            total_wins += result["wins"]
            total_losses += result["losses"]
            total_half_closes += result["half_closes"]
            if result["liquidated"]:
                liquidations += 1
        else:
            print(f"ERROR: No data for {symbol}")

    # Print combined summary
    print("\n" + "="*120)
    print("COMPREHENSIVE ANALYTICS - 90 DAY ENHANCED BOOST BACKTEST")
    print("="*120)

    # Detailed per-symbol analysis
    for r in all_results:
        wins = r["wins"]
        losses = r["losses"]
        total = r["total_trades"]
        days = 90
        trades_per_day = total / days
        wins_per_day = wins / days
        losses_per_day = losses / days

        # Calculate total win $ and total loss $
        total_win_dollars = r.get("total_win_dollars", 0)
        total_loss_dollars = r.get("total_loss_dollars", 0)
        avg_win = total_win_dollars / wins if wins > 0 else 0
        avg_loss = total_loss_dollars / losses if losses > 0 else 0

        # Profit factor
        profit_factor = abs(total_win_dollars / total_loss_dollars) if total_loss_dollars != 0 else float('inf')

        print(f"\n{'='*60}")
        print(f"  {r['symbol']} - DETAILED ANALYTICS")
        print(f"{'='*60}")
        print(f"  MARKET PERFORMANCE:")
        print(f"    Price Change:        {r['price_change']:+.1f}%")
        print(f"")
        print(f"  ACCOUNT PERFORMANCE:")
        print(f"    Starting Balance:    $100.00")
        print(f"    Ending Balance:      ${r['balance']:.2f}")
        print(f"    Net Profit/Loss:     ${r['balance'] - 100:+.2f}")
        print(f"    Return:              {r['return_pct']:+.1f}%")
        print(f"    Max Drawdown:        ${r.get('max_dd_dollars', 0):.2f} ({r['max_drawdown']:.1f}%)")
        print(f"")
        print(f"  TRADE STATISTICS:")
        print(f"    Total Trades:        {total}")
        print(f"    Winning Trades:      {wins} ({wins/total*100:.1f}%)")
        print(f"    Losing Trades:       {losses} ({losses/total*100:.1f}%)")
        print(f"    Win Rate:            {r['win_rate']:.1f}%")
        print(f"")
        print(f"  DAILY AVERAGES (90 days):")
        print(f"    Trades/Day:          {trades_per_day:.1f}")
        print(f"    Wins/Day:            {wins_per_day:.1f}")
        print(f"    Losses/Day:          {losses_per_day:.2f}")
        print(f"    Daily Return:        {r['return_pct']/90:.2f}%")
        print(f"")
        print(f"  PROFIT/LOSS BREAKDOWN:")
        print(f"    Total Won:           ${total_win_dollars:+.2f}")
        print(f"    Total Lost:          ${total_loss_dollars:+.2f}")
        print(f"    Avg Win:             ${avg_win:+.2f}")
        print(f"    Avg Loss:            ${avg_loss:+.2f}")
        print(f"    Best Trade:          ${r.get('best_trade', 0):+.2f}")
        print(f"    Worst Trade:         ${r.get('worst_trade', 0):+.2f}")
        print(f"    Profit Factor:       {profit_factor:.2f}x")
        print(f"")
        print(f"  BOOST MODE STATS:")
        print(f"    Boost Activations:   {r.get('boost_activations', 0)}")
        print(f"    Half-Close Cycles:   {r.get('half_closes', 0)}")
        status = "LIQUIDATED!" if r["liquidated"] else "SURVIVED"
        print(f"    Status:              {status}")

    # Summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"\n{'Symbol':<10} {'Market':<8} {'Return':<8} {'Trades':<7} {'Wins':<6} {'Losses':<7} {'WinRate':<8} {'Trades/Day':<11} {'MaxDD':<9} {'Best':<8} {'Worst':<10}")
    print("-"*120)
    for r in all_results:
        trades_day = r['total_trades'] / 90
        status = " *LIQD*" if r["liquidated"] else ""
        print(f"{r['symbol']:<10} {r['price_change']:+.1f}%{'':<2} {r['return_pct']:+.1f}%{'':<2} {r['total_trades']:<7} {r['wins']:<6} {r['losses']:<7} {r['win_rate']:.1f}%{'':<3} {trades_day:.1f}{'':<6} ${r.get('max_dd_dollars',0):<7.2f} ${r.get('best_trade',0):+.2f}{'':<2} ${r.get('worst_trade',0):+.2f}{status}")

    # Grand totals
    total_profit = total_ending - total_starting
    avg_return = total_profit / len(all_results)
    total_trades_per_day = total_trades / 90

    print(f"\n{'='*120}")
    print(f"GRAND TOTAL - PORTFOLIO PERFORMANCE")
    print(f"{'='*120}")
    print(f"  CAPITAL:")
    print(f"    Starting Capital:    ${total_starting:.2f} ({len(all_results)} symbols × $100)")
    print(f"    Ending Capital:      ${total_ending:.2f}")
    print(f"    Net Profit:          ${total_profit:+.2f}")
    print(f"    Portfolio Return:    {total_profit/total_starting*100:+.1f}%")
    print(f"    Avg Return/Symbol:   {total_profit/total_starting*100/len(all_results):.1f}%")
    print(f"")
    print(f"  TRADES:")
    print(f"    Total Trades:        {total_trades}")
    print(f"    Total Wins:          {total_wins}")
    print(f"    Total Losses:        {total_losses}")
    print(f"    Overall Win Rate:    {total_wins/total_trades*100:.1f}%")
    print(f"    Trades/Day (all):    {total_trades_per_day:.1f}")
    print(f"")
    print(f"  BOOST MODE:")
    print(f"    Half-Close Cycles:   {total_half_closes}")
    print(f"    Liquidations:        {liquidations}/{len(all_results)}")
    print(f"{'='*120}")


if __name__ == "__main__":
    run_90_day_test()
