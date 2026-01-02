#!/usr/bin/env python3
"""
Backtest Hedge + NO DCA + BOOST MODE Strategy v6
=================================================
NO DCA TEST - Only initial entry (DCA 0), no averaging down
DCA 0 made +$5,205 profit, DCA levels caused losses

Strategy:
- Initial entry only (NO DCA)
- Boost mode at -20% ROI (boost opposite side 1.5x)
- Strong trend mode blocks losing side
- Stop for day after SL hit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import csv
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
        self.budget_split = DCA_CONFIG["hedge_mode"]["budget_split"]  # 50%

        # NO DCA - Empty list means no DCA levels at all
        self.dca_levels = []  # NO DCA - only initial entry

        # Budget allocation - 10% for initial entry (conservative sizing)
        self.dca_pcts = [
            0.10,  # 10% for initial entry (conservative - avoid liquidation)
        ]

        # BOOST MODE - Triggers at -20% ROI (even without DCA)
        # When position hits -20% ROI, boost opposite side 1.5x
        self.boost_multiplier = 1.5  # 1.5x boost
        self.boost_trigger_roi = -0.20  # Trigger boost at -20% ROI (no DCA needed)
        self.boost_trigger_dca_level = 999  # Disabled - we use ROI trigger instead
        self.boost_tp_multiplier = 1.5  # Increase TP by 50% during boost mode
        self.trailing_activation_roi = 0.02  # Start trailing after 2% ROI profit
        self.trailing_distance_roi = 0.03    # Trail 3% behind peak

        # STOP FOR DAY AFTER SL - restart next day (CONFIGURABLE)
        self.stop_for_day_enabled = True  # Set to False to disable stop for day
        self.stopped_for_day = False  # True when SL hit, waiting for next day
        self.sl_hit_date = None  # Date when SL was hit

        # Log symbol-specific settings
        print(f"[{symbol}] TP ROI: {self.tp_roi*100:.0f}% | Boost Trigger: DCA {self.boost_trigger_dca_level} | Boost TP: +{(self.boost_tp_multiplier-1)*100:.0f}%")

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

        # SL stats
        self.sl_hits_long = 0
        self.sl_hits_short = 0
        self.days_stopped = 0  # Count of days stopped due to SL

        # Liquidation stats
        self.liquidations_long = 0
        self.liquidations_short = 0

        # Signal cooldown after SL (bars to wait before re-entering)
        self.cooldown_bars = 5  # Wait 5 candles after SL before re-entry
        self.cooldown_remaining = 0  # Current cooldown counter

        # STRONG TREND MODE - ADX-based trend detection
        self.strong_trend_mode = False
        self.trend_direction = None  # "UP" or "DOWN"
        self.adx_threshold = 40  # ADX > 40 = strong trend (lowered from 50 for more activations)
        self.adx_period = 14  # Standard ADX period
        self.current_adx = 0
        self.current_plus_di = 0
        self.current_minus_di = 0

        # Scale-in on consecutive TPs
        self.consecutive_tp_count = {"LONG": 0, "SHORT": 0}
        self.scale_in_after_tps = 2  # Scale in after 2 consecutive TPs
        self.scale_in_multiplier = 1.5  # 1.5x position on scale-in
        self.scaled_in_side = None  # Track which side is scaled in

        # Strong trend stats
        self.strong_trend_activations = 0
        self.scale_in_count = 0

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """
        Calculate ADX (Average Directional Index) for trend strength detection
        Returns: (adx, plus_di, minus_di) for each row
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
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()

        return adx, plus_di, minus_di

    def check_strong_trend(self, idx: int, adx_series: pd.Series, plus_di_series: pd.Series, minus_di_series: pd.Series) -> tuple:
        """
        Check if we're in a strong trend based on ADX
        Returns: (is_strong_trend, trend_direction)
        """
        if idx < self.adx_period * 2:
            return False, None

        adx = adx_series.iloc[idx]
        plus_di = plus_di_series.iloc[idx]
        minus_di = minus_di_series.iloc[idx]

        self.current_adx = adx
        self.current_plus_di = plus_di
        self.current_minus_di = minus_di

        if pd.isna(adx) or adx < self.adx_threshold:
            return False, None

        # Determine trend direction
        if plus_di > minus_di:
            return True, "UP"
        else:
            return True, "DOWN"

    def activate_strong_trend_mode(self, direction: str, timestamp):
        """Activate strong trend mode"""
        if self.strong_trend_mode:
            return  # Already active

        self.strong_trend_mode = True
        self.trend_direction = direction
        self.strong_trend_activations += 1
        print(f"[{timestamp}] >>> STRONG TREND MODE ACTIVATED! Direction: {direction} | ADX: {self.current_adx:.1f}")
        print(f"    [TREND LOGIC] Winner: 2x, half-close, trail | Loser: ALL DCA BLOCKED")

    def deactivate_strong_trend_mode(self, timestamp, reason: str):
        """Deactivate strong trend mode"""
        if not self.strong_trend_mode:
            return

        print(f"[{timestamp}] >>> STRONG TREND MODE ENDED - {reason}")
        self.strong_trend_mode = False
        self.trend_direction = None

    def apply_scale_in(self, position: dict, timestamp) -> dict:
        """Scale in position after consecutive TPs (1.5x)"""
        if position is None:
            return position

        old_margin = position["margin"]
        position["quantity"] *= self.scale_in_multiplier
        position["margin"] *= self.scale_in_multiplier
        position["is_scaled_in"] = True
        self.scaled_in_side = position["side"]
        self.scale_in_count += 1

        print(f"[{timestamp}] >>> SCALE-IN {position['side']}: margin ${old_margin:.2f} -> ${position['margin']:.2f} (after {self.scale_in_after_tps} consecutive TPs)")

        return position

    def get_historical_data(self, days: int = 30, interval: str = "1h", days_ago_start: int = 0):
        """Fetch historical klines from Binance MAINNET with pagination for large date ranges"""
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")
        print("Using Binance MAINNET for real historical data...")

        client = BinanceClient(testnet=False, use_demo=False)

        end_time = datetime.now() - timedelta(days=days_ago_start)
        start_time = end_time - timedelta(days=days)

        # Paginate to get all data (1000 candles per request max)
        all_data = []
        current_start = start_time

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

            # Move start to after last candle
            last_time = df_chunk.index[-1]
            current_start = last_time + timedelta(hours=1)

        if not all_data:
            print("No data returned!")
            return None

        # Combine all chunks
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        print(f"Price change over period: {price_change:+.2f}%")

        return df

    def stop_for_day(self, timestamp, exit_price: float, sl_side: str):
        """
        After SL hit on one side:
        - Keep the WINNING side open with trailing stop
        - Only fully stop when winning side closes via trailing
        - Then restart both sides next candle
        """
        self.days_stopped += 1

        # Determine winning side (opposite of SL side)
        winning_side = "SHORT" if sl_side == "LONG" else "LONG"
        winning_pos = self.short_position if sl_side == "LONG" else self.long_position

        # If winning side exists and is in profit, activate trailing and let it run
        if winning_pos:
            current_roi = self.calculate_roi(winning_pos["entry_price"], exit_price, winning_side)

            if current_roi > 0:
                # Winning side is profitable - activate trailing stop on it
                self.trailing_active = True
                self.boosted_peak_roi = current_roi
                winning_pos["trailing_for_close"] = True  # Mark for trailing close
                print(f"[{timestamp}] >>> {winning_side} PROFITABLE ({current_roi*100:.1f}% ROI) - Trailing activated")
                print(f"[{timestamp}] >>> Will close {winning_side} via trailing, then restart both sides")
                return  # Don't stop for day yet - let trailing run
            else:
                # Winning side not profitable - close it too
                pnl = self.close_position(winning_pos, exit_price, "STOP_DAY", timestamp)
                print(f"[{timestamp}] Closing {winning_side} for day stop @ ${exit_price:.4f} | P&L: ${pnl:+.2f}")
                if winning_side == "LONG":
                    self.long_position = None
                else:
                    self.short_position = None

        # Deactivate boost mode
        if self.boost_mode_active:
            self.deactivate_boost_mode(timestamp, "SL hit - stopping")

        # Now stop for day
        self.stopped_for_day = True
        self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        print(f"[{timestamp}] >>> STOPPED FOR THE DAY - Will restart tomorrow")

    def check_new_day(self, timestamp) -> bool:
        """Check if it's a new day and we should restart trading"""
        if not self.stopped_for_day:
            return False

        current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if current_date > self.sl_hit_date:
            self.stopped_for_day = False
            self.sl_hit_date = None
            return True
        return False

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

    def get_tp_price(self, entry_price: float, side: str, dca_level: int, is_boosted: bool = False) -> float:
        """Calculate TP price based on DCA level. INCREASE TP during boost mode."""
        if dca_level > 0 and dca_level <= len(self.dca_levels):
            tp_roi = self.dca_levels[dca_level - 1].get("tp_roi", self.tp_roi)
        else:
            tp_roi = self.tp_roi

        # INCREASE TP by 50% during boost mode
        if is_boosted and self.boost_mode_active:
            tp_roi = tp_roi * self.boost_tp_multiplier

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
        # Recalculate liquidation price after DCA
        position["liq_price"] = self.calculate_liquidation_price(new_entry, position["side"], new_margin, new_qty)

        return position

    def calculate_liquidation_price(self, entry_price: float, side: str, margin: float, quantity: float) -> float:
        """
        Calculate real liquidation price for isolated margin position.

        Binance Futures Liquidation Formula (Isolated):
        For LONG: Liq = Entry - (Margin - Maintenance Margin) / Quantity
        For SHORT: Liq = Entry + (Margin - Maintenance Margin) / Quantity

        Maintenance Margin = Position Value * MMR
        Position Value = Entry * Quantity
        MMR (Maintenance Margin Rate) = 0.4% for most pairs at <$50k position
        """
        mmr = 0.004  # 0.4% maintenance margin rate (Binance standard for most tiers)
        position_value = entry_price * quantity
        maintenance_margin = position_value * mmr

        # Liquidation happens when remaining margin = maintenance margin
        margin_buffer = margin - maintenance_margin

        if side == "LONG":
            # LONG liquidates when price drops enough to exhaust margin buffer
            liq_price = entry_price - (margin_buffer / quantity)
        else:
            # SHORT liquidates when price rises enough to exhaust margin buffer
            liq_price = entry_price + (margin_buffer / quantity)

        return max(0, liq_price)  # Price can't be negative

    def check_liquidation(self, position: dict, low: float, high: float) -> bool:
        """
        Check if position got liquidated based on price extremes.
        Returns True if liquidated.
        """
        if position is None:
            return False

        liq_price = position.get("liq_price", 0)
        if liq_price <= 0:
            return False

        if position["side"] == "LONG":
            # LONG liquidates if low hits or passes liquidation price
            return low <= liq_price
        else:
            # SHORT liquidates if high hits or passes liquidation price
            return high >= liq_price

    def open_position(self, side: str, price: float, boost_multiplier: float = 1.0) -> dict:
        """Open new position with optional boost multiplier"""
        budget = self.start_balance * self.budget_split
        initial_margin = budget * self.dca_pcts[0]

        # Apply boost multiplier
        initial_margin *= boost_multiplier

        quantity = (initial_margin * self.leverage) / price
        is_boosted = boost_multiplier > 1.0

        # Calculate real liquidation price
        liq_price = self.calculate_liquidation_price(price, side, initial_margin, quantity)

        position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0, is_boosted=is_boosted),  # Use boosted TP
            "sl_price": self.get_sl_price(price, side),
            "liq_price": liq_price,  # Real liquidation price
            "is_boosted": is_boosted,
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
        position["tp_price"] = self.get_tp_price(new_entry, position["side"], 0, is_boosted=True)  # Keep boost TP
        position["sl_price"] = self.get_sl_price(new_entry, position["side"])

        # Activate trailing after half-close cycle
        # Start tracking from current price (not 0) so trailing can work immediately
        self.trailing_active = True
        current_roi = self.calculate_roi(new_entry, exit_price, position["side"])
        self.boosted_peak_roi = max(current_roi, self.trailing_activation_roi)  # Start at activation threshold

        print(f"[{timestamp}] >>> HALF CLOSE @ ${exit_price:.4f} | Locked: ${half_pnl:+.2f} | Added 0.5x back | Trailing NOW ACTIVE (Peak: {self.boosted_peak_roi*100:.1f}%)")

        return position

    def activate_boost_mode(self, trigger_side: str, timestamp):
        """Activate enhanced boost mode when one side hits DCA 3"""
        # IMPORTANT: If strong trend mode is active, deactivate it first
        # Boost mode and Strong Trend mode should NEVER be active together
        if self.strong_trend_mode:
            self.deactivate_strong_trend_mode(timestamp, "Boost mode taking over")

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
        """Run the backtest with enhanced boost mode + Strong Trend Mode + Scale-in"""
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
        print(f"  - AFTER SL: STOP for the day, restart next day")
        print(f"STRONG TREND MODE: ADX > {self.adx_threshold}")
        print(f"  - Winner: 2x entry, half-close, trail")
        print(f"  - Loser: ALL DCA BLOCKED (no adding to losing side!)")
        print(f"SCALE-IN: After {self.scale_in_after_tps} consecutive TPs -> {self.scale_in_multiplier}x")
        print("="*70)

        # Calculate ADX for trend detection
        print("Calculating ADX for trend detection...")
        adx_series, plus_di_series, minus_di_series = self.calculate_adx(df, self.adx_period)

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

            # Decrement cooldown counter
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

            # Check if new day - restart trading after stop
            if self.check_new_day(timestamp):
                print(f"[{timestamp}] >>> NEW DAY - Restarting trading")
                self.cooldown_remaining = 0  # Reset cooldown on new day
                self.long_position = self.open_position("LONG", close)
                self.short_position = self.open_position("SHORT", close)
                print(f"[{timestamp}] Opened LONG @ ${close:.4f}")
                print(f"[{timestamp}] Opened SHORT @ ${close:.4f}")
                continue

            # Skip if stopped for the day
            if self.stopped_for_day:
                continue

            # Check for STRONG TREND MODE (ADX-based)
            is_strong_trend, trend_direction = self.check_strong_trend(i, adx_series, plus_di_series, minus_di_series)

            if is_strong_trend and not self.strong_trend_mode:
                # NEVER activate strong trend mode if BOOST MODE is already active
                # These two should NEVER happen together - boost mode takes priority
                if self.boost_mode_active:
                    pass  # Skip trend mode - boost mode already active
                else:
                    # Activate strong trend mode
                    self.activate_strong_trend_mode(trend_direction, timestamp)

                    # Determine winner/loser based on trend direction
                    # UP trend = LONG is winner, SHORT is loser
                    # DOWN trend = SHORT is winner, LONG is loser
                    winner_side = "LONG" if trend_direction == "UP" else "SHORT"
                    loser_side = "SHORT" if trend_direction == "UP" else "LONG"

                    # Apply 2x to winner if exists (trend boost)
                    # BUT ONLY if not already boosted from DCA boost mode
                    winner_pos = self.long_position if winner_side == "LONG" else self.short_position
                    if winner_pos and not winner_pos.get("is_trend_boosted", False) and not winner_pos.get("is_boosted", False):
                        old_margin = winner_pos["margin"]
                        winner_pos["quantity"] *= 2.0  # 2x for strong trend winner
                        winner_pos["margin"] *= 2.0
                        winner_pos["is_trend_boosted"] = True
                        winner_pos["is_boosted"] = True  # Enable trailing/half-close logic
                        print(f"[{timestamp}] >>> TREND BOOST {winner_side}: margin ${old_margin:.2f} -> ${winner_pos['margin']:.2f} (2x)")

            elif not is_strong_trend and self.strong_trend_mode:
                # Deactivate strong trend mode when ADX drops
                self.deactivate_strong_trend_mode(timestamp, f"ADX dropped below {self.adx_threshold}")

                # Remove trend boost flag from positions
                if self.long_position:
                    self.long_position["is_trend_boosted"] = False
                if self.short_position:
                    self.short_position["is_trend_boosted"] = False

            # Check LONG position
            if self.long_position:
                is_boosted = self.long_position.get("is_boosted", False)

                # LIQUIDATION CHECK - Check FIRST before any other exit conditions
                if self.check_liquidation(self.long_position, low, high):
                    liq_price = self.long_position["liq_price"]
                    margin_lost = self.long_position["margin"]
                    self.balance -= margin_lost  # Lose entire margin
                    if self.balance < 0:
                        self.balance = 0
                    self.total_losses += 1
                    self.total_pnl -= margin_lost
                    self.liquidations_long += 1
                    print(f"[{timestamp}] >>> LONG LIQUIDATED @ ${liq_price:.4f} | Lost: ${margin_lost:.2f} | DCA: {self.long_position['dca_level']}")
                    self.trades.append({
                        "timestamp": timestamp,
                        "side": "LONG",
                        "entry_price": self.long_position["entry_price"],
                        "exit_price": liq_price,
                        "quantity": self.long_position["quantity"],
                        "margin": margin_lost,
                        "dca_level": self.long_position["dca_level"],
                        "pnl": -margin_lost,
                        "exit_type": "LIQUIDATION",
                        "balance": self.balance,
                        "is_boosted": is_boosted,
                        "close_pct": 1.0
                    })
                    self.long_position = None
                    # Deactivate boost mode if this side was boosted
                    if self.boost_mode_active and self.boosted_side == "LONG":
                        self.deactivate_boost_mode(timestamp, "LONG liquidated")
                    self.stopped_for_day = True
                    self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                    continue

                # Check if this position is marked for trailing close (after SL on other side)
                if self.long_position.get("trailing_for_close", False) and self.trailing_active:
                    should_trail_close, trail_price = self.check_trailing_stop(self.long_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.long_position, trail_price, "TRAILING_CLOSE", timestamp)
                        print(f"[{timestamp}] LONG TRAILING CLOSE @ ${trail_price:.4f} | P&L: ${pnl:+.2f} | Peak ROI: {self.boosted_peak_roi*100:.1f}%")
                        self.long_position = None
                        self.trailing_active = False
                        self.boosted_peak_roi = 0
                        # STOP FOR DAY - Both sides will restart together tomorrow
                        self.stopped_for_day = True
                        self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                        print(f"[{timestamp}] >>> Trailing close complete - STOPPED FOR THE DAY")
                        continue

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
                        # Note: PNL already added to balance in close_position() called by half_close_and_readd()
                        continue

                # Check TP hit (non-boosted or full close)
                elif high >= self.long_position["tp_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["tp_price"], "TP", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] LONG TP @ ${self.long_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")

                    # Track consecutive TPs for scale-in
                    self.consecutive_tp_count["LONG"] += 1
                    self.consecutive_tp_count["SHORT"] = 0  # Reset opposite side

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode(timestamp, "LONG recovered (TP)")

                    # Re-enter (boost if in boost mode and this side is boosted)
                    boost_mult = self.boost_multiplier if (self.boost_mode_active and self.boosted_side == "LONG") else 1.0
                    self.long_position = self.open_position("LONG", close, boost_multiplier=boost_mult)

                    # SCALE-IN: After consecutive TPs, scale position to 1.5x
                    if self.consecutive_tp_count["LONG"] >= self.scale_in_after_tps and self.scaled_in_side != "LONG":
                        self.long_position = self.apply_scale_in(self.long_position, timestamp)

                    if boost_mult > 1.0:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                # Check SL hit
                elif low <= self.long_position["sl_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["sl_price"], "SL", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    self.sl_hits_long += 1
                    print(f"[{timestamp}] LONG SL @ ${self.long_position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")

                    # Reset consecutive TPs and scale-in on SL
                    self.consecutive_tp_count["LONG"] = 0
                    if self.scaled_in_side == "LONG":
                        self.scaled_in_side = None

                    self.long_position = None

                    # Deactivate boost if this was the trigger side
                    if self.boost_mode_active and self.boost_trigger_side == "LONG":
                        self.deactivate_boost_mode(timestamp, "Trigger side hit SL")

                    # Handle winning side - keep with trailing or stop for day (if enabled)
                    if self.stop_for_day_enabled:
                        self.stop_for_day(timestamp, close, sl_side="LONG")
                        continue
                    else:
                        # Set cooldown before re-entry
                        self.cooldown_remaining = self.cooldown_bars
                        # Just re-enter LONG immediately (cooldown affects next SL only)
                        self.long_position = self.open_position("LONG", close)
                        continue

                # NO DCA MODE - Check ROI-based boost trigger instead
                # Activate boost when position hits -20% ROI
                else:
                    long_roi = self.calculate_roi(self.long_position["entry_price"], close, "LONG")
                    if long_roi <= self.boost_trigger_roi and not self.boost_mode_active:
                        # STRONG TREND MODE: Don't boost if this is the loser side in a strong trend
                        if self.strong_trend_mode and self.trend_direction == "DOWN":
                            pass  # Don't activate boost on loser side during strong trend
                        else:
                            self.activate_boost_mode("LONG", timestamp)
                            print(f"[{timestamp}] >>> BOOST TRIGGERED! LONG at {long_roi*100:.1f}% ROI -> SHORT boosted 1.5x")
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

                # LIQUIDATION CHECK - Check FIRST before any other exit conditions
                if self.check_liquidation(self.short_position, low, high):
                    liq_price = self.short_position["liq_price"]
                    margin_lost = self.short_position["margin"]
                    self.balance -= margin_lost  # Lose entire margin
                    if self.balance < 0:
                        self.balance = 0
                    self.total_losses += 1
                    self.total_pnl -= margin_lost
                    self.liquidations_short += 1
                    print(f"[{timestamp}] >>> SHORT LIQUIDATED @ ${liq_price:.4f} | Lost: ${margin_lost:.2f} | DCA: {self.short_position['dca_level']}")
                    self.trades.append({
                        "timestamp": timestamp,
                        "side": "SHORT",
                        "entry_price": self.short_position["entry_price"],
                        "exit_price": liq_price,
                        "quantity": self.short_position["quantity"],
                        "margin": margin_lost,
                        "dca_level": self.short_position["dca_level"],
                        "pnl": -margin_lost,
                        "exit_type": "LIQUIDATION",
                        "balance": self.balance,
                        "is_boosted": is_boosted,
                        "close_pct": 1.0
                    })
                    self.short_position = None
                    # Deactivate boost mode if this side was boosted
                    if self.boost_mode_active and self.boosted_side == "SHORT":
                        self.deactivate_boost_mode(timestamp, "SHORT liquidated")
                    self.stopped_for_day = True
                    self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                    continue

                # Check if this position is marked for trailing close (after SL on other side)
                if self.short_position.get("trailing_for_close", False) and self.trailing_active:
                    should_trail_close, trail_price = self.check_trailing_stop(self.short_position, close)
                    if should_trail_close:
                        pnl = self.close_position(self.short_position, trail_price, "TRAILING_CLOSE", timestamp)
                        print(f"[{timestamp}] SHORT TRAILING CLOSE @ ${trail_price:.4f} | P&L: ${pnl:+.2f} | Peak ROI: {self.boosted_peak_roi*100:.1f}%")
                        self.short_position = None
                        self.trailing_active = False
                        self.boosted_peak_roi = 0
                        # STOP FOR DAY - Both sides will restart together tomorrow
                        self.stopped_for_day = True
                        self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                        print(f"[{timestamp}] >>> Trailing close complete - STOPPED FOR THE DAY")
                        continue

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
                        # Note: PNL already added to balance in close_position() called by half_close_and_readd()
                        continue

                # Check TP hit (non-boosted or full close)
                elif low <= self.short_position["tp_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["tp_price"], "TP", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    print(f"[{timestamp}] SHORT TP @ ${self.short_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")

                    # Track consecutive TPs for scale-in
                    self.consecutive_tp_count["SHORT"] += 1
                    self.consecutive_tp_count["LONG"] = 0  # Reset opposite side

                    # Check if this was the trigger side - deactivate boost
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode(timestamp, "SHORT recovered (TP)")

                    # Re-enter (boost if in boost mode and this side is boosted)
                    boost_mult = self.boost_multiplier if (self.boost_mode_active and self.boosted_side == "SHORT") else 1.0
                    self.short_position = self.open_position("SHORT", close, boost_multiplier=boost_mult)

                    # SCALE-IN: After consecutive TPs, scale position to 1.5x
                    if self.consecutive_tp_count["SHORT"] >= self.scale_in_after_tps and self.scaled_in_side != "SHORT":
                        self.short_position = self.apply_scale_in(self.short_position, timestamp)

                    if boost_mult > 1.0:
                        self.boosted_peak_roi = 0
                        self.trailing_active = False

                # Check SL hit
                elif high >= self.short_position["sl_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["sl_price"], "SL", timestamp)
                    if is_boosted:
                        self.boost_profits += pnl
                    self.sl_hits_short += 1
                    print(f"[{timestamp}] SHORT SL @ ${self.short_position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")

                    # Reset consecutive TPs and scale-in on SL
                    self.consecutive_tp_count["SHORT"] = 0
                    if self.scaled_in_side == "SHORT":
                        self.scaled_in_side = None

                    self.short_position = None

                    # Deactivate boost if this was the trigger side
                    if self.boost_mode_active and self.boost_trigger_side == "SHORT":
                        self.deactivate_boost_mode(timestamp, "Trigger side hit SL")

                    # Handle winning side - keep with trailing or stop for day (if enabled)
                    if self.stop_for_day_enabled:
                        self.stop_for_day(timestamp, close, sl_side="SHORT")
                        continue
                    else:
                        # Set cooldown before re-entry
                        self.cooldown_remaining = self.cooldown_bars
                        # Just re-enter SHORT immediately (cooldown affects next SL only)
                        self.short_position = self.open_position("SHORT", close)
                        continue

                # Check DCA - BUT NOT if this side is boosted!
                # NO DCA MODE - Check ROI-based boost trigger instead
                # Activate boost when position hits -20% ROI
                else:
                    short_roi = self.calculate_roi(self.short_position["entry_price"], close, "SHORT")
                    if short_roi <= self.boost_trigger_roi and not self.boost_mode_active:
                        # STRONG TREND MODE: Don't boost if this is the loser side in a strong trend
                        if self.strong_trend_mode and self.trend_direction == "UP":
                            pass  # Don't activate boost on loser side during strong trend
                        else:
                            self.activate_boost_mode("SHORT", timestamp)
                            print(f"[{timestamp}] >>> BOOST TRIGGERED! SHORT at {short_roi*100:.1f}% ROI -> LONG boosted 1.5x")
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

        # Export trade journal to CSV
        self.export_trade_journal()

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
            "strong_trend_activations": self.strong_trend_activations,
            "scale_in_count": self.scale_in_count,
            "liquidations_long": self.liquidations_long,
            "liquidations_short": self.liquidations_short,
            "liquidations_total": self.liquidations_long + self.liquidations_short,
            "liquidated": self.balance <= 0
        }

    def export_trade_journal(self, filename: str = None):
        """Export all trades to CSV trade journal"""
        if not self.trades:
            print("No trades to export")
            return

        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_journal_{self.symbol}_{timestamp_str}.csv"

        fieldnames = [
            "timestamp", "symbol", "side", "entry_price", "exit_price",
            "quantity", "margin", "dca_level", "pnl", "pnl_pct", "exit_type",
            "balance_after", "is_boosted", "close_pct", "cumulative_pnl"
        ]

        cumulative_pnl = 0
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for trade in self.trades:
                cumulative_pnl += trade["pnl"]
                pnl_pct = (trade["pnl"] / trade["margin"] * 100) if trade["margin"] > 0 else 0

                writer.writerow({
                    "timestamp": trade["timestamp"],
                    "symbol": self.symbol,
                    "side": trade["side"],
                    "entry_price": f"{trade['entry_price']:.6f}",
                    "exit_price": f"{trade['exit_price']:.6f}",
                    "quantity": f"{trade['quantity']:.6f}",
                    "margin": f"{trade['margin']:.2f}",
                    "dca_level": trade["dca_level"],
                    "pnl": f"{trade['pnl']:.2f}",
                    "pnl_pct": f"{pnl_pct:.2f}",
                    "exit_type": trade["exit_type"],
                    "balance_after": f"{trade['balance']:.2f}",
                    "is_boosted": trade.get("is_boosted", False),
                    "close_pct": f"{trade.get('close_pct', 1.0):.2f}",
                    "cumulative_pnl": f"{cumulative_pnl:.2f}"
                })

        print(f"\n>>> Trade journal exported to: {filename}")
        print(f"    Total trades: {len(self.trades)}")

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

        # Stop For Day Analysis
        print(f"\n>>> STOP FOR DAY Analysis:")
        print(f"  SL Hits (LONG):      {self.sl_hits_long}")
        print(f"  SL Hits (SHORT):     {self.sl_hits_short}")
        print(f"  Total SL Hits:       {self.sl_hits_long + self.sl_hits_short}")
        print(f"  Days Stopped:        {self.days_stopped}")

        # Liquidation Analysis
        print(f"\n>>> LIQUIDATION Analysis (Real Liq Prices):")
        print(f"  LONG Liquidations:   {self.liquidations_long}")
        print(f"  SHORT Liquidations:  {self.liquidations_short}")
        print(f"  Total Liquidations:  {self.liquidations_long + self.liquidations_short}")

        # Strong Trend Mode Analysis
        print(f"\n>>> STRONG TREND MODE Analysis:")
        print(f"  Trend Activations:   {self.strong_trend_activations}")
        print(f"  ADX Threshold:       {self.adx_threshold}")

        # Scale-in Analysis
        print(f"\n>>> SCALE-IN Analysis:")
        print(f"  Scale-in Count:      {self.scale_in_count}")
        print(f"  After Consecutive:   {self.scale_in_after_tps} TPs")

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
    """Test Enhanced Boost over specified days with selected pairs"""
    # Configuration
    BACKTEST_DAYS = 365  # 12 months
    symbols = ["BTCUSDT", "DOTUSDT", "BNBUSDT", "AVAXUSDT", "SOLUSDT"]

    print("="*80)
    print(f"ENHANCED BOOST MODE - {BACKTEST_DAYS} DAY BACKTEST")
    print("="*80)
    print("Strategy: At DCA 3 -> Boost opposite 1.5x")
    print("          At TP: Close HALF, lock profit, add 0.5x back")
    print("          Trailing starts AFTER each half-close")
    print("          Continue until losing side recovers or SL")
    print("="*80)

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
        df = backtester.get_historical_data(days=BACKTEST_DAYS, interval="1h")

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

    # Calculate portfolio-level metrics
    num_symbols = len(all_results)
    portfolio_capital = total_starting  # Total portfolio capital (e.g., $500 for 5 symbols)
    allocation_per_symbol = portfolio_capital / num_symbols  # $100 per symbol

    # Calculate portfolio max drawdown (sum of all symbol drawdowns at worst point)
    total_max_dd_dollars = sum(r.get('max_dd_dollars', 0) for r in all_results)
    portfolio_max_dd_pct = (total_max_dd_dollars / portfolio_capital) * 100

    # Detailed per-symbol analysis
    for r in all_results:
        wins = r["wins"]
        losses = r["losses"]
        total = r["total_trades"]
        trades_per_day = total / BACKTEST_DAYS
        wins_per_day = wins / BACKTEST_DAYS
        losses_per_day = losses / BACKTEST_DAYS

        # Calculate total win $ and total loss $
        total_win_dollars = r.get("total_win_dollars", 0)
        total_loss_dollars = r.get("total_loss_dollars", 0)
        avg_win = total_win_dollars / wins if wins > 0 else 0
        avg_loss = total_loss_dollars / losses if losses > 0 else 0

        # Profit factor
        profit_factor = abs(total_win_dollars / total_loss_dollars) if total_loss_dollars != 0 else float('inf')

        # Max DD calculations - both for symbol allocation AND portfolio
        max_dd_dollars = r.get('max_dd_dollars', 0)
        max_dd_pct_of_allocation = r['max_drawdown']  # % of $100
        max_dd_pct_of_portfolio = (max_dd_dollars / portfolio_capital) * 100  # % of $500

        print(f"\n{'='*60}")
        print(f"  {r['symbol']} - DETAILED ANALYTICS")
        print(f"{'='*60}")
        print(f"  MARKET PERFORMANCE:")
        print(f"    Price Change:        {r['price_change']:+.1f}%")
        print(f"")
        print(f"  ACCOUNT PERFORMANCE:")
        print(f"    Allocation:          ${allocation_per_symbol:.0f} ({100/num_symbols:.0f}% of ${portfolio_capital:.0f} portfolio)")
        print(f"    Ending Balance:      ${r['balance']:.2f}")
        print(f"    Net Profit/Loss:     ${r['balance'] - allocation_per_symbol:+.2f}")
        print(f"    Return (on alloc):   {r['return_pct']:+.1f}%")
        print(f"    Return (portfolio):  {(r['balance'] - allocation_per_symbol) / portfolio_capital * 100:+.2f}%")
        print(f"")
        print(f"  MAX DRAWDOWN:")
        print(f"    DD on Allocation:    ${max_dd_dollars:.2f} ({max_dd_pct_of_allocation:.1f}% of ${allocation_per_symbol:.0f})")
        print(f"    DD on Portfolio:     ${max_dd_dollars:.2f} ({max_dd_pct_of_portfolio:.1f}% of ${portfolio_capital:.0f})")
        print(f"")
        print(f"  TRADE STATISTICS:")
        print(f"    Total Trades:        {total}")
        print(f"    Winning Trades:      {wins} ({wins/total*100:.1f}%)")
        print(f"    Losing Trades:       {losses} ({losses/total*100:.1f}%)")
        print(f"    Win Rate:            {r['win_rate']:.1f}%")
        print(f"")
        print(f"  DAILY AVERAGES ({BACKTEST_DAYS} days):")
        print(f"    Trades/Day:          {trades_per_day:.1f}")
        print(f"    Wins/Day:            {wins_per_day:.1f}")
        print(f"    Losses/Day:          {losses_per_day:.2f}")
        print(f"    Daily Return:        {r['return_pct']/BACKTEST_DAYS:.2f}%")
        print(f"    Daily P&L:           ${(r['balance'] - allocation_per_symbol)/BACKTEST_DAYS:+.2f}")
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

    # Summary table with portfolio-based DD
    print("\n" + "="*140)
    print("SUMMARY TABLE")
    print("="*140)
    print(f"\n{'Symbol':<10} {'Market':<8} {'Return':<8} {'Trades':<7} {'Wins':<6} {'Losses':<7} {'WinRate':<8} {'T/Day':<6} {'TotalWon':<10} {'TotalLost':<11} {'DD($)':<8} {'DD(%Port)':<10} {'PF':<6}")
    print("-"*140)
    for r in all_results:
        trades_day = r['total_trades'] / BACKTEST_DAYS
        max_dd = r.get('max_dd_dollars', 0)
        max_dd_pct_port = (max_dd / portfolio_capital) * 100
        total_won = r.get('total_win_dollars', 0)
        total_lost = r.get('total_loss_dollars', 0)
        pf = abs(total_won / total_lost) if total_lost != 0 else 999
        status = " *LIQD*" if r["liquidated"] else ""
        print(f"{r['symbol']:<10} {r['price_change']:+.1f}%{'':<2} {r['return_pct']:+.1f}%{'':<2} {r['total_trades']:<7} {r['wins']:<6} {r['losses']:<7} {r['win_rate']:.1f}%{'':<3} {trades_day:.1f}{'':<2} ${total_won:<8.2f} ${total_lost:<9.2f} ${max_dd:<6.2f} {max_dd_pct_port:.1f}%{'':<5} {pf:.2f}x{status}")

    # Grand totals
    total_profit = total_ending - total_starting
    avg_return = total_profit / len(all_results)
    total_trades_per_day = total_trades / BACKTEST_DAYS

    # Calculate total won/lost across all symbols
    grand_total_won = sum(r.get('total_win_dollars', 0) for r in all_results)
    grand_total_lost = sum(r.get('total_loss_dollars', 0) for r in all_results)
    grand_profit_factor = abs(grand_total_won / grand_total_lost) if grand_total_lost != 0 else 999

    print(f"\n{'='*140}")
    print(f"GRAND TOTAL - PORTFOLIO PERFORMANCE (Based on ${portfolio_capital:.0f} Capital) - {BACKTEST_DAYS} DAYS")
    print(f"{'='*140}")
    print(f"  CAPITAL:")
    print(f"    Starting Capital:    ${total_starting:.2f} ({len(all_results)} symbols × ${allocation_per_symbol:.0f})")
    print(f"    Ending Capital:      ${total_ending:.2f}")
    print(f"    Net Profit:          ${total_profit:+.2f}")
    print(f"    Portfolio Return:    {total_profit/total_starting*100:+.1f}%")
    print(f"    Avg Return/Symbol:   {total_profit/total_starting*100/len(all_results):.1f}%")
    print(f"    Daily Avg Return:    {total_profit/total_starting*100/BACKTEST_DAYS:.2f}%")
    print(f"    Monthly Avg Return:  {total_profit/total_starting*100/(BACKTEST_DAYS/30):.1f}%")
    print(f"")
    print(f"  DRAWDOWN (Portfolio-Based):")
    print(f"    Max DD (Sum):        ${total_max_dd_dollars:.2f}")
    print(f"    Max DD % of Port:    {portfolio_max_dd_pct:.1f}%")
    print(f"    Worst Symbol DD:     ${max(r.get('max_dd_dollars', 0) for r in all_results):.2f} ({max(r.get('max_dd_dollars', 0) for r in all_results)/portfolio_capital*100:.1f}% of portfolio)")
    print(f"")
    print(f"  PROFIT/LOSS:")
    print(f"    Total Won (all):     ${grand_total_won:+.2f}")
    print(f"    Total Lost (all):    ${grand_total_lost:+.2f}")
    print(f"    Portfolio P/F:       {grand_profit_factor:.2f}x")
    print(f"")
    print(f"  TRADES:")
    print(f"    Total Trades:        {total_trades}")
    print(f"    Total Wins:          {total_wins}")
    print(f"    Total Losses:        {total_losses}")
    print(f"    Overall Win Rate:    {total_wins/total_trades*100:.1f}%")
    print(f"    Trades/Day (all):    {total_trades_per_day:.1f}")
    print(f"    Wins/Day:            {total_wins/BACKTEST_DAYS:.1f}")
    print(f"    Losses/Day:          {total_losses/BACKTEST_DAYS:.2f}")
    print(f"")
    print(f"  BOOST MODE:")
    print(f"    Half-Close Cycles:   {total_half_closes}")
    print(f"    Liquidations:        {liquidations}/{len(all_results)}")
    print(f"{'='*140}")


if __name__ == "__main__":
    run_90_day_test()
