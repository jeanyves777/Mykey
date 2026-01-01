#!/usr/bin/env python3
"""
Portfolio Backtest - MATCHES LIVE TRADING SYSTEM EXACTLY
=========================================================
Features matching live_trading_engine.py:
1. SHARED BALANCE POOL - All symbols draw from same capital
2. RESERVE FUND - 50% of profits protected (never traded)
3. 5% ALLOCATION BUFFER - For fees/safety
4. ISOLATED MARGIN - Each position can only lose its margin
5. HEDGE MODE - LONG + SHORT simultaneously
6. DCA LEVELS - Progressive averaging
7. BOOST MODE - 1.5x at DCA 3 trigger
8. STOP FOR DAY - After SL hit, stop until next day
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG, SYMBOL_SETTINGS


class PortfolioBacktester:
    """
    Portfolio-level backtester that matches the live trading system.
    All symbols share one balance pool with reserve fund protection.
    """

    def __init__(self, symbols: List[str], total_capital: float = 600.0):
        self.symbols = symbols
        self.num_symbols = len(symbols)

        # SHARED PORTFOLIO BALANCE
        self.total_capital = total_capital
        self.trading_capital = total_capital  # Active trading capital
        self.reserve_fund = 0.0  # Protected profits (50% of gains)
        self.reserve_pct = 0.50  # 50% of profits go to reserve

        # ALLOCATION BUFFER (5%)
        self.allocation_buffer = 0.05

        # Per-symbol state
        self.symbol_state: Dict[str, dict] = {}
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Initialize each symbol
        for symbol in symbols:
            self._init_symbol(symbol)

        # Portfolio stats
        self.all_trades = []
        self.peak_capital = total_capital
        self.max_drawdown = 0
        self.max_drawdown_dollars = 0

        # Reserve fund history
        self.reserve_history = []

    def _init_symbol(self, symbol: str):
        """Initialize state for a single symbol"""
        symbol_config = SYMBOL_SETTINGS.get(symbol, {})

        self.symbol_state[symbol] = {
            # Strategy params
            "tp_roi": symbol_config.get("tp_roi", DCA_CONFIG["take_profit_roi"]),
            "sl_roi": DCA_CONFIG["stop_loss_roi"],
            "dca_levels": symbol_config.get("dca_levels", DCA_CONFIG["levels"]),
            "budget_split": DCA_CONFIG["hedge_mode"]["budget_split"],

            # DCA percentages
            "dca_pcts": [
                DCA_CONFIG.get("initial_entry_pct", 0.10),
                DCA_CONFIG.get("dca1_pct", 0.15),
                DCA_CONFIG.get("dca2_pct", 0.20),
                DCA_CONFIG.get("dca3_pct", 0.25),
                DCA_CONFIG.get("dca4_pct", 0.30),
            ],

            # Boost params
            "boost_multiplier": 1.5,
            "boost_trigger_dca_level": 3,
            "boost_tp_multiplier": 1.5,

            # Positions
            "long_position": None,
            "short_position": None,

            # Boost state
            "boost_mode_active": False,
            "boosted_side": None,
            "boost_trigger_side": None,
            "boost_cycle_count": 0,
            "boost_locked_profit": 0,
            "trailing_active": False,
            "boosted_peak_roi": 0,

            # Stop for day
            "stopped_for_day": False,
            "sl_hit_date": None,

            # Stats
            "trades": [],
            "total_pnl": 0,
            "wins": 0,
            "losses": 0,
            "sl_hits_long": 0,
            "sl_hits_short": 0,
            "days_stopped": 0,
            "boost_activations": 0,
            "boost_profits": 0,
            "half_close_count": 0,
        }

    def get_available_capital(self) -> float:
        """Get available trading capital (excludes reserve fund and applies buffer)"""
        available = self.trading_capital * (1 - self.allocation_buffer)
        return max(0, available)

    def get_per_symbol_budget(self) -> float:
        """Get budget allocation per symbol"""
        return self.get_available_capital() / self.num_symbols

    def apply_pnl(self, pnl: float, symbol: str):
        """
        Apply PNL with SMART COMPOUNDING:
        - 50% of profits go to reserve fund (protected)
        - 50% of profits compound to trading capital
        - Losses only affect trading capital (reserve protected)
        """
        if pnl > 0:
            # PROFIT: Split between compound and reserve
            compound_amount = pnl * (1 - self.reserve_pct)
            reserve_amount = pnl * self.reserve_pct

            self.trading_capital += compound_amount
            self.reserve_fund += reserve_amount

        else:
            # LOSS: Only affects trading capital (reserve protected)
            self.trading_capital += pnl  # pnl is negative

            # Can't go below zero
            if self.trading_capital < 0:
                self.trading_capital = 0

        # Track peak and drawdown
        current_total = self.trading_capital + self.reserve_fund
        if current_total > self.peak_capital:
            self.peak_capital = current_total

        drawdown_dollars = self.peak_capital - current_total
        drawdown_pct = (drawdown_dollars / self.peak_capital * 100) if self.peak_capital > 0 else 0

        if drawdown_dollars > self.max_drawdown_dollars:
            self.max_drawdown_dollars = drawdown_dollars
            self.max_drawdown = drawdown_pct

    def get_historical_data(self, symbol: str, days: int = 365, interval: str = "1h"):
        """Fetch historical klines from Binance MAINNET with pagination"""
        print(f"Fetching {days} days of {interval} data for {symbol}...")
        print("Using Binance MAINNET for real historical data...")

        client = BinanceClient(testnet=False, use_demo=False)

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Paginate to get all data (1000 candles per request max)
        all_data = []
        current_start = start_time

        while current_start < end_time:
            df_chunk = client.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
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

        return df

    def get_tp_price(self, entry_price: float, side: str, dca_level: int,
                     symbol: str, is_boosted: bool = False) -> float:
        """Calculate take profit price"""
        state = self.symbol_state[symbol]
        tp_roi = state["tp_roi"]

        # Boosted positions get higher TP
        if is_boosted:
            tp_roi *= state["boost_tp_multiplier"]

        tp_move = tp_roi / self.leverage

        if side == "LONG":
            return entry_price * (1 + tp_move)
        else:
            return entry_price * (1 - tp_move)

    def get_sl_price(self, entry_price: float, side: str, symbol: str) -> float:
        """Calculate stop loss price"""
        state = self.symbol_state[symbol]
        sl_move = state["sl_roi"] / self.leverage

        if side == "LONG":
            return entry_price * (1 - sl_move)
        else:
            return entry_price * (1 + sl_move)

    def get_dca_trigger_price(self, entry_price: float, side: str,
                               dca_level: int, symbol: str) -> float:
        """Get price that triggers next DCA level"""
        state = self.symbol_state[symbol]
        if dca_level >= len(state["dca_levels"]):
            return None

        dca_drop = state["dca_levels"][dca_level]

        if side == "LONG":
            return entry_price * (1 - dca_drop)
        else:
            return entry_price * (1 + dca_drop)

    def open_position(self, symbol: str, side: str, price: float,
                      boost_multiplier: float = 1.0) -> dict:
        """Open new position with current budget allocation"""
        state = self.symbol_state[symbol]

        # Get current per-symbol budget
        per_symbol_budget = self.get_per_symbol_budget()
        budget = per_symbol_budget * state["budget_split"]
        initial_margin = budget * state["dca_pcts"][0]

        # Apply boost multiplier
        initial_margin *= boost_multiplier

        quantity = (initial_margin * self.leverage) / price
        is_boosted = boost_multiplier > 1.0

        position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0, symbol, is_boosted),
            "sl_price": self.get_sl_price(price, side, symbol),
            "is_boosted": is_boosted,
            "boost_multiplier": boost_multiplier,
            "symbol_budget": per_symbol_budget  # Track budget at open time
        }
        return position

    def add_dca(self, position: dict, current_price: float, symbol: str) -> dict:
        """Add DCA to existing position"""
        state = self.symbol_state[symbol]
        dca_level = position["dca_level"]

        if dca_level >= len(state["dca_pcts"]) - 1:
            return position

        # Get current per-symbol budget
        per_symbol_budget = self.get_per_symbol_budget()
        budget = per_symbol_budget * state["budget_split"]
        add_margin = budget * state["dca_pcts"][dca_level + 1]

        add_qty = (add_margin * self.leverage) / current_price

        # Calculate new average entry
        old_qty = position["quantity"]
        new_qty = old_qty + add_qty
        new_margin = position["margin"] + add_margin

        old_value = old_qty * position["entry_price"]
        add_value = add_qty * current_price
        new_entry = (old_value + add_value) / new_qty

        is_boosted = position.get("is_boosted", False)

        position["quantity"] = new_qty
        position["margin"] = new_margin
        position["entry_price"] = new_entry
        position["dca_level"] = dca_level + 1
        position["tp_price"] = self.get_tp_price(new_entry, position["side"],
                                                  dca_level + 1, symbol, is_boosted)
        position["sl_price"] = self.get_sl_price(new_entry, position["side"], symbol)

        return position

    def close_position(self, position: dict, exit_price: float, exit_type: str,
                       timestamp, symbol: str, close_pct: float = 1.0) -> float:
        """Close position and apply PNL to portfolio"""
        state = self.symbol_state[symbol]

        price_change_pct = (exit_price - position["entry_price"]) / position["entry_price"]

        if position["side"] == "LONG":
            roi = price_change_pct * self.leverage
        else:
            roi = -price_change_pct * self.leverage

        # Calculate PNL for closed portion
        closed_margin = position["margin"] * close_pct
        pnl = closed_margin * roi

        # Cap loss at margin (isolated mode)
        if pnl < -closed_margin:
            pnl = -closed_margin

        # Apply PNL with reserve fund logic
        self.apply_pnl(pnl, symbol)

        state["total_pnl"] += pnl

        if pnl > 0:
            state["wins"] += 1
        else:
            state["losses"] += 1

        # Record trade
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "quantity": position["quantity"] * close_pct,
            "margin": closed_margin,
            "dca_level": position["dca_level"],
            "pnl": pnl,
            "exit_type": exit_type,
            "trading_capital": self.trading_capital,
            "reserve_fund": self.reserve_fund,
            "is_boosted": position.get("is_boosted", False),
            "close_pct": close_pct
        }
        state["trades"].append(trade)
        self.all_trades.append(trade)

        return pnl

    def check_dca_trigger(self, position: dict, current_price: float, symbol: str) -> bool:
        """Check if DCA should trigger"""
        state = self.symbol_state[symbol]
        dca_level = position["dca_level"]

        if dca_level >= len(state["dca_levels"]):
            return False

        trigger_price = self.get_dca_trigger_price(
            position["entry_price"], position["side"], dca_level, symbol
        )

        if trigger_price is None:
            return False

        if position["side"] == "LONG":
            return current_price <= trigger_price
        else:
            return current_price >= trigger_price

    def activate_boost_mode(self, symbol: str, trigger_side: str, timestamp):
        """Activate boost mode for symbol"""
        state = self.symbol_state[symbol]
        state["boost_mode_active"] = True
        state["boost_trigger_side"] = trigger_side
        state["boosted_side"] = "SHORT" if trigger_side == "LONG" else "LONG"
        state["boost_cycle_count"] = 0
        state["boost_locked_profit"] = 0
        state["trailing_active"] = False
        state["boosted_peak_roi"] = 0
        state["boost_activations"] += 1
        print(f"[{timestamp}] {symbol} BOOST ACTIVATED: {state['boosted_side']} side boosted 1.5x")

    def deactivate_boost_mode(self, symbol: str, timestamp, reason: str = ""):
        """Deactivate boost mode for symbol"""
        state = self.symbol_state[symbol]
        if state["boost_mode_active"]:
            print(f"[{timestamp}] {symbol} BOOST DEACTIVATED: {reason}")
        state["boost_mode_active"] = False
        state["boosted_side"] = None
        state["boost_trigger_side"] = None
        state["trailing_active"] = False
        state["boosted_peak_roi"] = 0

    def stop_for_day(self, symbol: str, timestamp, exit_price: float):
        """Stop trading symbol for the day after SL hit"""
        state = self.symbol_state[symbol]
        state["stopped_for_day"] = True
        state["sl_hit_date"] = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        state["days_stopped"] += 1

        # Close remaining positions
        if state["long_position"]:
            pnl = self.close_position(state["long_position"], exit_price,
                                      "STOP_DAY", timestamp, symbol)
            print(f"[{timestamp}] {symbol} Closing LONG for day stop | P&L: ${pnl:+.2f}")
            state["long_position"] = None

        if state["short_position"]:
            pnl = self.close_position(state["short_position"], exit_price,
                                      "STOP_DAY", timestamp, symbol)
            print(f"[{timestamp}] {symbol} Closing SHORT for day stop | P&L: ${pnl:+.2f}")
            state["short_position"] = None

        # Deactivate boost
        if state["boost_mode_active"]:
            self.deactivate_boost_mode(symbol, timestamp, "Stopped for day")

        print(f"[{timestamp}] {symbol} >>> STOPPED FOR THE DAY")

    def check_new_day(self, symbol: str, timestamp) -> bool:
        """Check if new day and should restart trading"""
        state = self.symbol_state[symbol]

        if not state["stopped_for_day"]:
            return False

        current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if current_date > state["sl_hit_date"]:
            state["stopped_for_day"] = False
            state["sl_hit_date"] = None
            return True
        return False

    def calculate_roi(self, entry_price: float, current_price: float, side: str) -> float:
        """Calculate ROI for a position"""
        if side == "LONG":
            price_pct = (current_price - entry_price) / entry_price
        else:
            price_pct = (entry_price - current_price) / entry_price
        return price_pct * self.leverage

    def run_backtest(self, all_data: Dict[str, pd.DataFrame], days: int = 365):
        """Run portfolio backtest across all symbols"""

        print("\n" + "="*80)
        print("PORTFOLIO BACKTEST - MATCHES LIVE TRADING SYSTEM")
        print("="*80)
        print(f"Total Capital: ${self.total_capital:.2f}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Per Symbol: ${self.get_per_symbol_budget():.2f}")
        print(f"Reserve Fund: {self.reserve_pct*100:.0f}% of profits protected")
        print(f"Allocation Buffer: {self.allocation_buffer*100:.0f}%")
        print(f"Leverage: {self.leverage}x")
        print("="*80)

        # Get common timestamp index (intersection of all symbols)
        common_index = None
        for symbol, df in all_data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        print(f"\nCommon timeframe: {len(common_index)} candles")
        print(f"From: {common_index[0]} to {common_index[-1]}")

        # Open initial positions for all symbols
        for symbol in self.symbols:
            state = self.symbol_state[symbol]
            first_price = all_data[symbol].loc[common_index[0], 'close']

            state["long_position"] = self.open_position(symbol, "LONG", first_price)
            state["short_position"] = self.open_position(symbol, "SHORT", first_price)

            print(f"[{symbol}] Opened LONG+SHORT @ ${first_price:.4f}")

        # Main simulation loop
        for i, timestamp in enumerate(common_index):
            # Check if portfolio is liquidated
            if self.trading_capital <= 0:
                print(f"\n[{timestamp}] PORTFOLIO LIQUIDATED!")
                break

            # Process each symbol
            for symbol in self.symbols:
                state = self.symbol_state[symbol]
                row = all_data[symbol].loc[timestamp]

                high = row['high']
                low = row['low']
                close = row['close']

                # Check new day - restart after stop
                if self.check_new_day(symbol, timestamp):
                    print(f"[{timestamp}] {symbol} >>> NEW DAY - Restarting")
                    state["long_position"] = self.open_position(symbol, "LONG", close)
                    state["short_position"] = self.open_position(symbol, "SHORT", close)
                    continue

                # Skip if stopped for day
                if state["stopped_for_day"]:
                    continue

                # Process LONG position
                if state["long_position"]:
                    self._process_position(symbol, "LONG", timestamp, high, low, close)

                # Process SHORT position
                if state["short_position"]:
                    self._process_position(symbol, "SHORT", timestamp, high, low, close)

            # Record reserve fund history periodically
            if i % 24 == 0:  # Every 24 hours
                self.reserve_history.append({
                    "timestamp": timestamp,
                    "trading_capital": self.trading_capital,
                    "reserve_fund": self.reserve_fund,
                    "total": self.trading_capital + self.reserve_fund
                })

        # Print results
        self._print_results(days)

        return self._get_results()

    def _process_position(self, symbol: str, side: str, timestamp,
                          high: float, low: float, close: float):
        """Process a single position for TP/SL/DCA"""
        state = self.symbol_state[symbol]
        position = state["long_position"] if side == "LONG" else state["short_position"]

        if position is None:
            return

        is_boosted = position.get("is_boosted", False)

        # Check TP
        if side == "LONG" and high >= position["tp_price"]:
            pnl = self.close_position(position, position["tp_price"], "TP",
                                      timestamp, symbol)
            if is_boosted:
                state["boost_profits"] += pnl
            print(f"[{timestamp}] {symbol} LONG TP @ ${position['tp_price']:.4f} | P&L: ${pnl:+.2f}")
            state["long_position"] = None

            # Re-enter
            state["long_position"] = self.open_position(symbol, "LONG", close)

            # Check if this closes boost mode
            if state["boost_mode_active"] and state["boosted_side"] == "LONG":
                self.deactivate_boost_mode(symbol, timestamp, "Boosted side hit TP")

        elif side == "SHORT" and low <= position["tp_price"]:
            pnl = self.close_position(position, position["tp_price"], "TP",
                                      timestamp, symbol)
            if is_boosted:
                state["boost_profits"] += pnl
            print(f"[{timestamp}] {symbol} SHORT TP @ ${position['tp_price']:.4f} | P&L: ${pnl:+.2f}")
            state["short_position"] = None

            # Re-enter
            state["short_position"] = self.open_position(symbol, "SHORT", close)

            # Check if this closes boost mode
            if state["boost_mode_active"] and state["boosted_side"] == "SHORT":
                self.deactivate_boost_mode(symbol, timestamp, "Boosted side hit TP")

        # Check SL
        elif side == "LONG" and low <= position["sl_price"]:
            pnl = self.close_position(position, position["sl_price"], "SL",
                                      timestamp, symbol)
            if is_boosted:
                state["boost_profits"] += pnl
            state["sl_hits_long"] += 1
            print(f"[{timestamp}] {symbol} LONG SL @ ${position['sl_price']:.4f} | P&L: ${pnl:+.2f}")
            state["long_position"] = None

            # Deactivate boost if this was the trigger side
            if state["boost_mode_active"] and state["boost_trigger_side"] == "LONG":
                self.deactivate_boost_mode(symbol, timestamp, "Trigger side hit SL")

            # STOP FOR DAY
            self.stop_for_day(symbol, timestamp, close)

        elif side == "SHORT" and high >= position["sl_price"]:
            pnl = self.close_position(position, position["sl_price"], "SL",
                                      timestamp, symbol)
            if is_boosted:
                state["boost_profits"] += pnl
            state["sl_hits_short"] += 1
            print(f"[{timestamp}] {symbol} SHORT SL @ ${position['sl_price']:.4f} | P&L: ${pnl:+.2f}")
            state["short_position"] = None

            # Deactivate boost if this was the trigger side
            if state["boost_mode_active"] and state["boost_trigger_side"] == "SHORT":
                self.deactivate_boost_mode(symbol, timestamp, "Trigger side hit SL")

            # STOP FOR DAY
            self.stop_for_day(symbol, timestamp, close)

        # Check DCA
        elif self.check_dca_trigger(position, close, symbol):
            # Skip DCA if boosted side
            if state["boost_mode_active"] and state["boosted_side"] == side:
                pass
            else:
                old_level = position["dca_level"]
                position = self.add_dca(position, close, symbol)
                new_level = position["dca_level"]

                if side == "LONG":
                    state["long_position"] = position
                else:
                    state["short_position"] = position

                print(f"[{timestamp}] {symbol} {side} DCA {old_level} -> {new_level} @ ${close:.4f}")

                # Check for boost trigger at DCA 3
                if new_level == state["boost_trigger_dca_level"] and not state["boost_mode_active"]:
                    self.activate_boost_mode(symbol, side, timestamp)

                    # Boost opposite side
                    opposite_side = "SHORT" if side == "LONG" else "LONG"
                    opposite_pos = state["short_position"] if side == "LONG" else state["long_position"]

                    if opposite_pos:
                        # Close and reopen with boost
                        self.close_position(opposite_pos, close, "BOOST_CLOSE", timestamp, symbol)
                        boosted_pos = self.open_position(symbol, opposite_side, close,
                                                         boost_multiplier=state["boost_multiplier"])

                        if opposite_side == "LONG":
                            state["long_position"] = boosted_pos
                        else:
                            state["short_position"] = boosted_pos

                        print(f"[{timestamp}] {symbol} {opposite_side} BOOSTED 1.5x @ ${close:.4f}")

    def _print_results(self, days: int):
        """Print comprehensive results"""
        print("\n" + "="*100)
        print("PORTFOLIO BACKTEST RESULTS - MATCHING LIVE SYSTEM")
        print("="*100)

        total_capital = self.trading_capital + self.reserve_fund
        net_profit = total_capital - self.total_capital

        print(f"\n>>> PORTFOLIO SUMMARY:")
        print(f"  Starting Capital:      ${self.total_capital:.2f}")
        print(f"  Trading Capital:       ${self.trading_capital:.2f}")
        print(f"  Reserve Fund:          ${self.reserve_fund:.2f} (PROTECTED)")
        print(f"  Total Value:           ${total_capital:.2f}")
        print(f"  Net Profit:            ${net_profit:+.2f}")
        print(f"  Portfolio Return:      {net_profit/self.total_capital*100:+.1f}%")
        print(f"  Max Drawdown:          ${self.max_drawdown_dollars:.2f} ({self.max_drawdown:.1f}%)")

        print(f"\n>>> RESERVE FUND ANALYSIS:")
        print(f"  Reserve Protected:     ${self.reserve_fund:.2f}")
        print(f"  % of Total Profit:     {self.reserve_fund/net_profit*100:.1f}%" if net_profit > 0 else "  N/A (no profit)")
        print(f"  This capital CANNOT be lost in trading!")

        print(f"\n>>> PER-SYMBOL BREAKDOWN:")
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for symbol in self.symbols:
            state = self.symbol_state[symbol]
            trades = len(state["trades"])
            wins = state["wins"]
            losses = state["losses"]
            pnl = state["total_pnl"]

            total_trades += trades
            total_wins += wins
            total_losses += losses

            win_rate = (wins / trades * 100) if trades > 0 else 0
            print(f"  {symbol}: {trades} trades | {wins}W/{losses}L ({win_rate:.1f}%) | PnL: ${pnl:+.2f}")
            print(f"           SL Hits: {state['sl_hits_long']}L/{state['sl_hits_short']}S | Days Stopped: {state['days_stopped']}")

        print(f"\n>>> TRADE STATISTICS:")
        print(f"  Total Trades:          {total_trades}")
        print(f"  Wins:                  {total_wins}")
        print(f"  Losses:                {total_losses}")
        print(f"  Win Rate:              {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "  N/A")
        print(f"  Trades/Day:            {total_trades/days:.1f}")

        print(f"\n>>> KEY INSIGHT:")
        print(f"  With Reserve Fund, ${self.reserve_fund:.2f} is PROTECTED and cannot be lost.")
        print(f"  Even if trading capital goes to $0, reserve fund remains safe.")
        print(f"  This is why live system is more resilient than simple backtest!")

        print("="*100)

    def _get_results(self) -> dict:
        """Get results dictionary"""
        total_capital = self.trading_capital + self.reserve_fund

        return {
            "starting_capital": self.total_capital,
            "trading_capital": self.trading_capital,
            "reserve_fund": self.reserve_fund,
            "total_value": total_capital,
            "net_profit": total_capital - self.total_capital,
            "return_pct": (total_capital - self.total_capital) / self.total_capital * 100,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_dollars": self.max_drawdown_dollars,
            "total_trades": len(self.all_trades),
            "symbol_results": {s: self.symbol_state[s] for s in self.symbols}
        }


def run_portfolio_backtest():
    """Run the portfolio backtest"""

    # Configuration
    BACKTEST_DAYS = 365
    TOTAL_CAPITAL = 600.0  # $600 total ($100 per symbol)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOTUSDT", "BNBUSDT", "AVAXUSDT"]

    print("="*100)
    print(f"PORTFOLIO BACKTEST - {BACKTEST_DAYS} DAYS")
    print("="*100)
    print("This backtest MATCHES the live trading system exactly:")
    print("  - Shared balance pool across all symbols")
    print("  - Reserve Fund: 50% of profits protected")
    print("  - 5% allocation buffer for fees/safety")
    print("  - Isolated margin per position")
    print("="*100)

    # Initialize portfolio backtester
    backtester = PortfolioBacktester(symbols, total_capital=TOTAL_CAPITAL)

    # Fetch data for all symbols
    all_data = {}
    for symbol in symbols:
        df = backtester.get_historical_data(symbol, days=BACKTEST_DAYS)
        if df is not None and len(df) > 0:
            all_data[symbol] = df
        else:
            print(f"ERROR: Could not fetch data for {symbol}")
            return

    # Run backtest
    results = backtester.run_backtest(all_data, days=BACKTEST_DAYS)

    return results


if __name__ == "__main__":
    run_portfolio_backtest()
