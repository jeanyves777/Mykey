"""
Run Simple Trend Following Trading
==================================

NO ML - Just follow H1 trend direction.

Strategy:
1. Check H1 EMA50 vs EMA200 for each pair
2. Open position IMMEDIATELY in trend direction
3. When closed (TP/SL/4H max), reopen in same trend direction
4. DCA handles pullbacks (limit orders staged upfront)

Usage:
    cd C:\Users\Jean-Yves\thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.run_trend_trading
"""

import warnings
import os
import time
from datetime import datetime, timedelta
import pandas as pd

warnings.filterwarnings('ignore')

from .config.trading_config import load_config
from .data.oanda_client import OandaClient
from .data.data_loader import ForexDataLoader
from .risk_management.position_manager import PositionManager
from .risk_management.trade_executor import TradeExecutor
from .risk_management.trend_filter import TrendFilter


class SimpleTrendEngine:
    """Simple trend-following engine - NO ML needed."""

    def __init__(self, config):
        self.config = config
        self.symbols = config.symbols

        # OANDA connection
        self.client = OandaClient(
            api_key=config.oanda.api_key,
            account_id=config.oanda.account_id,
            environment=config.oanda.environment
        )

        self.data_loader = ForexDataLoader(self.client, config)
        self.position_manager = PositionManager(config)
        self.trade_executor = TradeExecutor(self.client, config)
        self.trend_filter = TrendFilter(config)

        self.equity = 5000.0

        # Track position open times for 4H max hold
        self.position_open_times = {}
        self.max_hold_hours = 4

        # Track pending DCA orders per symbol
        self.pending_dca_orders = {}

    def format_price(self, symbol: str, price: float) -> str:
        """Format price correctly for OANDA (3 decimals for JPY, 5 for others)."""
        if 'JPY' in symbol:
            return f"{price:.3f}"
        else:
            return f"{price:.5f}"

    def get_h1_trend(self, symbol: str) -> tuple:
        """Get H1 trend direction using EMA50 vs EMA200."""
        df_h1 = self.data_loader.get_latest_bars(symbol, count=250, granularity='H1')
        return self.trend_filter.check_higher_timeframe_trend(df_h1)

    def get_open_trades(self) -> dict:
        """Get all open trades from OANDA, organized by symbol."""
        success, trades = self.trade_executor.get_open_trades()
        if not success or not trades:
            return {}

        trades_by_symbol = {}
        for t in trades:
            symbol = t.get('instrument', '')
            if symbol:
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(t)
        return trades_by_symbol

    def has_position(self, symbol: str, trades_by_symbol: dict = None) -> bool:
        """Check if we have an open position for this symbol on OANDA."""
        if trades_by_symbol is None:
            trades_by_symbol = self.get_open_trades()
        return symbol in trades_by_symbol

    def calculate_smart_dca_units(self, existing_units: float, avg_entry: float,
                                     dca_price: float, target_recovery_pips: float,
                                     direction: str, pip_value: float) -> int:
        """
        Calculate DCA units needed to exit at target_recovery_pips profit.

        Formula: We want the new average entry to be close enough to current price
        so that a small recovery (target_recovery_pips) puts us in profit.

        For BUY: new_avg = (old_cost + dca_cost) / (old_units + dca_units)
                 We want: dca_price + target_pips = new_avg

        Args:
            existing_units: Total units already in position
            avg_entry: Current weighted average entry price
            dca_price: Price at which DCA will execute
            target_recovery_pips: Pips of recovery needed to exit (e.g., 2 pips)
            direction: 'BUY' or 'SELL'
            pip_value: Pip value for the symbol (0.0001 or 0.01 for JPY)

        Returns:
            Units needed for DCA to achieve quick recovery
        """
        # Target average = DCA price + recovery pips (for BUY)
        # This means when price recovers by target_recovery_pips, we're at breakeven+
        if direction == 'BUY':
            target_avg = dca_price + (target_recovery_pips * pip_value)
        else:
            target_avg = dca_price - (target_recovery_pips * pip_value)

        # Formula derivation:
        # new_avg = (existing_units * avg_entry + dca_units * dca_price) / (existing_units + dca_units)
        # target_avg = (existing_units * avg_entry + dca_units * dca_price) / (existing_units + dca_units)
        # target_avg * (existing_units + dca_units) = existing_units * avg_entry + dca_units * dca_price
        # target_avg * existing_units + target_avg * dca_units = existing_units * avg_entry + dca_units * dca_price
        # target_avg * dca_units - dca_units * dca_price = existing_units * avg_entry - target_avg * existing_units
        # dca_units * (target_avg - dca_price) = existing_units * (avg_entry - target_avg)
        # dca_units = existing_units * (avg_entry - target_avg) / (target_avg - dca_price)

        denominator = target_avg - dca_price
        if abs(denominator) < 0.000001:
            # Avoid division by zero - use fallback multiplier
            return int(existing_units * 2.0)

        dca_units = existing_units * (avg_entry - target_avg) / denominator

        # Ensure positive units and reasonable limits
        dca_units = abs(dca_units)

        # Cap at 5x existing to avoid excessive exposure
        max_units = existing_units * 5.0
        dca_units = min(dca_units, max_units)

        # Minimum 1 unit
        return max(1, int(dca_units))

    def place_dca_limit_orders(self, symbol: str, direction: str, entry_price: float, initial_units: int) -> None:
        """Place all DCA limit orders upfront after opening initial position.

        NOW USES SMART SIZING: Calculates exact units needed to exit at +2 pips.
        """
        dca = self.config.dca
        pip_value = self.config.get_pip_value(symbol)

        self.pending_dca_orders[symbol] = []

        # Track cumulative position for smart sizing
        cumulative_units = initial_units
        cumulative_cost = initial_units * entry_price

        for level in range(1, dca.max_dca_levels + 1):
            trigger_pips = dca.get_dca_trigger_pips(level)

            # Calculate limit price
            if direction == 'BUY':
                limit_price = entry_price - (trigger_pips * pip_value)
            else:
                limit_price = entry_price + (trigger_pips * pip_value)

            # Calculate current average entry
            avg_entry = cumulative_cost / cumulative_units

            # SMART DCA: Calculate units needed to exit at +2 pips from DCA price
            target_recovery = 2.0  # Exit at +2 pips profit
            dca_units = self.calculate_smart_dca_units(
                existing_units=cumulative_units,
                avg_entry=avg_entry,
                dca_price=limit_price,
                target_recovery_pips=target_recovery,
                direction=direction,
                pip_value=pip_value
            )

            # Update cumulative for next level calculation
            cumulative_units += dca_units
            cumulative_cost += dca_units * limit_price

            # Place limit order
            success, result = self.trade_executor.place_limit_order(
                symbol=symbol,
                units=dca_units,
                direction=direction,
                price=limit_price,
                stop_loss=None,  # Will share main position SL
                take_profit=None  # Will share main position TP
            )

            if success:
                order_id = result.get('order_id', '')
                self.pending_dca_orders[symbol].append({
                    'order_id': order_id,
                    'level': level,
                    'price': limit_price,
                    'units': dca_units
                })
                # Show smart sizing info
                new_avg = cumulative_cost / cumulative_units
                print(f"       [DCA L{level}] @ {self.format_price(symbol, limit_price)} | {dca_units:,} units | New Avg→{self.format_price(symbol, new_avg)} (+2p exit)")
            else:
                print(f"       [DCA L{level}] FAILED: {result.get('error', 'Unknown')}")

    def cancel_pending_dca_orders(self, symbol: str) -> None:
        """Cancel all pending DCA limit orders for a symbol."""
        if symbol not in self.pending_dca_orders:
            return

        for order in self.pending_dca_orders[symbol]:
            order_id = order.get('order_id', '')
            if order_id:
                self.trade_executor.cancel_order(order_id)

        del self.pending_dca_orders[symbol]

    def open_position(self, symbol: str, direction: str) -> bool:
        """Open a position in the given direction with DCA limit orders."""
        prices = self.client.get_pricing([symbol])
        if not prices or symbol not in prices:
            print(f"  {symbol}: Failed to get price")
            return False

        entry_price = prices[symbol]['ask'] if direction == 'BUY' else prices[symbol]['bid']
        pip_value = self.config.get_pip_value(symbol)

        # DCA sizing
        dca = self.config.dca
        sl_pips = dca.initial_sl_pips  # 40 pips
        tp_pips = dca.take_profit_pips  # 10 pips

        # Calculate SL/TP
        if direction == 'BUY':
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
        else:
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)

        # Position size (small for DCA)
        risk_pct = self.config.risk.position_size_pct
        risk_amt = self.equity * risk_pct / dca.initial_size_divisor
        units = int(risk_amt / (sl_pips * pip_value))

        # Minimum units check
        if units < 1:
            units = 1

        print(f"  {symbol}: Opening {direction} @ {self.format_price(symbol, entry_price)}")
        print(f"           Units: {units} | SL: {sl_pips:.0f}p | TP: {tp_pips:.0f}p")

        success, result = self.trade_executor.place_market_order(
            symbol=symbol,
            units=units,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if success:
            fill_price = result.get('fill_price', entry_price)
            trade_id = result.get('trade_id', '')

            self.position_manager.open_position(
                symbol=symbol,
                direction=direction,
                units=units,
                entry_price=fill_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trade_id=trade_id
            )

            # Track open time for 4H max hold
            self.position_open_times[symbol] = datetime.now()

            print(f"  {symbol}: >>> OPENED {direction} <<< (Trade ID: {trade_id})")

            # Place DCA limit orders
            print(f"       Staging {dca.max_dca_levels} DCA limit orders...")
            self.place_dca_limit_orders(symbol, direction, fill_price, units)

            return True
        else:
            print(f"  {symbol}: FAILED - {result.get('error', 'Unknown')}")
            return False

    def close_position(self, symbol: str, reason: str) -> bool:
        """Close all trades for a symbol and cancel pending DCAs."""
        # Cancel pending DCA orders
        self.cancel_pending_dca_orders(symbol)

        # Close the position on OANDA
        success, result = self.trade_executor.close_position(symbol)

        if success:
            print(f"  {symbol}: >>> CLOSED ({reason}) <<<")
            # Clean up tracking
            if symbol in self.position_open_times:
                del self.position_open_times[symbol]
            self.position_manager.close_position(symbol)
            return True
        else:
            print(f"  {symbol}: Close FAILED - {result.get('error', 'Unknown')}")
            return False

    def check_4h_max_hold(self, symbol: str) -> bool:
        """Check if position has exceeded 4H max hold time."""
        if symbol not in self.position_open_times:
            return False

        open_time = self.position_open_times[symbol]
        elapsed = datetime.now() - open_time
        max_hold = timedelta(hours=self.max_hold_hours)

        return elapsed >= max_hold

    def calculate_average_entry(self, trades: list) -> tuple:
        """Calculate weighted average entry price for DCA positions.

        Returns:
            Tuple of (avg_entry_price, total_units, direction)
        """
        total_cost = 0.0
        total_units = 0.0

        for t in trades:
            units = abs(float(t.get('currentUnits', 0)))
            price = float(t.get('price', 0))
            total_cost += units * price
            total_units += units

        if total_units == 0:
            return 0.0, 0.0, 'BUY'

        avg_price = total_cost / total_units
        direction = 'BUY' if float(trades[0].get('currentUnits', 0)) > 0 else 'SELL'

        return avg_price, total_units, direction

    def update_dca_tp_to_breakeven(self, symbol: str, trades: list) -> None:
        """Update all DCA trades to have TP at breakeven + small profit.

        When multiple DCAs are triggered, we want to exit at breakeven
        rather than waiting for original TP which may never be reached.
        """
        if len(trades) < 2:
            return  # Only for DCA positions (2+ trades)

        avg_entry, total_units, direction = self.calculate_average_entry(trades)
        pip_value = self.config.get_pip_value(symbol)

        # Set TP at average entry + 2 pips profit (just escape with small win)
        breakeven_tp_pips = 2.0

        if direction == 'BUY':
            new_tp = avg_entry + (breakeven_tp_pips * pip_value)
        else:
            new_tp = avg_entry - (breakeven_tp_pips * pip_value)

        # Check if current price is close to our new TP
        prices = self.client.get_pricing([symbol])
        if prices and symbol in prices:
            current_price = prices[symbol]['bid'] if direction == 'BUY' else prices[symbol]['ask']

            if direction == 'BUY':
                pips_to_tp = (new_tp - current_price) / pip_value
            else:
                pips_to_tp = (current_price - new_tp) / pip_value

            # If within 1 pip of breakeven, close NOW
            if pips_to_tp <= 1.0:
                print(f"  {symbol}: >>> BREAKEVEN REACHED ({pips_to_tp:.1f} pips to TP) - CLOSING <<<")
                self.close_position(symbol, "BREAKEVEN EXIT")
                return

        # Update TP for all trades to breakeven level
        print(f"  {symbol}: DCA Avg Entry={self.format_price(symbol, avg_entry)}, New TP={self.format_price(symbol, new_tp)} (+{breakeven_tp_pips}p)")

        for t in trades:
            trade_id = t.get('id', '')
            current_tp = None
            tp_order = t.get('takeProfitOrder', {})
            if tp_order:
                current_tp = float(tp_order.get('price', 0))

            # Only update if TP is different (more than 1 pip away)
            if current_tp is None or abs(current_tp - new_tp) / pip_value > 1.0:
                success, result = self.trade_executor.modify_trade(
                    trade_id=trade_id,
                    take_profit=new_tp,
                    symbol=symbol
                )
                if success:
                    print(f"       Trade {trade_id}: TP updated to {self.format_price(symbol, new_tp)}")

    def manage_position(self, symbol: str, trades: list) -> None:
        """Manage an open position - check 4H timeout, update P&L, manage DCA."""
        # Check 4H max hold
        if self.check_4h_max_hold(symbol):
            elapsed = datetime.now() - self.position_open_times[symbol]
            print(f"  {symbol}: 4H MAX HOLD reached ({elapsed.total_seconds()/3600:.1f}h)")
            self.close_position(symbol, "4H MAX HOLD")
            return

        # Calculate current P&L from OANDA trades
        total_pnl = sum(float(t.get('unrealizedPL', 0)) for t in trades)
        total_units = sum(abs(float(t.get('currentUnits', 0))) for t in trades)
        num_trades = len(trades)

        # Time remaining
        if symbol in self.position_open_times:
            elapsed = datetime.now() - self.position_open_times[symbol]
            remaining = timedelta(hours=self.max_hold_hours) - elapsed
            remaining_min = max(0, remaining.total_seconds() / 60)
        else:
            remaining_min = 240

        # Calculate average entry for DCA positions
        if num_trades >= 2:
            avg_entry, _, direction = self.calculate_average_entry(trades)
            pip_value = self.config.get_pip_value(symbol)

            # Get current price
            prices = self.client.get_pricing([symbol])
            if prices and symbol in prices:
                current_price = prices[symbol]['bid'] if direction == 'BUY' else prices[symbol]['ask']

                if direction == 'BUY':
                    pips_from_avg = (current_price - avg_entry) / pip_value
                else:
                    pips_from_avg = (avg_entry - current_price) / pip_value

                status = f"DCA x{num_trades} ({total_units:.0f} units) | Avg: {self.format_price(symbol, avg_entry)} | {pips_from_avg:+.1f}p"

                # If in profit from average, close immediately
                if pips_from_avg >= 2.0:
                    print(f"  {symbol}: {status} | P&L: ${total_pnl:+.2f}")
                    print(f"  {symbol}: >>> DCA RECOVERY +{pips_from_avg:.1f} pips - CLOSING <<<")
                    self.close_position(symbol, "DCA RECOVERY")
                    return

                # Update TP to breakeven if we have 2+ DCAs
                self.update_dca_tp_to_breakeven(symbol, trades)
            else:
                status = f"DCA x{num_trades} ({total_units:.0f} units)"
        else:
            status = f"Open ({num_trades} trades, {total_units:.0f} units)"

        pnl_str = f"${total_pnl:+.2f}" if total_pnl != 0 else "$0.00"
        print(f"  {symbol}: {status} | P&L: {pnl_str} | Time left: {remaining_min:.0f}m")

    def check_and_fix_sl_tp(self, trades_by_symbol: dict) -> None:
        """Check and fix incorrect SL/TP on existing positions.

        This fixes issues like JPY pairs having broken SL values (e.g., 493.8 instead of 149.xxx).
        Also fixes trades that have NO SL/TP set at all.
        """
        print("\n[*] Checking SL/TP on existing positions...")

        dca = self.config.dca
        fixes_needed = []

        for symbol, trades in trades_by_symbol.items():
            pip_value = self.config.get_pip_value(symbol)

            for trade in trades:
                trade_id = trade.get('id', '')
                entry_price = float(trade.get('price', 0))
                current_units = float(trade.get('currentUnits', 0))
                direction = 'BUY' if current_units > 0 else 'SELL'

                # Fetch FULL trade details to get SL/TP info
                # The /openTrades endpoint doesn't always return stopLossOrder/takeProfitOrder
                success, full_trade = self.trade_executor.get_trade_details(trade_id)
                if success and full_trade:
                    trade = full_trade  # Use full details

                # Get current SL/TP from trade
                current_sl = None
                current_tp = None

                sl_order = trade.get('stopLossOrder', {})
                if sl_order:
                    current_sl = float(sl_order.get('price', 0))

                tp_order = trade.get('takeProfitOrder', {})
                if tp_order:
                    current_tp = float(tp_order.get('price', 0))

                print(f"  {symbol} (Trade {trade_id}): Entry={self.format_price(symbol, entry_price)}, SL={current_sl}, TP={current_tp}")

                # Calculate correct SL/TP
                sl_pips = dca.initial_sl_pips  # 40 pips
                tp_pips = dca.take_profit_pips  # 10 pips

                if direction == 'BUY':
                    correct_sl = entry_price - (sl_pips * pip_value)
                    correct_tp = entry_price + (tp_pips * pip_value)
                else:
                    correct_sl = entry_price + (sl_pips * pip_value)
                    correct_tp = entry_price - (tp_pips * pip_value)

                # Check if SL needs fixing
                sl_needs_fix = False
                tp_needs_fix = False

                if current_sl is not None:
                    # For JPY pairs, check if SL is way off (more than 10% difference)
                    sl_diff_pct = abs(current_sl - correct_sl) / entry_price
                    if sl_diff_pct > 0.01:  # More than 1% off = broken
                        sl_needs_fix = True
                        print(f"  {symbol} (Trade {trade_id}): SL is BROKEN!")
                        print(f"       Current SL: {current_sl} | Should be: {self.format_price(symbol, correct_sl)}")
                else:
                    sl_needs_fix = True
                    print(f"  {symbol} (Trade {trade_id}): NO SL set!")

                if current_tp is not None:
                    tp_diff_pct = abs(current_tp - correct_tp) / entry_price
                    if tp_diff_pct > 0.01:  # More than 1% off = broken
                        tp_needs_fix = True
                        print(f"  {symbol} (Trade {trade_id}): TP is BROKEN!")
                        print(f"       Current TP: {current_tp} | Should be: {self.format_price(symbol, correct_tp)}")
                else:
                    tp_needs_fix = True
                    print(f"  {symbol} (Trade {trade_id}): NO TP set!")

                if sl_needs_fix or tp_needs_fix:
                    # Always set BOTH SL and TP to ensure consistency
                    fixes_needed.append({
                        'trade_id': trade_id,
                        'symbol': symbol,
                        'correct_sl': correct_sl,  # Always set SL
                        'correct_tp': correct_tp   # Always set TP
                    })

        if not fixes_needed:
            print("  All SL/TP values are correct.")
            return

        # Apply fixes
        print(f"\n  >>> FIXING {len(fixes_needed)} trade(s) <<<")
        for fix in fixes_needed:
            trade_id = fix['trade_id']
            symbol = fix['symbol']
            new_sl = fix['correct_sl']
            new_tp = fix['correct_tp']

            success, result = self.trade_executor.modify_trade(
                trade_id=trade_id,
                stop_loss=new_sl,
                take_profit=new_tp,
                symbol=symbol  # Important for JPY price formatting
            )

            if success:
                fixes_str = []
                if new_sl:
                    fixes_str.append(f"SL: {self.format_price(symbol, new_sl)}")
                if new_tp:
                    fixes_str.append(f"TP: {self.format_price(symbol, new_tp)}")
                print(f"  {symbol} (Trade {trade_id}): FIXED - {', '.join(fixes_str)}")
            else:
                print(f"  {symbol} (Trade {trade_id}): FIX FAILED - {result.get('error', 'Unknown')}")

        print("  SL/TP check complete.")

    def sync_positions_from_oanda(self, trades_by_symbol: dict) -> None:
        """Sync local tracking with OANDA state."""
        # Check for positions closed on OANDA
        for symbol in list(self.position_open_times.keys()):
            if symbol not in trades_by_symbol:
                print(f"  {symbol}: Position closed on OANDA - cleaning up")
                self.cancel_pending_dca_orders(symbol)
                del self.position_open_times[symbol]
                if symbol in self.position_manager.positions:
                    del self.position_manager.positions[symbol]

        # Check for positions opened elsewhere (sync open times)
        for symbol in trades_by_symbol:
            if symbol not in self.position_open_times and symbol in self.symbols:
                # Position exists on OANDA but we don't have open time
                # Use earliest trade time
                trades = trades_by_symbol[symbol]
                earliest = None
                for t in trades:
                    open_time_str = t.get('openTime', '')
                    if open_time_str:
                        try:
                            trade_time = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
                            trade_time = trade_time.replace(tzinfo=None)  # Remove timezone
                            if earliest is None or trade_time < earliest:
                                earliest = trade_time
                        except:
                            pass

                if earliest:
                    self.position_open_times[symbol] = earliest
                else:
                    self.position_open_times[symbol] = datetime.now()

    def run(self, interval: int = 60):
        """Run the trading loop."""
        print("\n" + "=" * 60)
        print("SIMPLE TREND FOLLOWING - NO ML")
        print("=" * 60)

        # Test connection
        success, msg = self.client.test_connection()
        if not success:
            print(f"[X] Connection failed: {msg}")
            return
        print(f"[+] {msg}")

        # Get account equity
        account = self.client.get_account_summary()
        if account and 'account' in account:
            self.equity = float(account['account'].get('NAV', 5000))
            print(f"[+] Equity: ${self.equity:,.2f}")

        # ============================================================
        # CHECK FOR EXISTING POSITIONS ON STARTUP
        # ============================================================
        print(f"\n[*] Checking for existing positions on OANDA...")
        trades_by_symbol = self.get_open_trades()

        if trades_by_symbol:
            print(f"\n>>> FOUND {len(trades_by_symbol)} EXISTING POSITION(S) <<<")
            total_pnl = 0
            for symbol, trades in trades_by_symbol.items():
                pnl = sum(float(t.get('unrealizedPL', 0)) for t in trades)
                total_pnl += pnl
                units = sum(abs(float(t.get('currentUnits', 0))) for t in trades)
                direction = 'BUY' if float(trades[0].get('currentUnits', 0)) > 0 else 'SELL'

                # Get open time from OANDA
                open_time_str = trades[0].get('openTime', '')
                if open_time_str:
                    try:
                        trade_time = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
                        trade_time = trade_time.replace(tzinfo=None)
                        elapsed = datetime.now() - trade_time
                        elapsed_min = elapsed.total_seconds() / 60
                        self.position_open_times[symbol] = trade_time
                    except:
                        elapsed_min = 0
                        self.position_open_times[symbol] = datetime.now()
                else:
                    elapsed_min = 0
                    self.position_open_times[symbol] = datetime.now()

                print(f"  {symbol}: {direction} | {len(trades)} trade(s) | {units:.0f} units | P&L: ${pnl:+.2f} | Age: {elapsed_min:.0f}m")

            print(f"\n  Total P&L: ${total_pnl:+.2f}")
            print("-" * 60)

            # Ask what to do with existing positions
            print("\nOptions:")
            print("  [1] Continue managing existing positions (default)")
            print("  [2] Close all positions and start fresh")
            print("  [3] Cancel and exit")

            choice = input("\nChoice [1/2/3]: ").strip()

            if choice == '2':
                print("\nClosing all existing positions...")
                for symbol in list(trades_by_symbol.keys()):
                    self.close_position(symbol, "USER RESET")
                print("All positions closed. Starting fresh.")
                trades_by_symbol = {}
            elif choice == '3':
                print("Cancelled.")
                return
            else:
                print("\nContinuing with existing positions...")
                # CHECK AND FIX SL/TP ON EXISTING POSITIONS
                self.check_and_fix_sl_tp(trades_by_symbol)
        else:
            print("  No existing positions found.")

        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"DCA Levels: {self.config.dca.max_dca_levels}")
        print(f"Max Hold: {self.max_hold_hours}H")
        print(f"Interval: {interval}s")
        print("-" * 60)

        try:
            while True:
                now = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{now}] Checking {len(self.symbols)} pairs...")

                # Get all open trades from OANDA (one API call)
                trades_by_symbol = self.get_open_trades()

                # Sync local state with OANDA
                self.sync_positions_from_oanda(trades_by_symbol)

                for symbol in self.symbols:
                    # Check if already have position
                    if self.has_position(symbol, trades_by_symbol):
                        self.manage_position(symbol, trades_by_symbol[symbol])
                        continue

                    # No position - get H1 trend and open
                    trend_dir, trend_reason = self.get_h1_trend(symbol)
                    print(f"  {symbol}: {trend_reason}")

                    if trend_dir in ['BUY', 'SELL']:
                        self.open_position(symbol, trend_dir)
                    elif trend_dir == 'NONE':
                        print(f"  {symbol}: No trade - flat/ranging market")
                    else:
                        print(f"  {symbol}: Insufficient data")

                # Wait for next iteration
                print(f"\nSleeping {interval}s...")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[!] Stopped by user")

            # Show final status
            trades_by_symbol = self.get_open_trades()
            if trades_by_symbol:
                print(f"\n{len(trades_by_symbol)} position(s) still open:")
                for symbol, trades in trades_by_symbol.items():
                    total_pnl = sum(float(t.get('unrealizedPL', 0)) for t in trades)
                    print(f"  {symbol}: {len(trades)} trade(s) | P&L: ${total_pnl:+.2f}")

                response = input("\nClose all positions? [y/N]: ").strip().lower()
                if response == 'y':
                    for symbol in trades_by_symbol:
                        self.close_position(symbol, "USER EXIT")
                    print("All positions closed.")


def main():
    print("\n" + "=" * 60)
    print("SIMPLE TREND FOLLOWING SYSTEM")
    print("NO ML - Just H1 EMA50 vs EMA200")
    print("=" * 60)

    config = load_config()

    print(f"\nSymbols: {', '.join(config.symbols)}")
    print(f"DCA Levels: {config.dca.max_dca_levels}")
    print(f"Initial SL: {config.dca.initial_sl_pips}p")
    print(f"Take Profit: {config.dca.take_profit_pips}p")

    print("\n" + "-" * 60)
    response = input("Start trading? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return

    engine = SimpleTrendEngine(config)
    engine.run(interval=60)


if __name__ == '__main__':
    main()
