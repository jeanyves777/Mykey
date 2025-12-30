"""
Paper Trading Engine for Forex ML System
========================================

Main engine for paper trading with OANDA.
"""

import warnings
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import os
import csv
import sys

# Suppress all ML and pandas warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trading_system.Forex_Trading.Forex_Trading_ML_Version.data.oanda_client import OandaClient
from trading_system.Forex_Trading.Forex_Trading_ML_Version.data.data_loader import ForexDataLoader
from trading_system.Forex_Trading.Forex_Trading_ML_Version.features.feature_engineer import FeatureEngineer
from trading_system.Forex_Trading.Forex_Trading_ML_Version.ensemble.ensemble_voting import EnsembleVotingSystem
from trading_system.Forex_Trading.Forex_Trading_ML_Version.risk_management.risk_manager import RiskManager
from trading_system.Forex_Trading.Forex_Trading_ML_Version.risk_management.position_manager import PositionManager
from trading_system.Forex_Trading.Forex_Trading_ML_Version.risk_management.trade_executor import TradeExecutor
from trading_system.Forex_Trading.Forex_Trading_ML_Version.risk_management.trend_filter import TrendFilter
from trading_system.Forex_Trading.Forex_Trading_ML_Version.engine.trading_strategy import MLTradingStrategy
from trading_system.Forex_Trading.Forex_Trading_ML_Version.engine.momentum_signal import MasterMomentumSignal


class PaperTradingEngine:
    """Paper trading engine for ML Forex system."""

    def __init__(self, config, ensemble: EnsembleVotingSystem,
                 feature_engineer: FeatureEngineer):
        """
        Initialize paper trading engine.

        Args:
            config: TradingConfig object
            ensemble: Trained EnsembleVotingSystem
            feature_engineer: Configured FeatureEngineer
        """
        self.config = config
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer

        # Initialize OANDA client
        self.client = OandaClient(
            api_key=config.oanda.api_key,
            account_id=config.oanda.account_id,
            environment=config.oanda.environment
        )

        # Initialize components
        self.data_loader = ForexDataLoader(self.client, config)
        self.strategy = MLTradingStrategy(ensemble, feature_engineer, config)
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager(config)
        self.trade_executor = TradeExecutor(self.client, config)
        self.trend_filter = TrendFilter(config)  # Smart DCA trend filtering
        self.momentum_signal = MasterMomentumSignal(config)  # Fast momentum entry

        # Trading state
        self.is_running = False
        self.symbols = config.symbols
        self.timeframe = 'M5'  # 5-minute bars
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        self.last_bar_time: Dict[str, datetime] = {}

        # Display settings
        self.verbose = False  # Set to True for detailed logging
        self.last_status_time = None

        # Logging
        self.log_dir = config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.trade_log_file = os.path.join(self.log_dir, f'ml_trades_{datetime.now().strftime("%Y%m%d")}.csv')

    def initialize(self) -> bool:
        """Initialize engine and load initial data."""
        print("\n" + "=" * 60)
        print("FOREX ML PAPER TRADING ENGINE")
        print("=" * 60)

        # Test connection
        success, message = self.client.test_connection()
        if not success:
            print(f"[X] Connection failed: {message}")
            return False
        print(f"[+] {message}")

        # Get account info
        account_info = self.client.get_account_summary()
        if account_info and 'account' in account_info:
            account = account_info['account']
            equity = float(account.get('NAV', 100000))
            self.risk_manager.set_initial_equity(equity)
            print(f"[+] Account equity: ${equity:,.2f}")

        # Load initial data for all symbols
        print(f"\nLoading initial data for {len(self.symbols)} symbols...")
        for symbol in self.symbols:
            df = self.data_loader.get_latest_bars(symbol, count=500, granularity=self.timeframe)
            if df is not None and len(df) > 0:
                self.data_buffer[symbol] = df
                self.last_bar_time[symbol] = df.index[-1]
                print(f"  [+] {symbol}: {len(df)} bars loaded")
            else:
                print(f"  [X] {symbol}: Failed to load data")

        # Check ensemble is trained
        model_status = self.ensemble.get_model_status()
        trained = sum(model_status.values())
        print(f"\n[+] Ensemble: {trained}/{len(model_status)} models trained")

        if trained == 0:
            print("[X] No models trained! Please train the ensemble first.")
            return False

        # Sync existing OANDA positions
        self._sync_existing_positions()

        print("\n" + "=" * 60)
        return True

    def _sync_existing_positions(self) -> None:
        """Sync LIVE open positions from OANDA account - FRESH DATA ONLY.

        IMPORTANT: Consolidates multiple trades for the same symbol into ONE position.
        This properly tracks DCA positions where OANDA has multiple trades.
        """
        print(f"\nFetching LIVE positions from OANDA...")

        # ALWAYS fetch fresh data from OANDA API - no caching
        success, trades = self.trade_executor.get_open_trades()
        if not success:
            print("  [!] Could not fetch open trades from OANDA")
            return

        # Clear any stale local positions first
        self.position_manager.positions.clear()

        if not trades:
            print("  [*] No open positions on OANDA")
            return

        # Check if DCA is enabled
        dca_enabled = self.config.dca.enabled if hasattr(self.config, 'dca') else False

        # Group trades by symbol to consolidate DCA positions
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get('instrument', '')
            if symbol and symbol in self.symbols:
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)

        synced = 0
        positions_needing_dca = []

        # Process each symbol (consolidating multiple trades as DCA)
        for symbol, symbol_trades in trades_by_symbol.items():
            try:
                # Sort trades by entry price to determine original vs DCA entries
                # For SELL: highest entry price = original, lower = DCAs
                # For BUY: lowest entry price = original, higher = DCAs
                first_trade = symbol_trades[0]
                direction = 'BUY' if float(first_trade.get('currentUnits', 0)) > 0 else 'SELL'

                if direction == 'SELL':
                    symbol_trades.sort(key=lambda t: float(t.get('price', 0)), reverse=True)
                else:
                    symbol_trades.sort(key=lambda t: float(t.get('price', 0)))

                # First trade is the ORIGINAL entry
                original_trade = symbol_trades[0]
                original_entry = float(original_trade.get('price', 0))
                original_units = abs(float(original_trade.get('currentUnits', 0)))
                trade_id = original_trade.get('id', '')

                # Calculate TOTAL units and WEIGHTED AVERAGE entry
                total_units = 0.0
                total_cost = 0.0
                total_pnl = 0.0

                for t in symbol_trades:
                    t_units = abs(float(t.get('currentUnits', 0)))
                    t_price = float(t.get('price', 0))
                    t_pnl = float(t.get('unrealizedPL', 0))
                    total_units += t_units
                    total_cost += t_units * t_price
                    total_pnl += t_pnl

                avg_entry = total_cost / total_units if total_units > 0 else original_entry

                # DCA level = number of trades - 1 (first is original, rest are DCAs)
                dca_level = len(symbol_trades) - 1

                # Get SL/TP from first trade (all should have same)
                stop_loss = 0.0
                take_profit = 0.0
                if 'stopLossOrder' in original_trade and original_trade['stopLossOrder']:
                    stop_loss = float(original_trade['stopLossOrder'].get('price', 0))
                if 'takeProfitOrder' in original_trade and original_trade['takeProfitOrder']:
                    take_profit = float(original_trade['takeProfitOrder'].get('price', 0))

                pip_value = self.config.get_pip_value(symbol)

                # If DCA enabled, calculate proper SL/TP
                if dca_enabled and getattr(self.config.dca, 'use_pip_based_dca', False):
                    # SL from ORIGINAL entry
                    sl_pips = self.config.dca.sl_after_dca_pips if dca_level > 0 else self.config.dca.initial_sl_pips
                    # TP from AVERAGE entry with dynamic TP based on DCA level
                    tp_pips = self.config.dca.get_tp_for_dca_level(dca_level)

                    if direction == 'BUY':
                        stop_loss = original_entry - (sl_pips * pip_value)
                        take_profit = avg_entry + (tp_pips * pip_value)
                    else:
                        stop_loss = original_entry + (sl_pips * pip_value)
                        take_profit = avg_entry - (tp_pips * pip_value)

                # Create the consolidated position
                self.position_manager.open_position(
                    symbol=symbol,
                    direction=direction,
                    units=original_units,  # Original units
                    entry_price=original_entry,  # Original entry
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trade_id=trade_id
                )

                # Update position with DCA info
                position = self.position_manager.positions[symbol]
                position.unrealized_pnl = total_pnl
                position.total_units = total_units
                position.avg_entry_price = avg_entry
                position.dca_level = dca_level
                position.dca_active = dca_level > 0
                # Track last DCA time for Smart Recovery Exit
                if dca_level > 0:
                    from datetime import datetime
                    position.last_dca_time = datetime.now()

                # Add DCA entries for tracking
                for i, t in enumerate(symbol_trades[1:], 1):  # Skip original
                    from trading_system.Forex_Trading.Forex_Trading_ML_Version.risk_management.position_manager import DCAEntry
                    from datetime import datetime
                    dca_entry = DCAEntry(
                        level=i,
                        units=abs(float(t.get('currentUnits', 0))),
                        price=float(t.get('price', 0)),
                        time=datetime.now(),
                        trade_id=t.get('id', '')
                    )
                    position.dca_entries.append(dca_entry)

                # Get current price
                prices = self.client.get_pricing([symbol])
                if prices and symbol in prices:
                    if direction == 'BUY':
                        current_price = prices[symbol]['bid']
                    else:
                        current_price = prices[symbol]['ask']

                    position.current_price = current_price
                    position.highest_price = max(original_entry, current_price)
                    position.lowest_price = min(original_entry, current_price)

                    # Calculate profit from avg entry
                    if direction == 'BUY':
                        profit_pips = (current_price - avg_entry) / pip_value
                    else:
                        profit_pips = (avg_entry - current_price) / pip_value

                synced += 1
                if dca_level > 0:
                    print(f"  [+] {symbol}: {direction} {total_units:.0f} units (DCA{dca_level})")
                    print(f"      Original: {original_units:.0f} @ {original_entry:.5f} | Avg: {avg_entry:.5f}")
                    print(f"      Profit: {profit_pips:+.1f} pips | P&L: ${total_pnl:+.2f}")
                else:
                    print(f"  [+] {symbol}: {direction} {total_units:.0f} units @ {original_entry:.5f} | "
                          f"P&L: ${total_pnl:+.2f}")

            except Exception as e:
                print(f"  [!] Error syncing trade for {symbol}: {e}")

        if synced > 0:
            print(f"\n  [+] Synced {synced} existing position(s)")

        # Check DCA for underwater positions
        if dca_enabled and positions_needing_dca:
            print(f"\n  [DCA] Checking {len(positions_needing_dca)} underwater position(s)...")
            for symbol, position, price_drop, pips_loss in positions_needing_dca:
                self._check_and_apply_initial_dca(symbol, position, price_drop, pips_loss)

        # CRITICAL: Check and fix missing SL/TP on all OANDA trades
        self._check_and_fix_missing_sl_tp(trades)

        # CRITICAL: Check and place missing DCA pending orders for existing positions
        # ONLY if use_pending_orders=True (NOT for Smart DCA which uses market orders with trend check)
        if dca_enabled and getattr(self.config.dca, 'use_pip_based_dca', False) and getattr(self.config.dca, 'use_pending_orders', False):
            self._check_and_place_missing_dca_orders(trades)
        elif dca_enabled and getattr(self.config.dca, 'use_pip_based_dca', False):
            print(f"\n  [SMART DCA] Pending orders DISABLED - Using trend-based market orders instead")
            # Cancel any existing pending DCA orders that shouldn't be there
            self._cancel_unwanted_pending_orders()

        # CRITICAL: Check and apply trailing stops for positions in profit
        if dca_enabled and getattr(self.config.dca, 'use_pip_based_dca', False):
            self._check_and_apply_trailing_stops(trades)

    def _check_and_place_missing_dca_orders(self, oanda_trades: list) -> None:
        """
        Check existing positions for missing DCA pending orders and place them.

        When the system restarts, existing positions may not have their DCA limit
        orders staged on OANDA. This method:
        1. Gets all pending orders from OANDA
        2. For each position, checks if DCA orders are missing
        3. Places missing DCA limit orders

        Args:
            oanda_trades: List of open trades from OANDA
        """
        print(f"\n  [DCA ORDERS] Checking for missing pending DCA orders...")

        dca = self.config.dca

        # Get all pending orders from OANDA
        success, all_pending_orders = self.trade_executor.get_pending_orders()
        if not success:
            print("    [!] Could not fetch pending orders from OANDA")
            all_pending_orders = []

        print(f"    Found {len(all_pending_orders)} total pending order(s) on OANDA")

        # Group pending orders by symbol
        pending_by_symbol = {}
        for order in all_pending_orders:
            symbol = order.get('instrument', '')
            if symbol:
                if symbol not in pending_by_symbol:
                    pending_by_symbol[symbol] = []
                pending_by_symbol[symbol].append(order)

        # Count trades per symbol to determine DCA level
        trades_by_symbol = {}
        for trade in oanda_trades:
            symbol = trade.get('instrument', '')
            if symbol and symbol in self.symbols:
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)

        positions_fixed = 0

        for symbol, position in list(self.position_manager.positions.items()):
            # Count existing trades for this symbol
            symbol_trades = trades_by_symbol.get(symbol, [])
            num_trades = len(symbol_trades)

            if num_trades == 0:
                print(f"    {symbol}: SKIP - no trades found on OANDA")
                continue

            # Count pending DCA orders for this symbol
            symbol_pending = pending_by_symbol.get(symbol, [])
            num_pending = len(symbol_pending)

            # Expected: 1 original trade + 4 pending DCA orders (if no DCAs filled)
            # If we have N trades, we should have (4 - (N-1)) pending orders
            expected_pending = dca.max_dca_levels - (num_trades - 1)
            expected_pending = max(0, expected_pending)

            pip_value = self.config.get_pip_value(symbol)

            # Get the ORIGINAL entry price (first trade opened)
            # Sort trades by ID to get the oldest one
            sorted_trades = sorted(symbol_trades, key=lambda t: int(t.get('id', '0')))
            original_trade = sorted_trades[0]
            original_entry_price = float(original_trade.get('price', position.entry_price))

            # Determine which DCA levels are already filled based on trade prices
            filled_levels = set()
            for trade in sorted_trades[1:]:  # Skip original trade
                trade_price = float(trade.get('price', 0))
                if position.direction == 'BUY':
                    pips_from_entry = (original_entry_price - trade_price) / pip_value
                else:
                    pips_from_entry = (trade_price - original_entry_price) / pip_value

                # Find which DCA level this trade corresponds to
                for level in range(1, dca.max_dca_levels + 1):
                    trigger_pips = dca.get_dca_trigger_pips(level)
                    if abs(pips_from_entry - trigger_pips) < 2.0:  # Within 2 pip tolerance for filled trades
                        filled_levels.add(level)
                        break

            # Determine which levels have pending orders
            pending_levels = set()
            for order in symbol_pending:
                price = float(order.get('price', 0))
                if position.direction == 'BUY':
                    pips_from_entry = (original_entry_price - price) / pip_value
                else:
                    pips_from_entry = (price - original_entry_price) / pip_value

                for level in range(1, dca.max_dca_levels + 1):
                    trigger_pips = dca.get_dca_trigger_pips(level)
                    if abs(pips_from_entry - trigger_pips) < 1.0:  # Within 1 pip tolerance
                        pending_levels.add(level)
                        # Track locally
                        order_id = order.get('id', '')
                        units = abs(float(order.get('units', 0)))
                        position.add_pending_dca_order(level, units, price, order_id)
                        break

            # Calculate missing levels
            missing_levels = []
            for level in range(1, dca.max_dca_levels + 1):
                if level not in filled_levels and level not in pending_levels:
                    missing_levels.append(level)

            if not missing_levels:
                # All DCA levels are accounted for
                print(f"    {symbol}: OK - {num_trades} trade(s), {num_pending} pending | Filled: {sorted(filled_levels) if filled_levels else 'none'}")
                continue

            # Missing pending orders - need to place them
            print(f"    {symbol}: MISSING DCA orders!")
            print(f"      Trades: {num_trades} | Pending: {num_pending} | Expected pending: {expected_pending}")
            print(f"      Filled levels: {sorted(filled_levels) if filled_levels else 'none'}")
            print(f"      Pending levels: {sorted(pending_levels) if pending_levels else 'none'}")
            print(f"      Missing levels: {missing_levels}")
            print(f"      Original entry: {original_entry_price:.5f}")

            # Get base units from original trade
            base_units = abs(float(original_trade.get('currentUnits', 0)))
            if base_units == 0:
                base_units = position.units

            print(f"      Base units: {base_units:.0f}")

            # Place missing orders
            self._place_specific_dca_orders(symbol, position.direction, original_entry_price, base_units, missing_levels)
            positions_fixed += 1

        if positions_fixed > 0:
            print(f"\n  >>> Fixed DCA orders for {positions_fixed} position(s)")
        else:
            print(f"    >>> All positions have their DCA orders")

    def _place_specific_dca_orders(self, symbol: str, direction: str, entry_price: float,
                                   base_units: float, levels: list) -> None:
        """
        Place DCA limit orders for specific levels.

        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
            entry_price: Original entry price
            base_units: Base position units
            levels: List of DCA levels to place (e.g., [2, 3, 4])
        """
        dca = self.config.dca
        pip_value = self.config.get_pip_value(symbol)
        position = self.position_manager.get_position(symbol)

        if not position:
            return

        # Calculate SL/TP for DCA orders
        sl_pips = dca.sl_after_dca_pips
        tp_pips = dca.take_profit_pips

        if direction == 'BUY':
            dca_stop_loss = entry_price - (sl_pips * pip_value)
            dca_take_profit = entry_price + (tp_pips * pip_value)
        else:
            dca_stop_loss = entry_price + (sl_pips * pip_value)
            dca_take_profit = entry_price - (tp_pips * pip_value)

        for level in levels:
            trigger_pips = dca.get_dca_trigger_pips(level)
            multiplier = dca.get_dca_multiplier(level)

            # Calculate DCA units
            dca_units = self.position_manager.get_dca_units(symbol, base_units, level, dca)
            if dca_units <= 0:
                print(f"      DCA{level}: SKIPPED (position cap)")
                continue

            # Calculate trigger price
            if direction == 'BUY':
                trigger_price = entry_price - (trigger_pips * pip_value)
            else:
                trigger_price = entry_price + (trigger_pips * pip_value)

            # Place the limit order
            success, result = self.trade_executor.place_limit_order(
                symbol=symbol,
                units=int(dca_units),
                direction=direction,
                price=trigger_price,
                stop_loss=dca_stop_loss,
                take_profit=dca_take_profit
            )

            if success:
                order_id = result.get('order_id', '')
                order_type = result.get('type', 'LIMIT')

                if order_type == 'LIMIT_FILLED':
                    print(f"      DCA{level}: FILLED immediately @ {trigger_price:.5f}")
                    position.add_dca_entry(level, dca_units, trigger_price, result.get('trade_id', ''))
                else:
                    position.add_pending_dca_order(level, dca_units, trigger_price, order_id)
                    print(f"      DCA{level}: PLACED @ {trigger_price:.5f} ({trigger_pips:.0f}p) | Order: {order_id}")
            else:
                error = result.get('error', 'Unknown')
                print(f"      DCA{level}: FAILED - {error}")

    def _check_and_apply_trailing_stops(self, oanda_trades: list) -> None:
        """
        Check existing positions for trailing stop opportunities and apply them.

        When system restarts, positions that are in profit may need their
        trailing stops set. This checks each position and applies an actual
        OANDA Trailing Stop Order if conditions are met (5+ pips profit).

        OANDA Trailing Stop Order automatically moves the SL as price moves
        in your favor - much better than manual SL adjustments.

        Args:
            oanda_trades: List of open trades from OANDA
        """
        print(f"\n  [TRAILING] Checking for trailing stop opportunities...")

        dca = self.config.dca
        trailing_pips = max(dca.trailing_stop_pips, 5.0)  # OANDA minimum is 5 pips
        activation_pips = 5.0  # OANDA minimum

        positions_updated = 0

        for symbol, position in list(self.position_manager.positions.items()):
            # Get current price from OANDA
            prices = self.client.get_pricing([symbol])
            if not prices or symbol not in prices:
                continue

            pip_value = self.config.get_pip_value(symbol)

            if position.direction == 'BUY':
                current_price = prices[symbol]['bid']
                # Update highest price
                position.highest_price = max(position.highest_price, current_price)
                profit_pips = (current_price - position.entry_price) / pip_value
            else:
                current_price = prices[symbol]['ask']
                # Update lowest price
                position.lowest_price = min(position.lowest_price, current_price)
                profit_pips = (position.entry_price - current_price) / pip_value

            # Check if trailing should activate
            if profit_pips >= activation_pips:
                # Check if any trade for this symbol already has trailing stop set
                has_trailing = False
                for trade in oanda_trades:
                    if trade.get('instrument') == symbol:
                        if 'trailingStopLossOrder' in trade and trade['trailingStopLossOrder']:
                            has_trailing = True
                            break

                if has_trailing:
                    print(f"    {symbol}: OANDA Trailing Stop already set ({profit_pips:+.1f}p profit)")
                    continue

                # SET ACTUAL OANDA TRAILING STOP ORDER
                print(f"    {symbol}: SETTING OANDA TRAILING STOP!")
                print(f"      Current Profit: {profit_pips:+.1f} pips")
                print(f"      Trailing Distance: {trailing_pips:.0f} pips")
                print(f"      >>> Sending to OANDA...")

                # Set trailing stop on ALL trades for this symbol
                self._set_trailing_stop_all_trades(symbol, trailing_pips)
                positions_updated += 1
            else:
                print(f"    {symbol}: Profit {profit_pips:+.1f}p < {activation_pips}p activation threshold")

        if positions_updated > 0:
            print(f"\n  >>> Set OANDA trailing stops for {positions_updated} position(s)")
        else:
            print(f"    >>> No trailing stop updates needed")

    def _check_and_apply_initial_dca(self, symbol: str, position, price_drop: float, pips_loss: float = 0) -> None:
        """Check if existing position needs DCA and apply levels that should have triggered."""
        dca = self.config.dca
        use_pips = getattr(dca, 'use_pip_based_dca', False)

        # For initial sync, check how many levels SHOULD have triggered
        triggered_levels = 0

        if use_pips:
            # PIP-BASED: Check trigger levels at 3, 6, 10, 15 pips
            if pips_loss >= dca.get_dca_trigger_pips(1):  # 3 pips
                triggered_levels = 1
                if pips_loss >= dca.get_dca_trigger_pips(2):  # 6 pips
                    triggered_levels = 2
                    if pips_loss >= dca.get_dca_trigger_pips(3):  # 10 pips
                        triggered_levels = 3
                        if pips_loss >= dca.get_dca_trigger_pips(4):  # 15 pips
                            triggered_levels = 4

            if triggered_levels == 0:
                print(f"    {symbol}: Loss {pips_loss:.1f} pips < {dca.get_dca_trigger_pips(1):.0f} pips - no DCA triggered yet")
                return

            print(f"    {symbol}: Loss {pips_loss:.1f} pips - triggering {triggered_levels} DCA level(s)")
        else:
            # PERCENTAGE-BASED: Original logic
            dca_levels = dca.get_dca_levels()

            if price_drop >= dca_levels[0].trigger_pct:  # 1.4%
                triggered_levels = 1
                if price_drop >= dca_levels[0].trigger_pct + dca_levels[1].trigger_pct:  # 2.9%
                    triggered_levels = 2
                    if price_drop >= dca_levels[0].trigger_pct + dca_levels[1].trigger_pct + dca_levels[2].trigger_pct:  # 5.4%
                        triggered_levels = 3
                        if price_drop >= dca_levels[0].trigger_pct + dca_levels[1].trigger_pct + dca_levels[2].trigger_pct + dca_levels[3].trigger_pct:  # 7.4%
                            triggered_levels = 4

            if triggered_levels == 0:
                print(f"    {symbol}: Price drop {price_drop:.2%} < {dca_levels[0].trigger_pct:.1%} - no DCA triggered yet")
                return

            print(f"    {symbol}: Price drop {price_drop:.2%} - triggering {triggered_levels} DCA level(s)")

        # Execute each DCA level that should have triggered
        for level in range(1, triggered_levels + 1):
            if position.dca_level >= level:
                continue  # Already at or past this level

            self._execute_dca_entry(symbol, position, level)

    def _check_and_fix_missing_sl_tp(self, trades: list) -> None:
        """
        Check all OANDA trades for missing or mismatched SL/TP and send corrections.

        CRITICAL FOR DCA: All trades for the same symbol must have:
        - SAME SL: Based on ORIGINAL entry price (oldest trade)
        - SAME TP: Based on AVERAGE entry price with dynamic TP reduction

        This ensures when SL hits, ALL trades close together.

        Args:
            trades: List of OANDA trade objects
        """
        if not trades:
            return

        print(f"\n  [SL/TP CHECK] Verifying SL/TP on {len(trades)} OANDA trade(s)...")

        dca_enabled = self.config.dca.enabled if hasattr(self.config, 'dca') else False
        trades_fixed = 0
        trades_ok = 0

        # GROUP TRADES BY SYMBOL for DCA handling
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get('instrument', '')
            if symbol and symbol in self.symbols:
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)

        for symbol, symbol_trades in trades_by_symbol.items():
            try:
                pip_value = self.config.get_pip_value(symbol)
                num_trades = len(symbol_trades)

                # Sort by trade ID (oldest first = original entry)
                symbol_trades.sort(key=lambda t: int(t.get('id', 0)))

                # Get ORIGINAL entry price (first/oldest trade)
                original_trade = symbol_trades[0]
                original_entry = float(original_trade.get('price', 0))
                units = float(original_trade.get('currentUnits', 0))
                direction = 'BUY' if units > 0 else 'SELL'

                # Calculate AVERAGE entry price (for TP)
                total_units = 0
                weighted_price_sum = 0
                for trade in symbol_trades:
                    trade_units = abs(float(trade.get('currentUnits', 0)))
                    trade_price = float(trade.get('price', 0))
                    total_units += trade_units
                    weighted_price_sum += trade_units * trade_price
                avg_entry = weighted_price_sum / total_units if total_units > 0 else original_entry

                # DCA level = number of trades - 1 (original entry is level 0)
                dca_level = num_trades - 1

                # Calculate what SL/TP SHOULD be for ALL trades
                if dca_enabled and getattr(self.config.dca, 'use_pip_based_dca', False):
                    # SL from ORIGINAL ENTRY (same for all trades)
                    if dca_level > 0:
                        sl_pips = self.config.dca.sl_after_dca_pips  # 55 pips after DCA
                    else:
                        sl_pips = self.config.dca.initial_sl_pips  # 25 pips initial

                    # TP from AVERAGE ENTRY with dynamic reduction
                    if hasattr(self.config.dca, 'get_tp_for_dca_level'):
                        tp_pips = self.config.dca.get_tp_for_dca_level(dca_level)
                    else:
                        tp_pips = self.config.dca.dca_profit_target_pips

                    if direction == 'BUY':
                        expected_sl = original_entry - (sl_pips * pip_value)
                        expected_tp = avg_entry + (tp_pips * pip_value)
                    else:
                        expected_sl = original_entry + (sl_pips * pip_value)
                        expected_tp = avg_entry - (tp_pips * pip_value)
                elif dca_enabled:
                    # Percentage-based DCA SL/TP
                    if direction == 'BUY':
                        expected_sl = original_entry * (1 - self.config.dca.sl_after_dca_pct)
                        expected_tp = avg_entry * (1 + self.config.dca.dca_profit_target_pct)
                    else:
                        expected_sl = original_entry * (1 + self.config.dca.sl_after_dca_pct)
                        expected_tp = avg_entry * (1 - self.config.dca.dca_profit_target_pct)
                else:
                    # Non-DCA: Use default SL/TP
                    sl_pips = self.config.risk.default_stop_loss_pips
                    tp_pips = self.config.risk.default_take_profit_pips
                    if direction == 'BUY':
                        expected_sl = original_entry - (sl_pips * pip_value)
                        expected_tp = original_entry + (tp_pips * pip_value)
                    else:
                        expected_sl = original_entry + (sl_pips * pip_value)
                        expected_tp = original_entry - (tp_pips * pip_value)

                # Check and fix ALL trades for this symbol
                for trade in symbol_trades:
                    trade_id = trade.get('id', '')
                    if not trade_id:
                        continue

                    has_sl = 'stopLossOrder' in trade and trade['stopLossOrder']
                    has_tp = 'takeProfitOrder' in trade and trade['takeProfitOrder']
                    oanda_sl = float(trade['stopLossOrder'].get('price', 0)) if has_sl else 0
                    oanda_tp = float(trade['takeProfitOrder'].get('price', 0)) if has_tp else 0

                    # Check if missing OR different from expected
                    sl_missing = oanda_sl == 0
                    tp_missing = oanda_tp == 0
                    sl_mismatch = abs(oanda_sl - expected_sl) > (0.5 * pip_value) if oanda_sl > 0 else False
                    tp_mismatch = abs(oanda_tp - expected_tp) > (0.5 * pip_value) if oanda_tp > 0 else False

                    needs_fix = sl_missing or tp_missing or sl_mismatch or tp_mismatch

                    if needs_fix:
                        success, result = self.trade_executor.modify_trade(
                            trade_id=trade_id,
                            stop_loss=expected_sl,
                            take_profit=expected_tp
                        )

                        if success:
                            trades_fixed += 1
                            fix_reason = []
                            if sl_missing:
                                fix_reason.append("SL missing")
                            elif sl_mismatch:
                                fix_reason.append(f"SL mismatch ({oanda_sl:.5f}→{expected_sl:.5f})")
                            if tp_missing:
                                fix_reason.append("TP missing")
                            elif tp_mismatch:
                                fix_reason.append(f"TP mismatch ({oanda_tp:.5f}→{expected_tp:.5f})")
                            print(f"    [FIXED] {symbol} Trade {trade_id}: {', '.join(fix_reason)}")
                        else:
                            error = result.get('error', 'Unknown')
                            print(f"    [!] Failed to fix {symbol} trade {trade_id}: {error}")
                    else:
                        trades_ok += 1

                # Show summary for DCA positions
                if num_trades > 1:
                    print(f"    [{symbol}] DCA{dca_level}: {num_trades} trades | "
                          f"Orig: {original_entry:.5f} | Avg: {avg_entry:.5f}")
                    print(f"           SL: {expected_sl:.5f} ({sl_pips:.0f}p from orig) | "
                          f"TP: {expected_tp:.5f} ({tp_pips:.0f}p from avg)")

            except Exception as e:
                print(f"    [!] Error checking {symbol} trades: {e}")

        if trades_fixed > 0:
            print(f"\n    >>> Fixed {trades_fixed} trade(s) with missing/mismatched SL/TP")
        if trades_ok > 0:
            print(f"    >>> {trades_ok} trade(s) already have correct SL/TP")

    def run(self, interval_seconds: int = 60) -> None:
        """
        Run the paper trading loop.

        Args:
            interval_seconds: Seconds between iterations
        """
        self.is_running = True
        iteration = 0

        print(f"\n[>] Starting ML paper trading (interval: {interval_seconds}s)")
        print("    Press Ctrl+C to stop")
        print("-" * 60)

        try:
            while self.is_running:
                iteration += 1
                start_time = time.time()

                # Update data and check signals (silent)
                self._trading_iteration()

                # Print compact status line (overwrites previous)
                self._print_compact_status(iteration)

                # Sleep until next iteration
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n[!] Stopping paper trading...")
            self.is_running = False

        # Final summary
        self._print_final_summary()

    def _trading_iteration(self) -> None:
        """Execute one trading iteration - ML SCANNING IS PRIORITY."""
        from datetime import datetime
        now = datetime.now().strftime('%H:%M:%S')

        # Fetch LIVE data from OANDA ONCE per iteration
        account_info = self.client.get_account_summary()
        if account_info and 'account' in account_info:
            equity = float(account_info['account'].get('NAV', self.risk_manager.current_equity))
            self.risk_manager.set_equity(equity)

        # Fetch LIVE positions from OANDA ONCE (cache for this iteration)
        success, self._live_oanda_trades = self.trade_executor.get_open_trades()
        if not success:
            self._live_oanda_trades = []

        open_count = len(self._live_oanda_trades) if self._live_oanda_trades else 0

        # ML SCANNING HEADER
        print(f"\n{'='*60}")
        print(f"[{now}] ML SCANNING {len(self.symbols)} PAIRS | Equity: ${self.risk_manager.current_equity:,.2f} | Open: {open_count}")
        print(f"{'='*60}")

        # PRIORITY: ML SCANNING - Process each symbol for trading opportunities
        trades_opened = False
        for symbol in self.symbols:
            try:
                opened = self._process_symbol(symbol)
                if opened:
                    trades_opened = True
            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")

        # If any trades were opened, REFRESH the OANDA trades cache before sync
        # This prevents newly opened trades from being detected as "closed"
        if trades_opened:
            import time
            time.sleep(0.5)  # Small delay to let OANDA process the orders
            success, self._live_oanda_trades = self.trade_executor.get_open_trades()
            if not success:
                self._live_oanda_trades = []

        # Update positions with fresh OANDA data
        self._update_positions()

    def _process_symbol(self, symbol: str) -> bool:
        """
        Process trading logic for a symbol.

        NEW STRATEGY - TREND FOLLOWING (NO SIGNAL WAITING):
        1. Check H1 trend direction (EMA50 vs EMA200)
        2. If no position: Open IMMEDIATELY in trend direction
        3. If position closed: Reopen IMMEDIATELY in trend direction
        4. Let DCA handle any pullbacks

        Returns:
            bool: True if a new trade was opened, False otherwise
        """
        # Fetch M5 data for buffer
        df = self.data_loader.get_latest_bars(symbol, count=500, granularity=self.timeframe)
        if df is None or len(df) < 50:
            print(f"  {symbol}: Insufficient M5 data ({len(df) if df is not None else 0} bars)")
            return False

        self.data_buffer[symbol] = df
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0

        # Check if position exists on OANDA
        oanda_trades = getattr(self, '_live_oanda_trades', [])
        has_position = any(t.get('instrument') == symbol for t in oanda_trades)

        if not has_position:
            has_position = self.position_manager.has_position(symbol)

        if has_position:
            # Position exists - just manage it (DCA, trailing, etc. handled elsewhere)
            position = self.position_manager.get_position(symbol)
            if position is None:
                for trade in oanda_trades:
                    if trade.get('instrument') == symbol:
                        units = float(trade.get('currentUnits', 0))
                        direction = 'BUY' if units > 0 else 'SELL'
                        print(f"  {symbol}: POSITION OPEN ({direction}) @ {current_price:.5f} - managing...")
                        return False
                return False
            print(f"  {symbol}: POSITION OPEN ({position.direction}) @ {current_price:.5f}")
            return False
        else:
            # ============================================================
            # NO POSITION - GET H1 TREND AND OPEN IMMEDIATELY
            # ============================================================
            df_h1 = self.data_loader.get_latest_bars(symbol, count=250, granularity='H1')
            trend_direction, trend_reason = self.trend_filter.check_higher_timeframe_trend(df_h1)

            print(f"  {symbol}: {trend_reason}")

            if trend_direction == 'NONE':
                print(f"  {symbol}: NO TRADE - Market flat/consolidating")
                return False

            if trend_direction == 'BOTH':
                # Insufficient H1 data - use M5 EMA as fallback
                ema_fast = df['close'].ewm(span=8, adjust=False).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                trend_direction = 'BUY' if ema_fast > ema_slow else 'SELL'
                print(f"  {symbol}: M5 fallback trend: {trend_direction}")

            # Check risk limits
            can_trade, risk_reason = self.risk_manager.can_trade(
                self.position_manager.get_position_count()
            )

            if not can_trade:
                print(f"  {symbol}: Trade BLOCKED - {risk_reason}")
                return False

            # OPEN IMMEDIATELY IN TREND DIRECTION
            print(f"  {symbol}: >>> OPENING {trend_direction} (H1 trend) <<<")
            success = self._open_position(symbol, trend_direction, df, 0.80, f"H1 TREND: {trend_reason}")
            return success

    def _open_position(self, symbol: str, direction: str, df: pd.DataFrame,
                       confidence: float, reason: str) -> bool:
        """
        Open a new position in the given direction.

        NEW STRATEGY: Trend already checked in _process_symbol
        - Open immediately in the direction provided
        - DCA will handle any pullbacks

        Returns:
            bool: True if trade was opened successfully, False otherwise
        """
        # Get current price
        prices = self.client.get_pricing([symbol])
        if not prices or symbol not in prices:
            print(f"  {symbol}: Failed to get price")
            return False

        if direction == 'BUY':
            entry_price = prices[symbol]['ask']
        else:
            entry_price = prices[symbol]['bid']

        # Check if DCA is enabled
        dca_enabled = self.config.dca.enabled if hasattr(self.config, 'dca') else False

        if dca_enabled:
            pip_value = self.config.get_pip_value(symbol)

            if getattr(self.config.dca, 'use_pip_based_dca', False):
                # PIP-BASED SL/TP for DCA
                # Initial SL is 25 pips to allow all DCA levels (3, 6, 10, 15 pips)
                initial_sl_pips = self.config.dca.initial_sl_pips  # 25 pips
                tp_pips = self.config.dca.take_profit_pips  # 8 pips
                if direction == 'BUY':
                    stop_loss = entry_price - (initial_sl_pips * pip_value)
                    take_profit = entry_price + (tp_pips * pip_value)
                else:
                    stop_loss = entry_price + (initial_sl_pips * pip_value)
                    take_profit = entry_price - (tp_pips * pip_value)
            else:
                # PERCENTAGE-BASED SL/TP for DCA
                # Initial SL is wide (8%) to allow all DCA levels to trigger
                initial_sl = getattr(self.config.dca, 'initial_sl_pct', 0.08)
                if direction == 'BUY':
                    stop_loss = entry_price * (1 - initial_sl)
                    take_profit = entry_price * (1 + self.config.dca.take_profit_pct)
                else:
                    stop_loss = entry_price * (1 + initial_sl)
                    take_profit = entry_price * (1 - self.config.dca.take_profit_pct)
        else:
            # Use pip-based SL/TP
            atr = self.strategy.get_atr(df)
            stop_loss, take_profit = self.risk_manager.calculate_stop_take_profit(
                symbol, entry_price, direction, atr
            )

        # Calculate position size (reduced for DCA to leave room for averaging down)
        units, position_value = self.risk_manager.calculate_position_size(
            symbol, entry_price, stop_loss, self.risk_manager.current_equity,
            dca_enabled=dca_enabled
        )

        # Place order
        success, result = self.trade_executor.place_market_order(
            symbol=symbol,
            units=units,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if success:
            # Track position locally
            self.position_manager.open_position(
                symbol=symbol,
                direction=direction,
                units=units,
                entry_price=result.get('fill_price', entry_price),
                stop_loss=stop_loss,
                take_profit=take_profit,
                trade_id=result.get('trade_id', '')
            )

            self.risk_manager.record_trade_entry()

            pip_value = self.config.get_pip_value(symbol)
            sl_pips = abs(entry_price - stop_loss) / pip_value
            tp_pips = abs(take_profit - entry_price) / pip_value

            # Print trade notification on new line
            if dca_enabled:
                if getattr(self.config.dca, 'use_pip_based_dca', False):
                    # Show pip-based SL/TP
                    print(f"\n[OPEN] {symbol} {direction} @ {entry_price:.5f} | "
                          f"Units: {units:.0f} | SL: {sl_pips:.0f}p TP: {tp_pips:.0f}p (DCA) | "
                          f"Conf: {confidence:.0%}")
                else:
                    sl_pct = self.config.dca.trailing_stop_pct * 100
                    tp_pct = self.config.dca.take_profit_pct * 100
                    print(f"\n[OPEN] {symbol} {direction} @ {entry_price:.5f} | "
                          f"Units: {units:.0f} | SL: {sl_pct:.1f}% TP: {tp_pct:.1f}% (DCA) | "
                          f"Conf: {confidence:.0%}")
            else:
                print(f"\n[OPEN] {symbol} {direction} @ {entry_price:.5f} | "
                      f"Units: {units:.0f} | SL: {sl_pips:.1f}p TP: {tp_pips:.1f}p | "
                      f"Conf: {confidence:.0%}")

            # Log trade
            self._log_trade('OPEN', symbol, direction, entry_price, units, confidence, reason)

            # PLACE PENDING DCA LIMIT ORDERS UPFRONT (only if use_pending_orders is True)
            # When use_pending_orders=False (SMART DCA), we check trend conditions at each DCA level instead
            if dca_enabled and getattr(self.config.dca, 'use_pending_orders', False):
                self._place_pending_dca_orders(symbol, direction, entry_price, units)

            return True
        else:
            print(f"\n[ERR] {symbol}: Order failed - {result.get('error', 'Unknown')}")
            return False

    def _place_pending_dca_orders(self, symbol: str, direction: str, entry_price: float, base_units: float) -> None:
        """
        Place pending limit orders for all DCA levels upfront.

        When a position is opened, we immediately place limit orders at each DCA trigger price.
        This ensures:
        - Guaranteed execution at exact price levels
        - Faster fills with no slippage
        - Works even when system is offline
        - Better average entry price

        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
            entry_price: Original entry price
            base_units: Base position units (before DCA divisor)
        """
        dca = self.config.dca
        pip_value = self.config.get_pip_value(symbol)
        position = self.position_manager.get_position(symbol)

        if not position:
            return

        # Calculate SL/TP for DCA orders (they all share the same SL/TP)
        sl_pips = dca.sl_after_dca_pips  # 30 pips from original entry
        tp_pips = dca.take_profit_pips   # 8 pips

        if direction == 'BUY':
            dca_stop_loss = entry_price - (sl_pips * pip_value)
            dca_take_profit = entry_price + (tp_pips * pip_value)
        else:
            dca_stop_loss = entry_price + (sl_pips * pip_value)
            dca_take_profit = entry_price - (tp_pips * pip_value)

        print(f"\n[DCA ORDERS] Placing {dca.max_dca_levels} pending limit orders for {symbol}...")

        placed_count = 0
        for level in range(1, dca.max_dca_levels + 1):
            trigger_pips = dca.get_dca_trigger_pips(level)
            multiplier = dca.get_dca_multiplier(level)

            # Calculate DCA units (with position cap)
            dca_units = self.position_manager.get_dca_units(symbol, base_units, level, dca)
            if dca_units <= 0:
                print(f"  DCA{level}: SKIPPED (position cap reached)")
                continue

            # Calculate trigger price based on direction
            # For BUY: DCA triggers when price goes DOWN (we buy lower)
            # For SELL: DCA triggers when price goes UP (we sell higher)
            if direction == 'BUY':
                trigger_price = entry_price - (trigger_pips * pip_value)
            else:
                trigger_price = entry_price + (trigger_pips * pip_value)

            # Place the limit order
            success, result = self.trade_executor.place_limit_order(
                symbol=symbol,
                units=int(dca_units),
                direction=direction,
                price=trigger_price,
                stop_loss=dca_stop_loss,
                take_profit=dca_take_profit
            )

            if success:
                order_id = result.get('order_id', '')
                order_type = result.get('type', 'LIMIT')

                if order_type == 'LIMIT_FILLED':
                    # Order was filled immediately (price already reached)
                    print(f"  DCA{level}: FILLED immediately @ {trigger_price:.5f} | {dca_units:.0f} units")
                    # Update position with DCA entry
                    position.add_dca_entry(level, dca_units, trigger_price, result.get('trade_id', ''))
                else:
                    # Order is pending
                    position.add_pending_dca_order(level, dca_units, trigger_price, order_id)
                    print(f"  DCA{level}: PENDING @ {trigger_price:.5f} ({trigger_pips:.0f}p) | "
                          f"{dca_units:.0f} units ({multiplier:.2f}x) | Order: {order_id}")
                    placed_count += 1
            else:
                error = result.get('error', 'Unknown')
                print(f"  DCA{level}: FAILED - {error}")

        if placed_count > 0:
            print(f"  >>> {placed_count} DCA limit order(s) staged and ready")
        print("")

    def _cancel_pending_dca_orders(self, symbol: str, position) -> None:
        """
        Cancel all pending DCA limit orders for a position.

        When a position is closed (manually, by SL, or by TP), we must cancel
        all staged DCA limit orders that haven't been filled yet.

        Args:
            symbol: Trading pair
            position: Position object with pending order tracking
        """
        if not position or not position.pending_dca_orders:
            return

        pending_count = len(position.pending_dca_orders)
        print(f"\n[DCA CANCEL] Cancelling {pending_count} pending DCA order(s) for {symbol}...")

        cancelled = 0
        failed = 0

        for pending in position.pending_dca_orders:
            success, result = self.trade_executor.cancel_order(pending.order_id)
            if success:
                cancelled += 1
                print(f"  DCA{pending.level}: Cancelled (Order {pending.order_id})")
            else:
                failed += 1
                error = result.get('error', 'Unknown')
                # Order may have been filled or already cancelled
                if 'not found' in error.lower() or 'filled' in error.lower():
                    print(f"  DCA{pending.level}: Already filled or cancelled")
                else:
                    print(f"  DCA{pending.level}: Cancel failed - {error}")

        # Clear the pending orders from position
        position.clear_pending_orders()

        if cancelled > 0:
            print(f"  >>> {cancelled} DCA order(s) cancelled")

    def _cancel_unwanted_pending_orders(self) -> None:
        """
        Cancel ALL pending DCA limit orders on OANDA when Smart DCA mode is enabled.

        Smart DCA uses market orders with trend checking, not limit orders.
        Any existing limit orders are leftovers from before the mode was changed
        and should be cancelled.
        """
        print(f"\n  [SMART DCA CLEANUP] Checking for leftover pending DCA orders to cancel...")

        # Get all pending orders from OANDA
        success, all_pending_orders = self.trade_executor.get_pending_orders()
        if not success:
            print("    [!] Could not fetch pending orders from OANDA")
            return

        if not all_pending_orders:
            print("    No pending orders found - OK")
            return

        # Filter for limit orders on our symbols (DCA orders are LIMIT type)
        dca_orders = []
        for order in all_pending_orders:
            symbol = order.get('instrument', '')
            order_type = order.get('type', '')
            if symbol in self.symbols and order_type == 'LIMIT':
                dca_orders.append(order)

        if not dca_orders:
            print("    No DCA limit orders found - OK")
            return

        print(f"    Found {len(dca_orders)} pending DCA limit order(s) to cancel...")

        cancelled = 0
        for order in dca_orders:
            order_id = order.get('id', '')
            symbol = order.get('instrument', '')
            price = float(order.get('price', 0))

            success, result = self.trade_executor.cancel_order(order_id)
            if success:
                cancelled += 1
                print(f"    CANCELLED: {symbol} limit @ {price:.5f} (Order {order_id})")
            else:
                error = result.get('error', 'Unknown')
                print(f"    FAILED: {symbol} @ {price:.5f} - {error}")

        if cancelled > 0:
            print(f"  >>> Cancelled {cancelled} leftover DCA limit order(s)")
            print(f"  >>> Smart DCA will use trend-checked market orders instead")

    def _sync_filled_pending_orders(self, oanda_trade_count: dict) -> None:
        """
        Sync local tracking with OANDA when pending DCA orders are filled.

        When a pending limit order is filled by OANDA, we need to:
        1. Remove it from pending_dca_orders
        2. Add it to dca_entries with proper tracking
        3. Recalculate average entry price and total units

        Args:
            oanda_trade_count: Dict of symbol -> number of trades on OANDA
        """
        dca_enabled = self.config.dca.enabled if hasattr(self.config, 'dca') else False
        if not dca_enabled:
            return

        for symbol, position in list(self.position_manager.positions.items()):
            if not position.pending_dca_orders:
                continue

            # Get pending order status from OANDA
            success, pending_orders = self.trade_executor.get_pending_orders(symbol)
            if not success:
                continue

            # Get IDs of still-pending orders
            pending_order_ids = {o.get('id', '') for o in pending_orders}

            # Check each local pending order
            filled_orders = []
            for pending in position.pending_dca_orders:
                if pending.order_id not in pending_order_ids:
                    # Order is no longer pending - it was either filled or cancelled
                    # Check if we have more trades than expected (indicating a fill)
                    expected_trades = 1 + position.dca_level  # Original + DCAs already filled
                    actual_trades = oanda_trade_count.get(symbol, 0)

                    if actual_trades > expected_trades:
                        # Order was filled! Update position
                        filled_orders.append(pending)
                        print(f"\n[DCA FILLED] {symbol} DCA{pending.level} limit order filled @ {pending.trigger_price:.5f}")
                        print(f"       +{pending.units:.0f} units | Now {actual_trades} trades on OANDA")

            # Process filled orders
            for filled in filled_orders:
                # Add to DCA entries (this also removes from pending_dca_orders)
                position.add_dca_entry(
                    level=filled.level,
                    units=filled.units,
                    price=filled.trigger_price,
                    trade_id=''  # We don't have the trade ID, but it's on OANDA
                )

                # Apply updated SL/TP
                pip_value = self.config.get_pip_value(symbol)
                self.position_manager.apply_dca_sl_tp(symbol, self.config.dca, pip_value)

                # Update OANDA trades with new SL/TP
                self._modify_all_trades_sl_tp(symbol, position.stop_loss, position.take_profit)

    def _close_position(self, symbol: str, reason: str = 'SIGNAL') -> None:
        """Close an existing position (ALL trades for this symbol)."""
        position = self.position_manager.get_position(symbol)
        if not position:
            return

        # CANCEL PENDING DCA LIMIT ORDERS FIRST
        # When closing a position, we must cancel all staged DCA orders
        self._cancel_pending_dca_orders(symbol, position)

        # Get current price
        prices = self.client.get_pricing([symbol])
        if not prices or symbol not in prices:
            return

        if position.direction == 'BUY':
            exit_price = prices[symbol]['bid']
        else:
            exit_price = prices[symbol]['ask']

        # CLOSE ALL TRADES FOR THIS SYMBOL ON OANDA
        # This is critical for DCA - when SL/TP hits, ALL trades must close together
        self._close_all_trades_for_symbol(symbol)

        # Track locally
        closed_trade = self.position_manager.close_position(symbol, exit_price, reason)

        if closed_trade:
            self.risk_manager.record_trade(closed_trade.pnl)

            # Color-code by P&L
            pnl_str = f"${closed_trade.pnl:+.2f}"
            result_str = "WIN" if closed_trade.pnl > 0 else "LOSS"

            print(f"\n[CLOSE] {symbol} {position.direction} @ {exit_price:.5f} | "
                  f"{result_str}: {closed_trade.pnl_pips:+.1f}p ({pnl_str}) | {reason}")

            # Log trade
            self._log_trade('CLOSE', symbol, position.direction, exit_price,
                           position.units, 0, reason, closed_trade.pnl)

    def _close_all_trades_for_symbol(self, symbol: str) -> None:
        """
        Close ALL OANDA trades for a symbol.

        When SL/TP hits, we must close ALL trades (original + DCAs) together.
        OANDA treats each DCA as a separate trade, so we must close each one.
        """
        # Get all open trades from OANDA
        success, trades = self.trade_executor.get_open_trades()
        if not success or not trades:
            # Fallback: try to close position by symbol
            self.trade_executor.close_position(symbol)
            return

        # Find all trades for this symbol
        symbol_trades = [t for t in trades if t.get('instrument') == symbol]

        if not symbol_trades:
            return

        closed_count = 0
        total_pnl = 0.0

        for trade in symbol_trades:
            trade_id = trade.get('id', '')
            if not trade_id:
                continue

            # Close this trade
            success, result = self.trade_executor.close_trade(trade_id)
            if success:
                closed_count += 1
                # Get realized P&L from the close result if available
                if 'orderFillTransaction' in result:
                    pnl = float(result['orderFillTransaction'].get('pl', 0))
                    total_pnl += pnl

        if closed_count > 0:
            print(f"       [OANDA] Closed {closed_count} trade(s) for {symbol} | Total P&L: ${total_pnl:+.2f}")

    def _update_positions(self) -> None:
        """Update all open positions with LIVE data from OANDA."""
        # Use cached OANDA trades from this iteration (already fetched in _trading_iteration)
        oanda_trades = getattr(self, '_live_oanda_trades', [])

        # Build map of currently open trades on OANDA (counting trades per symbol)
        oanda_trade_map = {}
        oanda_trade_count = {}
        if oanda_trades:
            for trade in oanda_trades:
                symbol = trade.get('instrument', '')
                if symbol:
                    oanda_trade_map[symbol] = trade
                    oanda_trade_count[symbol] = oanda_trade_count.get(symbol, 0) + 1

        # SYNC FILLED PENDING ORDERS
        # Check if any pending DCA limit orders were filled by OANDA
        self._sync_filled_pending_orders(oanda_trade_count)

        # Remove locally tracked positions that are no longer on OANDA (closed externally)
        for symbol in list(self.position_manager.positions.keys()):
            if symbol not in oanda_trade_map:
                # Position was closed on OANDA - cancel pending DCAs and remove from local tracking
                position = self.position_manager.get_position(symbol)
                if position and position.pending_dca_orders:
                    self._cancel_pending_dca_orders(symbol, position)
                print(f"\n[SYNC] {symbol}: Position closed on OANDA - removing from local tracking")
                del self.position_manager.positions[symbol]

        # Check if DCA is enabled
        dca_enabled = self.config.dca.enabled if hasattr(self.config, 'dca') else False

        # Update remaining positions with fresh OANDA data
        for symbol in list(self.position_manager.positions.keys()):
            position = self.position_manager.get_position(symbol)

            # Always use real OANDA data
            if symbol in oanda_trade_map:
                trade = oanda_trade_map[symbol]
                # Update unrealized P&L from OANDA (real data)
                position.unrealized_pnl = float(trade.get('unrealizedPL', 0))
                # Update units from OANDA
                oanda_units = abs(float(trade.get('currentUnits', 0)))
                if oanda_units > 0:
                    position.units = oanda_units
                # Update entry price if different
                oanda_price = float(trade.get('price', 0))
                if oanda_price > 0:
                    position.entry_price = oanda_price

            # Get current price for SL/TP checking
            prices = self.client.get_pricing([symbol])
            if not prices or symbol not in prices:
                continue

            if position.direction == 'BUY':
                current_price = prices[symbol]['bid']
            else:
                current_price = prices[symbol]['ask']

            # Update position and check SL/TP
            trigger = self.position_manager.update_position(symbol, current_price)

            if trigger == 'TP':
                self._close_position(symbol, 'TP')
            elif trigger == 'SL':
                self._close_position(symbol, 'SL')
            else:
                # DCA: Check if we should add to position
                if dca_enabled:
                    pip_value = self.config.get_pip_value(symbol)

                    # SKIP market order DCA check if using pending limit orders
                    # The limit orders are already staged and will execute automatically
                    if not position.use_pending_orders:
                        dca_level = self.position_manager.check_dca_trigger(symbol, self.config.dca, pip_value)
                        if dca_level:
                            self._execute_dca_entry(symbol, position, dca_level)

                    # SMART DCA TRAILING SYSTEM
                    # For DCA positions, we manage trailing manually to move ALL trades together
                    self._smart_dca_trailing(symbol, position, current_price)

    def _execute_dca_entry(self, symbol: str, position, dca_level: int) -> None:
        """
        Execute a DCA entry to add to position.

        NEW STRATEGY:
        - L1-L4: Execute IMMEDIATELY when price level is reached (no confirmation needed)
        - L5-L6: Require reversal confirmation before executing

        This ensures fast DCA execution for normal recovery while being cautious
        at extreme levels where the trade might be wrong.
        """
        # Get current price
        prices = self.client.get_pricing([symbol])
        if not prices or symbol not in prices:
            return

        if position.direction == 'BUY':
            entry_price = prices[symbol]['ask']
        else:
            entry_price = prices[symbol]['bid']

        # Get pip value for this symbol
        pip_value = self.config.get_pip_value(symbol)

        # Determine if this level requires reversal confirmation
        # L1-L4: IMMEDIATE execution, L5-L6: Require reversal
        dca_immediate_levels = getattr(self.config.dca, 'dca_immediate_levels', 4)
        requires_reversal = dca_level > dca_immediate_levels

        if requires_reversal:
            # L5-L6: Check trend conditions before executing
            use_trend_filter = getattr(self.config.dca, 'use_trend_filter', True)
            if use_trend_filter and symbol in self.data_buffer:
                df = self.data_buffer[symbol]
                can_dca, reason = self.trend_filter.can_dca(df, position.direction, dca_level)

                if not can_dca:
                    # Trend too strong against us - wait for reversal
                    print(f"\n[DCA{dca_level}] {symbol}: REVERSAL REQUIRED - {reason}")
                    print(f"       >>> L{dca_level} requires reversal confirmation - waiting...")
                    return

                # Show trend analysis when DCA is allowed
                trend_summary = self.trend_filter.get_trend_summary(df)
                print(f"\n[DCA{dca_level}] {symbol}: REVERSAL CONFIRMED - {reason}")
                print(f"       {trend_summary}")
        else:
            # L1-L4: Execute immediately without trend check
            print(f"\n[DCA{dca_level}] {symbol}: IMMEDIATE EXECUTION (L{dca_level} <= L{dca_immediate_levels})")

        # Calculate DCA units
        base_units = position.units  # Original position size
        dca_units = self.position_manager.get_dca_units(symbol, base_units, dca_level, self.config.dca)

        if dca_units <= 0:
            return

        # Place DCA order (same direction as original position)
        success, result = self.trade_executor.place_market_order(
            symbol=symbol,
            units=int(dca_units),
            direction=position.direction
        )

        if success:
            # Update position with DCA entry
            trade_id = result.get('trade_id', '')
            fill_price = result.get('fill_price', entry_price)

            position.add_dca_entry(
                level=dca_level,
                units=dca_units,
                price=fill_price,
                trade_id=trade_id
            )

            # Apply new SL/TP (SL from ORIGINAL entry, TP from avg entry)
            self.position_manager.apply_dca_sl_tp(symbol, self.config.dca, pip_value)

            # Track last DCA time for Smart Recovery Exit
            from datetime import datetime
            position.last_dca_time = datetime.now()

            # Display DCA entry with pip-based info
            if getattr(self.config.dca, 'use_pip_based_dca', False):
                trigger_pips = self.config.dca.get_dca_trigger_pips(dca_level)
                multiplier = self.config.dca.get_dca_multiplier(dca_level)
                sl_pips = self.config.dca.sl_after_dca_pips
                tp_pips = self.config.dca.dca_profit_target_pips

                print(f"\n[DCA{dca_level}] {symbol} {position.direction} +{dca_units:.0f} units @ {fill_price:.5f}")
                print(f"       Trigger: {trigger_pips:.0f} pips | Mult: {multiplier:.2f}x")
                print(f"       Original Entry: {position.entry_price:.5f} | Avg Entry: {position.avg_entry_price:.5f}")
                print(f"       Total Units: {position.total_units:.0f}")
                print(f"       SL: {position.stop_loss:.5f} ({sl_pips:.0f}p from ORIGINAL) | "
                      f"TP: {position.take_profit:.5f} ({tp_pips:.0f}p from avg)")
            else:
                # Fallback to percentage display
                dca_levels = self.config.dca.get_dca_levels()
                level_config = dca_levels[dca_level - 1]

                print(f"\n[DCA{dca_level}] {symbol} {position.direction} +{dca_units:.0f} units @ {fill_price:.5f}")
                print(f"       Trigger: {level_config.trigger_pct:.1%} | Mult: {level_config.multiplier:.2f}x")
                print(f"       Avg Entry: {position.avg_entry_price:.5f} | Total Units: {position.total_units:.0f}")
                print(f"       New SL: {position.stop_loss:.5f} ({self.config.dca.sl_after_dca_pct:.1%}) | "
                      f"TP: {position.take_profit:.5f} ({self.config.dca.dca_profit_target_pct:.1%})")

            # CRITICAL: Send SL/TP modification to OANDA for ALL trades on this symbol
            # After DCA, we need to update the actual broker-side SL/TP
            self._modify_all_trades_sl_tp(symbol, position.stop_loss, position.take_profit)

            # Log DCA entry
            self._log_trade(f'DCA{dca_level}', symbol, position.direction, fill_price,
                          dca_units, 0, f'DCA Level {dca_level}')
        else:
            print(f"\n[ERR] {symbol}: DCA{dca_level} order failed - {result.get('error', 'Unknown')}")

    def _smart_dca_trailing(self, symbol: str, position, current_price: float) -> None:
        """
        SMART DCA TRAILING SYSTEM
        =========================

        Manages trailing stops and protection for DCA positions.
        Unlike OANDA native trailing (per-trade), this moves ALL trades together.

        Features:
        1. DYNAMIC TP - Reduces TP as DCA level increases (faster exit)
        2. BREAKEVEN PROTECTION - Lock breakeven for DCA 3-4 when +3 pips
        3. MANUAL TRAILING - Move all SLs together based on avg entry profit
        4. EMERGENCY EXIT - Close all DCA4 positions at +3 pips profit

        Args:
            symbol: Trading pair
            position: Position object
            current_price: Current market price
        """
        if not position:
            return

        dca = self.config.dca
        pip_value = self.config.get_pip_value(symbol)
        dca_level = position.dca_level  # 0 = initial, 1-4 = DCA levels

        # Calculate profit from AVERAGE entry (for DCA positions)
        avg_entry = position.avg_entry_price if position.avg_entry_price > 0 else position.entry_price
        if position.direction == 'BUY':
            profit_pips = (current_price - avg_entry) / pip_value
        else:
            profit_pips = (avg_entry - current_price) / pip_value

        # ========================================
        # 1. MANUAL TP for DCA3-4 (below OANDA 5 pip minimum)
        # ========================================
        # DCA4: Exit at 2 pips profit (manual market close)
        if dca_level >= 4:
            manual_tp_dca4 = getattr(dca, 'manual_tp_dca4_pips', 2.0)
            if profit_pips >= manual_tp_dca4:
                print(f"\n[MANUAL TP] {symbol}: DCA4 at +{profit_pips:.1f} pips - CLOSING ALL!")
                print(f"       DCA Level: {dca_level} | Total Units: {position.total_units:.0f}")
                print(f"       >>> Manual TP hit (OANDA min is 5p, we exit at {manual_tp_dca4:.0f}p)")
                self._close_all_trades_for_symbol(symbol, f'DCA4 Manual TP at +{profit_pips:.1f}p')
                return

        # DCA3: Exit at 3 pips profit (manual market close)
        if dca_level == 3:
            manual_tp_dca3 = getattr(dca, 'manual_tp_dca3_pips', 3.0)
            if profit_pips >= manual_tp_dca3:
                print(f"\n[MANUAL TP] {symbol}: DCA3 at +{profit_pips:.1f} pips - CLOSING ALL!")
                print(f"       DCA Level: {dca_level} | Total Units: {position.total_units:.0f}")
                print(f"       >>> Manual TP hit (OANDA min is 5p, we exit at {manual_tp_dca3:.0f}p)")
                self._close_all_trades_for_symbol(symbol, f'DCA3 Manual TP at +{profit_pips:.1f}p')
                return

        # ========================================
        # 2. SMART RECOVERY EXIT - Exit at breakeven after DCA recovery
        # ========================================
        # After DCA, if we recover to breakeven, EXIT immediately
        # Don't wait for full TP - risk of reversal is too high
        if dca_level >= 1 and getattr(dca, 'use_smart_recovery_exit', True):
            # Check minimum hold time after DCA (avoid premature exits)
            min_hold_minutes = getattr(dca, 'min_dca_hold_minutes', 15.0)
            last_dca_time = getattr(position, 'last_dca_time', None)

            minutes_since_dca = 0
            if last_dca_time:
                from datetime import datetime
                now = datetime.now()
                if isinstance(last_dca_time, datetime):
                    minutes_since_dca = (now - last_dca_time).total_seconds() / 60.0

            # Only check recovery exit if held DCA position for minimum time
            if minutes_since_dca >= min_hold_minutes or last_dca_time is None:
                # Get recovery exit threshold for this DCA level
                recovery_thresholds = {
                    1: getattr(dca, 'recovery_exit_dca1_pips', 1.0),
                    2: getattr(dca, 'recovery_exit_dca2_pips', 1.0),
                    3: getattr(dca, 'recovery_exit_dca3_pips', 0.5),
                    4: getattr(dca, 'recovery_exit_dca4_pips', 0.0),
                }
                recovery_exit_pips = recovery_thresholds.get(dca_level, 0.5)

                # Check if we've recovered to the threshold
                if profit_pips >= recovery_exit_pips and profit_pips > 0:
                    print(f"\n{'='*60}")
                    print(f"[SMART RECOVERY EXIT] {symbol}")
                    print(f"{'='*60}")
                    print(f"       DCA Level: {dca_level} | Recovered to: +{profit_pips:.1f} pips")
                    print(f"       Threshold: +{recovery_exit_pips:.1f} pips | Minutes since DCA: {minutes_since_dca:.0f}")
                    print(f"       >>> EXITING NOW - Don't risk another reversal!")
                    print(f"       >>> Better to lock in small gain than wait for full TP")
                    self._close_all_trades_for_symbol(symbol, f'Smart Recovery Exit DCA{dca_level} at +{profit_pips:.1f}p')
                    return

        # ========================================
        # 3. TIME-BASED EXIT - FORCE EXIT AFTER 4 HOURS MAX HOLD
        # ========================================
        # NEW STRATEGY: If position held > 4 hours, EXIT REGARDLESS OF LOSS
        # Take the loss and start over - don't let positions sit forever
        if getattr(dca, 'use_time_based_recovery', True):
            max_hold_hours = getattr(dca, 'max_hold_hours', 4.0)
            time_recovery_loss_pips = getattr(dca, 'time_recovery_loss_pips', -5.0)  # Accept up to -5p loss

            # Check position age
            position_age_hours = 0
            entry_time = getattr(position, 'entry_time', None)
            if entry_time:
                from datetime import datetime
                now = datetime.now()
                if isinstance(entry_time, datetime):
                    position_age_hours = (now - entry_time).total_seconds() / 3600.0

            # FORCE EXIT after max_hold_hours - regardless of P&L (but prefer small loss)
            if position_age_hours >= max_hold_hours:
                # If recovered to acceptable loss OR in profit, exit immediately
                if profit_pips >= time_recovery_loss_pips:
                    print(f"\n{'='*60}")
                    print(f"[4-HOUR MAX HOLD EXIT] {symbol}")
                    print(f"{'='*60}")
                    print(f"       DCA Level: {dca_level} | Position Age: {position_age_hours:.1f} hours")
                    print(f"       Current P&L: {profit_pips:+.1f} pips")
                    print(f"       >>> MAX HOLD TIME REACHED - EXITING NOW!")
                    print(f"       >>> Take the loss and start fresh with new trend")
                    self._close_all_trades_for_symbol(symbol, f'4H Max Hold Exit DCA{dca_level} at {profit_pips:+.1f}p')
                    return
                else:
                    # Loss is worse than threshold, but still close if held > 5 hours
                    if position_age_hours >= max_hold_hours + 1.0:  # Give 1 extra hour buffer
                        print(f"\n{'='*60}")
                        print(f"[FORCED EXIT - EXCEEDED MAX HOLD] {symbol}")
                        print(f"{'='*60}")
                        print(f"       DCA Level: {dca_level} | Position Age: {position_age_hours:.1f} hours")
                        print(f"       Current P&L: {profit_pips:+.1f} pips (worse than {time_recovery_loss_pips:+.1f}p threshold)")
                        print(f"       >>> FORCED EXIT - Position held too long!")
                        print(f"       >>> Accept the loss and move on")
                        self._close_all_trades_for_symbol(symbol, f'FORCED Exit DCA{dca_level} after {position_age_hours:.1f}h at {profit_pips:+.1f}p')
                        return

        # ========================================
        # 4. BREAKEVEN PROTECTION for DCA 3-4 (fallback if recovery exit not triggered)
        # ========================================
        if dca_level >= 3 and getattr(dca, 'use_breakeven_protection', True):
            be_activation = getattr(dca, 'breakeven_activation_pips', 3.0)
            be_buffer = getattr(dca, 'breakeven_buffer_pips', 1.0)

            # Check if we should lock breakeven
            if profit_pips >= be_activation:
                # Calculate breakeven SL (avg entry + buffer)
                if position.direction == 'BUY':
                    new_sl = avg_entry + (be_buffer * pip_value)
                else:
                    new_sl = avg_entry - (be_buffer * pip_value)

                # Only move SL if it's better than current
                current_sl = position.stop_loss
                should_move = False
                if position.direction == 'BUY' and new_sl > current_sl:
                    should_move = True
                elif position.direction == 'SELL' and new_sl < current_sl:
                    should_move = True

                if should_move and not getattr(position, '_breakeven_locked', False):
                    print(f"\n[BREAKEVEN] {symbol}: Locking breakeven for DCA{dca_level}!")
                    print(f"       Profit: +{profit_pips:.1f} pips | Avg Entry: {avg_entry:.5f}")
                    print(f"       Moving SL: {current_sl:.5f} -> {new_sl:.5f} (BE + {be_buffer:.0f}p buffer)")
                    print(f"       >>> Protecting {position.total_units:.0f} units from reversal!")

                    # Update position SL
                    position.stop_loss = new_sl
                    position._breakeven_locked = True

                    # Send to OANDA - move ALL trades SL
                    self._modify_all_trades_sl_tp(symbol, new_sl, position.take_profit)
                    return

        # ========================================
        # 5. MANUAL TRAILING for DCA positions
        # ========================================
        if dca_level >= 2 and getattr(dca, 'use_manual_dca_trailing', True):
            trailing_activation = getattr(dca, 'manual_trailing_activation_pips', 5.0)
            trailing_distance = getattr(dca, 'manual_trailing_distance_pips', 8.0)

            if profit_pips >= trailing_activation:
                # Calculate trailing SL
                if position.direction == 'BUY':
                    new_sl = current_price - (trailing_distance * pip_value)
                else:
                    new_sl = current_price + (trailing_distance * pip_value)

                # Only move SL if it's better than current
                current_sl = position.stop_loss
                should_move = False
                if position.direction == 'BUY' and new_sl > current_sl:
                    should_move = True
                elif position.direction == 'SELL' and new_sl < current_sl:
                    should_move = True

                if should_move:
                    sl_move_pips = abs(new_sl - current_sl) / pip_value
                    print(f"\n[TRAILING] {symbol}: Moving SL for DCA{dca_level} position")
                    print(f"       Profit: +{profit_pips:.1f} pips | Distance: {trailing_distance:.0f}p")
                    print(f"       SL: {current_sl:.5f} -> {new_sl:.5f} (+{sl_move_pips:.1f}p)")

                    # Update position SL
                    position.stop_loss = new_sl

                    # Track highest SL for trailing (don't move backwards)
                    if position.direction == 'BUY':
                        position.highest_price = max(getattr(position, 'highest_price', current_price), current_price)
                    else:
                        position.lowest_price = min(getattr(position, 'lowest_price', current_price), current_price)

                    # Send to OANDA - move ALL trades SL
                    self._modify_all_trades_sl_tp(symbol, new_sl, position.take_profit)
                    return

        # ========================================
        # 6. UPDATE DYNAMIC TP based on DCA level
        # ========================================
        if dca_level > 0:
            # Get dynamic TP for current DCA level
            new_tp_pips = dca.get_tp_for_dca_level(dca_level)

            # Calculate new TP price from avg entry
            if position.direction == 'BUY':
                new_tp = avg_entry + (new_tp_pips * pip_value)
            else:
                new_tp = avg_entry - (new_tp_pips * pip_value)

            # Check if TP needs updating (only if significantly different)
            current_tp = position.take_profit
            tp_diff = abs(new_tp - current_tp) / pip_value

            if tp_diff > 0.5:  # More than 0.5 pip difference
                print(f"\n[DYNAMIC TP] {symbol}: Adjusting TP for DCA{dca_level}")
                print(f"       New TP: {new_tp:.5f} ({new_tp_pips:.0f}p from avg)")

                position.take_profit = new_tp
                self._modify_all_trades_sl_tp(symbol, position.stop_loss, new_tp)

    def _close_all_trades_for_symbol(self, symbol: str, reason: str = "Manual close") -> None:
        """
        Close ALL trades for a symbol at market.

        Used for emergency exits when we need to close DCA4 positions quickly.

        Args:
            symbol: Trading pair
            reason: Reason for closing
        """
        print(f"\n[CLOSE ALL] {symbol}: {reason}")

        # Get all open trades for this symbol
        success, trades = self.trade_executor.get_open_trades()
        if not success or not trades:
            print(f"       [!] Could not fetch trades")
            return

        symbol_trades = [t for t in trades if t.get('instrument') == symbol]
        if not symbol_trades:
            print(f"       [!] No trades found for {symbol}")
            return

        closed_count = 0
        total_pnl = 0.0

        for trade in symbol_trades:
            trade_id = trade.get('id', '')
            if not trade_id:
                continue

            success, result = self.trade_executor.close_trade(trade_id)
            if success:
                closed_count += 1
                pnl = result.get('pnl', 0)
                total_pnl += pnl
                print(f"       Trade {trade_id}: CLOSED | P&L: ${pnl:+.2f}")
            else:
                error = result.get('error', 'Unknown')
                print(f"       [!] Failed to close trade {trade_id}: {error}")

        # Clear local position
        if symbol in self.position_manager.positions:
            del self.position_manager.positions[symbol]

        print(f"       >>> Closed {closed_count} trade(s) | Total P&L: ${total_pnl:+.2f}")

        # Log the close
        self._log_trade('CLOSE_ALL', symbol, 'N/A', 0, 0, 0, reason)

    def _modify_all_trades_sl_tp(self, symbol: str, stop_loss: float, take_profit: float) -> None:
        """
        Modify SL/TP on ALL OANDA trades for a symbol.

        OANDA creates separate trades for each order. After DCA, we need to update
        the SL/TP on ALL trades for this symbol to:
        - SL: 35 pips from ORIGINAL entry price (same for all trades)
        - TP: 8 pips from AVERAGE entry price

        When ANY trade hits SL, ALL trades for this symbol will be closed together.

        Args:
            symbol: Trading pair
            stop_loss: New stop loss price (from original entry)
            take_profit: New take profit price
        """
        # Get all open trades from OANDA
        success, trades = self.trade_executor.get_open_trades()
        if not success or not trades:
            print(f"       [!] Could not fetch OANDA trades for SL/TP modification")
            return

        # Find all trades for this symbol
        symbol_trades = [t for t in trades if t.get('instrument') == symbol]

        if not symbol_trades:
            print(f"       [!] No OANDA trades found for {symbol}")
            return

        modified_count = 0
        failed_count = 0

        for trade in symbol_trades:
            trade_id = trade.get('id', '')
            if not trade_id:
                continue

            # Send modification to OANDA
            success, result = self.trade_executor.modify_trade(
                trade_id=trade_id,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if success:
                modified_count += 1
            else:
                failed_count += 1
                error = result.get('error', 'Unknown')
                print(f"       [!] Failed to modify trade {trade_id}: {error}")

        if modified_count > 0:
            print(f"       [OANDA] Modified SL/TP on {modified_count} trade(s)")
        if failed_count > 0:
            print(f"       [!] Failed to modify {failed_count} trade(s)")

    def _set_trailing_stop_all_trades(self, symbol: str, trailing_pips: float) -> None:
        """
        Set actual OANDA Trailing Stop orders on ALL trades for a symbol.

        This sets the OANDA native trailing stop order which:
        - Automatically tracks price movements
        - Shows up in the TS field on OANDA
        - Works even when our system is offline
        - Uses OANDA's built-in trailing logic

        Args:
            symbol: Trading pair
            trailing_pips: Trailing stop distance in pips (minimum 5 for OANDA)
        """
        # Ensure minimum 5 pips (OANDA requirement)
        trailing_pips = max(trailing_pips, 5.0)

        # Get all open trades from OANDA
        success, trades = self.trade_executor.get_open_trades()
        if not success or not trades:
            print(f"       [!] Could not fetch OANDA trades for trailing stop")
            return

        # Find all trades for this symbol
        symbol_trades = [t for t in trades if t.get('instrument') == symbol]

        if not symbol_trades:
            print(f"       [!] No OANDA trades found for {symbol}")
            return

        set_count = 0
        failed_count = 0

        for trade in symbol_trades:
            trade_id = trade.get('id', '')
            if not trade_id:
                continue

            # Check if trailing stop already set
            if 'trailingStopLossOrder' in trade and trade['trailingStopLossOrder']:
                existing_distance = trade['trailingStopLossOrder'].get('distance', 'N/A')
                print(f"       Trade {trade_id}: Trailing stop already set (distance: {existing_distance})")
                continue

            # Set OANDA trailing stop order
            success, result = self.trade_executor.set_trailing_stop(
                trade_id=trade_id,
                distance_pips=trailing_pips,
                symbol=symbol
            )

            if success:
                set_count += 1
                print(f"       Trade {trade_id}: OANDA Trailing Stop SET ({trailing_pips:.0f} pips)")
            else:
                failed_count += 1
                error = result.get('error', 'Unknown')
                print(f"       [!] Failed to set trailing stop on trade {trade_id}: {error}")

        if set_count > 0:
            print(f"       [OANDA] Set Trailing Stop on {set_count} trade(s) - Distance: {trailing_pips:.0f} pips")
        if failed_count > 0:
            print(f"       [!] Failed to set trailing stop on {failed_count} trade(s)")

    def _print_compact_status(self, iteration: int) -> None:
        """Print session summary AFTER ML scan with LIVE OANDA data."""
        # Use cached OANDA trades (already fetched in _trading_iteration)
        oanda_trades = getattr(self, '_live_oanda_trades', [])

        # Count and calculate from cached data (no extra API calls)
        open_count = len(oanda_trades) if oanda_trades else 0
        total_unrealized = sum(float(t.get('unrealizedPL', 0)) for t in oanda_trades) if oanda_trades else 0.0

        # SESSION SUMMARY after ML scan
        print(f"\n--- SESSION STATUS (LIVE OANDA DATA) ---")

        # Show open positions with FULL OANDA details
        if open_count > 0:
            # GROUP TRADES BY SYMBOL first to handle DCA positions correctly
            trades_by_symbol = {}
            for trade in oanda_trades:
                symbol = trade.get('instrument', '')
                if not symbol or symbol not in self.symbols:
                    continue
                if symbol not in trades_by_symbol:
                    trades_by_symbol[symbol] = []
                trades_by_symbol[symbol].append(trade)

            # Count unique symbols (not individual trades)
            unique_positions = len(trades_by_symbol)
            print(f"OPEN POSITIONS ({unique_positions} symbols, {open_count} trades) | Unrealized P&L: ${total_unrealized:+.2f}")

            for symbol, symbol_trades in trades_by_symbol.items():
                # Aggregate data from all trades for this symbol
                total_units = 0
                total_unrealized_pnl = 0.0
                weighted_entry_sum = 0.0
                direction = None

                # Find TP/SL from first trade (they should be same for all trades in symbol)
                first_trade = symbol_trades[0]
                stop_loss = 0.0
                take_profit = 0.0
                if 'stopLossOrder' in first_trade and first_trade['stopLossOrder']:
                    stop_loss = float(first_trade['stopLossOrder'].get('price', 0))
                if 'takeProfitOrder' in first_trade and first_trade['takeProfitOrder']:
                    take_profit = float(first_trade['takeProfitOrder'].get('price', 0))

                # Check for trailing stop in any trade
                has_oanda_trailing = False
                trailing_distance = 0

                for trade in symbol_trades:
                    trade_units = abs(float(trade.get('currentUnits', 0)))
                    trade_entry = float(trade.get('price', 0))
                    trade_pnl = float(trade.get('unrealizedPL', 0))

                    total_units += trade_units
                    total_unrealized_pnl += trade_pnl
                    weighted_entry_sum += trade_units * trade_entry

                    # Get direction (same for all trades in symbol)
                    if direction is None:
                        direction = 'BUY' if float(trade.get('currentUnits', 0)) > 0 else 'SELL'

                    # Check for trailing stop
                    if 'trailingStopLossOrder' in trade and trade['trailingStopLossOrder']:
                        has_oanda_trailing = True
                        trailing_distance = trade['trailingStopLossOrder'].get('distance', 0)

                # Calculate weighted average entry
                avg_entry_price = weighted_entry_sum / total_units if total_units > 0 else 0
                num_trades = len(symbol_trades)

                # For display, use averaged values
                units = total_units
                entry_price = avg_entry_price
                unrealized_pnl = total_unrealized_pnl

                # Note: TP/SL already fetched from first_trade above

                # Convert trailing distance to pips if present
                pip_value = self.config.get_pip_value(symbol)
                if has_oanda_trailing and trailing_distance:
                    trailing_distance = float(trailing_distance) / pip_value

                # Get LIVE current price from OANDA
                prices = self.client.get_pricing([symbol])
                if prices and symbol in prices:
                    if direction == 'BUY':
                        current_price = prices[symbol]['bid']
                    else:
                        current_price = prices[symbol]['ask']

                    # Calculate distance to TP/SL in pips (LIVE)
                    if direction == 'BUY':
                        tp_pips = (take_profit - current_price) / pip_value if take_profit > 0 else 0
                        sl_pips = (current_price - stop_loss) / pip_value if stop_loss > 0 else 0
                    else:
                        tp_pips = (current_price - take_profit) / pip_value if take_profit > 0 else 0
                        sl_pips = (stop_loss - current_price) / pip_value if stop_loss > 0 else 0

                    # Current P&L in pips (from average entry)
                    if direction == 'BUY':
                        pnl_pips = (current_price - entry_price) / pip_value
                    else:
                        pnl_pips = (entry_price - current_price) / pip_value

                    # Display header - show DCA info if multiple trades
                    if num_trades > 1:
                        print(f"  {symbol}: {direction} {units:.0f} units @ {entry_price:.5f} (AVG) [DCA: {num_trades} trades]")
                    else:
                        print(f"  {symbol}: {direction} {units:.0f} @ {entry_price:.5f}")
                    print(f"    Current: {current_price:.5f} | P&L: {pnl_pips:+.1f} pips (${unrealized_pnl:+.2f})")
                    print(f"    TP: {take_profit:.5f} ({tp_pips:+.1f} pips away) | SL: {stop_loss:.5f} ({sl_pips:.1f} pips away)")

                    # Get local position for DCA tracking
                    local_pos = self.position_manager.get_position(symbol)

                    # === SMART DCA TREND FILTER STATUS ===
                    # Show trend analysis for ALL underwater positions waiting for reversal
                    # Works even without local_pos - uses OANDA trade count as DCA level
                    if pnl_pips < 0:
                        dca = self.config.dca
                        use_trend_filter = getattr(dca, 'use_trend_filter', True)

                        # Get current DCA level from local_pos OR OANDA trade count
                        current_dca_level = 0
                        if local_pos:
                            current_dca_level = local_pos.dca_level
                        else:
                            # Use OANDA trade count - 1 as DCA level (1 trade = level 0, 2 trades = level 1)
                            current_dca_level = num_trades - 1

                        # Calculate which DCA level WOULD trigger at current price drop (check all 6 levels)
                        loss_pips = abs(pnl_pips)
                        next_dca_level = 0

                        for level in range(1, 7):  # Check levels 1-6
                            if loss_pips >= dca.get_dca_trigger_pips(level) and current_dca_level < level:
                                next_dca_level = level
                                break  # Take the lowest untriggered level

                        # Get immediate execution threshold (L1-L4 immediate, L5-L6 need reversal)
                        dca_immediate_levels = getattr(dca, 'dca_immediate_levels', 4)

                        # If DCA should have triggered but hasn't, show status
                        if next_dca_level > 0:
                            # Check if this level requires reversal confirmation
                            requires_reversal = next_dca_level > dca_immediate_levels

                            if requires_reversal:
                                # L5-L6: Check trend filter
                                if use_trend_filter and symbol in self.data_buffer:
                                    df = self.data_buffer[symbol]
                                    can_dca, reason = self.trend_filter.can_dca(df, direction, next_dca_level)
                                    trend_summary = self.trend_filter.get_trend_summary(df)

                                    if not can_dca:
                                        print(f"    >>> [DCA{next_dca_level}] REVERSAL REQUIRED @ {loss_pips:.1f}p loss")
                                        print(f"        REASON: {reason}")
                                        print(f"        TREND: {trend_summary}")
                                        print(f"        >>> L{next_dca_level} needs reversal confirmation <<<")
                                    else:
                                        print(f"    >>> [DCA{next_dca_level}] REVERSAL CONFIRMED - {reason}")
                                else:
                                    print(f"    >>> [DCA{next_dca_level}] L{next_dca_level} @ {loss_pips:.1f}p - waiting for trend data")
                            else:
                                # L1-L4: IMMEDIATE EXECUTION
                                print(f"    >>> [DCA{next_dca_level}] IMMEDIATE @ {loss_pips:.1f}p (L1-L4 no reversal needed)")
                        elif loss_pips > 0:
                            # Position underwater but not at DCA trigger yet
                            next_level = min(current_dca_level + 1, 6)
                            next_trigger = dca.get_dca_trigger_pips(next_level)
                            pips_to_next = next_trigger - loss_pips
                            if pips_to_next > 0:
                                mode = "IMMEDIATE" if next_level <= dca_immediate_levels else "REVERSAL"
                                print(f"    >>> [DCA INFO] Next DCA{next_level} at {next_trigger:.0f}p ({pips_to_next:.1f}p away) - {mode}")

                    # Show DCA and trailing stop info
                    # Note: has_oanda_trailing and trailing_distance already set in aggregation loop above

                    # local_pos already fetched above for Smart DCA status
                    if local_pos:
                        # For DCA positions, show profit from AVERAGE entry
                        if local_pos.dca_active and local_pos.avg_entry_price > 0:
                            avg_entry = local_pos.avg_entry_price
                            if direction == 'BUY':
                                avg_profit_pips = (current_price - avg_entry) / pip_value
                            else:
                                avg_profit_pips = (avg_entry - current_price) / pip_value

                            print(f"    DCA Level: {local_pos.dca_level} | Avg Entry: {avg_entry:.5f} | Profit from Avg: {avg_profit_pips:+.1f}p")
                            print(f"    Total Units: {local_pos.total_units:.0f} | Total P&L: ${local_pos.unrealized_pnl:+.2f}")

                            # Show Smart DCA Trailing status
                            dca = self.config.dca
                            if local_pos.dca_level >= 4:
                                exit_pips = getattr(dca, 'dca4_exit_profit_pips', 3.0)
                                print(f"    >>> DCA4 EMERGENCY EXIT: At {avg_profit_pips:+.1f}p (exits at +{exit_pips:.0f}p)")
                            elif local_pos.dca_level >= 3:
                                be_pips = getattr(dca, 'breakeven_activation_pips', 3.0)
                                if getattr(local_pos, '_breakeven_locked', False):
                                    print(f"    >>> BREAKEVEN LOCKED!")
                                elif avg_profit_pips >= be_pips:
                                    print(f"    >>> BREAKEVEN ACTIVATING (profit {avg_profit_pips:+.1f}p >= {be_pips:.0f}p)")
                                else:
                                    print(f"    >>> Breakeven at +{be_pips:.0f}p (current: {avg_profit_pips:+.1f}p)")
                            elif local_pos.dca_level >= 2:
                                trail_act = getattr(dca, 'manual_trailing_activation_pips', 5.0)
                                if avg_profit_pips >= trail_act:
                                    print(f"    >>> TRAILING ACTIVE (profit {avg_profit_pips:+.1f}p >= {trail_act:.0f}p)")
                                else:
                                    print(f"    >>> Trailing at +{trail_act:.0f}p (current: {avg_profit_pips:+.1f}p)")
                        else:
                            # Non-DCA position - show regular trailing status
                            if has_oanda_trailing:
                                print(f"    Trailing: OANDA TS ACTIVE ({trailing_distance:.0f} pips distance)")
                            else:
                                if pnl_pips >= 5.0:
                                    print(f"    Trailing: PENDING (will set on next iteration)")
                                else:
                                    print(f"    Trailing: Activates at +5p (current: {pnl_pips:+.1f}p)")
                else:
                    # Fallback if pricing fails
                    print(f"  {symbol}: {direction} {units:.0f} @ {entry_price:.5f} | P&L: ${unrealized_pnl:+.2f}")
                    print(f"    TP: {take_profit:.5f} | SL: {stop_loss:.5f}")

            # Show TOTAL UNREALIZED P&L summary after all trades
            print(f"\n  ════════════════════════════════════════════════════════════")
            print(f"  TOTAL UNREALIZED P&L: ${total_unrealized:+.2f} ({open_count} trades)")
            print(f"  ════════════════════════════════════════════════════════════")
        else:
            print(f"No open positions")

        # Get local stats for closed trade history
        stats = self.position_manager.get_statistics()

        # Session stats (closed trades only)
        if stats['total_trades'] > 0:
            print(f"\n  ============================================================")
            print(f"  CLOSED TRADES: {stats['winning_trades']}W/{stats['losing_trades']}L "
                  f"({stats['win_rate']:.1%}) | Realized P&L: ${stats['total_pnl']:+.2f}")

            # Per-pair breakdown
            pair_stats = self._get_pair_statistics()
            if pair_stats:
                print(f"  ------------------------------------------------------------")
                print(f"  {'PAIR':<12} {'TRADES':>6} {'WINS':>6} {'LOSSES':>6} {'WIN%':>8} {'P&L':>10}")
                print(f"  ------------------------------------------------------------")
                for pair, ps in pair_stats.items():
                    win_pct = (ps['wins'] / ps['trades'] * 100) if ps['trades'] > 0 else 0
                    print(f"  {pair:<12} {ps['trades']:>6} {ps['wins']:>6} {ps['losses']:>6} "
                          f"{win_pct:>7.1f}% ${ps['pnl']:>+9.2f}")
                print(f"  ------------------------------------------------------------")
                total_wins = sum(p['wins'] for p in pair_stats.values())
                total_losses = sum(p['losses'] for p in pair_stats.values())
                total_trades = sum(p['trades'] for p in pair_stats.values())
                total_pnl = sum(p['pnl'] for p in pair_stats.values())
                total_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0
                print(f"  {'TOTAL':<12} {total_trades:>6} {total_wins:>6} {total_losses:>6} "
                      f"{total_win_pct:>7.1f}% ${total_pnl:>+9.2f}")
            print(f"  ============================================================")

    def _get_pair_statistics(self) -> Dict[str, Dict]:
        """Get per-pair trading statistics."""
        pair_stats = {}
        for trade in self.position_manager.trade_history:
            symbol = trade.symbol
            if symbol not in pair_stats:
                pair_stats[symbol] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}

            pair_stats[symbol]['trades'] += 1
            pair_stats[symbol]['pnl'] += trade.pnl
            if trade.pnl > 0:
                pair_stats[symbol]['wins'] += 1
            else:
                pair_stats[symbol]['losses'] += 1

        return pair_stats

    def _print_status(self) -> None:
        """Print detailed status (when verbose or on events)."""
        stats = self.position_manager.get_statistics()

        print(f"\n  Status: Equity=${self.risk_manager.current_equity:,.2f} | "
              f"Realized=${stats['total_pnl']:.2f} | "
              f"Win Rate={stats['win_rate']:.1%} | "
              f"Closed Trades={stats['total_trades']}")

        if stats['open_positions'] > 0:
            print(f"  Open: {stats['open_positions']} positions | "
                  f"Unrealized=${stats['unrealized_pnl']:.2f}")

    def _print_final_summary(self) -> None:
        """Print final trading summary."""
        stats = self.position_manager.get_statistics()
        pair_stats = self._get_pair_statistics()

        print("\n")
        print("=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)

        # Overall stats
        print(f"Total Closed Trades: {stats['total_trades']}")
        print(f"Wins: {stats['winning_trades']} | Losses: {stats['losing_trades']} | "
              f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Total P&L from Closed Trades: ${stats['total_pnl']:+.2f}")

        # Per-pair breakdown
        if pair_stats:
            print(f"\nPerformance by Pair:")
            print(f"{'PAIR':<12} {'TRADES':>8} {'WINS':>8} {'LOSSES':>8} {'WIN%':>10} {'P&L':>12}")
            print("-" * 70)
            for pair, ps in pair_stats.items():
                win_pct = (ps['wins'] / ps['trades'] * 100) if ps['trades'] > 0 else 0
                print(f"{pair:<12} {ps['trades']:>8} {ps['wins']:>8} {ps['losses']:>8} "
                      f"{win_pct:>9.1f}% ${ps['pnl']:>+11.2f}")
            print("-" * 70)

        # Financial summary
        print(f"\nFinancial Summary:")
        print(f"  Final Balance: ${self.risk_manager.current_equity:,.2f}")
        print(f"  Total Realized P&L: ${stats['total_pnl']:+.2f}")
        if stats['profit_factor'] != float('inf'):
            print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Avg Win: ${stats['avg_win']:.2f}")
        print(f"  Avg Loss: ${stats['avg_loss']:.2f}")

        # Open positions warning
        if stats['open_positions'] > 0:
            print(f"\n  [!] {stats['open_positions']} position(s) still open")
            print(f"  Unrealized P&L: ${stats['unrealized_pnl']:+.2f}")

        print("=" * 70)

    def _log_trade(self, action: str, symbol: str, direction: str, price: float,
                   units: float, confidence: float, reason: str, pnl: float = 0) -> None:
        """Log trade to CSV file."""
        file_exists = os.path.exists(self.trade_log_file)

        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'action', 'symbol', 'direction', 'price',
                               'units', 'confidence', 'reason', 'pnl'])
            writer.writerow([
                datetime.now().isoformat(),
                action, symbol, direction, f"{price:.5f}",
                f"{units:.0f}", f"{confidence:.4f}", reason, f"{pnl:.2f}"
            ])

    def stop(self) -> None:
        """Stop the trading engine."""
        self.is_running = False

    def close_all_positions(self) -> None:
        """
        Close ALL open positions directly from OANDA.

        This fetches ALL trades from OANDA and closes each one individually.
        This is important because DCA creates multiple trades per symbol that
        may not all be tracked locally.
        """
        # CANCEL ALL PENDING DCA ORDERS FIRST
        print("\n[CLEANUP] Cancelling all pending DCA orders...")
        for symbol, position in list(self.position_manager.positions.items()):
            if position.pending_dca_orders:
                self._cancel_pending_dca_orders(symbol, position)

        # Also cancel any pending orders directly from OANDA (in case local tracking missed some)
        for symbol in self.symbols:
            cancelled, _ = self.trade_executor.cancel_all_orders_for_symbol(symbol)
            if cancelled > 0:
                print(f"  [OANDA] Cancelled {cancelled} pending order(s) for {symbol}")

        # FETCH ALL OPEN TRADES DIRECTLY FROM OANDA (not from local tracking)
        success, trades = self.trade_executor.get_open_trades()

        if not success:
            print("[!] Could not fetch trades from OANDA")
            # Fallback to local tracking
            for symbol in list(self.position_manager.positions.keys()):
                self._close_position(symbol, 'MANUAL')
            return

        if not trades:
            print("[*] No open trades on OANDA")
            self.position_manager.positions.clear()
            return

        # Filter to only our tracked symbols
        our_trades = [t for t in trades if t.get('instrument', '') in self.symbols]

        if not our_trades:
            print("[*] No open trades for tracked symbols")
            self.position_manager.positions.clear()
            return

        print(f"\n[CLOSING] Found {len(our_trades)} trade(s) on OANDA")

        # Group trades by symbol for P&L reporting
        trades_by_symbol = {}
        for trade in our_trades:
            symbol = trade.get('instrument', '')
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)

        # Close ALL trades for each symbol
        total_closed = 0
        total_pnl = 0.0

        for symbol, symbol_trades in trades_by_symbol.items():
            symbol_pnl = 0.0
            closed_count = 0

            for trade in symbol_trades:
                trade_id = trade.get('id', '')
                units = abs(float(trade.get('currentUnits', 0)))
                direction = 'BUY' if float(trade.get('currentUnits', 0)) > 0 else 'SELL'
                entry_price = float(trade.get('price', 0))
                unrealized = float(trade.get('unrealizedPL', 0))

                if not trade_id:
                    continue

                # Close this trade on OANDA
                success, result = self.trade_executor.close_trade(trade_id)

                if success:
                    closed_count += 1
                    total_closed += 1

                    # Get realized P&L from close result
                    realized_pnl = 0.0
                    if 'orderFillTransaction' in result:
                        realized_pnl = float(result['orderFillTransaction'].get('pl', 0))
                    else:
                        realized_pnl = unrealized  # Use unrealized as estimate

                    symbol_pnl += realized_pnl
                    total_pnl += realized_pnl

                    print(f"  [X] Trade {trade_id}: {symbol} {direction} {units:.0f} @ {entry_price:.5f} | P&L: ${realized_pnl:+.2f}")
                else:
                    error = result.get('error', 'Unknown')
                    print(f"  [!] Failed to close trade {trade_id}: {error}")

            if closed_count > 0:
                print(f"  >>> {symbol}: Closed {closed_count} trade(s) | Total P&L: ${symbol_pnl:+.2f}")

        # Clear local position tracking
        self.position_manager.positions.clear()

        print(f"\n[DONE] Closed {total_closed} trade(s) | Total Realized P&L: ${total_pnl:+.2f}")
