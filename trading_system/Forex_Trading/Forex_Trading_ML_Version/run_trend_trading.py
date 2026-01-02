"""
Run Signal-Based Trading - NO DCA
==================================

Signal-based entries using ADX + Volatility + HTF confirmation.

Strategy:
1. Wait for entry signal:
   - ADX > 25 (trending market)
   - ATR expansion (volatility spike)
   - H1 EMA50 vs EMA200 confirms direction
2. Open position in signal direction
3. 20 pips Take Profit
4. 15 pips Stop Loss
5. Trailing stop: activates at +10 pips, trails 8 pips behind

NO DCA - Single entry only!

Usage:
    cd C:/Users/Jean-Yves/thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.run_trend_trading
"""

import warnings
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from .config.trading_config import load_config
from .data.oanda_client import OandaClient
from .data.data_loader import ForexDataLoader
from .risk_management.position_manager import PositionManager
from .risk_management.trade_executor import TradeExecutor
from .risk_management.trend_filter import TrendFilter


class SignalTradingEngine:
    """Signal-based trading engine - ADX + Volatility + HTF confirmation."""

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

        # Track position info for trailing stop
        self.position_open_times = {}
        self.trailing_data = {}  # {symbol: {'peak_profit_pips': x, 'trailing_active': bool}}

        # Signal tracking
        self.last_signal = {}  # {symbol: {'direction': x, 'time': x}}

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """Calculate ADX (Average Directional Index).
        Returns: (adx, plus_di, minus_di) as Series
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

        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)

        # When +DM > -DM, -DM = 0 and vice versa
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

        # Smoothed TR, +DM, -DM using Wilder's smoothing
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()

        return adx, plus_di, minus_di

    def check_entry_signal(self, symbol: str, df: pd.DataFrame) -> tuple:
        """Check for entry signal using ADX + ATR + HTF confirmation.

        Returns: (signal_direction, reason) where signal_direction is 'BUY', 'SELL', or None
        """
        dca = self.config.dca

        if len(df) < 50:
            return None, "Insufficient data"

        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(df, dca.adx_period)
        atr = self.calculate_atr(df, dca.atr_period)
        atr_avg = atr.rolling(window=20).mean()

        # Get latest values
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_atr = atr.iloc[-1]
        current_atr_avg = atr_avg.iloc[-1]

        if pd.isna(current_adx) or pd.isna(current_atr) or pd.isna(current_atr_avg):
            return None, "Indicators not ready"

        # Condition 1: ADX above threshold (trending market)
        if current_adx < dca.adx_threshold:
            return None, f"ADX {current_adx:.1f} < {dca.adx_threshold} (ranging)"

        # Condition 2: Volatility expansion (ATR > 1.5x average)
        if current_atr < current_atr_avg * dca.atr_expansion_mult:
            return None, f"ATR not expanding ({current_atr:.5f} < {current_atr_avg * dca.atr_expansion_mult:.5f})"

        # Condition 3: Get H1 trend direction
        htf_trend, htf_reason = self.get_h1_trend(symbol)

        if htf_trend not in ['BUY', 'SELL']:
            return None, f"No HTF trend ({htf_reason})"

        # Condition 4: DI confirms direction matches HTF
        if htf_trend == 'BUY' and current_plus_di > current_minus_di:
            reason = f"SIGNAL: ADX={current_adx:.1f}, +DI>{'-'}DI, HTF=UP, ATR expanding"
            return 'BUY', reason
        elif htf_trend == 'SELL' and current_minus_di > current_plus_di:
            reason = f"SIGNAL: ADX={current_adx:.1f}, -DI>{'+'}DI, HTF=DOWN, ATR expanding"
            return 'SELL', reason

        return None, f"DI direction ({'+' if current_plus_di > current_minus_di else '-'}) doesn't match HTF ({htf_trend})"

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

    def open_position(self, symbol: str, direction: str) -> bool:
        """Open a momentum position - NO DCA, just single entry with trailing."""
        prices = self.client.get_pricing([symbol])
        if not prices or symbol not in prices:
            print(f"  {symbol}: Failed to get price")
            return False

        entry_price = prices[symbol]['ask'] if direction == 'BUY' else prices[symbol]['bid']
        pip_value = self.config.get_pip_value(symbol)

        # Momentum strategy settings
        dca = self.config.dca
        sl_pips = dca.initial_sl_pips    # 15 pips
        tp_pips = dca.take_profit_pips   # 20 pips

        # Calculate SL/TP
        if direction == 'BUY':
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
        else:
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)

        # Position size - FULL SIZE (no DCA reserve)
        risk_pct = self.config.risk.position_size_pct
        risk_amt = self.equity * risk_pct
        units = int(risk_amt / (sl_pips * pip_value))

        # Minimum units check
        if units < 1:
            units = 1

        print(f"  {symbol}: Opening {direction} @ {self.format_price(symbol, entry_price)}")
        print(f"           Units: {units:,} | SL: {sl_pips:.0f}p | TP: {tp_pips:.0f}p")

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

            # Track open time
            self.position_open_times[symbol] = datetime.now()

            # Initialize trailing data
            self.trailing_data[symbol] = {
                'entry_price': fill_price,
                'peak_profit_pips': 0.0,
                'trailing_active': False,
                'trailing_sl': None
            }

            print(f"  {symbol}: >>> OPENED {direction} <<< (Trade ID: {trade_id})")
            print(f"           Trailing activates at +10p, trails 8p behind")

            return True
        else:
            print(f"  {symbol}: FAILED - {result.get('error', 'Unknown')}")
            return False

    def close_position(self, symbol: str, reason: str) -> bool:
        """Close position for a symbol."""
        # Close the position on OANDA
        success, result = self.trade_executor.close_position(symbol)

        if success:
            print(f"  {symbol}: >>> CLOSED ({reason}) <<<")
            # Clean up tracking
            if symbol in self.position_open_times:
                del self.position_open_times[symbol]
            if symbol in self.trailing_data:
                del self.trailing_data[symbol]
            self.position_manager.close_position(symbol)
            return True
        else:
            print(f"  {symbol}: Close FAILED - {result.get('error', 'Unknown')}")
            return False

    def manage_trailing_stop(self, symbol: str, trade: dict) -> None:
        """Manage trailing stop for a position.

        Trailing activates at +10 pips and trails 8 pips behind peak.
        """
        dca = self.config.dca
        pip_value = self.config.get_pip_value(symbol)

        # Get position info
        entry_price = float(trade.get('price', 0))
        current_units = float(trade.get('currentUnits', 0))
        direction = 'BUY' if current_units > 0 else 'SELL'
        trade_id = trade.get('id', '')

        # Get current price
        prices = self.client.get_pricing([symbol])
        if not prices or symbol not in prices:
            return

        current_price = prices[symbol]['bid'] if direction == 'BUY' else prices[symbol]['ask']

        # Calculate current profit in pips
        if direction == 'BUY':
            current_profit_pips = (current_price - entry_price) / pip_value
        else:
            current_profit_pips = (entry_price - current_price) / pip_value

        # Initialize trailing data if needed
        if symbol not in self.trailing_data:
            self.trailing_data[symbol] = {
                'entry_price': entry_price,
                'peak_profit_pips': 0.0,
                'trailing_active': False,
                'trailing_sl': None
            }

        trail_data = self.trailing_data[symbol]

        # Update peak profit
        if current_profit_pips > trail_data['peak_profit_pips']:
            trail_data['peak_profit_pips'] = current_profit_pips

        # Check if trailing should activate (at +10 pips)
        activation_pips = dca.trailing_activation_pips  # 10 pips
        trail_distance = dca.trailing_stop_pips         # 8 pips

        if not trail_data['trailing_active'] and trail_data['peak_profit_pips'] >= activation_pips:
            trail_data['trailing_active'] = True
            print(f"  {symbol}: >>> TRAILING ACTIVATED at +{trail_data['peak_profit_pips']:.1f} pips <<<")

        # If trailing is active, update the SL
        if trail_data['trailing_active']:
            # Calculate new trailing SL (peak - 8 pips)
            if direction == 'BUY':
                new_trailing_sl = entry_price + ((trail_data['peak_profit_pips'] - trail_distance) * pip_value)
            else:
                new_trailing_sl = entry_price - ((trail_data['peak_profit_pips'] - trail_distance) * pip_value)

            # Only update if trailing SL has moved in our favor
            current_sl = trail_data['trailing_sl']
            should_update = False

            if current_sl is None:
                should_update = True
            elif direction == 'BUY' and new_trailing_sl > current_sl:
                should_update = True
            elif direction == 'SELL' and new_trailing_sl < current_sl:
                should_update = True

            if should_update:
                # Update SL on OANDA
                success, result = self.trade_executor.modify_trade(
                    trade_id=trade_id,
                    stop_loss=new_trailing_sl,
                    symbol=symbol
                )

                if success:
                    trail_data['trailing_sl'] = new_trailing_sl
                    locked_pips = trail_data['peak_profit_pips'] - trail_distance
                    print(f"  {symbol}: Trailing SL → {self.format_price(symbol, new_trailing_sl)} (locks +{locked_pips:.1f}p)")

        return current_profit_pips

    def manage_position(self, symbol: str, trades: list) -> None:
        """Manage an open position - check trailing stop, update P&L."""
        if len(trades) == 0:
            return

        # Get the main trade (should only be 1 with no DCA)
        trade = trades[0]

        # Calculate current P&L
        total_pnl = sum(float(t.get('unrealizedPL', 0)) for t in trades)
        total_units = sum(abs(float(t.get('currentUnits', 0))) for t in trades)

        # Manage trailing stop
        current_profit_pips = self.manage_trailing_stop(symbol, trade)

        # Get trailing status
        trail_data = self.trailing_data.get(symbol, {})
        trailing_active = trail_data.get('trailing_active', False)
        peak_pips = trail_data.get('peak_profit_pips', 0)

        # Build status string
        if trailing_active:
            status = f"TRAILING | Peak: +{peak_pips:.1f}p | Now: {current_profit_pips:+.1f}p"
        else:
            status = f"Open | {current_profit_pips:+.1f}p (trail at +10p)"

        pnl_str = f"${total_pnl:+.2f}" if total_pnl != 0 else "$0.00"
        print(f"  {symbol}: {status} | P&L: {pnl_str}")

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
                del self.position_open_times[symbol]
                if symbol in self.trailing_data:
                    del self.trailing_data[symbol]
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
        print("SIGNAL-BASED TRADING - NO DCA")
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

        dca = self.config.dca
        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"Strategy: SIGNAL-BASED (ADX + ATR + HTF)")
        print(f"Entry: ADX > {dca.adx_threshold}, ATR > {dca.atr_expansion_mult}x avg")
        print(f"TP: {dca.take_profit_pips}p | SL: {dca.initial_sl_pips}p")
        print(f"Trailing: activates at +{dca.trailing_activation_pips}p, trails {dca.trailing_stop_pips}p behind")
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

                    # No position - check for entry signal
                    # Get M15 data for signal calculation
                    df_m15 = self.data_loader.get_latest_bars(symbol, count=100, granularity='M15')

                    if df_m15 is None or len(df_m15) < 50:
                        print(f"  {symbol}: Insufficient M15 data")
                        continue

                    signal_dir, signal_reason = self.check_entry_signal(symbol, df_m15)
                    print(f"  {symbol}: {signal_reason}")

                    if signal_dir in ['BUY', 'SELL']:
                        self.open_position(symbol, signal_dir)

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
    print("SIGNAL-BASED TRADING SYSTEM")
    print("ADX + ATR + HTF Confirmation - NO DCA")
    print("=" * 60)

    config = load_config()
    dca = config.dca

    print(f"\nSymbols: {', '.join(config.symbols)}")
    print(f"Strategy: SIGNAL-BASED (NO DCA)")
    print(f"Entry Conditions:")
    print(f"  - ADX > {dca.adx_threshold} (trending market)")
    print(f"  - ATR > {dca.atr_expansion_mult}x average (volatility spike)")
    print(f"  - H1 EMA{dca.htf_ema_fast} vs EMA{dca.htf_ema_slow} confirms direction")
    print(f"Take Profit: {dca.take_profit_pips}p")
    print(f"Stop Loss: {dca.initial_sl_pips}p")
    print(f"Trailing: activates +{dca.trailing_activation_pips}p, trails {dca.trailing_stop_pips}p behind")

    print("\n" + "-" * 60)
    response = input("Start trading? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return

    engine = SignalTradingEngine(config)
    engine.run(interval=60)


if __name__ == '__main__':
    main()
