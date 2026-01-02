#!/usr/bin/env python3
"""
Apply ALL patches to live_trading_engine.py:
1. Strong Trend Mode (ADX > 40 detection, DCA blocking)
2. Periodic order cleanup (every 5 minutes)

This script carefully inserts code at the right locations.
"""

import os
import re

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

def apply_patches():
    print("=" * 60)
    print("APPLYING ALL PATCHES TO LIVE TRADING ENGINE")
    print("=" * 60)

    with open(LIVE_ENGINE_PATH, 'r') as f:
        content = f.read()

    # Check if already patched
    if "strong_trend_mode: Dict" in content:
        print("ERROR: Patches already applied!")
        return False

    # =========================================================================
    # PATCH 1: Add Strong Trend Mode tracking variables
    # Insert after "self.boost_multiplier = 1.5" line
    # =========================================================================
    strong_trend_vars = '''
        # STRONG TREND MODE - ADX-based trend detection for DCA blocking
        # When ADX > 40, block DCA 2+ on loser side and 2x boost winner side
        # MUTUALLY EXCLUSIVE with Boost Mode (Boost Mode takes priority)
        self.strong_trend_mode: Dict[str, bool] = {}       # symbol -> True if strong trend active
        self.strong_trend_direction: Dict[str, str] = {}   # symbol -> "UP" or "DOWN"
        self.adx_threshold = 40                            # ADX > 40 = strong trend
        self.current_adx: Dict[str, float] = {}            # symbol -> current ADX value
        self.current_plus_di: Dict[str, float] = {}        # symbol -> current +DI value
        self.current_minus_di: Dict[str, float] = {}       # symbol -> current -DI value
        self.trend_boosted_side: Dict[str, str] = {}       # symbol -> side that got 2x trend boost'''

    marker1 = "self.boost_multiplier = 1.5                       # 1.5x boost"
    if marker1 in content:
        content = content.replace(marker1, marker1 + strong_trend_vars)
        print("✓ PATCH 1: Added Strong Trend Mode tracking variables")
    else:
        print("✗ PATCH 1 FAILED: Could not find boost_multiplier marker")
        return False

    # =========================================================================
    # PATCH 2: Add order cleanup tracking variables
    # Insert after "self.running = False" in __init__ (first occurrence only)
    # =========================================================================
    order_cleanup_vars = '''

        # Order cleanup tracking
        self.last_order_cleanup: datetime = datetime.now()
        self.order_cleanup_interval = 300  # Clean up orphaned orders every 5 minutes'''

    # Find the first "self.running = False" followed by method definition
    pattern = r'(        self\.running = False\n\n    def log\(self)'
    match = re.search(pattern, content)
    if match:
        content = content[:match.start()] + "        self.running = False" + order_cleanup_vars + "\n\n    def log(self" + content[match.end():]
        print("✓ PATCH 2: Added order cleanup tracking variables")
    else:
        print("✗ PATCH 2 FAILED: Could not find self.running pattern")
        return False

    # =========================================================================
    # PATCH 3: Add Strong Trend Mode functions
    # Insert BEFORE "def _get_boost_trigger_level" (which is a method, not in __init__)
    # =========================================================================
    strong_trend_functions = '''
    # =========================================================================
    # STRONG TREND MODE - ADX-BASED TREND DETECTION
    # =========================================================================
    # When ADX > 40, we're in a strong trend:
    # - Winner side (LONG in UP, SHORT in DOWN): Gets 2x entry boost
    # - Loser side (SHORT in UP, LONG in DOWN): DCA 2+ is BLOCKED
    # This prevents adding to losing positions in strong trending markets

    def calculate_adx_for_symbol(self, symbol: str, df=None) -> tuple:
        """
        Calculate ADX (Average Directional Index) for a symbol.
        Returns: (adx, plus_di, minus_di)
        """
        try:
            # Get market data if not provided
            if df is None:
                if symbol in self.data_buffer and self.data_buffer[symbol] is not None:
                    if isinstance(self.data_buffer[symbol], dict):
                        df = self.data_buffer[symbol].get("1m")
                    else:
                        df = self.data_buffer[symbol]

                if df is None:
                    market_data = self.client.get_market_data(symbol)
                    if market_data and "1m" in market_data:
                        df = market_data["1m"]

            if df is None or len(df) < 28:
                return 0.0, 0.0, 0.0

            period = 14
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)

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
            plus_dm_copy = plus_dm.copy()
            minus_dm_copy = minus_dm.copy()
            plus_dm_copy[(plus_dm < minus_dm)] = 0
            minus_dm_copy[(minus_dm < plus_dm)] = 0
            plus_dm = plus_dm_copy
            minus_dm = minus_dm_copy

            # Smoothed TR, +DM, -DM using Wilder's smoothing
            atr = tr.ewm(alpha=1/period, min_periods=period).mean()
            plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

            # Calculate +DI and -DI
            plus_di = 100 * plus_dm_smooth / atr
            minus_di = 100 * minus_dm_smooth / atr

            # Calculate DX
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, 0.0001)  # Avoid division by zero
            dx = 100 * abs(plus_di - minus_di) / di_sum

            # Calculate ADX (smoothed DX)
            adx = dx.ewm(alpha=1/period, min_periods=period).mean()

            # Return latest values
            adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
            plus_di_val = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
            minus_di_val = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

            return adx_val, plus_di_val, minus_di_val

        except Exception as e:
            self.log(f"[STRONG TREND] ADX calculation error for {symbol}: {e}", level="WARN")
            return 0.0, 0.0, 0.0

    def check_strong_trend_mode(self, symbol: str) -> tuple:
        """
        Check if we're in a strong trend based on ADX.
        Returns: (is_strong_trend, trend_direction)
        """
        adx, plus_di, minus_di = self.calculate_adx_for_symbol(symbol)

        self.current_adx[symbol] = adx
        self.current_plus_di[symbol] = plus_di
        self.current_minus_di[symbol] = minus_di

        if adx < self.adx_threshold:
            return False, None

        if plus_di > minus_di:
            return True, "UP"
        else:
            return True, "DOWN"

    def activate_strong_trend_mode(self, symbol: str, direction: str):
        """Activate strong trend mode for a symbol."""
        if self.boost_mode_active.get(symbol, False):
            self.log(f"[STRONG TREND] {symbol}: NOT activating - Boost Mode already active", level="INFO")
            return False

        if self.strong_trend_mode.get(symbol, False):
            return False

        self.strong_trend_mode[symbol] = True
        self.strong_trend_direction[symbol] = direction

        adx = self.current_adx.get(symbol, 0)
        self.log(f"[STRONG TREND] >>> {symbol} ACTIVATED! Direction: {direction} | ADX: {adx:.1f}")
        winner = "LONG" if direction == "UP" else "SHORT"
        loser = "SHORT" if direction == "UP" else "LONG"
        self.log(f"[STRONG TREND]     Winner ({winner}): 2x boost | Loser ({loser}): DCA 2+ BLOCKED")

        return True

    def deactivate_strong_trend_mode(self, symbol: str, reason: str):
        """Deactivate strong trend mode for a symbol."""
        if not self.strong_trend_mode.get(symbol, False):
            return

        self.log(f"[STRONG TREND] >>> {symbol} ENDED - {reason}")
        self.strong_trend_mode[symbol] = False
        self.strong_trend_direction[symbol] = None
        self.trend_boosted_side[symbol] = None

    def is_dca_blocked_by_strong_trend(self, symbol: str, position_side: str, dca_level: int) -> tuple:
        """
        Check if DCA should be blocked due to Strong Trend Mode.
        DCA 2+ (level >= 2) is blocked on the LOSER side during strong trends.
        Returns: (is_blocked, reason)
        """
        if dca_level < 2:
            return False, None

        if not self.strong_trend_mode.get(symbol, False):
            return False, None

        direction = self.strong_trend_direction.get(symbol)
        if not direction:
            return False, None

        loser_side = "SHORT" if direction == "UP" else "LONG"

        if position_side == loser_side:
            adx = self.current_adx.get(symbol, 0)
            reason = f"Strong {direction} trend (ADX: {adx:.1f}) - {position_side} is loser side"
            return True, reason

        return False, None

    def cleanup_orphaned_orders(self):
        """
        Clean up orphaned/stale orders that don't match current positions.
        Runs periodically to catch orders that weren't cancelled properly.
        """
        try:
            for symbol in self.symbols:
                open_orders = self.client.get_open_orders(symbol)
                if not open_orders:
                    continue

                long_key = f"{symbol}_LONG"
                short_key = f"{symbol}_SHORT"
                long_pos = self.positions.get(long_key)
                short_pos = self.positions.get(short_key)

                for order in open_orders:
                    order_id = order.get("orderId")
                    order_position_side = order.get("positionSide", "BOTH")
                    order_type = order.get("type", "")

                    is_orphaned = False

                    if self.hedge_mode:
                        if order_position_side == "LONG":
                            if not long_pos:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found orphaned LONG order for {symbol}: {order_type} #{order_id}")
                            elif long_pos.stop_loss_order_id != order_id and long_pos.take_profit_order_id != order_id:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found untracked LONG order for {symbol}: {order_type} #{order_id}")
                        elif order_position_side == "SHORT":
                            if not short_pos:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found orphaned SHORT order for {symbol}: {order_type} #{order_id}")
                            elif short_pos.stop_loss_order_id != order_id and short_pos.take_profit_order_id != order_id:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found untracked SHORT order for {symbol}: {order_type} #{order_id}")
                    else:
                        pos = self.positions.get(symbol)
                        if not pos:
                            is_orphaned = True
                            self.log(f"[CLEANUP] Found orphaned order for {symbol}: {order_type} #{order_id}")

                    if is_orphaned:
                        try:
                            is_algo = order_type in ["TAKE_PROFIT_MARKET", "STOP_MARKET", "TRAILING_STOP_MARKET"]
                            self.client.cancel_order(symbol, order_id, is_algo_order=is_algo)
                            self.log(f"[CLEANUP] Cancelled orphaned order #{order_id} for {symbol}")
                        except Exception as e:
                            self.log(f"[CLEANUP] Could not cancel order #{order_id}: {e}", level="WARN")

        except Exception as e:
            self.log(f"[CLEANUP] Error during order cleanup: {e}", level="WARN")

'''

    marker3 = "    def _get_boost_trigger_level(self, symbol: str) -> int:"
    if marker3 in content:
        content = content.replace(marker3, strong_trend_functions + "\n" + marker3)
        print("✓ PATCH 3: Added Strong Trend Mode and order cleanup functions")
    else:
        print("✗ PATCH 3 FAILED: Could not find _get_boost_trigger_level marker")
        return False

    # =========================================================================
    # PATCH 4: Add DCA blocking in check_dca function
    # =========================================================================
    dca_blocking = '''            # ================================================================
            # STRONG TREND MODE: Block DCA 2+ on loser side
            # In strong trends (ADX > 40), don't add to losing positions
            # ================================================================
            is_blocked, block_reason = self.is_dca_blocked_by_strong_trend(symbol, position.side, dca_level)
            if is_blocked:
                self.log(f"[STRONG TREND] {symbol} {position.side} DCA {dca_level} BLOCKED - {block_reason}", level="DCA")
                return

'''

    marker4 = "            # =================================================================\n            # HYBRID DCA FILTER"
    if marker4 in content:
        content = content.replace(marker4, dca_blocking + marker4)
        print("✓ PATCH 4: Added DCA blocking logic in check_dca")
    else:
        print("✗ PATCH 4 FAILED: Could not find HYBRID DCA FILTER marker")
        return False

    # =========================================================================
    # PATCH 5: Add mutual exclusivity in _check_boost_activation
    # =========================================================================
    exclusivity = '''        # IMPORTANT: Deactivate Strong Trend Mode when activating Boost Mode
        # These two should NEVER be active together - Boost Mode takes priority
        if self.strong_trend_mode.get(symbol, False):
            self.deactivate_strong_trend_mode(symbol, "Boost Mode taking over")

'''

    marker5 = '''        if self.boost_mode_active.get(symbol, False):
            return

        # Check if this DCA level triggers boost'''

    if marker5 in content:
        new_marker5 = '''        if self.boost_mode_active.get(symbol, False):
            return

''' + exclusivity + '''        # Check if this DCA level triggers boost'''
        content = content.replace(marker5, new_marker5)
        print("✓ PATCH 5: Added Boost/Strong Trend mutual exclusivity")
    else:
        print("✗ PATCH 5 FAILED: Could not find boost activation marker")
        return False

    # =========================================================================
    # PATCH 6: Add Strong Trend check in manage_positions
    # =========================================================================
    trend_check = '''
        # ================================================================
        # STRONG TREND MODE: Check and update for each symbol
        # When ADX > 40, activate Strong Trend Mode and block DCA 2+ on loser side
        # ================================================================
        processed_symbols = set()
        for pos_key in list(self.positions.keys()):
            sym = self.get_symbol_from_key(pos_key)
            if sym in processed_symbols:
                continue
            processed_symbols.add(sym)

            if self.boost_mode_active.get(sym, False):
                continue

            is_strong, direction = self.check_strong_trend_mode(sym)

            if is_strong and not self.strong_trend_mode.get(sym, False):
                self.activate_strong_trend_mode(sym, direction)
            elif not is_strong and self.strong_trend_mode.get(sym, False):
                self.deactivate_strong_trend_mode(sym, f"ADX dropped below {self.adx_threshold}")

'''

    marker6 = '''    def manage_positions(self):
        """Manage all open positions - check TP/SL, Trailing TP, Trend Change, and DCA"""
        # Sync with Binance
        self.sync_positions()

        # Check each position'''

    if marker6 in content:
        new_marker6 = '''    def manage_positions(self):
        """Manage all open positions - check TP/SL, Trailing TP, Trend Change, and DCA"""
        # Sync with Binance
        self.sync_positions()
''' + trend_check + '''
        # Check each position'''
        content = content.replace(marker6, new_marker6)
        print("✓ PATCH 6: Added Strong Trend check in manage_positions")
    else:
        print("✗ PATCH 6 FAILED: Could not find manage_positions marker")
        return False

    # =========================================================================
    # PATCH 7: Add periodic order cleanup in main loop
    # =========================================================================
    cleanup_call = '''
                # PERIODIC ORDER CLEANUP - Cancel orphaned/stale orders
                if (datetime.now() - self.last_order_cleanup).total_seconds() >= self.order_cleanup_interval:
                    self.cleanup_orphaned_orders()
                    self.last_order_cleanup = datetime.now()

'''

    marker7 = "                self.manage_positions()\n\n                if (datetime.now() - last_status)"
    if marker7 in content:
        new_marker7 = "                self.manage_positions()\n" + cleanup_call + "                if (datetime.now() - last_status)"
        content = content.replace(marker7, new_marker7)
        print("✓ PATCH 7: Added periodic order cleanup in main loop")
    else:
        print("✗ PATCH 7 FAILED: Could not find main loop marker")
        return False

    # Write the patched file
    with open(LIVE_ENGINE_PATH, 'w') as f:
        f.write(content)

    print("=" * 60)
    print("ALL PATCHES APPLIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFeatures added:")
    print("1. Strong Trend Mode (ADX > 40)")
    print("2. DCA 2+ blocking on loser side in strong trends")
    print("3. Boost Mode / Strong Trend Mode mutual exclusivity")
    print("4. Periodic order cleanup every 5 minutes")

    return True

if __name__ == "__main__":
    apply_patches()
