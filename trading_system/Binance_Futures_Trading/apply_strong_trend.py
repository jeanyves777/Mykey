#!/usr/bin/env python3
"""
Apply Strong Trend Mode patch to live_trading_engine.py
Run this script on the VPS to add Strong Trend Mode features.
"""

import os
import re

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

# Strong Trend Mode tracking variables to add after boost_multiplier
STRONG_TREND_VARS = '''
        # STRONG TREND MODE - ADX-based trend detection for DCA blocking
        # When ADX > 40, block DCA 2+ on loser side and 2x boost winner side
        # MUTUALLY EXCLUSIVE with Boost Mode (Boost Mode takes priority)
        self.strong_trend_mode: Dict[str, bool] = {}       # symbol -> True if strong trend active
        self.strong_trend_direction: Dict[str, str] = {}   # symbol -> "UP" or "DOWN"
        self.adx_threshold = 40                            # ADX > 40 = strong trend
        self.current_adx: Dict[str, float] = {}            # symbol -> current ADX value
        self.current_plus_di: Dict[str, float] = {}        # symbol -> current +DI value
        self.current_minus_di: Dict[str, float] = {}       # symbol -> current -DI value
        self.trend_boosted_side: Dict[str, str] = {}       # symbol -> side that got 2x trend boost
'''

# Strong Trend Mode functions (ADX calculation, trend detection, DCA blocking)
STRONG_TREND_FUNCTIONS = '''
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

        - ADX > 40 = Strong trend
        - +DI > -DI = UP trend (bullish)
        - -DI > +DI = DOWN trend (bearish)
        """
        adx, plus_di, minus_di = self.calculate_adx_for_symbol(symbol)

        # Store current values
        self.current_adx[symbol] = adx
        self.current_plus_di[symbol] = plus_di
        self.current_minus_di[symbol] = minus_di

        if adx < self.adx_threshold:
            return False, None

        # Determine trend direction
        if plus_di > minus_di:
            return True, "UP"
        else:
            return True, "DOWN"

    def activate_strong_trend_mode(self, symbol: str, direction: str):
        """Activate strong trend mode for a symbol."""
        # IMPORTANT: Don't activate if Boost Mode is already active
        # Boost Mode and Strong Trend Mode are MUTUALLY EXCLUSIVE
        if self.boost_mode_active.get(symbol, False):
            self.log(f"[STRONG TREND] {symbol}: NOT activating - Boost Mode already active", level="INFO")
            return False

        if self.strong_trend_mode.get(symbol, False):
            return False  # Already active

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

        DCA 2+ (level >= 2) is blocked on the LOSER side during strong trends:
        - In UP trend: SHORT DCA 2+ is blocked
        - In DOWN trend: LONG DCA 2+ is blocked

        Returns: (is_blocked, reason)
        """
        # Only block DCA 2+ (level >= 2)
        if dca_level < 2:
            return False, None

        # Check if strong trend mode is active
        if not self.strong_trend_mode.get(symbol, False):
            return False, None

        direction = self.strong_trend_direction.get(symbol)
        if not direction:
            return False, None

        # Determine if this position is on the loser side
        loser_side = "SHORT" if direction == "UP" else "LONG"

        if position_side == loser_side:
            adx = self.current_adx.get(symbol, 0)
            reason = f"Strong {direction} trend (ADX: {adx:.1f}) - {position_side} is loser side"
            return True, reason

        return False, None

'''

# DCA blocking code to add in check_dca function
DCA_BLOCKING_CODE = '''
            # ================================================================
            # STRONG TREND MODE: Block DCA 2+ on loser side
            # In strong trends (ADX > 40), don't add to losing positions
            # ================================================================
            is_blocked, block_reason = self.is_dca_blocked_by_strong_trend(symbol, position.side, dca_level)
            if is_blocked:
                self.log(f"[STRONG TREND] {symbol} {position.side} DCA {dca_level} BLOCKED - {block_reason}", level="DCA")
                return

'''

def apply_patch():
    print("=" * 60)
    print("APPLYING STRONG TREND MODE PATCH")
    print("=" * 60)

    # Read the file
    with open(LIVE_ENGINE_PATH, 'r') as f:
        content = f.read()

    # Check if already patched
    if "strong_trend_mode: Dict" in content:
        print("ERROR: Strong Trend Mode already exists in the file!")
        return False

    # PATCH 1: Add tracking variables after boost_multiplier
    marker1 = "self.boost_multiplier = 1.5                       # 1.5x boost"
    if marker1 in content:
        content = content.replace(marker1, marker1 + STRONG_TREND_VARS)
        print("✓ PATCH 1: Added Strong Trend Mode tracking variables")
    else:
        print("✗ PATCH 1 FAILED: Could not find boost_multiplier marker")
        return False

    # PATCH 2: Add functions before STOP FOR DAY section
    marker2 = "        # STOP FOR DAY - After SL hit, stop trading that symbol"
    if marker2 in content:
        content = content.replace(marker2, STRONG_TREND_FUNCTIONS + "\n" + marker2)
        print("✓ PATCH 2: Added Strong Trend Mode functions")
    else:
        print("✗ PATCH 2 FAILED: Could not find STOP FOR DAY marker")
        return False

    # PATCH 3: Add DCA blocking in check_dca function
    # Find the HYBRID DCA FILTER section and add our check before it
    marker3 = "            # =================================================================\n            # HYBRID DCA FILTER"
    if marker3 in content:
        content = content.replace(marker3, DCA_BLOCKING_CODE + marker3)
        print("✓ PATCH 3: Added DCA blocking logic in check_dca")
    else:
        print("✗ PATCH 3 FAILED: Could not find HYBRID DCA FILTER marker")
        return False

    # PATCH 4: Add mutual exclusivity in _check_boost_activation
    # Find the early return check and add deactivation after it
    marker4 = '''        if self.boost_mode_active.get(symbol, False):
            return'''
    exclusivity_code = '''        if self.boost_mode_active.get(symbol, False):
            return

        # IMPORTANT: Deactivate Strong Trend Mode when activating Boost Mode
        # These two should NEVER be active together - Boost Mode takes priority
        if self.strong_trend_mode.get(symbol, False):
            self.deactivate_strong_trend_mode(symbol, "Boost Mode taking over")'''

    if marker4 in content:
        content = content.replace(marker4, exclusivity_code)
        print("✓ PATCH 4: Added Boost Mode / Strong Trend Mode mutual exclusivity")
    else:
        print("✗ PATCH 4 FAILED: Could not find boost activation check")
        return False

    # Write the patched file
    with open(LIVE_ENGINE_PATH, 'w') as f:
        f.write(content)

    print("=" * 60)
    print("PATCH APPLIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nStrong Trend Mode features added:")
    print("1. ADX > 40 = Strong Trend detected")
    print("2. In UP trend: SHORT DCA 2+ is BLOCKED")
    print("3. In DOWN trend: LONG DCA 2+ is BLOCKED")
    print("4. Boost Mode and Strong Trend Mode are mutually exclusive")

    return True

if __name__ == "__main__":
    apply_patch()
