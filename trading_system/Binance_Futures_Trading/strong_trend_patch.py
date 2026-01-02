#!/usr/bin/env python3
"""
Strong Trend Mode Patch for live_trading_engine.py
==================================================
This script patches the live trading engine to add Strong Trend Mode:

1. When ADX > 40 = Strong Trend detected
2. Direction: +DI > -DI = UP trend, -DI > +DI = DOWN trend
3. In UP trend: LONG wins (2x), SHORT loses (DCA 2+ BLOCKED)
4. In DOWN trend: SHORT wins (2x), LONG loses (DCA 2+ BLOCKED)
5. Strong Trend Mode and Boost Mode are MUTUALLY EXCLUSIVE

To apply: Run this script from the trading system directory
"""

import os
import re

# Path to the live trading engine
LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def backup_file(path):
    backup_path = path + ".backup"
    content = read_file(path)
    write_file(backup_path, content)
    print(f"Backup created: {backup_path}")

# =====================================================================
# PATCH 1: Add Strong Trend Mode tracking variables after boost mode
# =====================================================================
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

# =====================================================================
# PATCH 2: ADX calculation function
# =====================================================================
ADX_CALC_FUNCTION = '''
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
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, 0.0001)  # Avoid division by zero
            dx = 100 * abs(plus_di - minus_di) / di_sum

            # Calculate ADX (smoothed DX)
            adx = dx.ewm(alpha=1/period, min_periods=period).mean()

            # Return latest values
            adx_val = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
            plus_di_val = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0.0
            minus_di_val = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0.0

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
        self.log(f"[STRONG TREND]     Winner ({self._get_trend_winner(direction)}): 2x boost | Loser ({self._get_trend_loser(direction)}): DCA 2+ BLOCKED")

        return True

    def deactivate_strong_trend_mode(self, symbol: str, reason: str):
        """Deactivate strong trend mode for a symbol."""
        if not self.strong_trend_mode.get(symbol, False):
            return

        self.log(f"[STRONG TREND] >>> {symbol} ENDED - {reason}")
        self.strong_trend_mode[symbol] = False
        self.strong_trend_direction[symbol] = None
        self.trend_boosted_side[symbol] = None

    def _get_trend_winner(self, direction: str) -> str:
        """Get the winning side for a trend direction."""
        return "LONG" if direction == "UP" else "SHORT"

    def _get_trend_loser(self, direction: str) -> str:
        """Get the losing side for a trend direction."""
        return "SHORT" if direction == "UP" else "LONG"

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
        loser_side = self._get_trend_loser(direction)

        if position_side == loser_side:
            adx = self.current_adx.get(symbol, 0)
            reason = f"Strong {direction} trend (ADX: {adx:.1f}) - {position_side} is loser side"
            return True, reason

        return False, None

'''

# =====================================================================
# PATCH 3: DCA blocking logic (to insert in check_dca function)
# =====================================================================
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

# =====================================================================
# PATCH 4: Boost mode mutual exclusivity check
# =====================================================================
BOOST_EXCLUSIVITY_CODE = '''
        # IMPORTANT: Deactivate Strong Trend Mode when activating Boost Mode
        # These two should NEVER be active together - Boost Mode takes priority
        if self.strong_trend_mode.get(symbol, False):
            self.deactivate_strong_trend_mode(symbol, "Boost Mode taking over")
'''

def apply_patches():
    """Apply all patches to the live trading engine."""
    print("=" * 60)
    print("STRONG TREND MODE PATCH")
    print("=" * 60)

    # Read current file
    content = read_file(LIVE_ENGINE_PATH)

    # Backup first
    backup_file(LIVE_ENGINE_PATH)

    # Check if already patched
    if "strong_trend_mode" in content:
        print("WARNING: Strong Trend Mode variables already exist!")
        print("File may already be patched. Aborting.")
        return False

    # PATCH 1: Add tracking variables after boost mode variables
    # Find the line with "self.boost_multiplier = 1.5"
    marker1 = "self.boost_multiplier = 1.5"
    if marker1 in content:
        content = content.replace(
            marker1,
            marker1 + STRONG_TREND_VARS
        )
        print("✓ PATCH 1: Added Strong Trend Mode tracking variables")
    else:
        print("✗ PATCH 1: Could not find boost_multiplier marker")
        return False

    # PATCH 2: Add ADX calculation function after boost mode functions
    # Find the marker "# STOP FOR DAY" section
    marker2 = "# STOP FOR DAY - After SL hit"
    if marker2 in content:
        # Find the line before STOP FOR DAY
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if marker2 in line:
                # Insert the functions before this line
                lines.insert(i, ADX_CALC_FUNCTION)
                break
        content = '\n'.join(lines)
        print("✓ PATCH 2: Added ADX calculation and Strong Trend Mode functions")
    else:
        print("✗ PATCH 2: Could not find STOP FOR DAY marker")
        return False

    # PATCH 3: Add DCA blocking in check_dca function
    # Find the hybrid DCA filter check and add our check before it
    marker3 = "# =================================================================\n            # HYBRID DCA FILTER"
    if marker3 in content:
        content = content.replace(
            marker3,
            DCA_BLOCKING_CODE + "\n            " + marker3
        )
        print("✓ PATCH 3: Added DCA blocking logic for Strong Trend Mode")
    else:
        # Try alternative marker
        marker3_alt = "# HYBRID DCA FILTER: Easy for L1-2"
        if marker3_alt in content:
            # Find the line with this marker and add our code before it
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if marker3_alt in line:
                    # Find the comment line before it (with ===)
                    if i > 0 and "===" in lines[i-1]:
                        lines.insert(i-1, DCA_BLOCKING_CODE)
                    else:
                        lines.insert(i, DCA_BLOCKING_CODE)
                    break
            content = '\n'.join(lines)
            print("✓ PATCH 3: Added DCA blocking logic (alternative marker)")
        else:
            print("✗ PATCH 3: Could not find HYBRID DCA FILTER marker")
            return False

    # PATCH 4: Add mutual exclusivity in boost activation
    marker4 = "def _check_boost_activation"
    if marker4 in content:
        # Find the function and add exclusivity check after "Already in boost mode" check
        marker4_body = "if self.boost_mode_active.get(symbol, False):\n            return"
        if marker4_body in content:
            content = content.replace(
                marker4_body,
                marker4_body + "\n" + BOOST_EXCLUSIVITY_CODE
            )
            print("✓ PATCH 4: Added Boost Mode / Strong Trend Mode mutual exclusivity")
        else:
            print("✗ PATCH 4: Could not find boost activation check")
            return False
    else:
        print("✗ PATCH 4: Could not find _check_boost_activation function")
        return False

    # Write patched file
    write_file(LIVE_ENGINE_PATH, content)

    print("=" * 60)
    print("PATCH APPLIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nStrong Trend Mode features added:")
    print("1. ADX > 40 = Strong Trend detected")
    print("2. Winner side (LONG in UP, SHORT in DOWN): Gets 2x boost")
    print("3. Loser side (SHORT in UP, LONG in DOWN): DCA 2+ BLOCKED")
    print("4. Boost Mode and Strong Trend Mode are mutually exclusive")
    print("\nBackup saved as: live_trading_engine.py.backup")

    return True

if __name__ == "__main__":
    apply_patches()
