#!/usr/bin/env python3
"""
Apply Strong Trend Mode PART 2 patch:
1. Add Strong Trend Mode check/activate in manage_positions
2. This ensures we check for strong trends on each cycle

Run this script on the VPS after apply_strong_trend.py
"""

import os

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

# Code to add at the start of manage_positions, after sync_positions()
STRONG_TREND_CHECK = '''
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

            # Skip if Boost Mode is active for this symbol (mutually exclusive)
            if self.boost_mode_active.get(sym, False):
                continue

            # Check for Strong Trend
            is_strong, direction = self.check_strong_trend_mode(sym)

            if is_strong and not self.strong_trend_mode.get(sym, False):
                # Activate Strong Trend Mode
                self.activate_strong_trend_mode(sym, direction)

            elif not is_strong and self.strong_trend_mode.get(sym, False):
                # Deactivate Strong Trend Mode
                self.deactivate_strong_trend_mode(sym, f"ADX dropped below {self.adx_threshold}")

'''

def apply_patch():
    print("=" * 60)
    print("APPLYING STRONG TREND MODE PATCH - PART 2")
    print("=" * 60)

    # Read the file
    with open(LIVE_ENGINE_PATH, 'r') as f:
        content = f.read()

    # Check if already patched
    if "STRONG TREND MODE: Check and update for each symbol" in content:
        print("ERROR: Part 2 patch already applied!")
        return False

    # Find the marker - right after self.sync_positions() in manage_positions
    marker = '''    def manage_positions(self):
        """Manage all open positions - check TP/SL, Trailing TP, Trend Change, and DCA"""
        # Sync with Binance
        self.sync_positions()

        # Check each position'''

    if marker in content:
        # Replace with marker + strong trend check
        new_code = '''    def manage_positions(self):
        """Manage all open positions - check TP/SL, Trailing TP, Trend Change, and DCA"""
        # Sync with Binance
        self.sync_positions()
''' + STRONG_TREND_CHECK + '''
        # Check each position'''
        content = content.replace(marker, new_code)
        print("✓ Added Strong Trend Mode check in manage_positions")
    else:
        print("✗ FAILED: Could not find manage_positions marker")
        return False

    # Write the patched file
    with open(LIVE_ENGINE_PATH, 'w') as f:
        f.write(content)

    print("=" * 60)
    print("PART 2 PATCH APPLIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nStrong Trend Mode will now:")
    print("1. Check ADX on each manage_positions cycle")
    print("2. Activate when ADX > 40")
    print("3. Deactivate when ADX drops below 40")

    return True

if __name__ == "__main__":
    apply_patch()
