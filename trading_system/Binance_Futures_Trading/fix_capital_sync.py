#!/usr/bin/env python3
"""
Fix: Make trading capital sync with ACTUAL Binance balance on startup.
Instead of using hardcoded initial_capital, fetch real balance.
"""

import os

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

# New _load_reserve_fund that syncs with actual balance
NEW_LOAD_RESERVE_FUND = '''    def _load_reserve_fund(self):
        """
        Load reserve fund state from file.
        IMPORTANT: On startup, sync trading_capital with ACTUAL Binance balance
        to prevent sizing based on outdated/wrong capital.
        """
        try:
            # FIRST: Get ACTUAL balance from Binance
            actual_balance = self.client.get_balance()

            if os.path.exists(self.reserve_file):
                with open(self.reserve_file, 'r') as f:
                    data = json.load(f)
                    self.reserve_fund = data.get("reserve_fund", 0.0)
                    self.total_realized_profit = data.get("total_realized_profit", 0.0)
                    saved_trading_capital = data.get("trading_capital", self.initial_capital)
                    saved_initial = data.get("initial_capital", self.initial_capital)

                    # CHECK: If actual balance is significantly different from saved capital,
                    # reset to actual balance (prevents sizing based on wrong capital)
                    capital_diff = abs(actual_balance - saved_trading_capital)

                    if capital_diff > 10:  # More than $10 difference = reset to actual
                        self.log(f"[SMART COMPOUND] WARNING: Saved capital ${saved_trading_capital:.2f} differs from actual ${actual_balance:.2f}")
                        self.log(f"[SMART COMPOUND] RESETTING to actual balance ${actual_balance:.2f}")
                        self.trading_capital = actual_balance
                        self.initial_capital = actual_balance
                        self.reserve_fund = 0.0
                        self.total_realized_profit = 0.0
                        self._save_reserve_fund()  # Save the reset state
                    else:
                        # Normal load - use saved values
                        self.trading_capital = saved_trading_capital
                        self.log(f"[SMART COMPOUND] Loaded: Trading=${self.trading_capital:.2f} | Reserve=${self.reserve_fund:.2f}")
            else:
                # No saved file - initialize from actual balance
                self.log(f"[SMART COMPOUND] No saved state - initializing from actual balance ${actual_balance:.2f}")
                self.trading_capital = actual_balance
                self.initial_capital = actual_balance
                self._save_reserve_fund()

        except Exception as e:
            self.log(f"[SMART COMPOUND] Error loading reserve fund: {e}", level="WARN")
            # Fallback to actual balance on error
            try:
                actual_balance = self.client.get_balance()
                self.trading_capital = actual_balance
                self.initial_capital = actual_balance
                self.log(f"[SMART COMPOUND] Fallback: Using actual balance ${actual_balance:.2f}")
            except:
                pass
'''

def apply_patch():
    print("=" * 60)
    print("FIXING CAPITAL SYNC - Use ACTUAL Balance")
    print("=" * 60)

    with open(LIVE_ENGINE_PATH, 'r') as f:
        content = f.read()

    # Find and replace the _load_reserve_fund function
    # Find the start
    start_marker = "    def _load_reserve_fund(self):"
    # Find the end (next method definition)
    end_marker = "    def _save_reserve_fund(self):"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print("✗ FAILED: Could not find _load_reserve_fund function")
        return False

    # Replace the function
    content = content[:start_idx] + NEW_LOAD_RESERVE_FUND + "\n\n" + content[end_idx:]

    with open(LIVE_ENGINE_PATH, 'w') as f:
        f.write(content)

    print("✓ Replaced _load_reserve_fund to sync with actual balance")
    print("=" * 60)
    print("Now the system will:")
    print("1. Fetch ACTUAL Binance balance on startup")
    print("2. If saved capital differs by >$10, reset to actual")
    print("3. Size positions based on REAL balance")
    print("=" * 60)

    return True

if __name__ == "__main__":
    apply_patch()
