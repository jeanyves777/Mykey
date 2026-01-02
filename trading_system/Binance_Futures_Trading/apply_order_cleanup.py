#!/usr/bin/env python3
"""
Add periodic order cleanup to the main loop.
This ensures orphaned/stale orders are cancelled regularly, not just at startup.
"""

import os

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

# Function to add for order cleanup
ORDER_CLEANUP_FUNCTION = '''
    def cleanup_orphaned_orders(self):
        """
        Clean up orphaned/stale orders that don't match current positions.
        This runs periodically to catch orders that weren't cancelled properly.

        Orphaned orders can occur when:
        - Bot crashed before cancelling orders
        - Network error during order cancellation
        - Position closed externally but orders remain
        """
        try:
            for symbol in self.symbols:
                # Get all open orders for this symbol
                open_orders = self.client.get_open_orders(symbol)
                if not open_orders:
                    continue

                # Get current positions for this symbol
                long_key = f"{symbol}_LONG"
                short_key = f"{symbol}_SHORT"
                long_pos = self.positions.get(long_key)
                short_pos = self.positions.get(short_key)

                for order in open_orders:
                    order_id = order.get("orderId")
                    order_side = order.get("side")  # BUY or SELL
                    order_position_side = order.get("positionSide", "BOTH")  # LONG, SHORT, or BOTH
                    order_type = order.get("type", "")

                    # Determine which position this order belongs to
                    is_orphaned = False

                    if self.hedge_mode:
                        # In hedge mode, check positionSide
                        if order_position_side == "LONG":
                            if not long_pos:
                                is_orphaned = True
                                self.log(f"[CLEANUP] Found orphaned LONG order for {symbol}: {order_type} #{order_id}")
                            elif long_pos.stop_loss_order_id != order_id and long_pos.take_profit_order_id != order_id:
                                # Order exists but not tracked by our position
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
                        # Normal mode - check if any position exists
                        pos = self.positions.get(symbol)
                        if not pos:
                            is_orphaned = True
                            self.log(f"[CLEANUP] Found orphaned order for {symbol}: {order_type} #{order_id}")

                    # Cancel orphaned order
                    if is_orphaned:
                        try:
                            # Check if it's an algo order (TP/SL)
                            is_algo = order_type in ["TAKE_PROFIT_MARKET", "STOP_MARKET", "TRAILING_STOP_MARKET"]
                            self.client.cancel_order(symbol, order_id, is_algo_order=is_algo)
                            self.log(f"[CLEANUP] Cancelled orphaned order #{order_id} for {symbol}")
                        except Exception as e:
                            self.log(f"[CLEANUP] Could not cancel order #{order_id}: {e}", level="WARN")

        except Exception as e:
            self.log(f"[CLEANUP] Error during order cleanup: {e}", level="WARN")

'''

# Tracking variable for cleanup interval
CLEANUP_TRACKING_VAR = '''
        # Order cleanup tracking
        self.last_order_cleanup: datetime = datetime.now()
        self.order_cleanup_interval = 300  # Clean up orphaned orders every 5 minutes
'''

# Code to add in main loop for periodic cleanup
MAIN_LOOP_CLEANUP = '''
                # PERIODIC ORDER CLEANUP - Cancel orphaned/stale orders
                if (datetime.now() - self.last_order_cleanup).total_seconds() >= self.order_cleanup_interval:
                    self.cleanup_orphaned_orders()
                    self.last_order_cleanup = datetime.now()

'''

def apply_patch():
    print("=" * 60)
    print("APPLYING ORDER CLEANUP PATCH")
    print("=" * 60)

    with open(LIVE_ENGINE_PATH, 'r') as f:
        content = f.read()

    # Check if already patched
    if "cleanup_orphaned_orders" in content:
        print("ERROR: Order cleanup already exists in the file!")
        return False

    # PATCH 1: Add tracking variable after order_cleanup_interval or similar
    # Find a good spot - after self.running = False in __init__
    marker1 = "self.running = False"
    if marker1 in content:
        # Find the first occurrence (in __init__)
        idx = content.find(marker1)
        # Find the end of that line
        end_idx = content.find("\n", idx)
        if end_idx > 0:
            content = content[:end_idx+1] + CLEANUP_TRACKING_VAR + content[end_idx+1:]
            print("✓ PATCH 1: Added order cleanup tracking variables")
    else:
        print("✗ PATCH 1 FAILED: Could not find self.running marker")
        return False

    # PATCH 2: Add the cleanup function before manage_positions
    marker2 = "    def manage_positions(self):"
    if marker2 in content:
        content = content.replace(marker2, ORDER_CLEANUP_FUNCTION + "\n" + marker2)
        print("✓ PATCH 2: Added cleanup_orphaned_orders function")
    else:
        print("✗ PATCH 2 FAILED: Could not find manage_positions marker")
        return False

    # PATCH 3: Add periodic cleanup call in main loop
    # Find the line with "self.manage_positions()"
    marker3 = "                self.manage_positions()"
    if marker3 in content:
        content = content.replace(marker3, MAIN_LOOP_CLEANUP + marker3)
        print("✓ PATCH 3: Added periodic cleanup call in main loop")
    else:
        print("✗ PATCH 3 FAILED: Could not find manage_positions call in loop")
        return False

    # Write patched file
    with open(LIVE_ENGINE_PATH, 'w') as f:
        f.write(content)

    print("=" * 60)
    print("ORDER CLEANUP PATCH APPLIED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFeatures added:")
    print("1. cleanup_orphaned_orders() function")
    print("2. Runs every 5 minutes in main loop")
    print("3. Cancels orders without matching positions")

    return True

if __name__ == "__main__":
    apply_patch()
