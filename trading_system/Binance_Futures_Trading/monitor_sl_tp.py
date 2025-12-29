#!/usr/bin/env python3
"""
SL/TP Order Monitor & Auto-Fix Script
======================================
Monitors all open positions and ensures they have correct SL/TP orders.
If orders are missing or incorrect, it will fix them automatically.

Run this script periodically to catch any missed SL/TP updates after DCA.

Usage:
    python monitor_sl_tp.py           # Check and report (dry run)
    python monitor_sl_tp.py --fix     # Check and fix any issues
    python monitor_sl_tp.py --loop    # Run continuously every 60 seconds
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient
from config.trading_config import (
    FUTURES_SYMBOLS, STRATEGY_CONFIG, DCA_CONFIG, RISK_CONFIG, SYMBOL_SETTINGS
)


class SLTPMonitor:
    """Monitor and fix SL/TP orders for all open positions"""

    def __init__(self, testnet: bool = True):
        self.client = BinanceClient(testnet=testnet)
        self.testnet = testnet
        self.leverage = STRATEGY_CONFIG["leverage"]

    def log(self, message: str, level: str = "INFO"):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = "DEMO" if self.testnet else "LIVE"
        print(f"[{timestamp}] [{mode}] [{level}] {message}")

    def detect_dca_level_from_margin(self, symbol: str, actual_margin: float, budget: float) -> int:
        """
        Auto-detect DCA level based on margin used.
        Similar to the engine's logic for consistency.
        """
        if actual_margin <= 0 or budget <= 0:
            return 0

        # Calculate cumulative margin thresholds
        entry_pct = RISK_CONFIG.get("initial_entry_pct", 0.10)
        dca1_pct = RISK_CONFIG.get("dca1_pct", 0.15)
        dca2_pct = RISK_CONFIG.get("dca2_pct", 0.20)
        dca3_pct = RISK_CONFIG.get("dca3_pct", 0.25)

        # Cumulative thresholds (with 10% tolerance)
        threshold_0 = budget * entry_pct * 1.1
        threshold_1 = budget * (entry_pct + dca1_pct) * 1.1
        threshold_2 = budget * (entry_pct + dca1_pct + dca2_pct) * 1.1
        threshold_3 = budget * (entry_pct + dca1_pct + dca2_pct + dca3_pct) * 1.1

        if actual_margin > threshold_3:
            return 4
        elif actual_margin > threshold_2:
            return 3
        elif actual_margin > threshold_1:
            return 2
        elif actual_margin > threshold_0:
            return 1
        else:
            return 0

    def get_expected_sl_tp(self, entry_price: float, side: str, dca_level: int) -> tuple:
        """
        Calculate expected SL and TP prices based on entry price and DCA level.
        Returns (expected_sl, expected_tp, tp_roi, sl_roi)
        """
        leverage = self.leverage

        if DCA_CONFIG["enabled"]:
            if dca_level > 0 and dca_level <= len(DCA_CONFIG["levels"]):
                # Use reduced TP from DCA level config
                dca_level_config = DCA_CONFIG["levels"][dca_level - 1]
                tp_roi = dca_level_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])
                sl_roi = DCA_CONFIG["sl_after_dca_roi"]  # Tighter SL after DCA
            else:
                # Initial entry - use default TP/SL
                tp_roi = DCA_CONFIG["take_profit_roi"]
                sl_roi = DCA_CONFIG["stop_loss_roi"]

            tp_price_pct = tp_roi / leverage
            sl_price_pct = sl_roi / leverage
        else:
            tp_price_pct = STRATEGY_CONFIG["take_profit_pct"]
            sl_price_pct = STRATEGY_CONFIG["stop_loss_pct"]
            tp_roi = tp_price_pct
            sl_roi = sl_price_pct

        if side == "LONG":
            expected_sl = entry_price * (1 - sl_price_pct)
            expected_tp = entry_price * (1 + tp_price_pct)
        else:
            expected_sl = entry_price * (1 + sl_price_pct)
            expected_tp = entry_price * (1 - tp_price_pct)

        return expected_sl, expected_tp, tp_roi, sl_roi

    def check_all_positions(self, auto_fix: bool = False) -> dict:
        """
        Check all positions and their SL/TP orders.
        Returns a report of issues found.
        """
        report = {
            "positions_checked": 0,
            "positions_ok": 0,
            "issues_found": [],
            "fixed": []
        }

        # Get balance for budget calculation
        balance = self.client.get_balance()
        num_symbols = len(FUTURES_SYMBOLS)
        budget_per_symbol = balance * (1 - RISK_CONFIG.get("allocation_buffer_pct", 0.05)) / num_symbols

        # Get all open positions
        positions = self.client.get_positions()

        if not positions:
            self.log("No open positions found")
            return report

        self.log(f"Checking {len(positions)} position(s)...")
        print("=" * 70)

        for pos in positions:
            symbol = pos["symbol"]
            side = pos["side"]
            entry_price = float(pos.get("entry_price", 0))
            quantity = float(pos.get("quantity", 0))
            margin_used = float(pos.get("isolated_wallet", 0)) or float(pos.get("isolatedWallet", 0))
            liq_price = float(pos.get("liquidation_price", 0))

            report["positions_checked"] += 1

            if entry_price <= 0 or quantity <= 0:
                issue = f"{symbol}: Invalid position data (entry={entry_price}, qty={quantity})"
                report["issues_found"].append(issue)
                self.log(issue, level="ERROR")
                continue

            # Detect DCA level from margin
            dca_level = self.detect_dca_level_from_margin(symbol, margin_used, budget_per_symbol)

            # Get expected SL/TP
            expected_sl, expected_tp, tp_roi, sl_roi = self.get_expected_sl_tp(entry_price, side, dca_level)

            # Adjust SL for liquidation protection
            if liq_price > 0:
                buffer_pct = DCA_CONFIG.get("liquidation_buffer_pct", 0.01)
                if side == "LONG":
                    min_sl = liq_price * (1 + buffer_pct)
                    if expected_sl < min_sl:
                        expected_sl = min_sl
                else:
                    max_sl = liq_price * (1 - buffer_pct)
                    if expected_sl > max_sl:
                        expected_sl = max_sl

            # Get current SL/TP orders
            orders = self.client.get_open_orders(symbol)

            actual_sl = None
            actual_tp = None
            sl_order_id = None
            tp_order_id = None
            sl_qty = 0
            tp_qty = 0

            for order in orders:
                order_type = order.get("type", "")
                order_side = order.get("side", "")
                stop_price = float(order.get("stopPrice", 0))
                order_qty = float(order.get("origQty", 0))

                if side == "LONG":
                    if order_type == "STOP_MARKET" and order_side == "SELL":
                        actual_sl = stop_price
                        sl_order_id = order.get("orderId")
                        sl_qty = order_qty
                    elif order_type == "TAKE_PROFIT_MARKET" and order_side == "SELL":
                        actual_tp = stop_price
                        tp_order_id = order.get("orderId")
                        tp_qty = order_qty
                else:
                    if order_type == "STOP_MARKET" and order_side == "BUY":
                        actual_sl = stop_price
                        sl_order_id = order.get("orderId")
                        sl_qty = order_qty
                    elif order_type == "TAKE_PROFIT_MARKET" and order_side == "BUY":
                        actual_tp = stop_price
                        tp_order_id = order.get("orderId")
                        tp_qty = order_qty

            # Check for issues
            issues = []
            needs_sl_fix = False
            needs_tp_fix = False

            # Check SL
            if actual_sl is None:
                issues.append("SL MISSING")
                needs_sl_fix = True
            elif abs(sl_qty - quantity) > quantity * 0.01:
                issues.append(f"SL QTY WRONG ({sl_qty:.4f} vs {quantity:.4f})")
                needs_sl_fix = True

            # Check TP
            if actual_tp is None:
                issues.append("TP MISSING")
                needs_tp_fix = True
            elif abs(tp_qty - quantity) > quantity * 0.01:
                issues.append(f"TP QTY WRONG ({tp_qty:.4f} vs {quantity:.4f})")
                needs_tp_fix = True
            else:
                # Check if TP price is correct for DCA level
                # Allow 2% tolerance for price differences
                tp_tolerance = 0.02
                if side == "LONG":
                    if actual_tp > expected_tp * (1 + tp_tolerance):
                        issues.append(f"TP TOO HIGH (${actual_tp:.4f} vs ${expected_tp:.4f})")
                        needs_tp_fix = True
                else:
                    if actual_tp < expected_tp * (1 - tp_tolerance):
                        issues.append(f"TP TOO LOW (${actual_tp:.4f} vs ${expected_tp:.4f})")
                        needs_tp_fix = True

            # Print status
            print(f"\n{symbol} {side} @ ${entry_price:,.4f} | DCA Level: {dca_level}/4")
            print(f"  Quantity: {quantity:.6f} | Margin: ${margin_used:.2f}")
            print(f"  Expected TP: ${expected_tp:,.4f} ({tp_roi*100:.1f}% ROI)")
            print(f"  Expected SL: ${expected_sl:,.4f} ({sl_roi*100:.1f}% ROI)")
            print(f"  Actual TP: ${actual_tp:,.4f}" if actual_tp else "  Actual TP: MISSING!")
            print(f"  Actual SL: ${actual_sl:,.4f}" if actual_sl else "  Actual SL: MISSING!")

            if issues:
                issue_str = ", ".join(issues)
                print(f"  ISSUES: {issue_str}")
                report["issues_found"].append(f"{symbol}: {issue_str}")

                if auto_fix:
                    # Fix the orders
                    close_side = "SELL" if side == "LONG" else "BUY"

                    if needs_sl_fix:
                        # Cancel old SL if exists
                        if sl_order_id:
                            try:
                                self.client.cancel_order(symbol, sl_order_id)
                                self.log(f"  Cancelled old SL order {sl_order_id}")
                            except Exception as e:
                                self.log(f"  Could not cancel old SL: {e}", level="WARN")

                        # Place new SL
                        sl_order = self.client.place_stop_loss(symbol, close_side, quantity, expected_sl)
                        if "orderId" in sl_order:
                            self.log(f"  Placed new SL @ ${expected_sl:,.4f} (ID: {sl_order['orderId']})")
                            report["fixed"].append(f"{symbol}: SL fixed")
                        else:
                            self.log(f"  Failed to place SL: {sl_order}", level="ERROR")

                    if needs_tp_fix:
                        # Cancel old TP if exists
                        if tp_order_id:
                            try:
                                self.client.cancel_order(symbol, tp_order_id)
                                self.log(f"  Cancelled old TP order {tp_order_id}")
                            except Exception as e:
                                self.log(f"  Could not cancel old TP: {e}", level="WARN")

                        # Place new TP
                        tp_order = self.client.place_take_profit(symbol, close_side, quantity, expected_tp)
                        if "orderId" in tp_order:
                            self.log(f"  Placed new TP @ ${expected_tp:,.4f} (ID: {tp_order['orderId']})")
                            report["fixed"].append(f"{symbol}: TP fixed")
                        else:
                            self.log(f"  Failed to place TP: {tp_order}", level="ERROR")
            else:
                print(f"  STATUS: OK")
                report["positions_ok"] += 1

        print("\n" + "=" * 70)

        return report

    def run_loop(self, interval: int = 60, auto_fix: bool = False):
        """Run the monitor in a continuous loop"""
        self.log(f"Starting SL/TP Monitor Loop (interval: {interval}s, auto_fix: {auto_fix})")

        try:
            while True:
                print("\n" + "=" * 70)
                print(f"SL/TP MONITOR CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 70)

                report = self.check_all_positions(auto_fix=auto_fix)

                # Print summary
                print("\n--- SUMMARY ---")
                print(f"  Positions checked: {report['positions_checked']}")
                print(f"  Positions OK: {report['positions_ok']}")
                print(f"  Issues found: {len(report['issues_found'])}")
                if auto_fix:
                    print(f"  Fixed: {len(report['fixed'])}")

                print(f"\nNext check in {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)

        except KeyboardInterrupt:
            self.log("Monitor stopped by user")


def main():
    parser = argparse.ArgumentParser(
        description="SL/TP Order Monitor & Auto-Fix Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python monitor_sl_tp.py           # Check and report (dry run)
    python monitor_sl_tp.py --fix     # Check and fix any issues
    python monitor_sl_tp.py --loop    # Run continuously every 60 seconds
    python monitor_sl_tp.py --loop --fix --interval 30  # Fix every 30s
        """
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix any issues found"
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in a loop"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds for loop mode (default: 60)"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Use mainnet (REAL MONEY!) instead of demo"
    )

    args = parser.parse_args()

    testnet = not args.live

    print("\n" + "=" * 70)
    mode = "DEMO" if testnet else "MAINNET (REAL MONEY!)"
    print(f"SL/TP ORDER MONITOR - {mode}")
    print("=" * 70)

    if args.live:
        print("\n!!! WARNING: MAINNET MODE !!!")
        print("This will modify REAL orders with REAL funds!")
        confirm = input("\nContinue? (y/n): ")
        if confirm.lower() != "y":
            print("Cancelled")
            return

    monitor = SLTPMonitor(testnet=testnet)

    if args.loop:
        monitor.run_loop(interval=args.interval, auto_fix=args.fix)
    else:
        report = monitor.check_all_positions(auto_fix=args.fix)

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Positions checked: {report['positions_checked']}")
        print(f"Positions OK: {report['positions_ok']}")
        print(f"Issues found: {len(report['issues_found'])}")

        if report['issues_found']:
            print("\nIssues:")
            for issue in report['issues_found']:
                print(f"  - {issue}")

        if args.fix and report['fixed']:
            print("\nFixed:")
            for fix in report['fixed']:
                print(f"  - {fix}")

        print("=" * 70)


if __name__ == "__main__":
    main()
