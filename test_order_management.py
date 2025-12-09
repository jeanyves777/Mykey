#!/usr/bin/env python3
"""
Test Order Management - BTC/USD

This script tests the order management flow:
1. Place a BUY order for BTC
2. Place a Stop-Limit SL order (only supported type for Alpaca crypto)
3. Monitor the position for a few minutes
4. Close with a SELL order (simulating TP)

This validates our order management before live trading.
"""

import sys
import time
import os
from datetime import datetime
from pathlib import Path

# Fix for Windows console encoding
if sys.platform == 'win32':
    os.system('')  # Enable ANSI escape codes on Windows

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_system.engine.alpaca_client import AlpacaClient, ALPACA_AVAILABLE
from trading_system.config.crypto_paper_trading_config import CryptoPaperTradingConfig

import pytz
UTC = pytz.UTC


def main():
    print("=" * 70)
    print("ORDER MANAGEMENT TEST - BTC/USD")
    print("=" * 70)
    print()

    # Load config
    config = CryptoPaperTradingConfig.load()
    if not config.is_configured():
        print("ERROR: Config not set up. Run: python -m trading_system.run_crypto_paper_trading --setup")
        return

    # Initialize client
    client = AlpacaClient(
        api_key=config.api_key,
        api_secret=config.api_secret,
        paper=True
    )

    # Test connection
    print("[1] Testing connection...")
    try:
        account = client.get_account()
        print(f"    Connected! Cash: ${account['cash']:,.2f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # Get current BTC price
    print("\n[2] Getting BTC/USD quote...")
    symbol = "BTC/USD"
    quote = client.get_latest_crypto_quote(symbol)
    if not quote:
        print("    ERROR: Could not get quote")
        return

    print(f"    Bid: ${quote.bid:,.2f}")
    print(f"    Ask: ${quote.ask:,.2f}")
    print(f"    Mid: ${quote.mid:,.2f}")

    # Calculate order parameters
    position_value = 100.0  # Small test position - $100
    qty = position_value / quote.ask
    entry_price = quote.mid
    sl_price = entry_price * 0.995  # 0.5% below
    sl_limit = sl_price * 0.995     # Limit 0.5% below stop for execution
    tp_price = entry_price * 1.01   # 1% above

    print(f"\n[3] Order Parameters:")
    print(f"    Position Value: ${position_value:.2f}")
    print(f"    Quantity: {qty:.6f} BTC")
    print(f"    Entry (expected): ${entry_price:,.2f}")
    print(f"    Stop Loss: ${sl_price:,.2f} (-0.5%) [Stop-Limit Order]")
    print(f"    SL Limit: ${sl_limit:,.2f}")
    print(f"    Take Profit: ${tp_price:,.2f} (+1%) [Manual TP]")

    # Confirm
    print("\n" + "=" * 70)
    print("READY TO PLACE ORDERS ON ALPACA PAPER TRADING")
    print("Alpaca Crypto supports: Market, Limit, Stop-Limit orders only")
    print("=" * 70)
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled.")
        return

    # Step 1: Place BUY order
    print("\n" + "=" * 70)
    print("[4] PLACING BUY ORDER...")
    print("=" * 70)
    try:
        buy_order = client.submit_crypto_market_order(
            symbol=symbol,
            qty=qty,
            side='buy'
        )
        buy_order_id = buy_order.get('id', 'N/A')
        buy_status = buy_order.get('status', 'unknown')
        buy_filled_price = buy_order.get('filled_avg_price')

        print(f"    Order ID: {buy_order_id}")
        print(f"    Status: {buy_status}")
        if buy_filled_price:
            print(f"    Fill Price: ${buy_filled_price:,.2f}")
            entry_price = buy_filled_price
            sl_price = entry_price * 0.995
            sl_limit = sl_price * 0.995
            tp_price = entry_price * 1.01
        print(f"    SUCCESS!")
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    # Wait for order to fill and get actual quantity from position
    print("    Waiting for order to fill...")
    time.sleep(3)

    # Get actual position quantity (important: may differ slightly from requested)
    actual_qty = qty
    try:
        positions = client.trading_client.get_all_positions()
        for p in positions:
            if p.symbol == "BTCUSD" or p.symbol == "BTC/USD":
                actual_qty = float(p.qty)
                entry_price = float(p.avg_entry_price)
                sl_price = entry_price * 0.995
                sl_limit = sl_price * 0.995
                tp_price = entry_price * 1.01
                print(f"    Actual Filled Qty: {actual_qty:.8f} BTC")
                print(f"    Actual Entry Price: ${entry_price:,.2f}")
                break
    except Exception as e:
        print(f"    Warning: Could not verify position: {e}")
        print(f"    Using requested qty: {qty:.8f}")

    # Step 2: Place SL order (Stop-Limit)
    print("\n" + "=" * 70)
    print("[5] PLACING STOP-LIMIT ORDER (SL)...")
    print("=" * 70)
    sl_order_id = None
    try:
        sl_order = client.submit_crypto_stop_limit_order(
            symbol=symbol,
            qty=actual_qty,  # Use actual filled qty
            side='sell',
            stop_price=sl_price,
            limit_price=sl_limit
        )
        sl_order_id = sl_order.get('id', 'N/A')
        sl_status = sl_order.get('status', 'unknown')

        print(f"    Order ID: {sl_order_id}")
        print(f"    Status: {sl_status}")
        print(f"    Stop Price: ${sl_price:,.2f}")
        print(f"    Limit Price: ${sl_limit:,.2f}")
        print(f"    SUCCESS!")
    except Exception as e:
        print(f"    ERROR: {e}")
        print(f"    WARNING: SL not placed, will manage manually")

    # Step 3: Monitor position
    print("\n" + "=" * 70)
    print("[6] MONITORING POSITION (30 seconds)...")
    print("=" * 70)
    print()

    monitor_seconds = 30
    for i in range(monitor_seconds):
        # Get current price
        quote = client.get_latest_crypto_quote(symbol)
        if quote:
            current = quote.mid
            pnl_pct = (current - entry_price) / entry_price * 100
            dist_to_sl = (current - sl_price) / current * 100
            dist_to_tp = (tp_price - current) / current * 100

            # Simple ASCII progress bar
            price_range = tp_price - sl_price
            if price_range > 0:
                progress = (current - sl_price) / price_range
                progress = max(0, min(1, progress))
                bar_width = 20
                filled = int(progress * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
            else:
                bar = "-" * 20

            print(f"  [{i+1:2d}/{monitor_seconds}s] BTC: ${current:,.2f} | P&L: {pnl_pct:+.3f}% | SL [{bar}] TP | SL:{dist_to_sl:.2f}% | TP:{dist_to_tp:.2f}%")

        time.sleep(1)

    print()

    # Step 4: Close position
    print("=" * 70)
    print("[7] CLOSING POSITION...")
    print("=" * 70)

    # First cancel the SL order
    if sl_order_id:
        print(f"    Cancelling SL order {sl_order_id[:12]}...")
        try:
            cancelled = client.cancel_order(sl_order_id)
            if cancelled:
                print(f"    SL cancelled successfully")
            else:
                print(f"    SL already filled or cancelled")
        except Exception as e:
            print(f"    Failed to cancel SL: {e}")

    # Then place SELL order
    print(f"    Placing SELL order...")
    sell_order_id = None
    try:
        sell_order = client.submit_crypto_market_order(
            symbol=symbol,
            qty=actual_qty,  # Use actual filled qty
            side='sell'
        )
        sell_order_id = sell_order.get('id', 'N/A')
        sell_status = sell_order.get('status', 'unknown')
        sell_filled_price = sell_order.get('filled_avg_price')

        print(f"    Order ID: {sell_order_id}")
        print(f"    Status: {sell_status}")
        if sell_filled_price:
            print(f"    Fill Price: ${sell_filled_price:,.2f}")
            exit_price = sell_filled_price
        else:
            exit_price = quote.mid if quote else entry_price

        # Calculate final P&L
        pnl = actual_qty * (exit_price - entry_price)
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        print(f"\n    FINAL RESULT:")
        print(f"      Entry: ${entry_price:,.2f}")
        print(f"      Exit:  ${exit_price:,.2f}")
        print(f"      P&L:   ${pnl:+.2f} ({pnl_pct:+.3f}%)")

    except Exception as e:
        print(f"    ERROR: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print("\nOrders placed:")
    print(f"  1. BUY:  {buy_order_id}")
    if sl_order_id:
        print(f"  2. SL:   {sl_order_id} (Stop-Limit)")
    if sell_order_id:
        print(f"  3. SELL: {sell_order_id}")
    print("\nCheck your Alpaca Paper Trading dashboard to verify:")
    print("  https://app.alpaca.markets/account/orders")


if __name__ == "__main__":
    main()
