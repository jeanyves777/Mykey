"""
Test Crypto Order with Manual TP/SL on Alpaca Paper Trading
=============================================================

NOTE: Alpaca does NOT support bracket orders (OTO/OTOCO/OCO) for crypto.
TP/SL must be managed manually through price monitoring.

This script:
1. Submits a simple market order for crypto
2. Shows the TP/SL levels that would be monitored locally
3. Verifies the order executes correctly

Run: python -m trading_system.High_frequency_crypto_tradin.test_bracket_order
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from trading_system.High_frequency_crypto_tradin.engine.alpaca_live_engine import AlpacaLiveEngine, AlpacaConfig


def test_bracket_order():
    """Test crypto order with manual TP/SL monitoring."""
    print("=" * 70)
    print("ALPACA CRYPTO ORDER TEST - Manual TP/SL")
    print("=" * 70)
    print("NOTE: Alpaca does NOT support bracket orders for crypto.")
    print("      TP/SL will be monitored locally by the trading engine.")
    print("=" * 70)
    print(f"Time: {datetime.now()}")
    print()

    # Check credentials
    api_key = os.getenv('ALPACA_CRYPTO_KEY')
    api_secret = os.getenv('ALPACA_CRYPTO_SECRET')

    if not api_key or not api_secret:
        print("ERROR: Alpaca crypto credentials not found!")
        print("Please set ALPACA_CRYPTO_KEY and ALPACA_CRYPTO_SECRET in .env file")
        return False

    print(f"API Key: {api_key[:8]}...")

    # Create engine
    config = AlpacaConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://paper-api.alpaca.markets",
        crypto_symbols=["BTC/USD", "ETH/USD"]
    )

    engine = AlpacaLiveEngine(config)

    if not engine.api:
        print("ERROR: Failed to connect to Alpaca API")
        return False

    # Get account info
    print("\n" + "-" * 70)
    print("ACCOUNT INFO")
    print("-" * 70)
    account = engine.get_account()
    if account:
        print(f"  Equity: ${account.get('equity', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")
    else:
        print("ERROR: Could not get account info")
        return False

    # Get current BTC price - use latest quote for accuracy
    print("\n" + "-" * 70)
    print("FETCHING CURRENT BTC PRICE")
    print("-" * 70)

    # First try to get quote for accurate pricing
    current_price = None
    try:
        from alpaca.data.requests import CryptoLatestQuoteRequest
        quote_request = CryptoLatestQuoteRequest(symbol_or_symbols="BTC/USD")
        quote = engine.crypto_api.get_crypto_latest_quote(quote_request)
        if quote and "BTC/USD" in quote:
            current_price = float(quote["BTC/USD"].ask_price)
            print(f"  Quote Ask Price: ${current_price:,.2f}")
    except Exception as e:
        print(f"  Quote fetch failed: {e}")

    # Fallback to historical bars
    if current_price is None:
        df = engine.get_historical_bars("BTC/USD", limit=5)
        if df.empty:
            print("ERROR: Could not get BTC price data")
            return False
        current_price = df['close'].iloc[-1]
        print(f"  Bar Close Price: ${current_price:,.2f}")

    print(f"  Current BTC/USD: ${current_price:,.2f}")

    # Calculate TP/SL using V18 optimal settings (0.8% TP, 0.6% SL)
    # Use slightly wider margins to account for price movement during order submission
    tp_pct = 0.01   # 1.0% TP (wider to ensure valid)
    sl_pct = 0.01   # 1.0% SL (wider to ensure valid)

    take_profit = round(current_price * (1 + tp_pct), 2)
    stop_loss = round(current_price * (1 - sl_pct), 2)

    # Small test position - must be >= $10 minimum on Alpaca
    # 0.00015 BTC at ~$88,000 = ~$13.20
    test_qty = 0.00015

    print("\n" + "-" * 70)
    print("TEST ORDER DETAILS")
    print("-" * 70)
    print(f"  Symbol: BTC/USD")
    print(f"  Side: BUY")
    print(f"  Quantity: {test_qty} BTC (${test_qty * current_price:.2f})")
    print(f"  Entry Price (market): ~${current_price:,.2f}")
    print(f"  Take Profit: ${take_profit:,.2f} (+{tp_pct*100:.2f}%)")
    print(f"  Stop Loss: ${stop_loss:,.2f} (-{sl_pct*100:.2f}%)")

    # Confirmation
    print("\n" + "-" * 70)
    print("READY TO SUBMIT TEST ORDER")
    print("-" * 70)
    print("This will place a small BUY order for BTC/USD with bracket TP/SL.")

    # Check for --auto flag to skip confirmation
    auto_mode = "--auto" in sys.argv
    if not auto_mode:
        try:
            response = input("\nPress Enter to submit order (or Ctrl+C to cancel)...")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled by user")
            return False
    else:
        print("\n[AUTO MODE] Submitting order automatically...")

    # Submit market order first
    print("\n" + "-" * 70)
    print("STEP 1: SUBMITTING ENTRY ORDER...")
    print("-" * 70)

    order = engine.submit_order(
        symbol="BTC/USD",
        qty=test_qty,
        side="buy",
        order_type="market",
        time_in_force="gtc"
    )

    if not order:
        print("ERROR: Entry order failed!")
        return False

    print(f"Entry order submitted: {order.get('id')}")

    # Wait for entry order to fill
    print("\nWaiting 2 seconds for entry order to fill...")
    time.sleep(2)

    # Get actual filled quantity from position
    positions = engine.get_positions()
    if "BTCUSD" in positions:
        actual_qty = positions["BTCUSD"]["qty"]
        print(f"Position filled: {actual_qty} BTC")
    else:
        actual_qty = test_qty
        print(f"Using requested qty: {actual_qty}")

    # Now place TP and SL orders with actual filled quantity
    print("\n" + "-" * 70)
    print("STEP 2: PLACING TP/SL ORDERS...")
    print("-" * 70)

    tp_order, sl_order = engine.place_tp_sl_orders(
        symbol="BTC/USD",
        qty=actual_qty,  # Use actual filled quantity
        entry_side="buy",
        take_profit=take_profit,
        stop_loss=stop_loss
    )

    # Show results
    print("\n" + "=" * 70)
    print("ORDERS SUMMARY")
    print("=" * 70)
    print(f"ENTRY ORDER:")
    print(f"  ID: {order.get('id', 'N/A')}")
    print(f"  Symbol: {order.get('symbol', 'N/A')}")
    print(f"  Side: {order.get('side', 'N/A')}")
    print(f"  Status: {order.get('status', 'N/A')}")

    if tp_order:
        print(f"\nTAKE PROFIT ORDER:")
        print(f"  ID: {tp_order.get('id', 'N/A')}")
        print(f"  Price: ${tp_order.get('price', 0):,.2f}")
        print(f"  Status: {tp_order.get('status', 'N/A')}")
    else:
        print(f"\nTAKE PROFIT ORDER: FAILED")

    if sl_order:
        print(f"\nSTOP LOSS ORDER:")
        print(f"  ID: {sl_order.get('id', 'N/A')}")
        print(f"  Price: ${sl_order.get('price', 0):,.2f}")
        print(f"  Status: {sl_order.get('status', 'N/A')}")
    else:
        print(f"\nSTOP LOSS ORDER: FAILED")

    # Wait a moment for orders to be visible
    print("\nWaiting 2 seconds for orders to be visible...")
    time.sleep(2)

    # Check positions
    print("\n" + "-" * 70)
    print("CHECKING POSITIONS (after order)")
    print("-" * 70)
    positions = engine.get_positions()
    if positions:
        for sym, pos in positions.items():
            print(f"  {sym}:")
            print(f"    Qty: {pos['qty']}")
            print(f"    Entry: ${pos['avg_entry_price']:,.2f}")
            print(f"    Current: ${pos['current_price']:,.2f}")
            print(f"    P&L: ${pos['unrealized_pl']:,.2f}")
    else:
        print("  No positions yet (order may still be filling)")

    # Check open orders (should show TP and SL orders)
    print("\n" + "-" * 70)
    print("CHECKING OPEN ORDERS (TP/SL)")
    print("-" * 70)
    try:
        orders = engine.api.get_orders()
        if orders:
            for ord in orders:
                order_type = ord.type.value if hasattr(ord.type, 'value') else str(ord.type)
                print(f"  Order: {ord.symbol} {ord.side} {ord.qty}")
                print(f"    Type: {order_type}")
                print(f"    Status: {ord.status}")
                if hasattr(ord, 'limit_price') and ord.limit_price:
                    print(f"    Limit Price: ${float(ord.limit_price):,.2f}")
                if hasattr(ord, 'stop_price') and ord.stop_price:
                    print(f"    Stop Price: ${float(ord.stop_price):,.2f}")
                print()
        else:
            print("  No pending orders found")
    except Exception as e:
        print(f"  Error getting orders: {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    if tp_order and sl_order:
        print("SUCCESS: Both TP and SL orders placed!")
    else:
        print("WARNING: Some orders may have failed")

    print("\nTo close the test position and cancel orders, run:")
    print("  python -m trading_system.High_frequency_crypto_tradin.test_bracket_order --close")

    return True


def close_all_positions():
    """Close all positions and cancel pending orders (cleanup)."""
    print("=" * 70)
    print("CLOSING ALL POSITIONS AND CANCELLING ORDERS")
    print("=" * 70)

    api_key = os.getenv('ALPACA_CRYPTO_KEY')
    api_secret = os.getenv('ALPACA_CRYPTO_SECRET')

    config = AlpacaConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://paper-api.alpaca.markets"
    )

    engine = AlpacaLiveEngine(config)

    if engine.api:
        # Cancel all pending orders first
        try:
            orders = engine.api.get_orders()
            if orders:
                print(f"Found {len(orders)} pending orders - cancelling...")
                engine.api.cancel_orders()
                print("All orders cancelled")
            else:
                print("No pending orders")
        except Exception as e:
            print(f"Error cancelling orders: {e}")

        # Close all positions
        positions = engine.get_positions()
        if positions:
            print(f"Found {len(positions)} open positions - closing...")
            engine.close_all_positions()
            print("All positions closed")
        else:
            print("No positions to close")

    print("\nCleanup complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--close":
        close_all_positions()
    else:
        test_bracket_order()
