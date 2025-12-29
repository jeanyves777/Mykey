"""
Test Tradovate API Connection

Run this FIRST after filling in your credentials
"""

import sys
sys.path.insert(0, '.')

from trading_system.Tradovate.engine.tradovate_client import TradovateClient
from trading_system.Tradovate.config import tradovate_config

def test_connection():
    """Test Tradovate connection"""
    print("=" * 80)
    print("TRADOVATE API CONNECTION TEST")
    print("=" * 80)

    # Check if credentials are filled
    if not tradovate_config.TRADOVATE_DEMO_USERNAME:
        print("\n❌ ERROR: Credentials not filled!")
        print("\nPlease edit trading_system/Tradovate/config/tradovate_config.py")
        print("and fill in your Tradovate credentials:")
        print("  - TRADOVATE_DEMO_USERNAME")
        print("  - TRADOVATE_DEMO_PASSWORD")
        print("  - TRADOVATE_DEMO_API_KEY")
        print("\nGet your API key from: Tradovate Settings → API")
        return

    try:
        # Initialize client
        print("\n[1/4] Initializing Tradovate client...")
        client = TradovateClient(
            username=tradovate_config.TRADOVATE_DEMO_USERNAME,
            password=tradovate_config.TRADOVATE_DEMO_PASSWORD,
            api_key=tradovate_config.TRADOVATE_DEMO_API_KEY,
            is_demo=tradovate_config.USE_DEMO
        )

        # Get account balance
        print("\n[2/4] Getting account balance...")
        balance = client.get_account_balance()
        print(f"\n  Account Balance:")
        print(f"    Cash Balance:      ${balance.get('balance', 0):,.2f}")
        print(f"    Equity:            ${balance.get('equity', 0):,.2f}")
        print(f"    Margin Available:  ${balance.get('margin_available', 0):,.2f}")
        print(f"    Unrealized P&L:    ${balance.get('unrealized_pnl', 0):+,.2f}")

        # Check positions
        print("\n[3/4] Checking positions...")
        positions = client.get_positions()
        if positions:
            print(f"  Open positions: {len(positions)}")
            for pos in positions:
                print(f"    - {pos['symbol']}: {pos['side'].upper()} {pos['quantity']} @ {pos['entry_price']}")
        else:
            print("  No open positions ✓")

        # Test market data
        print("\n[4/4] Testing market data...")
        print("  Fetching M6EU2 (Micro Euro) data...")

        df = client.get_historical_bars('M6EU2', timeframe='M15', bars=10)
        if len(df) > 0:
            print(f"  ✓ Successfully retrieved {len(df)} bars")
            print(f"  Latest bar: {df.index[-1]}")
            print(f"  Close: {df['close'].iloc[-1]:.5f}")
        else:
            print("  ⚠ No data returned (symbol might not be available)")

        # Success
        print("\n" + "=" * 80)
        print("✅ CONNECTION TEST PASSED!")
        print("=" * 80)
        print("\nYour Tradovate API is working correctly!")
        print("\nNext steps:")
        print("  1. Run backtest: python run_tradovate_backtest.py")
        print("  2. Run paper trading: python run_tradovate_paper.py")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ CONNECTION TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check your credentials in tradovate_config.py")
        print("  2. Make sure you're using DEMO credentials")
        print("  3. Verify API key is correct (from Tradovate Settings → API)")
        print("  4. Check your internet connection")
        print("=" * 80)


if __name__ == "__main__":
    test_connection()
