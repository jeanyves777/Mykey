"""
Test script to fetch current market prices for NinjaTrader Micro E-mini FX Futures

This validates that we can access price data for all trading pairs:
- M6E (Micro EUR/USD)
- M6B (Micro GBP/USD)
- MJY (Micro USD/JPY)
- MSF (Micro USD/CHF)
- MCD (Micro USD/CAD)

Uses OANDA data as proxy since we're fetching via OANDA API.
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.config.forex_trading_config import FOREX_INSTRUMENTS
from datetime import datetime

# Symbol mapping: OANDA → NinjaTrader
SYMBOL_MAP = {
    'EUR_USD': 'M6E',  # Micro EUR/USD
    'GBP_USD': 'M6B',  # Micro GBP/USD
    'USD_JPY': 'MJY',  # Micro USD/JPY
    'USD_CAD': 'MCD',  # Micro USD/CAD
    'USD_CHF': 'MSF',  # Micro USD/CHF
}

# Pair settings with tick information
PAIR_SETTINGS = {
    'M6E': {'tick_size': 0.00005, 'tick_value': 6.25, 'name': 'Micro EUR/USD'},
    'M6B': {'tick_size': 0.0001, 'tick_value': 6.25, 'name': 'Micro GBP/USD'},
    'MJY': {'tick_size': 0.000001, 'tick_value': 1.25, 'name': 'Micro USD/JPY'},
    'MCD': {'tick_size': 0.00005, 'tick_value': 5.00, 'name': 'Micro USD/CAD'},
    'MSF': {'tick_size': 0.00005, 'tick_value': 6.25, 'name': 'Micro USD/CHF'},
}


def fetch_and_display_prices():
    """Fetch and display current market prices for all NinjaTrader futures"""
    print("=" * 80)
    print("NINJATRADER MICRO E-MINI FX FUTURES - CURRENT MARKET PRICES")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize OANDA client
    print("[INIT] Connecting to OANDA...")
    try:
        oanda_client = OandaClient(account_type="practice")
        print("[INIT] OK - Connected to OANDA")
    except Exception as e:
        print(f"[ERROR] Failed to connect to OANDA: {e}")
        return

    print()
    print("=" * 80)
    print("FETCHING PRICES FOR ALL TRADING PAIRS")
    print("=" * 80)
    print()

    # Fetch prices for each symbol
    all_prices = []
    for oanda_symbol in FOREX_INSTRUMENTS:
        nt_symbol = SYMBOL_MAP.get(oanda_symbol)
        if not nt_symbol:
            print(f"[SKIP] {oanda_symbol} - No NinjaTrader mapping")
            continue

        try:
            # Get current price from OANDA
            price_data = oanda_client.get_current_price(oanda_symbol)

            if price_data:
                settings = PAIR_SETTINGS[nt_symbol]

                bid = price_data.get('bid', 0)
                ask = price_data.get('ask', 0)
                mid = (bid + ask) / 2
                spread = ask - bid
                spread_pips = spread / settings['tick_size']

                all_prices.append({
                    'oanda_symbol': oanda_symbol,
                    'nt_symbol': nt_symbol,
                    'name': settings['name'],
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'spread': spread,
                    'spread_pips': spread_pips,
                    'tick_size': settings['tick_size'],
                    'tick_value': settings['tick_value']
                })

                print(f"[OK] {nt_symbol} ({settings['name']})")
                print(f"  OANDA Pair: {oanda_symbol}")
                print(f"  Bid: {bid:.5f}")
                print(f"  Ask: {ask:.5f}")
                print(f"  Mid: {mid:.5f}")
                print(f"  Spread: {spread:.5f} ({spread_pips:.1f} pips)")
                print(f"  Tick Size: {settings['tick_size']}")
                print(f"  Tick Value: ${settings['tick_value']:.2f}")
                print()
            else:
                print(f"[FAIL] {nt_symbol} - No price data returned")
                print()

        except Exception as e:
            print(f"[ERROR] {nt_symbol} - Error fetching price: {e}")
            print()

    # Summary table
    if all_prices:
        print("=" * 80)
        print("SUMMARY - ALL NINJATRADER FUTURES PRICES")
        print("=" * 80)
        print()
        print(f"{'Symbol':<6} {'Name':<20} {'Bid':<12} {'Ask':<12} {'Mid':<12} {'Spread (pips)':<15}")
        print("-" * 80)

        for price in all_prices:
            print(f"{price['nt_symbol']:<6} {price['name']:<20} "
                  f"{price['bid']:<12.5f} {price['ask']:<12.5f} "
                  f"{price['mid']:<12.5f} {price['spread_pips']:<15.1f}")

        print()
        print("=" * 80)
        print(f"SUCCESS - Fetched prices for {len(all_prices)}/{len(FOREX_INSTRUMENTS)} symbols")
        print("=" * 80)
        print()

        # Additional info
        print("TRADING INFORMATION:")
        print(f"  - Total symbols configured: {len(FOREX_INSTRUMENTS)}")
        print(f"  - NinjaTrader futures available: {len(all_prices)}")
        print(f"  - Data source: OANDA (practice account)")
        print(f"  - Market status: {'OPEN' if all_prices else 'CLOSED or NO DATA'}")
        print()

    else:
        print("=" * 80)
        print("WARNING - NO PRICES AVAILABLE")
        print("=" * 80)
        print()
        print("Possible reasons:")
        print("  - Market is closed (Forex closed Friday 5pm ET - Sunday 5pm ET)")
        print("  - OANDA API connection issue")
        print("  - Invalid API credentials")
        print()


if __name__ == "__main__":
    try:
        fetch_and_display_prices()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
