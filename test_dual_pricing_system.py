"""
Test Dual Pricing System - OANDA vs NinjaTrader

This script validates:
1. NinjaTrader bridge price query functionality
2. OANDA price fetching
3. Price comparison display
4. Difference calculation in pips

BEFORE RUNNING:
1. Start NinjaTrader 8
2. Connect to Sim101 or FundedNext account
3. Run NinjaTraderBridge.exe
4. Then run this test script

This will show you the actual price differences between OANDA and NinjaTrader
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

import socket
import json
from datetime import datetime
from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.config.forex_trading_config import FOREX_INSTRUMENTS

# Symbol mapping: OANDA → NinjaTrader
SYMBOL_MAP = {
    'EUR_USD': 'M6E',
    'GBP_USD': 'M6B',
    'USD_JPY': 'MJY',
    'USD_CAD': 'MCD',
    'USD_CHF': 'MSF'
}

# Pair settings
PAIR_SETTINGS = {
    'M6E': {'tick_size': 0.00005, 'tick_value': 6.25, 'name': 'Micro EUR/USD'},
    'M6B': {'tick_size': 0.0001, 'tick_value': 6.25, 'name': 'Micro GBP/USD'},
    'MJY': {'tick_size': 0.000001, 'tick_value': 1.25, 'name': 'Micro USD/JPY'},
    'MCD': {'tick_size': 0.00005, 'tick_value': 5.00, 'name': 'Micro USD/CAD'},
    'MSF': {'tick_size': 0.00005, 'tick_value': 6.25, 'name': 'Micro USD/CHF'},
}


def query_ninjatrader_price(nt_symbol: str) -> dict:
    """Query NinjaTrader price via bridge"""
    try:
        query = {
            'Action': 'PRICE_QUERY',
            'Symbol': nt_symbol,
            'Timestamp': datetime.now().isoformat()
        }

        message = json.dumps(query)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 8888))
        sock.sendall(message.encode('utf-8'))

        response = sock.recv(4096).decode('utf-8')
        sock.close()

        price_data = json.loads(response)

        status = price_data.get('Status', '')

        # Handle different status codes
        if status == 'OK':
            return {
                'symbol': nt_symbol,
                'bid': price_data.get('Bid', 0),
                'ask': price_data.get('Ask', 0),
                'last': price_data.get('Last', 0),
                'mid': (price_data.get('Bid', 0) + price_data.get('Ask', 0)) / 2,
                'source': 'NinjaTrader',
                'status': 'LIVE'
            }
        elif status == 'MARKET_CLOSED':
            return {
                'symbol': nt_symbol,
                'bid': price_data.get('Bid', 0),
                'ask': price_data.get('Ask', 0),
                'last': price_data.get('Last', 0),
                'mid': (price_data.get('Bid', 0) + price_data.get('Ask', 0)) / 2,
                'source': 'NinjaTrader (cached)',
                'status': 'CACHED',
                'timestamp': price_data.get('Timestamp', '')
            }
        elif status == 'MARKET_CLOSED_NO_CACHE':
            return {
                'symbol': nt_symbol,
                'status': 'NO_DATA',
                'error': 'Market closed, no cached prices available'
            }
        else:
            return {
                'symbol': nt_symbol,
                'status': 'ERROR',
                'error': status
            }

    except Exception as e:
        return {
            'symbol': nt_symbol,
            'status': 'ERROR',
            'error': str(e)
        }


def test_dual_pricing():
    """Test dual pricing system"""
    print("=" * 80)
    print("DUAL PRICING SYSTEM TEST - OANDA vs NinjaTrader")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize OANDA client
    print("[1/3] Connecting to OANDA...")
    try:
        oanda_client = OandaClient(account_type="practice")
        print("✓ Connected to OANDA")
    except Exception as e:
        print(f"❌ Failed to connect to OANDA: {e}")
        return

    # Test NinjaTrader bridge connection
    print("\n[2/3] Testing NinjaTrader bridge connection...")
    try:
        test_price = query_ninjatrader_price('M6E')
        if test_price['status'] == 'OK':
            print("✓ Connected to NinjaTrader bridge")
        else:
            print(f"❌ NinjaTrader bridge error: {test_price.get('error')}")
            print("\nMake sure:")
            print("  1. NinjaTrader 8 is running")
            print("  2. Connected to your account (Sim101 or FundedNext)")
            print("  3. NinjaTraderBridge.exe is running")
            return
    except Exception as e:
        print(f"❌ Cannot connect to NinjaTrader bridge: {e}")
        print("\nMake sure NinjaTraderBridge.exe is running on port 8888")
        return

    # Fetch all prices
    print("\n[3/3] Fetching prices from both sources...")
    print()
    print("=" * 80)
    print("PRICE COMPARISON - OANDA vs NinjaTrader")
    print("=" * 80)
    print()

    all_data = []

    for oanda_symbol in FOREX_INSTRUMENTS:
        nt_symbol = SYMBOL_MAP.get(oanda_symbol)
        if not nt_symbol:
            continue

        settings = PAIR_SETTINGS[nt_symbol]

        # Fetch OANDA price
        try:
            oanda_data = oanda_client.get_current_price(oanda_symbol)
            oanda_bid = oanda_data['bid']
            oanda_ask = oanda_data['ask']
            oanda_mid = (oanda_bid + oanda_ask) / 2
        except Exception as e:
            print(f"❌ {nt_symbol} ({settings['name']}) - OANDA error: {e}")
            continue

        # Fetch NinjaTrader price
        nt_data = query_ninjatrader_price(nt_symbol)

        if nt_data['status'] not in ['LIVE', 'CACHED']:
            print(f"❌ {nt_symbol} ({settings['name']}) - NinjaTrader error: {nt_data.get('error')}")
            continue

        nt_bid = nt_data['bid']
        nt_ask = nt_data['ask']
        nt_mid = nt_data['mid']
        nt_source = nt_data.get('source', 'NinjaTrader')
        nt_timestamp = nt_data.get('timestamp', '')

        # Calculate difference
        mid_diff = abs(nt_mid - oanda_mid)
        diff_pips = mid_diff / settings['tick_size']

        all_data.append({
            'symbol': nt_symbol,
            'name': settings['name'],
            'oanda_bid': oanda_bid,
            'oanda_ask': oanda_ask,
            'oanda_mid': oanda_mid,
            'nt_bid': nt_bid,
            'nt_ask': nt_ask,
            'nt_mid': nt_mid,
            'nt_source': nt_source,
            'nt_timestamp': nt_timestamp,
            'nt_status': nt_data['status'],
            'diff_pips': diff_pips
        })

    # Display results
    if all_data:
        print(f"{'Symbol':<6} {'Source':<22} {'Bid':<12} {'Ask':<12} {'Mid':<12} {'Diff (pips)':<12}")
        print("-" * 82)

        for data in all_data:
            print(f"{data['symbol']:<6} {'OANDA':<22} {data['oanda_bid']:<12.5f} {data['oanda_ask']:<12.5f} {data['oanda_mid']:<12.5f} {'':<12}")

            # Show status indicator for NinjaTrader
            nt_label = data['nt_source']
            if data['nt_status'] == 'CACHED':
                nt_label += f" [{data['nt_timestamp']}]"

            print(f"{'':<6} {nt_label:<22} {data['nt_bid']:<12.5f} {data['nt_ask']:<12.5f} {data['nt_mid']:<12.5f} {data['diff_pips']:<12.1f}")
            print("-" * 82)

        print()
        print("=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        avg_diff = sum(d['diff_pips'] for d in all_data) / len(all_data)
        max_diff = max(all_data, key=lambda x: x['diff_pips'])
        min_diff = min(all_data, key=lambda x: x['diff_pips'])

        print(f"  Average price difference: {avg_diff:.2f} pips")
        print(f"  Maximum difference: {max_diff['diff_pips']:.2f} pips ({max_diff['symbol']})")
        print(f"  Minimum difference: {min_diff['diff_pips']:.2f} pips ({min_diff['symbol']})")
        print()

        print("INTERPRETATION:")
        if avg_diff < 2:
            print("  ✓ Excellent - Price differences are minimal (< 2 pips)")
        elif avg_diff < 5:
            print("  ✓ Good - Price differences are acceptable (2-5 pips)")
        elif avg_diff < 10:
            print("  ⚠ Moderate - Price differences are noticeable (5-10 pips)")
        else:
            print("  ⚠ High - Significant price differences (> 10 pips)")

        print()
        print("RECOMMENDATION:")
        print("  Use NinjaTrader prices for trade execution (TP/SL calculation)")
        print("  Use OANDA for signal generation (strategy logic)")
        print("  Monitor price differences to detect data issues")
        print()

    else:
        print("❌ No price data available")
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_dual_pricing()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
