"""
Test what the live trading status display will look like
Shows the periodic status update with current market prices
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.config.forex_trading_config import FOREX_INSTRUMENTS
from datetime import datetime

# Symbol mapping
SYMBOL_MAP = {
    'EUR_USD': 'M6E',
    'GBP_USD': 'M6B',
    'USD_JPY': 'MJY',
    'USD_CAD': 'MCD',
    'USD_CHF': 'MSF',
}

def get_current_prices(oanda_client):
    """Fetch current market prices for all NinjaTrader futures symbols"""
    prices = {}
    for oanda_symbol in FOREX_INSTRUMENTS:
        nt_symbol = SYMBOL_MAP.get(oanda_symbol)
        if not nt_symbol:
            continue

        try:
            price_data = oanda_client.get_current_price(oanda_symbol)
            if price_data:
                prices[nt_symbol] = {
                    'oanda_symbol': oanda_symbol,
                    'bid': price_data.get('bid', 0),
                    'ask': price_data.get('ask', 0),
                    'mid': (price_data.get('bid', 0) + price_data.get('ask', 0)) / 2
                }
        except Exception as e:
            pass

    return prices


def show_status_display():
    """Show what the periodic status update looks like"""
    print("=" * 80)
    print("LIVE TRADING STATUS DISPLAY - EXAMPLE")
    print("=" * 80)
    print()
    print("Connecting to OANDA to fetch live prices...")

    oanda_client = OandaClient(account_type="practice")

    # Simulate status display
    current_time = datetime.now()
    loop_count = 26

    # Mock account values
    current_balance = 25150.00
    current_threshold = 24000.00
    starting_balance_today = 25000.00
    total_profit = 150.00
    trades_today = 3
    max_trades_per_day = 50
    open_positions_count = 1
    max_concurrent = 5
    daily_loss_limit = -500
    profit_target = 1250

    daily_pnl = current_balance - starting_balance_today
    buffer = current_balance - current_threshold

    print()
    print("=" * 80)
    print(f"\n[{current_time.strftime('%H:%M:%S')}] Loop {loop_count} - Checking market...")
    print(f"  Market Status: Market open")
    print(f"  Balance: ${current_balance:,.2f}")
    print(f"  Threshold: ${current_threshold:,.2f} (EOD trailing stop)")
    print(f"  Buffer: ${buffer:,.2f} above threshold")
    print(f"  Daily P&L: ${daily_pnl:+,.2f} (Daily loss limit: ${daily_loss_limit})")
    print(f"  Total Profit: ${total_profit:+,.2f} (Target: ${profit_target})")
    print(f"  Trades today: {trades_today}/{max_trades_per_day}")
    print(f"  Open positions: {open_positions_count}/{max_concurrent}")

    # Display current market prices
    print(f"\n  Current Market Prices (NinjaTrader Futures):")
    current_prices = get_current_prices(oanda_client)
    if current_prices:
        print(f"  {'Symbol':<6} {'Bid':<12} {'Ask':<12} {'Mid':<12}")
        print(f"  {'-'*42}")
        for nt_symbol in ['M6E', 'M6B', 'MJY', 'MSF', 'MCD']:
            if nt_symbol in current_prices:
                p = current_prices[nt_symbol]
                print(f"  {nt_symbol:<6} {p['bid']:<12.5f} {p['ask']:<12.5f} {p['mid']:<12.5f}")
    else:
        print(f"  (No price data available)")

    print()
    print("=" * 80)
    print()
    print("This is what you'll see every 5 loops during live trading!")
    print()


if __name__ == "__main__":
    try:
        show_status_display()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
