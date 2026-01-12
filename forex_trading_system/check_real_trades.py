"""
Check Real Trade Sizes from OANDA
==================================
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.oanda_client import OANDAClient
from config.trading_config import OANDA_CONFIG

def main():
    client = OANDAClient(practice=OANDA_CONFIG["practice"])

    print("\n" + "=" * 80)
    print("CHECKING ACTUAL TRADES FROM OANDA API")
    print("=" * 80)

    # Get account info
    account = client.get_account_summary()
    if not account:
        print("ERROR: Could not get account info")
        return

    balance = float(account.get('balance', 0))
    unrealized_pl = float(account.get('unrealizedPL', 0))

    print(f"\nAccount Balance: ${balance:,.2f}")
    print(f"Unrealized P&L: ${unrealized_pl:+,.2f}")
    print(f"Total Equity: ${balance + unrealized_pl:,.2f}")

    # Get all open trades (not positions, but individual trades)
    print("\n" + "-" * 80)
    print("OPEN TRADES (Individual Orders):")
    print("-" * 80)

    try:
        trades = client.get_open_trades()

        if not trades:
            print("No open trades found")
            return

        print(f"\nTotal open trades: {len(trades)}\n")

        total_risk = 0

        for i, trade in enumerate(trades, 1):
            instrument = trade['instrument']
            units = trade['units']
            entry_price = trade['price']
            current_pl = trade['unrealized_pl']
            sl_price = trade.get('stop_loss', 0)
            tp_price = trade.get('take_profit', 0)

            # Calculate notional value
            notional_value = abs(units) * entry_price

            print(f"\n{i}. {instrument}")
            print(f"   Units: {abs(units):,.0f} ({'LONG' if units > 0 else 'SHORT'})")
            print(f"   Entry Price: {entry_price:.5f}")
            print(f"   Notional Value: ${notional_value:,.2f}")
            print(f"   Current P&L: ${current_pl:+,.2f}")

            # Check for stop loss
            if sl_price > 0:
                print(f"   Stop Loss: {sl_price:.5f}")

                # Calculate risk
                pip_location = -2 if 'JPY' in instrument else -4
                pip_size = 10 ** pip_location

                # Calculate SL distance in pips
                sl_distance_pips = abs(entry_price - sl_price) / pip_size

                # Calculate risk amount in USD
                # For pairs quoted in USD (XXX_USD): risk = units * pip_size * sl_distance_pips
                # For pairs with USD as base (USD_XXX): risk = (units * pip_size * sl_distance_pips) / entry_price

                if instrument.endswith('_USD'):
                    # Quote currency is USD (EUR_USD, AUD_USD, etc.)
                    risk_amount = abs(units) * pip_size * sl_distance_pips
                else:
                    # Base currency is USD (USD_JPY, USD_CAD, etc.)
                    risk_amount = (abs(units) * pip_size * sl_distance_pips) / entry_price

                print(f"   SL Distance: {sl_distance_pips:.1f} pips")
                print(f"   RISK AMOUNT: ${risk_amount:,.2f}")
                print(f"   Risk %: {(risk_amount / balance) * 100:.2f}%")

                total_risk += risk_amount
            else:
                print(f"   ⚠️  NO STOP LOSS SET!")

            # Check for take profit
            if tp_price > 0:
                print(f"   Take Profit: {tp_price:.5f}")

        print("\n" + "=" * 80)
        print(f"TOTAL RISK ACROSS ALL TRADES: ${total_risk:,.2f}")
        print(f"TOTAL RISK AS % OF BALANCE: {(total_risk / balance) * 100:.2f}%")
        print("=" * 80)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
