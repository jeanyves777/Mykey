"""Quick position check script."""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load .env file
load_dotenv(r'C:\Users\Jean-Yves\thevolumeainative\.env')

api_key = os.getenv('ALPACA_CRYPTO_KEY') or os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_CRYPTO_SECRET') or os.getenv('ALPACA_SECRET_KEY')

if not api_key or not api_secret:
    print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    exit(1)

client = TradingClient(api_key, api_secret, paper=True)
positions = client.get_all_positions()

print("=" * 60)
print("CURRENT ALPACA POSITIONS")
print("=" * 60)

total = 0.0
for p in positions:
    qty = float(p.qty)
    entry = float(p.avg_entry_price)
    value = float(p.market_value)
    pnl = float(p.unrealized_pl)
    total += value
    print(f"{p.symbol}: {qty:.6f} @ ${entry:,.2f} = ${value:,.2f} (P&L: ${pnl:,.2f})")

print("=" * 60)
print(f"TOTAL VALUE: ${total:,.2f}")
print(f"EXPECTED (~3 x $66.60): $199.80")
print(f"STATUS: {'OK - Correct sizing' if abs(total - 200) < 50 else 'DOUBLED - Need to close excess'}")
print("=" * 60)
