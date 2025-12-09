"""Test the dual signal validation system with different timeframes."""
from trading_system.strategies import COINDaily0DTEMomentum
from trading_system.engine.alpaca_client import AlpacaClient
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv

load_dotenv()

EST = pytz.timezone('America/New_York')
client = AlpacaClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_API_SECRET'),
    paper=True
)

# Get 1-MINUTE bars for Technical Scoring
start_time_1min = datetime.now(EST) - timedelta(hours=3)
bars_1min = client.get_stock_bars('COIN', timeframe='1Min', start=start_time_1min, limit=180)
print(f'Got {len(bars_1min)} 1-min bars')

# Get 5-MINUTE bars for Price Action
start_time_5min = datetime.now(EST) - timedelta(hours=6)
bars_5min = client.get_stock_bars('COIN', timeframe='5Min', start=start_time_5min, limit=100)
print(f'Got {len(bars_5min)} 5-min bars')

bars_1min = bars_1min[-60:]
bars_5min = bars_5min[-30:]

# Test Method 1: Technical Scoring (1-MIN)
print()
print('=' * 65)
print('  METHOD 1: Technical Scoring (1-MIN BARS)')
print('=' * 65)
tech = COINDaily0DTEMomentum.calculate_signal_from_bars(bars_1min)
print(f"Signal: {tech['signal']} | Confidence: {tech['confidence']}")
print(f"Bull: {tech['bullish_score']}/17 | Bear: {tech['bearish_score']}/17")
for sig in tech.get('bullish_signals', []):
    print(f"  + {sig}")
for sig in tech.get('bearish_signals', []):
    print(f"  - {sig}")

# Test Method 2: Price Action (5-MIN)
print()
print('=' * 65)
print('  METHOD 2: Price Action (5-MIN BARS)')
print('=' * 65)
pa = COINDaily0DTEMomentum.calculate_price_action_signal(bars_5min)
print(f"Signal: {pa['signal']} | Strength: {pa['strength']}")
print(f"Bull: {pa['bullish_points']} | Bear: {pa['bearish_points']}")
for r in pa['reasons']:
    print(f"  * {r}")

# Final decision
print()
print('=' * 65)
print('  FINAL DECISION')
print('=' * 65)
ts = tech['signal']
ps = pa['signal']

if ts == ps and ts != 'NEUTRAL':
    print(f"*** CONFIRMED {ts} - BOTH METHODS AGREE! ***")
elif ts == 'NEUTRAL' or ps == 'NEUTRAL':
    print(f"NO TRADE - One or both methods neutral (Tech 1-min: {ts}, PA 5-min: {ps})")
else:
    print(f"CONFLICTING SIGNALS - NO TRADE (Tech 1-min: {ts} vs PA 5-min: {ps})")
