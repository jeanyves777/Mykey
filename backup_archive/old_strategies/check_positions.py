import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Binance_Futures_Trading'))
from engine.binance_client import BinanceClient
bc = BinanceClient()
client = bc.client

positions = client.futures_position_information()
print('=== CURRENT POSITIONS ===')
found = False
for pos in positions:
    amt = float(pos['positionAmt'])
    if amt != 0:
        found = True
        symbol = pos['symbol']
        entry = float(pos['entryPrice'])
        mark = float(pos['markPrice'])
        pnl = float(pos['unRealizedProfit'])
        leverage = int(pos['leverage'])
        side = 'LONG' if amt > 0 else 'SHORT'

        if side == 'LONG':
            pnl_pct = ((mark - entry) / entry) * 100
        else:
            pnl_pct = ((entry - mark) / entry) * 100

        roi = pnl_pct * leverage
        print(f'{symbol}: {side} | Entry: {entry:.4f} | Now: {mark:.4f} | PnL: ${pnl:.2f} ({roi:.1f}% ROI)')

if not found:
    print('No open positions')
