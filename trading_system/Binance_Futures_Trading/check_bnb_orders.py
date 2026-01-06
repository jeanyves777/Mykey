#!/usr/bin/env python3
"""Check BNB orders on Binance"""
import sys
sys.path.insert(0, '.')
from engine.binance_client import BinanceClient

client = BinanceClient(testnet=False)

# Get BNB position
positions = client.get_account_info().get('positions', [])
bnb_pos = next((p for p in positions if p['symbol'] == 'BNBUSDT' and float(p['positionAmt']) != 0), None)
if bnb_pos:
    entry = float(bnb_pos['entryPrice'])
    print(f'BNB Entry: ${entry:.2f}')
    print(f'Expected TP (30% ROI): ${entry * 1.015:.2f}')
    print(f'Expected SL (10% ROI): ${entry * 0.995:.2f}')

# Get algo orders for BNB
print('\nAlgo orders:')
algo = client.get_algo_orders('BNBUSDT')
for o in algo:
    order_type = o.get('type', 'N/A')
    trigger = o.get('triggerPrice', 'N/A')
    print(f'  {order_type} @ {trigger}')

# Get regular orders for BNB
print('\nRegular orders:')
reg = client.get_open_orders('BNBUSDT')
for o in reg:
    order_type = o.get('type', 'N/A')
    stop = o.get('stopPrice', o.get('price', 'N/A'))
    print(f'  {order_type} @ {stop}')
