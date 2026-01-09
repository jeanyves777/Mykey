#!/usr/bin/env python3
"""
Quick Backtest for HTF Confluence Strategy
"""
import sys
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trading_system", "Binance_Futures_Trading"))

from strategies.htf_confluence_strategy_new import (
    HTFConfluenceStrategy,
    MODERATE_CONFIG,
    TrendDirection,
    SignalStrength
)


def fetch_binance_data(symbol: str, interval: str, days: int):
    """Fetch historical data from Binance Futures API"""
    try:
        base_url = "https://fapi.binance.com"
        endpoint = "/fapi/v1/klines"
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_klines = []
        limit = 1500
        
        while start_time < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "limit": limit
            }
            
            try:
                response = requests.get(base_url + endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"    Error fetching data: {e}")
                break
            
            if not data or isinstance(data, dict):
                print(f"    No more data or error response")
                break
            
            all_klines.extend(data)
            start_time = data[-1][0] + 1
            
            if len(data) < limit:
                break
        
        if not all_klines:
            print(f"    No klines data returned")
            return None
        
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"    Exception in fetch_binance_data: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_backtest(symbol: str = "BTCUSDT", days: int = 30):
    """Run a quick backtest"""
    print(f"\n{'='*60}")
    print(f"HTF CONFLUENCE STRATEGY BACKTEST - {symbol}")
    print(f"{'='*60}\n")
    
    # Fetch data
    print(f"Fetching data...")
    ltf_df = fetch_binance_data(symbol, "15m", days)
    htf_df = fetch_binance_data(symbol, "4h", days + 20)
    
    if ltf_df is None or htf_df is None:
        print("ERROR: Could not fetch data")
        return
    
    print(f"  LTF (15m): {len(ltf_df)} candles")
    print(f"  HTF (4h): {len(htf_df)} candles")
    
    # Initialize strategy
    strategy = HTFConfluenceStrategy(**MODERATE_CONFIG)
    
    # Backtest variables
    balance = 1000.0
    initial_balance = balance
    position = None
    trades = []
    
    print(f"\nRunning backtest...")
    print(f"  Initial Balance: ${balance:.2f}")
    print(f"  Leverage: {strategy.leverage}x")
    print(f"  TP: {strategy.tp_roi*100:.0f}% ROI | SL: {strategy.sl_roi*100:.0f}% ROI")
    
    # Iterate through LTF bars
    for i in range(100, len(ltf_df)):
        current_ltf = ltf_df.iloc[:i+1]
        current_time = ltf_df.index[i]
        
        # Get HTF data up to current time
        current_htf = htf_df[htf_df.index <= current_time]
        
        if len(current_htf) < 50:
            continue
        
        # Check for exit if in position
        if position is not None:
            current_price = current_ltf['close'].iloc[-1]
            
            # Check TP/SL
            if position['side'] == 'LONG':
                if current_price >= position['tp_price']:
                    pnl = (position['tp_price'] - position['entry_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['tp_price'],
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    print(f"  [{current_time}] CLOSE LONG @ ${position['tp_price']:.2f} (TP) | PNL: ${pnl:+.2f}")
                    position = None
                elif current_price <= position['sl_price']:
                    pnl = (position['sl_price'] - position['entry_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['sl_price'],
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    print(f"  [{current_time}] CLOSE LONG @ ${position['sl_price']:.2f} (SL) | PNL: ${pnl:+.2f}")
                    position = None
            else:  # SHORT
                if current_price <= position['tp_price']:
                    pnl = (position['entry_price'] - position['tp_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['tp_price'],
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    print(f"  [{current_time}] CLOSE SHORT @ ${position['tp_price']:.2f} (TP) | PNL: ${pnl:+.2f}")
                    position = None
                elif current_price >= position['sl_price']:
                    pnl = (position['entry_price'] - position['sl_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['sl_price'],
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    print(f"  [{current_time}] CLOSE SHORT @ ${position['sl_price']:.2f} (SL) | PNL: ${pnl:+.2f}")
                    position = None
        
        # Check for entry if no position
        if position is None:
            signal = strategy.should_enter(current_ltf, current_htf, i)
            
            if signal.action in ["BUY", "SELL"]:
                # Calculate position size (risk 2% of balance)
                risk_amount = balance * 0.02
                sl_pct = strategy.sl_roi / strategy.leverage
                margin = risk_amount / sl_pct
                
                position = {
                    'side': 'LONG' if signal.action == "BUY" else 'SHORT',
                    'entry_price': signal.entry_price,
                    'tp_price': signal.take_profit,
                    'sl_price': signal.stop_loss,
                    'margin': margin,
                    'entry_time': current_time,
                    'confluence': signal.confluence_score
                }
                
                print(f"\n  [{current_time}] OPEN {position['side']} @ ${signal.entry_price:.2f}")
                print(f"    Confluence: {signal.confluence_score}/4 | {signal.strength.value}")
                print(f"    TP: ${signal.take_profit:.2f} | SL: ${signal.stop_loss:.2f}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total PNL: ${balance - initial_balance:+.2f} ({((balance/initial_balance - 1)*100):+.1f}%)")
    print(f"\nTrades: {len(trades)}")
    
    if trades:
        wins = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']
        
        print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
        
        if wins:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            print(f"  Avg Win: ${avg_win:.2f}")
        
        if losses:
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            print(f"  Avg Loss: ${avg_loss:.2f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test with BTC
    run_backtest("BTCUSDT", days=30)
