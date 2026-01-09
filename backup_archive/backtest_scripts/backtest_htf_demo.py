#!/usr/bin/env python3
"""
Quick Demo Backtest for HTF Confluence Strategy with Synthetic Data
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trading_system", "Binance_Futures_Trading"))

from strategies.htf_confluence_strategy_new import (
    HTFConfluenceStrategy,
    MODERATE_CONFIG,
    TrendDirection,
    SignalStrength
)


def generate_trending_data(num_candles: int, start_price: float = 50000, trend: str = "bullish"):
    """Generate synthetic trending price data"""
    np.random.seed(42)
    
    prices = [start_price]
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(num_candles-1, -1, -1)]
    
    for i in range(num_candles - 1):
        if trend == "bullish":
            # Bullish trend with some volatility
            change = np.random.uniform(0.0005, 0.003)  # 0.05% to 0.3% up
            noise = np.random.uniform(-0.001, 0.001)   # Small noise
            new_price = prices[-1] * (1 + change + noise)
        elif trend == "bearish":
            # Bearish trend
            change = np.random.uniform(-0.003, -0.0005)
            noise = np.random.uniform(-0.001, 0.001)
            new_price = prices[-1] * (1 + change + noise)
        else:  # sideways
            change = np.random.uniform(-0.002, 0.002)
            new_price = prices[-1] * (1 + change)
        
        prices.append(new_price)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    })
    
    df.set_index('timestamp', inplace=True)
    return df


def run_backtest():
    """Run a demonstration backtest"""
    print(f"\n{'='*60}")
    print(f"HTF CONFLUENCE STRATEGY - DEMO BACKTEST")
    print(f"{'='*60}\n")
    
    # Generate synthetic data
    print("Generating synthetic trending market data...")
    
    # HTF (15m): 200 candles for EMA warmup
    print("  HTF (15m): Bullish trend")
    htf_df = generate_trending_data(200, start_price=50000, trend="bullish")
    
    # LTF (5m): 500 candles
    print("  LTF (5m): Following HTF trend with pullbacks")
    ltf_df = generate_trending_data(500, start_price=htf_df['close'].iloc[-1], trend="bullish")
    
    print(f"  HTF candles: {len(htf_df)}")
    print(f"  LTF candles: {len(ltf_df)}")
    print(f"  HTF Price: ${htf_df['close'].iloc[0]:.2f} -> ${htf_df['close'].iloc[-1]:.2f}")
    print(f"  LTF Price: ${ltf_df['close'].iloc[0]:.2f} -> ${ltf_df['close'].iloc[-1]:.2f}")
    
    # Initialize strategy
    strategy = HTFConfluenceStrategy(**MODERATE_CONFIG)
    
    # Backtest variables
    balance = 1000.0
    initial_balance = balance
    position = None
    trades = []
    
    print(f"\n{'='*60}")
    print(f"BACKTEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Initial Balance: ${balance:.2f}")
    print(f"Leverage: {strategy.leverage}x")
    print(f"TP: {strategy.tp_roi*100:.0f}% ROI ({strategy.tp_roi/strategy.leverage*100:.2f}% price move)")
    print(f"SL: {strategy.sl_roi*100:.0f}% ROI ({strategy.sl_roi/strategy.leverage*100:.2f}% price move)")
    print(f"Risk per trade: 2% of balance")
    print(f"Min Confluence: {strategy.min_confluence_score}/4 conditions")
    
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTEST")
    print(f"{'='*60}\n")
    
    # Iterate through LTF bars
    signals_checked = 0
    signals_generated = 0
    
    for i in range(100, len(ltf_df)):
        current_ltf = ltf_df.iloc[:i+1]
        current_time = ltf_df.index[i]
        current_price = current_ltf['close'].iloc[-1]
        
        signals_checked += 1
        
        # Check for exit if in position
        if position is not None:
            # Check TP/SL
            if position['side'] == 'LONG':
                if current_price >= position['tp_price']:
                    pnl = (position['tp_price'] - position['entry_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    roi = (pnl / position['margin']) * 100
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['tp_price'],
                        'pnl': pnl,
                        'roi': roi,
                        'result': 'WIN'
                    })
                    print(f"✅ LONG WIN @ ${position['tp_price']:.2f} (TP)")
                    print(f"   PNL: ${pnl:+.2f} | ROI: {roi:+.1f}% | Balance: ${balance:.2f}\n")
                    position = None
                elif current_price <= position['sl_price']:
                    pnl = (position['sl_price'] - position['entry_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    roi = (pnl / position['margin']) * 100
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['sl_price'],
                        'pnl': pnl,
                        'roi': roi,
                        'result': 'LOSS'
                    })
                    print(f"❌ LONG LOSS @ ${position['sl_price']:.2f} (SL)")
                    print(f"   PNL: ${pnl:+.2f} | ROI: {roi:+.1f}% | Balance: ${balance:.2f}\n")
                    position = None
            else:  # SHORT
                if current_price <= position['tp_price']:
                    pnl = (position['entry_price'] - position['tp_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    roi = (pnl / position['margin']) * 100
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['tp_price'],
                        'pnl': pnl,
                        'roi': roi,
                        'result': 'WIN'
                    })
                    print(f"✅ SHORT WIN @ ${position['tp_price']:.2f} (TP)")
                    print(f"   PNL: ${pnl:+.2f} | ROI: {roi:+.1f}% | Balance: ${balance:.2f}\n")
                    position = None
                elif current_price >= position['sl_price']:
                    pnl = (position['entry_price'] - position['sl_price']) / position['entry_price'] * strategy.leverage * position['margin']
                    balance += pnl
                    roi = (pnl / position['margin']) * 100
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': position['sl_price'],
                        'pnl': pnl,
                        'roi': roi,
                        'result': 'LOSS'
                    })
                    print(f"❌ SHORT LOSS @ ${position['sl_price']:.2f} (SL)")
                    print(f"   PNL: ${pnl:+.2f} | ROI: {roi:+.1f}% | Balance: ${balance:.2f}\n")
                    position = None
        
        # Check for entry if no position
        if position is None and i % 4 == 0:  # Check every 4 bars to reduce noise
            signal = strategy.should_enter(current_ltf, htf_df, i)
            
            if signal.action in ["BUY", "SELL"]:
                signals_generated += 1
                
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
                
                print(f"📊 {current_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"🎯 OPEN {position['side']} @ ${signal.entry_price:.2f}")
                print(f"   Confluence: {signal.confluence_score}/4 ({signal.strength.value})")
                print(f"   TP: ${signal.take_profit:.2f} | SL: ${signal.stop_loss:.2f}")
                print(f"   Margin: ${margin:.2f} | Risk: ${risk_amount:.2f}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}\n")
    
    print(f"💰 PERFORMANCE")
    print(f"   Initial Balance: ${initial_balance:.2f}")
    print(f"   Final Balance: ${balance:.2f}")
    total_return = ((balance/initial_balance - 1)*100)
    print(f"   Total Return: {total_return:+.1f}%")
    print(f"   Total PNL: ${balance - initial_balance:+.2f}")
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"   Signals Checked: {signals_checked}")
    print(f"   Signals Generated: {signals_generated}")
    print(f"   Total Trades: {len(trades)}")
    
    if trades:
        wins = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']
        
        win_rate = (len(wins)/len(trades)*100) if trades else 0
        print(f"   Wins: {len(wins)} ({win_rate:.1f}%)")
        print(f"   Losses: {len(losses)} ({100-win_rate:.1f}%)")
        
        if wins:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            avg_win_roi = sum(t['roi'] for t in wins) / len(wins)
            print(f"   Avg Win: ${avg_win:.2f} ({avg_win_roi:.1f}% ROI)")
        
        if losses:
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            avg_loss_roi = sum(t['roi'] for t in losses) / len(losses)
            print(f"   Avg Loss: ${avg_loss:.2f} ({avg_loss_roi:.1f}% ROI)")
        
        if wins and losses:
            profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses))
            print(f"   Profit Factor: {profit_factor:.2f}")
    else:
        print(f"   No trades executed")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run_backtest()
