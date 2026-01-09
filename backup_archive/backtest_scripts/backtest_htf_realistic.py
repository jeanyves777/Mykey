#!/usr/bin/env python3
"""
Realistic Backtest for HTF Confluence Strategy
Tests in various market conditions: bullish, bearish, sideways
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
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG,
    TrendDirection,
    SignalStrength
)


def generate_realistic_market(num_candles: int, start_price: float = 50000):
    """
    Generate realistic market data with different phases:
    - Trending up (30%)
    - Trending down (30%)
    - Sideways/choppy (40%)
    """
    np.random.seed(123)  # Different seed for variety
    
    prices = [start_price]
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(num_candles-1, -1, -1)]
    
    # Define market phases
    phase_lengths = [
        ("bullish", 100),
        ("sideways", 80),
        ("bullish", 60),
        ("bearish", 100),
        ("sideways", 80),
        ("bearish", 80),
    ]
    
    candle_idx = 0
    for phase, length in phase_lengths:
        for _ in range(length):
            if candle_idx >= num_candles - 1:
                break
            
            if phase == "bullish":
                # Bullish trend
                change = np.random.uniform(0.0003, 0.0025)
                noise = np.random.uniform(-0.0008, 0.0008)
            elif phase == "bearish":
                # Bearish trend
                change = np.random.uniform(-0.0025, -0.0003)
                noise = np.random.uniform(-0.0008, 0.0008)
            else:  # sideways
                # Choppy sideways
                change = np.random.uniform(-0.0015, 0.0015)
                noise = np.random.uniform(-0.001, 0.001)
            
            new_price = prices[-1] * (1 + change + noise)
            prices.append(new_price)
            candle_idx += 1
    
    # Fill remaining if needed
    while len(prices) < num_candles:
        change = np.random.uniform(-0.001, 0.001)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.004)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.004)) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    })
    
    df.set_index('timestamp', inplace=True)
    return df


def run_realistic_backtest(config_name="MODERATE"):
    """Run a realistic backtest with mixed market conditions"""
    
    # Select config
    configs = {
        "CONSERVATIVE": CONSERVATIVE_CONFIG,
        "MODERATE": MODERATE_CONFIG,
        "AGGRESSIVE": AGGRESSIVE_CONFIG
    }
    config = configs.get(config_name, MODERATE_CONFIG)
    
    print(f"\n{'='*70}")
    print(f"HTF CONFLUENCE STRATEGY - REALISTIC BACKTEST ({config_name})")
    print(f"{'='*70}\n")
    
    # Generate realistic market data
    print("Generating realistic market data (mixed conditions)...")
    print("  Market phases: Bullish → Sideways → Bullish → Bearish → Sideways → Bearish")
    
    # HTF (15m): Longer history
    htf_df = generate_realistic_market(250, start_price=50000)
    
    # LTF (5m): Match HTF end price
    ltf_df = generate_realistic_market(500, start_price=htf_df['close'].iloc[-200])
    
    print(f"  HTF (15m): {len(htf_df)} candles")
    print(f"  LTF (5m): {len(ltf_df)} candles")
    print(f"  HTF Price: ${htf_df['close'].iloc[0]:.2f} → ${htf_df['close'].iloc[-1]:.2f} ({(htf_df['close'].iloc[-1]/htf_df['close'].iloc[0]-1)*100:+.1f}%)")
    print(f"  LTF Price: ${ltf_df['close'].iloc[0]:.2f} → ${ltf_df['close'].iloc[-1]:.2f} ({(ltf_df['close'].iloc[-1]/ltf_df['close'].iloc[0]-1)*100:+.1f}%)")
    
    # Initialize strategy
    strategy = HTFConfluenceStrategy(**config)
    
    # Backtest variables
    balance = 1000.0
    initial_balance = balance
    max_balance = balance
    position = None
    trades = []
    equity_curve = [balance]
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Initial Balance: ${balance:.2f}")
    print(f"Leverage: {strategy.leverage}x")
    print(f"TP: {strategy.tp_roi*100:.0f}% ROI ({strategy.tp_roi/strategy.leverage*100:.2f}% price move)")
    print(f"SL: {strategy.sl_roi*100:.0f}% ROI ({strategy.sl_roi/strategy.leverage*100:.2f}% price move)")
    print(f"Min Confluence: {strategy.min_confluence_score}/4 conditions")
    print(f"HTF EMA: {strategy.htf_ema_period} period")
    print(f"LTF EMAs: {strategy.ema_fast}/{strategy.ema_slow} period")
    print(f"Cooldown: {strategy.min_bars_between_signals} bars")
    
    print(f"\n{'='*70}")
    print(f"EXECUTING TRADES")
    print(f"{'='*70}\n")
    
    # Iterate through LTF bars
    for i in range(100, len(ltf_df)):
        current_ltf = ltf_df.iloc[:i+1]
        current_time = ltf_df.index[i]
        current_price = current_ltf['close'].iloc[-1]
        
        # Check for exit if in position
        if position is not None:
            # Check TP/SL
            hit_tp = False
            hit_sl = False
            
            if position['side'] == 'LONG':
                hit_tp = current_price >= position['tp_price']
                hit_sl = current_price <= position['sl_price']
            else:  # SHORT
                hit_tp = current_price <= position['tp_price']
                hit_sl = current_price >= position['sl_price']
            
            if hit_tp or hit_sl:
                exit_price = position['tp_price'] if hit_tp else position['sl_price']
                exit_type = "TP" if hit_tp else "SL"
                result = "WIN" if hit_tp else "LOSS"
                
                # Calculate PNL
                if position['side'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) / position['entry_price'] * strategy.leverage * position['margin']
                else:
                    pnl = (position['entry_price'] - exit_price) / position['entry_price'] * strategy.leverage * position['margin']
                
                balance += pnl
                max_balance = max(max_balance, balance)
                roi = (pnl / position['margin']) * 100
                
                trade_duration = (current_time - position['entry_time']).total_seconds() / 3600  # hours
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'duration_hours': trade_duration,
                    'side': position['side'],
                    'entry': position['entry_price'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'roi': roi,
                    'result': result,
                    'confluence': position['confluence']
                })
                
                emoji = "✅" if result == "WIN" else "❌"
                print(f"{emoji} {position['side']} {result} @ ${exit_price:.2f} ({exit_type}) | PNL: ${pnl:+.2f} ({roi:+.1f}% ROI) | Balance: ${balance:.2f}")
                
                position = None
            
            equity_curve.append(balance)
        
        # Check for entry if no position
        if position is None:
            signal = strategy.should_enter(current_ltf, htf_df, i)
            
            if signal.action in ["BUY", "SELL"]:
                # Calculate position size (risk 2% of balance per trade)
                risk_amount = balance * 0.02
                sl_pct = strategy.sl_roi / strategy.leverage
                margin = min(risk_amount / sl_pct, balance * 0.2)  # Cap at 20% of balance
                
                if margin < 10:  # Skip if margin too small
                    continue
                
                position = {
                    'side': 'LONG' if signal.action == "BUY" else 'SHORT',
                    'entry_price': signal.entry_price,
                    'tp_price': signal.take_profit,
                    'sl_price': signal.stop_loss,
                    'margin': margin,
                    'entry_time': current_time,
                    'confluence': signal.confluence_score
                }
                
                print(f"\n🎯 OPEN {position['side']} @ ${signal.entry_price:.2f} | Confluence: {signal.confluence_score}/4 ({signal.strength.value})")
                print(f"   TP: ${signal.take_profit:.2f} | SL: ${signal.stop_loss:.2f} | Margin: ${margin:.2f}")
    
    # Close any remaining position at end
    if position is not None:
        print(f"\n⚠️  Position still open at end of backtest - closing at market")
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': ltf_df.index[-1],
            'duration_hours': 0,
            'side': position['side'],
            'entry': position['entry_price'],
            'exit': ltf_df['close'].iloc[-1],
            'pnl': 0,
            'roi': 0,
            'result': 'OPEN',
            'confluence': position['confluence']
        })
    
    # Calculate statistics
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")
    
    total_return = ((balance/initial_balance - 1)*100)
    max_drawdown_pct = ((max_balance - balance) / max_balance * 100) if max_balance > balance else 0
    
    print(f"💰 PROFIT & LOSS")
    print(f"   Initial Balance:    ${initial_balance:,.2f}")
    print(f"   Final Balance:      ${balance:,.2f}")
    print(f"   Total Return:       {total_return:+.2f}%")
    print(f"   Total PNL:          ${balance - initial_balance:+,.2f}")
    print(f"   Max Drawdown:       {max_drawdown_pct:.2f}%")
    
    print(f"\n📊 TRADE STATISTICS")
    completed_trades = [t for t in trades if t['result'] != 'OPEN']
    print(f"   Total Trades:       {len(completed_trades)}")
    
    if completed_trades:
        wins = [t for t in completed_trades if t['result'] == 'WIN']
        losses = [t for t in completed_trades if t['result'] == 'LOSS']
        
        win_rate = (len(wins)/len(completed_trades)*100)
        print(f"   Wins:               {len(wins)} ({win_rate:.1f}%)")
        print(f"   Losses:             {len(losses)} ({100-win_rate:.1f}%)")
        
        if wins:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            avg_win_roi = sum(t['roi'] for t in wins) / len(wins)
            avg_win_duration = sum(t['duration_hours'] for t in wins) / len(wins)
            print(f"   Avg Win:            ${avg_win:.2f} ({avg_win_roi:+.1f}% ROI)")
            print(f"   Avg Win Duration:   {avg_win_duration:.1f} hours")
        
        if losses:
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            avg_loss_roi = sum(t['roi'] for t in losses) / len(losses)
            avg_loss_duration = sum(t['duration_hours'] for t in losses) / len(losses)
            print(f"   Avg Loss:           ${avg_loss:.2f} ({avg_loss_roi:.1f}% ROI)")
            print(f"   Avg Loss Duration:  {avg_loss_duration:.1f} hours")
        
        if wins and losses:
            profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses))
            print(f"   Profit Factor:      {profit_factor:.2f}")
            
            expectancy = (len(wins)/len(completed_trades) * abs(avg_win)) - (len(losses)/len(completed_trades) * abs(avg_loss))
            print(f"   Expectancy:         ${expectancy:.2f} per trade")
    
    print(f"\n{'='*70}\n")
    
    return {
        'config': config_name,
        'balance': balance,
        'return_pct': total_return,
        'trades': len(completed_trades),
        'win_rate': win_rate if completed_trades else 0,
        'profit_factor': profit_factor if (wins and losses) else 0
    }


if __name__ == "__main__":
    # Test all three configs
    results = []
    
    for config_name in ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]:
        result = run_realistic_backtest(config_name)
        results.append(result)
        print("\n" + "="*70 + "\n")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print(f"CONFIGURATION COMPARISON")
    print(f"{'='*70}\n")
    print(f"{'Config':<15} {'Return':<12} {'Trades':<10} {'Win Rate':<12} {'Profit Factor':<15}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['config']:<15} {r['return_pct']:>+10.2f}%  {r['trades']:<10} {r['win_rate']:>9.1f}%  {r['profit_factor']:>14.2f}")
    
    print(f"\n{'='*70}\n")
