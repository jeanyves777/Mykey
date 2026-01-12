"""
Parameter Optimization Script
Tests different parameter combinations to find optimal settings
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.oanda_client import OANDAClient
from engine.htf_confluence_strategy import HTFConfluenceForexStrategy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import product
import json
from config.trading_config import OANDA_CONFIG, FOREX_SYMBOLS, SYMBOL_SETTINGS

# Parameter grid to test
PARAM_GRID = {
    'min_confluence_score': [4, 5, 6, 7],
    'damage_control_pips': [-8, -10, -12, -14],
    'trailing_lock_activation': [6, 8, 10],
    'profit_lock_activation': [10, 12, 15],
}

def run_optimization():
    """Run parameter optimization across all combinations"""
    
    print("=" * 70)
    print("PARAMETER OPTIMIZATION - HTF Confluence Strategy")
    print("=" * 70)
    print(f"Testing {len(list(product(*PARAM_GRID.values())))} parameter combinations")
    print(f"Symbols: {', '.join(FOREX_SYMBOLS)}")
    print(f"Period: 30 days")
    print("=" * 70)
    
    # Initialize client
    client = OANDAClient(OANDA_CONFIG)
    
    # Get account info
    account = client.get_account_summary()
    initial_balance = float(account['balance'])
    print(f"\nStarting Balance: ${initial_balance:,.2f}")
    print(f"Account: {OANDA_CONFIG['account_id']}")
    
    results = []
    combination_num = 0
    total_combinations = len(list(product(*PARAM_GRID.values())))
    
    # Test each parameter combination
    for confluence, damage_control, trailing_activation, profit_activation in product(
        PARAM_GRID['min_confluence_score'],
        PARAM_GRID['damage_control_pips'],
        PARAM_GRID['trailing_lock_activation'],
        PARAM_GRID['profit_lock_activation']
    ):
        combination_num += 1
        print(f"\n{'='*70}")
        print(f"Testing Combination {combination_num}/{total_combinations}")
        print(f"{'='*70}")
        print(f"Min Confluence: {confluence}/8")
        print(f"Damage Control: {damage_control} pips")
        print(f"Trailing Lock: {trailing_activation} pips")
        print(f"Profit Lock: {profit_activation} pips")
        
        # Run backtest with these parameters
        result = backtest_with_params(
            client,
            initial_balance,
            confluence,
            damage_control,
            trailing_activation,
            profit_activation
        )
        
        # Store results
        result['params'] = {
            'min_confluence_score': confluence,
            'damage_control_pips': damage_control,
            'trailing_lock_activation': trailing_activation,
            'profit_lock_activation': profit_activation,
        }
        results.append(result)
        
        # Print summary
        print(f"\n📊 Result: {result['total_trades']} trades, "
              f"{result['win_rate']:.1f}% WR, "
              f"ROI: {result['roi']:.1f}%, "
              f"Drawdown: {result['max_drawdown']:.1f}%")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    # Sort by ROI
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roi', ascending=False)
    
    print("\n🏆 Top 10 Configurations by ROI:")
    print("-" * 70)
    for idx, row in results_df.head(10).iterrows():
        params = row['params']
        print(f"\n#{idx+1}: ROI {row['roi']:.1f}% | Trades: {row['total_trades']} | WR: {row['win_rate']:.1f}% | DD: {row['max_drawdown']:.1f}%")
        print(f"   Confluence: {params['min_confluence_score']}/8 | "
              f"DC: {params['damage_control_pips']}p | "
              f"Trail: {params['trailing_lock_activation']}p | "
              f"Lock: {params['profit_lock_activation']}p")
    
    # Best by different metrics
    print("\n" + "=" * 70)
    print("BEST BY DIFFERENT METRICS")
    print("=" * 70)
    
    best_roi = results_df.iloc[0]
    best_winrate = results_df.sort_values('win_rate', ascending=False).iloc[0]
    best_drawdown = results_df.sort_values('max_drawdown', ascending=True).iloc[0]
    best_sharpe = results_df.sort_values('sharpe_ratio', ascending=False).iloc[0] if 'sharpe_ratio' in results_df.columns else None
    
    print(f"\n🎯 Best ROI: {best_roi['roi']:.1f}%")
    print(f"   Params: {best_roi['params']}")
    
    print(f"\n✅ Best Win Rate: {best_winrate['win_rate']:.1f}%")
    print(f"   Params: {best_winrate['params']}")
    
    print(f"\n🛡️ Best Drawdown: {best_drawdown['max_drawdown']:.1f}%")
    print(f"   Params: {best_drawdown['params']}")
    
    if best_sharpe is not None:
        print(f"\n📈 Best Sharpe Ratio: {best_sharpe['sharpe_ratio']:.2f}")
        print(f"   Params: {best_sharpe['params']}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to: {filename}")
    
    return results_df

def backtest_with_params(client, initial_balance, confluence, damage_control, trailing_activation, profit_activation):
    """Run backtest with specific parameters"""
    
    balance = initial_balance
    peak_balance = initial_balance
    trades = []
    symbol_stats = {symbol: {'wins': 0, 'losses': 0, 'pnl': 0} for symbol in FOREX_SYMBOLS}
    
    # Fetch data for all symbols
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in FOREX_SYMBOLS:
        # Fetch candles for all timeframes
        try:
            h4_candles = client.get_candles(symbol, 'H4', start_date, end_date)
            h1_candles = client.get_candles(symbol, 'H1', start_date, end_date)
            m15_candles = client.get_candles(symbol, 'M15', start_date, end_date)
            m5_candles = client.get_candles(symbol, 'M5', start_date, end_date)
            
            if not all([h4_candles, h1_candles, m15_candles, m5_candles]):
                continue
            
            # Create strategy instance with custom params
            strategy = HTFConfluenceForexStrategy(symbol)
            
            # Override parameters
            strategy.min_confluence_score = confluence
            
            # Scan for signals
            for i in range(len(m15_candles)):
                current_time = m15_candles[i]['time']
                
                # Get HTF context
                h4_idx = next((j for j, c in enumerate(h4_candles) if c['time'] <= current_time), None)
                h1_idx = next((j for j, c in enumerate(h1_candles) if c['time'] <= current_time), None)
                m5_idx = next((j for j, c in enumerate(m5_candles) if c['time'] <= current_time), None)
                
                if h4_idx is None or h1_idx is None or m5_idx is None:
                    continue
                
                # Check for signal
                signal = strategy.should_enter(
                    h4_candles[:h4_idx+1],
                    h1_candles[:h1_idx+1],
                    m15_candles[:i+1],
                    m5_candles[:m5_idx+1]
                )
                
                if signal['signal'] != 'NONE':
                    # Calculate exit levels
                    entry_price = m15_candles[i]['close']
                    position_size = balance * 0.02 / SYMBOL_SETTINGS[symbol]['sl_pips']  # 2% risk
                    
                    # Simulate trade outcome with custom parameters
                    outcome = simulate_trade_outcome(
                        symbol,
                        signal['signal'],
                        entry_price,
                        position_size,
                        m15_candles[i:],
                        damage_control,
                        trailing_activation,
                        profit_activation
                    )
                    
                    if outcome:
                        balance += outcome['pnl']
                        peak_balance = max(peak_balance, balance)
                        
                        trades.append(outcome)
                        
                        if outcome['pnl'] > 0:
                            symbol_stats[symbol]['wins'] += 1
                        else:
                            symbol_stats[symbol]['losses'] += 1
                        symbol_stats[symbol]['pnl'] += outcome['pnl']
                        
                        # Account protection
                        min_balance = initial_balance * 0.20
                        if balance < min_balance:
                            break
                        
                        max_dd_pct = ((peak_balance - balance) / peak_balance) * 100
                        if max_dd_pct > 50:
                            break
        
        except Exception as e:
            print(f"   ⚠️ Error processing {symbol}: {str(e)}")
            continue
    
    # Calculate metrics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    final_balance = balance
    total_pnl = final_balance - initial_balance
    roi = (total_pnl / initial_balance) * 100
    max_drawdown = ((peak_balance - balance) / peak_balance) * 100 if peak_balance > balance else 0
    
    # Calculate Sharpe ratio (simplified)
    if trades:
        returns = [t['pnl'] / initial_balance for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_pnl': total_pnl,
        'roi': roi,
        'peak_balance': peak_balance,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # Annualized
        'symbol_stats': symbol_stats,
    }

def simulate_trade_outcome(symbol, direction, entry_price, position_size, future_candles, 
                          damage_control, trailing_activation, profit_activation):
    """Simulate trade outcome with custom exit parameters"""
    
    settings = SYMBOL_SETTINGS[symbol]
    pip_value = 10 ** settings['pip_location']
    
    # Calculate exits
    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    
    if direction == 'BUY':
        tp_price = entry_price + (tp_pips * pip_value)
        sl_price = entry_price - (sl_pips * pip_value)
        dc_price = entry_price + (damage_control * pip_value)
        trailing_lock_price = entry_price + (trailing_activation * pip_value)
    else:  # SELL
        tp_price = entry_price - (tp_pips * pip_value)
        sl_price = entry_price + (sl_pips * pip_value)
        dc_price = entry_price - (damage_control * pip_value)
        trailing_lock_price = entry_price - (trailing_activation * pip_value)
    
    trailing_active = False
    highest_profit = 0
    
    # Simulate price movement
    for candle in future_candles[:200]:  # Max 200 candles (50 hours)
        high = candle['high']
        low = candle['low']
        
        if direction == 'BUY':
            # Check TP
            if high >= tp_price:
                pnl = (tp_price - entry_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'TP', 'duration': 15}
            
            # Check SL
            if low <= sl_price:
                pnl = (sl_price - entry_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'SL', 'duration': 15}
            
            # Trailing lock logic
            current_profit_pips = (high - entry_price) / pip_value
            if current_profit_pips >= trailing_activation:
                trailing_active = True
                highest_profit = max(highest_profit, current_profit_pips)
            
            # Check damage control
            if low <= dc_price:
                pnl = (dc_price - entry_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'DC', 'duration': 15}
            
            # Trailing stop
            if trailing_active and highest_profit - current_profit_pips >= 8:
                exit_price = entry_price + ((highest_profit - 8) * pip_value)
                pnl = (exit_price - entry_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'TRAIL', 'duration': 15}
        
        else:  # SELL
            # Check TP
            if low <= tp_price:
                pnl = (entry_price - tp_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'TP', 'duration': 15}
            
            # Check SL
            if high >= sl_price:
                pnl = (entry_price - sl_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'SL', 'duration': 15}
            
            # Trailing lock logic
            current_profit_pips = (entry_price - low) / pip_value
            if current_profit_pips >= trailing_activation:
                trailing_active = True
                highest_profit = max(highest_profit, current_profit_pips)
            
            # Check damage control
            if high >= dc_price:
                pnl = (entry_price - dc_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'DC', 'duration': 15}
            
            # Trailing stop
            if trailing_active and highest_profit - current_profit_pips >= 8:
                exit_price = entry_price - ((highest_profit - 8) * pip_value)
                pnl = (entry_price - exit_price) / pip_value * position_size
                return {'pnl': pnl, 'exit': 'TRAIL', 'duration': 15}
    
    # Timeout - close at market
    exit_price = future_candles[-1]['close']
    if direction == 'BUY':
        pnl = (exit_price - entry_price) / pip_value * position_size
    else:
        pnl = (entry_price - exit_price) / pip_value * position_size
    
    return {'pnl': pnl, 'exit': 'TIMEOUT', 'duration': 3000}

if __name__ == "__main__":
    try:
        results = run_optimization()
        print("\n✅ Optimization completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
