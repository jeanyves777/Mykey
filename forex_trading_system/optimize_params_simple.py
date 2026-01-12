"""
Quick Parameter Optimization Script
Tests key parameter combinations using existing backtest infrastructure
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from itertools import product

# Parameter grid to test
PARAM_GRID = {
    'min_confluence_score': [4, 5, 6, 7],
    'damage_control_pips': [-8, -10, -12, -14],
}

def run_backtest_with_params(confluence, damage_control):
    """
    Run backtest by temporarily modifying config file
    """
    config_file = '/workspaces/Mykey/forex_trading_system/config/trading_config.py'
    
    # Read current config
    with open(config_file, 'r') as f:
        original_content = f.read()
    
    try:
        # Modify config with test parameters
        modified_content = original_content
        
        # Update min_confluence_score for all symbols
        modified_content = modified_content.replace(
            '"min_confluence_score": 5,',
            f'"min_confluence_score": {confluence},'
        )
        
        # Update damage_control_pips
        modified_content = modified_content.replace(
            'damage_control_pips = -12  ',
            f'damage_control_pips = {damage_control}  '
        )
        
        # Write temporary config
        with open(config_file, 'w') as f:
            f.write(modified_content)
        
        # Run backtest
        print(f"\n{'='*70}")
        print(f"Testing: Confluence {confluence}/8 | Damage Control {damage_control}p")
        print(f"{'='*70}")
        
        result = subprocess.run(
            ['python', 'backtest_oanda_scalping.py'],
            cwd='/workspaces/Mykey/forex_trading_system',
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse results from output
        output = result.stdout
        
        # Extract key metrics
        metrics = extract_metrics(output)
        metrics['params'] = {
            'min_confluence_score': confluence,
            'damage_control_pips': damage_control,
        }
        
        return metrics
        
    finally:
        # Restore original config
        with open(config_file, 'w') as f:
            f.write(original_content)

def extract_metrics(output):
    """Extract metrics from backtest output"""
    metrics = {
        'total_trades': 0,
        'win_rate': 0.0,
        'roi': 0.0,
        'final_balance': 0.0,
        'max_drawdown': 0.0,
        'total_pnl': 0.0,
    }
    
    try:
        lines = output.split('\n')
        for line in lines:
            if 'Total Trades:' in line:
                metrics['total_trades'] = int(line.split(':')[1].strip())
            elif 'Wins:' in line and '(' in line:
                # Extract win rate percentage
                pct = line.split('(')[1].split('%')[0]
                metrics['win_rate'] = float(pct)
            elif 'ROI:' in line:
                roi = line.split(':')[1].strip().replace('%', '').replace('+', '')
                metrics['roi'] = float(roi)
            elif 'Final Balance:' in line:
                balance = line.split('$')[1].strip().replace(',', '')
                metrics['final_balance'] = float(balance)
            elif 'Max Drawdown:' in line:
                dd = line.split(':')[1].strip().replace('%', '')
                metrics['max_drawdown'] = float(dd)
            elif 'Total P&L:' in line and '$' in line:
                pnl = line.split('$')[1].strip().replace(',', '').replace('+', '')
                metrics['total_pnl'] = float(pnl)
    except:
        pass
    
    return metrics

def main():
    print("=" * 70)
    print("PARAMETER OPTIMIZATION - All 9 Forex Pairs")
    print("=" * 70)
    print(f"Testing {len(PARAM_GRID['min_confluence_score']) * len(PARAM_GRID['damage_control_pips'])} combinations")
    print("=" * 70)
    
    results = []
    combination_num = 0
    total_combinations = len(PARAM_GRID['min_confluence_score']) * len(PARAM_GRID['damage_control_pips'])
    
    for confluence, damage_control in product(
        PARAM_GRID['min_confluence_score'],
        PARAM_GRID['damage_control_pips']
    ):
        combination_num += 1
        print(f"\n\n{'#'*70}")
        print(f"COMBINATION {combination_num}/{total_combinations}")
        print(f"{'#'*70}")
        
        try:
            metrics = run_backtest_with_params(confluence, damage_control)
            results.append(metrics)
            
            print(f"\n📊 RESULT:")
            print(f"   Trades: {metrics['total_trades']}")
            print(f"   Win Rate: {metrics['win_rate']:.1f}%")
            print(f"   ROI: {metrics['roi']:.1f}%")
            print(f"   Max DD: {metrics['max_drawdown']:.1f}%")
            print(f"   Final Balance: ${metrics['final_balance']:,.2f}")
            
        except subprocess.TimeoutExpired:
            print("⚠️ TIMEOUT - Skipping this combination")
            continue
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            continue
    
    # Analyze results
    print("\n\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    # Sort by ROI
    results_sorted = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print("\n🏆 TOP 5 CONFIGURATIONS BY ROI:")
    print("-" * 70)
    for i, result in enumerate(results_sorted[:5], 1):
        params = result['params']
        print(f"\n#{i}: ROI {result['roi']:.1f}%")
        print(f"   Confluence: {params['min_confluence_score']}/8")
        print(f"   Damage Control: {params['damage_control_pips']} pips")
        print(f"   Trades: {result['total_trades']} | WR: {result['win_rate']:.1f}%")
        print(f"   Max DD: {result['max_drawdown']:.1f}% | Balance: ${result['final_balance']:,.2f}")
    
    # Best by win rate
    best_wr = max(results, key=lambda x: x['win_rate'])
    print(f"\n\n✅ BEST WIN RATE: {best_wr['win_rate']:.1f}%")
    print(f"   Params: {best_wr['params']}")
    print(f"   ROI: {best_wr['roi']:.1f}% | Trades: {best_wr['total_trades']}")
    
    # Best by drawdown
    best_dd = min(results, key=lambda x: x['max_drawdown'])
    print(f"\n🛡️ BEST DRAWDOWN: {best_dd['max_drawdown']:.1f}%")
    print(f"   Params: {best_dd['params']}")
    print(f"   ROI: {best_dd['roi']:.1f}% | WR: {best_dd['win_rate']:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n💾 Results saved to: {filename}")
    print("\n✅ Optimization complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
