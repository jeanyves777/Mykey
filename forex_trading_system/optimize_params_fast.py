"""
Fast Parameter Optimization
Tests parameter combinations on shorter period (7 days)
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from itertools import product

# Parameter grid - focused on key parameters
PARAM_GRID = {
    'min_confluence_score': [4, 5, 6],
    'damage_control_pips': [-10, -12, -14],
}

def run_quick_backtest(confluence, damage_control):
    """Run 7-day backtest with parameters"""
    config_file = '/workspaces/Mykey/forex_trading_system/config/trading_config.py'
    
    # Read current config
    with open(config_file, 'r') as f:
        original_content = f.read()
    
    try:
        # Modify config
        modified_content = original_content
        modified_content = modified_content.replace(
            '"min_confluence_score": 5,',
            f'"min_confluence_score": {confluence},'
        )
        modified_content = modified_content.replace(
            'damage_control_pips = -12  ',
            f'damage_control_pips = {damage_control}  '
        )
        
        # Write temp config
        with open(config_file, 'w') as f:
            f.write(modified_content)
        
        print(f"⚙️  Testing: Confluence {confluence}/8 | DC {damage_control}p...", end=' ', flush=True)
        
        # Run backtest (output suppressed for speed)
        result = subprocess.run(
            ['python', 'backtest_oanda_scalping.py'],
            cwd='/workspaces/Mykey/forex_trading_system',
            capture_output=True,
            text=True,
            timeout=180  # 3 min timeout
        )
        
        output = result.stdout
        metrics = extract_metrics(output)
        metrics['params'] = {
            'confluence': confluence,
            'damage_control': damage_control,
        }
        
        print(f"✅ ROI: {metrics['roi']:+.1f}% | WR: {metrics['win_rate']:.1f}% | DD: {metrics['max_drawdown']:.1f}% | Trades: {metrics['total_trades']}")
        
        return metrics
        
    finally:
        # Restore config
        with open(config_file, 'w') as f:
            f.write(original_content)

def extract_metrics(output):
    """Extract metrics from output"""
    metrics = {
        'total_trades': 0,
        'win_rate': 0.0,
        'roi': 0.0,
        'final_balance': 0.0,
        'max_drawdown': 0.0,
    }
    
    try:
        for line in output.split('\n'):
            if 'Total Trades:' in line:
                metrics['total_trades'] = int(line.split(':')[1].strip())
            elif 'Wins:' in line and '(' in line:
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
    except:
        pass
    
    return metrics

def main():
    print("\n" + "=" * 80)
    print("🔬 PARAMETER OPTIMIZATION - All 9 Forex Pairs (30-day backtest)")
    print("=" * 80)
    total = len(PARAM_GRID['min_confluence_score']) * len(PARAM_GRID['damage_control_pips'])
    print(f"Testing {total} parameter combinations\n")
    
    results = []
    combo_num = 0
    
    for confluence, damage_control in product(
        PARAM_GRID['min_confluence_score'],
        PARAM_GRID['damage_control_pips']
    ):
        combo_num += 1
        print(f"[{combo_num}/{total}] ", end='')
        
        try:
            metrics = run_quick_backtest(confluence, damage_control)
            results.append(metrics)
        except subprocess.TimeoutExpired:
            print("⚠️ TIMEOUT")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    # Results Analysis
    print("\n" + "=" * 80)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Sort by ROI
    results_sorted = sorted(results, key=lambda x: x['roi'], reverse=True)
    
    print(f"\n🏆 TOP {min(5, len(results_sorted))} CONFIGURATIONS:")
    print("-" * 80)
    for i, r in enumerate(results_sorted[:5], 1):
        p = r['params']
        print(f"\n#{i}  ROI: {r['roi']:+7.1f}%  |  WR: {r['win_rate']:5.1f}%  |  DD: {r['max_drawdown']:5.1f}%  |  Trades: {r['total_trades']:4d}")
        print(f"     Confluence: {p['confluence']}/8  |  Damage Control: {p['damage_control']} pips")
    
    # Best by metrics
    best_wr = max(results, key=lambda x: x['win_rate'])
    best_dd = min(results, key=lambda x: x['max_drawdown'])
    
    print(f"\n" + "=" * 80)
    print("📈 BEST BY METRICS")
    print("=" * 80)
    
    print(f"\n✅ Best Win Rate: {best_wr['win_rate']:.1f}%")
    print(f"   Confluence: {best_wr['params']['confluence']}/8 | DC: {best_wr['params']['damage_control']}p")
    print(f"   ROI: {best_wr['roi']:+.1f}% | DD: {best_wr['max_drawdown']:.1f}%")
    
    print(f"\n🛡️ Best Drawdown: {best_dd['max_drawdown']:.1f}%")
    print(f"   Confluence: {best_dd['params']['confluence']}/8 | DC: {best_dd['params']['damage_control']}p")
    print(f"   ROI: {best_dd['roi']:+.1f}% | WR: {best_dd['win_rate']:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("\n✅ Optimization complete!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
