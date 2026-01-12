"""
Enhanced Backtest Analysis
Analyzes session performance and consecutive losses from backtest results
"""
import json
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_backtest_results(filename='backtest_results_20260112_014139.json'):
    """Analyze backtest with session and consecutive loss tracking"""
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("ENHANCED BACKTEST ANALYSIS - All 9 Pairs")
    print("="*80)
    
    # Extract summary and trades
    summary = data.get('summary', {})
    all_trades = data.get('trades', [])
    
    # Overall Stats
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {summary.get('total_trades', 0)}")
    print(f"   Win Rate: {summary.get('win_rate', 0):.1f}%")
    print(f"   ROI: {summary.get('roi', 0):+.1f}%")
    print(f"   Max Drawdown: {summary.get('max_drawdown', 0):.1f}% {'⚠️ HIT LIMIT' if summary.get('max_drawdown', 0) > 50 else ''}")
    print(f"   Final Balance: ${summary.get('final_balance', 0):,.2f}")
    print(f"   Stop Reason: {summary.get('stop_reason', 'Completed')}")
    
    # Session Analysis
    print(f"\n{'='*80}")
    print("SESSION PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    session_trades = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
    
    # Parse trades from all_trades data
    if all_trades:
        for trade in all_trades:
            trade_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
            hour = trade_time.hour
            
            # Determine session (UTC times)
            if 0 <= hour < 8:
                session = 'Asian'
            elif 8 <= hour < 16:
                session = 'London'
            else:
                session = 'New York'
            
            pnl = trade.get('pnl_usd', 0)
            session_trades[session]['pnl'] += pnl
            if pnl > 0:
                session_trades[session]['wins'] += 1
            else:
                session_trades[session]['losses'] += 1
        
        # Display session stats
        for session in ['Asian', 'London', 'New York']:
            if session in session_trades:
                stats = session_trades[session]
                total = stats['wins'] + stats['losses']
                wr = (stats['wins'] / total * 100) if total > 0 else 0
                avg_pnl = stats['pnl'] / total if total > 0 else 0
                
                print(f"\n{session.upper()} SESSION ({hour_range(session)}):")
                print(f"   Trades: {total}")
                print(f"   Wins: {stats['wins']} | Losses: {stats['losses']}")
                print(f"   Win Rate: {wr:.1f}%")
                print(f"   Total P&L: ${stats['pnl']:,.2f}")
                print(f"   Avg P&L/Trade: ${avg_pnl:.2f}")
                print(f"   Status: {'✅ Profitable' if stats['pnl'] > 0 else '❌ Unprofitable'}")
    else:
        print("\n⚠️ No trade data available for session analysis")
    
    # Consecutive Losses Analysis
    print(f"\n{'='*80}")
    print("CONSECUTIVE LOSSES ANALYSIS")
    print(f"{'='*80}")
    
    if all_trades:
        max_consecutive = 0
        current_consecutive = 0
        max_loss_amount = 0
        current_loss_amount = 0
        max_streak_start = None
        max_streak_end = None
        current_streak_start = None
        max_streak_symbol = None
        
        consecutive_by_symbol = defaultdict(lambda: {'max': 0, 'current': 0})
        
        for i, trade in enumerate(all_trades):
            pnl = trade.get('pnl_usd', 0)
            symbol = trade['symbol']
            
            if pnl < 0:
                # Loss
                current_consecutive += 1
                current_loss_amount += abs(pnl)
                
                if current_consecutive == 1:
                    current_streak_start = trade['entry_time']
                
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
                    max_loss_amount = current_loss_amount
                    max_streak_start = current_streak_start
                    # Calculate exit time from entry + duration
                    entry = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                    duration_min = trade.get('duration_minutes', 0)
                    exit_time = entry + timedelta(minutes=duration_min)
                    max_streak_end = exit_time.isoformat()
                    max_streak_symbol = symbol
                
                # Track per symbol
                consecutive_by_symbol[symbol]['current'] += 1
                if consecutive_by_symbol[symbol]['current'] > consecutive_by_symbol[symbol]['max']:
                    consecutive_by_symbol[symbol]['max'] = consecutive_by_symbol[symbol]['current']
            else:
                # Win - reset streak
                current_consecutive = 0
                current_loss_amount = 0
                current_streak_start = None
                
                # Reset per symbol
                consecutive_by_symbol[symbol]['current'] = 0
        
        print(f"\n🔴 OVERALL CONSECUTIVE LOSSES:")
        print(f"   Max Consecutive Losses: {max_consecutive} trades")
        print(f"   Total Loss Amount: ${max_loss_amount:,.2f}")
        if max_streak_start:
            print(f"   Occurred: {max_streak_start[:19]} to {max_streak_end[:19]}")
            print(f"   Last Symbol in Streak: {max_streak_symbol}")
        
        print(f"\n📉 PER-SYMBOL MAX CONSECUTIVE LOSSES:")
        for symbol in sorted(consecutive_by_symbol.keys()):
            max_consec = consecutive_by_symbol[symbol]['max']
            print(f"   {symbol}: {max_consec} consecutive losses")
    else:
        print("\n⚠️ No trade data available for consecutive loss analysis")
    
    # Symbol Rankings
    print(f"\n{'='*80}")
    print("SYMBOL RANKINGS (Best to Worst)")
    print(f"{'='*80}")
    
    # Calculate symbol performance from trades
    symbol_stats = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})
    
    if all_trades:
        for trade in all_trades:
            symbol = trade['symbol']
            pnl = trade.get('pnl_usd', 0)
            symbol_stats[symbol]['pnl'] += pnl
            symbol_stats[symbol]['trades'] += 1
            if pnl > 0:
                symbol_stats[symbol]['wins'] += 1
        
        symbols = []
        for symbol, stats in symbol_stats.items():
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            symbols.append((symbol, stats['pnl'], win_rate, stats['trades']))
        
        symbols.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Rank':<6}{'Symbol':<12}{'P&L':<15}{'Win Rate':<12}{'Trades':<10}{'Status'}")
        print("-" * 80)
        
        for i, (symbol, pnl, wr, trades) in enumerate(symbols, 1):
            status = '✅ KEEP' if pnl > 0 and wr > 55 else '❌ REMOVE'
            print(f"{i:<6}{symbol:<12}${pnl:>12,.2f}  {wr:>6.1f}%     {trades:<10}{status}")
    else:
        print("\n⚠️ No trade data available for symbol ranking")

def hour_range(session):
    """Return hour range for session"""
    ranges = {
        'Asian': '00:00-07:59 UTC',
        'London': '08:00-15:59 UTC',
        'New York': '16:00-23:59 UTC'
    }
    return ranges.get(session, '')

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else 'backtest_results_20260112_014139.json'
    try:
        analyze_backtest_results(filename)
    except FileNotFoundError:
        print(f"❌ Backtest results file '{filename}' not found. Run backtest first.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
