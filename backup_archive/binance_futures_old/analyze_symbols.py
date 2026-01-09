"""
Analyze Symbol Performance from Backtest
Includes: Funding fees, Liquidations, and per-symbol breakdown
"""
import sys
sys.path.append('.')
from engine.backtest_engine import BinanceBacktestEngine
from config.trading_config import FUTURES_SYMBOLS

def main():
    # Run backtest
    print("Running 7-day backtest with funding fees & liquidation tracking...")
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    engine = BinanceBacktestEngine(symbols=FUTURES_SYMBOLS)
    result = engine.run_backtest(start_date=start_date, end_date=end_date)

    # Analyze by symbol
    symbol_stats = {}
    for trade in result.trades:
        sym = trade.symbol
        if sym not in symbol_stats:
            symbol_stats[sym] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'liquidations': 0}
        symbol_stats[sym]['trades'] += 1
        symbol_stats[sym]['pnl'] += trade.pnl
        if trade.pnl > 0:
            symbol_stats[sym]['wins'] += 1
        if trade.exit_reason == 'LIQUIDATION':
            symbol_stats[sym]['liquidations'] += 1

    print()
    print('='*75)
    print('SYMBOL PERFORMANCE RANKING (7-day backtest with funding & liquidation)')
    print('='*75)
    print(f"{'Symbol':<12} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'Liqs':>5} {'Total PnL':>12}")
    print('-'*75)

    # Sort by PnL
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)

    for sym, stats in sorted_symbols:
        trades = stats['trades']
        wins = stats['wins']
        win_rate = (wins/trades*100) if trades > 0 else 0
        total_pnl = stats['pnl']
        liqs = stats['liquidations']

        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
        print(f'{sym:<12} {trades:>7} {wins:>6} {win_rate:>7.1f}% {liqs:>5} {pnl_str:>12}')

    # Symbols with no trades
    no_trades = [s for s in FUTURES_SYMBOLS if s not in symbol_stats]
    if no_trades:
        print('-'*75)
        print(f"No trades: {', '.join(no_trades)}")

    print('='*75)

    # Recommendations
    print()
    print('TOP 3 PERFORMERS (Keep these):')
    for i, (sym, stats) in enumerate(sorted_symbols[:3], 1):
        wr = (stats['wins']/stats['trades']*100) if stats['trades'] > 0 else 0
        print(f"  {i}. {sym}: ${stats['pnl']:+.2f} | {stats['wins']}/{stats['trades']} wins ({wr:.0f}%)")

    print()
    print('BOTTOM 3 (Consider removing):')
    for sym, stats in sorted_symbols[-3:]:
        wr = (stats['wins']/stats['trades']*100) if stats['trades'] > 0 else 0
        print(f"  - {sym}: ${stats['pnl']:+.2f} | {stats['liquidations']} liquidations")

if __name__ == "__main__":
    main()
