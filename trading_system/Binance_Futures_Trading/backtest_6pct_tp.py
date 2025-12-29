"""
Backtest with 6% ROI TP (vs current 15%)
Tests if smaller TP can still cover fees while getting more wins
"""
import sys
sys.path.append('.')
from engine.backtest_engine import BinanceBacktestEngine
from config.trading_config import FUTURES_SYMBOLS, DCA_CONFIG
from datetime import datetime, timedelta

def main():
    # Save original config
    original_tp = DCA_CONFIG["take_profit_roi"]

    # Test with 6% ROI TP
    DCA_CONFIG["take_profit_roi"] = 0.06  # 6% ROI = 0.3% price move with 20x

    print("="*60)
    print("BACKTEST: 6% ROI TP (vs current 15%)")
    print("="*60)
    print(f"TP: 6% ROI = 0.3% price move")
    print(f"SL: {DCA_CONFIG['stop_loss_roi']*100:.0f}% ROI = {DCA_CONFIG['stop_loss_roi']/20*100:.1f}% price move")
    print(f"Symbols: {', '.join(FUTURES_SYMBOLS)}")
    print("="*60)

    # Run 7-day backtest
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    engine = BinanceBacktestEngine(symbols=FUTURES_SYMBOLS)
    result = engine.run_backtest(start_date=start_date, end_date=end_date)

    # Analyze results
    total_trades = len(result.trades)
    wins = sum(1 for t in result.trades if t.pnl > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    total_pnl = sum(t.pnl for t in result.trades)
    gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
    gross_loss = sum(t.pnl for t in result.trades if t.pnl < 0)

    # Fee analysis
    total_fees = sum(t.commission for t in result.trades)
    avg_win = gross_profit / wins if wins > 0 else 0
    avg_loss = abs(gross_loss) / losses if losses > 0 else 0

    print("\n" + "="*60)
    print("RESULTS: 6% ROI TP")
    print("="*60)
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"\nGross Profit: ${gross_profit:+,.2f}")
    print(f"Gross Loss: ${gross_loss:,.2f}")
    print(f"Total Fees: ${total_fees:,.2f}")
    print(f"Net P&L: ${total_pnl:+,.2f}")
    print(f"\nAvg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {abs(gross_profit/gross_loss):.2f}" if gross_loss != 0 else "N/A")

    # Final balance
    final_balance = result.final_balance
    roi = ((final_balance - 10000) / 10000) * 100
    print(f"\nStarting Balance: $10,000")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Total ROI: {roi:+.1f}%")

    # Fee coverage analysis
    print("\n" + "="*60)
    print("FEE COVERAGE ANALYSIS")
    print("="*60)
    print(f"With 6% ROI TP and 20x leverage:")
    print(f"  - Price move needed: 0.3%")
    print(f"  - Fees per round trip: ~1.6% ROI")
    print(f"  - Net profit per win: ~4.4% ROI (6% - 1.6%)")
    print(f"  - Avg actual win: ${avg_win:.2f}")

    if avg_win > 0:
        fee_pct_of_win = (total_fees / wins / avg_win * 100) if wins > 0 else 0
        print(f"  - Fees as % of avg win: {fee_pct_of_win:.1f}%")

    # Restore original config
    DCA_CONFIG["take_profit_roi"] = original_tp

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print("6% TP: Smaller moves, more frequent wins, but fees eat more %")
    print("15% TP: Larger moves, fewer wins, but fees are smaller % of profit")
    print("="*60)

if __name__ == "__main__":
    main()
