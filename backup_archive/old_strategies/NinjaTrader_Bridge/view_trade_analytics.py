"""
View Trade Analytics for NinjaTrader Bridge

Shows performance statistics and trade history from the forex trade logger
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

from trading_system.analytics.forex_trade_logger import ForexTradeLogger
from datetime import datetime, timedelta
import pandas as pd


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def main():
    # Load trade logger
    logger = ForexTradeLogger()

    print_section("FOREX FUTURES TRADE ANALYTICS")

    # Overall statistics
    all_trades = list(logger.trades.values())
    open_trades = logger.get_open_trades()
    closed_trades = logger.get_closed_trades()

    print(f"Total Trades: {len(all_trades)}")
    print(f"  Open: {len(open_trades)}")
    print(f"  Closed: {len(closed_trades)}")

    if not closed_trades:
        print("\nNo closed trades yet!")
        return

    # P&L Statistics
    print_section("P&L STATISTICS")

    winning_trades = [t for t in closed_trades if t.net_pnl_usd > 0]
    losing_trades = [t for t in closed_trades if t.net_pnl_usd < 0]
    breakeven_trades = [t for t in closed_trades if t.net_pnl_usd == 0]

    total_pnl = sum(t.net_pnl_usd for t in closed_trades)
    avg_win = sum(t.net_pnl_usd for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.net_pnl_usd for t in losing_trades) / len(losing_trades) if losing_trades else 0

    print(f"Total P&L: ${total_pnl:+,.2f}")
    print(f"Average P&L: ${total_pnl / len(closed_trades):+,.2f}")
    print()
    print(f"Winners: {len(winning_trades)} ({len(winning_trades)/len(closed_trades)*100:.1f}%)")
    print(f"  Average Win: ${avg_win:+,.2f}")
    print(f"  Best Win: ${max((t.net_pnl_usd for t in winning_trades), default=0):+,.2f}")
    print()
    print(f"Losers: {len(losing_trades)} ({len(losing_trades)/len(closed_trades)*100:.1f}%)")
    print(f"  Average Loss: ${avg_loss:+,.2f}")
    print(f"  Worst Loss: ${min((t.net_pnl_usd for t in losing_trades), default=0):+,.2f}")
    print()
    print(f"Breakeven: {len(breakeven_trades)}")

    if winning_trades and losing_trades:
        profit_factor = abs(sum(t.net_pnl_usd for t in winning_trades) / sum(t.net_pnl_usd for t in losing_trades))
        print(f"\nProfit Factor: {profit_factor:.2f}")

    # Per Symbol Statistics
    print_section("PER SYMBOL STATISTICS")

    symbols = set(t.symbol for t in closed_trades)
    for symbol in sorted(symbols):
        symbol_trades = [t for t in closed_trades if t.symbol == symbol]
        symbol_pnl = sum(t.net_pnl_usd for t in symbol_trades)
        symbol_winners = len([t for t in symbol_trades if t.net_pnl_usd > 0])
        symbol_win_rate = symbol_winners / len(symbol_trades) * 100 if symbol_trades else 0

        print(f"{symbol}:")
        print(f"  Trades: {len(symbol_trades)}")
        print(f"  P&L: ${symbol_pnl:+,.2f}")
        print(f"  Win Rate: {symbol_win_rate:.1f}%")
        print()

    # Exit Reasons
    print_section("EXIT REASONS")

    exit_reasons = {}
    for trade in closed_trades:
        reason = trade.exit_reason or "UNKNOWN"
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(trade)

    for reason, trades in sorted(exit_reasons.items()):
        reason_pnl = sum(t.net_pnl_usd for t in trades)
        print(f"{reason}: {len(trades)} trades, ${reason_pnl:+,.2f}")

    # Daily Performance
    print_section("DAILY PERFORMANCE (Last 7 Days)")

    today = datetime.now().date()
    for i in range(7):
        day = today - timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')
        stats = logger.get_daily_stats(day_str)

        if stats['total_trades'] > 0:
            print(f"{day_str}:")
            print(f"  Trades: {stats['closed_trades']} (Open: {stats['open_trades']})")
            print(f"  P&L: ${stats['total_pnl']:+,.2f}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Best: ${stats['best_trade']:+,.2f}, Worst: ${stats['worst_trade']:+,.2f}")
            print()

    # Recent Trades
    print_section("LAST 10 TRADES")

    recent_trades = sorted(closed_trades, key=lambda t: t.exit_time, reverse=True)[:10]

    for trade in recent_trades:
        pnl_str = f"${trade.net_pnl_usd:+,.2f}"
        print(f"{trade.trade_id}")
        print(f"  {trade.symbol} {trade.side} @ {trade.entry_price:.5f} → {trade.exit_price:.5f}")
        print(f"  Exit: {trade.exit_reason}, P&L: {pnl_str} ({trade.gross_pnl_ticks:+.1f} ticks)")
        print(f"  Hold: {trade.hold_minutes:.0f} min")
        print()

    # Open Trades
    if open_trades:
        print_section("OPEN TRADES")

        for trade in open_trades:
            print(f"{trade.trade_id}")
            print(f"  {trade.symbol} {trade.side} @ {trade.entry_price:.5f}")
            print(f"  TP: {trade.take_profit:.5f}, SL: {trade.stop_loss:.5f}")
            print(f"  Entry: {trade.entry_time}")
            print()

    # Export Options
    print_section("EXPORT OPTIONS")

    print("1. Export to CSV:")
    print(f"   {logger.csv_file}")
    print()
    print("2. Export to Excel:")
    print("   Run: logger.export_to_excel()")
    print()
    print("To export, use:")
    print("   from trading_system.analytics.forex_trade_logger import ForexTradeLogger")
    print("   logger = ForexTradeLogger()")
    print("   logger.export_to_excel()")


if __name__ == "__main__":
    main()
