#!/usr/bin/env python3
"""
Quick 7-Day Backtest - Current Live Strategy
============================================
Tests your EXACT live strategy:
- NO DCA (only initial entry)
- Boost mode at -20% ROI (boost opposite side 1.5x)
- Trailing TP (activate at +20%, trail by 15%)
- TP=8% ROI, SL=90% ROI
- Max 5 boost cycles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trading_system", "Binance_Futures_Trading"))

from trading_system.Binance_Futures_Trading.backtest_enhanced_boost import EnhancedBoostBacktester
from trading_system.Binance_Futures_Trading.engine.binance_client import BinanceClient


def run_quick_test():
    """Quick 7-day test to show what could happen"""
    BACKTEST_DAYS = 7  # Short test
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Same as live

    print("="*80)
    print(f"QUICK {BACKTEST_DAYS} DAY BACKTEST - YOUR CURRENT LIVE STRATEGY")
    print("="*80)
    print("Strategy:")
    print("  • NO DCA (only initial entry with 10% of capital)")
    print("  • Boost mode triggers at -20% ROI (boost opposite side 1.5x)")
    print("  • Trailing TP: Activate at +20% ROI, trail by 15%")
    print("  • TP=8% ROI, SL=90% ROI")
    print("  • Max 5 boost cycles")
    print("  • 20x leverage, hedge mode (both sides)")
    print("="*80)

    all_results = []
    portfolio_capital = 300.0  # Total capital ($100 per symbol)
    allocation_per_symbol = 100.0

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        backtester = EnhancedBoostBacktester(symbol, start_balance=allocation_per_symbol)
        
        # Get real historical data
        df = backtester.get_historical_data(days=BACKTEST_DAYS, interval="1h")
        
        if df is not None and len(df) > 0:
            result = backtester.run_backtest(df)
            all_results.append(result)
        else:
            print(f"ERROR: Failed to get data for {symbol}")
            all_results.append({
                'symbol': symbol,
                'balance': allocation_per_symbol,
                'return_pct': 0,
                'liquidated': False,
                'total_trades': 0,
                'total_wins': 0,
                'total_losses': 0,
                'win_rate': 0,
                'boost_activations': 0,
                'half_close_count': 0,
            })

    # Portfolio summary
    print("\n" + "="*80)
    print(f"PORTFOLIO SUMMARY - {BACKTEST_DAYS} DAY SIMULATION")
    print("="*80)
    
    total_starting = len(all_results) * allocation_per_symbol
    total_ending = sum(r['balance'] for r in all_results)
    total_profit = total_ending - total_starting
    total_trades = sum(r['total_trades'] for r in all_results)
    total_wins = sum(r['total_wins'] for r in all_results)
    total_losses = sum(r['total_losses'] for r in all_results)
    total_boost_activations = sum(r['boost_activations'] for r in all_results)
    total_half_closes = sum(r['half_close_count'] for r in all_results)
    liquidations = sum(1 for r in all_results if r['liquidated'])

    print(f"Starting Capital:     ${total_starting:.2f}")
    print(f"Ending Capital:       ${total_ending:.2f}")
    print(f"Net Profit:           ${total_profit:+.2f}")
    print(f"Portfolio Return:     {total_profit/total_starting*100:+.1f}%")
    print(f"Daily Avg Return:     {total_profit/total_starting*100/BACKTEST_DAYS:+.2f}%")
    print(f"")
    print(f"Total Trades:         {total_trades}")
    print(f"Wins/Losses:          {total_wins}W / {total_losses}L")
    print(f"Win Rate:             {total_wins/total_trades*100 if total_trades > 0 else 0:.1f}%")
    print(f"Trades per Day:       {total_trades/BACKTEST_DAYS:.1f}")
    print(f"")
    print(f"Boost Activations:    {total_boost_activations}")
    print(f"Half-Close Cycles:    {total_half_closes}")
    print(f"Liquidations:         {liquidations}/{len(all_results)}")
    print("="*80)

    # Per-symbol details
    print("\nPER-SYMBOL PERFORMANCE:")
    print("-"*80)
    for r in all_results:
        status = "❌ LIQUIDATED" if r['liquidated'] else "✅ ACTIVE"
        print(f"{r['symbol']:10} | ${r['balance']:.2f} | Return: {r['return_pct']:+.1f}% | Trades: {r['total_trades']:3} ({r['win_rate']:.0f}% WR) | Boosts: {r['boost_activations']} | {status}")
    print("="*80)

    # What could happen scenarios
    print("\n📊 WHAT THIS MEANS FOR YOUR LIVE TRADING:")
    print("-"*80)
    
    if total_profit > 0:
        print(f"✅ POSITIVE: If market conditions match the last {BACKTEST_DAYS} days:")
        print(f"   • You could make ~${total_profit:.2f} profit ({total_profit/total_starting*100:.1f}% return)")
        print(f"   • Expected ~{total_trades} trades ({total_trades/BACKTEST_DAYS:.1f} per day)")
        print(f"   • Boost mode activated {total_boost_activations} times")
        print(f"   • Win rate: {total_wins/total_trades*100 if total_trades > 0 else 0:.1f}%")
    else:
        print(f"⚠️  WARNING: If market conditions match the last {BACKTEST_DAYS} days:")
        print(f"   • You could lose ~${abs(total_profit):.2f} ({abs(total_profit)/total_starting*100:.1f}% drawdown)")
        print(f"   • Had {total_losses} losing trades vs {total_wins} winners")
        print(f"   • Boost mode triggered {total_boost_activations} times (didn't save enough)")
    
    if liquidations > 0:
        print(f"\n🚨 DANGER: {liquidations} symbol(s) got LIQUIDATED in this simulation!")
        print(f"   • Review your position sizing and leverage settings")
        print(f"   • Consider reducing leverage or increasing capital")
    
    print("-"*80)
    print(f"⚡ REMEMBER: This is historical data. Future results may vary.")
    print(f"   • Market conditions change")
    print(f"   • Past performance ≠ future results")
    print(f"   • Always monitor your live positions")
    print("="*80)


if __name__ == "__main__":
    run_quick_test()
