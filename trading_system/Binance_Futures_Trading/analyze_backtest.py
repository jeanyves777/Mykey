#!/usr/bin/env python3
"""Comprehensive backtest analysis with all trade details"""

import pandas as pd
import glob
import os

os.chdir('/root/thevolumeainative/trading_system/Binance_Futures_Trading')

# Find all trade journals
journals = glob.glob('trade_journal_*.csv')
print('='*120)
print('COMPREHENSIVE BACKTEST ANALYSIS - ALL SYMBOLS')
print('='*120)

all_trades = []

for journal in sorted(journals):
    symbol = journal.split('_')[2]
    df = pd.read_csv(journal)
    df['symbol'] = symbol
    all_trades.append(df)

    print(f'\n{"="*120}')
    print(f'{symbol} - DETAILED TRADE ANALYSIS')
    print('='*120)

    # Basic stats
    total_trades = len(df)
    wins = len(df[df['pnl'] > 0])
    losses = len(df[df['pnl'] <= 0])
    win_rate = wins/total_trades*100 if total_trades > 0 else 0

    total_profit = df[df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())
    net_pnl = df['pnl'].sum()

    # Balance tracking
    start_balance = 100.0
    final_balance = df['balance_after'].iloc[-1] if len(df) > 0 else start_balance

    # Calculate equity curve and drawdown
    df['equity'] = start_balance + df['cumulative_pnl']
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100
    max_dd = df['drawdown'].max()
    max_dd_row = df.loc[df['drawdown'].idxmax()]

    # Exit types breakdown
    exit_types = df['exit_type'].value_counts()

    print(f'\n>>> PERFORMANCE SUMMARY:')
    print(f'    Starting Balance:    $100.00')
    print(f'    Final Balance:       ${final_balance:.2f}')
    print(f'    Net P&L:             ${net_pnl:+.2f}')
    print(f'    Return:              {(final_balance-100)/100*100:+.1f}%')

    print(f'\n>>> TRADE STATISTICS:')
    print(f'    Total Trades:        {total_trades}')
    print(f'    Winning Trades:      {wins} ({win_rate:.1f}%)')
    print(f'    Losing Trades:       {losses} ({100-win_rate:.1f}%)')
    print(f'    Total Profit:        ${total_profit:+.2f}')
    print(f'    Total Loss:          ${total_loss:.2f}')
    if total_loss > 0:
        print(f'    Profit Factor:       {total_profit/total_loss:.2f}x')

    print(f'\n>>> DRAWDOWN ANALYSIS:')
    print(f'    Max Drawdown:        {max_dd:.1f}%')
    print(f'    Max DD ($):          ${max_dd/100*start_balance:.2f}')
    print(f'    Max DD Date:         {max_dd_row["timestamp"]}')
    print(f'    Equity at Max DD:    ${max_dd_row["equity"]:.2f}')

    print(f'\n>>> EXIT TYPES BREAKDOWN:')
    for exit_type, count in exit_types.items():
        pnl_for_type = df[df['exit_type'] == exit_type]['pnl'].sum()
        print(f'    {exit_type:<20} {count:>5} trades | P&L: ${pnl_for_type:+.2f}')

    # DCA Level breakdown
    print(f'\n>>> TRADES BY DCA LEVEL:')
    for dca in sorted(df['dca_level'].unique()):
        dca_trades = df[df['dca_level'] == dca]
        dca_pnl = dca_trades['pnl'].sum()
        dca_wins = len(dca_trades[dca_trades['pnl'] > 0])
        dca_losses = len(dca_trades[dca_trades['pnl'] <= 0])
        avg_pnl = dca_trades['pnl'].mean()
        print(f'    DCA {int(dca)}: {len(dca_trades):>4} trades | Wins: {dca_wins:>3} | Losses: {dca_losses:>3} | P&L: ${dca_pnl:+.2f} | Avg: ${avg_pnl:+.2f}')

    # Liquidations
    liqs = df[df['exit_type'] == 'LIQUIDATION']
    if len(liqs) > 0:
        print(f'\n>>> LIQUIDATION EVENTS ({len(liqs)} total):')
        for _, liq in liqs.iterrows():
            print(f'    {liq["timestamp"]} | {liq["side"]} | DCA {int(liq["dca_level"])} | Lost: ${abs(liq["pnl"]):.2f} | Balance: ${liq["balance_after"]:.2f}')

    # SL hits
    sls = df[df['exit_type'] == 'SL']
    if len(sls) > 0:
        print(f'\n>>> STOP LOSS HITS ({len(sls)} total):')
        for _, sl in sls.iterrows():
            print(f'    {sl["timestamp"]} | {sl["side"]} | DCA {int(sl["dca_level"])} | Lost: ${abs(sl["pnl"]):.2f} | Balance: ${sl["balance_after"]:.2f}')

    # Worst trades
    print(f'\n>>> TOP 5 WORST TRADES:')
    worst = df.nsmallest(5, 'pnl')
    for _, t in worst.iterrows():
        print(f'    {t["timestamp"]} | {t["side"]} | {t["exit_type"]} | DCA {int(t["dca_level"])} | P&L: ${t["pnl"]:+.2f}')

    # Best trades
    print(f'\n>>> TOP 5 BEST TRADES:')
    best = df.nlargest(5, 'pnl')
    for _, t in best.iterrows():
        print(f'    {t["timestamp"]} | {t["side"]} | {t["exit_type"]} | DCA {int(t["dca_level"])} | P&L: ${t["pnl"]:+.2f}')

# Combined analysis
print(f'\n{"="*120}')
print('PORTFOLIO COMBINED ANALYSIS')
print('='*120)

combined = pd.concat(all_trades, ignore_index=True)

total_trades = len(combined)
wins = len(combined[combined['pnl'] > 0])
losses = len(combined[combined['pnl'] <= 0])

total_profit = combined[combined['pnl'] > 0]['pnl'].sum()
total_loss = abs(combined[combined['pnl'] <= 0]['pnl'].sum())

liqs = combined[combined['exit_type'] == 'LIQUIDATION']
sl_trades = combined[combined['exit_type'] == 'SL']

print(f'\n>>> PORTFOLIO TOTALS:')
print(f'    Starting Capital:    $500.00 (5 x $100)')
print(f'    Total Trades:        {total_trades}')
print(f'    Winning Trades:      {wins} ({wins/total_trades*100:.1f}%)')
print(f'    Losing Trades:       {losses} ({losses/total_trades*100:.1f}%)')
print(f'    Total Profit:        ${total_profit:+.2f}')
print(f'    Total Loss:          ${total_loss:.2f}')
print(f'    Net P&L:             ${total_profit - total_loss:+.2f}')
print(f'    Profit Factor:       {total_profit/total_loss:.2f}x')

print(f'\n>>> CRITICAL LOSSES:')
print(f'    Liquidations:        {len(liqs)} trades | Lost: ${abs(liqs["pnl"].sum()):.2f}')
print(f'    Stop Losses:         {len(sl_trades)} trades | Lost: ${abs(sl_trades["pnl"].sum()):.2f}')

print(f'\n>>> LIQUIDATIONS BY SYMBOL:')
for sym in combined['symbol'].unique():
    sym_liqs = liqs[liqs['symbol'] == sym]
    if len(sym_liqs) > 0:
        long_liqs = len(sym_liqs[sym_liqs['side'] == 'LONG'])
        short_liqs = len(sym_liqs[sym_liqs['side'] == 'SHORT'])
        print(f'    {sym}: {len(sym_liqs)} liquidations (L:{long_liqs} S:{short_liqs}) | Lost: ${abs(sym_liqs["pnl"].sum()):.2f}')

print(f'\n>>> ALL LIQUIDATION EVENTS (Chronological):')
print(f'    {"Timestamp":<22} {"Symbol":<10} {"Side":<6} {"DCA":<5} {"Lost":>12} {"Balance":>12}')
print(f'    {"-"*75}')
for _, liq in liqs.sort_values('timestamp').iterrows():
    print(f'    {str(liq["timestamp"]):<22} {liq["symbol"]:<10} {liq["side"]:<6} {int(liq["dca_level"]):<5} ${abs(liq["pnl"]):>10.2f} ${liq["balance_after"]:>10.2f}')

# Summary by DCA level across all symbols
print(f'\n>>> P&L BY DCA LEVEL (ALL SYMBOLS):')
for dca in sorted(combined['dca_level'].unique()):
    dca_trades = combined[combined['dca_level'] == dca]
    dca_pnl = dca_trades['pnl'].sum()
    dca_wins = len(dca_trades[dca_trades['pnl'] > 0])
    dca_losses = len(dca_trades[dca_trades['pnl'] <= 0])
    dca_liqs = len(dca_trades[dca_trades['exit_type'] == 'LIQUIDATION'])
    print(f'    DCA {int(dca)}: {len(dca_trades):>5} trades | W:{dca_wins:>4} L:{dca_losses:>3} LIQ:{dca_liqs:>2} | P&L: ${dca_pnl:+.2f}')

print(f'\n{"="*120}')
