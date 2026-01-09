#!/usr/bin/env python3
"""
Quick 1-Week Backtest - All 17 Pairs
=====================================
$100 capital, $5 fixed per trade
Uses optimized settings from config
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add trading system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trading_system', 'Crypto_Margin_Trading'))

from strategies.crypto_margin_strategy import calculate_indicators
from config.crypto_paper_config import PAIR_SETTINGS, TRADING_PAIRS

# Configuration
INITIAL_BALANCE = 100.0  # $100 starting capital
FIXED_TRADE_SIZE = 5.0   # $5 per trade
ONE_WEEK_CANDLES = 7 * 24 * 60  # 1 week = 10,080 minutes


def generate_signals(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """Generate signals for a specific strategy"""
    n = len(df)
    signals = np.zeros(n)

    rsi = df['rsi'].values
    macd = df['macd'].values
    macd_signal = df['macd_signal'].values
    close = df['close'].values
    ema21 = df['ema21'].values
    ema50 = df['ema50'].values

    rsi_prev = np.roll(rsi, 1)
    macd_prev = np.roll(macd, 1)
    macd_signal_prev = np.roll(macd_signal, 1)

    if strategy == 'RSI_REVERSAL':
        buy = (rsi_prev < 35) & (rsi >= 35)
        sell = (rsi_prev > 65) & (rsi <= 65)
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    elif strategy == 'RSI_EXTREME':
        signals = np.where(rsi < 25, 1, np.where(rsi > 75, -1, 0))

    elif strategy == 'MACD_CROSS':
        buy = (macd_prev < macd_signal_prev) & (macd > macd_signal) & (macd < 0)
        sell = (macd_prev > macd_signal_prev) & (macd < macd_signal) & (macd > 0)
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    elif strategy == 'EMA_PULLBACK':
        uptrend = ema21 > ema50
        downtrend = ema21 < ema50
        price_near_ema = np.abs(close - ema21) / ema21 < 0.002
        buy = uptrend & price_near_ema & (rsi < 50)
        sell = downtrend & price_near_ema & (rsi > 50)
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    elif strategy == 'RSI_MACD_COMBO':
        buy = (rsi < 35) & (macd > macd_signal)
        sell = (rsi > 65) & (macd < macd_signal)
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    elif strategy == 'TRIPLE_CONFIRM':
        uptrend = ema21 > ema50
        downtrend = ema21 < ema50
        buy = (rsi < 40) & (macd > macd_signal) & uptrend
        sell = (rsi > 60) & (macd < macd_signal) & downtrend
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    # Set first 50 rows to 0 (indicator warmup)
    signals[:50] = 0
    return signals


def run_backtest_fixed_size(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signals: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    trade_size: float = 5.0,
) -> dict:
    """
    Run backtest with fixed $ per trade
    """
    n = len(close)
    trades = []

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    direction = 0
    take_profit = 0.0
    stop_loss = 0.0

    for i in range(50, n):
        current_price = close[i]

        if in_position:
            h = high[i]
            l = low[i]
            exit_price = None
            exit_reason = None

            if direction == 1:  # Long
                if l <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'SL'
                elif h >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TP'
            else:  # Short
                if h >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'SL'
                elif l <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TP'

            if exit_price is not None:
                # Fixed $ size - calculate P&L
                if direction == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Fixed $5 trade, P&L is proportional
                pnl = trade_size * (pnl_pct / 100)

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                })

                in_position = False

        else:
            sig = signals[i]
            if sig != 0:
                direction = int(sig)
                entry_price = current_price
                entry_idx = i

                if direction == 1:
                    take_profit = current_price * (1 + tp_pct / 100)
                    stop_loss = current_price * (1 - sl_pct / 100)
                else:
                    take_profit = current_price * (1 - tp_pct / 100)
                    stop_loss = current_price * (1 + sl_pct / 100)

                in_position = True

    # Calculate metrics
    total_trades = len(trades)
    if total_trades == 0:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pnl': 0, 'avg_trade': 0, 'best_trade': 0, 'worst_trade': 0
        }

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    total_pnl = sum(t['pnl'] for t in trades)
    avg_trade = total_pnl / total_trades
    best_trade = max(t['pnl'] for t in trades)
    worst_trade = min(t['pnl'] for t in trades)

    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0) or 0.001
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) or 0.001
    profit_factor = gross_profit / gross_loss

    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_trade': round(avg_trade, 3),
        'best_trade': round(best_trade, 3),
        'worst_trade': round(worst_trade, 3),
        'profit_factor': round(profit_factor, 2),
        'trade_list': trades,
    }


def main():
    print("\n" + "=" * 80)
    print("QUICK 1-WEEK BACKTEST - ALL 17 PAIRS")
    print("=" * 80)
    print(f"Capital: ${INITIAL_BALANCE:.2f} | Trade Size: ${FIXED_TRADE_SIZE:.2f}")
    print(f"Period: Last 7 days (~{ONE_WEEK_CANDLES:,} 1-minute candles)")
    print("=" * 80)

    data_dir = Path(__file__).parent / 'trading_system' / 'Crypto_Margin_Trading' / 'Crypto_Data_from_Binance'

    results = []
    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    total_losses = 0

    print(f"\n{'PAIR':<12} {'STRATEGY':<16} {'TP%':<5} {'SL%':<5} {'TRADES':<7} {'W/L':<8} {'WR%':<6} {'PnL':<10} {'PF':<6}")
    print("-" * 85)

    for pair in TRADING_PAIRS:
        settings = PAIR_SETTINGS[pair]
        strategy = settings['strategy']
        tp_pct = settings['tp_pct']
        sl_pct = settings['sl_pct']

        # Load data
        filepath = data_dir / f"{pair}_1m.csv"
        if not filepath.exists():
            print(f"{pair:<12} Data not found - skipping")
            continue

        df = pd.read_csv(filepath)

        # Take only last week of data
        if len(df) > ONE_WEEK_CANDLES:
            df = df.tail(ONE_WEEK_CANDLES).reset_index(drop=True)

        # Calculate indicators
        df = calculate_indicators(df)

        # Generate signals
        signals = generate_signals(df, strategy)

        # Run backtest
        result = run_backtest_fixed_size(
            close=df['close'].values,
            high=df['high'].values,
            low=df['low'].values,
            signals=signals,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            trade_size=FIXED_TRADE_SIZE,
        )

        # Store result
        result['pair'] = pair
        result['strategy'] = strategy
        result['tp_pct'] = tp_pct
        result['sl_pct'] = sl_pct
        results.append(result)

        # Accumulate totals
        total_trades += result['trades']
        total_pnl += result['total_pnl']
        total_wins += result['wins']
        total_losses += result['losses']

        # Print row
        pnl_str = f"${result['total_pnl']:+.2f}"
        pnl_color = "" if result['total_pnl'] >= 0 else ""

        print(f"{pair:<12} {strategy:<16} {tp_pct:<5.1f} {sl_pct:<5.1f} {result['trades']:<7} "
              f"{result['wins']}/{result['losses']:<5} {result['win_rate']:<6.1f} {pnl_str:<10} {result['profit_factor']:<6.2f}")

    # Summary
    print("=" * 85)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    final_balance = INITIAL_BALANCE + total_pnl
    return_pct = (total_pnl / INITIAL_BALANCE) * 100

    print(f"\n{'SUMMARY':^80}")
    print("-" * 80)
    print(f"Total Trades:     {total_trades}")
    print(f"Wins/Losses:      {total_wins}/{total_losses}")
    print(f"Overall Win Rate: {overall_wr:.1f}%")
    print(f"Total P&L:        ${total_pnl:+.2f}")
    print(f"Return:           {return_pct:+.2f}%")
    print(f"Starting Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance:    ${final_balance:.2f}")
    print("-" * 80)

    # Top performers
    profitable = [r for r in results if r['total_pnl'] > 0]
    profitable.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"\nTOP 5 PERFORMERS (Last 7 Days):")
    for i, r in enumerate(profitable[:5], 1):
        print(f"  {i}. {r['pair']:<10} {r['strategy']:<16} ${r['total_pnl']:+.2f} ({r['trades']} trades, {r['win_rate']:.0f}% WR)")

    # Worst performers
    losers = [r for r in results if r['total_pnl'] < 0]
    losers.sort(key=lambda x: x['total_pnl'])

    if losers:
        print(f"\nWORST PERFORMERS:")
        for r in losers[:3]:
            print(f"  - {r['pair']:<10} {r['strategy']:<16} ${r['total_pnl']:+.2f}")

    print("\n" + "=" * 80)
    print("Backtest complete!")


if __name__ == "__main__":
    main()
