#!/usr/bin/env python3
"""
REALISTIC 1-Week Backtest - All 17 Pairs
=========================================
$100 capital, $5 fixed per trade
WITH: Fees, Slippage, Spread simulation

Kraken Fee Structure (as of Dec 2025):
- Maker: 0.16%
- Taker: 0.26%
- We use taker fee (market orders) for entry/exit
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
INITIAL_BALANCE = 100.0       # $100 starting capital
FIXED_TRADE_SIZE = 5.0        # $5 per trade
ONE_WEEK_CANDLES = 7 * 24 * 60  # 1 week = 10,080 minutes

# REALISTIC TRADING COSTS
TAKER_FEE_PCT = 0.26          # Kraken taker fee 0.26%
SLIPPAGE_PCT = 0.02           # 0.02% average slippage (conservative)
SPREAD_PCT = 0.03             # 0.03% average spread (conservative for major pairs)

# Combined cost per side (entry OR exit)
COST_PER_SIDE = TAKER_FEE_PCT + SLIPPAGE_PCT + (SPREAD_PCT / 2)  # ~0.295%
ROUND_TRIP_COST = COST_PER_SIDE * 2  # ~0.59% per trade


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
    close_prev = np.roll(close, 1)

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

    elif strategy == 'MACD_ZERO':
        # MACD crosses zero line
        buy = (macd_prev < 0) & (macd >= 0)
        sell = (macd_prev > 0) & (macd <= 0)
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    # =========================================================================
    # NEW STRATEGIES
    # =========================================================================

    elif strategy == 'RSI_DIVERGENCE':
        # RSI making higher lows while price makes lower lows (bullish)
        close_prev2 = np.roll(close, 2)
        rsi_prev2 = np.roll(rsi, 2)
        price_lower_low = (close < close_prev) & (close_prev < close_prev2)
        rsi_higher_low = (rsi > rsi_prev) & (rsi_prev > rsi_prev2)
        bullish_div = price_lower_low & rsi_higher_low & (rsi < 40)

        price_higher_high = (close > close_prev) & (close_prev > close_prev2)
        rsi_lower_high = (rsi < rsi_prev) & (rsi_prev < rsi_prev2)
        bearish_div = price_higher_high & rsi_lower_high & (rsi > 60)
        signals = np.where(bullish_div, 1, np.where(bearish_div, -1, 0))

    elif strategy == 'SWING_REVERSAL':
        # Price makes swing high/low with RSI confirmation
        high = df['high'].values
        low = df['low'].values
        high_prev = np.roll(high, 1)
        low_prev = np.roll(low, 1)
        swing_low = (low < low_prev) & (low < np.roll(low, -1)) & (rsi < 35)
        swing_high = (high > high_prev) & (high > np.roll(high, -1)) & (rsi > 65)
        signals = np.where(swing_low, 1, np.where(swing_high, -1, 0))

    elif strategy == 'MOMENTUM_BREAKOUT':
        # Strong price move with trend confirmation
        close_pct_change = (close - close_prev) / close_prev * 100
        ema9 = df['ema9'].values
        strong_uptrend = (ema9 > ema21) & (ema21 > ema50)
        strong_downtrend = (ema9 < ema21) & (ema21 < ema50)
        strong_up_move = close_pct_change > 0.5
        strong_down_move = close_pct_change < -0.5
        buy = strong_up_move & strong_uptrend & (rsi > 50) & (rsi < 70)
        sell = strong_down_move & strong_downtrend & (rsi < 50) & (rsi > 30)
        signals = np.where(buy, 1, np.where(sell, -1, 0))

    elif strategy == 'RSI_TREND_FOLLOW':
        # Enter on RSI pullback in strong trend
        ema9 = df['ema9'].values
        strong_uptrend = (ema9 > ema21) & (ema21 > ema50)
        strong_downtrend = (ema9 < ema21) & (ema21 < ema50)
        rsi_pullback_bull = (rsi < 45) & (rsi > 30) & strong_uptrend & (macd > 0)
        rsi_pullback_bear = (rsi > 55) & (rsi < 70) & strong_downtrend & (macd < 0)
        signals = np.where(rsi_pullback_bull, 1, np.where(rsi_pullback_bear, -1, 0))

    # Set first 50 rows to 0 (indicator warmup)
    signals[:50] = 0
    return signals


def run_realistic_backtest(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signals: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    trade_size: float = 5.0,
    fee_pct: float = TAKER_FEE_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
) -> dict:
    """
    Run REALISTIC backtest with:
    - Fees (entry + exit)
    - Slippage (worse entry/exit prices)
    - One trade at a time per symbol
    """
    n = len(close)
    trades = []

    in_position = False
    entry_price = 0.0
    actual_entry = 0.0  # After slippage
    entry_idx = 0
    direction = 0
    take_profit = 0.0
    stop_loss = 0.0

    total_fees = 0.0

    for i in range(50, n):
        current_price = close[i]

        if in_position:
            h = high[i]
            l = low[i]
            exit_price = None
            exit_reason = None

            if direction == 1:  # Long
                # SL/TP check with slippage (exit at worse price)
                if l <= stop_loss:
                    # SL hit - slippage makes it worse
                    exit_price = stop_loss * (1 - slippage_pct / 100)
                    exit_reason = 'SL'
                elif h >= take_profit:
                    # TP hit - slippage still costs a bit
                    exit_price = take_profit * (1 - slippage_pct / 100)
                    exit_reason = 'TP'
            else:  # Short
                if h >= stop_loss:
                    exit_price = stop_loss * (1 + slippage_pct / 100)
                    exit_reason = 'SL'
                elif l <= take_profit:
                    exit_price = take_profit * (1 + slippage_pct / 100)
                    exit_reason = 'TP'

            if exit_price is not None:
                # Calculate P&L with actual entry (after slippage)
                if direction == 1:
                    pnl_pct = (exit_price - actual_entry) / actual_entry * 100
                else:
                    pnl_pct = (actual_entry - exit_price) / actual_entry * 100

                # Raw P&L
                raw_pnl = trade_size * (pnl_pct / 100)

                # Subtract fees (entry + exit)
                entry_fee = trade_size * (fee_pct / 100)
                exit_fee = trade_size * (fee_pct / 100)
                total_fee = entry_fee + exit_fee
                total_fees += total_fee

                # Net P&L after fees
                net_pnl = raw_pnl - total_fee

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'signal_price': entry_price,
                    'actual_entry': actual_entry,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'raw_pnl': raw_pnl,
                    'fees': total_fee,
                    'net_pnl': net_pnl,
                    'pnl_pct': pnl_pct,
                })

                in_position = False

        else:
            sig = signals[i]
            if sig != 0:
                direction = int(sig)
                entry_price = current_price
                entry_idx = i

                # Apply slippage to entry (we get worse price)
                if direction == 1:  # Buying - price goes up
                    actual_entry = current_price * (1 + slippage_pct / 100)
                    take_profit = actual_entry * (1 + tp_pct / 100)
                    stop_loss = actual_entry * (1 - sl_pct / 100)
                else:  # Selling - price goes down
                    actual_entry = current_price * (1 - slippage_pct / 100)
                    take_profit = actual_entry * (1 - tp_pct / 100)
                    stop_loss = actual_entry * (1 + sl_pct / 100)

                in_position = True

    # Calculate metrics
    total_trades = len(trades)
    if total_trades == 0:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'gross_pnl': 0, 'total_fees': 0, 'net_pnl': 0,
            'avg_trade': 0, 'profit_factor': 0
        }

    wins = sum(1 for t in trades if t['net_pnl'] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100

    gross_pnl = sum(t['raw_pnl'] for t in trades)
    net_pnl = sum(t['net_pnl'] for t in trades)
    avg_trade = net_pnl / total_trades

    gross_profit = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0) or 0.001
    gross_loss = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] < 0)) or 0.001
    profit_factor = gross_profit / gross_loss

    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'gross_pnl': round(gross_pnl, 2),
        'total_fees': round(total_fees, 2),
        'net_pnl': round(net_pnl, 2),
        'avg_trade': round(avg_trade, 4),
        'profit_factor': round(profit_factor, 2),
        'trade_list': trades,
    }


def main():
    print("\n" + "=" * 90)
    print("REALISTIC 1-WEEK BACKTEST - ALL 17 PAIRS (With Fees & Slippage)")
    print("=" * 90)
    print(f"Capital: ${INITIAL_BALANCE:.2f} | Trade Size: ${FIXED_TRADE_SIZE:.2f}")
    print(f"Period: Last 7 days (~{ONE_WEEK_CANDLES:,} 1-minute candles)")
    print("-" * 90)
    print(f"TRADING COSTS APPLIED:")
    print(f"  - Kraken Taker Fee: {TAKER_FEE_PCT}% per side ({TAKER_FEE_PCT*2}% round-trip)")
    print(f"  - Slippage:         {SLIPPAGE_PCT}% per side")
    print(f"  - Total Cost:       ~{ROUND_TRIP_COST:.2f}% per trade")
    print("=" * 90)

    data_dir = Path(__file__).parent / 'trading_system' / 'Crypto_Margin_Trading' / 'Crypto_Data_from_Binance'

    results = []
    total_trades = 0
    total_gross_pnl = 0.0
    total_fees = 0.0
    total_net_pnl = 0.0
    total_wins = 0
    total_losses = 0

    print(f"\n{'PAIR':<12} {'STRATEGY':<16} {'TRADES':<7} {'W/L':<8} {'WR%':<6} {'GROSS':<10} {'FEES':<8} {'NET PnL':<10} {'PF':<6}")
    print("-" * 95)

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

        # Run REALISTIC backtest
        result = run_realistic_backtest(
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
        total_gross_pnl += result['gross_pnl']
        total_fees += result['total_fees']
        total_net_pnl += result['net_pnl']
        total_wins += result['wins']
        total_losses += result['losses']

        # Print row
        print(f"{pair:<12} {strategy:<16} {result['trades']:<7} "
              f"{result['wins']}/{result['losses']:<5} {result['win_rate']:<6.1f} "
              f"${result['gross_pnl']:>+7.2f}  ${result['total_fees']:<6.2f} "
              f"${result['net_pnl']:>+8.2f}  {result['profit_factor']:<6.2f}")

    # Summary
    print("=" * 95)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    final_balance = INITIAL_BALANCE + total_net_pnl
    return_pct = (total_net_pnl / INITIAL_BALANCE) * 100
    fee_pct_of_gross = (total_fees / abs(total_gross_pnl) * 100) if total_gross_pnl != 0 else 0

    print(f"\n{'REALISTIC SUMMARY (After Fees & Slippage)':^90}")
    print("-" * 90)
    print(f"Total Trades:       {total_trades}")
    print(f"Wins/Losses:        {total_wins}/{total_losses}")
    print(f"Overall Win Rate:   {overall_wr:.1f}%")
    print(f"")
    print(f"Gross P&L:          ${total_gross_pnl:+.2f}")
    print(f"Total Fees Paid:    ${total_fees:.2f} ({fee_pct_of_gross:.1f}% of gross)")
    print(f"NET P&L:            ${total_net_pnl:+.2f}")
    print(f"")
    print(f"Starting Balance:   ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance:      ${final_balance:.2f}")
    print(f"Return (after fees):{return_pct:+.2f}%")
    print("-" * 90)

    # Comparison
    print(f"\n{'IMPACT OF TRADING COSTS':^90}")
    print("-" * 90)
    # What it would be without fees
    ideal_return = (total_gross_pnl / INITIAL_BALANCE) * 100
    print(f"Gross Return (no fees):    {ideal_return:+.2f}%")
    print(f"Net Return (with fees):    {return_pct:+.2f}%")
    print(f"Cost Impact:               {ideal_return - return_pct:.2f}% reduction")
    print(f"Avg Fee per Trade:         ${total_fees/total_trades:.4f}" if total_trades > 0 else "N/A")

    # Top performers after fees
    profitable = [r for r in results if r['net_pnl'] > 0]
    profitable.sort(key=lambda x: x['net_pnl'], reverse=True)

    print(f"\nTOP 5 PERFORMERS (After Fees):")
    for i, r in enumerate(profitable[:5], 1):
        print(f"  {i}. {r['pair']:<10} {r['strategy']:<16} Net: ${r['net_pnl']:+.2f} (Fees: ${r['total_fees']:.2f})")

    # Worst performers
    losers = [r for r in results if r['net_pnl'] < 0]
    losers.sort(key=lambda x: x['net_pnl'])

    if losers:
        print(f"\nPAIRS THAT BECAME UNPROFITABLE AFTER FEES:")
        for r in losers:
            print(f"  - {r['pair']:<10} Gross: ${r['gross_pnl']:+.2f} -> Net: ${r['net_pnl']:+.2f}")

    print("\n" + "=" * 90)
    print("Realistic backtest complete!")


if __name__ == "__main__":
    main()
