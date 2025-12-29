"""
Verify DCA logic matches the profitable test results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def backtest_original(df, signals, tp_pct=0.005, sl_pct=0.005, dca_pct=0.002,
                      initial_capital=5000, risk_pct=0.02):
    """Original profitable backtest from test_dca_strategy.py"""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    dca_multipliers = [0.25, 0.125, 0.25, 0.375, 0.5]

    capital = initial_capital
    trades = []
    last_trade_bar = -100

    i = 0
    while i < n - 300:
        if signals[i] == 1 and (i - last_trade_bar) > 50:
            entry_price = close[i]
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)

            dca_prices = [entry_price * (1 - dca_pct * (j + 1)) for j in range(4)]

            base_risk = capital * risk_pct
            position_value = base_risk * dca_multipliers[0] / sl_pct
            current_qty = position_value / entry_price
            avg_entry = entry_price
            dca_stage = 0
            total_investment = position_value

            for j in range(i + 1, min(i + 300, n)):
                bar_high = high[j]
                bar_low = low[j]

                # Check DCA triggers
                while dca_stage < 4 and bar_low <= dca_prices[dca_stage]:
                    dca_stage += 1
                    if dca_stage < len(dca_multipliers):
                        add_risk = base_risk * dca_multipliers[dca_stage]
                        add_value = add_risk / sl_pct
                        add_qty = add_value / dca_prices[dca_stage - 1]

                        total_cost = (avg_entry * current_qty) + (dca_prices[dca_stage - 1] * add_qty)
                        current_qty += add_qty
                        avg_entry = total_cost / current_qty
                        total_investment += add_value

                # Check TP (based on average entry)
                effective_tp = avg_entry * (1 + tp_pct)
                if bar_high >= effective_tp:
                    exit_price = effective_tp
                    pnl = current_qty * (exit_price - avg_entry)
                    trades.append({'pnl': pnl, 'dca_stages': dca_stage, 'outcome': 'TP'})
                    capital += pnl
                    last_trade_bar = j
                    break

                # Check SL
                if bar_low <= sl_price:
                    exit_price = sl_price
                    pnl = current_qty * (exit_price - avg_entry)
                    trades.append({'pnl': pnl, 'dca_stages': dca_stage, 'outcome': 'SL'})
                    capital += pnl
                    last_trade_bar = j
                    break
            else:
                exit_price = close[min(i + 300, n - 1)]
                pnl = current_qty * (exit_price - avg_entry)
                trades.append({'pnl': pnl, 'dca_stages': dca_stage, 'outcome': 'TIMEOUT'})
                capital += pnl
                last_trade_bar = min(i + 300, n - 1)

            i = last_trade_bar + 1
        else:
            i += 1

    return trades, capital


def main():
    print("=" * 70)
    print("VERIFYING DCA LOGIC")
    print("=" * 70)

    # Load data
    data_file = Path(__file__).parent / "Crypto_Data_Fresh" / "BTCUSD_1m.csv"
    df = pd.read_csv(data_file)
    df = df.tail(50000).reset_index(drop=True)
    print(f"Data: {len(df):,} bars")

    # Compute signals - SAME as original test
    df['rsi'] = compute_rsi(df['close'], 14).shift(1)
    df['momentum'] = df['close'].pct_change(5).shift(1) * 100

    # Original signal: momentum > 0.2 AND RSI < 65
    momentum_signals = ((df['momentum'] > 0.2) & (df['rsi'] < 65)).astype(int).values
    print(f"Momentum signals: {momentum_signals.sum():,}")

    # Test configurations
    configs = [
        {'tp': 0.005, 'sl': 0.005, 'dca': 0.002, 'name': '0.5% TP/SL (Original Test)'},
        {'tp': 0.008, 'sl': 0.005, 'dca': 0.003, 'name': '0.8% TP, 0.5% SL'},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")

        trades, final_capital = backtest_original(
            df, momentum_signals,
            tp_pct=cfg['tp'],
            sl_pct=cfg['sl'],
            dca_pct=cfg['dca']
        )

        if trades:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df['pnl'] > 0]

            print(f"  Trades: {len(trades_df)}")
            print(f"  Win Rate: {100*len(wins)/len(trades_df):.1f}%")
            print(f"  Final Capital: ${final_capital:,.2f}")
            print(f"  Return: {100*(final_capital/5000-1):+.1f}%")

            outcomes = trades_df['outcome'].value_counts()
            print(f"  Outcomes: {dict(outcomes)}")
            print(f"  Avg DCA stages: {trades_df['dca_stages'].mean():.1f}")

    # Now test WITHOUT momentum filter (just for comparison)
    print("\n" + "=" * 70)
    print("WITHOUT MOMENTUM FILTER (RSI < 35 only)")
    print("=" * 70)

    rsi_signals = (df['rsi'] < 35).astype(int).values
    print(f"RSI signals: {rsi_signals.sum():,}")

    for cfg in configs[:1]:  # Just test one config
        trades, final_capital = backtest_original(
            df, rsi_signals,
            tp_pct=cfg['tp'],
            sl_pct=cfg['sl'],
            dca_pct=cfg['dca']
        )

        if trades:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df['pnl'] > 0]

            print(f"  Trades: {len(trades_df)}")
            print(f"  Win Rate: {100*len(wins)/len(trades_df):.1f}%")
            print(f"  Final: ${final_capital:,.2f} ({100*(final_capital/5000-1):+.1f}%)")


if __name__ == "__main__":
    main()
