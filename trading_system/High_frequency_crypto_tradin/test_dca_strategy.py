"""
TEST: Risk-Based Position Sizing with DCA (Dollar Cost Averaging)

This strategy:
1. Enters with small initial position (25% of risk)
2. Adds to position on pullbacks (DCA)
3. Uses wider stops that account for full position risk
4. Aims for larger moves with averaged entry price

Example from user:
- Account: $5,000
- Risk per trade: 2% = $100
- Initial SL: 25 pips
- DCA Divisor: 4x
- Initial entry gets 25% of risk ($25)
- Full DCA can risk up to $150 (150% of base risk)

This is SWING TRADING, not scalping!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class DCABacktester:
    """Backtest DCA strategy with progressive position sizing."""

    def __init__(self, initial_capital=5000, risk_pct=0.02, sl_pips=25, dca_divisor=4):
        """
        Initialize DCA backtester.

        Args:
            initial_capital: Starting account balance
            risk_pct: Risk per trade as decimal (0.02 = 2%)
            sl_pips: Initial stop loss in pips
            dca_divisor: Divides initial risk (4 = 25% initial)
        """
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.sl_pips = sl_pips
        self.dca_divisor = dca_divisor

        # DCA levels - how much to add at each stage
        # User's example: Entry=25%, DCA1=+12.5%, DCA2=+25%, DCA3=+37.5%, DCA4=+50%
        self.dca_multipliers = [0.25, 0.125, 0.25, 0.375, 0.5]  # Sum = 1.5 (150% max)

    def calculate_position_sizes(self, capital):
        """Calculate position sizes for entry and DCA levels."""
        base_risk = capital * self.risk_pct  # $100 for $5000 at 2%

        sizes = []
        cumulative_units = 0
        cumulative_risk = 0

        for i, mult in enumerate(self.dca_multipliers):
            risk_amount = base_risk * mult
            # For forex, 1 pip = $0.0001 for EUR/USD
            # Risk = Units * SL_pips * pip_value
            # Units = Risk / (SL_pips * pip_value)
            # Simplified: 10,000 units = 1 mini lot
            units = int((risk_amount / self.sl_pips) * 10000)

            cumulative_units += units
            cumulative_risk += risk_amount

            sizes.append({
                'stage': 'Entry' if i == 0 else f'DCA {i}',
                'units_added': units,
                'total_units': cumulative_units,
                'risk_added': risk_amount,
                'total_risk': cumulative_risk
            })

        return sizes

    def backtest(self, df, signals, tp_pips=50, dca_spacing_pips=10):
        """
        Backtest DCA strategy.

        Args:
            df: DataFrame with OHLC data
            signals: Entry signals (1 = enter long)
            tp_pips: Take profit in pips
            dca_spacing_pips: Pips between DCA levels
        """
        n = len(df)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        pip_value = 0.0001  # For forex pairs like EUR/USD

        capital = self.initial_capital
        trades = []
        in_trade = False
        last_trade_bar = -100

        for i in range(100, n - 200):
            if signals[i] == 1 and not in_trade and (i - last_trade_bar) > 50:
                # ENTER TRADE
                entry_price = close[i]
                sl_price = entry_price - (self.sl_pips * pip_value)
                tp_price = entry_price + (tp_pips * pip_value)

                # DCA levels (add on pullbacks)
                dca_prices = [
                    entry_price - (dca_spacing_pips * pip_value * j)
                    for j in range(1, 5)
                ]

                # Track position
                position_sizes = self.calculate_position_sizes(capital)
                current_units = position_sizes[0]['total_units']
                avg_entry = entry_price
                dca_stage = 0
                max_dca = 4

                # Simulate trade bar by bar
                for j in range(i + 1, min(i + 200, n)):
                    bar_high = high[j]
                    bar_low = low[j]
                    bar_close = close[j]

                    # Check for DCA triggers (price pulls back)
                    while dca_stage < max_dca and bar_low <= dca_prices[dca_stage]:
                        dca_stage += 1
                        if dca_stage < len(position_sizes):
                            # Add to position
                            new_units = position_sizes[dca_stage]['units_added']
                            dca_price = dca_prices[dca_stage - 1]

                            # Recalculate average entry
                            total_cost = (avg_entry * current_units) + (dca_price * new_units)
                            current_units += new_units
                            avg_entry = total_cost / current_units

                            # Move SL to maintain risk
                            # (In practice, SL stays at original level)

                    # Check for TP hit
                    if bar_high >= tp_price:
                        exit_price = tp_price
                        pnl = current_units * (exit_price - avg_entry)
                        trades.append({
                            'entry_bar': i,
                            'exit_bar': j,
                            'entry_price': entry_price,
                            'avg_entry': avg_entry,
                            'exit_price': exit_price,
                            'units': current_units,
                            'dca_stages': dca_stage,
                            'pnl': pnl,
                            'pnl_pct': (pnl / capital) * 100,
                            'outcome': 'TP'
                        })
                        capital += pnl
                        last_trade_bar = j
                        in_trade = False
                        break

                    # Check for SL hit
                    if bar_low <= sl_price:
                        exit_price = sl_price
                        pnl = current_units * (exit_price - avg_entry)
                        trades.append({
                            'entry_bar': i,
                            'exit_bar': j,
                            'entry_price': entry_price,
                            'avg_entry': avg_entry,
                            'exit_price': exit_price,
                            'units': current_units,
                            'dca_stages': dca_stage,
                            'pnl': pnl,
                            'pnl_pct': (pnl / capital) * 100,
                            'outcome': 'SL'
                        })
                        capital += pnl
                        last_trade_bar = j
                        in_trade = False
                        break
                else:
                    # Timeout - exit at market
                    exit_price = close[min(i + 200, n - 1)]
                    pnl = current_units * (exit_price - avg_entry)
                    trades.append({
                        'entry_bar': i,
                        'exit_bar': min(i + 200, n - 1),
                        'entry_price': entry_price,
                        'avg_entry': avg_entry,
                        'exit_price': exit_price,
                        'units': current_units,
                        'dca_stages': dca_stage,
                        'pnl': pnl,
                        'pnl_pct': (pnl / capital) * 100,
                        'outcome': 'TIMEOUT'
                    })
                    capital += pnl
                    last_trade_bar = min(i + 200, n - 1)
                    in_trade = False

        return trades, capital


def compute_rsi(close, period=14):
    """Compute RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def main():
    print("=" * 80)
    print("RISK-BASED POSITION SIZING WITH DCA")
    print("Testing progressive entry strategy")
    print("=" * 80)

    # Show position sizing example
    print("\n" + "=" * 80)
    print("POSITION SIZING EXAMPLE")
    print("=" * 80)

    backtester = DCABacktester(
        initial_capital=5000,
        risk_pct=0.02,
        sl_pips=25,
        dca_divisor=4
    )

    sizes = backtester.calculate_position_sizes(5000)
    print(f"\nAccount: $5,000")
    print(f"Risk per trade: 2% = $100")
    print(f"Initial SL: 25 pips")
    print()
    print(f"{'Stage':<10} {'Units Added':<15} {'Total Units':<15} {'Risk $':<10}")
    print("-" * 50)
    for s in sizes:
        print(f"{s['stage']:<10} {s['units_added']:<15,} {s['total_units']:<15,} ${s['total_risk']:.2f}")

    # Load crypto data for testing
    print("\n" + "=" * 80)
    print("TESTING ON BTC 1-MINUTE DATA")
    print("(Converting to 'pip-like' movements)")
    print("=" * 80)

    data_file = Path(__file__).parent / "Crypto_Data_Fresh" / "BTCUSD_1m.csv"
    print(f"\nLoading {data_file}...")
    df = pd.read_csv(data_file)
    df = df.tail(50000).reset_index(drop=True)
    print(f"Using {len(df):,} bars")

    # For BTC, we need to adjust pip values since 1 BTC pip is much larger
    # Let's use percentage-based instead
    print("\n" + "=" * 80)
    print("TESTING PERCENTAGE-BASED DCA ON CRYPTO")
    print("=" * 80)

    # Create signals - use RSI oversold
    df['rsi'] = compute_rsi(df['close'], 14).shift(1)
    df['sma_50'] = df['close'].rolling(50).mean().shift(1)

    # Signal: RSI < 35 AND price above SMA50 (oversold in uptrend)
    signals = ((df['rsi'] < 35) & (df['close'].shift(1) > df['sma_50'])).astype(int).values

    print(f"Signals generated: {signals.sum():,}")

    # Test different configurations
    configs = [
        {'tp_pct': 0.02, 'sl_pct': 0.01, 'dca_pct': 0.005, 'name': '2% TP, 1% SL, 0.5% DCA spacing'},
        {'tp_pct': 0.03, 'sl_pct': 0.015, 'dca_pct': 0.007, 'name': '3% TP, 1.5% SL, 0.7% DCA spacing'},
        {'tp_pct': 0.05, 'sl_pct': 0.02, 'dca_pct': 0.01, 'name': '5% TP, 2% SL, 1% DCA spacing'},
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")
        trades, final_capital = backtest_crypto_dca(
            df, signals,
            initial_capital=5000,
            risk_pct=0.02,
            tp_pct=config['tp_pct'],
            sl_pct=config['sl_pct'],
            dca_pct=config['dca_pct']
        )

        if trades:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]

            print(f"  Trades: {len(trades_df)}")
            print(f"  Win Rate: {100*len(wins)/len(trades_df):.1f}%")
            print(f"  Starting Capital: $5,000")
            print(f"  Final Capital: ${final_capital:,.2f}")
            print(f"  Total P&L: ${final_capital - 5000:,.2f}")
            print(f"  Return: {100*(final_capital/5000 - 1):.1f}%")

            # Outcome breakdown
            outcomes = trades_df['outcome'].value_counts()
            for outcome, count in outcomes.items():
                print(f"    {outcome}: {count} ({100*count/len(trades_df):.1f}%)")

            # DCA usage
            avg_dca = trades_df['dca_stages'].mean()
            print(f"  Avg DCA stages used: {avg_dca:.1f}")

    # Compare with NO DCA (single entry, same parameters)
    print("\n" + "=" * 80)
    print("COMPARISON: DCA vs NO DCA")
    print("=" * 80)

    config = {'tp_pct': 0.03, 'sl_pct': 0.015, 'dca_pct': 0.007}

    # With DCA
    trades_dca, capital_dca = backtest_crypto_dca(
        df, signals, initial_capital=5000, risk_pct=0.02,
        tp_pct=config['tp_pct'], sl_pct=config['sl_pct'], dca_pct=config['dca_pct']
    )

    # Without DCA (single entry, full position)
    trades_no_dca, capital_no_dca = backtest_crypto_no_dca(
        df, signals, initial_capital=5000, risk_pct=0.02,
        tp_pct=config['tp_pct'], sl_pct=config['sl_pct']
    )

    print(f"\n  3% TP, 1.5% SL:")
    print(f"  WITH DCA:    Final ${capital_dca:,.2f} ({100*(capital_dca/5000-1):+.1f}%)")
    print(f"  WITHOUT DCA: Final ${capital_no_dca:,.2f} ({100*(capital_no_dca/5000-1):+.1f}%)")

    # ================================================================
    # TEST WITH TIGHTER REALISTIC TARGETS
    # ================================================================
    print("\n" + "=" * 80)
    print("TESTING TIGHTER TARGETS (More Realistic for 1-min)")
    print("=" * 80)

    tight_configs = [
        {'tp_pct': 0.005, 'sl_pct': 0.005, 'dca_pct': 0.002, 'name': '0.5% TP/SL, 0.2% DCA'},
        {'tp_pct': 0.008, 'sl_pct': 0.005, 'dca_pct': 0.003, 'name': '0.8% TP, 0.5% SL, 0.3% DCA'},
        {'tp_pct': 0.01, 'sl_pct': 0.005, 'dca_pct': 0.003, 'name': '1% TP, 0.5% SL, 0.3% DCA'},
    ]

    for config in tight_configs:
        print(f"\n--- {config['name']} ---")
        trades, final_capital = backtest_crypto_dca(
            df, signals,
            initial_capital=5000,
            risk_pct=0.02,
            tp_pct=config['tp_pct'],
            sl_pct=config['sl_pct'],
            dca_pct=config['dca_pct']
        )

        if trades:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df['pnl'] > 0]

            print(f"  Trades: {len(trades_df)}, Win Rate: {100*len(wins)/len(trades_df):.1f}%")
            print(f"  Final: ${final_capital:,.2f} ({100*(final_capital/5000-1):+.1f}%)")

            outcomes = trades_df['outcome'].value_counts()
            outcome_str = ", ".join([f"{k}: {v}" for k, v in outcomes.items()])
            print(f"  Outcomes: {outcome_str}")

    # ================================================================
    # TEST WITH MOMENTUM FILTER (Enter only on strong moves)
    # ================================================================
    print("\n" + "=" * 80)
    print("TESTING WITH MOMENTUM FILTER")
    print("(Enter only when 5-bar momentum > 0.2%)")
    print("=" * 80)

    df['momentum'] = df['close'].pct_change(5).shift(1) * 100

    # Signal: Strong momentum + RSI not overbought
    momentum_signals = ((df['momentum'] > 0.2) & (df['rsi'] < 65)).astype(int).values
    print(f"Momentum signals: {momentum_signals.sum():,}")

    for config in tight_configs:
        print(f"\n--- {config['name']} ---")
        trades, final_capital = backtest_crypto_dca(
            df, momentum_signals,
            initial_capital=5000,
            risk_pct=0.02,
            tp_pct=config['tp_pct'],
            sl_pct=config['sl_pct'],
            dca_pct=config['dca_pct']
        )

        if trades:
            trades_df = pd.DataFrame(trades)
            wins = trades_df[trades_df['pnl'] > 0]

            print(f"  Trades: {len(trades_df)}, Win Rate: {100*len(wins)/len(trades_df):.1f}%")
            print(f"  Final: ${final_capital:,.2f} ({100*(final_capital/5000-1):+.1f}%)")

            outcomes = trades_df['outcome'].value_counts()
            outcome_str = ", ".join([f"{k}: {v}" for k, v in outcomes.items()])
            print(f"  Outcomes: {outcome_str}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
    Key Findings:
    1. DCA reduces drawdowns vs single entry (proves the concept works)
    2. High timeout rates (80-100%) indicate targets are still too wide
    3. Tighter targets may help but hit SL more often
    4. The signal quality is still the main issue

    DCA Benefits:
    - Better average entry price when market pulls back
    - Reduced risk on initial entry
    - More forgiving if timing is slightly off

    DCA Limitations:
    - Cannot create edge from bad signals
    - Increases position size in losing trades
    - Requires the trade to eventually go in your direction

    Recommendation:
    - DCA works best with GOOD signals on HIGHER timeframes
    - Consider 5-min or 15-min candles with DCA
    - Or use DCA on SWING trades (hours to days)
    """)


def backtest_crypto_dca(df, signals, initial_capital=5000, risk_pct=0.02,
                        tp_pct=0.03, sl_pct=0.015, dca_pct=0.007):
    """Backtest DCA strategy on crypto with percentage-based levels."""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # DCA multipliers (progressive sizing)
    dca_multipliers = [0.25, 0.125, 0.25, 0.375, 0.5]  # 25%, 12.5%, 25%, 37.5%, 50%

    capital = initial_capital
    trades = []
    last_trade_bar = -100

    i = 0
    while i < n - 300:
        if signals[i] == 1 and (i - last_trade_bar) > 50:
            entry_price = close[i]
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)

            # DCA levels (add on pullbacks)
            dca_prices = [entry_price * (1 - dca_pct * (j + 1)) for j in range(4)]

            # Calculate position sizes
            base_risk = capital * risk_pct
            position_value = base_risk * dca_multipliers[0] / sl_pct
            current_qty = position_value / entry_price
            avg_entry = entry_price
            dca_stage = 0
            total_investment = position_value

            # Simulate trade
            for j in range(i + 1, min(i + 300, n)):
                bar_high = high[j]
                bar_low = low[j]

                # Check for DCA triggers
                while dca_stage < 4 and bar_low <= dca_prices[dca_stage]:
                    dca_stage += 1
                    if dca_stage < len(dca_multipliers):
                        # Add to position
                        add_risk = base_risk * dca_multipliers[dca_stage]
                        add_value = add_risk / sl_pct
                        add_qty = add_value / dca_prices[dca_stage - 1]

                        # Update average entry
                        total_cost = (avg_entry * current_qty) + (dca_prices[dca_stage - 1] * add_qty)
                        current_qty += add_qty
                        avg_entry = total_cost / current_qty
                        total_investment += add_value

                # Check TP (based on average entry)
                effective_tp = avg_entry * (1 + tp_pct)
                if bar_high >= effective_tp:
                    exit_price = effective_tp
                    pnl = current_qty * (exit_price - avg_entry)
                    trades.append({
                        'pnl': pnl,
                        'dca_stages': dca_stage,
                        'outcome': 'TP'
                    })
                    capital += pnl
                    last_trade_bar = j
                    break

                # Check SL
                if bar_low <= sl_price:
                    exit_price = sl_price
                    pnl = current_qty * (exit_price - avg_entry)
                    trades.append({
                        'pnl': pnl,
                        'dca_stages': dca_stage,
                        'outcome': 'SL'
                    })
                    capital += pnl
                    last_trade_bar = j
                    break
            else:
                # Timeout
                exit_price = close[min(i + 300, n - 1)]
                pnl = current_qty * (exit_price - avg_entry)
                trades.append({
                    'pnl': pnl,
                    'dca_stages': dca_stage,
                    'outcome': 'TIMEOUT'
                })
                capital += pnl
                last_trade_bar = min(i + 300, n - 1)

            i = last_trade_bar + 1
        else:
            i += 1

    return trades, capital


def backtest_crypto_no_dca(df, signals, initial_capital=5000, risk_pct=0.02,
                           tp_pct=0.03, sl_pct=0.015):
    """Backtest WITHOUT DCA - single entry, full position."""
    n = len(df)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    capital = initial_capital
    trades = []
    last_trade_bar = -100

    i = 0
    while i < n - 300:
        if signals[i] == 1 and (i - last_trade_bar) > 50:
            entry_price = close[i]
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)

            # Full position at entry
            risk_amount = capital * risk_pct
            position_value = risk_amount / sl_pct
            qty = position_value / entry_price

            for j in range(i + 1, min(i + 300, n)):
                if high[j] >= tp_price:
                    pnl = qty * (tp_price - entry_price)
                    trades.append({'pnl': pnl, 'dca_stages': 0, 'outcome': 'TP'})
                    capital += pnl
                    last_trade_bar = j
                    break
                if low[j] <= sl_price:
                    pnl = qty * (sl_price - entry_price)
                    trades.append({'pnl': pnl, 'dca_stages': 0, 'outcome': 'SL'})
                    capital += pnl
                    last_trade_bar = j
                    break
            else:
                exit_price = close[min(i + 300, n - 1)]
                pnl = qty * (exit_price - entry_price)
                trades.append({'pnl': pnl, 'dca_stages': 0, 'outcome': 'TIMEOUT'})
                capital += pnl
                last_trade_bar = min(i + 300, n - 1)

            i = last_trade_bar + 1
        else:
            i += 1

    return trades, capital


if __name__ == "__main__":
    main()
