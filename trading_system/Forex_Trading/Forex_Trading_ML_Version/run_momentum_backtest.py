"""
Backtest Momentum Strategy - NO DCA
====================================

Simple H1 trend following with:
- 20 pips Take Profit
- 15 pips Stop Loss
- Trailing stop: activates at +10 pips, trails 8 pips behind

Usage:
    cd C:/Users/Jean-Yves/thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.run_momentum_backtest
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class MomentumBacktester:
    """Backtest the momentum strategy with trailing stop."""

    def __init__(self, symbol: str, start_balance: float = 10000.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance

        # Strategy params - matching trading_config.py
        self.tp_pips = 20.0           # Take profit
        self.sl_pips = 15.0           # Stop loss
        self.trailing_activation = 10.0  # Activate trailing at +10 pips
        self.trailing_distance = 8.0     # Trail 8 pips behind

        # Pip value
        self.pip_value = 0.01 if 'JPY' in symbol else 0.0001

        # Position sizing
        self.risk_pct = 0.02  # 2% risk per trade

        # Stats
        self.trades = []
        self.equity_curve = [start_balance]
        self.peak_balance = start_balance
        self.max_drawdown = 0

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return prices.ewm(span=period, adjust=False).mean()

    def get_h1_trend(self, df: pd.DataFrame, idx: int) -> str:
        """Get H1 trend direction using EMA50 vs EMA200."""
        if idx < 200:
            return 'NONE'

        # Calculate EMAs on close prices up to current bar
        closes = df['close'].iloc[:idx+1]
        ema50 = self.calculate_ema(closes, 50).iloc[-1]
        ema200 = self.calculate_ema(closes, 200).iloc[-1]

        current_price = df['close'].iloc[idx]

        # Trend logic
        if ema50 > ema200 and current_price > ema50:
            return 'BUY'
        elif ema50 < ema200 and current_price < ema50:
            return 'SELL'
        else:
            return 'NONE'

    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest."""
        print(f"\n{'='*70}")
        print(f"MOMENTUM BACKTEST - {self.symbol}")
        print(f"{'='*70}")
        print(f"Strategy: NO DCA | TP: {self.tp_pips}p | SL: {self.sl_pips}p")
        print(f"Trailing: activates +{self.trailing_activation}p, trails {self.trailing_distance}p")
        print(f"Data: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        print(f"Starting Balance: ${self.start_balance:,.2f}")
        print("-" * 70)

        position = None
        trailing_active = False
        peak_profit_pips = 0.0
        trailing_sl = None

        # Track wins/losses
        wins = 0
        losses = 0
        tp_exits = 0
        sl_exits = 0
        trailing_exits = 0

        lookback = 200
        total_bars = len(df) - lookback - 1
        last_progress = 0

        for i in range(lookback, len(df) - 1):
            # Progress
            progress = int((i - lookback) / total_bars * 100)
            if progress >= last_progress + 10:
                print(f"  Progress: {progress}%...", end='\r')
                last_progress = progress

            current_bar = df.iloc[i]
            high = current_bar['high']
            low = current_bar['low']
            close = current_bar['close']

            # If in position, check exits
            if position is not None:
                entry_price = position['entry_price']
                direction = position['direction']

                # Calculate current profit in pips
                if direction == 'BUY':
                    current_profit_pips = (close - entry_price) / self.pip_value
                    high_profit_pips = (high - entry_price) / self.pip_value
                    low_profit_pips = (low - entry_price) / self.pip_value
                else:
                    current_profit_pips = (entry_price - close) / self.pip_value
                    high_profit_pips = (entry_price - low) / self.pip_value
                    low_profit_pips = (entry_price - high) / self.pip_value

                exit_reason = None
                exit_price = None

                # Check TP hit (use high/low for intrabar)
                if high_profit_pips >= self.tp_pips:
                    exit_reason = 'TP'
                    if direction == 'BUY':
                        exit_price = entry_price + (self.tp_pips * self.pip_value)
                    else:
                        exit_price = entry_price - (self.tp_pips * self.pip_value)
                    tp_exits += 1

                # Check SL hit
                elif low_profit_pips <= -self.sl_pips:
                    # But check if trailing SL was hit first
                    if trailing_active and trailing_sl is not None:
                        # Check trailing SL
                        if direction == 'BUY' and low <= trailing_sl:
                            exit_reason = 'TRAIL'
                            exit_price = trailing_sl
                            trailing_exits += 1
                        elif direction == 'SELL' and high >= trailing_sl:
                            exit_reason = 'TRAIL'
                            exit_price = trailing_sl
                            trailing_exits += 1
                        else:
                            exit_reason = 'SL'
                            if direction == 'BUY':
                                exit_price = entry_price - (self.sl_pips * self.pip_value)
                            else:
                                exit_price = entry_price + (self.sl_pips * self.pip_value)
                            sl_exits += 1
                    else:
                        exit_reason = 'SL'
                        if direction == 'BUY':
                            exit_price = entry_price - (self.sl_pips * self.pip_value)
                        else:
                            exit_price = entry_price + (self.sl_pips * self.pip_value)
                        sl_exits += 1

                # Check trailing stop hit (if active)
                elif trailing_active and trailing_sl is not None:
                    if direction == 'BUY' and low <= trailing_sl:
                        exit_reason = 'TRAIL'
                        exit_price = trailing_sl
                        trailing_exits += 1
                    elif direction == 'SELL' and high >= trailing_sl:
                        exit_reason = 'TRAIL'
                        exit_price = trailing_sl
                        trailing_exits += 1

                # Update trailing stop if active
                if exit_reason is None and trailing_active:
                    if high_profit_pips > peak_profit_pips:
                        peak_profit_pips = high_profit_pips
                        # Update trailing SL
                        locked_pips = peak_profit_pips - self.trailing_distance
                        if direction == 'BUY':
                            new_trailing_sl = entry_price + (locked_pips * self.pip_value)
                            if trailing_sl is None or new_trailing_sl > trailing_sl:
                                trailing_sl = new_trailing_sl
                        else:
                            new_trailing_sl = entry_price - (locked_pips * self.pip_value)
                            if trailing_sl is None or new_trailing_sl < trailing_sl:
                                trailing_sl = new_trailing_sl

                # Activate trailing if profit threshold reached
                if exit_reason is None and not trailing_active:
                    if high_profit_pips >= self.trailing_activation:
                        trailing_active = True
                        peak_profit_pips = high_profit_pips
                        locked_pips = peak_profit_pips - self.trailing_distance
                        if direction == 'BUY':
                            trailing_sl = entry_price + (locked_pips * self.pip_value)
                        else:
                            trailing_sl = entry_price - (locked_pips * self.pip_value)

                # Close position if exit triggered
                if exit_reason is not None:
                    # Calculate PnL
                    if direction == 'BUY':
                        pnl_pips = (exit_price - entry_price) / self.pip_value
                    else:
                        pnl_pips = (entry_price - exit_price) / self.pip_value

                    pnl_dollars = position['units'] * pnl_pips * self.pip_value

                    self.balance += pnl_dollars

                    if pnl_dollars > 0:
                        wins += 1
                    else:
                        losses += 1

                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_bar.name,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pips': pnl_pips,
                        'pnl_dollars': pnl_dollars,
                        'exit_reason': exit_reason
                    })

                    # Reset position
                    position = None
                    trailing_active = False
                    peak_profit_pips = 0.0
                    trailing_sl = None

            # If no position, check for entry
            if position is None:
                trend = self.get_h1_trend(df, i)

                if trend in ['BUY', 'SELL']:
                    # Calculate position size
                    risk_amount = self.balance * self.risk_pct
                    units = risk_amount / (self.sl_pips * self.pip_value)

                    position = {
                        'entry_time': current_bar.name,
                        'direction': trend,
                        'entry_price': close,
                        'units': units
                    }

            # Track equity curve and drawdown
            self.equity_curve.append(self.balance)
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            dd = (self.peak_balance - self.balance) / self.peak_balance * 100
            if dd > self.max_drawdown:
                self.max_drawdown = dd

        # Close any open position at end
        if position is not None:
            exit_price = df.iloc[-1]['close']
            if position['direction'] == 'BUY':
                pnl_pips = (exit_price - position['entry_price']) / self.pip_value
            else:
                pnl_pips = (position['entry_price'] - exit_price) / self.pip_value

            pnl_dollars = position['units'] * pnl_pips * self.pip_value
            self.balance += pnl_dollars

            self.trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl_pips': pnl_pips,
                'pnl_dollars': pnl_dollars,
                'exit_reason': 'END'
            })

        # Print results
        self.print_results(wins, losses, tp_exits, sl_exits, trailing_exits)

        return {
            'symbol': self.symbol,
            'balance': self.balance,
            'return_pct': (self.balance - self.start_balance) / self.start_balance * 100,
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.trades) * 100 if self.trades else 0,
            'max_drawdown': self.max_drawdown,
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'trailing_exits': trailing_exits
        }

    def print_results(self, wins, losses, tp_exits, sl_exits, trailing_exits):
        """Print backtest results."""
        total_trades = len(self.trades)
        if total_trades == 0:
            print("No trades executed")
            return

        total_pnl = self.balance - self.start_balance
        win_rate = wins / total_trades * 100

        # Calculate avg win/loss
        winning_trades = [t for t in self.trades if t['pnl_dollars'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_dollars'] <= 0]

        avg_win = np.mean([t['pnl_dollars'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_dollars'] for t in losing_trades]) if losing_trades else 0
        avg_win_pips = np.mean([t['pnl_pips'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pips = np.mean([t['pnl_pips'] for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t['pnl_dollars'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl_dollars'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        print(f"\n{'='*70}")
        print(f"RESULTS - {self.symbol}")
        print(f"{'='*70}")
        print(f"\nPerformance:")
        print(f"  Starting Balance:  ${self.start_balance:,.2f}")
        print(f"  Ending Balance:    ${self.balance:,.2f}")
        print(f"  Net P&L:           ${total_pnl:+,.2f}")
        print(f"  Return:            {total_pnl/self.start_balance*100:+.2f}%")
        print(f"  Max Drawdown:      {self.max_drawdown:.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:      {total_trades}")
        print(f"  Wins:              {wins} ({win_rate:.1f}%)")
        print(f"  Losses:            {losses}")
        print(f"  Profit Factor:     {profit_factor:.2f}")

        print(f"\nAverage Trade:")
        print(f"  Avg Win:           ${avg_win:+.2f} ({avg_win_pips:+.1f} pips)")
        print(f"  Avg Loss:          ${avg_loss:+.2f} ({avg_loss_pips:+.1f} pips)")

        print(f"\nExit Breakdown:")
        print(f"  Take Profit:       {tp_exits}")
        print(f"  Stop Loss:         {sl_exits}")
        print(f"  Trailing Stop:     {trailing_exits}")

        print(f"{'='*70}")


def load_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Load historical data from HistData CSV files."""
    symbol_map = {
        'EUR_USD': 'EURUSD',
        'GBP_USD': 'GBPUSD',
        'USD_JPY': 'USDJPY',
    }

    histdata_symbol = symbol_map.get(symbol, symbol.replace('_', ''))

    # Path to HistData 2024 files
    data_dir = Path(__file__).parent.parent / "Backtesting_data_histdata" / "2024"
    csv_file = data_dir / f"DAT_MT_{histdata_symbol}_M1_2024.csv"

    if not csv_file.exists():
        print(f"Data file not found: {csv_file}")
        return None

    print(f"Loading {symbol} data from {csv_file.name}...")

    try:
        df = pd.read_csv(
            csv_file,
            names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates={'datetime': ['date', 'time']},
            date_format='%Y.%m.%d %H:%M'
        )

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Resample to H1 for trend detection (our strategy uses H1 EMA)
        df_h1 = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Filter to last N days
        if days > 0 and days < 365:
            cutoff = df_h1.index[-1] - timedelta(days=days)
            df_h1 = df_h1[df_h1.index > cutoff]

        print(f"Loaded {len(df_h1):,} H1 candles")
        return df_h1

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run backtest on multiple symbols."""
    print("=" * 70)
    print("MOMENTUM STRATEGY BACKTEST - NO DCA")
    print("=" * 70)
    print("Strategy: H1 EMA50 vs EMA200 trend following")
    print("TP: 20 pips | SL: 15 pips")
    print("Trailing: activates +10 pips, trails 8 pips behind")
    print("=" * 70)

    symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    days = 90  # 3 months backtest

    all_results = []
    total_starting = 0
    total_ending = 0

    for symbol in symbols:
        df = load_historical_data(symbol, days=days)

        if df is not None and len(df) > 200:
            backtester = MomentumBacktester(symbol, start_balance=10000.0)
            result = backtester.run_backtest(df)
            all_results.append(result)
            total_starting += 10000
            total_ending += result['balance']
        else:
            print(f"Skipping {symbol} - insufficient data")

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print("PORTFOLIO SUMMARY")
        print(f"{'='*70}")

        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['wins'] for r in all_results)
        total_losses = sum(r['losses'] for r in all_results)
        total_tp = sum(r['tp_exits'] for r in all_results)
        total_sl = sum(r['sl_exits'] for r in all_results)
        total_trail = sum(r['trailing_exits'] for r in all_results)

        print(f"\n{'Symbol':<12} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>8}")
        print("-" * 50)
        for r in all_results:
            print(f"{r['symbol']:<12} {r['return_pct']:>+9.2f}% {r['total_trades']:>8} {r['win_rate']:>7.1f}% {r['max_drawdown']:>7.2f}%")

        print("-" * 50)
        total_pnl = total_ending - total_starting
        overall_return = total_pnl / total_starting * 100
        overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

        print(f"\nPortfolio Performance:")
        print(f"  Total Starting:    ${total_starting:,.2f}")
        print(f"  Total Ending:      ${total_ending:,.2f}")
        print(f"  Net P&L:           ${total_pnl:+,.2f}")
        print(f"  Return:            {overall_return:+.2f}%")
        print(f"  Total Trades:      {total_trades}")
        print(f"  Overall Win Rate:  {overall_win_rate:.1f}%")

        print(f"\nExit Breakdown (All Symbols):")
        print(f"  Take Profit:       {total_tp}")
        print(f"  Stop Loss:         {total_sl}")
        print(f"  Trailing Stop:     {total_trail}")


if __name__ == "__main__":
    main()
