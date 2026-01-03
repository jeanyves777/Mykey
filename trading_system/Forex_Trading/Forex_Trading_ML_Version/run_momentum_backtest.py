"""
Backtest Signal-Based Strategy - NO DCA
========================================

Signal-based entries using ADX + Volatility + HTF confirmation:
- ADX > 25 (trending market)
- ATR expansion (volatility spike)
- H1 EMA50 vs EMA200 confirms direction
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


class SignalBacktester:
    """Backtest the signal-based strategy with ADX + ATR + HTF."""

    def __init__(self, symbol: str, start_balance: float = 10000.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance

        # Strategy params - OPTIMIZED #1 (Best Return)
        self.tp_pips = 50.0           # Take profit
        self.sl_pips = 20.0           # Stop loss
        self.trailing_activation = 8.0  # Activate trailing at +8 pips
        self.trailing_distance = 5.0     # Trail 5 pips behind

        # Signal params - OPTIMIZED #1
        self.adx_period = 14
        self.adx_threshold = 20.0     # ADX > 20 = more signals
        self.atr_period = 14
        self.atr_expansion_mult = 1.2  # ATR > 1.2x avg (more signals)

        # Pip value
        self.pip_value = 0.01 if 'JPY' in symbol else 0.0001

        # Position sizing - $5 per pip (OPTIMIZED)
        self.dollars_per_pip = 5.0  # $5 per pip

        # Stats
        self.trades = []
        self.equity_curve = [start_balance]
        self.peak_balance = start_balance
        self.max_drawdown = 0

        # Signal stats
        self.signals_checked = 0
        self.signals_triggered = 0

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """Calculate ADX. Returns (adx, plus_di, minus_di)."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)

        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()

        return adx, plus_di, minus_di

    def get_h1_trend(self, df: pd.DataFrame, idx: int) -> str:
        """Get H1 trend direction using EMA50 vs EMA200."""
        if idx < 200:
            return 'NONE'

        closes = df['close'].iloc[:idx+1]
        ema50 = self.calculate_ema(closes, 50).iloc[-1]
        ema200 = self.calculate_ema(closes, 200).iloc[-1]

        current_price = df['close'].iloc[idx]

        if ema50 > ema200 and current_price > ema50:
            return 'BUY'
        elif ema50 < ema200 and current_price < ema50:
            return 'SELL'
        else:
            return 'NONE'

    def check_entry_signal(self, df: pd.DataFrame, idx: int, adx: float,
                           plus_di: float, minus_di: float,
                           atr: float, atr_avg: float) -> str:
        """Check for entry signal using ADX + ATR + HTF.
        Returns 'BUY', 'SELL', or None.
        """
        self.signals_checked += 1

        # Condition 1: ADX above threshold
        if adx < self.adx_threshold:
            return None

        # Condition 2: Volatility expansion
        if atr < atr_avg * self.atr_expansion_mult:
            return None

        # Condition 3: HTF trend direction
        htf_trend = self.get_h1_trend(df, idx)
        if htf_trend not in ['BUY', 'SELL']:
            return None

        # Condition 4: DI confirms direction
        if htf_trend == 'BUY' and plus_di > minus_di:
            self.signals_triggered += 1
            return 'BUY'
        elif htf_trend == 'SELL' and minus_di > plus_di:
            self.signals_triggered += 1
            return 'SELL'

        return None

    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest."""
        print(f"\n{'='*70}")
        print(f"SIGNAL BACKTEST - {self.symbol}")
        print(f"{'='*70}")
        print(f"Strategy: ADX > {self.adx_threshold} | ATR > {self.atr_expansion_mult}x avg | HTF confirm")
        print(f"TP: {self.tp_pips}p | SL: {self.sl_pips}p")
        print(f"Trailing: activates +{self.trailing_activation}p, trails {self.trailing_distance}p")
        print(f"Data: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        print(f"Starting Balance: ${self.start_balance:,.2f}")
        print("-" * 70)

        # Pre-calculate indicators
        adx_series, plus_di_series, minus_di_series = self.calculate_adx(df, self.adx_period)
        atr_series = self.calculate_atr(df, self.atr_period)
        atr_avg_series = atr_series.rolling(window=20).mean()

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

            # Get indicator values
            adx = adx_series.iloc[i]
            plus_di = plus_di_series.iloc[i]
            minus_di = minus_di_series.iloc[i]
            atr = atr_series.iloc[i]
            atr_avg = atr_avg_series.iloc[i]

            if pd.isna(adx) or pd.isna(atr) or pd.isna(atr_avg):
                continue

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

            # If no position, check for entry signal
            if position is None:
                signal = self.check_entry_signal(df, i, adx, plus_di, minus_di, atr, atr_avg)

                if signal in ['BUY', 'SELL']:
                    # Calculate position size for $1 per pip
                    # For EUR_USD: 1 pip = 0.0001, so 10,000 units = $1/pip
                    # For USD_JPY: 1 pip = 0.01, so 100 units = $1/pip (approx)
                    units = self.dollars_per_pip / self.pip_value

                    position = {
                        'entry_time': current_bar.name,
                        'direction': signal,
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

        print(f"\nSignal Statistics:")
        print(f"  Bars Checked:      {self.signals_checked:,}")
        print(f"  Signals Triggered: {self.signals_triggered}")
        signal_rate = self.signals_triggered / self.signals_checked * 100 if self.signals_checked > 0 else 0
        print(f"  Signal Rate:       {signal_rate:.2f}%")

        print(f"{'='*70}")


def load_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Load historical data from HistData CSV files."""
    symbol_map = {
        'EUR_USD': 'EURUSD',
        'GBP_USD': 'GBPUSD',
        'USD_JPY': 'USDJPY',
        'USD_CHF': 'USDCHF',
        'USD_CAD': 'USDCAD',
        'AUD_USD': 'AUDUSD',
        'NZD_USD': 'NZDUSD',
        'EUR_GBP': 'EURGBP',
        'EUR_JPY': 'EURJPY',
        'GBP_JPY': 'GBPJPY',
        'AUD_JPY': 'AUDJPY',
        'EUR_CAD': 'EURCAD',
        'AUD_CHF': 'AUDCHF',
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
    """Run backtest comparing $1-$5 per pip."""
    print("=" * 80)
    print("POSITION SIZE COMPARISON: $1/pip to $5/pip - USD_JPY 90 DAYS")
    print("=" * 80)
    print("Strategy: ADX > 20 | ATR > 1.2x | TP: 50p | SL: 20p | Trail: +8p/5p")
    print("=" * 80)

    symbol = 'USD_JPY'
    days = 90
    start_balance = 500.0

    df = load_historical_data(symbol, days=days)

    if df is None or len(df) < 200:
        print("Insufficient data")
        return

    print(f"\nData: {len(df):,} H1 candles from {df.index[0]} to {df.index[-1]}")
    print(f"Starting Balance: ${start_balance:.2f}")
    print("\n" + "=" * 80)
    print(f"{'$/pip':<8} {'End Bal':>12} {'P&L':>12} {'Return':>10} {'Trades':>8} {'Wins':>6} {'Loss':>6} {'Win%':>8} {'MaxDD':>8} {'Risk/Trade':>12}")
    print("-" * 80)

    all_results = []

    for dollars_per_pip in [1, 2, 3, 4, 5]:
        # Create backtester with specific $/pip
        backtester = SignalBacktester(symbol, start_balance=start_balance)
        backtester.dollars_per_pip = dollars_per_pip

        # Run silently
        backtester.signals_checked = 0
        backtester.signals_triggered = 0

        # Pre-calculate indicators
        adx_series, plus_di_series, minus_di_series = backtester.calculate_adx(df, 14)
        atr_series = backtester.calculate_atr(df, 14)
        atr_avg_series = atr_series.rolling(window=20).mean()

        position = None
        trailing_active = False
        peak_profit_pips = 0.0
        trailing_sl = None
        wins = 0
        losses = 0
        peak_balance = start_balance
        max_dd = 0

        for i in range(200, len(df) - 1):
            current_bar = df.iloc[i]
            high = current_bar['high']
            low = current_bar['low']
            close = current_bar['close']

            adx = adx_series.iloc[i]
            plus_di = plus_di_series.iloc[i]
            minus_di = minus_di_series.iloc[i]
            atr = atr_series.iloc[i]
            atr_avg = atr_avg_series.iloc[i]

            if pd.isna(adx) or pd.isna(atr) or pd.isna(atr_avg):
                continue

            # Check exits
            if position is not None:
                entry_price = position['entry_price']
                direction = position['direction']

                if direction == 'BUY':
                    high_profit_pips = (high - entry_price) / backtester.pip_value
                    low_profit_pips = (low - entry_price) / backtester.pip_value
                else:
                    high_profit_pips = (entry_price - low) / backtester.pip_value
                    low_profit_pips = (entry_price - high) / backtester.pip_value

                exit_reason = None
                exit_pips = 0

                # TP check
                if high_profit_pips >= backtester.tp_pips:
                    exit_reason = 'TP'
                    exit_pips = backtester.tp_pips
                # SL check
                elif low_profit_pips <= -backtester.sl_pips:
                    if trailing_active and trailing_sl is not None:
                        if (direction == 'BUY' and low <= trailing_sl) or (direction == 'SELL' and high >= trailing_sl):
                            exit_reason = 'TRAIL'
                            exit_pips = (trailing_sl - entry_price) / backtester.pip_value if direction == 'BUY' else (entry_price - trailing_sl) / backtester.pip_value
                        else:
                            exit_reason = 'SL'
                            exit_pips = -backtester.sl_pips
                    else:
                        exit_reason = 'SL'
                        exit_pips = -backtester.sl_pips
                # Trailing check
                elif trailing_active and trailing_sl is not None:
                    if (direction == 'BUY' and low <= trailing_sl) or (direction == 'SELL' and high >= trailing_sl):
                        exit_reason = 'TRAIL'
                        exit_pips = (trailing_sl - entry_price) / backtester.pip_value if direction == 'BUY' else (entry_price - trailing_sl) / backtester.pip_value

                # Update trailing
                if exit_reason is None and trailing_active and high_profit_pips > peak_profit_pips:
                    peak_profit_pips = high_profit_pips
                    locked_pips = peak_profit_pips - backtester.trailing_distance
                    if direction == 'BUY':
                        new_sl = entry_price + (locked_pips * backtester.pip_value)
                        if trailing_sl is None or new_sl > trailing_sl:
                            trailing_sl = new_sl
                    else:
                        new_sl = entry_price - (locked_pips * backtester.pip_value)
                        if trailing_sl is None or new_sl < trailing_sl:
                            trailing_sl = new_sl

                # Activate trailing
                if exit_reason is None and not trailing_active and high_profit_pips >= backtester.trailing_activation:
                    trailing_active = True
                    peak_profit_pips = high_profit_pips
                    locked_pips = peak_profit_pips - backtester.trailing_distance
                    if direction == 'BUY':
                        trailing_sl = entry_price + (locked_pips * backtester.pip_value)
                    else:
                        trailing_sl = entry_price - (locked_pips * backtester.pip_value)

                # Close position
                if exit_reason:
                    pnl = exit_pips * dollars_per_pip
                    backtester.balance += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
                    trailing_active = False
                    peak_profit_pips = 0.0
                    trailing_sl = None

            # Entry check
            if position is None:
                if adx >= backtester.adx_threshold and atr >= atr_avg * backtester.atr_expansion_mult:
                    htf_trend = backtester.get_h1_trend(df, i)
                    if htf_trend == 'BUY' and plus_di > minus_di:
                        position = {'entry_price': close, 'direction': 'BUY'}
                    elif htf_trend == 'SELL' and minus_di > plus_di:
                        position = {'entry_price': close, 'direction': 'SELL'}

            # Track DD
            if backtester.balance > peak_balance:
                peak_balance = backtester.balance
            dd = (peak_balance - backtester.balance) / peak_balance * 100
            if dd > max_dd:
                max_dd = dd

        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        pnl = backtester.balance - start_balance
        return_pct = pnl / start_balance * 100
        risk_per_trade = dollars_per_pip * backtester.sl_pips

        print(f"${dollars_per_pip}/pip   ${backtester.balance:>10.2f} ${pnl:>+10.2f} {return_pct:>+9.1f}% "
              f"{total_trades:>8} {wins:>6} {losses:>6} {win_rate:>7.1f}% {max_dd:>7.1f}% ${risk_per_trade:>10.0f}")

        all_results.append({
            'dollars_per_pip': dollars_per_pip,
            'balance': backtester.balance,
            'pnl': pnl,
            'return_pct': return_pct,
            'trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_dd': max_dd,
            'risk_per_trade': risk_per_trade
        })

    print("=" * 80)

    # Now run one more time to get trade sequence for $5/pip
    print("\n" + "=" * 80)
    print("TRADE SEQUENCE ANALYSIS (checking consecutive losses)")
    print("=" * 80)

    backtester = SignalBacktester(symbol, start_balance=start_balance)
    backtester.dollars_per_pip = 5  # Use $5/pip for analysis

    adx_series, plus_di_series, minus_di_series = backtester.calculate_adx(df, 14)
    atr_series = backtester.calculate_atr(df, 14)
    atr_avg_series = atr_series.rolling(window=20).mean()

    position = None
    trailing_active = False
    peak_profit_pips = 0.0
    trailing_sl = None
    trade_results = []  # Track W/L sequence

    for i in range(200, len(df) - 1):
        current_bar = df.iloc[i]
        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']

        adx = adx_series.iloc[i]
        plus_di = plus_di_series.iloc[i]
        minus_di = minus_di_series.iloc[i]
        atr = atr_series.iloc[i]
        atr_avg = atr_avg_series.iloc[i]

        if pd.isna(adx) or pd.isna(atr) or pd.isna(atr_avg):
            continue

        if position is not None:
            entry_price = position['entry_price']
            direction = position['direction']

            if direction == 'BUY':
                high_profit_pips = (high - entry_price) / backtester.pip_value
                low_profit_pips = (low - entry_price) / backtester.pip_value
            else:
                high_profit_pips = (entry_price - low) / backtester.pip_value
                low_profit_pips = (entry_price - high) / backtester.pip_value

            exit_reason = None
            exit_pips = 0

            if high_profit_pips >= backtester.tp_pips:
                exit_reason = 'TP'
                exit_pips = backtester.tp_pips
            elif low_profit_pips <= -backtester.sl_pips:
                if trailing_active and trailing_sl is not None:
                    if (direction == 'BUY' and low <= trailing_sl) or (direction == 'SELL' and high >= trailing_sl):
                        exit_reason = 'TRAIL'
                        exit_pips = (trailing_sl - entry_price) / backtester.pip_value if direction == 'BUY' else (entry_price - trailing_sl) / backtester.pip_value
                    else:
                        exit_reason = 'SL'
                        exit_pips = -backtester.sl_pips
                else:
                    exit_reason = 'SL'
                    exit_pips = -backtester.sl_pips
            elif trailing_active and trailing_sl is not None:
                if (direction == 'BUY' and low <= trailing_sl) or (direction == 'SELL' and high >= trailing_sl):
                    exit_reason = 'TRAIL'
                    exit_pips = (trailing_sl - entry_price) / backtester.pip_value if direction == 'BUY' else (entry_price - trailing_sl) / backtester.pip_value

            if exit_reason is None and trailing_active and high_profit_pips > peak_profit_pips:
                peak_profit_pips = high_profit_pips
                locked_pips = peak_profit_pips - backtester.trailing_distance
                if direction == 'BUY':
                    new_sl = entry_price + (locked_pips * backtester.pip_value)
                    if trailing_sl is None or new_sl > trailing_sl:
                        trailing_sl = new_sl
                else:
                    new_sl = entry_price - (locked_pips * backtester.pip_value)
                    if trailing_sl is None or new_sl < trailing_sl:
                        trailing_sl = new_sl

            if exit_reason is None and not trailing_active and high_profit_pips >= backtester.trailing_activation:
                trailing_active = True
                peak_profit_pips = high_profit_pips
                locked_pips = peak_profit_pips - backtester.trailing_distance
                if direction == 'BUY':
                    trailing_sl = entry_price + (locked_pips * backtester.pip_value)
                else:
                    trailing_sl = entry_price - (locked_pips * backtester.pip_value)

            if exit_reason:
                pnl = exit_pips * 5  # $5/pip
                result = 'W' if pnl > 0 else 'L'
                trade_results.append({'result': result, 'pnl': pnl, 'reason': exit_reason})
                position = None
                trailing_active = False
                peak_profit_pips = 0.0
                trailing_sl = None

        if position is None:
            if adx >= backtester.adx_threshold and atr >= atr_avg * backtester.atr_expansion_mult:
                htf_trend = backtester.get_h1_trend(df, i)
                if htf_trend == 'BUY' and plus_di > minus_di:
                    position = {'entry_price': close, 'direction': 'BUY'}
                elif htf_trend == 'SELL' and minus_di > plus_di:
                    position = {'entry_price': close, 'direction': 'SELL'}

    # Analyze consecutive losses
    sequence = ''.join([t['result'] for t in trade_results])
    print(f"\nTrade Sequence (60 trades):")

    # Print in rows of 20
    for i in range(0, len(sequence), 20):
        chunk = sequence[i:i+20]
        print(f"  Trades {i+1:2d}-{min(i+20, len(sequence)):2d}: {chunk}")

    # Show details of the losing streak (trades 10-13)
    print(f"\n--- ANALYZING THE 4-LOSS STREAK (Trades 10-13) ---")
    loss_trades = []
    for idx, t in enumerate(trade_results[9:13], start=10):
        print(f"  Trade {idx}: {t['result']} | Exit: {t['reason']} | P&L: ${t['pnl']:+.0f}")
        loss_trades.append(t)

    # Find max consecutive losses
    max_consec_loss = 0
    current_consec = 0
    loss_streaks = []

    for i, r in enumerate(sequence):
        if r == 'L':
            current_consec += 1
        else:
            if current_consec > 0:
                loss_streaks.append((i - current_consec + 1, current_consec))
            if current_consec > max_consec_loss:
                max_consec_loss = current_consec
            current_consec = 0

    if current_consec > max_consec_loss:
        max_consec_loss = current_consec
    if current_consec > 0:
        loss_streaks.append((len(sequence) - current_consec + 1, current_consec))

    # Find max consecutive wins
    max_consec_win = 0
    current_consec = 0

    for r in sequence:
        if r == 'W':
            current_consec += 1
        else:
            if current_consec > max_consec_win:
                max_consec_win = current_consec
            current_consec = 0
    if current_consec > max_consec_win:
        max_consec_win = current_consec

    print(f"\nStreak Analysis:")
    print(f"  Max Consecutive WINS:   {max_consec_win}")
    print(f"  Max Consecutive LOSSES: {max_consec_loss}")

    # Show where the big loss streaks happened
    big_streaks = [s for s in loss_streaks if s[1] >= 3]
    if big_streaks:
        print(f"\n  Loss Streaks of 3+:")
        for start, count in big_streaks:
            loss_amount = count * 100  # $100 per loss at $5/pip
            print(f"    Trades {start}-{start+count-1}: {count} losses in a row = -${loss_amount}")

    print(f"\nDrawdown Explanation:")
    print(f"  At $5/pip with 20 pip SL = $100 risk per trade")
    print(f"  Max {max_consec_loss} consecutive losses = -${max_consec_loss * 100} potential DD")
    print(f"  But actual max DD was 48% because losses were spread out")
    print(f"  and wins in between recovered some losses")

    print("=" * 80)

    # ========================================================================
    # SAFETY MEASURES SIMULATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAFETY MEASURES - WHAT IF THIS HAPPENED ON DAY 1?")
    print("=" * 80)

    print("\nPROBLEM: Starting with $500, if you hit 4 losses on day 1:")
    print(f"  At $5/pip: 4 x $100 = -$400 = 80% of account GONE!")
    print(f"  At $2/pip: 4 x $40  = -$160 = 32% of account gone")
    print(f"  At $1/pip: 4 x $20  = -$80  = 16% of account gone")

    print("\n" + "-" * 80)
    print("SAFETY MEASURE #1: DAILY LOSS LIMIT")
    print("-" * 80)
    print("  Stop trading for the day after X% loss")
    print("  Example: 10% daily limit = stop after losing $50 on $500 account")
    print("  At $5/pip: Stop after 0.5 losses (basically 1 loss)")
    print("  At $2/pip: Stop after 1.25 losses (stop after 1-2 losses)")
    print("  At $1/pip: Stop after 2.5 losses (stop after 2-3 losses)")

    print("\n" + "-" * 80)
    print("SAFETY MEASURE #2: CONSECUTIVE LOSS LIMIT")
    print("-" * 80)
    print("  Stop trading after N consecutive losses")
    print("  Example: Stop after 2 consecutive losses, wait for next session")
    print("  This would have stopped the bleeding at -$200 instead of -$400")

    print("\n" + "-" * 80)
    print("SAFETY MEASURE #3: PROGRESSIVE POSITION SIZING")
    print("-" * 80)
    print("  Reduce position size after each loss:")
    print("  Trade 1: $2/pip (normal)")
    print("  After 1 loss: $1.5/pip (reduce 25%)")
    print("  After 2 losses: $1/pip (reduce 50%)")
    print("  After win: Reset to $2/pip")
    print("  ")
    print("  4 losses would cost: $40 + $30 + $20 + $20 = $110 instead of $160")

    print("\n" + "-" * 80)
    print("SAFETY MEASURE #4: ACCOUNT SIZE THRESHOLD")
    print("-" * 80)
    print("  CRITICAL: $500 is TOO SMALL for $5/pip!")
    print("  ")
    print("  Professional Risk: 1-2% per trade max")
    print("  $500 x 2% = $10 risk per trade")
    print("  With 20 pip SL: $10 / 20 = $0.50/pip max!")
    print("  ")
    print("  For $1/pip ($20 risk): Need $1,000+ account")
    print("  For $2/pip ($40 risk): Need $2,000+ account")
    print("  For $5/pip ($100 risk): Need $5,000+ account")

    print("\n" + "-" * 80)
    print("RECOMMENDED SETTINGS FOR $500 ACCOUNT:")
    print("-" * 80)
    print("  Position Size:    $0.50 - $1/pip max")
    print("  Daily Loss Limit: 6% ($30)")
    print("  Consec Loss Stop: 3 losses = take a break")
    print("  ")
    print("  With $1/pip:")
    print("    4 losses = -$80 (16% DD) - painful but recoverable")
    print("    Worst case day: -$60 (3 losses then stop)")
    print("=" * 80)

    return

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
