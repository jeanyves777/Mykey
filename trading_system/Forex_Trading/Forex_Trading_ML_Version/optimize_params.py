"""
Parameter Optimization for Signal-Based Strategy
=================================================

Goal: Find parameters that achieve 50%+ monthly profit
Tests multiple combinations of:
- TP/SL ratios
- ADX thresholds
- ATR expansion multipliers
- Trailing stop settings
- Dollars per pip

Usage:
    cd C:/Users/Jean-Yves/thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.optimize_params
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import product


class ParameterOptimizer:
    """Optimize trading parameters for maximum monthly profit."""

    def __init__(self, symbol: str, start_balance: float = 500.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.pip_value = 0.01 if 'JPY' in symbol else 0.0001

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
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

    def get_h1_trend(self, df: pd.DataFrame, idx: int, ema_fast: int = 50, ema_slow: int = 200) -> str:
        if idx < ema_slow:
            return 'NONE'
        closes = df['close'].iloc[:idx+1]
        ema_f = self.calculate_ema(closes, ema_fast).iloc[-1]
        ema_s = self.calculate_ema(closes, ema_slow).iloc[-1]
        current_price = df['close'].iloc[idx]

        if ema_f > ema_s and current_price > ema_f:
            return 'BUY'
        elif ema_f < ema_s and current_price < ema_f:
            return 'SELL'
        return 'NONE'

    def run_backtest(self, df: pd.DataFrame, params: dict, session_hours: tuple = (0, 24)) -> dict:
        """Run backtest with given parameters and session filter."""
        tp_pips = params['tp_pips']
        sl_pips = params['sl_pips']
        trailing_activation = params['trailing_activation']
        trailing_distance = params['trailing_distance']
        adx_threshold = params['adx_threshold']
        atr_expansion_mult = params['atr_expansion_mult']
        dollars_per_pip = params['dollars_per_pip']
        max_losses_per_day = params.get('max_losses_per_day', 999)  # Safety limit
        session_start, session_end = session_hours  # Trading hours filter

        balance = self.start_balance
        peak_balance = self.start_balance

        # Pre-calculate indicators
        adx_series, plus_di_series, minus_di_series = self.calculate_adx(df, 14)
        atr_series = self.calculate_atr(df, 14)
        atr_avg_series = atr_series.rolling(window=20).mean()

        position = None
        trailing_active = False
        peak_profit_pips = 0.0
        trailing_sl = None

        wins = 0
        losses = 0
        total_pnl = 0
        max_drawdown = 0

        # Daily loss tracking
        current_day = None
        daily_losses = 0
        trading_paused = False
        days_paused = 0

        for i in range(200, len(df) - 1):
            current_bar = df.iloc[i]
            bar_date = current_bar.name.date()
            high = current_bar['high']
            low = current_bar['low']
            close = current_bar['close']

            # Reset daily loss counter on new day
            if bar_date != current_day:
                current_day = bar_date
                daily_losses = 0
                trading_paused = False

            adx = adx_series.iloc[i]
            plus_di = plus_di_series.iloc[i]
            minus_di = minus_di_series.iloc[i]
            atr = atr_series.iloc[i]
            atr_avg = atr_avg_series.iloc[i]

            if pd.isna(adx) or pd.isna(atr) or pd.isna(atr_avg):
                continue

            # Check exits if in position
            if position is not None:
                entry_price = position['entry_price']
                direction = position['direction']

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

                # Check TP
                if high_profit_pips >= tp_pips:
                    exit_reason = 'TP'
                    if direction == 'BUY':
                        exit_price = entry_price + (tp_pips * self.pip_value)
                    else:
                        exit_price = entry_price - (tp_pips * self.pip_value)

                # Check SL
                elif low_profit_pips <= -sl_pips:
                    if trailing_active and trailing_sl is not None:
                        if direction == 'BUY' and low <= trailing_sl:
                            exit_reason = 'TRAIL'
                            exit_price = trailing_sl
                        elif direction == 'SELL' and high >= trailing_sl:
                            exit_reason = 'TRAIL'
                            exit_price = trailing_sl
                        else:
                            exit_reason = 'SL'
                            exit_price = entry_price - (sl_pips * self.pip_value) if direction == 'BUY' else entry_price + (sl_pips * self.pip_value)
                    else:
                        exit_reason = 'SL'
                        exit_price = entry_price - (sl_pips * self.pip_value) if direction == 'BUY' else entry_price + (sl_pips * self.pip_value)

                # Check trailing stop
                elif trailing_active and trailing_sl is not None:
                    if direction == 'BUY' and low <= trailing_sl:
                        exit_reason = 'TRAIL'
                        exit_price = trailing_sl
                    elif direction == 'SELL' and high >= trailing_sl:
                        exit_reason = 'TRAIL'
                        exit_price = trailing_sl

                # Update trailing stop
                if exit_reason is None and trailing_active:
                    if high_profit_pips > peak_profit_pips:
                        peak_profit_pips = high_profit_pips
                        locked_pips = peak_profit_pips - trailing_distance
                        if direction == 'BUY':
                            new_trailing_sl = entry_price + (locked_pips * self.pip_value)
                            if trailing_sl is None or new_trailing_sl > trailing_sl:
                                trailing_sl = new_trailing_sl
                        else:
                            new_trailing_sl = entry_price - (locked_pips * self.pip_value)
                            if trailing_sl is None or new_trailing_sl < trailing_sl:
                                trailing_sl = new_trailing_sl

                # Activate trailing
                if exit_reason is None and not trailing_active:
                    if high_profit_pips >= trailing_activation:
                        trailing_active = True
                        peak_profit_pips = high_profit_pips
                        locked_pips = peak_profit_pips - trailing_distance
                        if direction == 'BUY':
                            trailing_sl = entry_price + (locked_pips * self.pip_value)
                        else:
                            trailing_sl = entry_price - (locked_pips * self.pip_value)

                # Close position
                if exit_reason is not None:
                    if direction == 'BUY':
                        pnl_pips = (exit_price - entry_price) / self.pip_value
                    else:
                        pnl_pips = (entry_price - exit_price) / self.pip_value

                    pnl_dollars = pnl_pips * dollars_per_pip
                    balance += pnl_dollars
                    total_pnl += pnl_dollars

                    if pnl_dollars > 0:
                        wins += 1
                    else:
                        losses += 1
                        daily_losses += 1
                        # Check if daily loss limit hit
                        if daily_losses >= max_losses_per_day:
                            trading_paused = True
                            days_paused += 1

                    position = None
                    trailing_active = False
                    peak_profit_pips = 0.0
                    trailing_sl = None

            # Check for entry (only if not paused and within session hours)
            bar_hour = current_bar.name.hour
            in_session = (session_start <= bar_hour < session_end) if session_start < session_end else (bar_hour >= session_start or bar_hour < session_end)

            if position is None and not trading_paused and in_session:
                # Signal conditions - STRENGTHENED
                # 1. ADX must be above threshold (strong trend)
                # 2. ATR must show volatility expansion
                # 3. DI difference must be significant (>5 = strong directional move)
                di_diff = abs(plus_di - minus_di)
                min_di_diff = 5.0  # Minimum DI difference for strong signal

                if adx >= adx_threshold and atr >= atr_avg * atr_expansion_mult and di_diff >= min_di_diff:
                    htf_trend = self.get_h1_trend(df, i)
                    if htf_trend == 'BUY' and plus_di > minus_di:
                        position = {
                            'entry_price': close,
                            'direction': 'BUY',
                            'units': dollars_per_pip / self.pip_value
                        }
                    elif htf_trend == 'SELL' and minus_di > plus_di:
                        position = {
                            'entry_price': close,
                            'direction': 'SELL',
                            'units': dollars_per_pip / self.pip_value
                        }

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100
            if dd > max_drawdown:
                max_drawdown = dd

        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        return_pct = (balance - self.start_balance) / self.start_balance * 100

        return {
            'balance': balance,
            'return_pct': return_pct,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'days_paused': days_paused,
            'params': params
        }


def load_historical_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Load historical data from HistData CSV files."""
    symbol_map = {
        'EUR_USD': 'EURUSD',
        'GBP_USD': 'GBPUSD',
        'USD_JPY': 'USDJPY',
        'USD_CHF': 'USDCHF',
        'USD_CAD': 'USDCAD',
    }

    histdata_symbol = symbol_map.get(symbol, symbol.replace('_', ''))
    data_dir = Path(__file__).parent.parent / "Backtesting_data_histdata" / "2024"
    csv_file = data_dir / f"DAT_MT_{histdata_symbol}_M1_2024.csv"

    if not csv_file.exists():
        return None

    try:
        df = pd.read_csv(
            csv_file,
            names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates={'datetime': ['date', 'time']},
            date_format='%Y.%m.%d %H:%M'
        )
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Use M1 data directly for scalping (no resampling!)
        # Filter to last N days
        if days > 0 and days < 365:
            cutoff = df.index[-1] - timedelta(days=days)
            df = df[df.index > cutoff]

        print(f"  {symbol}: Loaded {len(df):,} M1 candles")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def main():
    """Run parameter optimization with $5/pip, NO loss limit, STRONGER signals."""
    print("=" * 80)
    print("M1 SCALPING BACKTEST - $5/PIP + STRONG SIGNALS")
    print("=" * 80)
    print("Timeframe: M1 (1-minute candles) - REAL SCALPING")
    print("Goal: Quick entries and exits with tight TP/SL")
    print("=" * 80)

    # Trading sessions (UTC hours) - focus on best ones
    sessions = {
        'ALL_DAY': (0, 24),      # Trade any time
        'LONDON': (8, 16),       # 08:00 - 16:00 UTC (best win rate)
        'LONDON_NY': (8, 21),    # London + NY overlap
    }

    # Parameter ranges - STRONGER SIGNAL FILTERS
    param_grid = {
        'tp_pips': [40, 50, 60, 70, 80],      # Higher TP for stronger signals
        'sl_pips': [15, 20, 25],
        'trailing_activation': [10, 15, 20],   # Later activation = more confirmed move
        'trailing_distance': [5, 8],
        'adx_threshold': [25, 30, 35, 40],     # STRICTER ADX (stronger trend)
        'atr_expansion_mult': [1.5, 2.0, 2.5], # STRICTER ATR (bigger volatility)
        'dollars_per_pip': [5],  # FIXED $5/pip
        'max_losses_per_day': [999],  # NO LIMIT
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    total_combos = len(combinations)

    print(f"\nTesting {total_combos} param combos × {len(sessions)} sessions = {total_combos * len(sessions)} total")
    print(f"Fixed: $5/pip | STRONGER signals (ADX 25-40, ATR 1.5-2.5x)")
    print("-" * 80)

    # Test on TOP 4 PROFITABLE PAIRS only
    symbols = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CAD']
    days = 7  # 1 week of M1 data = ~10,000 candles per pair

    # SCALPING parameters for M1 timeframe
    best_params = {
        'tp_pips': 15,           # Scalping: smaller TP (15 pips)
        'sl_pips': 10,           # Scalping: tight SL (10 pips)
        'trailing_activation': 8, # Activate trail at +8 pips
        'trailing_distance': 3,   # Lock in profit at 3 pips behind
        'adx_threshold': 25,      # ADX > 25 for trend
        'atr_expansion_mult': 1.2, # ATR > 1.2x average
        'dollars_per_pip': 5,
        'max_losses_per_day': 999,
    }

    print(f"\nTesting M1 SCALPING parameters on 4 PAIRS (7 days)...")
    print(f"Parameters: TP:{best_params['tp_pips']}p SL:{best_params['sl_pips']}p ADX>{best_params['adx_threshold']} ATR>{best_params['atr_expansion_mult']}x DI>5")
    print("-" * 90)

    all_results = []
    total_starting = 0
    total_ending = 0

    for symbol in symbols:
        df = load_historical_data(symbol, days=days)

        if df is None or len(df) < 200:
            print(f"  {symbol}: Skipping - insufficient data")
            continue

        optimizer = ParameterOptimizer(symbol, start_balance=500.0)
        result = optimizer.run_backtest(df, best_params, (0, 24))  # ALL_DAY
        result['symbol'] = symbol

        all_results.append(result)
        total_starting += 500
        total_ending += result['balance']

    # Display results by pair
    print(f"\n{'='*90}")
    print("STRONG SIGNAL RESULTS - ALL 8 PAIRS ($5/pip)")
    print(f"{'='*90}")
    print(f"{'Pair':<10} {'End Bal':>10} {'Return':>10} {'MaxDD':>8} {'Trades':>7} {'W/L':>8} {'Win%':>7}")
    print("-" * 90)

    # Sort by return
    all_results.sort(key=lambda x: x['return_pct'], reverse=True)

    profitable = 0
    losing = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0

    for r in all_results:
        status = "PROFIT" if r['return_pct'] > 0 else "LOSS"
        print(f"{r['symbol']:<10} ${r['balance']:>9.2f} {r['return_pct']:>+9.1f}% {r['max_drawdown']:>7.1f}% "
              f"{r['total_trades']:>7} {r['wins']:>3}/{r['losses']:<3} {r['win_rate']:>6.1f}% {status}")

        if r['return_pct'] > 0:
            profitable += 1
        else:
            losing += 1
        total_trades += r['total_trades']
        total_wins += r['wins']
        total_losses += r['losses']

    # Portfolio summary
    print(f"\n{'='*90}")
    print("PORTFOLIO SUMMARY - STRONG SIGNALS")
    print(f"{'='*90}")

    total_pnl = total_ending - total_starting
    portfolio_return = total_pnl / total_starting * 100
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    avg_max_dd = sum(r['max_drawdown'] for r in all_results) / len(all_results)

    print(f"\n  Pairs Tested:      {len(all_results)}")
    print(f"  Profitable:        {profitable}")
    print(f"  Losing:            {losing}")
    print(f"\n  Starting Capital:  ${total_starting:,.2f} (${500} × {len(all_results)} pairs)")
    print(f"  Ending Capital:    ${total_ending:,.2f}")
    print(f"  Net P&L:           ${total_pnl:+,.2f}")
    print(f"  Portfolio Return:  {portfolio_return:+.1f}%")
    print(f"\n  Total Trades:      {total_trades}")
    print(f"  Total Wins:        {total_wins}")
    print(f"  Total Losses:      {total_losses}")
    print(f"  Overall Win Rate:  {overall_win_rate:.1f}%")
    print(f"  Avg Max Drawdown:  {avg_max_dd:.1f}%")

    # Show the parameters used
    print(f"\n{'='*90}")
    print("PARAMETERS USED (STRONG SIGNALS)")
    print(f"{'='*90}")
    print(f"  Take Profit:       {best_params['tp_pips']} pips")
    print(f"  Stop Loss:         {best_params['sl_pips']} pips")
    print(f"  Trailing Start:    +{best_params['trailing_activation']} pips")
    print(f"  Trailing Distance: {best_params['trailing_distance']} pips")
    print(f"  ADX Threshold:     >{best_params['adx_threshold']} (STRONG)")
    print(f"  ATR Expansion:     >{best_params['atr_expansion_mult']}x avg")
    print(f"  DI Difference:     >5 (STRONG)")
    print(f"  Position Size:     $5/pip")
    print(f"  Daily Loss Limit:  NONE")

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
