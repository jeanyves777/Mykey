"""
Full Parameter Comparison - All Results for Decision Making
============================================================

Compares multiple parameter sets across all pairs and timeframes.

Usage:
    cd C:/Users/Jean-Yves/thevolumeainative
    py -m trading_system.Forex_Trading.Forex_Trading_ML_Version.full_comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class FullBacktester:
    """Full backtest with configurable parameters."""

    def __init__(self, symbol: str, start_balance: float, params: dict):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.params = params
        self.pip_value = 0.01 if 'JPY' in symbol else 0.0001
        self.trades = []

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

    def get_h1_trend(self, df: pd.DataFrame, idx: int) -> str:
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
        return 'NONE'

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Run backtest with given parameters."""
        p = self.params
        tp_pips = p['tp_pips']
        sl_pips = p['sl_pips']
        trailing_activation = p['trailing_activation']
        trailing_distance = p['trailing_distance']
        adx_threshold = p['adx_threshold']
        atr_expansion_mult = p['atr_expansion_mult']
        dollars_per_pip = p['dollars_per_pip']

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
        tp_exits = 0
        sl_exits = 0
        trail_exits = 0
        peak_balance = self.start_balance
        max_drawdown = 0

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

            # Check exits if in position
            if position is not None:
                entry_price = position['entry_price']
                direction = position['direction']

                if direction == 'BUY':
                    high_profit_pips = (high - entry_price) / self.pip_value
                    low_profit_pips = (low - entry_price) / self.pip_value
                else:
                    high_profit_pips = (entry_price - low) / self.pip_value
                    low_profit_pips = (entry_price - high) / self.pip_value

                exit_reason = None
                exit_price = None

                # Check TP
                if high_profit_pips >= tp_pips:
                    exit_reason = 'TP'
                    tp_exits += 1
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
                            trail_exits += 1
                        elif direction == 'SELL' and high >= trailing_sl:
                            exit_reason = 'TRAIL'
                            exit_price = trailing_sl
                            trail_exits += 1
                        else:
                            exit_reason = 'SL'
                            sl_exits += 1
                            exit_price = entry_price - (sl_pips * self.pip_value) if direction == 'BUY' else entry_price + (sl_pips * self.pip_value)
                    else:
                        exit_reason = 'SL'
                        sl_exits += 1
                        exit_price = entry_price - (sl_pips * self.pip_value) if direction == 'BUY' else entry_price + (sl_pips * self.pip_value)

                # Check trailing stop
                elif trailing_active and trailing_sl is not None:
                    if direction == 'BUY' and low <= trailing_sl:
                        exit_reason = 'TRAIL'
                        exit_price = trailing_sl
                        trail_exits += 1
                    elif direction == 'SELL' and high >= trailing_sl:
                        exit_reason = 'TRAIL'
                        exit_price = trailing_sl
                        trail_exits += 1

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
                    self.balance += pnl_dollars

                    if pnl_dollars > 0:
                        wins += 1
                    else:
                        losses += 1

                    self.trades.append({
                        'pnl_pips': pnl_pips,
                        'pnl_dollars': pnl_dollars,
                        'exit_reason': exit_reason
                    })

                    position = None
                    trailing_active = False
                    peak_profit_pips = 0.0
                    trailing_sl = None

            # Check for entry
            if position is None:
                if adx >= adx_threshold and atr >= atr_avg * atr_expansion_mult:
                    htf_trend = self.get_h1_trend(df, i)
                    if htf_trend == 'BUY' and plus_di > minus_di:
                        position = {
                            'entry_price': close,
                            'direction': 'BUY',
                        }
                    elif htf_trend == 'SELL' and minus_di > plus_di:
                        position = {
                            'entry_price': close,
                            'direction': 'SELL',
                        }

            # Track drawdown
            if self.balance > peak_balance:
                peak_balance = self.balance
            dd = (peak_balance - self.balance) / peak_balance * 100
            if dd > max_drawdown:
                max_drawdown = dd

        total_trades = wins + losses
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        return_pct = (self.balance - self.start_balance) / self.start_balance * 100

        # Calculate profit factor
        winning_trades = [t for t in self.trades if t['pnl_dollars'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_dollars'] <= 0]
        gross_profit = sum(t['pnl_dollars'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl_dollars'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_win = np.mean([t['pnl_dollars'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_dollars'] for t in losing_trades]) if losing_trades else 0

        return {
            'symbol': self.symbol,
            'balance': self.balance,
            'return_pct': return_pct,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'trail_exits': trail_exits,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
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

        df_h1 = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if days > 0 and days < 365:
            cutoff = df_h1.index[-1] - timedelta(days=days)
            df_h1 = df_h1[df_h1.index > cutoff]

        return df_h1
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def main():
    """Run full comparison."""
    print("=" * 100)
    print("FULL PARAMETER COMPARISON - ALL RESULTS FOR YOUR DECISION")
    print("=" * 100)

    # Define parameter sets to test
    param_sets = {
        'CURRENT (Conservative)': {
            'tp_pips': 20, 'sl_pips': 15,
            'trailing_activation': 10, 'trailing_distance': 8,
            'adx_threshold': 25, 'atr_expansion_mult': 1.5,
            'dollars_per_pip': 1
        },
        'OPTIMIZED #1 (Best Return)': {
            'tp_pips': 50, 'sl_pips': 20,
            'trailing_activation': 8, 'trailing_distance': 5,
            'adx_threshold': 20, 'atr_expansion_mult': 1.2,
            'dollars_per_pip': 5
        },
        'OPTIMIZED #4 (Low DD)': {
            'tp_pips': 40, 'sl_pips': 20,
            'trailing_activation': 8, 'trailing_distance': 5,
            'adx_threshold': 25, 'atr_expansion_mult': 1.2,
            'dollars_per_pip': 5
        },
        'MODERATE RISK': {
            'tp_pips': 40, 'sl_pips': 20,
            'trailing_activation': 10, 'trailing_distance': 5,
            'adx_threshold': 25, 'atr_expansion_mult': 1.2,
            'dollars_per_pip': 3
        },
        'HIGH RISK': {
            'tp_pips': 50, 'sl_pips': 15,
            'trailing_activation': 8, 'trailing_distance': 5,
            'adx_threshold': 20, 'atr_expansion_mult': 1.2,
            'dollars_per_pip': 5
        },
    }

    symbols = ['USD_JPY', 'EUR_USD', 'USD_CHF', 'USD_CAD']
    start_balance = 500.0

    # Test periods
    periods = [30, 60, 90]  # 1, 2, 3 months

    for period_days in periods:
        print(f"\n{'='*100}")
        print(f"BACKTEST PERIOD: {period_days} DAYS ({period_days//30} MONTH{'S' if period_days > 30 else ''})")
        print(f"{'='*100}")

        for param_name, params in param_sets.items():
            print(f"\n{'-'*100}")
            print(f"STRATEGY: {param_name}")
            print(f"TP: {params['tp_pips']}p | SL: {params['sl_pips']}p | "
                  f"Trail: +{params['trailing_activation']}p/{params['trailing_distance']}p | "
                  f"ADX>{params['adx_threshold']} | ATR>{params['atr_expansion_mult']}x | "
                  f"${params['dollars_per_pip']}/pip")
            print(f"{'-'*100}")

            total_return = 0
            total_trades = 0
            total_wins = 0
            results = []

            print(f"\n{'Symbol':<10} {'Return':>10} {'P&L':>12} {'Trades':>8} {'Win%':>8} {'PF':>6} {'MaxDD':>8} {'TP':>5} {'SL':>5} {'Trail':>6}")
            print("-" * 90)

            for symbol in symbols:
                df = load_historical_data(symbol, days=period_days)

                if df is None or len(df) < 200:
                    print(f"{symbol:<10} {'N/A':>10}")
                    continue

                backtester = FullBacktester(symbol, start_balance, params)
                result = backtester.run_backtest(df)
                results.append(result)

                pnl = result['balance'] - start_balance
                total_return += result['return_pct']
                total_trades += result['total_trades']
                total_wins += result['wins']

                print(f"{symbol:<10} {result['return_pct']:>+9.1f}% ${pnl:>+10.2f} "
                      f"{result['total_trades']:>8} {result['win_rate']:>7.1f}% "
                      f"{result['profit_factor']:>5.2f} {result['max_drawdown']:>7.1f}% "
                      f"{result['tp_exits']:>5} {result['sl_exits']:>5} {result['trail_exits']:>6}")

            if results:
                avg_return = total_return / len(results)
                overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
                total_pnl = sum(r['balance'] - start_balance for r in results)
                max_dd = max(r['max_drawdown'] for r in results)

                print("-" * 90)
                print(f"{'PORTFOLIO':<10} {avg_return:>+9.1f}% ${total_pnl:>+10.2f} "
                      f"{total_trades:>8} {overall_win_rate:>7.1f}% "
                      f"{'':>5} {max_dd:>7.1f}%")

    # Summary comparison
    print(f"\n\n{'='*100}")
    print("SUMMARY: MONTHLY (30-DAY) RESULTS COMPARISON")
    print(f"{'='*100}")

    print(f"\n{'Strategy':<30} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>8} {'Risk':>15}")
    print("-" * 85)

    for param_name, params in param_sets.items():
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        max_dd = 0

        for symbol in symbols:
            df = load_historical_data(symbol, days=30)
            if df is None or len(df) < 200:
                continue

            backtester = FullBacktester(symbol, start_balance, params)
            result = backtester.run_backtest(df)
            total_pnl += result['balance'] - start_balance
            total_trades += result['total_trades']
            total_wins += result['wins']
            if result['max_drawdown'] > max_dd:
                max_dd = result['max_drawdown']

        avg_return = total_pnl / (start_balance * len(symbols)) * 100
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

        # Risk level based on $/pip
        risk_level = "LOW" if params['dollars_per_pip'] == 1 else "MEDIUM" if params['dollars_per_pip'] <= 3 else "HIGH"

        print(f"{param_name:<30} {avg_return:>+9.1f}% {total_trades:>8} {win_rate:>7.1f}% {max_dd:>7.1f}% {risk_level:>15}")

    print(f"\n{'='*100}")
    print("RISK ANALYSIS")
    print(f"{'='*100}")
    print("""
Position Sizing Impact (with $500 account):

  $1/pip = Risk $15-20 per trade (3-4% of account) = LOW RISK
  $3/pip = Risk $45-60 per trade (9-12% of account) = MEDIUM RISK
  $5/pip = Risk $75-100 per trade (15-20% of account) = HIGH RISK

Key Considerations:
  - Higher $/pip = Higher returns BUT higher drawdowns
  - Lower ADX threshold = More signals BUT lower quality
  - Wider SL = Fewer stopouts BUT larger losses when hit
  - Tight trailing = Locks profits BUT may exit early

Recommendations:
  - Conservative: $1/pip, ADX>25, tight TP/SL
  - Balanced: $3/pip, ADX>25, wider TP
  - Aggressive: $5/pip, ADX>20, wide TP, tight trailing
""")


if __name__ == "__main__":
    main()
