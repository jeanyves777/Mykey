"""
Enhanced Backtest with Daily Performance & Volatility Analysis
Tracks:
- Daily win rate and P&L for each pair
- Volatility conditions during entries (ATR, spread, candle size)
- Volume/momentum indicators at entry
- Identifies bad days vs good days patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Add trading system to path
sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading'))
from config import optimized_paper_config as config

# ============================================================================
# CONFIGURATION
# ============================================================================
HISTDATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024')

PAIR_TO_HISTDATA = {
    'EUR_USD': 'EURUSD',
    'GBP_USD': 'GBPUSD',
    'USD_JPY': 'USDJPY',
    'USD_CHF': 'USDCHF',
    'USD_CAD': 'USDCAD',
    'NZD_USD': 'NZDUSD',
    'AUD_JPY': 'AUDJPY',
}

# Spread simulation (OANDA typical)
SPREAD_PIPS = {
    'EUR_USD': 1.0,
    'GBP_USD': 1.2,
    'USD_JPY': 1.0,
    'USD_CHF': 1.5,
    'USD_CAD': 1.2,
    'NZD_USD': 1.5,
    'AUD_JPY': 1.8,
}

# ============================================================================
# DATA LOADING WITH VOLATILITY INDICATORS
# ============================================================================
def load_data_with_volatility(pair: str) -> pd.DataFrame:
    """Load data and calculate all indicators including volatility metrics."""
    histdata_name = PAIR_TO_HISTDATA.get(pair)
    if not histdata_name:
        return pd.DataFrame()

    # Find data file (HistData format: DAT_MT_EURUSD_M1_2024.csv)
    data_file = None
    for f in HISTDATA_DIR.glob(f'DAT_MT_{histdata_name}_M1_*.csv'):
        data_file = f
        break

    if not data_file:
        print(f"  No data file found for {pair}")
        return pd.DataFrame()

    print(f"  Loading {data_file.name}...")

    # Load CSV (HistData MT format: date, time, open, high, low, close, volume)
    # Format: 2024.01.01,17:00,1.104270,1.104290,1.104250,1.104290,0
    df = pd.read_csv(data_file, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df = df.drop(columns=['date', 'time'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Loaded {len(df):,} bars")

    # Calculate standard indicators
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Candle properties
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']

    # ============================================================================
    # VOLATILITY INDICATORS
    # ============================================================================

    # ATR (14-period Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # ATR as percentage of price (normalized volatility)
    df['atr_pct'] = (df['atr'] / df['close']) * 100

    # Volatility regime (rolling std of returns)
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std() * 100

    # Volume indicators (if available)
    if df['volume'].sum() > 0:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    else:
        df['volume_sma'] = 1
        df['volume_ratio'] = 1

    # Momentum (rate of change)
    df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
    df['momentum_10'] = (df['close'] / df['close'].shift(10) - 1) * 100

    # Bollinger Band width (volatility measure)
    df['bb_sma'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_width'] = (df['bb_std'] * 4) / df['bb_sma'] * 100  # Width as % of price

    # Hour of day (for session analysis)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date

    return df


def run_backtest_with_analysis(pair: str, df: pd.DataFrame) -> dict:
    """
    Run backtest and collect detailed trade data including volatility conditions.
    """
    if len(df) < 200:
        return {'trades': [], 'daily': {}}

    settings = config.get_pair_settings(pair)
    pip_mult = config.PIP_MULTIPLIERS[pair]
    spread_pips = SPREAD_PIPS.get(pair, 1.0)

    tp_pips = settings['tp_pips']
    sl_pips = settings['sl_pips']
    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult
    spread_dist = spread_pips / pip_mult

    # Pre-extract arrays for speed
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    timestamps = df['timestamp'].values
    rsi_arr = df['rsi'].values
    is_green_arr = df['is_green'].values
    is_red_arr = df['is_red'].values
    macd_arr = df['macd'].values
    macd_signal_arr = df['macd_signal'].values
    ema9_arr = df['ema9'].values
    ema21_arr = df['ema21'].values
    ema50_arr = df['ema50'].values

    # Volatility arrays
    atr_arr = df['atr'].values
    atr_pct_arr = df['atr_pct'].values
    volatility_arr = df['volatility_20'].values
    volume_ratio_arr = df['volume_ratio'].values
    momentum_5_arr = df['momentum_5'].values
    bb_width_arr = df['bb_width'].values
    candle_range_arr = df['candle_range'].values
    hour_arr = df['hour'].values
    date_arr = df['date'].values

    trades = []
    daily_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl_pips': 0})

    for i in range(50, len(df) - 100):
        # Check signal based on strategy
        signal = None
        rsi = rsi_arr[i]
        prev_rsi = rsi_arr[i-1]

        if settings['strategy'] == 'RSI_30_70':
            oversold = settings.get('rsi_oversold', 30)
            overbought = settings.get('rsi_overbought', 70)

            if prev_rsi < oversold and rsi >= oversold and is_green_arr[i]:
                signal = 'BUY'
            elif prev_rsi > overbought and rsi <= overbought and is_red_arr[i]:
                signal = 'SELL'

        elif settings['strategy'] == 'MACD_CROSS':
            macd = macd_arr[i]
            macd_signal = macd_signal_arr[i]
            prev_macd = macd_arr[i-1]
            prev_macd_signal = macd_signal_arr[i-1]

            if prev_macd <= prev_macd_signal and macd > macd_signal and is_green_arr[i]:
                signal = 'BUY'
            elif prev_macd >= prev_macd_signal and macd < macd_signal and is_red_arr[i]:
                signal = 'SELL'

        elif settings['strategy'] == 'STRONG_TREND':
            ema9 = ema9_arr[i]
            ema21 = ema21_arr[i]
            ema50 = ema50_arr[i]

            if ema9 > ema21 > ema50 and 35 <= rsi <= 50 and is_green_arr[i]:
                signal = 'BUY'
            elif ema9 < ema21 < ema50 and 50 <= rsi <= 65 and is_red_arr[i]:
                signal = 'SELL'

        if signal is None:
            continue

        # Entry with spread cost
        entry = closes[i]
        if signal == 'BUY':
            entry += spread_dist  # Pay spread on buy
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            entry -= spread_dist  # Pay spread on sell
            tp = entry - tp_dist
            sl = entry + sl_dist

        entry_time = pd.Timestamp(timestamps[i])
        entry_date = date_arr[i]

        # Capture volatility conditions at entry
        entry_volatility = {
            'atr': atr_arr[i],
            'atr_pct': atr_pct_arr[i],
            'volatility_20': volatility_arr[i],
            'volume_ratio': volume_ratio_arr[i],
            'momentum_5': momentum_5_arr[i],
            'bb_width': bb_width_arr[i],
            'candle_range': candle_range_arr[i],
            'hour': hour_arr[i],
            'rsi': rsi,
        }

        # Simulate trade outcome
        result = None
        exit_price = entry
        duration_bars = 0

        for j in range(i + 1, min(i + 500, len(df))):
            duration_bars += 1
            high = highs[j]
            low = lows[j]

            if signal == 'BUY':
                if low <= sl:
                    result = 'LOSS'
                    exit_price = sl
                    break
                elif high >= tp:
                    result = 'WIN'
                    exit_price = tp
                    break
            else:
                if high >= sl:
                    result = 'LOSS'
                    exit_price = sl
                    break
                elif low <= tp:
                    result = 'WIN'
                    exit_price = tp
                    break

        if result is None:
            continue

        # Calculate P&L in pips
        if signal == 'BUY':
            pnl_pips = (exit_price - entry) * pip_mult
        else:
            pnl_pips = (entry - exit_price) * pip_mult

        # Record trade
        trade = {
            'entry_time': entry_time,
            'date': entry_date,
            'signal': signal,
            'entry': entry,
            'exit': exit_price,
            'result': result,
            'pnl_pips': pnl_pips,
            'duration_bars': duration_bars,
            **entry_volatility
        }
        trades.append(trade)

        # Update daily stats
        daily_stats[entry_date]['pnl_pips'] += pnl_pips
        if result == 'WIN':
            daily_stats[entry_date]['wins'] += 1
        else:
            daily_stats[entry_date]['losses'] += 1

    return {
        'trades': trades,
        'daily': dict(daily_stats)
    }


def analyze_results(pair: str, results: dict):
    """Analyze backtest results with focus on volatility patterns."""
    trades = results['trades']
    daily = results['daily']

    if not trades:
        print(f"\n{pair}: No trades")
        return

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    print(f"\n{'=' * 80}")
    print(f"{pair} - DETAILED ANALYSIS")
    print(f"{'=' * 80}")

    # Overall stats
    total = len(trades)
    win_rate = len(wins) / total * 100
    total_pips = sum(t['pnl_pips'] for t in trades)

    print(f"\nOVERALL: {total} trades | {len(wins)}W/{len(losses)}L | {win_rate:.1f}% WR | {total_pips:+.1f} pips")

    # =========================================================================
    # DAILY BREAKDOWN
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("DAILY PERFORMANCE (showing worst 10 days)")
    print(f"{'=' * 60}")

    # Sort days by P&L
    sorted_days = sorted(daily.items(), key=lambda x: x[1]['pnl_pips'])

    bad_days = 0
    good_days = 0
    for date, stats in sorted_days:
        if stats['pnl_pips'] < 0:
            bad_days += 1
        else:
            good_days += 1

    print(f"Total Days: {len(sorted_days)} | Good: {good_days} | Bad: {bad_days}")
    print(f"\nWorst Days:")
    print(f"{'DATE':<12} {'TRADES':<8} {'WINS':<6} {'LOSSES':<8} {'WIN%':<8} {'P&L PIPS'}")
    print("-" * 55)

    for date, stats in sorted_days[:10]:  # Worst 10 days
        total_day = stats['wins'] + stats['losses']
        wr = stats['wins'] / total_day * 100 if total_day > 0 else 0
        print(f"{str(date):<12} {total_day:<8} {stats['wins']:<6} {stats['losses']:<8} {wr:<7.1f}% {stats['pnl_pips']:+8.1f}")

    # =========================================================================
    # VOLATILITY ANALYSIS - WINS vs LOSSES
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("VOLATILITY CONDITIONS: WINS vs LOSSES")
    print(f"{'=' * 60}")

    # Helper to calculate average
    def avg(values):
        valid = [v for v in values if v is not None and not np.isnan(v)]
        return np.mean(valid) if valid else 0

    metrics = ['atr_pct', 'volatility_20', 'volume_ratio', 'momentum_5', 'bb_width', 'candle_range']

    print(f"\n{'METRIC':<18} {'WINS AVG':<15} {'LOSSES AVG':<15} {'DIFFERENCE':<12} {'INSIGHT'}")
    print("-" * 75)

    insights = []

    for metric in metrics:
        win_avg = avg([t[metric] for t in wins])
        loss_avg = avg([t[metric] for t in losses])
        diff = win_avg - loss_avg
        diff_pct = (diff / loss_avg * 100) if loss_avg != 0 else 0

        # Generate insight
        insight = ""
        if abs(diff_pct) > 20:
            if metric == 'atr_pct' and diff < 0:
                insight = "Wins in LOWER volatility"
                insights.append(f"Filter: ATR% < {loss_avg:.3f}")
            elif metric == 'volatility_20' and diff < 0:
                insight = "Wins in CALMER markets"
                insights.append(f"Filter: Vol20 < {loss_avg:.4f}")
            elif metric == 'volume_ratio' and diff > 0:
                insight = "Wins need HIGHER volume"
                insights.append(f"Filter: VolumeRatio > {win_avg:.2f}")
            elif metric == 'bb_width' and diff < 0:
                insight = "Wins in TIGHTER BBands"
                insights.append(f"Filter: BBWidth < {loss_avg:.3f}")
            elif metric == 'candle_range' and diff < 0:
                insight = "Wins on SMALLER candles"

        print(f"{metric:<18} {win_avg:<15.4f} {loss_avg:<15.4f} {diff:+11.4f} {insight}")

    # =========================================================================
    # HOUR ANALYSIS
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("PERFORMANCE BY HOUR (UTC)")
    print(f"{'=' * 60}")

    hour_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})
    for t in trades:
        hour = t['hour']
        if t['result'] == 'WIN':
            hour_stats[hour]['wins'] += 1
        else:
            hour_stats[hour]['losses'] += 1

    print(f"\n{'HOUR':<6} {'TRADES':<8} {'WIN%':<8} {'STATUS'}")
    print("-" * 35)

    bad_hours = []
    for hour in sorted(hour_stats.keys()):
        stats = hour_stats[hour]
        total_h = stats['wins'] + stats['losses']
        wr = stats['wins'] / total_h * 100 if total_h > 0 else 0
        status = "AVOID" if wr < 70 and total_h >= 10 else ""
        if status == "AVOID":
            bad_hours.append(hour)
        print(f"{hour:02d}:00  {total_h:<8} {wr:<7.1f}% {status}")

    # =========================================================================
    # RSI AT ENTRY ANALYSIS
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("RSI AT ENTRY: WINS vs LOSSES")
    print(f"{'=' * 60}")

    win_rsi = avg([t['rsi'] for t in wins])
    loss_rsi = avg([t['rsi'] for t in losses])

    print(f"Average RSI at WIN entries:  {win_rsi:.1f}")
    print(f"Average RSI at LOSS entries: {loss_rsi:.1f}")

    # RSI ranges
    rsi_ranges = [(0, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 100)]
    print(f"\n{'RSI RANGE':<12} {'WINS':<8} {'LOSSES':<8} {'WIN%'}")
    print("-" * 40)

    for low, high in rsi_ranges:
        range_wins = len([t for t in wins if low <= t['rsi'] < high])
        range_losses = len([t for t in losses if low <= t['rsi'] < high])
        total_r = range_wins + range_losses
        if total_r > 0:
            wr = range_wins / total_r * 100
            print(f"{low:2d}-{high:2d}       {range_wins:<8} {range_losses:<8} {wr:.1f}%")

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("RECOMMENDED FILTERS")
    print(f"{'=' * 60}")

    if insights:
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight}")

    if bad_hours:
        print(f"  • Avoid hours: {bad_hours}")

    return {
        'pair': pair,
        'win_rate': win_rate,
        'total_pips': total_pips,
        'bad_days': bad_days,
        'good_days': good_days,
        'insights': insights,
        'bad_hours': bad_hours
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED BACKTEST WITH DAILY & VOLATILITY ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")

    # Focus on problematic pairs first, then all
    problem_pairs = ['EUR_USD', 'USD_CAD', 'AUD_JPY']
    other_pairs = [p for p in config.OPTIMIZED_PAIRS if p not in problem_pairs]

    all_results = {}

    print("\n" + "=" * 80)
    print("ANALYZING PROBLEMATIC PAIRS FIRST")
    print("=" * 80)

    for pair in problem_pairs:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing {pair}...")
        df = load_data_with_volatility(pair)
        if len(df) > 0:
            results = run_backtest_with_analysis(pair, df)
            analysis = analyze_results(pair, results)
            all_results[pair] = analysis

    print("\n" + "=" * 80)
    print("ANALYZING OTHER PAIRS FOR COMPARISON")
    print("=" * 80)

    for pair in other_pairs:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing {pair}...")
        df = load_data_with_volatility(pair)
        if len(df) > 0:
            results = run_backtest_with_analysis(pair, df)
            analysis = analyze_results(pair, results)
            all_results[pair] = analysis

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'PAIR':<10} {'WIN%':<8} {'PIPS':<12} {'GOOD DAYS':<12} {'BAD DAYS':<12} {'STATUS'}")
    print("-" * 65)

    for pair in problem_pairs + other_pairs:
        if pair in all_results:
            r = all_results[pair]
            status = "PROBLEM" if r['win_rate'] < 80 else "OK"
            print(f"{pair:<10} {r['win_rate']:<7.1f}% {r['total_pips']:+11.1f} {r['good_days']:<12} {r['bad_days']:<12} {status}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analysis complete!")
