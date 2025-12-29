"""
COMPREHENSIVE STRATEGY PARAMETER OPTIMIZER
==========================================
Optimizes TP, SL, and strategy parameters per pair using HistData.
Finds the best parameter combinations for maximum profitability.

Usage:
    py optimize_strategy_params.py                    # Optimize all pairs
    py optimize_strategy_params.py --pair GBP_USD    # Optimize specific pair
    py optimize_strategy_params.py --pair USD_JPY --strategies RSI_30_70,STRONG_TREND
"""

import sys
import os
import argparse
from itertools import product
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

sys.path.insert(0, os.path.abspath('.'))

# Import shared config for pip multipliers
from trading_system.Forex_Trading.config import optimized_paper_config as config

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directory
HISTDATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_histdata\2024')

# Pair to HistData filename mapping
PAIR_TO_HISTDATA = {
    'EUR_USD': 'EURUSD',
    'GBP_USD': 'GBPUSD',
    'USD_JPY': 'USDJPY',
    'USD_CHF': 'USDCHF',
    'USD_CAD': 'USDCAD',
    'NZD_USD': 'NZDUSD',
    'GBP_JPY': 'GBPJPY',
    'AUD_JPY': 'AUDJPY',
    'EUR_JPY': 'EURJPY',
    'EUR_GBP': 'EURGBP',
    'AUD_CHF': 'AUDCHF',
    'EUR_CAD': 'EURCAD',
}

# Pip multipliers for new pairs
PIP_MULTIPLIERS = {
    'EUR_USD': 10000,
    'GBP_USD': 10000,
    'USD_JPY': 100,
    'USD_CHF': 10000,
    'USD_CAD': 10000,
    'NZD_USD': 10000,
    'GBP_JPY': 100,
    'AUD_JPY': 100,
    'EUR_JPY': 100,
    'EUR_GBP': 10000,
    'AUD_CHF': 10000,
    'EUR_CAD': 10000,
}

# Spread simulation (OANDA typical)
SPREAD_PIPS = {
    'EUR_USD': 1.0,
    'GBP_USD': 1.2,
    'USD_JPY': 1.0,
    'USD_CHF': 1.5,
    'USD_CAD': 1.2,
    'NZD_USD': 1.5,
    'GBP_JPY': 2.0,
    'AUD_JPY': 1.8,
    'EUR_JPY': 1.5,
    'EUR_GBP': 1.5,
    'AUD_CHF': 2.5,
    'EUR_CAD': 2.0,
}

# ============================================================================
# PARAMETER RANGES TO TEST
# ============================================================================

# Take Profit ranges (in pips)
TP_RANGE = [3, 4, 5, 6, 7, 8, 10, 12, 15]

# Stop Loss ranges (in pips)
SL_RANGE = [10, 15, 20, 25, 30, 35, 40]

# RSI thresholds for RSI_30_70 strategy
RSI_OVERSOLD_RANGE = [25, 30, 35]  # Buy when RSI crosses up through this
RSI_OVERBOUGHT_RANGE = [65, 70, 75]  # Sell when RSI crosses down through this

# RSI zones for STRONG_TREND strategy
RSI_PULLBACK_BUY_RANGE = [(30, 45), (35, 50), (40, 55)]  # (min, max) for buy pullback
RSI_PULLBACK_SELL_RANGE = [(45, 60), (50, 65), (55, 70)]  # (min, max) for sell bounce

# Cooldown periods (in bars/minutes)
COOLDOWN_RANGE = [15, 30, 45, 60]

# Trade timeout (max bars to wait for TP/SL)
TIMEOUT_RANGE = [100, 150, 200, 300]

# Available strategies
STRATEGIES = ['RSI_30_70', 'STRONG_TREND', 'RSI_EXTREME', 'MACD_CROSS']


@dataclass
class OptimizationResult:
    """Stores results of a single parameter combination test"""
    pair: str
    strategy: str
    tp_pips: int
    sl_pips: int
    extra_params: Dict
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pips: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators needed for strategies"""
    df = df.copy()

    # EMAs
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
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Candle properties
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

    # ATR for volatility-based stops
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    return df


def load_pair_data(pair: str) -> Optional[pd.DataFrame]:
    """Load and prepare data for a single pair"""
    histdata_pair = PAIR_TO_HISTDATA.get(pair, pair.replace('_', ''))
    csv_path = HISTDATA_DIR / f"DAT_MT_{histdata_pair}_M1_2024.csv"

    if not csv_path.exists():
        print(f"  ERROR: Data file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, header=None,
                     names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M')
    df = calculate_indicators(df)

    return df


def check_signal(strategy: str, i: int, arrays: Dict, params: Dict) -> Tuple[Optional[str], str]:
    """
    Check for trading signal based on strategy and parameters.

    Args:
        strategy: Strategy name
        i: Current bar index
        arrays: Dict of numpy arrays (rsi, ema9, etc.)
        params: Strategy-specific parameters

    Returns:
        Tuple of (signal, reason) where signal is 'BUY', 'SELL', or None
    """
    rsi = arrays['rsi'][i]
    prev_rsi = arrays['rsi'][i-1]
    is_green = arrays['is_green'][i]
    is_red = arrays['is_red'][i]

    if strategy == 'RSI_30_70':
        oversold = params.get('rsi_oversold', 30)
        overbought = params.get('rsi_overbought', 70)

        # BUY: RSI crosses up through oversold level + green candle
        if prev_rsi < oversold and rsi >= oversold and is_green:
            return 'BUY', f"RSI crossed UP through {oversold}"

        # SELL: RSI crosses down through overbought level + red candle
        if prev_rsi > overbought and rsi <= overbought and is_red:
            return 'SELL', f"RSI crossed DOWN through {overbought}"

    elif strategy == 'STRONG_TREND':
        ema9 = arrays['ema9'][i]
        ema21 = arrays['ema21'][i]
        ema50 = arrays['ema50'][i]

        buy_min, buy_max = params.get('rsi_pullback_buy', (35, 50))
        sell_min, sell_max = params.get('rsi_pullback_sell', (50, 65))

        # Strong uptrend: EMA9 > EMA21 > EMA50
        if ema9 > ema21 > ema50:
            if buy_min <= rsi <= buy_max and is_green:
                return 'BUY', f"Uptrend pullback (RSI: {rsi:.1f})"

        # Strong downtrend: EMA9 < EMA21 < EMA50
        elif ema9 < ema21 < ema50:
            if sell_min <= rsi <= sell_max and is_red:
                return 'SELL', f"Downtrend bounce (RSI: {rsi:.1f})"

    elif strategy == 'RSI_EXTREME':
        # More extreme RSI levels for higher probability
        extreme_oversold = params.get('rsi_extreme_oversold', 20)
        extreme_overbought = params.get('rsi_extreme_overbought', 80)

        if prev_rsi < extreme_oversold and rsi >= extreme_oversold and is_green:
            return 'BUY', f"Extreme oversold bounce (RSI: {rsi:.1f})"

        if prev_rsi > extreme_overbought and rsi <= extreme_overbought and is_red:
            return 'SELL', f"Extreme overbought reversal (RSI: {rsi:.1f})"

    elif strategy == 'MACD_CROSS':
        macd = arrays['macd'][i]
        prev_macd = arrays['macd'][i-1]
        signal_line = arrays['macd_signal'][i]
        prev_signal = arrays['macd_signal'][i-1]

        # MACD crosses above signal line
        if prev_macd < prev_signal and macd >= signal_line and is_green:
            return 'BUY', "MACD crossed above signal"

        # MACD crosses below signal line
        if prev_macd > prev_signal and macd <= signal_line and is_red:
            return 'SELL', "MACD crossed below signal"

    return None, ""


def backtest_params(
    pair: str,
    df: pd.DataFrame,
    strategy: str,
    tp_pips: int,
    sl_pips: int,
    cooldown: int = 30,
    timeout: int = 200,
    extra_params: Dict = None
) -> OptimizationResult:
    """
    Run backtest with specific parameters.

    Returns:
        OptimizationResult with all metrics
    """
    if extra_params is None:
        extra_params = {}

    pip_mult = PIP_MULTIPLIERS.get(pair, config.PIP_MULTIPLIERS.get(pair, 10000))
    spread_pips = SPREAD_PIPS.get(pair, 1.0)

    tp_dist = tp_pips / pip_mult
    sl_dist = sl_pips / pip_mult
    spread_dist = spread_pips / pip_mult

    # Pre-extract numpy arrays
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    arrays = {
        'rsi': df['rsi'].values,
        'ema9': df['ema9'].values,
        'ema21': df['ema21'].values,
        'ema50': df['ema50'].values,
        'macd': df['macd'].values,
        'macd_signal': df['macd_signal'].values,
        'is_green': df['is_green'].values,
        'is_red': df['is_red'].values,
    }

    trades = []
    cooldown_until = 0
    equity_curve = [0]  # Track cumulative P&L for drawdown calculation

    for i in range(50, len(df) - 100):
        if i < cooldown_until:
            continue

        signal, _ = check_signal(strategy, i, arrays, extra_params)

        if signal is None:
            continue

        entry = closes[i]

        # Simulate trade
        trade_closed = False
        max_j = min(i + timeout, len(highs))

        if signal == 'BUY':
            tp_target = entry + tp_dist + spread_dist
            sl_target = entry - sl_dist
            net_win = tp_pips - spread_pips
            net_loss = -sl_pips - spread_pips

            for j in range(i + 1, max_j):
                if highs[j] >= tp_target:
                    trades.append({'result': 'WIN', 'pips': net_win})
                    equity_curve.append(equity_curve[-1] + net_win)
                    cooldown_until = j + cooldown
                    trade_closed = True
                    break
                if lows[j] <= sl_target:
                    trades.append({'result': 'LOSS', 'pips': net_loss})
                    equity_curve.append(equity_curve[-1] + net_loss)
                    cooldown_until = j + cooldown
                    trade_closed = True
                    break
        else:  # SELL
            tp_target = entry - tp_dist - spread_dist
            sl_target = entry + sl_dist
            net_win = tp_pips - spread_pips
            net_loss = -sl_pips - spread_pips

            for j in range(i + 1, max_j):
                if lows[j] <= tp_target:
                    trades.append({'result': 'WIN', 'pips': net_win})
                    equity_curve.append(equity_curve[-1] + net_win)
                    cooldown_until = j + cooldown
                    trade_closed = True
                    break
                if highs[j] >= sl_target:
                    trades.append({'result': 'LOSS', 'pips': net_loss})
                    equity_curve.append(equity_curve[-1] + net_loss)
                    cooldown_until = j + cooldown
                    trade_closed = True
                    break

    # Calculate metrics
    if not trades:
        return OptimizationResult(
            pair=pair, strategy=strategy, tp_pips=tp_pips, sl_pips=sl_pips,
            extra_params=extra_params, trades=0, wins=0, losses=0,
            win_rate=0, total_pips=0, profit_factor=0, avg_win=0, avg_loss=0,
            max_drawdown=0, sharpe_ratio=0
        )

    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100 if trades else 0

    total_pips = sum(t['pips'] for t in trades)
    win_pips = sum(t['pips'] for t in trades if t['pips'] > 0)
    loss_pips = abs(sum(t['pips'] for t in trades if t['pips'] < 0))

    profit_factor = win_pips / loss_pips if loss_pips > 0 else 0
    avg_win = win_pips / wins if wins > 0 else 0
    avg_loss = loss_pips / losses if losses > 0 else 0

    # Max drawdown
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = running_max - equity_arr
    max_drawdown = np.max(drawdowns)

    # Sharpe ratio (simplified, using daily returns)
    pips_arr = np.array([t['pips'] for t in trades])
    sharpe_ratio = np.mean(pips_arr) / np.std(pips_arr) * np.sqrt(252) if np.std(pips_arr) > 0 else 0

    return OptimizationResult(
        pair=pair, strategy=strategy, tp_pips=tp_pips, sl_pips=sl_pips,
        extra_params=extra_params, trades=len(trades), wins=wins, losses=losses,
        win_rate=win_rate, total_pips=total_pips, profit_factor=profit_factor,
        avg_win=avg_win, avg_loss=avg_loss, max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio
    )


def optimize_pair(
    pair: str,
    strategies: List[str] = None,
    tp_range: List[int] = None,
    sl_range: List[int] = None,
    quick_mode: bool = False
) -> List[OptimizationResult]:
    """
    Run parameter optimization for a single pair.

    Args:
        pair: Currency pair
        strategies: List of strategies to test (default: all)
        tp_range: TP values to test
        sl_range: SL values to test
        quick_mode: Use reduced parameter ranges for faster testing

    Returns:
        List of OptimizationResult sorted by total_pips
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING: {pair}")
    print(f"{'='*70}")

    # Load data
    df = load_pair_data(pair)
    if df is None:
        return []

    print(f"  Loaded {len(df):,} bars")

    # Default ranges
    if strategies is None:
        strategies = STRATEGIES
    if tp_range is None:
        tp_range = [4, 5, 6, 8, 10] if quick_mode else TP_RANGE
    if sl_range is None:
        sl_range = [15, 20, 25, 30] if quick_mode else SL_RANGE

    results = []
    total_combinations = 0

    for strategy in strategies:
        # Generate parameter combinations based on strategy
        if strategy == 'RSI_30_70':
            param_combos = []
            for oversold, overbought in product(RSI_OVERSOLD_RANGE, RSI_OVERBOUGHT_RANGE):
                if oversold < overbought:  # Valid combination
                    param_combos.append({
                        'rsi_oversold': oversold,
                        'rsi_overbought': overbought
                    })

        elif strategy == 'STRONG_TREND':
            param_combos = []
            for buy_zone, sell_zone in product(RSI_PULLBACK_BUY_RANGE, RSI_PULLBACK_SELL_RANGE):
                param_combos.append({
                    'rsi_pullback_buy': buy_zone,
                    'rsi_pullback_sell': sell_zone
                })

        elif strategy == 'RSI_EXTREME':
            param_combos = [
                {'rsi_extreme_oversold': 15, 'rsi_extreme_overbought': 85},
                {'rsi_extreme_oversold': 20, 'rsi_extreme_overbought': 80},
                {'rsi_extreme_oversold': 25, 'rsi_extreme_overbought': 75},
            ]

        elif strategy == 'MACD_CROSS':
            param_combos = [{}]  # No extra params for MACD

        else:
            param_combos = [{}]

        # Test all combinations
        combos = list(product(tp_range, sl_range, param_combos))
        total_combinations += len(combos)

        print(f"\n  Strategy: {strategy}")
        print(f"  Testing {len(combos)} combinations...")

        tested = 0
        best_pips = -999999
        best_combo = None

        for tp, sl, extra_params in combos:
            # Skip invalid TP/SL ratios (TP should be reasonable vs SL)
            if tp > sl:  # Risk/reward too aggressive
                continue

            result = backtest_params(
                pair=pair,
                df=df,
                strategy=strategy,
                tp_pips=tp,
                sl_pips=sl,
                extra_params=extra_params
            )

            results.append(result)
            tested += 1

            # Track best so far
            if result.total_pips > best_pips:
                best_pips = result.total_pips
                best_combo = result

            # Progress update every 10 combinations
            if tested % 10 == 0:
                best_info = f"| Best: TP{best_combo.tp_pips}/SL{best_combo.sl_pips} = {best_pips:+.0f}p ({best_combo.win_rate:.1f}%)" if best_combo else ""
                print(f"    [{tested}/{len(combos)}] {best_info}")

    print(f"\n  Total combinations tested: {len(results)}")

    # Sort by total pips (profitability)
    results.sort(key=lambda x: x.total_pips, reverse=True)

    return results


def print_top_results(results: List[OptimizationResult], top_n: int = 20):
    """Print the top N results in a formatted table"""
    if not results:
        print("No results to display")
        return

    print(f"\n{'='*100}")
    print(f"TOP {top_n} PARAMETER COMBINATIONS")
    print(f"{'='*100}")

    header = f"{'#':<3} {'STRATEGY':<14} {'TP':<4} {'SL':<4} {'TRADES':<7} {'WIN%':<7} {'P&L':<10} {'PF':<6} {'DD':<8} {'PARAMS'}"
    print(header)
    print("-" * 100)

    for i, r in enumerate(results[:top_n], 1):
        params_str = str(r.extra_params) if r.extra_params else ""
        if len(params_str) > 30:
            params_str = params_str[:27] + "..."

        print(f"{i:<3} {r.strategy:<14} {r.tp_pips:<4} {r.sl_pips:<4} "
              f"{r.trades:<7} {r.win_rate:<6.1f}% {r.total_pips:>+8.0f}p "
              f"{r.profit_factor:<5.2f} {r.max_drawdown:>7.0f}p {params_str}")

    print("-" * 100)


def print_strategy_comparison(results: List[OptimizationResult]):
    """Print best result for each strategy"""
    print(f"\n{'='*80}")
    print("BEST RESULT PER STRATEGY")
    print(f"{'='*80}")

    strategies_seen = set()

    for r in results:
        if r.strategy not in strategies_seen:
            strategies_seen.add(r.strategy)
            print(f"\n{r.strategy}:")
            print(f"  TP: {r.tp_pips} | SL: {r.sl_pips}")
            print(f"  Trades: {r.trades} | Win Rate: {r.win_rate:.1f}%")
            print(f"  Total P&L: {r.total_pips:+.0f} pips | Profit Factor: {r.profit_factor:.2f}")
            print(f"  Max Drawdown: {r.max_drawdown:.0f} pips")
            if r.extra_params:
                print(f"  Params: {r.extra_params}")


def save_results(results: List[OptimizationResult], pair: str, filename: str = None):
    """Save results to JSON file"""
    if filename is None:
        filename = f"optimization_results_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    data = []
    for r in results:
        data.append({
            'pair': r.pair,
            'strategy': r.strategy,
            'tp_pips': r.tp_pips,
            'sl_pips': r.sl_pips,
            'extra_params': r.extra_params,
            'trades': r.trades,
            'wins': r.wins,
            'losses': r.losses,
            'win_rate': r.win_rate,
            'total_pips': r.total_pips,
            'profit_factor': r.profit_factor,
            'avg_win': r.avg_win,
            'avg_loss': r.avg_loss,
            'max_drawdown': r.max_drawdown,
            'sharpe_ratio': r.sharpe_ratio
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument('--pair', type=str, help="Specific pair to optimize (e.g., GBP_USD)")
    parser.add_argument('--pairs', type=str, help="Comma-separated list of pairs")
    parser.add_argument('--strategies', type=str, help="Comma-separated list of strategies")
    parser.add_argument('--quick', action='store_true', help="Quick mode with reduced ranges")
    parser.add_argument('--top', type=int, default=20, help="Number of top results to show")
    parser.add_argument('--save', action='store_true', help="Save results to JSON file")

    args = parser.parse_args()

    print("=" * 70)
    print("STRATEGY PARAMETER OPTIMIZER")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # Determine pairs to optimize
    if args.pair:
        pairs = [args.pair.upper()]
    elif args.pairs:
        pairs = [p.strip().upper() for p in args.pairs.split(',')]
    else:
        pairs = list(PAIR_TO_HISTDATA.keys())

    # Determine strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip().upper() for s in args.strategies.split(',')]

    print(f"Pairs: {pairs}")
    print(f"Strategies: {strategies or 'ALL'}")
    print(f"Quick mode: {args.quick}")

    # Run optimization for each pair
    all_results = {}

    for pair in pairs:
        if pair not in PAIR_TO_HISTDATA:
            print(f"\nWARNING: Unknown pair {pair}, skipping...")
            continue

        results = optimize_pair(
            pair=pair,
            strategies=strategies,
            quick_mode=args.quick
        )

        all_results[pair] = results

        # Print results for this pair
        print_top_results(results, top_n=args.top)
        print_strategy_comparison(results)

        if args.save:
            save_results(results, pair)

    # Final summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"End time: {datetime.now()}")

    # Print best overall for each pair
    print("\nBEST PARAMETERS PER PAIR:")
    print("-" * 70)

    for pair, results in all_results.items():
        if results:
            best = results[0]
            print(f"\n{pair}:")
            print(f"  Strategy: {best.strategy}")
            print(f"  TP: {best.tp_pips} pips | SL: {best.sl_pips} pips")
            print(f"  Win Rate: {best.win_rate:.1f}% | P&L: {best.total_pips:+.0f} pips")
            print(f"  Profit Factor: {best.profit_factor:.2f}")
            if best.extra_params:
                print(f"  Extra Params: {best.extra_params}")


if __name__ == "__main__":
    main()
