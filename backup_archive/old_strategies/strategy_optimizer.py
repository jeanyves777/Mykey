"""
FOREX STRATEGY OPTIMIZER
========================
Tests multiple strategy configurations to find 70%+ win rate combinations.

Strategies tested:
1. EMA Crossover (different periods)
2. RSI Reversal (different levels)
3. Bollinger Band Bounce
4. MACD Divergence
5. Combined filters

Parameters optimized:
- TP/SL ratios (tighter TP for higher win rate)
- RSI thresholds
- EMA periods
- Entry filters (trend strength, volatility)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = Path(r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data")

# Pairs to test
PAIRS = {
    'EUR_USD': ('EURUSD', 10000),
    'GBP_USD': ('GBPUSD', 10000),
    'USD_JPY': ('USDJPY', 100),
    'USD_CHF': ('USDCHF', 10000),
    'USD_CAD': ('USDCAD', 10000)
}


def load_data(pair: str, timeframe: str = "5") -> pd.DataFrame:
    """Load CSV data for a pair"""
    prefix = PAIRS[pair][0]
    filepath = DATA_DIR / f"{prefix}{timeframe}.csv"

    df = pd.read_csv(filepath, sep='\t',
                     names=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'])
    return df


def calculate_indicators(df: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21,
                         rsi_period: int = 14, bb_period: int = 20) -> pd.DataFrame:
    """Calculate all indicators"""
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    df['ema_trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR for volatility
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Candle patterns
    df['body'] = df['close'] - df['open']
    df['is_green'] = df['body'] > 0
    df['is_red'] = df['body'] < 0

    # Momentum
    df['momentum'] = df['close'].pct_change(5) * 100

    return df


def strategy_ema_crossover(df: pd.DataFrame, i: int, direction_filter: str = None) -> str:
    """EMA Crossover Strategy - trade when fast crosses slow"""
    if i < 2:
        return None

    prev_trend = df.iloc[i-1]['ema_trend']
    curr_trend = df.iloc[i]['ema_trend']

    # Crossover detection
    if prev_trend == -1 and curr_trend == 1:
        if direction_filter is None or direction_filter == 'BUY':
            return 'BUY'
    elif prev_trend == 1 and curr_trend == -1:
        if direction_filter is None or direction_filter == 'SELL':
            return 'SELL'
    return None


def strategy_rsi_reversal(df: pd.DataFrame, i: int, oversold: float = 30,
                          overbought: float = 70, confirm_candle: bool = True) -> str:
    """RSI Reversal - buy oversold, sell overbought"""
    if i < 2:
        return None

    rsi = df.iloc[i]['rsi']
    prev_rsi = df.iloc[i-1]['rsi']
    is_green = df.iloc[i]['is_green']
    is_red = df.iloc[i]['is_red']

    # Buy when RSI exits oversold
    if prev_rsi < oversold and rsi >= oversold:
        if not confirm_candle or is_green:
            return 'BUY'

    # Sell when RSI exits overbought
    if prev_rsi > overbought and rsi <= overbought:
        if not confirm_candle or is_red:
            return 'SELL'

    return None


def strategy_bb_bounce(df: pd.DataFrame, i: int, threshold: float = 0.05) -> str:
    """Bollinger Band Bounce - buy at lower band, sell at upper"""
    if i < 2:
        return None

    bb_pct = df.iloc[i]['bb_pct']
    prev_bb_pct = df.iloc[i-1]['bb_pct']
    is_green = df.iloc[i]['is_green']
    is_red = df.iloc[i]['is_red']

    # Buy near lower band with reversal candle
    if bb_pct < threshold and prev_bb_pct < bb_pct and is_green:
        return 'BUY'

    # Sell near upper band with reversal candle
    if bb_pct > (1 - threshold) and prev_bb_pct > bb_pct and is_red:
        return 'SELL'

    return None


def strategy_trend_pullback(df: pd.DataFrame, i: int, rsi_buy_zone: tuple = (40, 55),
                            rsi_sell_zone: tuple = (45, 60)) -> str:
    """Trend Pullback - enter pullbacks in strong trends"""
    if i < 5:
        return None

    ema_trend = df.iloc[i]['ema_trend']
    rsi = df.iloc[i]['rsi']
    is_green = df.iloc[i]['is_green']
    is_red = df.iloc[i]['is_red']
    close = df.iloc[i]['close']
    ema_slow = df.iloc[i]['ema_slow']

    # BUY: Uptrend, RSI pulled back, green candle near EMA
    if ema_trend == 1 and rsi_buy_zone[0] <= rsi <= rsi_buy_zone[1]:
        if is_green and close > ema_slow * 0.998:  # Within 0.2% of EMA
            return 'BUY'

    # SELL: Downtrend, RSI bounced, red candle near EMA
    if ema_trend == -1 and rsi_sell_zone[0] <= rsi <= rsi_sell_zone[1]:
        if is_red and close < ema_slow * 1.002:
            return 'SELL'

    return None


def strategy_macd_cross(df: pd.DataFrame, i: int, require_trend: bool = True) -> str:
    """MACD Crossover Strategy"""
    if i < 2:
        return None

    macd = df.iloc[i]['macd']
    signal = df.iloc[i]['macd_signal']
    prev_macd = df.iloc[i-1]['macd']
    prev_signal = df.iloc[i-1]['macd_signal']
    ema_trend = df.iloc[i]['ema_trend']

    # Bullish crossover
    if prev_macd < prev_signal and macd >= signal:
        if not require_trend or ema_trend == 1:
            return 'BUY'

    # Bearish crossover
    if prev_macd > prev_signal and macd <= signal:
        if not require_trend or ema_trend == -1:
            return 'SELL'

    return None


def strategy_momentum_breakout(df: pd.DataFrame, i: int, mom_threshold: float = 0.15) -> str:
    """Momentum Breakout - trade strong momentum moves"""
    if i < 2:
        return None

    momentum = df.iloc[i]['momentum']
    ema_trend = df.iloc[i]['ema_trend']
    rsi = df.iloc[i]['rsi']

    # Strong upward momentum in uptrend
    if momentum > mom_threshold and ema_trend == 1 and rsi < 70:
        return 'BUY'

    # Strong downward momentum in downtrend
    if momentum < -mom_threshold and ema_trend == -1 and rsi > 30:
        return 'SELL'

    return None


def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   tp_pips: float, sl_pips: float, pip_mult: float,
                   max_bars: int = 100) -> dict:
    """Simulate a single trade"""
    entry_price = df.iloc[entry_idx]['close']
    tp_price_dist = tp_pips / pip_mult
    sl_price_dist = sl_pips / pip_mult

    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        high = df.iloc[j]['high']
        low = df.iloc[j]['low']

        if direction == 'BUY':
            tp_hit = high >= entry_price + tp_price_dist
            sl_hit = low <= entry_price - sl_price_dist
        else:
            tp_hit = low <= entry_price - tp_price_dist
            sl_hit = high >= entry_price + sl_price_dist

        if tp_hit and sl_hit:
            # Assume SL hit first (conservative)
            return {'won': False, 'pnl': -sl_pips, 'exit': 'SL', 'bars': j - entry_idx}
        elif tp_hit:
            return {'won': True, 'pnl': tp_pips, 'exit': 'TP', 'bars': j - entry_idx}
        elif sl_hit:
            return {'won': False, 'pnl': -sl_pips, 'exit': 'SL', 'bars': j - entry_idx}

    # Timeout
    exit_price = df.iloc[min(entry_idx + max_bars - 1, len(df) - 1)]['close']
    if direction == 'BUY':
        pnl = (exit_price - entry_price) * pip_mult
    else:
        pnl = (entry_price - exit_price) * pip_mult

    return {'won': pnl > 0, 'pnl': pnl, 'exit': 'TIMEOUT', 'bars': max_bars}


def run_optimization(pair: str, strategies: list, tp_sl_combos: list,
                     sample_rate: int = 20, min_trades: int = 30) -> list:
    """Run optimization for a pair"""
    print(f"\n{'='*60}")
    print(f"OPTIMIZING {pair}")
    print(f"{'='*60}")

    prefix, pip_mult = PAIRS[pair]
    df = load_data(pair)
    print(f"Loaded {len(df)} candles")

    results = []

    for strategy_name, strategy_func, strategy_params in strategies:
        for tp_pips, sl_pips in tp_sl_combos:
            # Calculate indicators with strategy-specific params
            df_calc = calculate_indicators(df.copy(), **strategy_params.get('indicators', {}))

            trades = []
            cooldown_until = 0

            # Sample candles
            for i in range(100, len(df_calc) - 50, sample_rate):
                if i < cooldown_until:
                    continue

                # Get signal
                signal = strategy_func(df_calc, i, **strategy_params.get('entry', {}))

                if signal:
                    result = simulate_trade(df_calc, i, signal, tp_pips, sl_pips, pip_mult)
                    trades.append(result)
                    cooldown_until = i + max(10, result['bars'])

            if len(trades) >= min_trades:
                wins = len([t for t in trades if t['won']])
                win_rate = wins / len(trades) * 100
                total_pnl = sum(t['pnl'] for t in trades)
                avg_bars = np.mean([t['bars'] for t in trades])

                # Profit factor
                gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
                gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
                pf = gross_profit / gross_loss if gross_loss > 0 else 999

                results.append({
                    'pair': pair,
                    'strategy': strategy_name,
                    'tp': tp_pips,
                    'sl': sl_pips,
                    'rr': tp_pips / sl_pips,
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'pnl': total_pnl,
                    'pf': pf,
                    'avg_bars': avg_bars,
                    'params': strategy_params
                })

    return results


def main():
    """Run full optimization"""
    print("="*70)
    print("FOREX STRATEGY OPTIMIZER - Finding 70%+ Win Rate Configurations")
    print("="*70)
    print(f"Data: {DATA_DIR}")
    print(f"Pairs: {list(PAIRS.keys())}")

    # Define strategies to test
    strategies = [
        # Strategy 1: EMA Crossover (basic)
        ("EMA_Cross_9_21", strategy_ema_crossover, {
            'indicators': {'ema_fast': 9, 'ema_slow': 21},
            'entry': {}
        }),

        # Strategy 2: EMA Crossover (slower)
        ("EMA_Cross_12_26", strategy_ema_crossover, {
            'indicators': {'ema_fast': 12, 'ema_slow': 26},
            'entry': {}
        }),

        # Strategy 3: RSI Reversal (standard)
        ("RSI_30_70", strategy_rsi_reversal, {
            'indicators': {'rsi_period': 14},
            'entry': {'oversold': 30, 'overbought': 70, 'confirm_candle': True}
        }),

        # Strategy 4: RSI Reversal (tighter)
        ("RSI_25_75", strategy_rsi_reversal, {
            'indicators': {'rsi_period': 14},
            'entry': {'oversold': 25, 'overbought': 75, 'confirm_candle': True}
        }),

        # Strategy 5: RSI Reversal (extreme)
        ("RSI_20_80", strategy_rsi_reversal, {
            'indicators': {'rsi_period': 14},
            'entry': {'oversold': 20, 'overbought': 80, 'confirm_candle': True}
        }),

        # Strategy 6: Bollinger Band Bounce
        ("BB_Bounce_5pct", strategy_bb_bounce, {
            'indicators': {'bb_period': 20},
            'entry': {'threshold': 0.05}
        }),

        # Strategy 7: Bollinger Band Bounce (tighter)
        ("BB_Bounce_3pct", strategy_bb_bounce, {
            'indicators': {'bb_period': 20},
            'entry': {'threshold': 0.03}
        }),

        # Strategy 8: Trend Pullback (standard)
        ("Pullback_40_55", strategy_trend_pullback, {
            'indicators': {'ema_fast': 9, 'ema_slow': 21},
            'entry': {'rsi_buy_zone': (40, 55), 'rsi_sell_zone': (45, 60)}
        }),

        # Strategy 9: Trend Pullback (tighter RSI)
        ("Pullback_35_50", strategy_trend_pullback, {
            'indicators': {'ema_fast': 9, 'ema_slow': 21},
            'entry': {'rsi_buy_zone': (35, 50), 'rsi_sell_zone': (50, 65)}
        }),

        # Strategy 10: MACD Cross with trend
        ("MACD_Trend", strategy_macd_cross, {
            'indicators': {},
            'entry': {'require_trend': True}
        }),

        # Strategy 11: MACD Cross without trend filter
        ("MACD_NoTrend", strategy_macd_cross, {
            'indicators': {},
            'entry': {'require_trend': False}
        }),

        # Strategy 12: Momentum Breakout
        ("Momentum_0.15", strategy_momentum_breakout, {
            'indicators': {},
            'entry': {'mom_threshold': 0.15}
        }),

        # Strategy 13: Momentum Breakout (stronger)
        ("Momentum_0.20", strategy_momentum_breakout, {
            'indicators': {},
            'entry': {'mom_threshold': 0.20}
        }),
    ]

    # TP/SL combinations to test (tighter TP = higher win rate)
    # Key insight: smaller TP relative to SL = higher win rate
    tp_sl_combos = [
        # Very tight (high win rate expected)
        (5, 15),   # 1:3 risk (TP is 1/3 of SL)
        (6, 12),   # 1:2 risk
        (8, 16),   # 1:2 risk
        (10, 20),  # 1:2 risk

        # Tight
        (8, 12),   # 2:3 risk
        (10, 15),  # 2:3 risk
        (12, 18),  # 2:3 risk

        # Standard
        (10, 10),  # 1:1 risk
        (12, 12),  # 1:1 risk
        (15, 15),  # 1:1 risk

        # Current settings for reference
        (15, 12),  # 1.25:1 (current USD_CHF)
        (18, 14),  # 1.29:1 (current USD_CAD)
        (20, 16),  # 1.25:1 (current EUR_USD)
        (25, 20),  # 1.25:1 (current USD_JPY)
        (35, 28),  # 1.25:1 (current GBP_USD)
    ]

    all_results = []

    # Run optimization for each pair
    for pair in PAIRS.keys():
        results = run_optimization(pair, strategies, tp_sl_combos)
        all_results.extend(results)

        # Show top 5 for this pair
        sorted_results = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:5]
        print(f"\nTop 5 for {pair}:")
        for r in sorted_results:
            print(f"  {r['strategy']}: TP={r['tp']}p SL={r['sl']}p | "
                  f"WR={r['win_rate']:.1f}% | Trades={r['trades']} | PnL={r['pnl']:+.1f}p | PF={r['pf']:.2f}")

    # Final summary - top strategies with 70%+ win rate
    print("\n" + "="*70)
    print("STRATEGIES WITH 70%+ WIN RATE")
    print("="*70)

    high_wr = [r for r in all_results if r['win_rate'] >= 70]
    high_wr_sorted = sorted(high_wr, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)

    if high_wr_sorted:
        print(f"\nFound {len(high_wr_sorted)} configurations with 70%+ win rate!\n")
        print(f"{'Pair':<10} {'Strategy':<18} {'TP':<4} {'SL':<4} {'R:R':<5} {'Trades':<7} {'Win%':<7} {'PnL':<10} {'PF':<6}")
        print("-" * 80)

        for r in high_wr_sorted[:30]:  # Top 30
            rr = f"{r['rr']:.2f}"
            pnl = f"{r['pnl']:+.1f}p"
            pf = f"{r['pf']:.2f}" if r['pf'] < 100 else "INF"
            print(f"{r['pair']:<10} {r['strategy']:<18} {r['tp']:<4.0f} {r['sl']:<4.0f} {rr:<5} "
                  f"{r['trades']:<7} {r['win_rate']:<6.1f}% {pnl:<10} {pf:<6}")
    else:
        print("\nNo strategies found with 70%+ win rate.")
        print("\nTop 20 overall:")
        top20 = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)[:20]
        print(f"\n{'Pair':<10} {'Strategy':<18} {'TP':<4} {'SL':<4} {'Win%':<7} {'Trades':<7} {'PnL':<10}")
        print("-" * 70)
        for r in top20:
            print(f"{r['pair']:<10} {r['strategy']:<18} {r['tp']:<4.0f} {r['sl']:<4.0f} "
                  f"{r['win_rate']:<6.1f}% {r['trades']:<7} {r['pnl']:+.1f}p")

    # Profitable strategies (positive P&L)
    print("\n" + "="*70)
    print("PROFITABLE STRATEGIES (Positive P&L)")
    print("="*70)

    profitable = [r for r in all_results if r['pnl'] > 0]
    profitable_sorted = sorted(profitable, key=lambda x: x['pnl'], reverse=True)[:20]

    if profitable_sorted:
        print(f"\nFound {len(profitable)} profitable configurations!\n")
        print(f"{'Pair':<10} {'Strategy':<18} {'TP':<4} {'SL':<4} {'Win%':<7} {'Trades':<7} {'PnL':<10} {'PF':<6}")
        print("-" * 75)

        for r in profitable_sorted:
            pf = f"{r['pf']:.2f}" if r['pf'] < 100 else "INF"
            print(f"{r['pair']:<10} {r['strategy']:<18} {r['tp']:<4.0f} {r['sl']:<4.0f} "
                  f"{r['win_rate']:<6.1f}% {r['trades']:<7} {r['pnl']:+.1f}p    {pf:<6}")

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
