"""
TEST: Can ANY simple indicator predict 1-minute crypto direction?

This tests basic indicators BEFORE spending more time on ML.
If simple indicators can't beat 50%, the timeframe may be unpredictable.

Indicators tested:
1. RSI oversold/overbought
2. MACD crossover
3. Momentum positive
4. Price above SMA
5. Bollinger Band bounce
6. Volume spike + direction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def compute_rsi(close, period=14):
    """Compute RSI."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def test_signal(df, signal_mask, tp_pct, sl_pct, max_bars=60, name=""):
    """Test a signal's win rate."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    wins = 0
    losses = 0
    timeouts = 0

    signal_indices = np.where(signal_mask)[0]

    for i in signal_indices:
        if i < 100 or i >= len(df) - max_bars:
            continue

        entry = close[i]
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)

        outcome = 'timeout'
        for j in range(1, max_bars + 1):
            idx = i + j
            if high[idx] >= tp_price:
                outcome = 'win'
                break
            if low[idx] <= sl_price:
                outcome = 'loss'
                break

        if outcome == 'win':
            wins += 1
        elif outcome == 'loss':
            losses += 1
        else:
            timeouts += 1

    total = wins + losses + timeouts
    if total == 0:
        return {'name': name, 'signals': 0, 'win_rate': 0, 'wins': 0, 'losses': 0, 'timeouts': 0}

    return {
        'name': name,
        'signals': total,
        'win_rate': wins / total,
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'timeout_rate': timeouts / total
    }


def main():
    print("=" * 80)
    print("TESTING SIMPLE INDICATORS ON 1-MINUTE CRYPTO")
    print("Can ANY indicator predict direction better than random?")
    print("=" * 80)

    # Load data
    data_file = Path(__file__).parent / "Crypto_Data_Fresh" / "BTCUSD_1m.csv"
    print(f"\nLoading {data_file}...")
    df = pd.read_csv(data_file)
    df = df.tail(100000).reset_index(drop=True)  # Last 100k bars (~70 days)
    print(f"Using {len(df):,} bars")

    # Compute indicators (ALL SHIFTED by 1 to prevent look-ahead)
    print("\nComputing indicators (all shifted by 1 bar)...")

    # RSI
    df['rsi'] = compute_rsi(df['close'], 14).shift(1)

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean().shift(1)
    exp26 = df['close'].ewm(span=26, adjust=False).mean().shift(1)
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # SMAs
    df['sma_10'] = df['close'].rolling(10).mean().shift(1)
    df['sma_20'] = df['close'].rolling(20).mean().shift(1)
    df['sma_50'] = df['close'].rolling(50).mean().shift(1)

    # Momentum
    df['mom_5'] = df['close'].pct_change(5).shift(1)
    df['mom_10'] = df['close'].pct_change(10).shift(1)

    # Bollinger Bands
    rolling_std = df['close'].rolling(20).std().shift(1)
    df['bb_upper'] = df['sma_20'] + 2 * rolling_std
    df['bb_lower'] = df['sma_20'] - 2 * rolling_std
    df['bb_position'] = (df['close'].shift(1) - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean().shift(1)
    df['volume_ratio'] = df['volume'].shift(1) / (df['volume_sma'] + 1e-10)

    # Previous candle direction
    df['prev_green'] = (df['close'].shift(1) > df['open'].shift(1))

    # Previous close relative to SMA
    prev_close = df['close'].shift(1)

    # Test parameters: 0.3% TP, 0.3% SL (1:1 R:R)
    tp_pct = 0.003
    sl_pct = 0.003
    max_bars = 30  # 30 minutes max hold

    print(f"\nTest parameters: TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, Max hold={max_bars} bars")
    print(f"Required win rate for breakeven (1:1 R:R): 50%")

    results = []

    # ================================================================
    # TEST 1: RSI Oversold (< 30)
    # ================================================================
    signal = df['rsi'] < 30
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "RSI < 30 (oversold)")
    results.append(result)

    # TEST 2: RSI < 25 (more oversold)
    signal = df['rsi'] < 25
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "RSI < 25 (very oversold)")
    results.append(result)

    # TEST 3: RSI < 35 (slightly oversold)
    signal = df['rsi'] < 35
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "RSI < 35")
    results.append(result)

    # ================================================================
    # TEST 4: MACD Bullish Crossover
    # ================================================================
    signal = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "MACD bullish crossover")
    results.append(result)

    # TEST 5: MACD Histogram positive and increasing
    signal = (df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1))
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "MACD hist positive & rising")
    results.append(result)

    # ================================================================
    # TEST 6: Price above SMA20 (trend following)
    # ================================================================
    signal = prev_close > df['sma_20']
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "Price > SMA20")
    results.append(result)

    # TEST 7: Price above all SMAs (strong uptrend)
    signal = (prev_close > df['sma_10']) & (prev_close > df['sma_20']) & (prev_close > df['sma_50'])
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "Price > all SMAs")
    results.append(result)

    # ================================================================
    # TEST 8: Positive momentum
    # ================================================================
    signal = df['mom_5'] > 0.001  # 0.1% momentum over 5 bars
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "5-bar momentum > 0.1%")
    results.append(result)

    signal = df['mom_10'] > 0.002  # 0.2% momentum over 10 bars
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "10-bar momentum > 0.2%")
    results.append(result)

    # ================================================================
    # TEST 9: Bollinger Band bounce
    # ================================================================
    signal = df['bb_position'] < 0.2  # Near lower band
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "BB position < 0.2 (near lower)")
    results.append(result)

    signal = df['bb_position'] < 0.1  # Very near lower band
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "BB position < 0.1 (very near lower)")
    results.append(result)

    # ================================================================
    # TEST 10: Volume spike + green candle
    # ================================================================
    signal = (df['volume_ratio'] > 2.0) & df['prev_green']
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "Volume spike + green candle")
    results.append(result)

    # ================================================================
    # TEST 11: Combination - RSI oversold + uptrend
    # ================================================================
    signal = (df['rsi'] < 35) & (prev_close > df['sma_50'])
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "RSI<35 + above SMA50")
    results.append(result)

    # ================================================================
    # TEST 12: 3 consecutive green candles
    # ================================================================
    df['green_streak'] = df['prev_green'].rolling(3).sum()
    signal = df['green_streak'] == 3
    result = test_signal(df, signal, tp_pct, sl_pct, max_bars, "3 consecutive green candles")
    results.append(result)

    # ================================================================
    # RANDOM BASELINE
    # ================================================================
    np.random.seed(42)
    random_signal = np.random.random(len(df)) > 0.90  # ~10% random signals
    result = test_signal(df, random_signal, tp_pct, sl_pct, max_bars, "RANDOM (10% baseline)")
    results.append(result)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: WIN RATE BY INDICATOR")
    print("=" * 80)
    print(f"\n{'Indicator':<35} {'Signals':<10} {'Win Rate':<10} {'T/O Rate':<10} {'Edge'}")
    print("-" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('win_rate', ascending=False)

    # Get random baseline
    random_wr = results_df[results_df['name'].str.contains('RANDOM')]['win_rate'].values[0]

    for _, row in results_df.iterrows():
        edge = row['win_rate'] - random_wr
        edge_str = f"+{edge*100:.1f}%" if edge > 0 else f"{edge*100:.1f}%"
        marker = " ***" if row['win_rate'] > 0.50 else ""
        print(f"{row['name']:<35} {row['signals']:<10} {row['win_rate']*100:.1f}%     "
              f"{row['timeout_rate']*100:.1f}%     {edge_str}{marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    above_50 = results_df[results_df['win_rate'] > 0.50]
    print(f"\nIndicators with >50% win rate: {len(above_50)}/{len(results_df)}")

    above_random = results_df[results_df['win_rate'] > random_wr + 0.02]  # 2% edge
    print(f"Indicators beating random by >2%: {len(above_random)}/{len(results_df)}")

    if len(above_50) == 0:
        print("\n*** NO INDICATOR ACHIEVED >50% WIN RATE ***")
        print("This suggests 1-minute crypto may be inherently unpredictable.")
        print("\nRecommendations:")
        print("  1. Try 5-minute or 15-minute timeframe")
        print("  2. Use wider TP/SL (e.g., 1% TP / 0.5% SL)")
        print("  3. Consider mean-reversion on higher timeframes")
        print("  4. Consider different assets (stocks, forex)")
    else:
        best = above_50.iloc[0]
        print(f"\nBest indicator: {best['name']}")
        print(f"  Win rate: {best['win_rate']*100:.1f}%")
        print(f"  This could be worth exploring further!")

    # Test with wider TP/SL
    print("\n" + "=" * 80)
    print("TESTING WIDER TP/SL (0.5% TP, 0.3% SL)")
    print("=" * 80)

    tp_pct = 0.005
    sl_pct = 0.003
    max_bars = 60

    # Test best indicators with wider targets
    best_signals = [
        (df['rsi'] < 30, "RSI < 30"),
        ((df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1)), "MACD hist rising"),
        (df['bb_position'] < 0.2, "BB near lower"),
        (df['mom_5'] > 0.001, "Momentum > 0.1%"),
    ]

    print(f"\nTP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, Max hold={max_bars} bars")
    print(f"Required win rate for breakeven (1.67:1 R:R): ~38%")

    for signal, name in best_signals:
        result = test_signal(df, signal, tp_pct, sl_pct, max_bars, name)
        marker = " ***" if result['win_rate'] > 0.38 else ""
        print(f"  {name:<35} WR: {result['win_rate']*100:.1f}% ({result['signals']} signals){marker}")

    # ================================================================
    # TEST MUCH WIDER TARGETS (1% TP, 0.5% SL)
    # ================================================================
    print("\n" + "=" * 80)
    print("TESTING MUCH WIDER TARGETS (1% TP, 0.5% SL)")
    print("=" * 80)

    tp_pct = 0.01   # 1% TP
    sl_pct = 0.005  # 0.5% SL
    max_bars = 120  # 2 hours

    print(f"\nTP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, Max hold={max_bars} bars")
    print(f"Required win rate for breakeven (2:1 R:R): ~34%")
    print()

    for signal, name in best_signals:
        result = test_signal(df, signal, tp_pct, sl_pct, max_bars, name)
        marker = " ***" if result['win_rate'] > 0.34 else ""
        print(f"  {name:<35} WR: {result['win_rate']*100:.1f}% T/O: {result['timeout_rate']*100:.1f}% ({result['signals']} signals){marker}")

    # ================================================================
    # TEST VERY WIDE TARGETS (2% TP, 1% SL)
    # ================================================================
    print("\n" + "=" * 80)
    print("TESTING VERY WIDE TARGETS (2% TP, 1% SL)")
    print("=" * 80)

    tp_pct = 0.02   # 2% TP
    sl_pct = 0.01   # 1% SL
    max_bars = 240  # 4 hours

    print(f"\nTP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, Max hold={max_bars} bars")
    print(f"Required win rate for breakeven (2:1 R:R): ~34%")
    print()

    for signal, name in best_signals:
        result = test_signal(df, signal, tp_pct, sl_pct, max_bars, name)
        marker = " ***" if result['win_rate'] > 0.34 else ""
        print(f"  {name:<35} WR: {result['win_rate']*100:.1f}% T/O: {result['timeout_rate']*100:.1f}% ({result['signals']} signals){marker}")

    # ================================================================
    # TEST TIGHT SL, WIDE TP (scalping with trailing)
    # ================================================================
    print("\n" + "=" * 80)
    print("TESTING ASYMMETRIC (1.5% TP, 0.3% SL) - Tight stops, let winners run")
    print("=" * 80)

    tp_pct = 0.015   # 1.5% TP
    sl_pct = 0.003   # 0.3% SL (tight)
    max_bars = 120

    print(f"\nTP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, Max hold={max_bars} bars")
    print(f"Required win rate for breakeven (5:1 R:R): ~17%")
    print()

    for signal, name in best_signals:
        result = test_signal(df, signal, tp_pct, sl_pct, max_bars, name)
        marker = " ***" if result['win_rate'] > 0.17 else ""
        print(f"  {name:<35} WR: {result['win_rate']*100:.1f}% T/O: {result['timeout_rate']*100:.1f}% ({result['signals']} signals){marker}")


if __name__ == "__main__":
    main()
