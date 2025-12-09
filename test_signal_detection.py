"""
Test script for Options Trend Detection Signal Logic
Tests with simulated bar data since market is closed on weekends
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MockBar:
    """Mock bar for testing"""
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime = None


def calculate_ema(prices: list, period: int) -> float:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return sum(prices) / len(prices)
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def test_signal_detection(bars: list, scenario_name: str) -> str:
    """Test the signal detection logic with mock data"""
    print(f'\n{"="*70}')
    print(f'SCENARIO: {scenario_name}')
    print(f'{"="*70}')

    if len(bars) < 10:
        print('Not enough bar data for trend analysis')
        return 'NEUTRAL'

    current_price = bars[-1].close
    closes = [b.close for b in bars]
    ema_20 = calculate_ema(closes, 20)
    ema_9 = calculate_ema(closes, 9)

    bullish_score = 0
    bearish_score = 0
    trend_signals = []

    # 1. PRICE vs EMA-20
    if current_price > ema_20:
        bullish_score += 2
        trend_signals.append('Price > EMA20')
    elif current_price < ema_20:
        bearish_score += 2
        trend_signals.append('Price < EMA20')

    # 2. EMA-9 vs EMA-20
    if ema_9 > ema_20:
        bullish_score += 2
        trend_signals.append('EMA9 > EMA20')
    elif ema_9 < ema_20:
        bearish_score += 2
        trend_signals.append('EMA9 < EMA20')

    # 3. Last 5 bars momentum
    last_5_bars = bars[-5:]
    price_5_bars_ago = last_5_bars[0].open
    price_change_5 = ((current_price - price_5_bars_ago) / price_5_bars_ago) * 100

    if price_change_5 > 0.05:
        bullish_score += 1
        trend_signals.append(f'5-bar momentum: +{price_change_5:.2f}%')
    elif price_change_5 < -0.05:
        bearish_score += 1
        trend_signals.append(f'5-bar momentum: {price_change_5:.2f}%')

    # 4. Last 10 bars momentum
    last_10_bars = bars[-10:]
    price_10_bars_ago = last_10_bars[0].open
    price_change_10 = ((current_price - price_10_bars_ago) / price_10_bars_ago) * 100

    if price_change_10 > 0.1:
        bullish_score += 1
        trend_signals.append(f'10-bar momentum: +{price_change_10:.2f}%')
    elif price_change_10 < -0.1:
        bearish_score += 1
        trend_signals.append(f'10-bar momentum: {price_change_10:.2f}%')

    # 5. Bar color count
    green_bars = sum(1 for b in last_5_bars if b.close > b.open)
    red_bars = sum(1 for b in last_5_bars if b.close < b.open)

    if green_bars >= 4:
        bullish_score += 1
        trend_signals.append(f'Bar colors: {green_bars} green / {red_bars} red')
    elif red_bars >= 4:
        bearish_score += 1
        trend_signals.append(f'Bar colors: {green_bars} green / {red_bars} red')

    # 6. Higher highs / Lower lows
    recent_highs = [b.high for b in bars[-5:]]
    recent_lows = [b.low for b in bars[-5:]]

    if all(recent_highs[i] <= recent_highs[i+1] for i in range(len(recent_highs)-1)):
        bullish_score += 1
        trend_signals.append('Higher highs')
    if all(recent_lows[i] >= recent_lows[i+1] for i in range(len(recent_lows)-1)):
        bearish_score += 1
        trend_signals.append('Lower lows')

    # Print analysis
    print(f'  Current: ${current_price:.2f} | EMA9: ${ema_9:.2f} | EMA20: ${ema_20:.2f}')
    print(f'  5-bar change: {price_change_5:+.2f}% | 10-bar change: {price_change_10:+.2f}%')
    print(f'  Green/Red bars (last 5): {green_bars}/{red_bars}')
    print(f'  Signals: {trend_signals}')
    print(f'\n  BULLISH Score: {bullish_score} | BEARISH Score: {bearish_score}')

    # Decision
    MIN_TREND_SCORE = 4

    if bullish_score >= MIN_TREND_SCORE and bullish_score > bearish_score + 1:
        signal = 'BULLISH'
        print(f'\n  >>> RESULT: BULLISH CONFIRMED (score: {bullish_score}) -> BUY CALLS')
    elif bearish_score >= MIN_TREND_SCORE and bearish_score > bullish_score + 1:
        signal = 'BEARISH'
        print(f'\n  >>> RESULT: BEARISH CONFIRMED (score: {bearish_score}) -> BUY PUTS')
    else:
        signal = 'BULLISH'  # Default to BULLISH for neutral
        print(f'\n  >>> RESULT: NEUTRAL - defaulting to BULLISH (CALLs)')

    return signal


def main():
    print('\n' + '='*70)
    print('TESTING TREND DETECTION SIGNAL LOGIC')
    print('='*70)
    print('Testing with simulated bar data (market is closed on Sunday)')

    # Scenario 1: CLEAR DOWNTREND
    downtrend_bars = []
    base_price = 300.0
    for i in range(20):
        drop = i * 1.5
        open_p = base_price - drop
        close_p = base_price - drop - 1.0
        high_p = open_p + 0.5
        low_p = close_p - 0.5
        downtrend_bars.append(MockBar(open=open_p, high=high_p, low=low_p, close=close_p))

    result1 = test_signal_detection(downtrend_bars, 'CLEAR DOWNTREND (20 consecutive red bars, price falling)')

    # Scenario 2: CLEAR UPTREND
    uptrend_bars = []
    base_price = 270.0
    for i in range(20):
        rise = i * 1.5
        open_p = base_price + rise
        close_p = base_price + rise + 1.0
        low_p = open_p - 0.5
        high_p = close_p + 0.5
        uptrend_bars.append(MockBar(open=open_p, high=high_p, low=low_p, close=close_p))

    result2 = test_signal_detection(uptrend_bars, 'CLEAR UPTREND (20 consecutive green bars, price rising)')

    # Scenario 3: CHOPPY MARKET
    choppy_bars = []
    base_price = 285.0
    for i in range(20):
        if i % 2 == 0:
            open_p = base_price - 0.5
            close_p = base_price + 0.5
        else:
            open_p = base_price + 0.5
            close_p = base_price - 0.5
        high_p = max(open_p, close_p) + 0.3
        low_p = min(open_p, close_p) - 0.3
        choppy_bars.append(MockBar(open=open_p, high=high_p, low=low_p, close=close_p))

    result3 = test_signal_detection(choppy_bars, 'CHOPPY/SIDEWAYS MARKET (alternating bars, no trend)')

    # Scenario 4: REVERSAL (was down, now up)
    reversal_bars = []
    base_price = 300.0
    for i in range(15):
        drop = i * 0.5
        open_p = base_price - drop
        close_p = base_price - drop - 0.3
        high_p = open_p + 0.2
        low_p = close_p - 0.2
        reversal_bars.append(MockBar(open=open_p, high=high_p, low=low_p, close=close_p))

    for i in range(5):
        rise = i * 2.0
        open_p = 290.0 + rise
        close_p = 291.5 + rise
        low_p = open_p - 0.3
        high_p = close_p + 0.3
        reversal_bars.append(MockBar(open=open_p, high=high_p, low=low_p, close=close_p))

    result4 = test_signal_detection(reversal_bars, 'REVERSAL (15 bars down, then 5 bars strong bounce)')

    # Summary
    print('\n' + '='*70)
    print('TEST SUMMARY')
    print('='*70)
    print(f'Scenario 1 (Downtrend):    {result1:8} - Expected: BEARISH')
    print(f'Scenario 2 (Uptrend):      {result2:8} - Expected: BULLISH')
    print(f'Scenario 3 (Choppy):       {result3:8} - Expected: BULLISH (neutral default)')
    print(f'Scenario 4 (Reversal up):  {result4:8} - Expected: BULLISH or NEUTRAL')
    print('='*70)

    # Validation
    all_passed = True
    if result1 != 'BEARISH':
        print('FAIL: Downtrend should have detected BEARISH!')
        all_passed = False
    if result2 != 'BULLISH':
        print('FAIL: Uptrend should have detected BULLISH!')
        all_passed = False

    if all_passed:
        print('\nALL TESTS PASSED - Signal detection is working correctly!')
    else:
        print('\nSOME TESTS FAILED - Review the signal detection logic')

    return all_passed


if __name__ == '__main__':
    main()
