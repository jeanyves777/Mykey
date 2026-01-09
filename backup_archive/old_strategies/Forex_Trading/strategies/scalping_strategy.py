"""
STRONG SCALPING STRATEGY - MULTI-INDICATOR VALIDATION
======================================================

Ultra-precise scalping with 5+ pip targets
Combines 4+ indicators for HIGH CONFIDENCE entries
NO OVERTRADING - strict entry requirements

Strategy Requirements (ALL must align):
1. Trend confirmation: EMA alignment
2. Momentum confirmation: RSI in specific zones
3. Entry trigger: MACD crossover or RSI extreme
4. Volume/volatility filter: Recent price action
5. Candle pattern confirmation

TARGET: 5-15 pip scalps with 75%+ win rate
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all indicators needed for multi-indicator validation.

    Args:
        df: DataFrame with OHLC data (columns: open, high, low, close)

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # EMAs - Multiple timeframes for trend strength
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # RSI - Momentum confirmation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9) - Entry trigger
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands - Volatility and extremes
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

    # ATR - Volatility filter
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    # Candle patterns
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / df['candle_range'].replace(0, 1)

    # Price momentum - recent strength
    df['price_change_3'] = df['close'].pct_change(3) * 100
    df['price_change_5'] = df['close'].pct_change(5) * 100

    return df


def check_trend_alignment(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    Check if EMA trend is strongly aligned for the direction.
    
    Requirements:
    - Strong uptrend: EMA9 > EMA21 > EMA50 (all aligned)
    - Strong downtrend: EMA9 < EMA21 < EMA50 (all aligned)
    - Price near EMA9 (not extended)
    """
    if len(df) < 1:
        return False, "Insufficient data"

    current = df.iloc[-1]
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']
    close = current['close']

    # Check if price is not too extended from EMA9 (within 0.15%)
    ema9_distance = abs(close - ema9) / ema9 * 100

    if direction == 'BUY':
        if ema9 > ema21 > ema50:
            if ema9_distance > 0.15:
                return False, f"Price too far from EMA9 ({ema9_distance:.2f}%)"
            return True, f"Strong uptrend: EMA9>{ema21:.5f}>EMA50>{ema50:.5f}"
        return False, "No uptrend alignment"
    
    elif direction == 'SELL':
        if ema9 < ema21 < ema50:
            if ema9_distance > 0.15:
                return False, f"Price too far from EMA9 ({ema9_distance:.2f}%)"
            return True, f"Strong downtrend: EMA9<{ema21:.5f}<EMA50<{ema50:.5f}"
        return False, "No downtrend alignment"

    return False, "Unknown direction"


def check_momentum_confirmation(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    Check RSI momentum is in the right zone for scalping.
    
    Requirements:
    - BUY: RSI between 40-55 (slight pullback from strength)
    - SELL: RSI between 45-60 (slight bounce from weakness)
    - NOT oversold/overbought extremes (avoid reversals)
    """
    if len(df) < 1:
        return False, "Insufficient data"

    current = df.iloc[-1]
    rsi = current['rsi']

    if direction == 'BUY':
        if 40 <= rsi <= 55:
            return True, f"RSI in buy zone: {rsi:.1f}"
        return False, f"RSI outside buy zone (40-55): {rsi:.1f}"
    
    elif direction == 'SELL':
        if 45 <= rsi <= 60:
            return True, f"RSI in sell zone: {rsi:.1f}"
        return False, f"RSI outside sell zone (45-60): {rsi:.1f}"

    return False, "Unknown direction"


def check_macd_trigger(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    Check MACD crossover trigger with histogram momentum.
    
    Requirements:
    - BUY: MACD crosses above signal OR histogram increasing + positive
    - SELL: MACD crosses below signal OR histogram decreasing + negative
    """
    if len(df) < 2:
        return False, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    macd = current['macd']
    macd_signal = current['macd_signal']
    macd_hist = current['macd_hist']
    prev_hist = prev['macd_hist']

    if direction == 'BUY':
        # Crossover or increasing histogram
        if macd > macd_signal and macd_hist > prev_hist:
            return True, f"MACD bullish: {macd:.5f}>{macd_signal:.5f}, hist rising"
        return False, f"MACD not bullish: {macd:.5f} vs {macd_signal:.5f}"
    
    elif direction == 'SELL':
        # Crossover or decreasing histogram
        if macd < macd_signal and macd_hist < prev_hist:
            return True, f"MACD bearish: {macd:.5f}<{macd_signal:.5f}, hist falling"
        return False, f"MACD not bearish: {macd:.5f} vs {macd_signal:.5f}"

    return False, "Unknown direction"


def check_candle_confirmation(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    Check candle pattern confirms the direction.
    
    Requirements:
    - BUY: Green candle with body > 40% of range (strong buying)
    - SELL: Red candle with body > 40% of range (strong selling)
    - Not a doji or indecisive candle
    """
    if len(df) < 1:
        return False, "Insufficient data"

    current = df.iloc[-1]
    is_green = current['is_green']
    is_red = current['is_red']
    body_ratio = current['body_ratio']

    # Need strong candle (not doji)
    if body_ratio < 0.4:
        return False, f"Weak candle (body ratio: {body_ratio:.2f})"

    if direction == 'BUY':
        if is_green:
            return True, f"Strong green candle (ratio: {body_ratio:.2f})"
        return False, "Not a green candle"
    
    elif direction == 'SELL':
        if is_red:
            return True, f"Strong red candle (ratio: {body_ratio:.2f})"
        return False, "Not a red candle"

    return False, "Unknown direction"


def check_volatility_filter(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check volatility is in acceptable range (not too low, not too high).
    
    Requirements:
    - ATR is above minimum (market moving)
    - Not in extreme volatility (avoid whipsaws)
    """
    if len(df) < 1:
        return False, "Insufficient data"

    current = df.iloc[-1]
    atr = current['atr']
    close = current['close']

    # ATR as percentage of price
    atr_pct = (atr / close) * 100

    # Ideal range: 0.015% - 0.10% (for forex)
    if atr_pct < 0.015:
        return False, f"ATR too low ({atr_pct:.4f}%) - market too quiet"
    if atr_pct > 0.10:
        return False, f"ATR too high ({atr_pct:.4f}%) - too volatile"

    return True, f"ATR acceptable: {atr_pct:.4f}%"


def get_scalping_signal(instrument: str, df: pd.DataFrame, config=None) -> Tuple[Optional[str], str]:
    """
    Get high-confidence scalping signal requiring ALL indicators to align.
    
    Args:
        instrument: Currency pair (e.g., 'EUR_USD')
        df: DataFrame with OHLC data
        config: Config module (optional)
    
    Returns:
        Tuple of (signal, reason) where signal is 'BUY', 'SELL', or None
    """
    # Calculate indicators
    df = calculate_indicators(df)

    if len(df) < 50:
        return None, "Insufficient data for indicators"

    # Test both directions
    for direction in ['BUY', 'SELL']:
        # Check all 5 validations
        trend_ok, trend_msg = check_trend_alignment(df, direction)
        if not trend_ok:
            continue  # No trend, skip this direction

        momentum_ok, momentum_msg = check_momentum_confirmation(df, direction)
        if not momentum_ok:
            continue  # No momentum, skip this direction

        macd_ok, macd_msg = check_macd_trigger(df, direction)
        if not macd_ok:
            continue  # No MACD trigger, skip this direction

        candle_ok, candle_msg = check_candle_confirmation(df, direction)
        if not candle_ok:
            continue  # No candle confirmation, skip this direction

        volatility_ok, volatility_msg = check_volatility_filter(df)
        if not volatility_ok:
            continue  # Volatility not good, skip this direction

        # ALL 5 checks passed!
        reason = f"{direction} - ALL CONFIRMED: " \
                 f"Trend({trend_msg}), " \
                 f"Momentum({momentum_msg}), " \
                 f"MACD({macd_msg}), " \
                 f"Candle({candle_msg}), " \
                 f"Vol({volatility_msg})"
        
        return direction, reason

    # No direction passed all checks
    return None, "No multi-indicator alignment"


def get_signal(instrument: str, df: pd.DataFrame, config=None) -> Tuple[Optional[str], str]:
    """
    Main entry point - uses scalping strategy.
    """
    return get_scalping_signal(instrument, df, config)
