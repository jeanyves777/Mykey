"""
ENHANCED SCALPING STRATEGY - MULTI-LEVEL VALIDATION
====================================================

Uses COIN strategy's proven safeguards:
✅ SCORING SYSTEM (0-15 points) instead of binary checks
✅ MINIMUM SCORE threshold (8+ required)
✅ MINIMUM LEAD requirement (4+ point advantage)
✅ Multiple indicator confirmation (EMA, RSI, MACD, BB, Volume)
✅ Volatility filter (ATR)
✅ Per-indicator weighting (4 points for trend, 3 for RSI/MACD, etc.)
✅ High-probability entries only - NO WEAK SIGNALS

Target: 80%+ win rate with strict filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ALL indicators for scoring."""
    df = df.copy()
    
    # EMAs - Trend confirmation
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI - Momentum
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD - Entry trigger
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands - Extremes & position
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
    
    # Volume analysis
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
    
    return df


def analyze_market_with_scoring(df: pd.DataFrame, verbose: bool = False) -> Tuple[Optional[str], str, int, int]:
    """
    HIGH-PROBABILITY SIGNAL ANALYSIS with SCORING SYSTEM
    
    Returns: (signal, reason, bullish_score, bearish_score)
    - signal: 'BUY', 'SELL', or None
    - reason: Detailed explanation
    - bullish_score: Total bullish points (0-15)
    - bearish_score: Total bearish points (0-15)
    """
    if len(df) < 50:
        return None, "Insufficient data", 0, 0
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    bullish_score = 0
    bearish_score = 0
    bullish_signals = []
    bearish_signals = []
    
    # ========== 1. EMA STACK (up to 4 points) ==========
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']
    close = current['close']
    
    if ema9 > ema21 > ema50:
        if close > ema9:
            bullish_score += 4
            bullish_signals.append("EMA Stack STRONG (Price > EMA9 > EMA21 > EMA50)")
        else:
            bullish_score += 2
            bullish_signals.append("EMA alignment bullish (EMA9 > EMA21 > EMA50)")
    elif ema9 < ema21 < ema50:
        if close < ema9:
            bearish_score += 4
            bearish_signals.append("EMA Stack STRONG (Price < EMA9 < EMA21 < EMA50)")
        else:
            bearish_score += 2
            bearish_signals.append("EMA alignment bearish (EMA9 < EMA21 < EMA50)")
    
    # ========== 2. RSI CONFIRMATION (up to 3 points) ==========
    rsi = current['rsi']
    
    if 30 <= rsi <= 50:  # Recovering from oversold - bullish
        bullish_score += 3
        bullish_signals.append(f"RSI bullish zone ({rsi:.1f}) - recovering")
    elif 50 < rsi <= 65:  # Strong momentum, not overbought
        bullish_score += 2
        bullish_signals.append(f"RSI momentum ({rsi:.1f}) - strong")
    elif rsi > 70:  # Overbought - bearish reversal potential
        bearish_score += 3
        bearish_signals.append(f"RSI OVERBOUGHT ({rsi:.1f}) - reversal risk")
    elif rsi < 30:  # Oversold - bullish bounce potential
        bullish_score += 3
        bullish_signals.append(f"RSI OVERSOLD ({rsi:.1f}) - bounce setup")
    elif 50 <= rsi <= 70:  # Neutral to high
        bearish_score += 1
        bearish_signals.append(f"RSI neutral-high ({rsi:.1f})")
    
    # ========== 3. MACD CONFIRMATION (up to 3 points) ==========
    macd = current['macd']
    macd_signal = current['macd_signal']
    macd_hist = current['macd_hist']
    prev_hist = prev['macd_hist']
    
    if macd > macd_signal and macd_hist > prev_hist:
        bullish_score += 3
        bullish_signals.append(f"MACD bullish (hist: {macd_hist:.5f}, increasing)")
    elif macd < macd_signal and macd_hist < prev_hist:
        bearish_score += 3
        bearish_signals.append(f"MACD bearish (hist: {macd_hist:.5f}, decreasing)")
    
    # ========== 4. BOLLINGER BANDS (up to 3 points) ==========
    bb_upper = current['bb_upper']
    bb_middle = current['bb_middle']
    bb_lower = current['bb_lower']
    
    if not pd.isna(bb_upper) and not pd.isna(bb_lower):
        bb_width = bb_upper - bb_lower
        if bb_width > 0:
            bb_position = (close - bb_lower) / bb_width
            
            if close > bb_middle:
                bullish_score += 2
                bullish_signals.append(f"Above BB middle ({bb_position:.1%} position)")
                if bb_position < 0.8:  # Not too extended
                    bullish_score += 1
                    bullish_signals.append("BB not overbought (room to run)")
            else:
                bearish_score += 2
                bearish_signals.append(f"Below BB middle ({bb_position:.1%} position)")
                if bb_position > 0.2:  # Not too extended
                    bearish_score += 1
                    bearish_signals.append("BB not oversold (room to fall)")
    
    # ========== 5. VOLUME CONFIRMATION (up to 2 points) ==========
    vol_ratio = current['volume_ratio']
    
    if not pd.isna(vol_ratio):
        if vol_ratio > 1.5:  # High volume
            if bullish_score > bearish_score:
                bullish_score += 2
                bullish_signals.append(f"High volume confirms bullish ({vol_ratio:.1f}x)")
            else:
                bearish_score += 2
                bearish_signals.append(f"High volume confirms bearish ({vol_ratio:.1f}x)")
        elif vol_ratio > 1.0:  # Above average
            if bullish_score > bearish_score:
                bullish_score += 1
                bullish_signals.append(f"Volume above avg ({vol_ratio:.1f}x)")
            else:
                bearish_score += 1
                bearish_signals.append(f"Volume above avg ({vol_ratio:.1f}x)")
    
    # ========== 6. ATR VOLATILITY CHECK (score reducer) ==========
    atr = current['atr']
    atr_pct = (atr / close) * 100 if close > 0 and not pd.isna(atr) else 0
    
    if atr_pct > 0.10:  # Too volatile (>0.10% for forex)
        bullish_score = int(bullish_score * 0.7)
        bearish_score = int(bearish_score * 0.7)
        volatility_warning = f"HIGH VOLATILITY ({atr_pct:.3f}%) - scores reduced"
    else:
        volatility_warning = None
    
    # ========== DECISION LOGIC - STRICT THRESHOLDS ==========
    MIN_SCORE = 8  # Need 8+ points (out of 15 possible)
    MIN_LEAD = 4   # Need 4+ point advantage over opposite direction
    
    reason_parts = []
    reason_parts.append(f"BULLISH: {bullish_score}/15 | BEARISH: {bearish_score}/15")
    
    if verbose:
        if bullish_signals:
            reason_parts.append("Bullish: " + ", ".join(bullish_signals))
        if bearish_signals:
            reason_parts.append("Bearish: " + ", ".join(bearish_signals))
        if volatility_warning:
            reason_parts.append(volatility_warning)
    
    reason = " | ".join(reason_parts)
    
    # HIGH-PROBABILITY entry only
    if bullish_score >= MIN_SCORE and bullish_score >= bearish_score + MIN_LEAD:
        return 'BUY', f"HIGH-PROB BUY ({reason})", bullish_score, bearish_score
    elif bearish_score >= MIN_SCORE and bearish_score >= bullish_score + MIN_LEAD:
        return 'SELL', f"HIGH-PROB SELL ({reason})", bullish_score, bearish_score
    else:
        return None, f"NO HIGH-PROB SIGNAL ({reason})", bullish_score, bearish_score


def get_signal(instrument: str, df: pd.DataFrame, config=None) -> Tuple[Optional[str], str]:
    """
    Main entry point using multi-level validation.
    
    Args:
        instrument: Currency pair
        df: DataFrame with OHLC data
        config: Optional config (not used, for compatibility)
    
    Returns:
        Tuple of (signal, reason)
    """
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Analyze with scoring system
    signal, reason, bull_score, bear_score = analyze_market_with_scoring(df, verbose=True)
    
    return signal, reason
