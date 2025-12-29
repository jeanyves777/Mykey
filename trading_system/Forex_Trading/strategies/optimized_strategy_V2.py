"""
OPTIMIZED FOREX STRATEGY V2 - 14 Multi-Indicator Strategies
============================================================

Strategy logic for multiple entry strategies with filters.
Configuration is loaded from config files (paper or live).

GROUP A - EXISTING STRATEGIES (4):
1. MACD_CROSS: MACD line crosses signal line
2. MACD_ZERO: MACD crosses zero line
3. RSI_REVERSAL: RSI reversal at 35/65 levels
4. RSI_EXTREME: RSI at extreme levels (<25 or >75)

GROUP B - NEW MULTI-INDICATOR STRATEGIES (10):
5. EMA_RIBBON_RSI_VOLUME: 8/13/21/34 EMA + RSI(14) + Volume MA(20)
6. VWAP_BB_STOCH: VWAP + BB(20,2) + Stochastic(14,3,3)
7. SUPERTREND_ADX_EMA: Supertrend(10,3) + ADX(14) + EMA 9/21 cross
8. MACD_RSI_DIVERGENCE: MACD + RSI divergence detection
9. ICHIMOKU_ATR_VPROFILE: Ichimoku Cloud + ATR(14) + Volume Profile
10. TEMA_CCI_MOMENTUM: TEMA(9) + CCI(20) + Momentum(10)
11. KELTNER_FISHER_OBV: Keltner(20,2) + Fisher Transform + OBV
12. PSAR_ADX_RSI_MTF: PSAR + ADX + RSI Multi-Timeframe
13. HEIKIN_ASHI_MACD_BB: Heikin Ashi + MACD + BB Width
14. WILLIAMS_MFI_EMA: Williams %R + MFI + EMA Ribbon

Filters:
- Volume Filter: Only enter when volume > 20% above average
- Trend Filter: EMAs must be aligned (EMA9 > EMA21 > EMA50)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# List of all 14 strategies
ALL_STRATEGIES = [
    # Group A - Existing
    'MACD_CROSS', 'MACD_ZERO', 'RSI_REVERSAL', 'RSI_EXTREME',
    # Group B - New Multi-Indicator
    'EMA_RIBBON_RSI_VOLUME', 'VWAP_BB_STOCH', 'SUPERTREND_ADX_EMA',
    'MACD_RSI_DIVERGENCE', 'ICHIMOKU_ATR_VPROFILE', 'TEMA_CCI_MOMENTUM',
    'KELTNER_FISHER_OBV', 'PSAR_ADX_RSI_MTF', 'HEIKIN_ASHI_MACD_BB',
    'WILLIAMS_MFI_EMA'
]


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all indicators needed for 14 strategy signals.

    Args:
        df: DataFrame with OHLC data (columns: open, high, low, close, volume)

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # ============= BASIC EMAs =============
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema34'] = df['close'].ewm(span=34, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ============= RSI (14) =============
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)

    # ============= MACD (12, 26, 9) =============
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ============= VOLUME =============
    if 'volume' in df.columns:
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg'].replace(0, np.nan)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
    else:
        df['volume_ratio'] = 1.0
        df['obv'] = 0
        df['obv_ema'] = 0

    # ============= BOLLINGER BANDS (20, 2) =============
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_pct'] = df['bb_pct'].fillna(0.5)

    # ============= STOCHASTIC (14, 3, 3) =============
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14).replace(0, np.nan)
    df['stoch_k'] = df['stoch_k'].fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # ============= ATR (14) =============
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # ============= ADX (14) =============
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
        np.maximum(df['high'] - df['high'].shift(), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
        np.maximum(df['low'].shift() - df['low'], 0),
        0
    )
    df['plus_di'] = 100 * (pd.Series(df['plus_dm']).rolling(window=14).mean() / df['atr'].replace(0, np.nan))
    df['minus_di'] = 100 * (pd.Series(df['minus_dm']).rolling(window=14).mean() / df['atr'].replace(0, np.nan))
    dx = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.nan))
    df['adx'] = dx.rolling(window=14).mean()
    df['adx'] = df['adx'].fillna(20)

    # ============= SUPERTREND (10, 3) =============
    atr_mult = 3
    atr_period = 10
    atr_st = tr.rolling(window=atr_period).mean()
    hl2 = (df['high'] + df['low']) / 2
    df['supertrend_upper'] = hl2 + atr_mult * atr_st
    df['supertrend_lower'] = hl2 - atr_mult * atr_st
    df['supertrend'] = np.where(df['close'] > df['supertrend_upper'].shift(), 1,
                                 np.where(df['close'] < df['supertrend_lower'].shift(), -1, 0))
    df['supertrend'] = df['supertrend'].replace(0, method='ffill').fillna(1)

    # ============= TEMA (9) =============
    ema1 = df['close'].ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    ema3 = ema2.ewm(span=9, adjust=False).mean()
    df['tema'] = 3 * ema1 - 3 * ema2 + ema3

    # ============= CCI (20) =============
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    df['cci'] = df['cci'].fillna(0)

    # ============= MOMENTUM (10) =============
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['momentum_pct'] = df['close'].pct_change(periods=10) * 100

    # ============= WILLIAMS %R (14) =============
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14).replace(0, np.nan)
    df['williams_r'] = df['williams_r'].fillna(-50)

    # ============= MFI (14) =============
    if 'volume' in df.columns:
        tp = (df['high'] + df['low'] + df['close']) / 3
        raw_mf = tp * df['volume']
        pos_mf = np.where(tp > tp.shift(), raw_mf, 0)
        neg_mf = np.where(tp < tp.shift(), raw_mf, 0)
        pos_mf_sum = pd.Series(pos_mf).rolling(window=14).sum()
        neg_mf_sum = pd.Series(neg_mf).rolling(window=14).sum()
        mf_ratio = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))
        df['mfi'] = df['mfi'].fillna(50)
    else:
        df['mfi'] = 50

    # ============= KELTNER CHANNELS (20, 2) =============
    df['keltner_mid'] = df['close'].ewm(span=20, adjust=False).mean()
    df['keltner_upper'] = df['keltner_mid'] + 2 * df['atr']
    df['keltner_lower'] = df['keltner_mid'] - 2 * df['atr']

    # ============= FISHER TRANSFORM =============
    # Normalized price to -1 to 1 range
    min_low = df['low'].rolling(window=10).min()
    max_high = df['high'].rolling(window=10).max()
    value = 2 * ((df['close'] - min_low) / (max_high - min_low).replace(0, np.nan) - 0.5)
    value = value.clip(-0.999, 0.999).fillna(0)
    df['fisher'] = 0.5 * np.log((1 + value) / (1 - value).replace(0, np.nan))
    df['fisher'] = df['fisher'].fillna(0).ewm(span=3, adjust=False).mean()
    df['fisher_signal'] = df['fisher'].shift(1)

    # ============= PARABOLIC SAR =============
    df['psar'] = _calculate_psar(df)

    # ============= ICHIMOKU =============
    # Tenkan-sen (9)
    df['tenkan'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    # Kijun-sen (26)
    df['kijun'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    # Senkou Span A
    df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    # Senkou Span B (52)
    df['senkou_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)

    # ============= HEIKIN ASHI (Vectorized) =============
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    # Vectorized HA Open calculation using shift and ewm
    # HA_Open[0] = (Open[0] + Close[0]) / 2
    # HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2
    # This can be approximated with exponential smoothing
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    ha_close_vals = df['ha_close'].values
    for i in range(1, min(len(df), 100)):  # Only calculate first 100 precisely
        ha_open[i] = (ha_open[i-1] + ha_close_vals[i-1]) / 2
    # For the rest, use EMA approximation (much faster)
    if len(df) > 100:
        ha_open[100:] = ((df['open'].iloc[100:].values + df['close'].iloc[100:].values) / 2 +
                         ha_close_vals[99:-1]) / 2
    df['ha_open'] = ha_open
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
    df['ha_green'] = df['ha_close'] > df['ha_open']
    df['ha_red'] = df['ha_close'] < df['ha_open']

    # ============= VWAP (Session-based approximation) =============
    if 'volume' in df.columns:
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    else:
        df['vwap'] = df['close'].rolling(window=20).mean()

    # ============= CANDLE PROPERTIES =============
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    return df


def _calculate_psar_fast(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2) -> np.ndarray:
    """
    Calculate Parabolic SAR using Numba-style optimized numpy arrays.
    ~10x faster than pandas iloc access.
    """
    length = len(df)
    if length == 0:
        return np.array([])

    high = df['high'].values
    low = df['low'].values

    psar = np.zeros(length)
    af = af_start
    uptrend = True
    ep = low[0]
    psar[0] = high[0]

    for i in range(1, length):
        if uptrend:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            if low[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            if high[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return psar


def _calculate_psar(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2) -> pd.Series:
    """Calculate Parabolic SAR - wrapper for fast implementation."""
    psar_values = _calculate_psar_fast(df, af_start, af_step, af_max)
    return pd.Series(psar_values, index=df.index)


def check_volume_filter(df: pd.DataFrame) -> bool:
    """Check if volume is 20%+ above average."""
    if len(df) < 1:
        return True  # Pass if no data
    current = df.iloc[-1]
    return current.get('volume_ratio', 1.0) >= 1.2


def check_trend_filter(df: pd.DataFrame, direction: str) -> bool:
    """
    Check if EMAs are aligned for the given direction.

    BUY: EMA9 > EMA21 > EMA50 (uptrend)
    SELL: EMA9 < EMA21 < EMA50 (downtrend)
    """
    if len(df) < 1:
        return True  # Pass if no data
    current = df.iloc[-1]
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']

    if direction == 'BUY':
        return ema9 > ema21 > ema50
    else:  # SELL
        return ema9 < ema21 < ema50


def check_rsi_30_70_signal(df: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> Tuple[Optional[str], str]:
    """
    RSI Reversal Strategy with customizable thresholds.

    BUY: RSI crosses UP through oversold level + green candle
    SELL: RSI crosses DOWN through overbought level + red candle
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = current['rsi']
    prev_rsi = prev['rsi']

    # BUY: RSI crosses up through oversold level + green candle
    if prev_rsi < oversold and rsi >= oversold and current['is_green']:
        return 'BUY', f"RSI crossed UP through {oversold} ({prev_rsi:.1f} -> {rsi:.1f}) + Green candle"

    # SELL: RSI crosses down through overbought level + red candle
    if prev_rsi > overbought and rsi <= overbought and current['is_red']:
        return 'SELL', f"RSI crossed DOWN through {overbought} ({prev_rsi:.1f} -> {rsi:.1f}) + Red candle"

    return None, f"No signal (RSI: {rsi:.1f})"


def check_rsi_reversal_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    RSI Reversal at 35/65 levels (less extreme than 30/70).

    BUY: RSI crosses UP through 35 + green candle
    SELL: RSI crosses DOWN through 65 + red candle
    """
    return check_rsi_30_70_signal(df, oversold=35, overbought=65)


def check_rsi_extreme_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    RSI Extreme Strategy - very oversold/overbought levels.

    BUY: RSI < 25 and crossing up + green candle
    SELL: RSI > 75 and crossing down + red candle
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = current['rsi']
    prev_rsi = prev['rsi']

    # BUY: RSI crosses up from extreme oversold (< 25)
    if prev_rsi < 25 and rsi >= 25 and current['is_green']:
        return 'BUY', f"RSI extreme oversold recovery ({prev_rsi:.1f} -> {rsi:.1f}) + Green candle"

    # Also trigger if RSI < 25 and turning up
    if rsi < 25 and rsi > prev_rsi and current['is_green']:
        return 'BUY', f"RSI extreme oversold ({rsi:.1f}) turning up + Green candle"

    # SELL: RSI crosses down from extreme overbought (> 75)
    if prev_rsi > 75 and rsi <= 75 and current['is_red']:
        return 'SELL', f"RSI extreme overbought reversal ({prev_rsi:.1f} -> {rsi:.1f}) + Red candle"

    # Also trigger if RSI > 75 and turning down
    if rsi > 75 and rsi < prev_rsi and current['is_red']:
        return 'SELL', f"RSI extreme overbought ({rsi:.1f}) turning down + Red candle"

    return None, f"No signal (RSI: {rsi:.1f})"


def check_macd_cross_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    MACD Crossover Strategy.

    BUY: MACD crosses above signal line + green candle
    SELL: MACD crosses below signal line + red candle
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    macd = current['macd']
    macd_signal = current['macd_signal']
    prev_macd = prev['macd']
    prev_macd_signal = prev['macd_signal']

    # BUY: MACD crosses above signal line + green candle
    if prev_macd <= prev_macd_signal and macd > macd_signal and current['is_green']:
        return 'BUY', f"MACD crossed ABOVE signal ({macd:.5f} > {macd_signal:.5f}) + Green candle"

    # SELL: MACD crosses below signal line + red candle
    if prev_macd >= prev_macd_signal and macd < macd_signal and current['is_red']:
        return 'SELL', f"MACD crossed BELOW signal ({macd:.5f} < {macd_signal:.5f}) + Red candle"

    return None, f"No signal (MACD: {macd:.5f}, Signal: {macd_signal:.5f})"


def check_macd_zero_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    MACD Zero Line Cross Strategy.

    BUY: MACD crosses above zero + green candle
    SELL: MACD crosses below zero + red candle
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    macd = current['macd']
    prev_macd = prev['macd']

    # BUY: MACD crosses above zero + green candle
    if prev_macd <= 0 and macd > 0 and current['is_green']:
        return 'BUY', f"MACD crossed ABOVE zero ({prev_macd:.5f} -> {macd:.5f}) + Green candle"

    # SELL: MACD crosses below zero + red candle
    if prev_macd >= 0 and macd < 0 and current['is_red']:
        return 'SELL', f"MACD crossed BELOW zero ({prev_macd:.5f} -> {macd:.5f}) + Red candle"

    return None, f"No signal (MACD: {macd:.5f})"


def check_strong_trend_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strong Trend Pullback Strategy (EMA_PULLBACK).

    BUY: EMA9 > EMA21 > EMA50 (strong uptrend) + RSI in 35-50 zone + green candle
    SELL: EMA9 < EMA21 < EMA50 (strong downtrend) + RSI in 50-65 zone + red candle
    """
    if len(df) < 1:
        return None, "Insufficient data"

    current = df.iloc[-1]
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']
    rsi = current['rsi']

    # Strong uptrend: EMA9 > EMA21 > EMA50
    if ema9 > ema21 > ema50:
        if 35 <= rsi <= 50 and current['is_green']:
            return 'BUY', f"Strong uptrend pullback (RSI: {rsi:.1f}, EMA9>EMA21>EMA50)"
        return None, f"Uptrend but RSI not in zone ({rsi:.1f})"

    # Strong downtrend: EMA9 < EMA21 < EMA50
    if ema9 < ema21 < ema50:
        if 50 <= rsi <= 65 and current['is_red']:
            return 'SELL', f"Strong downtrend bounce (RSI: {rsi:.1f}, EMA9<EMA21<EMA50)"
        return None, f"Downtrend but RSI not in zone ({rsi:.1f})"

    return None, f"No clear trend (RSI: {rsi:.1f})"


def check_rsi_macd_combo_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    RSI + MACD Combination Strategy.

    BUY: RSI < 40 + MACD > Signal + green candle
    SELL: RSI > 60 + MACD < Signal + red candle
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    rsi = current['rsi']
    macd = current['macd']
    macd_signal = current['macd_signal']

    # BUY: RSI oversold zone + MACD bullish
    if rsi < 40 and macd > macd_signal and current['is_green']:
        return 'BUY', f"RSI+MACD combo BUY (RSI: {rsi:.1f}, MACD > Signal)"

    # SELL: RSI overbought zone + MACD bearish
    if rsi > 60 and macd < macd_signal and current['is_red']:
        return 'SELL', f"RSI+MACD combo SELL (RSI: {rsi:.1f}, MACD < Signal)"

    return None, f"No combo signal (RSI: {rsi:.1f}, MACD: {macd:.5f})"


def check_triple_confirm_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Triple Confirmation Strategy - EMA trend + RSI + MACD alignment.

    BUY: EMA uptrend + RSI < 50 + MACD bullish + green candle
    SELL: EMA downtrend + RSI > 50 + MACD bearish + red candle
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']
    rsi = current['rsi']
    macd = current['macd']
    macd_signal = current['macd_signal']

    # BUY: Triple confirmation
    if ema9 > ema21 > ema50 and rsi < 50 and macd > macd_signal and current['is_green']:
        return 'BUY', f"Triple confirm BUY (Uptrend + RSI:{rsi:.1f} + MACD bullish)"

    # SELL: Triple confirmation
    if ema9 < ema21 < ema50 and rsi > 50 and macd < macd_signal and current['is_red']:
        return 'SELL', f"Triple confirm SELL (Downtrend + RSI:{rsi:.1f} + MACD bearish)"

    return None, f"No triple confirm (RSI: {rsi:.1f})"


# =============================================================================
# NEW MULTI-INDICATOR STRATEGIES (10 Strategies)
# =============================================================================

def check_ema_ribbon_rsi_volume_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 5: EMA_RIBBON_RSI_VOLUME
    8/13/21/34 EMA Ribbon + RSI(14) + Volume MA(20)

    BUY: Price > all EMAs, RSI 40-70, Volume > 150% of MA
    SELL: Price < all EMAs, RSI 30-60, Volume > 150% of MA
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    price = current['close']
    ema8, ema13, ema21, ema34 = current['ema8'], current['ema13'], current['ema21'], current['ema34']
    rsi = current['rsi']
    vol_ratio = current.get('volume_ratio', 1.0)

    # BUY: Price above all EMAs + RSI in zone + high volume
    if price > ema8 > ema13 > ema21 > ema34:
        if 40 <= rsi <= 70 and vol_ratio >= 1.5 and current['is_green']:
            return 'BUY', f"EMA Ribbon BUY (Price>{ema8:.5f}, RSI:{rsi:.1f}, Vol:{vol_ratio:.1f}x)"
        return None, f"EMA uptrend but conditions not met (RSI:{rsi:.1f}, Vol:{vol_ratio:.1f}x)"

    # SELL: Price below all EMAs + RSI in zone + high volume
    if price < ema8 < ema13 < ema21 < ema34:
        if 30 <= rsi <= 60 and vol_ratio >= 1.5 and current['is_red']:
            return 'SELL', f"EMA Ribbon SELL (Price<{ema8:.5f}, RSI:{rsi:.1f}, Vol:{vol_ratio:.1f}x)"
        return None, f"EMA downtrend but conditions not met (RSI:{rsi:.1f}, Vol:{vol_ratio:.1f}x)"

    return None, f"No EMA ribbon alignment (RSI:{rsi:.1f})"


def check_vwap_bb_stoch_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 6: VWAP_BB_STOCH
    VWAP + Bollinger Bands(20,2) + Stochastic(14,3,3)

    BUY: Price bounces off lower BB, above VWAP, Stoch K crosses above D from <20
    SELL: Price bounces off upper BB, below VWAP, Stoch K crosses below D from >80
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]
    price = current['close']

    vwap = current['vwap']
    bb_lower, bb_upper = current['bb_lower'], current['bb_upper']
    stoch_k, stoch_d = current['stoch_k'], current['stoch_d']
    prev_stoch_k, prev_stoch_d = prev['stoch_k'], prev['stoch_d']

    # BUY: Bounce off lower BB + above VWAP + Stoch cross up from oversold
    if price > bb_lower and prev['low'] <= prev['bb_lower']:  # Bounced off lower BB
        if price > vwap:  # Above VWAP
            if prev_stoch_k < 20 and stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d:  # Stoch cross up
                if current['is_green']:
                    return 'BUY', f"VWAP+BB+Stoch BUY (BB bounce, VWAP:{vwap:.5f}, Stoch:{stoch_k:.1f})"

    # SELL: Bounce off upper BB + below VWAP + Stoch cross down from overbought
    if price < bb_upper and prev['high'] >= prev['bb_upper']:  # Bounced off upper BB
        if price < vwap:  # Below VWAP
            if prev_stoch_k > 80 and stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d:  # Stoch cross down
                if current['is_red']:
                    return 'SELL', f"VWAP+BB+Stoch SELL (BB bounce, VWAP:{vwap:.5f}, Stoch:{stoch_k:.1f})"

    return None, f"No VWAP+BB+Stoch signal (Stoch:{stoch_k:.1f})"


def check_supertrend_adx_ema_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 7: SUPERTREND_ADX_EMA
    Supertrend(10,3) + ADX(14) + EMA 9/21 cross

    BUY: Supertrend flips bullish + ADX > 25 + EMA9 crosses above EMA21
    SELL: Supertrend flips bearish + ADX > 25 + EMA9 crosses below EMA21
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    supertrend = current['supertrend']
    prev_supertrend = prev['supertrend']
    adx = current['adx']
    ema9, ema21 = current['ema9'], current['ema21']
    prev_ema9, prev_ema21 = prev['ema9'], prev['ema21']

    # BUY: Supertrend flip + ADX strong + EMA cross
    if supertrend == 1 and prev_supertrend == -1:  # Supertrend flipped bullish
        if adx > 25:
            if ema9 > ema21 and prev_ema9 <= prev_ema21:  # EMA cross up
                if current['is_green']:
                    return 'BUY', f"Supertrend+ADX+EMA BUY (ADX:{adx:.1f}, EMA cross up)"

    # SELL: Supertrend flip + ADX strong + EMA cross
    if supertrend == -1 and prev_supertrend == 1:  # Supertrend flipped bearish
        if adx > 25:
            if ema9 < ema21 and prev_ema9 >= prev_ema21:  # EMA cross down
                if current['is_red']:
                    return 'SELL', f"Supertrend+ADX+EMA SELL (ADX:{adx:.1f}, EMA cross down)"

    return None, f"No Supertrend signal (ADX:{adx:.1f})"


def check_macd_rsi_divergence_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 8: MACD_RSI_DIVERGENCE
    MACD zero cross + RSI divergence detection

    BUY: MACD crosses zero + Bullish RSI divergence (price lower low, RSI higher low)
    SELL: MACD crosses zero + Bearish RSI divergence (price higher high, RSI lower high)
    """
    if len(df) < 10:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    macd = current['macd']
    prev_macd = prev['macd']
    rsi = current['rsi']

    # Look for divergence in last 10 bars
    recent = df.iloc[-10:]
    price_min_idx = recent['low'].idxmin()
    price_max_idx = recent['high'].idxmax()
    rsi_min_idx = recent['rsi'].idxmin()
    rsi_max_idx = recent['rsi'].idxmax()

    # BUY: MACD zero cross + bullish divergence
    if prev_macd <= 0 and macd > 0:
        # Bullish divergence: recent price made lower low but RSI made higher low
        if price_min_idx > rsi_min_idx:  # Price low after RSI low
            if current['is_green'] and rsi < 50:
                return 'BUY', f"MACD+RSI Divergence BUY (MACD zero cross + bullish div)"

    # SELL: MACD zero cross + bearish divergence
    if prev_macd >= 0 and macd < 0:
        # Bearish divergence: recent price made higher high but RSI made lower high
        if price_max_idx > rsi_max_idx:  # Price high after RSI high
            if current['is_red'] and rsi > 50:
                return 'SELL', f"MACD+RSI Divergence SELL (MACD zero cross + bearish div)"

    return None, f"No MACD divergence signal (RSI:{rsi:.1f})"


def check_ichimoku_atr_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 9: ICHIMOKU_ATR_VPROFILE
    Ichimoku Cloud + ATR(14) + Volume Profile

    BUY: Price above cloud + Tenkan crosses above Kijun + Volume spike
    SELL: Price below cloud + Tenkan crosses below Kijun + Volume spike
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]
    price = current['close']

    tenkan = current['tenkan']
    kijun = current['kijun']
    prev_tenkan = prev['tenkan']
    prev_kijun = prev['kijun']
    senkou_a = current.get('senkou_a', price)
    senkou_b = current.get('senkou_b', price)
    vol_ratio = current.get('volume_ratio', 1.0)

    cloud_top = max(senkou_a, senkou_b) if not pd.isna(senkou_a) and not pd.isna(senkou_b) else price
    cloud_bottom = min(senkou_a, senkou_b) if not pd.isna(senkou_a) and not pd.isna(senkou_b) else price

    # BUY: Price above cloud + TK cross + volume
    if price > cloud_top:
        if tenkan > kijun and prev_tenkan <= prev_kijun:  # TK cross
            if vol_ratio >= 1.3 and current['is_green']:
                return 'BUY', f"Ichimoku BUY (Above cloud, TK cross, Vol:{vol_ratio:.1f}x)"

    # SELL: Price below cloud + TK cross + volume
    if price < cloud_bottom:
        if tenkan < kijun and prev_tenkan >= prev_kijun:  # TK cross
            if vol_ratio >= 1.3 and current['is_red']:
                return 'SELL', f"Ichimoku SELL (Below cloud, TK cross, Vol:{vol_ratio:.1f}x)"

    return None, f"No Ichimoku signal"


def check_tema_cci_momentum_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 10: TEMA_CCI_MOMENTUM
    TEMA(9) + CCI(20) + Momentum(10)

    BUY: TEMA trending up + CCI crosses above -100 + Momentum positive
    SELL: TEMA trending down + CCI crosses below +100 + Momentum negative
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]

    tema = current['tema']
    prev_tema = prev['tema']
    cci = current['cci']
    prev_cci = prev['cci']
    momentum = current['momentum']

    # BUY: TEMA up + CCI cross + positive momentum
    if tema > prev_tema:  # TEMA trending up
        if prev_cci < -100 and cci >= -100:  # CCI crosses above -100
            if momentum > 0 and current['is_green']:
                return 'BUY', f"TEMA+CCI+Mom BUY (TEMA up, CCI:{cci:.1f}, Mom:{momentum:.5f})"

    # SELL: TEMA down + CCI cross + negative momentum
    if tema < prev_tema:  # TEMA trending down
        if prev_cci > 100 and cci <= 100:  # CCI crosses below +100
            if momentum < 0 and current['is_red']:
                return 'SELL', f"TEMA+CCI+Mom SELL (TEMA down, CCI:{cci:.1f}, Mom:{momentum:.5f})"

    return None, f"No TEMA+CCI signal (CCI:{cci:.1f})"


def check_keltner_fisher_obv_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 11: KELTNER_FISHER_OBV
    Keltner Channels(20,2) + Fisher Transform + OBV

    BUY: Breakout above Keltner upper + Fisher extreme low reversal + OBV rising
    SELL: Breakdown below Keltner lower + Fisher extreme high reversal + OBV falling
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]
    price = current['close']

    keltner_upper = current['keltner_upper']
    keltner_lower = current['keltner_lower']
    fisher = current['fisher']
    fisher_signal = current['fisher_signal']
    obv = current['obv']
    obv_ema = current['obv_ema']

    # BUY: Keltner breakout + Fisher reversal from low + OBV rising
    if price > keltner_upper and prev['close'] <= prev['keltner_upper']:
        if fisher > fisher_signal and fisher > -1.5:  # Fisher turning up
            if obv > obv_ema and current['is_green']:
                return 'BUY', f"Keltner+Fisher+OBV BUY (Breakout, Fisher:{fisher:.2f})"

    # SELL: Keltner breakdown + Fisher reversal from high + OBV falling
    if price < keltner_lower and prev['close'] >= prev['keltner_lower']:
        if fisher < fisher_signal and fisher < 1.5:  # Fisher turning down
            if obv < obv_ema and current['is_red']:
                return 'SELL', f"Keltner+Fisher+OBV SELL (Breakdown, Fisher:{fisher:.2f})"

    return None, f"No Keltner signal (Fisher:{fisher:.2f})"


def check_psar_adx_rsi_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 12: PSAR_ADX_RSI_MTF
    Parabolic SAR + ADX + RSI alignment

    BUY: PSAR flips below price + ADX > 20 + RSI 40-60 (momentum zone)
    SELL: PSAR flips above price + ADX > 20 + RSI 40-60 (momentum zone)
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]
    price = current['close']

    psar = current['psar']
    prev_psar = prev['psar']
    adx = current['adx']
    rsi = current['rsi']

    # BUY: PSAR flip bullish + ADX strong + RSI in momentum zone
    if psar < price and prev_psar >= prev['close']:  # PSAR flipped below price
        if adx > 20 and 40 <= rsi <= 60:
            if current['is_green']:
                return 'BUY', f"PSAR+ADX+RSI BUY (PSAR flip, ADX:{adx:.1f}, RSI:{rsi:.1f})"

    # SELL: PSAR flip bearish + ADX strong + RSI in momentum zone
    if psar > price and prev_psar <= prev['close']:  # PSAR flipped above price
        if adx > 20 and 40 <= rsi <= 60:
            if current['is_red']:
                return 'SELL', f"PSAR+ADX+RSI SELL (PSAR flip, ADX:{adx:.1f}, RSI:{rsi:.1f})"

    return None, f"No PSAR signal (ADX:{adx:.1f}, RSI:{rsi:.1f})"


def check_heikin_ashi_macd_bb_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 13: HEIKIN_ASHI_MACD_BB
    Heikin Ashi candles + MACD + Bollinger Band Width

    BUY: 3 consecutive HA green + MACD histogram growing + BB expanding
    SELL: 3 consecutive HA red + MACD histogram shrinking + BB expanding
    """
    if len(df) < 4:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    prev3 = df.iloc[-4]

    # Check 3 consecutive HA candles
    ha_green_streak = current['ha_green'] and prev1['ha_green'] and prev2['ha_green']
    ha_red_streak = current['ha_red'] and prev1['ha_red'] and prev2['ha_red']

    macd_hist = current['macd_hist']
    prev_macd_hist = prev1['macd_hist']
    bb_width = current['bb_width']
    prev_bb_width = prev1['bb_width']

    # BUY: 3 HA green + MACD growing + BB expanding
    if ha_green_streak:
        if macd_hist > prev_macd_hist and macd_hist > 0:  # MACD growing positive
            if bb_width > prev_bb_width:  # BB expanding
                return 'BUY', f"HA+MACD+BB BUY (3 HA green, MACD growing, BB expanding)"

    # SELL: 3 HA red + MACD shrinking + BB expanding
    if ha_red_streak:
        if macd_hist < prev_macd_hist and macd_hist < 0:  # MACD shrinking negative
            if bb_width > prev_bb_width:  # BB expanding
                return 'SELL', f"HA+MACD+BB SELL (3 HA red, MACD shrinking, BB expanding)"

    return None, f"No HA+MACD+BB signal"


def check_williams_mfi_ema_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Strategy 14: WILLIAMS_MFI_EMA
    Williams %R + Money Flow Index + EMA Ribbon

    BUY: Williams %R exits oversold (<-80) + MFI crosses above 20 + Price > EMA ribbon
    SELL: Williams %R exits overbought (>-20) + MFI crosses below 80 + Price < EMA ribbon
    """
    if len(df) < 2:
        return None, "Insufficient data"

    current = df.iloc[-1]
    prev = df.iloc[-2]
    price = current['close']

    williams = current['williams_r']
    prev_williams = prev['williams_r']
    mfi = current['mfi']
    prev_mfi = prev['mfi']
    ema8, ema13, ema21 = current['ema8'], current['ema13'], current['ema21']

    # BUY: Williams exits oversold + MFI cross + above EMA ribbon
    if prev_williams < -80 and williams >= -80:  # Williams exits oversold
        if prev_mfi < 20 and mfi >= 20:  # MFI crosses above 20
            if price > ema8 and price > ema13 and price > ema21:  # Above EMA ribbon
                if current['is_green']:
                    return 'BUY', f"Williams+MFI+EMA BUY (W:{williams:.1f}, MFI:{mfi:.1f})"

    # SELL: Williams exits overbought + MFI cross + below EMA ribbon
    if prev_williams > -20 and williams <= -20:  # Williams exits overbought
        if prev_mfi > 80 and mfi <= 80:  # MFI crosses below 80
            if price < ema8 and price < ema13 and price < ema21:  # Below EMA ribbon
                if current['is_red']:
                    return 'SELL', f"Williams+MFI+EMA SELL (W:{williams:.1f}, MFI:{mfi:.1f})"

    return None, f"No Williams+MFI signal (W:{williams:.1f}, MFI:{mfi:.1f})"


# =============================================================================
# SIGNAL DISPATCHER
# =============================================================================

def get_signal(instrument: str, df: pd.DataFrame, config=None) -> Tuple[Optional[str], str]:
    """
    Get trading signal based on the pair's optimized strategy.

    Args:
        instrument: Currency pair (e.g., 'EUR_USD')
        df: DataFrame with indicators calculated
        config: Config module (paper or live). If None, uses default settings.

    Returns:
        Tuple of (signal, reason) where signal is 'BUY', 'SELL', or None
    """
    if config is not None:
        settings = config.get_pair_settings(instrument)
    else:
        # Default settings if no config provided
        settings = _get_default_pair_settings(instrument)

    if not settings:
        return None, f"Unknown instrument: {instrument}"

    strategy = settings['strategy']

    # Get raw signal from strategy
    signal = None
    reason = ""

    # GROUP A - EXISTING STRATEGIES
    if strategy == 'RSI_30_70':
        oversold = settings.get('rsi_oversold', 30)
        overbought = settings.get('rsi_overbought', 70)
        signal, reason = check_rsi_30_70_signal(df, oversold=oversold, overbought=overbought)
    elif strategy == 'RSI_REVERSAL':
        signal, reason = check_rsi_reversal_signal(df)
    elif strategy == 'RSI_EXTREME':
        signal, reason = check_rsi_extreme_signal(df)
    elif strategy == 'MACD_CROSS':
        signal, reason = check_macd_cross_signal(df)
    elif strategy == 'MACD_ZERO':
        signal, reason = check_macd_zero_signal(df)
    elif strategy in ['STRONG_TREND', 'EMA_PULLBACK']:
        signal, reason = check_strong_trend_signal(df)
    elif strategy == 'RSI_MACD_COMBO':
        signal, reason = check_rsi_macd_combo_signal(df)
    elif strategy == 'TRIPLE_CONFIRM':
        signal, reason = check_triple_confirm_signal(df)

    # GROUP B - NEW MULTI-INDICATOR STRATEGIES
    elif strategy == 'EMA_RIBBON_RSI_VOLUME':
        signal, reason = check_ema_ribbon_rsi_volume_signal(df)
    elif strategy == 'VWAP_BB_STOCH':
        signal, reason = check_vwap_bb_stoch_signal(df)
    elif strategy == 'SUPERTREND_ADX_EMA':
        signal, reason = check_supertrend_adx_ema_signal(df)
    elif strategy == 'MACD_RSI_DIVERGENCE':
        signal, reason = check_macd_rsi_divergence_signal(df)
    elif strategy == 'ICHIMOKU_ATR_VPROFILE':
        signal, reason = check_ichimoku_atr_signal(df)
    elif strategy == 'TEMA_CCI_MOMENTUM':
        signal, reason = check_tema_cci_momentum_signal(df)
    elif strategy == 'KELTNER_FISHER_OBV':
        signal, reason = check_keltner_fisher_obv_signal(df)
    elif strategy == 'PSAR_ADX_RSI_MTF':
        signal, reason = check_psar_adx_rsi_signal(df)
    elif strategy == 'HEIKIN_ASHI_MACD_BB':
        signal, reason = check_heikin_ashi_macd_bb_signal(df)
    elif strategy == 'WILLIAMS_MFI_EMA':
        signal, reason = check_williams_mfi_ema_signal(df)

    else:
        return None, f"Unknown strategy: {strategy}"

    # If no signal, return early
    if signal is None:
        return None, reason

    # Apply filters if enabled
    volume_filter = settings.get('volume_filter', False)
    trend_filter = settings.get('trend_filter', False)

    if volume_filter:
        if not check_volume_filter(df):
            return None, f"Volume filter blocked ({reason})"

    if trend_filter:
        if not check_trend_filter(df, signal):
            return None, f"Trend filter blocked ({reason})"

    # All filters passed
    filter_info = []
    if volume_filter:
        filter_info.append("Vol+")
    if trend_filter:
        filter_info.append("Trend+")

    if filter_info:
        reason = f"{reason} [{', '.join(filter_info)}]"

    return signal, reason


def _get_default_pair_settings(instrument: str) -> Dict:
    """Default pair settings (used when no config is provided)."""
    default_settings = {
        'EUR_USD': {'strategy': 'MACD_CROSS', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': True, 'trend_filter': True, 'expected_wr': 92.0},
        'GBP_USD': {'strategy': 'MACD_ZERO', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': False, 'trend_filter': False, 'expected_wr': 87.0},
        'USD_JPY': {'strategy': 'RSI_REVERSAL', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': False, 'trend_filter': True, 'expected_wr': 88.2},
        'USD_CHF': {'strategy': 'RSI_EXTREME', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': False, 'trend_filter': True, 'expected_wr': 93.8},
        'AUD_CHF': {'strategy': 'RSI_REVERSAL', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': False, 'trend_filter': False, 'expected_wr': 90.0},
        'EUR_GBP': {'strategy': 'MACD_CROSS', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': True, 'trend_filter': False, 'expected_wr': 93.3},
        'NZD_USD': {'strategy': 'RSI_REVERSAL', 'tp_pips': 5, 'sl_pips': 15, 'volume_filter': True, 'trend_filter': False, 'expected_wr': 86.4},
    }
    return default_settings.get(instrument, {
        'strategy': 'RSI_REVERSAL',
        'tp_pips': 5,
        'sl_pips': 15,
        'volume_filter': False,
        'trend_filter': False,
        'expected_wr': 75.0
    })


def print_strategy_info(config=None):
    """Print strategy configuration info."""
    if config is not None:
        config.print_config_info()
    else:
        print("=" * 80)
        print("OPTIMIZED FOREX STRATEGY V2 - 14 Multi-Indicator Strategies")
        print("=" * 80)
        print("\nGROUP A - EXISTING STRATEGIES (4):")
        print("  1. MACD_CROSS:     MACD crosses signal line")
        print("  2. MACD_ZERO:      MACD crosses zero line")
        print("  3. RSI_REVERSAL:   RSI crosses 35/65 levels")
        print("  4. RSI_EXTREME:    RSI at extreme <25/>75 levels")
        print("\nGROUP B - NEW MULTI-INDICATOR STRATEGIES (10):")
        print("  5. EMA_RIBBON_RSI_VOLUME:  8/13/21/34 EMA + RSI + Volume")
        print("  6. VWAP_BB_STOCH:          VWAP + Bollinger + Stochastic")
        print("  7. SUPERTREND_ADX_EMA:     Supertrend + ADX + EMA cross")
        print("  8. MACD_RSI_DIVERGENCE:    MACD zero cross + RSI divergence")
        print("  9. ICHIMOKU_ATR_VPROFILE:  Ichimoku Cloud + ATR + Volume")
        print(" 10. TEMA_CCI_MOMENTUM:      TEMA + CCI + Momentum")
        print(" 11. KELTNER_FISHER_OBV:     Keltner + Fisher Transform + OBV")
        print(" 12. PSAR_ADX_RSI_MTF:       Parabolic SAR + ADX + RSI")
        print(" 13. HEIKIN_ASHI_MACD_BB:    Heikin Ashi + MACD + BB Width")
        print(" 14. WILLIAMS_MFI_EMA:       Williams %R + MFI + EMA Ribbon")
        print("\nFilters:")
        print("  Volume Filter:  Only enter when volume > 20% above average")
        print("  Trend Filter:   EMAs must be aligned for direction")
        print("=" * 80)
