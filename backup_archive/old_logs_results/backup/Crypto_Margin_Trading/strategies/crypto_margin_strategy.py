"""
Crypto Margin Trading Strategy
==============================
Technical analysis strategies for crypto margin trading.
Adapted from the Forex optimized strategy with crypto-specific adjustments.

Strategy Types:
1. RSI_REVERSAL: RSI crosses 35/65 levels + candle confirmation
2. RSI_30_70: RSI crosses 30/70 levels (more extreme)
3. RSI_EXTREME: RSI at <25 or >75 levels
4. MACD_CROSS: MACD/Signal line crossover
5. MACD_ZERO: MACD crosses zero line
6. EMA_PULLBACK: Pullback in strong EMA trend
7. RSI_MACD_COMBO: Combined RSI + MACD signals
8. TRIPLE_CONFIRM: EMA trend + RSI + MACD alignment

Key Differences from Forex:
- Uses percentage-based TP/SL instead of pips
- 24/7 market (no weekend close)
- Higher volatility considerations
- Volume data always available (crypto)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for strategy signals.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # Ensure we have enough data
    if len(df) < 50:
        return df

    # EMAs (Exponential Moving Averages)
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI (Relative Strength Index - 14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Volume analysis
    if 'volume' in df.columns:
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
    else:
        df['volume_ratio'] = 1.0

    # ATR (Average True Range - for volatility)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100  # ATR as % of price

    # Candle color
    df['is_green'] = df['close'] > df['open']
    df['is_red'] = df['close'] < df['open']

    # Price change percentage
    df['change_pct'] = df['close'].pct_change() * 100

    return df


def check_volume_filter(df: pd.DataFrame, threshold: float = 1.2) -> bool:
    """
    Check if volume is above average (default 20% above).

    Args:
        df: DataFrame with indicators
        threshold: Volume ratio threshold (1.2 = 20% above average)

    Returns:
        True if volume filter passes
    """
    if len(df) < 1:
        return True  # Pass if no data
    current = df.iloc[-1]
    return current.get('volume_ratio', 1.0) >= threshold


def check_trend_filter(df: pd.DataFrame, direction: str) -> bool:
    """
    Check if EMAs are aligned for the given direction.

    BUY: EMA9 > EMA21 > EMA50 (uptrend)
    SELL: EMA9 < EMA21 < EMA50 (downtrend)

    Args:
        df: DataFrame with indicators
        direction: 'BUY' or 'SELL'

    Returns:
        True if trend filter passes
    """
    if len(df) < 1:
        return True
    current = df.iloc[-1]
    ema9 = current['ema9']
    ema21 = current['ema21']
    ema50 = current['ema50']

    if direction == 'BUY':
        return ema9 > ema21 > ema50
    else:  # SELL
        return ema9 < ema21 < ema50


def check_volatility_filter(df: pd.DataFrame, max_atr_pct: float = 2.0) -> bool:
    """
    Check if volatility is within acceptable range.

    Args:
        df: DataFrame with indicators
        max_atr_pct: Maximum ATR as percentage of price

    Returns:
        True if volatility is acceptable
    """
    if len(df) < 1:
        return True
    current = df.iloc[-1]
    atr_pct = current.get('atr_pct', 0)
    return atr_pct <= max_atr_pct


# ==================== STRATEGY SIGNALS ====================

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
        return 'BUY', f"MACD crossed ABOVE signal + Green candle"

    # SELL: MACD crosses below signal line + red candle
    if prev_macd >= prev_macd_signal and macd < macd_signal and current['is_red']:
        return 'SELL', f"MACD crossed BELOW signal + Red candle"

    return None, f"No signal (MACD: {macd:.6f})"


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
        return 'BUY', f"MACD crossed ABOVE zero + Green candle"

    # SELL: MACD crosses below zero + red candle
    if prev_macd >= 0 and macd < 0 and current['is_red']:
        return 'SELL', f"MACD crossed BELOW zero + Red candle"

    return None, f"No signal (MACD: {macd:.6f})"


def check_ema_pullback_signal(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    EMA Pullback Strategy (Strong Trend Pullback).

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
            return 'BUY', f"Uptrend pullback (RSI: {rsi:.1f}, EMA9>EMA21>EMA50)"
        return None, f"Uptrend but RSI not in zone ({rsi:.1f})"

    # Strong downtrend: EMA9 < EMA21 < EMA50
    if ema9 < ema21 < ema50:
        if 50 <= rsi <= 65 and current['is_red']:
            return 'SELL', f"Downtrend bounce (RSI: {rsi:.1f}, EMA9<EMA21<EMA50)"
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

    return None, f"No combo signal (RSI: {rsi:.1f})"


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


# ==================== MAIN SIGNAL FUNCTION ====================

def get_signal(pair: str, df: pd.DataFrame, config=None) -> Tuple[Optional[str], str]:
    """
    Get trading signal based on the pair's configured strategy.

    Args:
        pair: Trading pair (e.g., 'BTCUSDT')
        df: DataFrame with indicators calculated
        config: Config module with pair settings

    Returns:
        Tuple of (signal, reason) where signal is 'BUY', 'SELL', or None
    """
    if config is not None:
        settings = config.get_pair_settings(pair)
    else:
        settings = _get_default_pair_settings(pair)

    if not settings:
        return None, f"Unknown pair: {pair}"

    strategy = settings['strategy']

    # Get raw signal from strategy
    signal = None
    reason = ""

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
        signal, reason = check_ema_pullback_signal(df)
    elif strategy == 'RSI_MACD_COMBO':
        signal, reason = check_rsi_macd_combo_signal(df)
    elif strategy == 'TRIPLE_CONFIRM':
        signal, reason = check_triple_confirm_signal(df)
    else:
        return None, f"Unknown strategy: {strategy}"

    # If no signal, return early
    if signal is None:
        return None, reason

    # Apply filters if enabled
    volume_filter = settings.get('volume_filter', False)
    trend_filter = settings.get('trend_filter', False)
    volatility_filter = settings.get('volatility_filter', False)
    max_atr_pct = settings.get('max_atr_pct', 3.0)

    if volume_filter:
        if not check_volume_filter(df):
            return None, f"Volume filter blocked ({reason})"

    if trend_filter:
        if not check_trend_filter(df, signal):
            return None, f"Trend filter blocked ({reason})"

    if volatility_filter:
        if not check_volatility_filter(df, max_atr_pct):
            return None, f"Volatility too high ({reason})"

    # All filters passed
    filter_info = []
    if volume_filter:
        filter_info.append("Vol+")
    if trend_filter:
        filter_info.append("Trend+")
    if volatility_filter:
        filter_info.append("ATR+")

    if filter_info:
        reason = f"{reason} [{', '.join(filter_info)}]"

    return signal, reason


def _get_default_pair_settings(pair: str) -> Dict:
    """
    Default pair settings (used when no config is provided).

    These are starting points - should be optimized through backtesting.
    """
    default_settings = {
        # Major pairs - RSI Reversal as default
        'BTCUSDT': {'strategy': 'RSI_REVERSAL', 'tp_pct': 0.5, 'sl_pct': 1.2, 'volume_filter': False, 'trend_filter': False},
        'ETHUSDT': {'strategy': 'RSI_REVERSAL', 'tp_pct': 0.5, 'sl_pct': 1.2, 'volume_filter': False, 'trend_filter': False},
        'SOLUSDT': {'strategy': 'MACD_CROSS', 'tp_pct': 0.8, 'sl_pct': 2.0, 'volume_filter': False, 'trend_filter': False},
        'XRPUSDT': {'strategy': 'RSI_REVERSAL', 'tp_pct': 0.5, 'sl_pct': 1.2, 'volume_filter': False, 'trend_filter': False},
        'DOGEUSDT': {'strategy': 'RSI_EXTREME', 'tp_pct': 1.0, 'sl_pct': 2.5, 'volume_filter': True, 'trend_filter': False},
        'LTCUSDT': {'strategy': 'RSI_REVERSAL', 'tp_pct': 0.5, 'sl_pct': 1.2, 'volume_filter': False, 'trend_filter': False},
        'ADAUSDT': {'strategy': 'RSI_REVERSAL', 'tp_pct': 0.5, 'sl_pct': 1.2, 'volume_filter': False, 'trend_filter': False},
        'LINKUSDT': {'strategy': 'MACD_CROSS', 'tp_pct': 0.6, 'sl_pct': 1.5, 'volume_filter': False, 'trend_filter': False},
        'AVAXUSDT': {'strategy': 'RSI_MACD_COMBO', 'tp_pct': 0.7, 'sl_pct': 1.8, 'volume_filter': False, 'trend_filter': False},
        'DOTUSDT': {'strategy': 'RSI_REVERSAL', 'tp_pct': 0.5, 'sl_pct': 1.2, 'volume_filter': False, 'trend_filter': False},
    }

    return default_settings.get(pair, {
        'strategy': 'RSI_REVERSAL',
        'tp_pct': 0.5,
        'sl_pct': 1.2,
        'volume_filter': False,
        'trend_filter': False
    })


def calculate_tp_sl(
    entry_price: float,
    direction: str,
    tp_pct: float,
    sl_pct: float
) -> Tuple[float, float]:
    """
    Calculate Take Profit and Stop Loss prices based on percentages.

    Args:
        entry_price: Entry price
        direction: 'BUY' or 'SELL'
        tp_pct: Take profit percentage
        sl_pct: Stop loss percentage

    Returns:
        Tuple of (take_profit_price, stop_loss_price)
    """
    if direction == 'BUY':
        take_profit = entry_price * (1 + tp_pct / 100)
        stop_loss = entry_price * (1 - sl_pct / 100)
    else:  # SELL
        take_profit = entry_price * (1 - tp_pct / 100)
        stop_loss = entry_price * (1 + sl_pct / 100)

    return take_profit, stop_loss


def print_strategy_info(config=None):
    """Print strategy configuration info."""
    print("=" * 80)
    print("CRYPTO MARGIN TRADING STRATEGY")
    print("=" * 80)
    print("\nStrategy Types:")
    print("  RSI_30_70:      RSI crosses 30/70 levels + candle confirmation")
    print("  RSI_REVERSAL:   RSI crosses 35/65 levels (default)")
    print("  RSI_EXTREME:    RSI at extreme <25/>75 levels")
    print("  MACD_CROSS:     MACD crosses signal line")
    print("  MACD_ZERO:      MACD crosses zero line")
    print("  EMA_PULLBACK:   Pullback in strong EMA trend")
    print("  RSI_MACD_COMBO: Combined RSI + MACD signals")
    print("  TRIPLE_CONFIRM: EMA trend + RSI + MACD alignment")
    print("\nFilters:")
    print("  Volume Filter:    Only enter when volume > 20% above average")
    print("  Trend Filter:     EMAs must be aligned for direction")
    print("  Volatility Filter: ATR must be below threshold")
    print("=" * 80)


if __name__ == "__main__":
    print_strategy_info()
