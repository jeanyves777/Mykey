"""
Trend Filter for Smart DCA System
==================================

This module provides trend analysis to prevent DCA entries against strong trends.

Key Features:
1. ADX - Measures trend STRENGTH (not direction)
2. EMA crossover - Detects bullish/bearish reversals
3. Reversal candle recognition - Engulfing patterns, strong reversals

Usage:
    trend_filter = TrendFilter(config)
    can_add, reason = trend_filter.can_dca(df, position_direction='long', dca_level=2)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrendFilterConfig:
    """Configuration for trend filtering."""
    # ADX thresholds
    adx_period: int = 14
    adx_strong_trend: float = 30.0      # Don't DCA when ADX above this
    adx_weak_trend: float = 20.0        # Safe to DCA when ADX below this

    # EMA for trend direction
    ema_fast_period: int = 8
    ema_slow_period: int = 21

    # Reversal detection
    require_reversal_candle: bool = True
    min_reversal_body_ratio: float = 0.6  # Body must be 60% of candle range

    # Higher DCA levels need stronger confirmation
    high_dca_threshold: int = 3  # DCA level 3+ needs stronger reversal


class TrendFilter:
    """
    Trend filter to prevent DCA entries against strong trends.

    The goal is to avoid "averaging down into a falling knife" by:
    1. Checking trend strength with ADX
    2. Looking for reversal signals before DCA
    3. Being more conservative at higher DCA levels
    """

    def __init__(self, config: TrendFilterConfig = None):
        self.config = config or TrendFilterConfig()

    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index).
        ADX measures trend STRENGTH, not direction:
        - ADX < 20: Weak/no trend (consolidation)
        - ADX 20-30: Moderate trend
        - ADX > 30: Strong trend
        - ADX > 40: Very strong trend
        """
        df = df.copy()
        period = self.config.adx_period

        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']

        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'], 0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'], 0
        )

        # Smoothed averages
        df['atr'] = df['tr'].rolling(period).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(period).mean() / df['atr'])

        # DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        df['adx'] = df['dx'].rolling(period).mean()

        # Trend direction based on DI
        df['trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)

        return df

    def calculate_ema_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA crossover signals for reversal detection."""
        df = df.copy()

        df['ema_fast'] = df['close'].ewm(span=self.config.ema_fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config.ema_slow_period, adjust=False).mean()

        # Bullish crossover (fast crosses above slow)
        df['ema_bullish'] = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        )

        # Bearish crossover (fast crosses below slow)
        df['ema_bearish'] = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        )

        # Current EMA relationship
        df['ema_trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)

        return df

    def detect_reversal_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect reversal candle patterns:
        1. Bullish engulfing
        2. Bearish engulfing
        3. Strong reversal candles (large body in direction of reversal)
        """
        df = df.copy()

        # Candle metrics
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / (df['range'] + 1e-10)
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']

        # Previous candle info
        df['prev_body'] = df['body'].shift(1)
        df['prev_bullish'] = df['is_bullish'].shift(1)
        df['prev_bearish'] = df['is_bearish'].shift(1)

        # Bullish engulfing: Current bullish candle body > previous bearish body
        df['bullish_engulfing'] = (
            df['is_bullish'] &
            df['prev_bearish'] &
            (df['body'] > df['prev_body']) &
            (df['body_ratio'] >= self.config.min_reversal_body_ratio)
        )

        # Bearish engulfing: Current bearish candle body > previous bullish body
        df['bearish_engulfing'] = (
            df['is_bearish'] &
            df['prev_bullish'] &
            (df['body'] > df['prev_body']) &
            (df['body_ratio'] >= self.config.min_reversal_body_ratio)
        )

        # Strong bullish reversal: Large green candle after red
        df['strong_bullish'] = (
            df['is_bullish'] &
            df['prev_bearish'] &
            (df['body_ratio'] >= 0.7)  # Very strong body
        )

        # Strong bearish reversal: Large red candle after green
        df['strong_bearish'] = (
            df['is_bearish'] &
            df['prev_bullish'] &
            (df['body_ratio'] >= 0.7)
        )

        # Combined reversal signals
        df['bullish_reversal'] = df['bullish_engulfing'] | df['strong_bullish']
        df['bearish_reversal'] = df['bearish_engulfing'] | df['strong_bearish']

        return df

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all trend analysis on dataframe."""
        df = self.calculate_adx(df)
        df = self.calculate_ema_signals(df)
        df = self.detect_reversal_candle(df)
        return df

    def can_dca(self, df: pd.DataFrame, position_direction: str = 'long',
                dca_level: int = 0) -> Tuple[bool, str]:
        """
        Check if DCA entry is safe based on trend analysis.

        Args:
            df: OHLC dataframe (must have recent data)
            position_direction: 'long' or 'short'
            dca_level: Current DCA level (0=entry, 1-4=DCA stages)

        Returns:
            (can_dca, reason): Tuple of boolean and explanation string
        """
        if len(df) < 30:
            return True, "Insufficient data for trend analysis"

        # Run analysis
        df = self.analyze(df)
        latest = df.iloc[-1]

        adx = latest.get('adx', 0)
        trend_dir = latest.get('trend_direction', 0)  # 1=bullish, -1=bearish
        ema_trend = latest.get('ema_trend', 0)

        # Position direction as number
        pos_dir = 1 if position_direction.lower() == 'long' else -1

        # Check if trend is against our position
        trend_against = (trend_dir != pos_dir)
        ema_against = (ema_trend != pos_dir)

        # Weak trend - always OK to DCA
        if adx < self.config.adx_weak_trend:
            return True, f"Weak trend (ADX={adx:.1f}<{self.config.adx_weak_trend}) - safe to DCA"

        # Moderate trend (20-30 ADX)
        if adx < self.config.adx_strong_trend:
            if not trend_against:
                return True, f"Moderate trend in our favor (ADX={adx:.1f}) - OK to DCA"

            # Trend against us - check for reversal
            if self.config.require_reversal_candle:
                bullish_rev = latest.get('bullish_reversal', False)
                bearish_rev = latest.get('bearish_reversal', False)

                if pos_dir == 1 and bullish_rev:
                    return True, f"Bullish reversal detected (ADX={adx:.1f}) - DCA allowed"
                elif pos_dir == -1 and bearish_rev:
                    return True, f"Bearish reversal detected (ADX={adx:.1f}) - DCA allowed"
                else:
                    return False, f"Trend against us (ADX={adx:.1f}), no reversal - WAIT"

            return True, f"Moderate trend against us (ADX={adx:.1f}) - proceed with caution"

        # Strong trend (ADX > 30)
        if trend_against:
            # Higher DCA levels need stronger confirmation
            if dca_level >= self.config.high_dca_threshold:
                # Must have reversal AND EMA crossover
                bullish_cross = latest.get('ema_bullish', False)
                bearish_cross = latest.get('ema_bearish', False)
                bullish_rev = latest.get('bullish_reversal', False)
                bearish_rev = latest.get('bearish_reversal', False)

                if pos_dir == 1 and (bullish_cross or bullish_rev):
                    return True, f"Strong reversal at DCA {dca_level} (ADX={adx:.1f}) - DCA allowed"
                elif pos_dir == -1 and (bearish_cross or bearish_rev):
                    return True, f"Strong reversal at DCA {dca_level} (ADX={adx:.1f}) - DCA allowed"
                else:
                    return False, f"STRONG TREND (ADX={adx:.1f}) at DCA {dca_level} - BLOCK DCA"

            # Lower DCA levels - check for any reversal
            bullish_rev = latest.get('bullish_reversal', False)
            bearish_rev = latest.get('bearish_reversal', False)

            if pos_dir == 1 and bullish_rev:
                return True, f"Reversal in strong trend (ADX={adx:.1f}) - DCA allowed"
            elif pos_dir == -1 and bearish_rev:
                return True, f"Reversal in strong trend (ADX={adx:.1f}) - DCA allowed"

            return False, f"STRONG TREND AGAINST US (ADX={adx:.1f}) - NO REVERSAL - BLOCK DCA"

        # Strong trend in our favor
        return True, f"Strong trend in our favor (ADX={adx:.1f}) - OK to DCA"

    def validate_entry(self, df: pd.DataFrame, signal_direction: str = 'long') -> Tuple[bool, str]:
        """
        Validate if initial entry is safe based on trend.
        Prevents entering trades that go strongly against the current trend.

        Args:
            df: OHLC dataframe
            signal_direction: 'long' or 'short' from ML signal

        Returns:
            (can_enter, reason): Tuple of boolean and explanation
        """
        if len(df) < 30:
            return True, "Insufficient data for trend validation"

        df = self.analyze(df)
        latest = df.iloc[-1]

        adx = latest.get('adx', 0)
        trend_dir = latest.get('trend_direction', 0)
        ema_trend = latest.get('ema_trend', 0)

        signal_dir = 1 if signal_direction.lower() == 'long' else -1

        # If trend is very strong and against signal, block entry
        if adx > self.config.adx_strong_trend:
            if trend_dir != signal_dir and ema_trend != signal_dir:
                return False, f"STRONG COUNTER-TREND (ADX={adx:.1f}) - BLOCK ENTRY"

        # Check for momentum confirmation
        if adx > self.config.adx_weak_trend:
            if trend_dir == signal_dir:
                return True, f"Entry WITH trend (ADX={adx:.1f}) - CONFIRMED"
            else:
                # Counter-trend but not extremely strong
                bullish_rev = latest.get('bullish_reversal', False)
                bearish_rev = latest.get('bearish_reversal', False)

                if signal_dir == 1 and bullish_rev:
                    return True, f"Counter-trend LONG with reversal (ADX={adx:.1f}) - OK"
                elif signal_dir == -1 and bearish_rev:
                    return True, f"Counter-trend SHORT with reversal (ADX={adx:.1f}) - OK"

                return True, f"Counter-trend entry (ADX={adx:.1f}) - proceed with caution"

        # Weak trend - always OK
        return True, f"Weak trend (ADX={adx:.1f}) - entry OK"

    def get_trend_summary(self, df: pd.DataFrame) -> str:
        """Get a one-line summary of current market conditions."""
        if len(df) < 30:
            return "Insufficient data"

        df = self.analyze(df)
        latest = df.iloc[-1]

        adx = latest.get('adx', 0)
        trend_dir = latest.get('trend_direction', 0)
        ema_trend = latest.get('ema_trend', 0)

        # Trend strength
        if adx < self.config.adx_weak_trend:
            strength = "WEAK"
        elif adx < self.config.adx_strong_trend:
            strength = "MODERATE"
        else:
            strength = "STRONG"

        # Trend direction
        direction = "BULLISH" if trend_dir == 1 else "BEARISH"
        ema_dir = "above" if ema_trend == 1 else "below"

        # Reversal signals
        bullish_rev = latest.get('bullish_reversal', False)
        bearish_rev = latest.get('bearish_reversal', False)
        rev_status = ""
        if bullish_rev:
            rev_status = " | BULLISH REVERSAL"
        elif bearish_rev:
            rev_status = " | BEARISH REVERSAL"

        return f"{strength} {direction} (ADX={adx:.1f}) | EMA {ema_dir} slow{rev_status}"


# Test the trend filter
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Test with sample data
    data_dir = Path(__file__).parent / "Crypto_Data_Fresh"
    data_file = data_dir / "BTCUSD_1m.csv"

    if data_file.exists():
        df = pd.read_csv(data_file)
        df = df.tail(1000).reset_index(drop=True)

        print("=" * 70)
        print("TREND FILTER TEST")
        print("=" * 70)

        # Create filter
        config = TrendFilterConfig()
        filter = TrendFilter(config)

        # Analyze
        df = filter.analyze(df)

        # Get summary
        summary = filter.get_trend_summary(df)
        print(f"\nCurrent Market: {summary}")

        # Test DCA decisions
        print("\nDCA Decisions (LONG position):")
        for dca_level in range(5):
            can_dca, reason = filter.can_dca(df, 'long', dca_level)
            status = "OK" if can_dca else "BLOCK"
            print(f"  DCA {dca_level}: [{status}] {reason}")

        # Test entry validation
        print("\nEntry Validation:")
        for direction in ['long', 'short']:
            can_enter, reason = filter.validate_entry(df, direction)
            status = "OK" if can_enter else "BLOCK"
            print(f"  {direction.upper()}: [{status}] {reason}")

        # Show recent ADX values
        print("\nRecent ADX values:")
        for i in [-5, -4, -3, -2, -1]:
            row = df.iloc[i]
            print(f"  Bar {i}: ADX={row['adx']:.1f}, Trend={'UP' if row['trend_direction']==1 else 'DOWN'}")
    else:
        print(f"Data file not found: {data_file}")
