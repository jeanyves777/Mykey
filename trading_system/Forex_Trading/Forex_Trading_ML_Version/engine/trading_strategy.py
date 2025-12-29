"""
ML Trading Strategy for Forex ML System
======================================

Strategy logic using ensemble ML predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime


class MLTradingStrategy:
    """ML-powered trading strategy."""

    # Signal constants
    SELL = 0
    HOLD = 1
    BUY = 2

    def __init__(self, ensemble, feature_engineer, config=None):
        """
        Initialize ML trading strategy.

        Args:
            ensemble: EnsembleVotingSystem instance
            feature_engineer: FeatureEngineer instance
            config: TradingConfig object
        """
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer
        self.config = config

        # Signal thresholds
        self.confidence_threshold = 0.60
        self.min_agreement = 3
        self.total_models_display = 3  # Display as X/3

        if config and hasattr(config, 'ml'):
            self.confidence_threshold = config.ml.confidence_threshold
            self.min_agreement = config.ml.min_model_agreement
            self.total_models_display = getattr(config.ml, 'total_models_for_display', 3)

        # State tracking
        self.last_signal: Dict[str, int] = {}
        self.last_confidence: Dict[str, float] = {}
        self.signal_count: Dict[str, Dict[str, int]] = {}

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[Optional[str], float, int, str]:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Trading pair
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (signal, confidence, agreement, reason)
        """
        if df is None or len(df) < 200:
            return None, 0.0, 0, "Insufficient data"

        try:
            # Engineer features
            featured_df = self.feature_engineer.engineer_features(df.copy(), fit=False)

            # Get latest features
            if len(featured_df) == 0:
                return None, 0.0, 0, "Feature engineering failed"

            X = self.feature_engineer.get_feature_matrix(featured_df)
            if len(X) == 0:
                return None, 0.0, 0, "No features available"

            # Get prediction from ensemble
            prediction, confidence, agreement, details = self.ensemble.predict_single(X[-1])

            # Store state
            self.last_signal[symbol] = prediction
            self.last_confidence[symbol] = confidence

            # Track signal counts
            if symbol not in self.signal_count:
                self.signal_count[symbol] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

            signal_name = self._get_signal_name(prediction)
            self.signal_count[symbol][signal_name] += 1

            # Check if signal passes thresholds
            if not self.ensemble.should_trade(confidence, agreement):
                reason = f"Below threshold (conf={confidence:.2f}, agree={agreement}/{self.total_models_display})"
                return None, confidence, agreement, reason

            # Convert prediction to signal
            if prediction == self.BUY:
                signal = 'BUY'
                reason = f"ML BUY signal (conf={confidence:.2f}, agree={agreement}/{self.total_models_display})"
            elif prediction == self.SELL:
                signal = 'SELL'
                reason = f"ML SELL signal (conf={confidence:.2f}, agree={agreement}/{self.total_models_display})"
            else:
                signal = None
                reason = f"HOLD signal (conf={confidence:.2f})"

            # Add model details to reason
            if signal:
                model_votes = [f"{k[:2].upper()}:{self._get_signal_name(v)}"
                              for k, v in details['model_predictions'].items()]
                reason += f" | Votes: {', '.join(model_votes)}"

            return signal, confidence, agreement, reason

        except Exception as e:
            return None, 0.0, 0, f"Error: {str(e)}"

    def should_close_position(self, symbol: str, position_direction: str,
                              df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if position should be closed based on ML signal.

        Args:
            symbol: Trading pair
            position_direction: Current position direction ('BUY' or 'SELL')
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (should_close, reason)
        """
        signal, confidence, agreement, _ = self.generate_signal(symbol, df)

        if signal is None:
            return False, "No signal"

        # Close if opposite signal with high confidence
        if position_direction == 'BUY' and signal == 'SELL':
            if confidence >= self.confidence_threshold and agreement >= self.min_agreement:
                return True, f"Opposite SELL signal (conf={confidence:.2f})"

        if position_direction == 'SELL' and signal == 'BUY':
            if confidence >= self.confidence_threshold and agreement >= self.min_agreement:
                return True, f"Opposite BUY signal (conf={confidence:.2f})"

        return False, "No close signal"

    def get_position_direction(self, signal: str) -> str:
        """Get position direction from signal."""
        return signal if signal in ['BUY', 'SELL'] else None

    def get_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from dataframe."""
        if df is None or len(df) < period:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = np.mean(tr[-period:])
        return float(atr)

    def _get_signal_name(self, prediction: int) -> str:
        """Convert prediction to signal name."""
        if prediction == self.BUY:
            return 'BUY'
        elif prediction == self.SELL:
            return 'SELL'
        else:
            return 'HOLD'

    def get_signal_stats(self, symbol: str = None) -> Dict:
        """Get signal statistics."""
        if symbol:
            return self.signal_count.get(symbol, {'BUY': 0, 'SELL': 0, 'HOLD': 0})

        total = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for counts in self.signal_count.values():
            for signal, count in counts.items():
                total[signal] += count
        return total

    def reset_stats(self, symbol: str = None) -> None:
        """Reset signal statistics."""
        if symbol:
            self.signal_count[symbol] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        else:
            self.signal_count.clear()
            self.last_signal.clear()
            self.last_confidence.clear()
