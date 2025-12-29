"""
High-Frequency Trading Strategy
================================

Core strategy logic for the HF Crypto Trading System.
Combines ensemble ML predictions with risk management rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """Represents a trading signal with metadata."""
    signal: SignalType
    confidence: float
    symbol: str
    timestamp: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    model_agreement: int
    features: Dict[str, float] = None
    reason: str = ""


@dataclass
class StrategyState:
    """Current state of the trading strategy."""
    is_active: bool = True
    current_position: SignalType = SignalType.HOLD
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    position_size: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = 0.0
    trades_today: int = 0
    daily_pnl: float = 0.0
    last_signal_time: Optional[datetime] = None


class HFTradingStrategy:
    """
    High-Frequency Trading Strategy using Ensemble ML for SPOT TRADING.

    Features:
    - BUY-ONLY signals (no short selling for spot crypto)
    - DCA (Dollar Cost Averaging) support for trades going against us
    - ML-based signal generation
    - Multiple confirmation filters
    - Dynamic position sizing
    - Risk management rules
    - Session-based trading
    """

    def __init__(self,
                 ensemble=None,
                 feature_engineer=None,
                 config: Dict = None):
        """
        Initialize the HF Trading Strategy.

        Args:
            ensemble: EnsembleVotingSystem instance
            feature_engineer: FeatureEngineer instance
            config: Strategy configuration dictionary
        """
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer
        self.state = StrategyState()

        # Default configuration for BUY-only HF day trading with DCA
        self.config = {
            'min_confidence': 0.60,        # V18 optimal
            'min_agreement': 5,            # V18 optimal: 5/5 unanimous
            'position_size_pct': 0.02,
            'max_position_value': 10000,
            'stop_loss_pct': 0.025,        # 2.5% final SL (after DCA exhausted)
            'take_profit_pct': 0.01,       # 1% TP (quick exit)
            'trailing_stop_pct': 0.006,    # 0.6% trailing stop
            'max_trades_per_day': 50,
            'max_daily_loss_pct': 0.05,
            'cooldown_seconds': 180,       # 3 min cooldown
            # NOTE: No max_holding_minutes - DCA handles exits
            'volume_filter_threshold': 1.2,
            'volatility_filter_max': 0.05,
            'trend_confirmation': True,
            'momentum_confirmation': True,
            'volume_confirmation': True,
            # DCA settings - TIGHT for HF day trading
            'buy_only_mode': True,         # BUY-only for spot trading
            'dca_enabled': True,
            'dca_trigger_pct': 0.005,      # 0.5% drop triggers first DCA
            'dca_spacing_pct': 0.005,      # 0.5% between each DCA level
            'max_dca_entries': 3,
            'dca_multiplier': 1.5,
            'dca_profit_target_pct': 0.004, # 0.4% profit on avg to exit
        }

        if config:
            self.config.update(config)

        # Data buffer for feature calculation
        self.data_buffer: List[Dict] = []
        self.buffer_size = 200  # Minimum bars needed for features

    def update_data(self, bar: Dict):
        """
        Update the data buffer with a new bar.

        Args:
            bar: OHLCV bar data
        """
        self.data_buffer.append(bar)

        # Keep buffer at manageable size
        if len(self.data_buffer) > self.buffer_size * 2:
            self.data_buffer = self.data_buffer[-self.buffer_size:]

    def get_data_as_df(self) -> pd.DataFrame:
        """Convert data buffer to DataFrame."""
        if not self.data_buffer:
            return pd.DataFrame()

        return pd.DataFrame(self.data_buffer)

    def generate_signal(self,
                        current_price: float,
                        account_equity: float,
                        symbol: str = "CRYPTO") -> Optional[TradeSignal]:
        """
        Generate a trading signal based on current market conditions.

        Args:
            current_price: Current market price
            account_equity: Current account equity
            symbol: Trading symbol

        Returns:
            TradeSignal if conditions are met, None otherwise
        """
        now = datetime.now()

        # Check if we have enough data
        if len(self.data_buffer) < self.buffer_size:
            logger.info(f"  [DATA] Buffering: {len(self.data_buffer)}/{self.buffer_size} bars")
            return None

        logger.info(f"  [DATA] Buffer ready: {len(self.data_buffer)} bars")

        # Check cooldown
        if self.state.last_signal_time:
            seconds_since_last = (now - self.state.last_signal_time).total_seconds()
            if seconds_since_last < self.config['cooldown_seconds']:
                remaining = self.config['cooldown_seconds'] - seconds_since_last
                logger.info(f"  [COOLDOWN] Waiting {remaining:.0f}s (cooldown: {self.config['cooldown_seconds']}s)")
                return None

        # Check daily limits
        if self.state.trades_today >= self.config['max_trades_per_day']:
            logger.info(f"  [LIMIT] Daily trade limit reached: {self.state.trades_today}/{self.config['max_trades_per_day']}")
            return None

        max_daily_loss = account_equity * self.config['max_daily_loss_pct']
        if self.state.daily_pnl < -max_daily_loss:
            logger.info(f"  [LIMIT] Daily loss limit reached: ${self.state.daily_pnl:.2f} (max: ${-max_daily_loss:.2f})")
            return None

        logger.info(f"  [STATUS] Trades today: {self.state.trades_today} | Daily P&L: ${self.state.daily_pnl:.2f}")

        # Get features and prediction
        try:
            df = self.get_data_as_df()
            logger.info(f"  [FEATURES] Generating features from {len(df)} bars...")

            if self.feature_engineer:
                features = self.feature_engineer.prepare_inference_data(df)
                logger.info(f"  [FEATURES] Generated {features.shape[1] if hasattr(features, 'shape') else 'N/A'} features")
            else:
                features = df

            if self.ensemble and self.ensemble.is_trained:
                logger.info(f"  [ML] Running 5-model ensemble prediction...")
                prediction, confidence, details = self.ensemble.get_trade_signal(features)
                model_agreement = details.get('agreement_count', [0])[0] if 'agreement_count' in details else 3

                # LOG: Detailed ML predictions
                logger.info(f"  [ML] ===== ENSEMBLE PREDICTION RESULTS =====")
                logger.info(f"  [ML] Prediction: {prediction} ({'BUY' if prediction == 1 else 'SELL' if prediction == -1 else 'HOLD'})")
                logger.info(f"  [ML] Confidence: {confidence:.2%}")
                logger.info(f"  [ML] Model Agreement: {model_agreement}/5 models agree")

                # Log individual model predictions if available
                if 'individual_predictions' in details:
                    logger.info(f"  [ML] ----- Individual Model Votes -----")
                    for model_name, pred in details['individual_predictions'].items():
                        vote = 'BUY' if pred == 1 else 'SELL' if pred == -1 else 'HOLD'
                        logger.info(f"  [ML]   {model_name}: {vote}")

                if 'individual_confidences' in details:
                    logger.info(f"  [ML] ----- Individual Confidences -----")
                    for model_name, conf in details['individual_confidences'].items():
                        logger.info(f"  [ML]   {model_name}: {conf:.2%}")

                logger.info(f"  [ML] ========================================")
            else:
                # Fallback to technical analysis
                logger.info(f"  [TA] Using technical analysis (ML not available)")
                prediction, confidence, model_agreement = self._technical_analysis_signal(df)
                logger.info(f"  [TA] Prediction: {prediction} | Confidence: {confidence:.2%} | Agreement: {model_agreement}")

        except Exception as e:
            logger.error(f"  [ERROR] Error generating prediction: {e}")
            return None

        # Check minimum confidence
        if confidence < self.config['min_confidence']:
            logger.info(f"  [FILTER] REJECTED - Confidence {confidence:.2%} < {self.config['min_confidence']:.2%} required")
            return None

        # Check minimum model agreement
        if model_agreement < self.config['min_agreement']:
            logger.info(f"  [FILTER] REJECTED - Agreement {model_agreement}/5 < {self.config['min_agreement']}/5 required")
            return None

        logger.info(f"  [FILTER] PASSED - Confidence: {confidence:.2%} | Agreement: {model_agreement}/5")

        # Apply filters
        if not self._apply_filters(df, prediction):
            logger.info(f"  [FILTER] REJECTED by trading filters")
            return None

        logger.info(f"  [FILTER] All filters PASSED")

        # Determine signal type - BUY ONLY for spot trading
        logger.info(f"  [SIGNAL] ===== GENERATING TRADE SIGNAL =====")
        logger.info(f"  [SIGNAL] Mode: {'BUY-ONLY (SPOT)' if self.config.get('buy_only_mode', True) else 'LONG/SHORT'}")

        if prediction == 1:
            signal_type = SignalType.BUY
            logger.info(f"  [SIGNAL] Direction: BUY (prediction=1)")
        elif prediction == -1:
            # BUY-ONLY MODE: Ignore SELL signals for spot trading
            if self.config.get('buy_only_mode', True):
                logger.info(f"  [SIGNAL] Direction: SELL (prediction=-1) - IGNORED (BUY-only mode)")
                logger.info(f"  [SIGNAL] Spot trading does not support short selling")
                return None
            else:
                signal_type = SignalType.SELL
                logger.info(f"  [SIGNAL] Direction: SELL (prediction=-1)")
        else:
            logger.info(f"  [SIGNAL] Direction: HOLD (prediction=0) - No trade")
            return None

        # Calculate position sizing
        position_size = self._calculate_position_size(
            account_equity, current_price, confidence
        )
        logger.info(f"  [SIGNAL] Position Size: {position_size:.6f} units")
        logger.info(f"  [SIGNAL] Position Value: ${position_size * current_price:.2f}")

        # Calculate stop loss and take profit
        atr = self._calculate_atr(df)
        stop_loss = self._calculate_stop_loss(current_price, signal_type, atr)
        take_profit = self._calculate_take_profit(current_price, signal_type, atr)

        # Calculate SL/TP percentages
        sl_pct = abs(stop_loss - current_price) / current_price * 100
        tp_pct = abs(take_profit - current_price) / current_price * 100
        rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0

        logger.info(f"  [SIGNAL] Entry Price: ${current_price:.2f}")
        logger.info(f"  [SIGNAL] Stop Loss: ${stop_loss:.2f} ({sl_pct:.2f}%)")
        logger.info(f"  [SIGNAL] Take Profit: ${take_profit:.2f} ({tp_pct:.2f}%)")
        logger.info(f"  [SIGNAL] Risk:Reward Ratio: 1:{rr_ratio:.2f}")
        logger.info(f"  [SIGNAL] ATR(14): ${atr:.2f}")

        # Create signal
        signal = TradeSignal(
            signal=signal_type,
            confidence=confidence,
            symbol=symbol,
            timestamp=now,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            model_agreement=model_agreement,
            reason=self._get_signal_reason(df, signal_type)
        )

        # Update state
        self.state.last_signal_time = now

        logger.info(f"  [SIGNAL] ========================================")
        logger.info(f"  [TRADE] >>> SIGNAL GENERATED: {signal_type.name} @ ${current_price:.2f}")
        logger.info(f"  [TRADE] >>> Confidence: {confidence:.2%} | Agreement: {model_agreement}/5")
        logger.info(f"  [TRADE] >>> Reason: {signal.reason}")
        logger.info(f"  [SIGNAL] ========================================")

        return signal

    def _technical_analysis_signal(self, df: pd.DataFrame) -> Tuple[int, float, int]:
        """
        Generate signal using technical analysis when ML is not available.

        Returns:
            Tuple of (prediction, confidence, agreement)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        signals = []

        # RSI
        rsi = self._calculate_rsi(close, 14)
        if rsi < 30:
            signals.append(1)
        elif rsi > 70:
            signals.append(-1)
        else:
            signals.append(0)

        # MACD
        macd, signal_line = self._calculate_macd(close)
        if macd > signal_line:
            signals.append(1)
        else:
            signals.append(-1)

        # Moving average crossover
        sma_10 = np.mean(close[-10:])
        sma_20 = np.mean(close[-20:])
        if sma_10 > sma_20:
            signals.append(1)
        else:
            signals.append(-1)

        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger(close, 20)
        current_close = close[-1]
        if current_close < bb_lower:
            signals.append(1)
        elif current_close > bb_upper:
            signals.append(-1)
        else:
            signals.append(0)

        # Stochastic
        stoch_k = self._calculate_stochastic(high, low, close, 14)
        if stoch_k < 20:
            signals.append(1)
        elif stoch_k > 80:
            signals.append(-1)
        else:
            signals.append(0)

        # Aggregate signals
        avg_signal = np.mean(signals)
        agreement = sum(1 for s in signals if s == np.sign(avg_signal))

        if avg_signal > 0.3:
            prediction = 1
        elif avg_signal < -0.3:
            prediction = -1
        else:
            prediction = 0

        confidence = (abs(avg_signal) + 0.5) / 1.5  # Scale to 0.33 - 1.0

        return prediction, confidence, agreement

    def _apply_filters(self, df: pd.DataFrame, prediction: int) -> bool:
        """Apply trading filters to validate signal."""
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        logger.info(f"  [FILTER] Checking trading filters for prediction={prediction}...")

        # Volume filter
        if self.config['volume_confirmation']:
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]

            # Skip volume filter if no volume data available (crypto often has 0 volume from some APIs)
            if avg_volume == 0 or current_volume == 0:
                logger.info(f"  [FILTER] Volume: SKIPPED (no volume data available)")
            else:
                volume_ratio = current_volume / avg_volume
                logger.info(f"  [FILTER] Volume: {volume_ratio:.2f}x avg (threshold: {self.config['volume_filter_threshold']}x)")
                if current_volume < avg_volume * self.config['volume_filter_threshold']:
                    logger.info(f"  [FILTER] REJECTED - Volume too low ({volume_ratio:.2f}x < {self.config['volume_filter_threshold']}x)")
                    return False

        # Volatility filter
        if len(close) >= 21:
            # Calculate returns properly - use close[-21:] to get 21 values, then diff gives 20 returns
            close_window = close[-21:]
            returns = np.diff(close_window) / close_window[:-1]  # 20 returns from 21 prices
            volatility = np.std(returns)
            logger.info(f"  [FILTER] Volatility: {volatility:.4f} (max: {self.config['volatility_filter_max']})")
            if volatility > self.config['volatility_filter_max']:
                logger.info(f"  [FILTER] REJECTED - Volatility too high ({volatility:.4f} > {self.config['volatility_filter_max']})")
                return False

        # Trend confirmation
        if self.config['trend_confirmation']:
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
            current_price = close[-1]
            price_vs_sma = (current_price / sma_50 - 1) * 100
            logger.info(f"  [FILTER] Trend: Price ${current_price:.2f} vs SMA50 ${sma_50:.2f} ({price_vs_sma:+.2f}%)")

            if prediction == 1 and current_price < sma_50 * 0.98:
                logger.info(f"  [FILTER] REJECTED - BUY signal but price {price_vs_sma:.2f}% below SMA50 (need >-2%)")
                return False
            if prediction == -1 and current_price > sma_50 * 1.02:
                logger.info(f"  [FILTER] REJECTED - SELL signal but price {price_vs_sma:.2f}% above SMA50 (need <+2%)")
                return False

        # Momentum confirmation
        if self.config['momentum_confirmation']:
            roc_5 = (close[-1] - close[-6]) / close[-6] * 100 if len(close) >= 6 else 0
            logger.info(f"  [FILTER] Momentum (5-bar ROC): {roc_5:+.2f}%")

            if prediction == 1 and roc_5 < -2.0:
                logger.info(f"  [FILTER] REJECTED - BUY but momentum {roc_5:.2f}% (need >-2%)")
                return False
            if prediction == -1 and roc_5 > 2.0:
                logger.info(f"  [FILTER] REJECTED - SELL but momentum {roc_5:.2f}% (need <+2%)")
                return False

        logger.info(f"  [FILTER] All trading filters PASSED")
        return True

    def _calculate_position_size(self,
                                  equity: float,
                                  price: float,
                                  confidence: float) -> float:
        """Calculate position size based on risk parameters."""
        # Base position size
        base_size = equity * self.config['position_size_pct']

        # Adjust for confidence
        confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5x

        # Calculate quantity
        position_value = min(base_size * confidence_multiplier, self.config['max_position_value'])
        quantity = position_value / price

        return round(quantity, 8)  # Crypto precision

    def _calculate_stop_loss(self,
                              price: float,
                              signal: SignalType,
                              atr: float) -> float:
        """Calculate stop loss price."""
        # ATR-based stop loss
        atr_multiplier = 1.5
        atr_stop = atr * atr_multiplier

        # Percentage-based stop loss
        pct_stop = price * self.config['stop_loss_pct']

        # Use the larger of the two
        stop_distance = max(atr_stop, pct_stop)

        if signal == SignalType.BUY:
            return price - stop_distance
        else:
            return price + stop_distance

    def _calculate_take_profit(self,
                                price: float,
                                signal: SignalType,
                                atr: float) -> float:
        """Calculate take profit price."""
        # ATR-based take profit
        atr_multiplier = 2.5
        atr_tp = atr * atr_multiplier

        # Percentage-based take profit
        pct_tp = price * self.config['take_profit_pct']

        # Use the larger of the two
        tp_distance = max(atr_tp, pct_tp)

        if signal == SignalType.BUY:
            return price + tp_distance
        else:
            return price - tp_distance

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[-period:] - low[-period:],
            np.maximum(
                np.abs(high[-period:] - np.roll(close, 1)[-period:]),
                np.abs(low[-period:] - np.roll(close, 1)[-period:])
            )
        )

        return np.mean(tr)

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, close: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD."""
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_12 = ema(close, 12)
        ema_26 = ema(close, 26)
        macd_line = ema_12 - ema_26
        signal_line = ema(macd_line, 9)

        return macd_line[-1], signal_line[-1]

    def _calculate_bollinger(self, close: np.ndarray, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    def _calculate_stochastic(self,
                               high: np.ndarray,
                               low: np.ndarray,
                               close: np.ndarray,
                               period: int = 14) -> float:
        """Calculate Stochastic %K."""
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])

        if highest_high == lowest_low:
            return 50

        return 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)

    def _get_signal_reason(self, df: pd.DataFrame, signal: SignalType) -> str:
        """Generate human-readable reason for signal."""
        close = df['close'].values
        reasons = []

        rsi = self._calculate_rsi(close, 14)
        if signal == SignalType.BUY and rsi < 40:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif signal == SignalType.SELL and rsi > 60:
            reasons.append(f"RSI overbought ({rsi:.1f})")

        macd, signal_line = self._calculate_macd(close)
        if signal == SignalType.BUY and macd > signal_line:
            reasons.append("MACD bullish cross")
        elif signal == SignalType.SELL and macd < signal_line:
            reasons.append("MACD bearish cross")

        return "; ".join(reasons) if reasons else "ML ensemble signal"

    def update_position_state(self,
                               current_price: float,
                               position: SignalType,
                               entry_price: float,
                               entry_time: datetime,
                               position_size: float):
        """Update the strategy state with current position."""
        self.state.current_position = position
        self.state.entry_price = entry_price
        self.state.entry_time = entry_time
        self.state.position_size = position_size

        if position == SignalType.BUY:
            self.state.unrealized_pnl = (current_price - entry_price) * position_size
        elif position == SignalType.SELL:
            self.state.unrealized_pnl = (entry_price - current_price) * position_size

        self.state.highest_price_since_entry = max(
            self.state.highest_price_since_entry, current_price
        )
        self.state.lowest_price_since_entry = min(
            self.state.lowest_price_since_entry or current_price, current_price
        )

    def check_exit_conditions(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if current position should be exited.

        Returns:
            Tuple of (should_exit, reason)
        """
        if self.state.current_position == SignalType.HOLD:
            return False, ""

        entry_price = self.state.entry_price
        position = self.state.current_position

        # Calculate P&L percentage
        if position == SignalType.BUY:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Stop loss
        if pnl_pct <= -self.config['stop_loss_pct']:
            return True, "Stop loss triggered"

        # Take profit
        if pnl_pct >= self.config['take_profit_pct']:
            return True, "Take profit triggered"

        # Trailing stop
        if position == SignalType.BUY:
            peak_pnl = (self.state.highest_price_since_entry - entry_price) / entry_price
            if peak_pnl > 0.01:  # Only if we were in profit
                trail_price = self.state.highest_price_since_entry * (1 - self.config['trailing_stop_pct'])
                if current_price <= trail_price:
                    return True, "Trailing stop triggered"
        else:
            trough_pnl = (entry_price - self.state.lowest_price_since_entry) / entry_price
            if trough_pnl > 0.01:
                trail_price = self.state.lowest_price_since_entry * (1 + self.config['trailing_stop_pct'])
                if current_price >= trail_price:
                    return True, "Trailing stop triggered"

        # NOTE: No max holding time - DCA handles exits (profit target or SL)

        return False, ""

    def on_trade_closed(self, pnl: float):
        """Update state when a trade is closed."""
        self.state.trades_today += 1
        self.state.daily_pnl += pnl
        self.state.current_position = SignalType.HOLD
        self.state.entry_price = 0.0
        self.state.entry_time = None
        self.state.position_size = 0.0
        self.state.unrealized_pnl = 0.0
        self.state.highest_price_since_entry = 0.0
        self.state.lowest_price_since_entry = 0.0

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new day)."""
        self.state.trades_today = 0
        self.state.daily_pnl = 0.0
