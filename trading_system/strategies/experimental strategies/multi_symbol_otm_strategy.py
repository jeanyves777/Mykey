"""
Multi-Symbol OTM Day Trading Strategy

Uses 4-layer signal validation:
- Layer 1: Price Action (candlestick patterns)
- Layer 2: Technical Indicators (RSI, MACD, Bollinger)
- Layer 3: Momentum Weighting
- Layer 4: Pullback Detection (5-min HTF)

Trades OTM options on: SPY, QQQ, AMD, IWM, PLTR, BAC
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SignalResult:
    """Result of signal analysis."""
    direction: str  # 'BULLISH', 'BEARISH', or 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    layer_scores: Dict[str, float]
    details: Dict[str, any]


class MultiSymbolOTMStrategy:
    """
    Multi-Symbol OTM Day Trading Strategy with 4-layer signal validation.
    """

    # EMA period for trend detection
    EMA_PERIOD = 20

    # Signal thresholds
    MIN_BULLISH_SCORE = 0.6
    MIN_BEARISH_SCORE = 0.6

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return [prices[-1]] * len(prices) if prices else []

        ema = []
        multiplier = 2 / (period + 1)

        # Start with SMA for first EMA value
        sma = sum(prices[:period]) / period
        ema.append(sma)

        for i in range(period, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)

        # Pad beginning with first EMA value
        return [ema[0]] * (len(prices) - len(ema)) + ema

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        # EMA 12
        ema12 = MultiSymbolOTMStrategy.calculate_ema(prices, 12)
        # EMA 26
        ema26 = MultiSymbolOTMStrategy.calculate_ema(prices, 26)

        # MACD line
        macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]

        # Signal line (9-period EMA of MACD)
        signal_line = MultiSymbolOTMStrategy.calculate_ema(macd_line, 9)

        # Histogram
        histogram = macd_line[-1] - signal_line[-1] if signal_line else 0.0

        return macd_line[-1], signal_line[-1], histogram

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]

        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        std = np.std(recent_prices)

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return upper, middle, lower

    @staticmethod
    def analyze_layer1_price_action(bars: List[dict]) -> Dict:
        """
        Layer 1: Price Action Analysis (Candlestick Patterns)
        """
        if len(bars) < 5:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'patterns': []}

        patterns = []
        bullish_count = 0
        bearish_count = 0

        # Analyze last 5 candles
        for i in range(-5, 0):
            bar = bars[i]
            open_price = bar['open']
            close_price = bar['close']
            high_price = bar['high']
            low_price = bar['low']

            body = abs(close_price - open_price)
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            total_range = high_price - low_price

            if total_range == 0:
                continue

            # Bullish patterns
            if close_price > open_price:
                bullish_count += 1

                # Strong bullish candle
                if body > 0.7 * total_range:
                    patterns.append('STRONG_BULLISH')
                    bullish_count += 1

                # Hammer (bullish reversal)
                if lower_wick > 2 * body and upper_wick < 0.1 * total_range:
                    patterns.append('HAMMER')
                    bullish_count += 2

            # Bearish patterns
            elif close_price < open_price:
                bearish_count += 1

                # Strong bearish candle
                if body > 0.7 * total_range:
                    patterns.append('STRONG_BEARISH')
                    bearish_count += 1

                # Shooting star (bearish reversal)
                if upper_wick > 2 * body and lower_wick < 0.1 * total_range:
                    patterns.append('SHOOTING_STAR')
                    bearish_count += 2

            # Doji (indecision)
            if body < 0.1 * total_range:
                patterns.append('DOJI')

        # Calculate direction and score
        total = bullish_count + bearish_count
        if total == 0:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'patterns': patterns}

        if bullish_count > bearish_count:
            score = bullish_count / (total + 3)  # Normalize
            direction = 'BULLISH'
        elif bearish_count > bullish_count:
            score = bearish_count / (total + 3)
            direction = 'BEARISH'
        else:
            score = 0.0
            direction = 'NEUTRAL'

        return {'score': min(score, 1.0), 'direction': direction, 'patterns': patterns}

    @staticmethod
    def analyze_layer2_technical(bars: List[dict]) -> Dict:
        """
        Layer 2: Technical Indicator Analysis (RSI, MACD, Bollinger)
        """
        if len(bars) < 26:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'indicators': {}}

        closes = [b['close'] for b in bars]
        current_price = closes[-1]

        # Calculate indicators
        rsi = MultiSymbolOTMStrategy.calculate_rsi(closes)
        macd, signal, histogram = MultiSymbolOTMStrategy.calculate_macd(closes)
        bb_upper, bb_middle, bb_lower = MultiSymbolOTMStrategy.calculate_bollinger_bands(closes)
        ema20 = MultiSymbolOTMStrategy.calculate_ema(closes, 20)[-1]

        bullish_signals = 0
        bearish_signals = 0
        total_signals = 4

        # RSI analysis
        if rsi < 30:
            bullish_signals += 1  # Oversold
        elif rsi > 70:
            bearish_signals += 1  # Overbought
        elif rsi < 45:
            bullish_signals += 0.5
        elif rsi > 55:
            bearish_signals += 0.5

        # MACD analysis
        if histogram > 0 and macd > signal:
            bullish_signals += 1
        elif histogram < 0 and macd < signal:
            bearish_signals += 1

        # Bollinger Bands analysis
        if current_price < bb_lower:
            bullish_signals += 1  # Below lower band
        elif current_price > bb_upper:
            bearish_signals += 1  # Above upper band
        elif current_price < bb_middle:
            bullish_signals += 0.5
        elif current_price > bb_middle:
            bearish_signals += 0.5

        # EMA trend
        if current_price > ema20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # Determine direction
        if bullish_signals > bearish_signals:
            score = bullish_signals / total_signals
            direction = 'BULLISH'
        elif bearish_signals > bullish_signals:
            score = bearish_signals / total_signals
            direction = 'BEARISH'
        else:
            score = 0.0
            direction = 'NEUTRAL'

        return {
            'score': min(score, 1.0),
            'direction': direction,
            'indicators': {
                'rsi': rsi,
                'macd': macd,
                'signal': signal,
                'histogram': histogram,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'ema20': ema20,
                'price': current_price
            }
        }

    @staticmethod
    def analyze_layer3_momentum(bars: List[dict]) -> Dict:
        """
        Layer 3: Momentum Analysis (Volume-weighted)
        """
        if len(bars) < 10:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'momentum': {}}

        # Calculate price momentum
        closes = [b['close'] for b in bars]
        volumes = [b.get('volume', 1) for b in bars]

        # Short-term momentum (5 bars)
        short_momentum = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] != 0 else 0

        # Medium-term momentum (10 bars)
        mid_momentum = (closes[-1] - closes[-10]) / closes[-10] * 100 if closes[-10] != 0 else 0

        # Volume momentum (increasing or decreasing)
        avg_recent_vol = sum(volumes[-3:]) / 3
        avg_prev_vol = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else avg_recent_vol
        vol_ratio = avg_recent_vol / avg_prev_vol if avg_prev_vol > 0 else 1.0

        # Calculate momentum score
        bullish_score = 0
        bearish_score = 0

        # Price momentum scoring
        if short_momentum > 0.5:
            bullish_score += 1
        elif short_momentum < -0.5:
            bearish_score += 1

        if mid_momentum > 1.0:
            bullish_score += 1
        elif mid_momentum < -1.0:
            bearish_score += 1

        # Volume confirmation
        if vol_ratio > 1.2:
            # High volume confirms trend
            if short_momentum > 0:
                bullish_score += 1
            elif short_momentum < 0:
                bearish_score += 1

        # Determine direction
        if bullish_score > bearish_score:
            score = bullish_score / 3
            direction = 'BULLISH'
        elif bearish_score > bullish_score:
            score = bearish_score / 3
            direction = 'BEARISH'
        else:
            score = 0.0
            direction = 'NEUTRAL'

        return {
            'score': min(score, 1.0),
            'direction': direction,
            'momentum': {
                'short_momentum': short_momentum,
                'mid_momentum': mid_momentum,
                'volume_ratio': vol_ratio
            }
        }

    @staticmethod
    def check_pullback_entry_htf(bars_5min: List[dict], direction: str) -> Dict:
        """
        Layer 4: Pullback Detection using 5-minute bars (Higher TimeFrame = less noise)

        For BULLISH entry:
        - Price pulled back from recent high
        - Now showing recovery (green candle, higher lows)
        - RSI recovering from oversold or neutral zone

        For BEARISH entry:
        - Price pulled back from recent low (bounced)
        - Now showing weakness (red candle, lower highs)
        - RSI declining from overbought or neutral zone
        """
        if len(bars_5min) < 20:
            return {
                'valid': False,
                'reason': 'Insufficient 5-min data',
                'pullback_score': 0,
                'recovery_score': 0,
                'details': {}
            }

        closes = [b['close'] for b in bars_5min]
        highs = [b['high'] for b in bars_5min]
        lows = [b['low'] for b in bars_5min]
        opens = [b['open'] for b in bars_5min]

        current_close = closes[-1]
        current_open = opens[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        # Calculate EMA 20 on 5-min
        ema20_htf = MultiSymbolOTMStrategy.calculate_ema(closes, 20)[-1]

        # Calculate RSI on 5-min
        rsi_htf = MultiSymbolOTMStrategy.calculate_rsi(closes, 14)

        # Recent high/low (last 20 bars = 100 minutes)
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])

        pullback_score = 0
        recovery_score = 0
        details = {
            'ema20_htf': ema20_htf,
            'rsi_htf': rsi_htf,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'current_price': current_close
        }

        if direction == 'BULLISH':
            # === BULLISH PULLBACK DETECTION ===

            # 1. Price below recent high (pullback occurred)
            pullback_pct = (recent_high - current_close) / recent_high * 100
            if pullback_pct >= 0.3:  # At least 0.3% pullback
                pullback_score += 1
                details['pullback_pct'] = pullback_pct
            if pullback_pct >= 0.5:
                pullback_score += 1
            if pullback_pct >= 1.0:
                pullback_score += 1

            # 2. Price near EMA support
            ema_distance = (current_close - ema20_htf) / ema20_htf * 100
            if -0.5 <= ema_distance <= 1.0:  # Near or just above EMA
                pullback_score += 1
                details['near_ema_support'] = True

            # 3. Higher lows forming (recovery)
            if len(lows) >= 3:
                if lows[-1] > lows[-2] and lows[-2] > lows[-3]:
                    recovery_score += 2
                    details['higher_lows'] = True
                elif lows[-1] > lows[-2]:
                    recovery_score += 1

            # 4. Current candle is green (bullish)
            if current_close > current_open:
                recovery_score += 2
                details['green_candle'] = True

            # 5. RSI recovering (not overbought)
            if 30 <= rsi_htf <= 60:
                recovery_score += 1
                details['rsi_recovering'] = True
            elif rsi_htf < 30:
                recovery_score += 2  # Oversold = better entry
                details['rsi_oversold'] = True

            # 6. Price holding above recent low
            if current_close > recent_low * 1.005:  # At least 0.5% above low
                recovery_score += 1
                details['above_recent_low'] = True

            # Entry is valid if good pullback AND showing recovery
            valid = pullback_score >= 3 and recovery_score >= 4 and current_close > current_open

        else:  # BEARISH
            # === BEARISH PULLBACK DETECTION ===

            # 1. Price above recent low (bounce occurred)
            bounce_pct = (current_close - recent_low) / recent_low * 100
            if bounce_pct >= 0.3:
                pullback_score += 1
                details['bounce_pct'] = bounce_pct
            if bounce_pct >= 0.5:
                pullback_score += 1
            if bounce_pct >= 1.0:
                pullback_score += 1

            # 2. Price near EMA resistance
            ema_distance = (current_close - ema20_htf) / ema20_htf * 100
            if -1.0 <= ema_distance <= 0.5:  # Near or just below EMA
                pullback_score += 1
                details['near_ema_resistance'] = True

            # 3. Lower highs forming (weakness)
            if len(highs) >= 3:
                if highs[-1] < highs[-2] and highs[-2] < highs[-3]:
                    recovery_score += 2
                    details['lower_highs'] = True
                elif highs[-1] < highs[-2]:
                    recovery_score += 1

            # 4. Current candle is red (bearish)
            if current_close < current_open:
                recovery_score += 2
                details['red_candle'] = True

            # 5. RSI declining (not oversold)
            if 40 <= rsi_htf <= 70:
                recovery_score += 1
                details['rsi_declining'] = True
            elif rsi_htf > 70:
                recovery_score += 2  # Overbought = better entry
                details['rsi_overbought'] = True

            # 6. Price holding below recent high
            if current_close < recent_high * 0.995:
                recovery_score += 1
                details['below_recent_high'] = True

            # Entry is valid if good bounce AND showing weakness
            valid = pullback_score >= 3 and recovery_score >= 4 and current_close < current_open

        reason = "VALID" if valid else f"Pullback={pullback_score}/4, Recovery={recovery_score}/6"

        return {
            'valid': valid,
            'reason': reason,
            'pullback_score': pullback_score,
            'recovery_score': recovery_score,
            'details': details
        }

    @classmethod
    def generate_signal(cls, bars_1min: List[dict], bars_5min: List[dict]) -> SignalResult:
        """
        Generate trading signal using 4-layer validation.

        Args:
            bars_1min: 1-minute bars for layers 1-3
            bars_5min: 5-minute bars for layer 4 (pullback detection)

        Returns:
            SignalResult with direction, confidence, and details
        """
        # Layer 1: Price Action
        layer1 = cls.analyze_layer1_price_action(bars_1min)

        # Layer 2: Technical Indicators
        layer2 = cls.analyze_layer2_technical(bars_1min)

        # Layer 3: Momentum
        layer3 = cls.analyze_layer3_momentum(bars_1min)

        # Determine preliminary direction (weighted voting)
        directions = [layer1['direction'], layer2['direction'], layer3['direction']]
        weights = [0.25, 0.35, 0.40]  # Layer 2 and 3 weighted more

        bullish_weight = sum(w for d, w in zip(directions, weights) if d == 'BULLISH')
        bearish_weight = sum(w for d, w in zip(directions, weights) if d == 'BEARISH')

        if bullish_weight > bearish_weight and bullish_weight >= 0.5:
            prelim_direction = 'BULLISH'
            prelim_confidence = bullish_weight
        elif bearish_weight > bullish_weight and bearish_weight >= 0.5:
            prelim_direction = 'BEARISH'
            prelim_confidence = bearish_weight
        else:
            return SignalResult(
                direction='NEUTRAL',
                confidence=0.0,
                layer_scores={'layer1': layer1['score'], 'layer2': layer2['score'], 'layer3': layer3['score']},
                details={'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'reason': 'No clear direction from L1-L3'}
            )

        # Layer 4: Pullback Detection (final gate)
        layer4 = cls.check_pullback_entry_htf(bars_5min, prelim_direction)

        if not layer4['valid']:
            return SignalResult(
                direction='NEUTRAL',
                confidence=prelim_confidence * 0.5,  # Reduced confidence
                layer_scores={
                    'layer1': layer1['score'],
                    'layer2': layer2['score'],
                    'layer3': layer3['score'],
                    'layer4_pullback': layer4['pullback_score'] / 4,
                    'layer4_recovery': layer4['recovery_score'] / 6
                },
                details={
                    'layer1': layer1,
                    'layer2': layer2,
                    'layer3': layer3,
                    'layer4': layer4,
                    'reason': f'V4 Pullback not valid: {layer4["reason"]}'
                }
            )

        # All layers agree - high confidence signal
        final_confidence = prelim_confidence * (0.5 + (layer4['pullback_score'] + layer4['recovery_score']) / 20)

        return SignalResult(
            direction=prelim_direction,
            confidence=min(final_confidence, 1.0),
            layer_scores={
                'layer1': layer1['score'],
                'layer2': layer2['score'],
                'layer3': layer3['score'],
                'layer4_pullback': layer4['pullback_score'] / 4,
                'layer4_recovery': layer4['recovery_score'] / 6
            },
            details={
                'layer1': layer1,
                'layer2': layer2,
                'layer3': layer3,
                'layer4': layer4,
                'reason': 'All 4 layers validated'
            }
        )

    @staticmethod
    def select_strike(current_price: float, option_chain: List[dict], direction: str,
                      otm_min: int = 1, otm_max: int = 2) -> Optional[dict]:
        """
        Select the best OTM strike from the option chain.

        Args:
            current_price: Current underlying price
            option_chain: List of available options
            direction: 'BULLISH' for calls, 'BEARISH' for puts
            otm_min: Minimum OTM strikes
            otm_max: Maximum OTM strikes

        Returns:
            Selected option contract or None
        """
        if not option_chain:
            return None

        option_type = 'call' if direction == 'BULLISH' else 'put'

        # Filter by option type
        filtered = [o for o in option_chain if o.get('option_type', '').lower() == option_type]

        if not filtered:
            return None

        # Sort by strike
        if direction == 'BULLISH':
            # For calls, we want strikes above current price (OTM)
            otm_options = sorted([o for o in filtered if o['strike'] > current_price], key=lambda x: x['strike'])
        else:
            # For puts, we want strikes below current price (OTM)
            otm_options = sorted([o for o in filtered if o['strike'] < current_price], key=lambda x: x['strike'], reverse=True)

        # Select from the OTM range
        if len(otm_options) < otm_min:
            return None

        # Try to get option in the range [otm_min, otm_max]
        for i in range(otm_min - 1, min(otm_max, len(otm_options))):
            option = otm_options[i]
            # Check if option has reasonable bid/ask
            if option.get('ask', 0) > 0 and option.get('bid', 0) >= 0:
                return option

        return otm_options[otm_min - 1] if len(otm_options) >= otm_min else None
