"""
ULTIMATE TRADING SYSTEM - Target: 90% Win Rate
================================================

Philosophy:
- Trade LESS, win MORE
- Only trade when ALL conditions align
- Strict risk management
- $200 capital test

Key Strategies for High Win Rate:
1. EXTREME SELECTIVITY - Only trade top 1-2% setups
2. REGIME FILTERING - Only trade in favorable market conditions
3. MULTI-CONFIRMATION - Require 4+ models to agree
4. CONFIDENCE THRESHOLD - Only trade >80% confidence
5. RISK/REWARD FILTER - Only take 2:1+ R:R trades
6. VOLATILITY FILTER - Avoid low/extreme volatility
7. MOMENTUM CONFIRMATION - Trade with trend only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UltimateRiskConfig:
    """Ultra-conservative risk management for 90% win rate target."""

    # Capital
    initial_capital: float = 200.0  # $200 test

    # Position sizing - ULTRA CONSERVATIVE
    max_position_pct: float = 0.05  # Max 5% per trade ($10 on $200)
    min_position_size: float = 5.0  # Minimum $5 position

    # Win rate optimization
    min_confidence: float = 0.75    # Only trade >75% confidence
    min_model_agreement: int = 4    # 4 out of 5 models must agree

    # Regime filters
    min_volatility: float = 0.001   # Avoid dead markets
    max_volatility: float = 0.05    # Avoid crazy volatility
    min_volume_ratio: float = 1.2   # Above average volume only
    require_trend_alignment: bool = True

    # Risk/Reward
    min_risk_reward: float = 2.0    # Only 2:1+ R:R trades
    stop_loss_pct: float = 0.01     # 1% stop loss (tight)
    take_profit_pct: float = 0.02   # 2% take profit

    # Trade management
    max_trades_per_day: int = 3     # Quality over quantity
    max_concurrent_positions: int = 1
    cooldown_minutes: int = 30      # Wait between trades

    # Daily limits
    max_daily_loss_pct: float = 0.05  # Stop at 5% daily loss
    daily_profit_target_pct: float = 0.03  # Target 3% daily profit

    # Trading costs (Binance)
    commission_rate: float = 0.001  # 0.1% maker/taker
    slippage_pct: float = 0.0005    # 0.05% slippage estimate


class RegimeDetector:
    """Detect market regime - only trade favorable conditions."""

    def __init__(self):
        self.current_regime = "UNKNOWN"

    def analyze(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Analyze market regime.

        Returns dict with:
        - regime: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
        - tradeable: True/False
        - confidence: 0-1
        """
        if len(df) < lookback + 10:
            return {'regime': 'UNKNOWN', 'tradeable': False, 'confidence': 0}

        close = df['close'].values[-lookback:]
        high = df['high'].values[-lookback:]
        low = df['low'].values[-lookback:]
        volume = df['volume'].values[-lookback:]

        # Calculate indicators
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns)
        trend = (close[-1] - close[0]) / close[0]

        # ADX-like trend strength
        tr = np.maximum(high[1:] - low[1:],
                       np.maximum(np.abs(high[1:] - close[:-1]),
                                 np.abs(low[1:] - close[:-1])))
        atr = np.mean(tr)

        # Volume analysis
        avg_volume = np.mean(volume)
        recent_volume = np.mean(volume[-5:])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # Determine regime (very permissive - let ML be the main filter)
        if volatility > 0.10:  # 10% volatility = extreme (rare)
            regime = "VOLATILE"
            tradeable = False
            confidence = 0.3
        elif abs(trend) > 0.005:  # Any detectable trend
            if trend > 0:
                regime = "TRENDING_UP"
            else:
                regime = "TRENDING_DOWN"
            tradeable = True
            confidence = min(abs(trend) / 0.02, 1.0) * 0.8 + 0.2
        elif volatility < 0.0001:  # Only truly dead markets (0.01% vol)
            regime = "DEAD"
            tradeable = False
            confidence = 0.2
        else:
            regime = "RANGING"
            tradeable = True  # Allow all ranging markets
            confidence = 0.5

        # Volume confirmation (disabled for more trades)
        # if volume_ratio < 0.8:
        #     tradeable = False
        #     confidence *= 0.7

        self.current_regime = regime

        return {
            'regime': regime,
            'tradeable': tradeable,
            'confidence': confidence,
            'volatility': volatility,
            'trend': trend,
            'volume_ratio': volume_ratio,
            'atr': atr
        }


class SignalFilter:
    """Filter signals for maximum win rate."""

    def __init__(self, config: UltimateRiskConfig):
        self.config = config
        self.regime_detector = RegimeDetector()

    def should_trade(self,
                     prediction: int,
                     confidence: float,
                     model_votes: Dict[str, int],
                     regime_info: Dict,
                     df: pd.DataFrame,
                     current_idx: int) -> Tuple[bool, str]:
        """
        Apply all filters to determine if we should trade.

        Returns: (should_trade, reason)
        """
        reasons = []

        # 1. Confidence filter
        if confidence < self.config.min_confidence:
            return False, f"Confidence {confidence:.2f} < {self.config.min_confidence}"

        # 2. Model agreement filter
        votes = list(model_votes.values())
        agreement = sum(1 for v in votes if v == prediction)
        if agreement < self.config.min_model_agreement:
            return False, f"Only {agreement}/{len(votes)} models agree"

        # 3. Regime filter
        if not regime_info.get('tradeable', False):
            return False, f"Regime {regime_info.get('regime')} not tradeable"

        # 4. Volatility filter
        vol = regime_info.get('volatility', 0)
        if vol < self.config.min_volatility:
            return False, f"Volatility {vol:.4f} too low"
        if vol > self.config.max_volatility:
            return False, f"Volatility {vol:.4f} too high"

        # 5. Volume filter
        vol_ratio = regime_info.get('volume_ratio', 0)
        if vol_ratio < self.config.min_volume_ratio:
            return False, f"Volume ratio {vol_ratio:.2f} too low"

        # 6. Trend alignment (if enabled)
        if self.config.require_trend_alignment:
            trend = regime_info.get('trend', 0)
            if prediction == 1 and trend < -0.005:  # BUY against downtrend
                return False, "BUY signal against downtrend"
            if prediction == 0 and trend > 0.005:   # SELL against uptrend
                return False, "SELL signal against uptrend"

        # 7. Risk/Reward check (can be disabled via min_risk_reward=0)
        if self.config.min_risk_reward > 0 and current_idx > 0 and current_idx < len(df) - 1:
            current_price = df['close'].iloc[current_idx]
            atr = regime_info.get('atr', current_price * 0.01)

            potential_profit = atr * 2  # Expected move
            potential_loss = current_price * self.config.stop_loss_pct

            rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
            if rr_ratio < self.config.min_risk_reward:
                return False, f"R:R {rr_ratio:.1f} < {self.config.min_risk_reward}"

        # All filters passed
        return True, "ALL_FILTERS_PASSED"


class UltimateTradingEngine:
    """
    Ultimate Trading Engine targeting 90% win rate.

    Key principle: Trade less, win more.
    """

    def __init__(self, config: UltimateRiskConfig = None):
        self.config = config or UltimateRiskConfig()
        self.signal_filter = SignalFilter(self.config)
        self.regime_detector = RegimeDetector()

        # State
        self.capital = self.config.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.current_date = None

        # Stats
        self.total_signals = 0
        self.filtered_signals = 0
        self.trades_taken = 0
        self.wins = 0
        self.losses = 0

    def reset(self):
        """Reset engine state."""
        self.capital = self.config.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self.current_date = None
        self.total_signals = 0
        self.filtered_signals = 0
        self.trades_taken = 0
        self.wins = 0
        self.losses = 0

    def run_backtest(self,
                     df: pd.DataFrame,
                     predictions: np.ndarray,
                     confidences: np.ndarray,
                     model_votes: List[Dict[str, int]] = None,
                     verbose: bool = True) -> Dict:
        """
        Run backtest with all filters and risk management.

        Args:
            df: OHLCV data
            predictions: Model predictions (0=SELL, 1=BUY)
            confidences: Confidence scores (0-1)
            model_votes: Per-sample votes from each model
            verbose: Print progress
        """
        self.reset()

        n_bars = min(len(df), len(predictions))

        if verbose:
            print("=" * 70)
            print("ULTIMATE TRADING SYSTEM - $200 CAPITAL TEST")
            print("=" * 70)
            print(f"Initial Capital: ${self.config.initial_capital:.2f}")
            print(f"Total Bars: {n_bars}")
            print(f"Min Confidence: {self.config.min_confidence}")
            print(f"Min Model Agreement: {self.config.min_model_agreement}/5")
            print(f"Max Trades/Day: {self.config.max_trades_per_day}")
            print("=" * 70)

        # Get price data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Get datetime
        if 'datetime' in df.columns:
            datetimes = pd.to_datetime(df['datetime'])
        else:
            datetimes = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')

        filter_reasons = {}

        for i in range(20, n_bars):  # Start after warmup
            current_time = datetimes[i]
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Check for new day
            current_day = current_time.date() if hasattr(current_time, 'date') else None
            if current_day != self.current_date:
                self.current_date = current_day
                self.daily_trades = 0
                self.daily_pnl = 0.0

            # Update position if we have one
            if self.position is not None:
                self._update_position(current_price, current_high, current_low, current_time)

                # Check exit conditions
                exit_signal = self._check_exit(current_price, current_high, current_low)
                if exit_signal:
                    self._close_position(current_price, current_time, exit_signal)

            # Check for new entry
            if self.position is None and i < len(predictions):
                self.total_signals += 1

                prediction = predictions[i]
                confidence = confidences[i]

                # Get model votes (if available)
                votes = model_votes[i] if model_votes else {'model': prediction}

                # Analyze regime
                regime_info = self.regime_detector.analyze(df.iloc[:i+1])

                # Apply all filters
                should_trade, reason = self.signal_filter.should_trade(
                    prediction=prediction,
                    confidence=confidence,
                    model_votes=votes,
                    regime_info=regime_info,
                    df=df,
                    current_idx=i
                )

                if not should_trade:
                    self.filtered_signals += 1
                    filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                    continue

                # Check daily limits
                if self.daily_trades >= self.config.max_trades_per_day:
                    continue

                if self.daily_pnl / self.config.initial_capital < -self.config.max_daily_loss_pct:
                    continue

                # Check cooldown
                if self.last_trade_time is not None:
                    minutes_since = (current_time - self.last_trade_time).total_seconds() / 60
                    if minutes_since < self.config.cooldown_minutes:
                        continue

                # OPEN POSITION
                self._open_position(prediction, current_price, current_time, confidence)

            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)

        # Close any remaining position
        if self.position is not None:
            self._close_position(closes[-1], datetimes.iloc[-1], "END_OF_DATA")

        # Calculate results
        results = self._calculate_results()

        if verbose:
            self._print_results(results, filter_reasons)

        return results

    def _open_position(self, prediction: int, price: float, time, confidence: float):
        """Open a new position."""
        # Calculate position size
        position_value = self.capital * self.config.max_position_pct
        position_value = max(position_value, self.config.min_position_size)
        position_value = min(position_value, self.capital * 0.5)  # Never more than 50%

        # Apply slippage
        if prediction == 1:  # BUY
            entry_price = price * (1 + self.config.slippage_pct)
            side = "LONG"
        else:  # SELL
            entry_price = price * (1 - self.config.slippage_pct)
            side = "SHORT"

        # Calculate commission
        commission = position_value * self.config.commission_rate

        # Calculate quantity
        quantity = (position_value - commission) / entry_price

        # Set stop loss and take profit
        if side == "LONG":
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
            take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)
            take_profit = entry_price * (1 - self.config.take_profit_pct)

        self.position = {
            'side': side,
            'entry_price': entry_price,
            'entry_time': time,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'commission': commission,
        }

        self.daily_trades += 1
        self.last_trade_time = time
        self.trades_taken += 1

    def _update_position(self, price: float, high: float, low: float, time):
        """Update position tracking."""
        if self.position is None:
            return

        self.position['highest_price'] = max(self.position['highest_price'], high)
        self.position['lowest_price'] = min(self.position['lowest_price'], low)

    def _check_exit(self, price: float, high: float, low: float) -> Optional[str]:
        """Check if we should exit position."""
        if self.position is None:
            return None

        side = self.position['side']
        stop_loss = self.position['stop_loss']
        take_profit = self.position['take_profit']

        if side == "LONG":
            if low <= stop_loss:
                return "STOP_LOSS"
            if high >= take_profit:
                return "TAKE_PROFIT"
        else:  # SHORT
            if high >= stop_loss:
                return "STOP_LOSS"
            if low <= take_profit:
                return "TAKE_PROFIT"

        return None

    def _close_position(self, price: float, time, reason: str):
        """Close current position."""
        if self.position is None:
            return

        side = self.position['side']
        entry_price = self.position['entry_price']
        quantity = self.position['quantity']

        # Determine exit price based on reason
        if reason == "STOP_LOSS":
            exit_price = self.position['stop_loss']
        elif reason == "TAKE_PROFIT":
            exit_price = self.position['take_profit']
        else:
            exit_price = price

        # Apply slippage on exit
        if side == "LONG":
            exit_price *= (1 - self.config.slippage_pct)
        else:
            exit_price *= (1 + self.config.slippage_pct)

        # Calculate P&L
        if side == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        # Subtract exit commission
        exit_commission = quantity * exit_price * self.config.commission_rate
        pnl -= exit_commission

        # Total commission
        total_commission = self.position['commission'] + exit_commission

        # Update capital
        self.capital += pnl
        self.daily_pnl += pnl

        # Track win/loss
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Record trade
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': time,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl / (entry_price * quantity) * 100,
            'commission': total_commission,
            'exit_reason': reason,
            'confidence': self.position['confidence'],
        })

        self.position = None

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity."""
        equity = self.capital

        if self.position is not None:
            entry_price = self.position['entry_price']
            quantity = self.position['quantity']
            side = self.position['side']

            if side == "LONG":
                unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = (entry_price - current_price) * quantity

            equity += unrealized

        return equity

    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if len(self.trades) == 0:
            return {
                'initial_capital': self.config.initial_capital,
                'final_capital': self.capital,
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'total_signals': self.total_signals,
                'filtered_signals': self.filtered_signals,
                'filter_rate': self.filtered_signals / self.total_signals if self.total_signals > 0 else 0,
            }

        # Basic stats
        total_trades = len(self.trades)
        win_rate = self.wins / total_trades if total_trades > 0 else 0

        # Returns
        final_capital = self.capital
        total_return = final_capital - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital * 100

        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown_pct = np.max(drawdown) * 100

        # Sharpe ratio (simplified)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 12)  # Annualized for 5-min
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if self.wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if self.losses > 0 else 0

        return {
            'initial_capital': self.config.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown_pct,
            'total_signals': self.total_signals,
            'filtered_signals': self.filtered_signals,
            'filter_rate': self.filtered_signals / self.total_signals if self.total_signals > 0 else 0,
        }

    def _print_results(self, results: Dict, filter_reasons: Dict):
        """Print detailed results."""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        print(f"\n{'CAPITAL':-^50}")
        print(f"  Initial:     ${results['initial_capital']:.2f}")
        print(f"  Final:       ${results['final_capital']:.2f}")
        print(f"  Return:      ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")

        print(f"\n{'TRADES':-^50}")
        print(f"  Total Signals:    {results['total_signals']}")
        print(f"  Filtered Out:     {results['filtered_signals']} ({results['filter_rate']*100:.1f}%)")
        print(f"  Trades Taken:     {results['total_trades']}")

        print(f"\n{'PERFORMANCE':-^50}")
        print(f"  Win Rate:         {results['win_rate']*100:.1f}%  ({results['wins']}W / {results['losses']}L)")
        print(f"  Avg Win:          ${results['avg_win']:.2f}")
        print(f"  Avg Loss:         ${results['avg_loss']:.2f}")
        print(f"  Profit Factor:    {results['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown_pct']:.2f}%")

        if filter_reasons:
            print(f"\n{'FILTER REASONS':-^50}")
            sorted_reasons = sorted(filter_reasons.items(), key=lambda x: -x[1])[:10]
            for reason, count in sorted_reasons:
                print(f"  {reason}: {count}")

        # Individual trades
        if len(self.trades) > 0 and len(self.trades) <= 20:
            print(f"\n{'TRADE LOG':-^50}")
            for i, t in enumerate(self.trades):
                pnl_str = f"+${t['pnl']:.2f}" if t['pnl'] > 0 else f"-${abs(t['pnl']):.2f}"
                print(f"  {i+1}. {t['side']:5} @ ${t['entry_price']:.4f} -> ${t['exit_price']:.4f} | {pnl_str} ({t['exit_reason']})")

        print("\n" + "=" * 70)

        # FINAL VERDICT
        if results['win_rate'] >= 0.9:
            print("*** TARGET ACHIEVED: 90%+ WIN RATE ***")
        elif results['win_rate'] >= 0.8:
            print("*** EXCELLENT: 80%+ WIN RATE ***")
        elif results['win_rate'] >= 0.7:
            print("*** GOOD: 70%+ WIN RATE ***")
        elif results['win_rate'] >= 0.6:
            print("*** ACCEPTABLE: 60%+ WIN RATE ***")
        else:
            print("*** NEEDS IMPROVEMENT ***")


def run_ultimate_test():
    """Run the ultimate $200 capital test."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from features import FeatureEngineer
    from ensemble import EnsembleVotingSystem
    from data.data_aggregator import DataAggregator

    print("\n" + "=" * 70)
    print("ULTIMATE $200 TRADING SYSTEM TEST")
    print("=" * 70)

    # Load saved models
    model_path = Path(__file__).parent / "saved_models_improved"
    if not model_path.exists():
        print("ERROR: No trained models found. Run train_improved.py first.")
        return

    # Load ensemble
    ensemble = EnsembleVotingSystem()
    ensemble.load(str(model_path / "ensemble"))
    print("Loaded trained ensemble model")

    # Load feature config
    fe = FeatureEngineer()
    fe.load_feature_config(str(model_path / "feature_config.json"))
    print(f"Loaded feature config ({len(fe.feature_columns)} features)")

    # Load 5-minute data
    data_path = Path(__file__).parent / "Crypto_Data_5m"
    symbols = ['BTCUSD', 'ETHUSD']  # Test on BTC and ETH only

    all_data = []
    for symbol in symbols:
        filepath = data_path / f"{symbol}_5m.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['symbol'] = symbol
            # Use last 10000 bars for testing (unseen data)
            df = df.tail(10000).reset_index(drop=True)
            all_data.append(df)
            print(f"Loaded {symbol}: {len(df)} bars")

    if not all_data:
        print("ERROR: No data found")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal test data: {len(combined_df)} bars")

    # Engineer features
    print("\nEngineering features...")
    fe_new = FeatureEngineer()
    featured_df = fe_new.compute_all_features(combined_df)

    # Prepare features for prediction
    X = featured_df[fe.feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get predictions
    print("Getting predictions from ensemble...")
    predictions = ensemble.predict(X)
    confidences = ensemble.predict_proba(X).max(axis=1)

    print(f"Predictions: {len(predictions)}")
    print(f"Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")

    # Get individual model votes
    model_votes = []
    for i in range(len(X)):
        votes = {}
        for name, model in ensemble.models.items():
            try:
                pred = model.predict(X.iloc[[i]])
                votes[name] = int(pred[0])
            except:
                votes[name] = predictions[i]
        model_votes.append(votes)

    # Run ultimate test
    config = UltimateRiskConfig(
        initial_capital=200.0,
        min_confidence=0.70,        # Start with 70%, can increase
        min_model_agreement=4,      # 4/5 models
        max_trades_per_day=5,
        min_risk_reward=1.5,
    )

    engine = UltimateTradingEngine(config=config)

    results = engine.run_backtest(
        df=featured_df,
        predictions=predictions,
        confidences=confidences,
        model_votes=model_votes,
        verbose=True
    )

    # Try different confidence thresholds
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS - CONFIDENCE THRESHOLDS")
    print("=" * 70)

    for min_conf in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        config.min_confidence = min_conf
        engine = UltimateTradingEngine(config=config)
        results = engine.run_backtest(
            df=featured_df,
            predictions=predictions,
            confidences=confidences,
            model_votes=model_votes,
            verbose=False
        )
        wr = results['win_rate'] * 100
        trades = results['total_trades']
        ret = results['total_return_pct']
        print(f"Conf >= {min_conf:.0%}: {trades:3} trades, {wr:5.1f}% win rate, {ret:+6.2f}% return")

    return results


if __name__ == "__main__":
    run_ultimate_test()
