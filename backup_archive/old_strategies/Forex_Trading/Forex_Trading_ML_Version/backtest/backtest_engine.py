"""
Backtest Engine for Forex ML System
===================================

Realistic backtesting with slippage and commission.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class BacktestTrade:
    """Represents a backtest trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    pnl_pips: float
    exit_reason: str
    confidence: float
    agreement: int


@dataclass
class BacktestResults:
    """Backtest results summary."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_time: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """Backtesting engine for ML strategy."""

    def __init__(self, config, ensemble, feature_engineer):
        """
        Initialize backtest engine.

        Args:
            config: TradingConfig object
            ensemble: Trained EnsembleVotingSystem
            feature_engineer: Configured FeatureEngineer
        """
        self.config = config
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer

        # Backtest settings
        if hasattr(config, 'backtest'):
            self.initial_capital = config.backtest.initial_capital
            self.commission_rate = config.backtest.commission_rate
            self.slippage_pips = config.backtest.slippage_pips
        else:
            self.initial_capital = 100000
            self.commission_rate = 0.00002
            self.slippage_pips = 0.5

        # ML settings
        self.confidence_threshold = config.ml.confidence_threshold
        self.min_agreement = config.ml.min_model_agreement

    def run_backtest(self, data: Dict[str, pd.DataFrame],
                     start_date: str = None, end_date: str = None) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResults object
        """
        results = BacktestResults()
        equity = self.initial_capital
        peak_equity = equity
        results.equity_curve = [equity]

        # Track open positions
        positions: Dict[str, dict] = {}

        # Process each symbol
        for symbol, df in data.items():
            print(f"\nBacktesting {symbol}...")

            # Filter by date if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            if len(df) < 200:
                print(f"  Insufficient data: {len(df)} bars")
                continue

            # Engineer features
            featured_df = self.feature_engineer.engineer_features(df.copy(), fit=False)

            # Iterate through bars
            for i in range(200, len(featured_df)):
                current_bar = featured_df.iloc[i]
                current_time = featured_df.index[i]
                current_price = current_bar['close']

                # Check existing position
                if symbol in positions:
                    pos = positions[symbol]

                    # Check SL/TP
                    exit_reason = None
                    if pos['direction'] == 'BUY':
                        if current_bar['low'] <= pos['stop_loss']:
                            exit_price = pos['stop_loss']
                            exit_reason = 'SL'
                        elif current_bar['high'] >= pos['take_profit']:
                            exit_price = pos['take_profit']
                            exit_reason = 'TP'
                    else:  # SELL
                        if current_bar['high'] >= pos['stop_loss']:
                            exit_price = pos['stop_loss']
                            exit_reason = 'SL'
                        elif current_bar['low'] <= pos['take_profit']:
                            exit_price = pos['take_profit']
                            exit_reason = 'TP'

                    if exit_reason:
                        # Close position
                        pip_value = self._get_pip_value(symbol)
                        if pos['direction'] == 'BUY':
                            pnl_pips = (exit_price - pos['entry_price']) / pip_value
                        else:
                            pnl_pips = (pos['entry_price'] - exit_price) / pip_value

                        # Apply slippage
                        pnl_pips -= self.slippage_pips

                        pnl = pnl_pips * pos['units'] * 10 * pip_value
                        pnl -= pos['commission']

                        equity += pnl
                        results.equity_curve.append(equity)

                        # Track trade
                        trade = BacktestTrade(
                            entry_time=pos['entry_time'],
                            exit_time=current_time,
                            symbol=symbol,
                            direction=pos['direction'],
                            entry_price=pos['entry_price'],
                            exit_price=exit_price,
                            units=pos['units'],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            exit_reason=exit_reason,
                            confidence=pos['confidence'],
                            agreement=pos['agreement']
                        )
                        results.trades.append(trade)

                        if pnl > 0:
                            results.winning_trades += 1
                        else:
                            results.losing_trades += 1

                        results.total_pnl += pnl
                        results.total_pips += pnl_pips
                        results.total_trades += 1

                        # Track drawdown
                        peak_equity = max(peak_equity, equity)
                        drawdown = peak_equity - equity
                        if drawdown > results.max_drawdown:
                            results.max_drawdown = drawdown
                            results.max_drawdown_pct = drawdown / peak_equity

                        del positions[symbol]

                else:
                    # Check for new signal
                    X = self.feature_engineer.get_feature_matrix(featured_df.iloc[:i+1])
                    if len(X) == 0:
                        continue

                    prediction, confidence, agreement, details = self.ensemble.predict_single(X[-1])

                    # Check thresholds
                    if confidence < self.confidence_threshold or agreement < self.min_agreement:
                        continue

                    if prediction == 2:  # BUY
                        direction = 'BUY'
                    elif prediction == 0:  # SELL
                        direction = 'SELL'
                    else:
                        continue

                    # Calculate entry with slippage
                    entry_price = current_price
                    if direction == 'BUY':
                        entry_price += self.slippage_pips * self._get_pip_value(symbol)
                    else:
                        entry_price -= self.slippage_pips * self._get_pip_value(symbol)

                    # Calculate SL/TP
                    settings = self.config.get_pair_settings(symbol)
                    pip_value = self._get_pip_value(symbol)
                    tp_pips = settings.get('tp_pips', 8)
                    sl_pips = settings.get('sl_pips', 20)

                    if direction == 'BUY':
                        stop_loss = entry_price - (sl_pips * pip_value)
                        take_profit = entry_price + (tp_pips * pip_value)
                    else:
                        stop_loss = entry_price + (sl_pips * pip_value)
                        take_profit = entry_price - (tp_pips * pip_value)

                    # Position sizing
                    risk_amount = equity * 0.02
                    units = risk_amount / (sl_pips * pip_value * 10)

                    # Commission
                    commission = units * entry_price * self.commission_rate * 2

                    positions[symbol] = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'entry_time': current_time,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'units': units,
                        'commission': commission,
                        'confidence': confidence,
                        'agreement': agreement
                    }

        # Calculate final metrics
        if results.total_trades > 0:
            results.win_rate = results.winning_trades / results.total_trades

            wins = [t.pnl for t in results.trades if t.pnl > 0]
            losses = [t.pnl for t in results.trades if t.pnl <= 0]

            results.avg_win = sum(wins) / len(wins) if wins else 0
            results.avg_loss = abs(sum(losses) / len(losses)) if losses else 0

            if losses:
                results.profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')

            # Average holding time
            holding_times = [(t.exit_time - t.entry_time).total_seconds() / 3600
                            for t in results.trades]
            results.avg_holding_time = sum(holding_times) / len(holding_times)

            # Sharpe ratio (simplified)
            returns = np.diff(results.equity_curve) / results.equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                results.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288)

        return results

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        if self.config and hasattr(self.config, 'get_pip_value'):
            return self.config.get_pip_value(symbol)
        return 0.01 if 'JPY' in symbol else 0.0001

    def print_results(self, results: BacktestResults) -> None:
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print(f"\nTotal P&L: ${results.total_pnl:,.2f}")
        print(f"Total Pips: {results.total_pips:.1f}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"\nMax Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.1%})")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"\nAvg Win: ${results.avg_win:.2f}")
        print(f"Avg Loss: ${results.avg_loss:.2f}")
        print(f"Avg Holding Time: {results.avg_holding_time:.1f} hours")
        print("=" * 60)
