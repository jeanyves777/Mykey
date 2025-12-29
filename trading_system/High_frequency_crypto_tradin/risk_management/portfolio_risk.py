"""
Portfolio Risk Analyzer
========================

Advanced portfolio-level risk analysis including:
- Correlation analysis
- Value at Risk (VaR)
- Expected Shortfall
- Beta and volatility metrics
- Concentration risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk metrics."""
    total_value: float = 0.0
    total_exposure: float = 0.0
    cash: float = 0.0

    # Volatility
    portfolio_volatility: float = 0.0
    portfolio_beta: float = 0.0

    # VaR metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0

    # Concentration
    max_position_weight: float = 0.0
    herfindahl_index: float = 0.0  # Concentration measure

    # Correlation
    avg_correlation: float = 0.0
    max_correlation: float = 0.0

    # Drawdown
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0


class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analysis.

    Provides advanced risk metrics for portfolio management.
    """

    def __init__(self, risk_manager=None, position_manager=None):
        """
        Initialize Portfolio Risk Analyzer.

        Args:
            risk_manager: RiskManager instance
            position_manager: PositionManager instance
        """
        self.risk_manager = risk_manager
        self.position_manager = position_manager

        # Historical data for calculations
        self.returns_history: Dict[str, List[float]] = {}
        self.portfolio_values: List[float] = []
        self.peak_value = 0.0

    def update_returns(self, symbol: str, return_value: float):
        """Add a return observation for a symbol."""
        if symbol not in self.returns_history:
            self.returns_history[symbol] = []
        self.returns_history[symbol].append(return_value)

        # Keep history manageable
        if len(self.returns_history[symbol]) > 1000:
            self.returns_history[symbol] = self.returns_history[symbol][-500:]

    def update_portfolio_value(self, value: float):
        """Update portfolio value for tracking."""
        self.portfolio_values.append(value)
        self.peak_value = max(self.peak_value, value)

        if len(self.portfolio_values) > 1000:
            self.portfolio_values = self.portfolio_values[-500:]

    # ==========================================
    # CORRELATION ANALYSIS
    # ==========================================

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for tracked symbols."""
        if len(self.returns_history) < 2:
            return pd.DataFrame()

        # Build DataFrame from returns
        min_length = min(len(v) for v in self.returns_history.values())
        if min_length < 10:
            return pd.DataFrame()

        data = {
            symbol: returns[-min_length:]
            for symbol, returns in self.returns_history.items()
        }

        df = pd.DataFrame(data)
        return df.corr()

    def get_correlation_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get pairs of symbols with high correlation."""
        corr_matrix = self.calculate_correlation_matrix()
        if corr_matrix.empty:
            return []

        pairs = []
        symbols = list(corr_matrix.columns)

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    corr = corr_matrix.loc[sym1, sym2]
                    if abs(corr) >= threshold:
                        pairs.append((sym1, sym2, corr))

        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def check_correlation_risk(self, symbol: str, threshold: float = 0.7) -> Tuple[bool, List[str]]:
        """
        Check if adding a position would create correlation risk.

        Args:
            symbol: Symbol to check
            threshold: Correlation threshold

        Returns:
            Tuple of (has_risk, correlated_symbols)
        """
        if not self.position_manager:
            return False, []

        corr_matrix = self.calculate_correlation_matrix()
        if corr_matrix.empty or symbol not in corr_matrix.columns:
            return False, []

        open_symbols = self.position_manager.get_symbols()
        correlated = []

        for open_sym in open_symbols:
            if open_sym in corr_matrix.columns:
                corr = corr_matrix.loc[symbol, open_sym]
                if abs(corr) >= threshold:
                    correlated.append(open_sym)

        return len(correlated) > 0, correlated

    # ==========================================
    # VALUE AT RISK (VaR)
    # ==========================================

    def calculate_var(self,
                      portfolio_value: float,
                      confidence: float = 0.95,
                      horizon_days: int = 1) -> float:
        """
        Calculate Value at Risk using historical simulation.

        Args:
            portfolio_value: Current portfolio value
            confidence: Confidence level (0.95 = 95%)
            horizon_days: Time horizon in days

        Returns:
            VaR in dollar terms
        """
        if len(self.portfolio_values) < 20:
            # Not enough data, use rough estimate
            return portfolio_value * 0.02 * np.sqrt(horizon_days)  # 2% daily vol estimate

        # Calculate portfolio returns
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        if len(returns) < 10:
            return portfolio_value * 0.02 * np.sqrt(horizon_days)

        # Historical VaR
        var_percentile = np.percentile(returns, (1 - confidence) * 100)
        var = abs(var_percentile) * portfolio_value * np.sqrt(horizon_days)

        return var

    def calculate_expected_shortfall(self,
                                     portfolio_value: float,
                                     confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (CVaR).

        Args:
            portfolio_value: Current portfolio value
            confidence: Confidence level

        Returns:
            Expected Shortfall in dollar terms
        """
        if len(self.portfolio_values) < 20:
            return portfolio_value * 0.03  # Rough estimate

        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        if len(returns) < 10:
            return portfolio_value * 0.03

        var_percentile = (1 - confidence) * 100
        threshold = np.percentile(returns, var_percentile)
        tail_returns = returns[returns <= threshold]

        if len(tail_returns) == 0:
            return self.calculate_var(portfolio_value, confidence)

        es = abs(np.mean(tail_returns)) * portfolio_value
        return es

    # ==========================================
    # VOLATILITY & BETA
    # ==========================================

    def calculate_portfolio_volatility(self, annualize: bool = True) -> float:
        """Calculate portfolio volatility."""
        if len(self.portfolio_values) < 10:
            return 0.0

        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        volatility = np.std(returns)

        if annualize:
            # Assuming minute data, 252 days * 24 hours * 60 minutes
            volatility *= np.sqrt(252 * 24 * 60)

        return volatility

    def calculate_individual_volatility(self, symbol: str, annualize: bool = True) -> float:
        """Calculate volatility for a specific symbol."""
        if symbol not in self.returns_history or len(self.returns_history[symbol]) < 10:
            return 0.0

        returns = np.array(self.returns_history[symbol])
        volatility = np.std(returns)

        if annualize:
            volatility *= np.sqrt(252 * 24 * 60)

        return volatility

    # ==========================================
    # CONCENTRATION RISK
    # ==========================================

    def calculate_concentration_metrics(self) -> Dict:
        """Calculate portfolio concentration metrics."""
        if not self.position_manager:
            return {'herfindahl_index': 0, 'max_weight': 0}

        positions = self.position_manager.get_all_positions()
        if not positions:
            return {'herfindahl_index': 0, 'max_weight': 0}

        total_value = sum(pos.market_value for pos in positions)
        if total_value == 0:
            return {'herfindahl_index': 0, 'max_weight': 0}

        weights = [pos.market_value / total_value for pos in positions]

        # Herfindahl-Hirschman Index (sum of squared weights)
        hhi = sum(w ** 2 for w in weights)

        return {
            'herfindahl_index': hhi,
            'max_weight': max(weights) if weights else 0,
            'min_weight': min(weights) if weights else 0,
            'avg_weight': np.mean(weights) if weights else 0,
            'position_count': len(positions),
            'effective_positions': 1 / hhi if hhi > 0 else 0  # Effective number of positions
        }

    # ==========================================
    # DRAWDOWN ANALYSIS
    # ==========================================

    def calculate_drawdown(self) -> Tuple[float, float]:
        """
        Calculate current and maximum drawdown.

        Returns:
            Tuple of (current_drawdown, max_drawdown)
        """
        if len(self.portfolio_values) < 2:
            return 0.0, 0.0

        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak

        current_dd = drawdown[-1]
        max_dd = np.max(drawdown)

        return current_dd, max_dd

    def get_drawdown_periods(self) -> List[Dict]:
        """Get list of drawdown periods."""
        if len(self.portfolio_values) < 10:
            return []

        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak

        periods = []
        in_drawdown = False
        start_idx = 0

        for i, dd in enumerate(drawdown):
            if dd > 0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_idx = i
            elif dd < 0.005 and in_drawdown:  # End of drawdown (<0.5%)
                in_drawdown = False
                max_dd = np.max(drawdown[start_idx:i])
                periods.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'duration': i - start_idx,
                    'max_drawdown': max_dd
                })

        return periods

    # ==========================================
    # COMPREHENSIVE ANALYSIS
    # ==========================================

    def get_full_risk_report(self, portfolio_value: float = None) -> PortfolioRiskMetrics:
        """
        Generate comprehensive risk report.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            PortfolioRiskMetrics with all calculated metrics
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_values[-1] if self.portfolio_values else 0

        metrics = PortfolioRiskMetrics()
        metrics.total_value = portfolio_value

        # Position metrics
        if self.position_manager:
            summary = self.position_manager.get_portfolio_summary()
            metrics.total_exposure = summary.get('total_exposure', 0)

        # Concentration
        conc = self.calculate_concentration_metrics()
        metrics.max_position_weight = conc.get('max_weight', 0)
        metrics.herfindahl_index = conc.get('herfindahl_index', 0)

        # Volatility
        metrics.portfolio_volatility = self.calculate_portfolio_volatility()

        # VaR
        metrics.var_95 = self.calculate_var(portfolio_value, 0.95)
        metrics.var_99 = self.calculate_var(portfolio_value, 0.99)
        metrics.expected_shortfall = self.calculate_expected_shortfall(portfolio_value)

        # Correlation
        corr_matrix = self.calculate_correlation_matrix()
        if not corr_matrix.empty:
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_corrs = corr_matrix.values[mask]
            if len(upper_corrs) > 0:
                metrics.avg_correlation = np.mean(np.abs(upper_corrs))
                metrics.max_correlation = np.max(np.abs(upper_corrs))

        # Drawdown
        metrics.current_drawdown, metrics.max_drawdown = self.calculate_drawdown()

        return metrics

    def get_risk_alerts(self, config=None) -> List[str]:
        """
        Get list of risk alerts based on current portfolio state.

        Args:
            config: RiskConfig with thresholds

        Returns:
            List of alert messages
        """
        alerts = []

        report = self.get_full_risk_report()

        # Concentration alerts
        if report.max_position_weight > 0.25:
            alerts.append(f"High concentration: {report.max_position_weight*100:.1f}% in single position")

        # Correlation alerts
        if report.avg_correlation > 0.6:
            alerts.append(f"High average correlation: {report.avg_correlation:.2f}")

        # Drawdown alerts
        if report.current_drawdown > 0.10:
            alerts.append(f"Significant drawdown: {report.current_drawdown*100:.1f}%")

        if report.max_drawdown > 0.20:
            alerts.append(f"Large max drawdown: {report.max_drawdown*100:.1f}%")

        # Volatility alerts
        if report.portfolio_volatility > 0.50:  # 50% annualized
            alerts.append(f"High portfolio volatility: {report.portfolio_volatility*100:.1f}% annualized")

        return alerts

    def reset(self):
        """Reset all historical data."""
        self.returns_history.clear()
        self.portfolio_values.clear()
        self.peak_value = 0.0
