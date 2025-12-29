"""
Centralized Risk Management Module
====================================

This module provides unified risk management for:
- Training (feature target creation with risk-adjusted returns)
- Backtesting (realistic position sizing and risk limits)
- Simulation (paper trading with risk controls)
- Live Trading (real-time risk monitoring and enforcement)

All components use the same risk rules to ensure consistency.
"""

from .risk_manager import RiskManager, RiskConfig, RiskLevel
from .position_manager import PositionManager, Position, PositionSide
from .trade_executor import TradeExecutor, Order, OrderType, OrderSide, OrderStatus, ExecutionMode
from .portfolio_risk import PortfolioRiskAnalyzer

__all__ = [
    'RiskManager',
    'RiskConfig',
    'RiskLevel',
    'PositionManager',
    'Position',
    'PositionSide',
    'TradeExecutor',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'ExecutionMode',
    'PortfolioRiskAnalyzer'
]
