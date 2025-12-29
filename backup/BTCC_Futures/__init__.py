"""
BTCC Futures Trading System
===========================
Complete trading system for BTCC.com Futures platform.

Features:
- REST API client with MD5 signature authentication
- Heartbeat mechanism (every 20 seconds)
- Position management (open, close, update TP/SL)
- Pending orders support
- Multiple trading strategies
- Paper and live trading modes
"""

from .btcc_api_client import BTCCAPIClient
from .btcc_config import BTCCConfig

__all__ = ['BTCCAPIClient', 'BTCCConfig']
__version__ = '1.0.0'
