"""
Forex Trading System
====================
HTF Confluence Strategy for Forex markets using OANDA API v20.

This is a direct port of the successful Binance Futures trading strategy
to Forex markets, maintaining identical strategy logic while adapting for
OANDA API and pip-based calculations.

Strategy:
- HTF (Higher Timeframe) Trend + Confluence
- 8/8 confluence requirement for entry
- Multi-timeframe analysis (4H, 1H, 15m, 5m, 1m)
- Smart profit lock and trailing profit lock
- Fakeout protection and damage control

Usage:
    python main.py              # Run on practice account
    python main.py --live       # Run on live account (CAUTION)
    python test_connection.py   # Test OANDA API connection
"""

__version__ = "1.0.0"
__author__ = "Mykey Trading Systems"
__license__ = "Private"

# Package info
PACKAGE_NAME = "forex_trading_system"
DESCRIPTION = "HTF Confluence Strategy for Forex (OANDA API v20)"

# Import main components for easy access
from engine.oanda_client import OANDAClient
from engine.htf_confluence_strategy import HTFConfluenceForexStrategy
from engine.htf_confluence_live_engine import HTFConfluenceForexEngine

__all__ = [
    "OANDAClient",
    "HTFConfluenceForexStrategy",
    "HTFConfluenceForexEngine",
]
