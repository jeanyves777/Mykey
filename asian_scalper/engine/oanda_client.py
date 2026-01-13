"""
OANDA API Client for Asian Scalper
===================================
Dedicated client for portfolio scalping strategy
Uses account: 101-001-8364309-002
"""

import requests
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OANDA_CONFIG

logger = logging.getLogger(__name__)


class AsianScalperOANDAClient:
    """
    OANDA v20 API Client for Asian Session Portfolio Scalper

    Handles:
    - Account information
    - Market data (candles)
    - Batch order placement (all 15 pairs at once)
    - Portfolio-level position management
    - Individual trade management
    """

    def __init__(self, practice: bool = True):
        """Initialize OANDA client for Asian Scalper."""
        self.practice = practice
        self.api_key = OANDA_CONFIG["api_key"]

        # Set base URL based on environment
        if self.practice:
            self.base_url = OANDA_CONFIG["practice_url"]
            self.stream_url = OANDA_CONFIG["practice_stream_url"]
        else:
            self.base_url = OANDA_CONFIG["live_url"]
            self.stream_url = OANDA_CONFIG["live_stream_url"]

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }

        # DEDICATED Account ID for Asian Scalper
        self.account_id = OANDA_CONFIG["account_id"]

        logger.info(f"Asian Scalper OANDA Client initialized")
        logger.info(f"Account: {self.account_id} ({'PRACTICE' if practice else 'LIVE'})")

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                logger.info(f"Connected to OANDA - Account ID: {self.account_id}")
                return True
            else:
                logger.error(f"Failed to verify account: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_account_summary(self) -> Dict:
        """Get account summary (balance, NAV, P&L)."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                return response.json()["account"]
            else:
                logger.error(f"Failed to get account summary: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}

    def get_balance(self) -> float:
        """Get account balance."""
        summary = self.get_account_summary()
        return float(summary.get("balance", 0))

    def get_nav(self) -> float:
        """Get Net Asset Value (balance + unrealized P&L)."""
        summary = self.get_account_summary()
        return float(summary.get("NAV", 0))

    def get_unrealized_pl(self) -> float:
        """Get total unrealized P&L across all positions."""
        summary = self.get_account_summary()
        return float(summary.get("unrealizedPL", 0))

    def get_margin_available(self) -> float:
        """Get available margin for trading."""
        summary = self.get_account_summary()
        return float(summary.get("marginAvailable", 0))

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        count: int = 100
    ) -> pd.DataFrame:
        """
        Fetch candlestick data.

        Args:
            instrument: Instrument name (e.g., "EUR_USD")
            granularity: Timeframe (M1, M5, M15, H1, H4, D)
            count: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            url = f"{self.base_url}/v3/instruments/{instrument}/candles"
            params = {
                "granularity": granularity,
                "count": count,
                "price": "M"  # Mid prices
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                candles = data.get("candles", [])

                if not candles:
                    logger.warning(f"No candles returned for {instrument} {granularity}")
                    return pd.DataFrame()

                # Convert to DataFrame
                df_data = []
                for candle in candles:
                    if candle["complete"]:  # Only use completed candles
                        mid = candle["mid"]
                        df_data.append({
                            "time": candle["time"],
                            "open": float(mid["o"]),
                            "high": float(mid["h"]),
                            "low": float(mid["l"]),
                            "close": float(mid["c"]),
                            "volume": int(candle["volume"])
                        })

                df = pd.DataFrame(df_data)

                if not df.empty:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index("time")

                return df
            else:
                logger.error(f"Failed to get candles: {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching candles for {instrument}: {e}")
            return pd.DataFrame()

    def get_current_price(self, instrument: str) -> Dict:
        """Get current bid/ask prices."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
            params = {"instruments": instrument}

            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])

                if prices:
                    price = prices[0]
                    return {
                        "instrument": price["instrument"],
                        "bid": float(price["bids"][0]["price"]),
                        "ask": float(price["asks"][0]["price"]),
                        "mid": (float(price["bids"][0]["price"]) + float(price["asks"][0]["price"])) / 2,
                        "spread": float(price["asks"][0]["price"]) - float(price["bids"][0]["price"]),
                        "time": price["time"]
                    }

            return {}

        except Exception as e:
            logger.error(f"Error getting price for {instrument}: {e}")
            return {}

    def get_multiple_prices(self, instruments: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple instruments at once."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
            params = {"instruments": ",".join(instruments)}

            response = requests.get(url, headers=self.headers, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                prices = {}

                for price in data.get("prices", []):
                    instrument = price["instrument"]
                    prices[instrument] = {
                        "bid": float(price["bids"][0]["price"]),
                        "ask": float(price["asks"][0]["price"]),
                        "mid": (float(price["bids"][0]["price"]) + float(price["asks"][0]["price"])) / 2,
                        "spread": float(price["asks"][0]["price"]) - float(price["bids"][0]["price"]),
                        "time": price["time"]
                    }

                return prices

            return {}

        except Exception as e:
            logger.error(f"Error getting multiple prices: {e}")
            return {}

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Dict:
        """
        Place a market order.

        Args:
            instrument: Instrument name (e.g., "EUR_USD")
            units: Position size (positive = buy, negative = sell)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Order response
        """
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"

            order = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",  # Fill or Kill
                    "positionFill": "DEFAULT"
                }
            }

            # Add stop loss if specified
            if stop_loss is not None:
                order["order"]["stopLossOnFill"] = {
                    "price": str(round(stop_loss, 5)),
                    "timeInForce": "GTC"
                }

            # Add take profit if specified
            if take_profit is not None:
                order["order"]["takeProfitOnFill"] = {
                    "price": str(round(take_profit, 5)),
                    "timeInForce": "GTC"
                }

            response = requests.post(
                url,
                headers=self.headers,
                data=json.dumps(order),
                timeout=10
            )

            if response.status_code == 201:
                result = response.json()
                logger.info(f"Order placed: {instrument} {units} units")
                return result
            else:
                logger.error(f"Order failed: {response.status_code} - {response.text}")
                return {"error": response.text}

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/openTrades"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                trades = []

                for trade in data.get("trades", []):
                    trades.append({
                        "id": trade["id"],
                        "instrument": trade["instrument"],
                        "units": float(trade["currentUnits"]),
                        "price": float(trade["price"]),
                        "unrealized_pl": float(trade["unrealizedPL"]),
                        "open_time": trade["openTime"],
                        "stop_loss": float(trade.get("stopLossOrder", {}).get("price", 0)),
                        "take_profit": float(trade.get("takeProfitOrder", {}).get("price", 0))
                    })

                return trades

            return []

        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                positions = []

                for pos in data.get("positions", []):
                    instrument = pos["instrument"]
                    long_units = float(pos["long"]["units"])
                    short_units = float(pos["short"]["units"])

                    if long_units != 0:
                        positions.append({
                            "instrument": instrument,
                            "side": "LONG",
                            "units": long_units,
                            "average_price": float(pos["long"]["averagePrice"]),
                            "unrealized_pl": float(pos["long"]["unrealizedPL"]),
                        })

                    if short_units != 0:
                        positions.append({
                            "instrument": instrument,
                            "side": "SHORT",
                            "units": abs(short_units),
                            "average_price": float(pos["short"]["averagePrice"]),
                            "unrealized_pl": float(pos["short"]["unrealizedPL"]),
                        })

                return positions

            return []

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def close_trade(self, trade_id: str) -> Dict:
        """Close a specific trade by ID."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/close"

            response = requests.put(
                url,
                headers=self.headers,
                data=json.dumps({}),
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Trade {trade_id} closed")
                return result
            else:
                logger.error(f"Failed to close trade: {response.text}")
                return {"error": response.text}

        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return {"error": str(e)}

    def close_all_trades(self) -> List[Dict]:
        """Close ALL open trades - used when portfolio target is met."""
        trades = self.get_open_trades()
        results = []

        for trade in trades:
            result = self.close_trade(trade["id"])
            results.append({
                "trade_id": trade["id"],
                "instrument": trade["instrument"],
                "units": trade["units"],
                "unrealized_pl": trade["unrealized_pl"],
                "result": result
            })

        logger.info(f"Closed {len(results)} trades")
        return results

    def close_position(self, instrument: str, side: str) -> Dict:
        """Close a position for a specific instrument."""
        try:
            side_key = "long" if side == "LONG" else "short"
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"

            data = {
                side_key + "Units": "ALL"
            }

            response = requests.put(
                url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to close position: {response.text}")
                return {"error": response.text}

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"error": str(e)}

    def get_portfolio_unrealized_pl(self) -> float:
        """Get total unrealized P&L for all open positions."""
        return self.get_unrealized_pl()

    def get_trade_count(self) -> int:
        """Get number of open trades."""
        trades = self.get_open_trades()
        return len(trades)
