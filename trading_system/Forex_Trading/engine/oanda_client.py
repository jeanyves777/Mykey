"""
OANDA API Client Wrapper
Handles all API interactions with OANDA v20 REST API
"""
import os
import json
import time
from datetime import datetime, timedelta
import pytz
from typing import Optional, Dict, List, Any, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()


def create_retry_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class OandaClient:
    """OANDA v20 API client for forex trading"""

    # Price precision for different instruments (decimal places)
    # JPY pairs: 3 decimals, others: 5 decimals
    PRICE_PRECISION = {
        "USD_JPY": 3,
        "EUR_JPY": 3,
        "GBP_JPY": 3,
        "AUD_JPY": 3,
        "NZD_JPY": 3,
        "CAD_JPY": 3,
        "CHF_JPY": 3,
        # All other pairs default to 5
    }

    def _get_precision(self, instrument: str) -> int:
        """Get the price precision (decimal places) for an instrument"""
        return self.PRICE_PRECISION.get(instrument, 5)

    def _round_price(self, price: float, instrument: str) -> float:
        """Round price to the correct precision for the instrument"""
        precision = self._get_precision(instrument)
        return round(price, precision)

    def __init__(self, account_type: str = "practice"):
        """
        Initialize OANDA client

        Args:
            account_type: "practice" or "live"
        """
        self.account_type = account_type

        # Load credentials
        if account_type == "practice":
            self.api_key = os.getenv("OANDA_PRACTICE_API_KEY")
            self.account_id = os.getenv("OANDA_PRACTICE_ACCOUNT_ID")
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.api_key = os.getenv("OANDA_LIVE_API_KEY")
            self.account_id = os.getenv("OANDA_LIVE_ACCOUNT_ID")
            self.base_url = "https://api-fxtrade.oanda.com"

        if not self.api_key or not self.account_id:
            raise ValueError(f"OANDA {account_type} credentials not found in .env file")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Create session with retry logic for transient errors
        self.session = create_retry_session(retries=3, backoff_factor=1.0)
        self.session.headers.update(self.headers)

        print(f"[OANDA] Initialized {account_type} client")
        print(f"[OANDA] Account ID: {self.account_id}")

    def get_account_info(self) -> Dict:
        """Get account information including balance and positions"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()["account"]
        except Exception as e:
            print(f"[OANDA] Error getting account info: {e}")
            return {}

    def get_balance(self) -> float:
        """Get account balance"""
        account_info = self.get_account_info()
        return float(account_info.get("balance", 0))

    def get_nav(self) -> float:
        """Get Net Asset Value (NAV)"""
        account_info = self.get_account_info()
        return float(account_info.get("NAV", 0))

    def get_unrealized_pl(self) -> float:
        """Get unrealized P&L"""
        account_info = self.get_account_info()
        return float(account_info.get("unrealizedPL", 0))

    def get_margin_available(self) -> float:
        """Get available margin for trading"""
        account_info = self.get_account_info()
        return float(account_info.get("marginAvailable", 0))

    def get_candles(
        self,
        instrument: str,
        granularity: str = "M1",
        count: int = 500,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get historical candles

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: M1, M5, M15, M30, H1, H4, D (minute, hour, day)
            count: Number of candles (max 5000)
            from_time: Start time (optional)
            to_time: End time (optional)

        Returns:
            List of candles with OHLCV data
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"

        params = {
            "granularity": granularity,
            "price": "MBA"  # Mid, Bid, Ask prices
        }

        if from_time and to_time:
            params["from"] = from_time.isoformat() + "Z"
            params["to"] = to_time.isoformat() + "Z"
        else:
            params["count"] = min(count, 5000)

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            candles_raw = response.json()["candles"]

            # Convert to standard format
            candles = []
            for candle in candles_raw:
                if candle["complete"]:  # Only use completed candles
                    candles.append({
                        "time": datetime.fromisoformat(candle["time"].replace("Z", "+00:00")),
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": int(candle["volume"])
                    })

            return candles

        except Exception as e:
            print(f"[OANDA] Error getting candles for {instrument}: {e}")
            return []

    def get_current_price(self, instrument: str) -> Dict:
        """Get current bid/ask prices"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            pricing = response.json()["prices"][0]

            return {
                "instrument": pricing["instrument"],
                "bid": float(pricing["bids"][0]["price"]),
                "ask": float(pricing["asks"][0]["price"]),
                "spread": float(pricing["asks"][0]["price"]) - float(pricing["bids"][0]["price"]),
                "time": datetime.fromisoformat(pricing["time"].replace("Z", "+00:00"))
            }
        except Exception as e:
            print(f"[OANDA] Error getting current price for {instrument}: {e}")
            return {}

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            positions = response.json()["positions"]

            result = []
            for pos in positions:
                if float(pos["long"]["units"]) != 0 or float(pos["short"]["units"]) != 0:
                    result.append({
                        "instrument": pos["instrument"],
                        "long_units": float(pos["long"]["units"]),
                        "short_units": float(pos["short"]["units"]),
                        "unrealized_pl": float(pos["unrealizedPL"])
                    })

            return result

        except Exception as e:
            print(f"[OANDA] Error getting open positions: {e}")
            return []

    def get_open_trades(self) -> Optional[List[Dict]]:
        """
        Get all open trades

        Returns:
            List of trade dicts on success, None on connection error
            (None allows caller to distinguish between "no trades" and "API error")
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openTrades"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            trades = response.json()["trades"]

            result = []
            for trade in trades:
                result.append({
                    "id": trade["id"],
                    "instrument": trade["instrument"],
                    "price": float(trade["price"]),
                    "units": float(trade["initialUnits"]),
                    "current_units": float(trade["currentUnits"]),
                    "unrealized_pl": float(trade["unrealizedPL"]),
                    "open_time": datetime.fromisoformat(trade["openTime"].replace("Z", "+00:00")),
                    "take_profit": float(trade.get("takeProfitOrder", {}).get("price", 0)),
                    "stop_loss": float(trade.get("stopLossOrder", {}).get("price", 0)),
                    "trailing_stop": trade.get("trailingStopLossOrder", {}).get("distance")
                })

            return result

        except Exception as e:
            print(f"[OANDA] Error getting open trades: {e}")
            return None  # Return None to indicate error, not empty list

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None
    ) -> Dict:
        """
        Place a market order

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            units: Number of units (positive for BUY, negative for SELL)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            trailing_stop_distance: Trailing stop distance in pips (optional)

        Returns:
            Order response dict
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"

        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
                "positionFill": "DEFAULT"
            }
        }

        # Add stop loss (with instrument-specific precision)
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(self._round_price(stop_loss, instrument))
            }

        # Add take profit (with instrument-specific precision)
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(self._round_price(take_profit, instrument))
            }

        # Add trailing stop
        if trailing_stop_distance:
            order_data["order"]["trailingStopLossOnFill"] = {
                "distance": str(self._round_price(trailing_stop_distance, instrument))
            }

        try:
            response = self.session.post(url, json=order_data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if "orderFillTransaction" in result:
                fill = result["orderFillTransaction"]
                return {
                    "success": True,
                    "order_id": fill["orderID"],
                    "trade_id": fill["tradeOpened"]["tradeID"] if "tradeOpened" in fill else None,
                    "filled_price": float(fill["price"]),
                    "filled_units": float(fill["units"]),
                    "instrument": fill["instrument"],
                    "time": datetime.fromisoformat(fill["time"].replace("Z", "+00:00"))
                }
            else:
                return {"success": False, "error": "Order not filled"}

        except requests.exceptions.HTTPError as e:
            # Get detailed error message from OANDA
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get("errorMessage", str(e))
                print(f"[OANDA] Error placing order: {error_msg}")
                print(f"[OANDA] Full error: {error_detail}")
                return {"success": False, "error": error_msg, "detail": error_detail}
            except:
                print(f"[OANDA] Error placing order: {e}")
                return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"[OANDA] Error placing order: {e}")
            return {"success": False, "error": str(e)}

    def close_trade(self, trade_id: str, units: Optional[int] = None) -> Dict:
        """Close a trade (partially or fully)"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/close"

        data = {}
        if units:
            data["units"] = str(units)

        try:
            response = self.session.put(url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if "orderFillTransaction" in result:
                fill = result["orderFillTransaction"]
                return {
                    "success": True,
                    "trade_id": trade_id,
                    "closed_price": float(fill["price"]),
                    "pl": float(fill["pl"]),
                    "units": float(fill["units"])
                }
            else:
                return {"success": False, "error": "Trade not closed"}

        except Exception as e:
            print(f"[OANDA] Error closing trade: {e}")
            return {"success": False, "error": str(e)}

    def modify_trade(
        self,
        trade_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None,
        instrument: Optional[str] = None
    ) -> Dict:
        """Modify trade's stop loss, take profit, or trailing stop"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/orders"

        # Default to 5 decimals if instrument not provided
        precision = self._get_precision(instrument) if instrument else 5

        orders = {}

        if stop_loss:
            orders["stopLoss"] = {"price": str(round(stop_loss, precision))}

        if take_profit:
            orders["takeProfit"] = {"price": str(round(take_profit, precision))}

        if trailing_stop_distance:
            orders["trailingStopLoss"] = {"distance": str(round(trailing_stop_distance, precision))}

        try:
            response = self.session.put(url, json=orders, timeout=30)
            response.raise_for_status()
            return {"success": True}
        except Exception as e:
            print(f"[OANDA] Error modifying trade: {e}")
            return {"success": False, "error": str(e)}

    def get_trade_history(self, count: int = 50) -> List[Dict]:
        """Get closed trades history with exit reason detection"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades"
        params = {"state": "CLOSED", "count": count}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            trades = response.json()["trades"]

            result = []
            for trade in trades:
                # Determine exit reason from various OANDA fields
                exit_reason = "MANUAL"
                closing_ids = trade.get("closingTransactionIDs", [])

                # Method 1: Check specific fill transaction IDs (most reliable)
                if trade.get("stopLossOrderFillTransactionID"):
                    exit_reason = "SL"
                elif trade.get("takeProfitOrderFillTransactionID"):
                    exit_reason = "TP"
                elif trade.get("trailingStopLossOrderFillTransactionID"):
                    exit_reason = "TRAIL"
                else:
                    # Method 2: Check the closing transaction types
                    # Fetch the last closing transaction to determine type
                    if closing_ids:
                        try:
                            last_closing_id = closing_ids[-1]
                            tx_details = self.get_transaction_details(last_closing_id)
                            tx_type = tx_details.get("type", "")
                            tx_reason = tx_details.get("reason", "")

                            # OANDA transaction types for order fills:
                            # STOP_LOSS_ORDER, TAKE_PROFIT_ORDER, TRAILING_STOP_LOSS_ORDER
                            if "STOP_LOSS" in tx_type or "STOP_LOSS" in tx_reason:
                                if "TRAILING" in tx_type or "TRAILING" in tx_reason:
                                    exit_reason = "TRAIL"
                                else:
                                    exit_reason = "SL"
                            elif "TAKE_PROFIT" in tx_type or "TAKE_PROFIT" in tx_reason:
                                exit_reason = "TP"
                            elif "TRAILING" in tx_type or "TRAILING" in tx_reason:
                                exit_reason = "TRAIL"
                            elif tx_type == "ORDER_FILL" and tx_reason == "STOP_LOSS_ORDER":
                                exit_reason = "SL"
                            elif tx_type == "ORDER_FILL" and tx_reason == "TAKE_PROFIT_ORDER":
                                exit_reason = "TP"
                            elif tx_type == "ORDER_FILL" and tx_reason == "TRAILING_STOP_LOSS_ORDER":
                                exit_reason = "TRAIL"
                        except Exception:
                            pass  # Keep as MANUAL if we can't determine

                result.append({
                    "id": trade["id"],
                    "instrument": trade["instrument"],
                    "open_price": float(trade["price"]),
                    "average_close_price": float(trade["averageClosePrice"]),
                    "close_price": float(trade["averageClosePrice"]),
                    "units": float(trade["initialUnits"]),
                    "realized_pl": float(trade["realizedPL"]),
                    "open_time": datetime.fromisoformat(trade["openTime"].replace("Z", "+00:00")),
                    "close_time": datetime.fromisoformat(trade["closeTime"].replace("Z", "+00:00")),
                    "exit_reason": exit_reason,
                    "closing_transaction_ids": closing_ids
                })

            return result

        except Exception as e:
            print(f"[OANDA] Error getting trade history: {e}")
            return []

    def get_transaction_details(self, transaction_id: str) -> Dict:
        """Get details of a specific transaction to determine close reason"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/transactions/{transaction_id}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json().get("transaction", {})
        except Exception as e:
            # Silent fail - don't spam logs
            return {}


if __name__ == "__main__":
    # Test connection
    print("Testing OANDA connection...")
    client = OandaClient(account_type="practice")

    account_info = client.get_account_info()
    print(f"\nAccount Balance: ${client.get_balance():,.2f}")
    print(f"NAV: ${client.get_nav():,.2f}")
    print(f"Margin Available: ${client.get_margin_available():,.2f}")

    # Test getting EUR/USD price
    price = client.get_current_price("EUR_USD")
    print(f"\nEUR/USD: Bid={price['bid']}, Ask={price['ask']}, Spread={price['spread']:.5f}")

    # Test getting candles
    candles = client.get_candles("EUR_USD", granularity="M1", count=10)
    print(f"\nLast 10 1-min candles retrieved: {len(candles)}")
    if candles:
        print(f"Latest: {candles[-1]}")
