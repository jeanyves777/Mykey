"""
OANDA API Client
===============
Handles all interactions with OANDA v20 API
"""

import requests
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from config.trading_config import OANDA_CONFIG

logger = logging.getLogger(__name__)


class OANDAClient:
    """
    OANDA v20 API Client
    
    Handles:
    - Account information
    - Market data (candles)
    - Order placement (market, limit, stop)
    - Position management
    - Trade history
    """
    
    def __init__(self, practice: bool = True):
        """
        Initialize OANDA client.
        
        Args:
            practice: Use practice account (True) or live (False)
        """
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
        
        # Account ID from config (or will be fetched)
        self.account_id = OANDA_CONFIG["account_id"] or None
        
        logger.info(f"OANDA Client initialized ({'PRACTICE' if practice else 'LIVE'})")
    
    def test_connection(self) -> bool:
        """Test API connection and fetch account ID if not set."""
        try:
            # If account ID already set, just verify it
            if self.account_id:
                url = f"{self.base_url}/v3/accounts/{self.account_id}"
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"Connected to OANDA - Account ID: {self.account_id}")
                    return True
                else:
                    logger.error(f"Failed to verify account: {response.status_code}")
                    return False
            
            # Otherwise, fetch account list
            url = f"{self.base_url}/v3/accounts"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                accounts = data.get("accounts", [])
                
                if accounts:
                    self.account_id = accounts[0]["id"]
                    logger.info(f"Connected to OANDA - Account ID: {self.account_id}")
                    return True
                else:
                    logger.error("No accounts found")
                    return False
            else:
                logger.error(f"Connection failed: {response.status_code} - {response.text}")
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
        """Get total unrealized P&L."""
        summary = self.get_account_summary()
        return float(summary.get("unrealizedPL", 0))
    
    def get_margin_available(self) -> float:
        """Get available margin for trading."""
        summary = self.get_account_summary()
        return float(summary.get("marginAvailable", 0))

    def get_margin_rate(self, instrument: str) -> float:
        """Get margin rate for a specific instrument."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
            params = {"instruments": instrument}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                instruments = data.get("instruments", [])
                if instruments:
                    return float(instruments[0].get("marginRate", 0.02))  # Default 2% if not found
            return 0.02  # Default 2% margin (50:1 leverage)
        except Exception as e:
            logger.error(f"Error getting margin rate for {instrument}: {e}")
            return 0.02

    def get_candles(
        self,
        instrument: str,
        granularity: str,
        count: int = 500
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
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add take profit if specified
            if take_profit is not None:
                order["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            response = requests.post(
                url,
                headers=self.headers,
                data=json.dumps(order),
                timeout=10
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                logger.error(f"Order failed: {response.status_code} - {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}
    
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
                    
                    # Add long position if exists
                    if long_units != 0:
                        positions.append({
                            "instrument": instrument,
                            "side": "LONG",
                            "units": long_units,
                            "average_price": float(pos["long"]["averagePrice"]),
                            "unrealized_pl": float(pos["long"]["unrealizedPL"]),
                            "financing": float(pos["long"].get("financing", 0))
                        })
                    
                    # Add short position if exists
                    if short_units != 0:
                        positions.append({
                            "instrument": instrument,
                            "side": "SHORT",
                            "units": abs(short_units),
                            "average_price": float(pos["short"]["averagePrice"]),
                            "unrealized_pl": float(pos["short"]["unrealizedPL"]),
                            "financing": float(pos["short"].get("financing", 0))
                        })
                
                return positions
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, instrument: str) -> Optional[Dict]:
        """Get position for specific instrument."""
        positions = self.get_open_positions()
        
        for pos in positions:
            if pos["instrument"] == instrument:
                return pos
        
        return None
    
    def close_position(self, instrument: str, side: str) -> Dict:
        """
        Close a position.
        
        Args:
            instrument: Instrument name
            side: "LONG" or "SHORT"
            
        Returns:
            Close response
        """
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
    
    def modify_trade(
        self,
        trade_id: str,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Dict:
        """Modify stop loss or take profit on existing trade."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/orders"
            
            # Prepare order modifications
            data = {}
            
            if stop_loss is not None:
                data["stopLoss"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            if take_profit is not None:
                data["takeProfit"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            if not data:
                return {"error": "No modifications specified"}
            
            response = requests.put(
                url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to modify trade: {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            logger.error(f"Error modifying trade: {e}")
            return {"error": str(e)}
    
    def get_transaction_history(
        self,
        from_time: str = None,
        to_time: str = None,
        count: int = 100
    ) -> List[Dict]:
        """Get transaction history."""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/transactions"
            params = {"count": count}
            
            if from_time:
                params["from"] = from_time
            if to_time:
                params["to"] = to_time
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("transactions", [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []
