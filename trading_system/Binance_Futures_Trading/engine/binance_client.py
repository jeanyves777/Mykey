"""
Binance Futures API Client
==========================
Handles all API interactions with Binance Futures (Testnet & Mainnet)
"""

import os
import time
import hmac
import hashlib
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.trading_config import BINANCE_CONFIG, SYMBOL_SETTINGS


class BinanceClient:
    """Binance Futures API Client"""

    def __init__(self, testnet: bool = True, use_demo: bool = True):
        """
        Initialize Binance client

        Args:
            testnet: If True, use testnet/demo. If False, use mainnet.
            use_demo: If True, use demo.binance.com API (demo-fapi.binance.com)
        """
        self.testnet = testnet
        self.use_demo = use_demo

        # Use different API keys for demo vs live
        if testnet:
            self.api_key = BINANCE_CONFIG["api_key"]
            self.api_secret = BINANCE_CONFIG["api_secret"]
        else:
            # LIVE MODE - Use live API keys
            self.api_key = BINANCE_CONFIG["live_api_key"]
            self.api_secret = BINANCE_CONFIG["live_api_secret"]

        # Set base URL based on mode
        if testnet:
            if use_demo:
                # Use demo.binance.com API endpoint
                self.base_url = BINANCE_CONFIG["futures_demo_url"]
            else:
                # Use old testnet
                self.base_url = BINANCE_CONFIG["futures_testnet_url"]
        else:
            self.base_url = BINANCE_CONFIG["futures_mainnet_url"]

        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Time offset for server synchronization
        self.time_offset = 0
        self._sync_time()

    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for authenticated endpoints"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _sync_time(self):
        """Synchronize local time with Binance server"""
        try:
            url = f"{self.base_url}/fapi/v1/time"
            response = requests.get(url)
            if response.status_code == 200:
                server_time = response.json().get("serverTime", 0)
                local_time = int(time.time() * 1000)
                self.time_offset = server_time - local_time
                print(f"[INFO] Time sync: offset = {self.time_offset}ms")
        except Exception as e:
            print(f"[WARN] Time sync failed: {e}")
            self.time_offset = 0

    def _get_timestamp(self) -> int:
        """Get synchronized timestamp"""
        return int(time.time() * 1000) + self.time_offset

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(self, method: str, endpoint: str, params: Dict = None,
                 signed: bool = False) -> Dict:
        """
        Make API request

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request

        Returns:
            API response as dict
        """
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            params["timestamp"] = self._get_timestamp()
            params["signature"] = self._generate_signature(params)

        try:
            if method == "GET":
                response = self.session.get(url, params=params)
            elif method == "POST":
                response = self.session.post(url, params=params)
            elif method == "DELETE":
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[ERROR] Response: {e.response.text}")
            return {}

    # =========================================================================
    # Public Endpoints (No authentication required)
    # =========================================================================

    def get_server_time(self) -> int:
        """Get Binance server time"""
        result = self._request("GET", "/fapi/v1/time")
        return result.get("serverTime", 0)

    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol info"""
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get info for a specific symbol"""
        exchange_info = self.get_exchange_info()
        for s in exchange_info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return {}

    def get_current_price(self, symbol: str) -> Dict:
        """
        Get current price for a symbol

        Returns:
            Dict with 'price', 'bid', 'ask'
        """
        # Get ticker price
        ticker = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})

        # Get order book for bid/ask
        book = self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})

        return {
            "symbol": symbol,
            "price": float(ticker.get("price", 0)),
            "bid": float(book.get("bidPrice", 0)),
            "ask": float(book.get("askPrice", 0)),
            "timestamp": int(time.time() * 1000)
        }

    def get_klines(self, symbol: str, interval: str = "1m",
                   limit: int = 500, start_time: int = None,
                   end_time: int = None) -> pd.DataFrame:
        """
        Get candlestick/kline data

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1500)
            start_time: Start time in ms
            end_time: End time in ms

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500)
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._request("GET", "/fapi/v1/klines", params)

        if not data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df.set_index("open_time", inplace=True)

        return df

    def get_market_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get market data for all timeframes

        Args:
            symbol: Trading pair

        Returns:
            Dict with DataFrames for each timeframe
        """
        from config.trading_config import DATA_CONFIG

        data = {}

        # Fetch all timeframes
        data["1m"] = self.get_klines(symbol, "1m", DATA_CONFIG["candles_1m"])
        data["5m"] = self.get_klines(symbol, "5m", DATA_CONFIG["candles_5m"])
        data["15m"] = self.get_klines(symbol, "15m", DATA_CONFIG["candles_15m"])
        data["1h"] = self.get_klines(symbol, "1h", DATA_CONFIG["candles_1h"])

        return data

    # =========================================================================
    # Private Endpoints (Authentication required)
    # =========================================================================

    def get_account_info(self) -> Dict:
        """Get futures account information"""
        # Use v2 which has consistent position format with entryPrice
        # v3 has different position structure that may not include entryPrice
        result = self._request("GET", "/fapi/v2/account", signed=True)
        if not result or "code" in result:
            # Fallback to v3 if v2 fails
            result = self._request("GET", "/fapi/v3/account", signed=True)
        return result if result else {}

    def get_balance(self) -> float:
        """Get USDT balance"""
        account = self.get_account_info()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["walletBalance"])
        return 0.0

    def get_available_balance(self) -> float:
        """Get available USDT balance (excluding margin)"""
        account = self.get_account_info()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                return float(asset["availableBalance"])
        return 0.0

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        account = self.get_account_info()
        positions = []

        for pos in account.get("positions", []):
            try:
                position_amt = float(pos.get("positionAmt", 0))
                if position_amt != 0:
                    # Handle different API versions (v2/v3 may have different field names)
                    entry_price = float(pos.get("entryPrice", pos.get("avgPrice", 0)))
                    liq_price = float(pos.get("liquidationPrice", 0))
                    # Get isolated margin wallet (margin used for this position)
                    isolated_wallet = float(pos.get("isolatedWallet", pos.get("isolated_wallet", 0)))
                    positions.append({
                        "symbol": pos["symbol"],
                        "side": "LONG" if position_amt > 0 else "SHORT",
                        "quantity": abs(position_amt),
                        "entry_price": entry_price,
                        "unrealized_pnl": float(pos.get("unrealizedProfit", pos.get("unRealizedProfit", 0))),
                        "leverage": int(pos.get("leverage", 1)),
                        "liquidation_price": liq_price,
                        "isolated_wallet": isolated_wallet,
                    })
            except (KeyError, ValueError, TypeError) as e:
                # Skip malformed position entries
                continue

        return positions

    def get_position(self, symbol: str, position_side: str = None) -> Optional[Dict]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            position_side: Optional side filter for hedge mode ('LONG' or 'SHORT')
                          In hedge mode, you may have both LONG and SHORT positions.

        Returns:
            Position dict or None if not found
        """
        positions = self.get_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                # If position_side specified, match it (for hedge mode)
                if position_side and pos["side"] != position_side:
                    continue
                return pos
        return None

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol"""
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        return self._request("POST", "/fapi/v1/leverage", params, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict:
        """Set margin type (ISOLATED or CROSSED)"""
        params = {
            "symbol": symbol,
            "marginType": margin_type
        }
        return self._request("POST", "/fapi/v1/marginType", params, signed=True)

    def modify_isolated_position_margin(self, symbol: str, amount: float, add: bool = True, position_side: str = "BOTH") -> Dict:
        """
        Modify isolated position margin (add or reduce margin).

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            amount: Amount to add or reduce
            add: True to add margin, False to reduce
            position_side: 'BOTH' for One-Way mode, 'LONG'/'SHORT' for Hedge mode

        Returns:
            API response
        """
        params = {
            "symbol": symbol,
            "amount": str(amount),
            "type": 1 if add else 2,  # 1=Add, 2=Reduce
            "positionSide": position_side
        }
        return self._request("POST", "/fapi/v1/positionMargin", params, signed=True)

    def get_position_mode(self) -> Dict:
        """Get current position mode (One-Way or Hedge Mode)"""
        return self._request("GET", "/fapi/v1/positionSide/dual", {}, signed=True)

    def set_position_mode(self, hedge_mode: bool = True) -> Dict:
        """
        Enable or disable Hedge Mode (Dual Side Position Mode).

        Args:
            hedge_mode: True = Hedge Mode (LONG/SHORT), False = One-Way Mode

        Returns:
            API response

        NOTE: This can only be changed when there are NO open positions!
        """
        params = {
            "dualSidePosition": "true" if hedge_mode else "false"
        }
        return self._request("POST", "/fapi/v1/positionSide/dual", params, signed=True)

    def ensure_hedge_mode(self) -> bool:
        """
        Ensure hedge mode is enabled on the account.
        Returns True if hedge mode is enabled, False otherwise.
        """
        try:
            result = self.get_position_mode()
            is_hedge = result.get("dualSidePosition", False)

            if not is_hedge:
                print("[INFO] Enabling Hedge Mode (Dual Side Position)...")
                enable_result = self.set_position_mode(True)
                if "code" in enable_result:
                    print(f"[ERROR] Failed to enable Hedge Mode: {enable_result}")
                    return False
                print("[INFO] Hedge Mode enabled successfully!")
                return True
            else:
                print("[INFO] Hedge Mode already enabled")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to check/enable Hedge Mode: {e}")
            return False

    # =========================================================================
    # Order Endpoints
    # =========================================================================

    def place_market_order(self, symbol: str, side: str, quantity: float,
                           position_side: str = None) -> Dict:
        """
        Place a market order

        Args:
            symbol: Trading pair
            side: "BUY" or "SELL"
            quantity: Order quantity
            position_side: "LONG" or "SHORT" (required for hedge mode)
                           For LONG: BUY to open, SELL to close
                           For SHORT: SELL to open, BUY to close

        Returns:
            Order response
        """
        # Round quantity to symbol precision
        settings = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = settings.get("qty_precision", 3)
        quantity = round(quantity, qty_precision)

        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity
        }

        # Add positionSide for hedge mode
        if position_side:
            params["positionSide"] = position_side

        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_limit_order(self, symbol: str, side: str, quantity: float,
                          price: float, time_in_force: str = "GTC") -> Dict:
        """
        Place a limit order

        Args:
            symbol: Trading pair
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Limit price
            time_in_force: GTC, IOC, FOK

        Returns:
            Order response
        """
        settings = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = settings.get("qty_precision", 3)
        price_precision = settings.get("price_precision", 2)

        quantity = round(quantity, qty_precision)
        price = round(price, price_precision)

        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": time_in_force
        }

        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_loss(self, symbol: str, side: str, quantity: float,
                        stop_price: float, position_side: str = None) -> Dict:
        """
        Place a stop loss order

        Args:
            symbol: Trading pair
            side: "BUY" to close short, "SELL" to close long
            quantity: Order quantity
            stop_price: Stop trigger price
            position_side: "LONG" or "SHORT" (required for hedge mode)

        Returns:
            Order response
        """
        settings = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = settings.get("qty_precision", 3)
        price_precision = settings.get("price_precision", 2)

        quantity = round(quantity, qty_precision)
        stop_price = round(stop_price, price_precision)

        # As of Dec 2025, Binance mainnet requires Algo Order API for conditional orders
        # Demo still uses the regular endpoint
        if self.testnet:
            # Demo mode - use regular order endpoint
            params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "quantity": quantity,
                "stopPrice": stop_price,
            }
            # In hedge mode, use positionSide instead of reduceOnly
            if position_side:
                params["positionSide"] = position_side
            else:
                params["reduceOnly"] = "true"
            return self._request("POST", "/fapi/v1/order", params, signed=True)
        else:
            # Mainnet - use Algo Order API
            params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "algoType": "CONDITIONAL",
                "quantity": quantity,
                "triggerPrice": stop_price,
                "workingType": "CONTRACT_PRICE"
            }
            if position_side:
                params["positionSide"] = position_side
            result = self._request("POST", "/fapi/v1/algoOrder", params, signed=True)
            if result and "algoId" in result:
                result["orderId"] = result.get("algoId")
            return result

    def place_take_profit(self, symbol: str, side: str, quantity: float,
                          take_profit_price: float, position_side: str = None) -> Dict:
        """
        Place a take profit order

        Args:
            symbol: Trading pair
            side: "BUY" to close short, "SELL" to close long
            quantity: Order quantity
            take_profit_price: Take profit trigger price
            position_side: "LONG" or "SHORT" (required for hedge mode)

        Returns:
            Order response
        """
        settings = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = settings.get("qty_precision", 3)
        price_precision = settings.get("price_precision", 2)

        quantity = round(quantity, qty_precision)
        take_profit_price = round(take_profit_price, price_precision)

        # As of Dec 2025, Binance mainnet requires Algo Order API for conditional orders
        # Demo still uses the regular endpoint
        if self.testnet:
            # Demo mode - use regular order endpoint
            params = {
                "symbol": symbol,
                "side": side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": quantity,
                "stopPrice": take_profit_price,
            }
            # In hedge mode, use positionSide instead of reduceOnly
            if position_side:
                params["positionSide"] = position_side
            else:
                params["reduceOnly"] = "true"
            return self._request("POST", "/fapi/v1/order", params, signed=True)
        else:
            # Mainnet - use Algo Order API
            params = {
                "symbol": symbol,
                "side": side,
                "type": "TAKE_PROFIT_MARKET",
                "algoType": "CONDITIONAL",
                "quantity": quantity,
                "triggerPrice": take_profit_price,
                "workingType": "CONTRACT_PRICE"
            }
            if position_side:
                params["positionSide"] = position_side
            result = self._request("POST", "/fapi/v1/algoOrder", params, signed=True)
            if result and "algoId" in result:
                result["orderId"] = result.get("algoId")
            return result

    def cancel_order(self, symbol: str, order_id: int, is_algo_order: bool = False) -> Dict:
        """Cancel an order (regular or algo order on mainnet)"""
        if is_algo_order and not self.testnet:
            # Cancel algo order on mainnet
            return self.cancel_algo_order(symbol, order_id)
        else:
            # Cancel regular order
            params = {
                "symbol": symbol,
                "orderId": order_id
            }
            return self._request("DELETE", "/fapi/v1/order", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancel all open orders for a symbol (including algo orders on mainnet)"""
        results = {}

        # Cancel regular orders
        params = {"symbol": symbol}
        results["regular"] = self._request("DELETE", "/fapi/v1/allOpenOrders", params, signed=True)

        # On mainnet, also cancel algo orders
        if not self.testnet:
            algo_orders = self.get_algo_orders(symbol)
            results["algo_cancelled"] = []
            for order in algo_orders:
                algo_id = order.get("algoId")
                if algo_id:
                    try:
                        cancel_result = self.cancel_algo_order(symbol, algo_id)
                        results["algo_cancelled"].append({"algoId": algo_id, "result": cancel_result})
                    except:
                        pass

        return results

    def cancel_orders_for_side(self, symbol: str, position_side: str) -> Dict:
        """
        Cancel all open orders for a specific position side (LONG or SHORT).
        Used in hedge mode to only cancel orders for one side.
        """
        results = {"cancelled": [], "failed": []}

        # Get all open orders for this symbol
        all_orders = self.get_open_orders(symbol)

        for order in all_orders:
            order_position_side = order.get("positionSide", "BOTH")

            # Only cancel orders matching the specified side
            if order_position_side == position_side:
                order_id = order.get("orderId") or order.get("algoId")
                if order_id:
                    try:
                        # Check if it's an algo order
                        is_algo = "algoId" in order
                        if is_algo:
                            cancel_result = self.cancel_algo_order(symbol, order_id)
                        else:
                            cancel_result = self.cancel_order(symbol, order_id)
                        results["cancelled"].append({"orderId": order_id, "type": order.get("type", "unknown")})
                    except Exception as e:
                        results["failed"].append({"orderId": order_id, "error": str(e)})

        return results

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders (including algo orders on mainnet)"""
        params = {}
        if symbol:
            params["symbol"] = symbol

        # Get regular open orders
        regular_orders = self._request("GET", "/fapi/v1/openOrders", params, signed=True)
        if not isinstance(regular_orders, list):
            regular_orders = []

        # On mainnet, also get algo orders (TP/SL orders are in algo orders endpoint)
        if not self.testnet:
            algo_orders = self.get_algo_orders(symbol)
            # Combine both lists
            return regular_orders + algo_orders

        return regular_orders

    def get_algo_orders(self, symbol: str = None) -> List[Dict]:
        """Get algo orders (conditional orders like TP/SL on mainnet)"""
        params = {}
        if symbol:
            params["symbol"] = symbol

        # Use /fapi/v1/openAlgoOrders for current open algo orders
        result = self._request("GET", "/fapi/v1/openAlgoOrders", params, signed=True)

        if not isinstance(result, list):
            # Try different response format
            if isinstance(result, dict) and "orders" in result:
                result = result.get("orders", [])
            else:
                return []

        # Normalize algo order format to match regular order format
        normalized = []
        for order in result:
            # Algo orders may use different field names
            order_type = order.get("orderType", order.get("type", ""))
            # Normalize order type to match regular order format
            # Algo orders may return "STOP" instead of "STOP_MARKET"
            if order_type == "STOP":
                order_type = "STOP_MARKET"
            elif order_type == "TAKE_PROFIT":
                order_type = "TAKE_PROFIT_MARKET"

            normalized.append({
                "orderId": order.get("algoId", order.get("orderId")),
                "algoId": order.get("algoId"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "positionSide": order.get("positionSide", "BOTH"),  # Include positionSide for hedge mode
                "type": order_type,
                "status": order.get("algoStatus", order.get("status")),
                "triggerPrice": float(order.get("triggerPrice", 0)),
                "stopPrice": float(order.get("triggerPrice", 0)),
                "quantity": float(order.get("quantity", order.get("origQty", 0))),
                "origQty": float(order.get("quantity", order.get("origQty", 0))),
                "isAlgoOrder": True
            })

        return normalized

    def cancel_algo_order(self, symbol: str, algo_id: int) -> Dict:
        """Cancel an algo order on mainnet"""
        params = {
            "symbol": symbol,
            "algoId": algo_id
        }
        return self._request("DELETE", "/fapi/v1/algoOrder", params, signed=True)

    def get_income_history(self, symbol: str = None, income_type: str = "REALIZED_PNL",
                           limit: int = 10) -> List[Dict]:
        """
        Get income history (realized PNL, funding fees, etc.)

        Args:
            symbol: Trading pair (optional)
            income_type: Type of income - REALIZED_PNL, FUNDING_FEE, COMMISSION, etc.
            limit: Number of records to return

        Returns:
            List of income records
        """
        params = {
            "incomeType": income_type,
            "limit": limit
        }
        if symbol:
            params["symbol"] = symbol

        result = self._request("GET", "/fapi/v1/income", params, signed=True)
        return result if isinstance(result, list) else []

    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get recent account trades for a symbol

        Args:
            symbol: Trading pair
            limit: Number of trades to return

        Returns:
            List of trade records
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        result = self._request("GET", "/fapi/v1/userTrades", params, signed=True)
        return result if isinstance(result, list) else []

    def close_position(self, symbol: str) -> Dict:
        """
        Close an entire position

        Args:
            symbol: Trading pair

        Returns:
            Order response
        """
        position = self.get_position(symbol)
        if not position:
            return {"error": "No position found"}

        # Determine close side (opposite of position)
        close_side = "SELL" if position["side"] == "LONG" else "BUY"

        return self.place_market_order(symbol, close_side, position["quantity"])

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            server_time = self.get_server_time()
            if server_time > 0:
                mode = "Demo (demo.binance.com)" if self.use_demo else ("Testnet" if self.testnet else "Mainnet")
                print(f"[OK] Connected to Binance {mode}")
                print(f"[OK] Base URL: {self.base_url}")
                print(f"[OK] Server time: {datetime.fromtimestamp(server_time/1000)}")
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Connection test failed: {e}")
            print(f"[ERROR] Base URL: {self.base_url}")
            return False

    def get_historical_klines(self, symbol: str, interval: str,
                               start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Get historical klines for backtesting

        Args:
            symbol: Trading pair
            interval: Kline interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to now

        Returns:
            DataFrame with historical OHLCV data
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        all_data = []
        current_start = start_ms

        while current_start < end_ms:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1500,
                start_time=current_start,
                end_time=end_ms
            )

            if df.empty:
                break

            all_data.append(df)

            # Move to next batch
            current_start = int(df.index[-1].timestamp() * 1000) + 1

            # Rate limiting
            time.sleep(0.2)

        if all_data:
            return pd.concat(all_data)

        return pd.DataFrame()


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Test connection
    client = BinanceClient(testnet=True)

    if client.test_connection():
        # Get current price
        price = client.get_current_price("BTCUSDT")
        print(f"\nBTCUSDT Price: ${price['price']:,.2f}")
        print(f"  Bid: ${price['bid']:,.2f}")
        print(f"  Ask: ${price['ask']:,.2f}")

        # Get recent candles
        print("\nFetching 1-minute candles...")
        df = client.get_klines("BTCUSDT", "1m", 10)
        print(df[["open", "high", "low", "close", "volume"]].tail())

        # Try to get account info (may fail on testnet without proper setup)
        try:
            balance = client.get_balance()
            print(f"\nAccount Balance: ${balance:,.2f} USDT")
        except Exception as e:
            print(f"\n[NOTE] Could not get balance (testnet may require setup): {e}")
