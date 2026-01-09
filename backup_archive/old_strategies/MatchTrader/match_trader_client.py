"""
Match-Trader API Client for FundedNext Prop Trading
====================================================
REST API integration for Match-Trader platform used by FundedNext.

API Docs: https://app.theneo.io/match-trade/platform-api/introduction
Rate Limit: 500 requests/minute
Token Expiry: 15 minutes (refresh max 4/hour)
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position"""
    position_id: str
    instrument: str
    side: str  # "BUY" or "SELL"
    volume: float
    entry_price: float
    current_price: float
    sl_price: float
    tp_price: float
    profit: float
    open_time: str


@dataclass
class AccountBalance:
    """Account balance information"""
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    profit_loss: float


class MatchTraderClient:
    """
    Match-Trader API Client

    Usage:
        client = MatchTraderClient(
            base_url="https://your-broker-url.match-trader.com",
            broker_id="your_broker_id"
        )
        client.login("email@example.com", "password")

        # Get balance
        balance = client.get_balance()

        # Open position
        order_id = client.open_position(
            instrument="EURUSD",
            side="BUY",
            volume=0.01,
            sl_price=1.0500,
            tp_price=1.0600
        )

        # Close position
        client.close_position(position_id, "EURUSD", "BUY", 0.01)
    """

    def __init__(self, base_url: str, broker_id: str, system_uuid: str = None):
        """
        Initialize Match-Trader client

        Args:
            base_url: Broker's Match-Trader URL (e.g., https://mtr-demo-prod.match-trader.com)
            broker_id: Broker identifier for login
            system_uuid: System UUID for API calls (obtained after login)
        """
        self.base_url = base_url.rstrip('/')
        self.broker_id = broker_id
        self.system_uuid = system_uuid

        # Authentication tokens
        self.trading_api_token: Optional[str] = None
        self.auth_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.12  # ~500 requests/minute = 120ms between requests

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication tokens"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if self.trading_api_token:
            headers['Auth-trading-api'] = self.trading_api_token
        return headers

    def _get_cookies(self) -> Dict[str, str]:
        """Get authentication cookies"""
        if self.auth_token:
            return {'co-auth': self.auth_token}
        return {}

    def _check_token_expiry(self):
        """Check if token needs refresh"""
        if self.token_expiry and datetime.now() >= self.token_expiry - timedelta(minutes=2):
            logger.info("Token expiring soon, refreshing...")
            self.refresh_token()

    def _make_request(self, method: str, endpoint: str, data: Dict = None,
                      params: Dict = None, require_auth: bool = True) -> Dict:
        """
        Make API request with error handling

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            require_auth: Whether authentication is required

        Returns:
            Response JSON as dictionary
        """
        self._rate_limit()

        if require_auth:
            self._check_token_expiry()

        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers() if require_auth else {'Content-Type': 'application/json'}
        cookies = self._get_cookies() if require_auth else {}

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers,
                cookies=cookies,
                timeout=30
            )

            # Log request/response for debugging
            logger.debug(f"{method} {url} -> {response.status_code}")

            if response.status_code == 401:
                logger.warning("Unauthorized - attempting token refresh")
                self.refresh_token()
                # Retry request
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=self._get_headers(),
                    cookies=self._get_cookies(),
                    timeout=30
                )

            response.raise_for_status()

            if response.text:
                return response.json()
            return {'status': 'OK'}

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    # ==================== Authentication ====================

    def login(self, email: str, password: str) -> bool:
        """
        Login to Match-Trader platform

        Args:
            email: Account email
            password: Account password

        Returns:
            True if login successful
        """
        endpoint = "/manager/mtr-login"
        data = {
            "email": email,
            "password": password,
            "brokerId": self.broker_id
        }

        try:
            response = self._make_request("POST", endpoint, data, require_auth=False)

            # Extract tokens
            self.trading_api_token = response.get('tradingApiToken')
            self.auth_token = response.get('token')
            self.system_uuid = response.get('systemUuid') or self.system_uuid

            # Token valid for 15 minutes
            self.token_expiry = datetime.now() + timedelta(minutes=15)

            logger.info(f"Login successful - System UUID: {self.system_uuid}")
            return True

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def refresh_token(self) -> bool:
        """
        Refresh authentication token (max 4 times per hour)

        Returns:
            True if refresh successful
        """
        endpoint = "/manager/refresh-token"

        try:
            response = self._make_request("POST", endpoint, require_auth=True)

            self.trading_api_token = response.get('tradingApiToken', self.trading_api_token)
            self.auth_token = response.get('token', self.auth_token)
            self.token_expiry = datetime.now() + timedelta(minutes=15)

            logger.info("Token refreshed successfully")
            return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    # ==================== Account Information ====================

    def get_balance(self) -> Optional[AccountBalance]:
        """
        Get account balance and equity

        Returns:
            AccountBalance object or None on error
        """
        endpoint = f"/mtr-api/{self.system_uuid}/balance"

        try:
            response = self._make_request("GET", endpoint)

            return AccountBalance(
                balance=float(response.get('balance', 0)),
                equity=float(response.get('equity', 0)),
                margin_used=float(response.get('marginUsed', 0)),
                margin_free=float(response.get('marginFree', 0)),
                profit_loss=float(response.get('profitLoss', 0))
            )

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    def get_instruments(self) -> List[Dict]:
        """
        Get available trading instruments

        Returns:
            List of instrument details
        """
        endpoint = f"/mtr-api/{self.system_uuid}/effective-instruments"

        try:
            response = self._make_request("GET", endpoint)
            return response.get('instruments', [])
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return []

    # ==================== Market Data ====================

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current market quote for a symbol

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            Quote dictionary with bid, ask, last price
        """
        endpoint = f"/mtr-api/{self.system_uuid}/quotations"
        params = {"symbols": symbol}

        try:
            response = self._make_request("GET", endpoint, params=params)
            quotes = response.get('quotations', [])

            for quote in quotes:
                if quote.get('symbol') == symbol:
                    return {
                        'symbol': symbol,
                        'bid': float(quote.get('bid', 0)),
                        'ask': float(quote.get('ask', 0)),
                        'last': float(quote.get('last', 0)),
                        'time': quote.get('time')
                    }
            return None

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_candles(self, symbol: str, interval: str = "M15",
                    from_time: str = None, to_time: str = None) -> List[Dict]:
        """
        Get historical candle data

        Args:
            symbol: Trading symbol
            interval: Candle interval (M1, M5, M15, M30, H1, H4, D1, W1)
            from_time: Start time (ISO format)
            to_time: End time (ISO format)

        Returns:
            List of candle data
        """
        endpoint = f"/mtr-api/{self.system_uuid}/candles"
        params = {
            "symbol": symbol,
            "interval": interval
        }
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time

        try:
            response = self._make_request("GET", endpoint, params=params)
            return response.get('candles', [])
        except Exception as e:
            logger.error(f"Failed to get candles: {e}")
            return []

    # ==================== Position Management ====================

    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions

        Returns:
            List of Position objects
        """
        endpoint = f"/mtr-api/{self.system_uuid}/open-positions"

        try:
            response = self._make_request("GET", endpoint)
            positions = []

            for pos in response.get('positions', []):
                positions.append(Position(
                    position_id=str(pos.get('positionId')),
                    instrument=pos.get('instrument'),
                    side=pos.get('side'),
                    volume=float(pos.get('volume', 0)),
                    entry_price=float(pos.get('openPrice', 0)),
                    current_price=float(pos.get('currentPrice', 0)),
                    sl_price=float(pos.get('slPrice', 0)),
                    tp_price=float(pos.get('tpPrice', 0)),
                    profit=float(pos.get('profit', 0)),
                    open_time=pos.get('openTime', '')
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def open_position(self, instrument: str, side: str, volume: float,
                      sl_price: float = 0, tp_price: float = 0) -> Optional[str]:
        """
        Open a new trading position

        Args:
            instrument: Trading symbol (e.g., "EURUSD")
            side: "BUY" or "SELL"
            volume: Lot size (e.g., 0.01)
            sl_price: Stop loss price (0 for none)
            tp_price: Take profit price (0 for none)

        Returns:
            Order ID if successful, None on error
        """
        endpoint = f"/mtr-api/{self.system_uuid}/position/open"
        data = {
            "instrument": instrument,
            "orderSide": side.upper(),
            "volume": volume,
            "slPrice": sl_price,
            "tpPrice": tp_price,
            "isMobile": False
        }

        try:
            response = self._make_request("POST", endpoint, data)
            order_id = response.get('orderId')

            logger.info(f"Position opened: {side} {volume} {instrument} @ market - Order ID: {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return None

    def close_position(self, position_id: str, instrument: str,
                       side: str, volume: float) -> bool:
        """
        Close an existing position

        Args:
            position_id: Position ID to close
            instrument: Trading symbol
            side: Original position side ("BUY" or "SELL")
            volume: Volume to close

        Returns:
            True if successful
        """
        endpoint = f"/mtr-api/{self.system_uuid}/position/close"
        data = {
            "positionId": position_id,
            "instrument": instrument,
            "orderSide": side.upper(),
            "volume": volume
        }

        try:
            response = self._make_request("POST", endpoint, data)
            logger.info(f"Position closed: {position_id} - {side} {volume} {instrument}")
            return response.get('status') == 'OK'

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def partial_close(self, position_id: str, instrument: str,
                      side: str, volume: float) -> bool:
        """
        Partially close a position

        Args:
            position_id: Position ID
            instrument: Trading symbol
            side: Position side
            volume: Volume to close (must be less than total)

        Returns:
            True if successful
        """
        endpoint = f"/mtr-api/{self.system_uuid}/position/partial-close"
        data = {
            "positionId": position_id,
            "instrument": instrument,
            "orderSide": side.upper(),
            "volume": volume
        }

        try:
            response = self._make_request("POST", endpoint, data)
            logger.info(f"Partial close: {position_id} - {volume} lots")
            return response.get('status') == 'OK'

        except Exception as e:
            logger.error(f"Failed to partial close: {e}")
            return False

    def edit_position(self, position_id: str, instrument: str,
                      sl_price: float = None, tp_price: float = None) -> bool:
        """
        Modify stop loss and/or take profit of an existing position

        Args:
            position_id: Position ID to modify
            instrument: Trading symbol
            sl_price: New stop loss price (None to keep current)
            tp_price: New take profit price (None to keep current)

        Returns:
            True if successful
        """
        endpoint = f"/mtr-api/{self.system_uuid}/position/edit"
        data = {
            "positionId": position_id,
            "instrument": instrument
        }

        if sl_price is not None:
            data["slPrice"] = sl_price
        if tp_price is not None:
            data["tpPrice"] = tp_price

        try:
            response = self._make_request("POST", endpoint, data)
            logger.info(f"Position modified: {position_id} - SL: {sl_price}, TP: {tp_price}")
            return response.get('status') == 'OK'

        except Exception as e:
            logger.error(f"Failed to edit position: {e}")
            return False

    # ==================== Order Management ====================

    def create_pending_order(self, instrument: str, side: str, volume: float,
                             price: float, order_type: str = "LIMIT",
                             sl_price: float = 0, tp_price: float = 0) -> Optional[str]:
        """
        Create a pending order (limit or stop)

        Args:
            instrument: Trading symbol
            side: "BUY" or "SELL"
            volume: Lot size
            price: Entry price
            order_type: "LIMIT" or "STOP"
            sl_price: Stop loss price
            tp_price: Take profit price

        Returns:
            Order ID if successful
        """
        endpoint = f"/mtr-api/{self.system_uuid}/order/create-pending-order"
        data = {
            "instrument": instrument,
            "orderSide": side.upper(),
            "volume": volume,
            "price": price,
            "orderType": order_type,
            "slPrice": sl_price,
            "tpPrice": tp_price
        }

        try:
            response = self._make_request("POST", endpoint, data)
            order_id = response.get('orderId')
            logger.info(f"Pending order created: {order_type} {side} {volume} {instrument} @ {price}")
            return order_id

        except Exception as e:
            logger.error(f"Failed to create pending order: {e}")
            return None

    def get_pending_orders(self) -> List[Dict]:
        """
        Get all active pending orders

        Returns:
            List of pending order details
        """
        endpoint = f"/mtr-api/{self.system_uuid}/order/get-active-orders"

        try:
            response = self._make_request("GET", endpoint)
            return response.get('orders', [])
        except Exception as e:
            logger.error(f"Failed to get pending orders: {e}")
            return []

    def cancel_pending_order(self, order_id: str) -> bool:
        """
        Cancel a pending order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        endpoint = f"/mtr-api/{self.system_uuid}/order/cancel-pending-order"
        data = {"orderId": order_id}

        try:
            response = self._make_request("POST", endpoint, data)
            logger.info(f"Pending order cancelled: {order_id}")
            return response.get('status') == 'OK'
        except Exception as e:
            logger.error(f"Failed to cancel pending order: {e}")
            return False

    # ==================== Trade History ====================

    def get_closed_positions(self, from_date: str, to_date: str) -> List[Dict]:
        """
        Get closed position history

        Args:
            from_date: Start date (ISO format)
            to_date: End date (ISO format)

        Returns:
            List of closed position details
        """
        endpoint = f"/mtr-api/{self.system_uuid}/closed-positions"
        data = {
            "from": from_date,
            "to": to_date
        }

        try:
            response = self._make_request("POST", endpoint, data)
            return response.get('positions', [])
        except Exception as e:
            logger.error(f"Failed to get closed positions: {e}")
            return []


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example configuration for FundedNext
    # You'll need to get the actual broker_id and base_url from FundedNext

    config = {
        "base_url": "https://mtr-demo-prod.match-trader.com",  # Demo URL - replace with actual
        "broker_id": "fundednext",  # Replace with actual broker ID
        "email": "your_email@example.com",
        "password": "your_password"
    }

    # Initialize client
    client = MatchTraderClient(
        base_url=config["base_url"],
        broker_id=config["broker_id"]
    )

    # Login
    if client.login(config["email"], config["password"]):
        print("Login successful!")

        # Get account balance
        balance = client.get_balance()
        if balance:
            print(f"\nAccount Balance:")
            print(f"  Balance: ${balance.balance:,.2f}")
            print(f"  Equity: ${balance.equity:,.2f}")
            print(f"  Free Margin: ${balance.margin_free:,.2f}")

        # Get quote
        quote = client.get_quote("EURUSD")
        if quote:
            print(f"\nEURUSD Quote:")
            print(f"  Bid: {quote['bid']}")
            print(f"  Ask: {quote['ask']}")

        # Get open positions
        positions = client.get_open_positions()
        print(f"\nOpen Positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos.side} {pos.volume} {pos.instrument} @ {pos.entry_price} -> P/L: ${pos.profit:.2f}")

        # Example: Open a position (commented out for safety)
        # order_id = client.open_position(
        #     instrument="EURUSD",
        #     side="BUY",
        #     volume=0.01,
        #     sl_price=1.0500,
        #     tp_price=1.0600
        # )

    else:
        print("Login failed!")
