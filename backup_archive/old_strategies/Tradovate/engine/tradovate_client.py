"""
Tradovate REST + WebSocket Client
Handles authentication, market data, orders, and positions
"""

import requests
import json
import time
import websocket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


class TradovateClient:
    """
    Tradovate API Client

    REST API: https://api.tradovate.com (live) or https://demo.tradovate.com (demo)
    WebSocket: wss://md.tradovate.com/v1/websocket (live) or wss://md-demo.tradovate.com/v1/websocket (demo)

    Documentation: https://api.tradovate.com/v1/docs
    """

    def __init__(self, username: str, password: str, api_key: str, is_demo: bool = True):
        """
        Initialize Tradovate client

        Args:
            username: Tradovate username
            password: Tradovate password
            api_key: Tradovate API key (from Tradovate settings)
            is_demo: True for demo account, False for live
        """
        self.username = username
        self.password = password
        self.api_key = api_key
        self.is_demo = is_demo

        # API endpoints (from official OpenAPI spec)
        if is_demo:
            self.base_url = "https://demo.tradovateapi.com/v1"
            self.ws_url = "wss://md-demo.tradovateapi.com/v1/websocket"
        else:
            self.base_url = "https://live.tradovateapi.com/v1"
            self.ws_url = "wss://md.tradovateapi.com/v1/websocket"

        # Session
        self.access_token = None
        self.account_id = None
        self.session = requests.Session()

        # WebSocket
        self.ws = None
        self.ws_connected = False
        self.market_data = {}

        # Authenticate
        self._authenticate()
        self._get_account()

    def _authenticate(self):
        """Authenticate with Tradovate and get access token"""
        url = f"{self.base_url}/auth/accesstokenrequest"
        payload = {
            "name": self.username,
            "password": self.password,
            "appId": "TradingSystem",
            "appVersion": "1.0",
            "cid": 0,
            "sec": self.api_key  # API secret/key goes here
        }

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get('accessToken')
            print(f"[TRADOVATE] Authenticated successfully ({'DEMO' if self.is_demo else 'LIVE'})")
        else:
            raise Exception(f"Authentication failed: {response.text}")

    def _get_account(self):
        """Get account ID"""
        url = f"{self.base_url}/account/list"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        response = self.session.get(url, headers=headers)

        if response.status_code == 200:
            accounts = response.json()
            if accounts:
                self.account_id = accounts[0]['id']
                print(f"[TRADOVATE] Account ID: {self.account_id}")
        else:
            raise Exception(f"Failed to get account: {response.text}")

    # ==================== ACCOUNT INFO ====================

    def get_account_balance(self) -> Dict:
        """Get account balance and margin info"""
        url = f"{self.base_url}/account/item"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"id": self.account_id}

        response = self.session.get(url, headers=headers, params=params)

        if response.status_code == 200:
            account = response.json()
            return {
                'balance': account.get('cashBalance', 0),
                'equity': account.get('netLiquidatingValue', 0),
                'margin_used': account.get('marginUsed', 0),
                'margin_available': account.get('marginAvailable', 0),
                'unrealized_pnl': account.get('unrealizedPnL', 0),
                'realized_pnl': account.get('realizedPnL', 0)
            }
        else:
            print(f"[TRADOVATE] Error getting account balance: {response.text}")
            return {}

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        url = f"{self.base_url}/position/list"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        response = self.session.get(url, headers=headers)

        if response.status_code == 200:
            positions = response.json()
            return [{
                'symbol': p.get('contractId'),  # Need to map contractId to symbol
                'side': 'long' if p.get('netPos', 0) > 0 else 'short',
                'quantity': abs(p.get('netPos', 0)),
                'entry_price': p.get('avgPrice', 0),
                'unrealized_pnl': p.get('unrealizedPnL', 0)
            } for p in positions if p.get('netPos', 0) != 0]
        else:
            print(f"[TRADOVATE] Error getting positions: {response.text}")
            return []

    # ==================== MARKET DATA ====================

    def get_contract_id(self, symbol: str) -> Optional[int]:
        """
        Get contract ID for a symbol

        Args:
            symbol: Symbol name (e.g., 'M6EU2' for Micro Euro Dec 2025)

        Returns:
            Contract ID (integer)
        """
        url = f"{self.base_url}/contract/find"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"name": symbol}

        response = self.session.get(url, headers=headers, params=params)

        if response.status_code == 200:
            contract = response.json()
            return contract.get('id')
        else:
            print(f"[TRADOVATE] Error finding contract {symbol}: {response.text}")
            return None

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = 'M15',
        bars: int = 500
    ) -> pd.DataFrame:
        """
        Get historical bars for backtesting

        Args:
            symbol: Symbol name (e.g., 'M6EU2')
            timeframe: Timeframe ('M1', 'M5', 'M15', 'H1', 'D1')
            bars: Number of bars to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        contract_id = self.get_contract_id(symbol)
        if not contract_id:
            return pd.DataFrame()

        # Map timeframe to Tradovate format
        tf_map = {
            'M1': {'unit': 'Minute', 'size': 1},
            'M5': {'unit': 'Minute', 'size': 5},
            'M15': {'unit': 'Minute', 'size': 15},
            'H1': {'unit': 'Hour', 'size': 1},
            'D1': {'unit': 'Day', 'size': 1}
        }

        tf = tf_map.get(timeframe, {'unit': 'Minute', 'size': 15})

        url = f"{self.base_url}/chart/getBars"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {
            'symbol': symbol,
            'chartDescription': {
                'underlyingType': 'MinuteBar',
                'elementSize': tf['size'],
                'elementSizeUnit': tf['unit'],
                'withHistogram': False
            },
            'timeRange': {
                'asMuchAsElements': bars
            }
        }

        response = self.session.post(url, headers=headers, json=params)

        if response.status_code == 200:
            data = response.json()
            bars_data = data.get('bars', [])

            if not bars_data:
                return pd.DataFrame()

            df = pd.DataFrame(bars_data)
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('time', inplace=True)

            return df
        else:
            print(f"[TRADOVATE] Error getting historical bars: {response.text}")
            return pd.DataFrame()

    # ==================== ORDER MANAGEMENT ====================

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place a market order

        Args:
            symbol: Symbol name (e.g., 'M6EU2')
            side: 'buy' or 'sell'
            quantity: Number of contracts
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Order details
        """
        contract_id = self.get_contract_id(symbol)
        if not contract_id:
            return None

        url = f"{self.base_url}/order/placeorder"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        order = {
            "accountId": self.account_id,
            "accountSpec": self.username,
            "action": "Buy" if side.lower() == 'buy' else "Sell",
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": "Market",
            "isAutomated": True
        }

        response = self.session.post(url, headers=headers, json=order)

        if response.status_code == 200:
            order_response = response.json()
            print(f"[TRADOVATE] Order placed: {side.upper()} {quantity} {symbol}")

            # Place stop loss and take profit as separate orders if provided
            if stop_loss:
                self._place_stop_loss(symbol, quantity, stop_loss, side)
            if take_profit:
                self._place_take_profit(symbol, quantity, take_profit, side)

            return order_response
        else:
            print(f"[TRADOVATE] Error placing order: {response.text}")
            return None

    def _place_stop_loss(self, symbol: str, quantity: int, price: float, original_side: str):
        """Place stop loss order"""
        url = f"{self.base_url}/order/placeorder"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        # Stop loss is opposite side of entry
        side = "Sell" if original_side.lower() == 'buy' else "Buy"

        order = {
            "accountId": self.account_id,
            "accountSpec": self.username,
            "action": side,
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": "Stop",
            "stopPrice": price,
            "isAutomated": True
        }

        response = self.session.post(url, headers=headers, json=order)

        if response.status_code == 200:
            print(f"[TRADOVATE] Stop loss placed at {price}")
        else:
            print(f"[TRADOVATE] Error placing stop loss: {response.text}")

    def _place_take_profit(self, symbol: str, quantity: int, price: float, original_side: str):
        """Place take profit order"""
        url = f"{self.base_url}/order/placeorder"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        # Take profit is opposite side of entry
        side = "Sell" if original_side.lower() == 'buy' else "Buy"

        order = {
            "accountId": self.account_id,
            "accountSpec": self.username,
            "action": side,
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": "Limit",
            "price": price,
            "isAutomated": True
        }

        response = self.session.post(url, headers=headers, json=order)

        if response.status_code == 200:
            print(f"[TRADOVATE] Take profit placed at {price}")
        else:
            print(f"[TRADOVATE] Error placing take profit: {response.text}")

    def close_position(self, symbol: str, quantity: int, side: str):
        """Close a position"""
        # Close is opposite side of position
        close_side = 'sell' if side.lower() == 'buy' else 'buy'
        return self.place_market_order(symbol, close_side, quantity)

    def close_all_positions(self):
        """Close all open positions"""
        positions = self.get_positions()
        for pos in positions:
            self.close_position(
                pos['symbol'],
                pos['quantity'],
                pos['side']
            )

    # ==================== WEBSOCKET (for real-time data) ====================

    def connect_websocket(self):
        """Connect to Tradovate WebSocket for real-time market data"""
        def on_message(ws, message):
            data = json.loads(message)
            # Handle real-time quotes
            if data.get('e') == 'md':  # Market data update
                symbol = data.get('s')
                self.market_data[symbol] = {
                    'bid': data.get('b'),
                    'ask': data.get('a'),
                    'last': data.get('l'),
                    'time': datetime.now()
                }

        def on_open(ws):
            self.ws_connected = True
            print("[TRADOVATE] WebSocket connected")
            # Authorize WebSocket
            ws.send(json.dumps({
                "op": "authorize",
                "args": [self.access_token]
            }))

        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
            print("[TRADOVATE] WebSocket disconnected")

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_open=on_open,
            on_close=on_close
        )

        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

    def subscribe_market_data(self, symbol: str):
        """Subscribe to real-time market data for a symbol"""
        if self.ws and self.ws_connected:
            self.ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"md/{symbol}"]
            }))
            print(f"[TRADOVATE] Subscribed to {symbol}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price from WebSocket data"""
        if symbol in self.market_data:
            return self.market_data[symbol].get('last')
        return None
