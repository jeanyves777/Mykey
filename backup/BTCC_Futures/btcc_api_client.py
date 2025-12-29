"""
BTCC Futures API Client
=======================
REST API client for BTCC.com Futures trading platform.

Features:
- MD5 signature authentication
- Session management with token
- Heartbeat mechanism (every 20 seconds)
- All trading endpoints (open/close positions, pending orders, etc.)
"""

import hashlib
import requests
import threading
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BTCCAPIClient:
    """BTCC Futures REST API Client with MD5 Authentication"""

    BASE_URL = "https://api1.btloginc.com:9081"
    HEARTBEAT_INTERVAL = 20  # seconds

    def __init__(self, api_key: str, secret_key: str, user_name: str = None,
                 password: str = None, company_id: int = 1):
        """
        Initialize BTCC API Client.

        Args:
            api_key: API access key
            secret_key: Secret key for signature
            user_name: Email or mobile number for login
            password: Account password
            company_id: Company ID (default: 1)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.user_name = user_name
        self.password = password
        self.company_id = company_id

        self.token = None
        self.account_id = None
        self.account_no = None
        self.account_group_id = None
        self.user_id = None

        self._heartbeat_thread = None
        self._heartbeat_running = False

        self.session = requests.Session()
        self.session.verify = True

    def _generate_signature(self, params: Dict) -> str:
        """
        Generate MD5 signature for API request.

        Steps:
        1. Sort parameters by key in ASCII order
        2. Add secret_key to sorted params
        3. Generate MD5 hash
        """
        # Remove 'sign' if present
        params_copy = {k: v for k, v in params.items() if k != 'sign' and v is not None}

        # Add secret_key
        params_copy['secret_key'] = self.secret_key

        # Sort by key in ASCII order
        sorted_params = sorted(params_copy.items(), key=lambda x: x[0])

        # Build query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])

        # Generate MD5 hash
        signature = hashlib.md5(query_string.encode('utf-8')).hexdigest()

        return signature

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated API request."""
        url = f"{self.BASE_URL}{endpoint}"

        if params is None:
            params = {}

        # Add signature
        params['sign'] = self._generate_signature(params)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            else:
                response = self.session.post(url, params=params, timeout=30)

            response.raise_for_status()
            data = response.json()

            if data.get('code', -1) != 0:
                logger.error(f"API Error: {data.get('msg', 'Unknown error')}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {'code': -1, 'msg': str(e)}

    # ==================== Authentication ====================

    def login(self, user_name: str = None, password: str = None) -> Dict:
        """
        Login to BTCC and get session token.

        Args:
            user_name: Email or mobile number
            password: Account password

        Returns:
            Login response with token and account info
        """
        user_name = user_name or self.user_name
        password = password or self.password

        if not user_name or not password:
            return {'code': -1, 'msg': 'Username and password required'}

        params = {
            'user_name': user_name,
            'password': password,
            'company_id': self.company_id,
            'api_key': self.api_key,
        }

        result = self._make_request('POST', '/v1/user/login', params)

        if result.get('code') == 0:
            self.token = result.get('token')
            account = result.get('account', {})
            self.account_id = account.get('id')
            self.account_no = account.get('account_no')
            self.account_group_id = account.get('account_groupid')
            self.user_id = account.get('userid')

            logger.info(f"Login successful. Account: {self.account_no}, Balance: {account.get('balance')}")

            # Start heartbeat
            self._start_heartbeat()

        return result

    def keepalive(self) -> Dict:
        """Send heartbeat to maintain session."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
        }

        return self._make_request('GET', '/v1/user/keepalive', params)

    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while self._heartbeat_running:
            try:
                result = self.keepalive()
                if result.get('code') != 0:
                    logger.warning(f"Heartbeat failed: {result.get('msg')}")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            time.sleep(self.HEARTBEAT_INTERVAL)

    def _start_heartbeat(self):
        """Start background heartbeat thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Heartbeat started (every 20 seconds)")

    def _stop_heartbeat(self):
        """Stop background heartbeat thread."""
        self._heartbeat_running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

    # ==================== Account Info ====================

    def get_account(self) -> Dict:
        """Get account information."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
        }

        return self._make_request('GET', '/v1/account/account', params)

    def get_symbols(self, account_group_id: int = None) -> Dict:
        """Get list of tradable products/symbols."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
        }

        if account_group_id:
            params['account_group_id'] = account_group_id

        return self._make_request('GET', '/v1/config/symbollist', params)

    def get_account_group(self, account_group_id: int = None) -> Dict:
        """Get account group information."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'account_groupid': account_group_id or self.account_group_id,
        }

        return self._make_request('GET', '/v1/config/accountgroup', params)

    # ==================== Position Management ====================

    def get_positions(self, symbol: str = None, position_id: int = None,
                      page_no: int = 1, page_size: int = 100) -> Dict:
        """
        Get list of positions.

        Args:
            symbol: Filter by symbol name
            position_id: Filter by position ID
            page_no: Page number
            page_size: Items per page
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'pageNo': page_no,
            'pageSize': page_size,
        }

        if symbol:
            params['symbol'] = symbol
        if position_id:
            params['id'] = position_id

        return self._make_request('GET', '/v1/account/positionlist', params)

    def open_position(self, symbol: str, direction: int, volume: float, price: float,
                      leverage: int, stop_loss: float = None, take_profit: float = None,
                      refid: int = None) -> Dict:
        """
        Open a new position.

        Args:
            symbol: Product/symbol name
            direction: 1=Buy, 2=Sell
            volume: Position size
            price: Entry price
            leverage: Leverage multiplier
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            refid: Client reference ID (optional)

        Returns:
            Position info if successful
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'direction': direction,
            'symbol': symbol,
            'request_volume': volume,
            'request_price': price,
            'multiple': leverage,
        }

        if stop_loss:
            params['stop_loss'] = stop_loss
        if take_profit:
            params['take_profit'] = take_profit
        if refid:
            params['refid'] = refid

        result = self._make_request('POST', '/v1/account/openposition', params)

        if result.get('code') == 0:
            position = result.get('position', {})
            logger.info(f"Position opened: {symbol} {'BUY' if direction == 1 else 'SELL'} "
                       f"x{volume} @ {price}, ID: {position.get('id')}")

        return result

    def close_position(self, position_id: int, symbol: str, direction: int,
                       volume: float, price: float, refid: int = None) -> Dict:
        """
        Close a position at market price.

        Args:
            position_id: Position ID to close
            symbol: Product/symbol name
            direction: 1=Buy, 2=Sell (same as position)
            volume: Volume to close
            price: Close price
            refid: Client reference ID (optional)
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'positionid': position_id,
            'direction': direction,
            'symbol': symbol,
            'request_volume': volume,
            'request_price': price,
        }

        if refid:
            params['refid'] = refid

        result = self._make_request('POST', '/v1/account/closeposition', params)

        if result.get('code') == 0:
            position = result.get('position', {})
            logger.info(f"Position closed: ID {position_id}, P/L: {position.get('exec_profit')}")

        return result

    def update_position(self, position_id: int, symbol: str,
                        stop_loss: float, take_profit: float) -> Dict:
        """
        Update position stop loss and take profit.

        Args:
            position_id: Position ID
            symbol: Product/symbol name
            stop_loss: New stop loss price
            take_profit: New take profit price
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'id': position_id,
            'symbol': symbol,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
        }

        return self._make_request('POST', '/v1/account/updateposition', params)

    # ==================== Pending Orders ====================

    def create_pending_order(self, symbol: str, direction: int, order_type: int,
                             volume: float, price: float, leverage: int,
                             stop_loss: float = None, take_profit: float = None,
                             position_id: int = None, refid: int = None) -> Dict:
        """
        Create a pending order.

        Args:
            symbol: Product/symbol name
            direction: 1=Buy, 2=Sell
            order_type: 1=LIMIT open, 2=STOP open, 3=Limit close, 4=Market SL, 5=Market TP
            volume: Order volume
            price: Order price
            leverage: Leverage multiplier
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            position_id: Required for type 3, 4, 5
            refid: Client reference ID (optional)
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'direction': direction,
            'type': order_type,
            'symbol': symbol,
            'request_volume': volume,
            'request_price': price,
            'multiple': leverage,
        }

        if stop_loss:
            params['stop_loss'] = stop_loss
        if take_profit:
            params['take_profit'] = take_profit
        if position_id:
            params['positionid'] = position_id
        if refid:
            params['refid'] = refid

        return self._make_request('POST', '/v1/account/openpending', params)

    def create_sltp_order(self, symbol: str, position_id: int, volume: float,
                          leverage: int, stop_loss: float = None,
                          take_profit: float = None) -> Dict:
        """
        Create stop loss / take profit order for existing position.

        Args:
            symbol: Product/symbol name
            position_id: Position ID
            volume: Volume
            leverage: Leverage of original position
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'symbol': symbol,
            'positionid': position_id,
            'request_volume': volume,
            'multiple': leverage,
        }

        if stop_loss:
            params['stop_loss'] = stop_loss
        if take_profit:
            params['take_profit'] = take_profit

        return self._make_request('POST', '/v1/account/opensltp', params)

    def update_pending_order(self, order_id: int, direction: int, order_type: int,
                             volume: float, price: float, symbol: str = None,
                             stop_loss: float = None, take_profit: float = None) -> Dict:
        """Update an existing pending order."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'id': order_id,
            'direction': direction,
            'type': order_type,
            'request_volume': volume,
            'request_price': price,
        }

        if symbol:
            params['symbol'] = symbol
        if stop_loss:
            params['stop_loss'] = stop_loss
        if take_profit:
            params['take_profit'] = take_profit

        return self._make_request('POST', '/v1/account/updatepending', params)

    def cancel_pending_order(self, order_id: int) -> Dict:
        """Cancel a pending order."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
            'id': order_id,
        }

        return self._make_request('POST', '/v1/account/cancelpending', params)

    def get_pending_orders(self, order_id: int = None) -> Dict:
        """Get list of pending orders."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
        }

        if order_id:
            params['id'] = order_id

        return self._make_request('GET', '/v1/account/orderList', params)

    # ==================== Trade History ====================

    def get_today_trades(self, execution_id: int = None) -> Dict:
        """Get today's executed trades."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
        }

        if execution_id:
            params['id'] = execution_id

        return self._make_request('GET', '/v1/account/dealTodayList', params)

    def get_today_pnl(self, execution_id: int = None) -> Dict:
        """Get today's P/L list."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'accountid': self.account_id,
        }

        if execution_id:
            params['id'] = execution_id

        return self._make_request('GET', '/v1/account/profitTodayList', params)

    def get_profit_history(self, page_no: int = 1, page_size: int = 100,
                           begin_time: int = None, end_time: int = None,
                           symbol: str = None, direction: int = None) -> Dict:
        """Get historical P/L report."""
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'pageNo': page_no,
            'pageSize': page_size,
        }

        if begin_time:
            params['beginTime'] = begin_time
        if end_time:
            params['endTime'] = end_time
        if symbol:
            params['symbol'] = symbol
        if direction:
            params['openDirection'] = direction

        return self._make_request('GET', '/v1/user/getProfitReport', params)

    def get_trade_history(self, page_no: int = 1, page_size: int = 100,
                          begin_time: int = None, end_time: int = None,
                          symbol: str = None, direction: int = None,
                          report_type: str = None) -> Dict:
        """
        Get transaction history.

        Args:
            report_type: 'OPEN' for opens, 'CLOSE' for closes
        """
        if not self.token:
            return {'code': -1, 'msg': 'Not logged in'}

        params = {
            'token': self.token,
            'pageNo': page_no,
            'pageSize': page_size,
        }

        if begin_time:
            params['beginTime'] = begin_time
        if end_time:
            params['endTime'] = end_time
        if symbol:
            params['symbol'] = symbol
        if direction:
            params['direction'] = direction
        if report_type:
            params['reportType'] = report_type

        return self._make_request('GET', '/v1/user/getTradeReport', params)

    # ==================== Utility Methods ====================

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions (status = 1)."""
        result = self.get_positions(page_size=100)

        if result.get('code') != 0:
            return []

        positions = result.get('positions', [])
        return [p for p in positions if p.get('status') == 1]

    def close_all_positions(self) -> List[Dict]:
        """Close all open positions."""
        positions = self.get_open_positions()
        results = []

        for pos in positions:
            result = self.close_position(
                position_id=pos['id'],
                symbol=pos['symbol'],
                direction=pos['direction'],
                volume=pos['volume'],
                price=pos.get('close_price', pos['open_price'])
            )
            results.append(result)

        return results

    def get_balance(self) -> float:
        """Get current account balance."""
        result = self.get_account()

        if result.get('code') == 0:
            return result.get('account', {}).get('balance', 0)
        return 0

    def get_equity(self) -> float:
        """Get current account equity."""
        result = self.get_account()

        if result.get('code') == 0:
            return result.get('account', {}).get('equity', 0)
        return 0

    def get_free_margin(self) -> float:
        """Get available margin."""
        result = self.get_account()

        if result.get('code') == 0:
            return result.get('account', {}).get('free_margin', 0)
        return 0

    def disconnect(self):
        """Disconnect and stop heartbeat."""
        self._stop_heartbeat()
        self.token = None
        logger.info("Disconnected from BTCC")

    def __del__(self):
        """Cleanup on destruction."""
        self._stop_heartbeat()
