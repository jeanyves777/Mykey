"""
Trade Executor for Forex ML System
==================================

Execute trades through OANDA API.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime
import json


class TradeExecutor:
    """Executes trades through broker API."""

    def __init__(self, api_client, config=None):
        """
        Initialize trade executor.

        Args:
            api_client: OANDA API client
            config: TradingConfig object
        """
        self.api = api_client
        self.config = config
        self.last_error: str = ""

    def _format_price(self, symbol: str, price: float) -> str:
        """Format price correctly for OANDA API.

        JPY pairs use 3 decimal places, all others use 5.
        """
        if 'JPY' in symbol:
            return f"{price:.3f}"
        else:
            return f"{price:.5f}"

    def place_market_order(self, symbol: str, units: float, direction: str,
                           stop_loss: float = None, take_profit: float = None) -> Tuple[bool, Dict]:
        """
        Place a market order.

        Args:
            symbol: Trading pair (e.g., 'EUR_USD')
            units: Number of units (positive)
            direction: 'BUY' or 'SELL'
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Adjust units for direction
            order_units = units if direction == 'BUY' else -units

            # Build order data
            order_data = {
                "order": {
                    "instrument": symbol,
                    "units": str(int(order_units)),
                    "type": "MARKET",
                    "timeInForce": "FOK"
                }
            }

            # Add stop loss (format correctly for JPY pairs)
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": self._format_price(symbol, stop_loss)
                }

            # Add take profit (format correctly for JPY pairs)
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": self._format_price(symbol, take_profit)
                }

            # Place order
            response = self.api.create_order(order_data)

            if response and 'orderFillTransaction' in response:
                fill = response['orderFillTransaction']
                return True, {
                    'trade_id': fill.get('id', ''),
                    'symbol': symbol,
                    'direction': direction,
                    'units': abs(float(fill.get('units', units))),
                    'fill_price': float(fill.get('price', 0)),
                    'time': fill.get('time', datetime.now().isoformat())
                }
            else:
                self.last_error = response.get('errorMessage', 'Unknown error')
                return False, {'error': self.last_error}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def close_trade(self, trade_id: str) -> Tuple[bool, Dict]:
        """
        Close a specific trade.

        Args:
            trade_id: Trade ID to close

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            response = self.api.close_trade(trade_id)

            if response and 'orderFillTransaction' in response:
                fill = response['orderFillTransaction']
                return True, {
                    'trade_id': trade_id,
                    'close_price': float(fill.get('price', 0)),
                    'pnl': float(fill.get('pl', 0)),
                    'time': fill.get('time', datetime.now().isoformat())
                }
            else:
                self.last_error = response.get('errorMessage', 'Unknown error')
                return False, {'error': self.last_error}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def close_position(self, symbol: str, direction: str = None) -> Tuple[bool, Dict]:
        """
        Close all positions for a symbol.

        Args:
            symbol: Trading pair
            direction: 'long', 'short', or None for all

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            close_data = {}
            if direction == 'long' or direction is None:
                close_data['longUnits'] = 'ALL'
            if direction == 'short' or direction is None:
                close_data['shortUnits'] = 'ALL'

            response = self.api.close_position(symbol, close_data)

            if response:
                return True, {
                    'symbol': symbol,
                    'closed': True,
                    'time': datetime.now().isoformat()
                }
            else:
                return False, {'error': 'Failed to close position'}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def modify_trade(self, trade_id: str, stop_loss: float = None,
                     take_profit: float = None, symbol: str = None) -> Tuple[bool, Dict]:
        """
        Modify stop loss or take profit for a trade.

        Args:
            trade_id: Trade ID to modify
            stop_loss: New stop loss price
            take_profit: New take profit price
            symbol: Trading pair (for price formatting, optional)

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            modify_data = {}
            if stop_loss:
                if symbol:
                    modify_data['stopLoss'] = {'price': self._format_price(symbol, stop_loss)}
                else:
                    modify_data['stopLoss'] = {'price': f"{stop_loss:.5f}"}
            if take_profit:
                if symbol:
                    modify_data['takeProfit'] = {'price': self._format_price(symbol, take_profit)}
                else:
                    modify_data['takeProfit'] = {'price': f"{take_profit:.5f}"}

            response = self.api.modify_trade(trade_id, modify_data)

            if response:
                return True, {
                    'trade_id': trade_id,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'modified': True
                }
            else:
                return False, {'error': 'Failed to modify trade'}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def get_open_trades(self) -> Tuple[bool, list]:
        """
        Get all open trades.

        Returns:
            Tuple of (success, list of trades)
        """
        try:
            response = self.api.get_open_trades()

            if response and 'trades' in response:
                return True, response['trades']
            else:
                return True, []

        except Exception as e:
            self.last_error = str(e)
            return False, []

    def get_open_positions(self) -> Tuple[bool, list]:
        """
        Get all open positions.

        Returns:
            Tuple of (success, list of positions)
        """
        try:
            response = self.api.get_open_positions()

            if response and 'positions' in response:
                return True, response['positions']
            else:
                return True, []

        except Exception as e:
            self.last_error = str(e)
            return False, []

    def get_account_info(self) -> Tuple[bool, Dict]:
        """
        Get account information.

        Returns:
            Tuple of (success, account_dict)
        """
        try:
            response = self.api.get_account()

            if response and 'account' in response:
                account = response['account']
                return True, {
                    'balance': float(account.get('balance', 0)),
                    'equity': float(account.get('NAV', 0)),
                    'margin_used': float(account.get('marginUsed', 0)),
                    'margin_available': float(account.get('marginAvailable', 0)),
                    'unrealized_pnl': float(account.get('unrealizedPL', 0)),
                    'open_trade_count': int(account.get('openTradeCount', 0))
                }
            else:
                return False, {}

        except Exception as e:
            self.last_error = str(e)
            return False, {}

    def place_limit_order(self, symbol: str, units: float, direction: str,
                          price: float, stop_loss: float = None,
                          take_profit: float = None) -> Tuple[bool, Dict]:
        """
        Place a limit order (pending order) for DCA.

        Args:
            symbol: Trading pair (e.g., 'EUR_USD')
            units: Number of units (positive)
            direction: 'BUY' or 'SELL'
            price: Limit price at which to execute
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            Tuple of (success, result_dict with order_id)
        """
        try:
            # Adjust units for direction
            order_units = units if direction == 'BUY' else -units

            # Build limit order data (format prices correctly for JPY pairs)
            order_data = {
                "order": {
                    "instrument": symbol,
                    "units": str(int(order_units)),
                    "type": "LIMIT",
                    "price": self._format_price(symbol, price),
                    "timeInForce": "GTC"  # Good Till Cancelled
                }
            }

            # Add stop loss (format correctly for JPY pairs)
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": self._format_price(symbol, stop_loss)
                }

            # Add take profit (format correctly for JPY pairs)
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": self._format_price(symbol, take_profit)
                }

            # Place order
            response = self.api.create_order(order_data)

            if response and 'orderCreateTransaction' in response:
                order = response['orderCreateTransaction']
                return True, {
                    'order_id': order.get('id', ''),
                    'symbol': symbol,
                    'direction': direction,
                    'units': abs(float(order.get('units', units))),
                    'price': float(order.get('price', price)),
                    'type': 'LIMIT',
                    'time': order.get('time', datetime.now().isoformat())
                }
            elif response and 'orderFillTransaction' in response:
                # Order was filled immediately (price already reached)
                fill = response['orderFillTransaction']
                return True, {
                    'order_id': fill.get('id', ''),
                    'trade_id': fill.get('id', ''),
                    'symbol': symbol,
                    'direction': direction,
                    'units': abs(float(fill.get('units', units))),
                    'fill_price': float(fill.get('price', price)),
                    'type': 'LIMIT_FILLED',
                    'time': fill.get('time', datetime.now().isoformat())
                }
            else:
                self.last_error = response.get('errorMessage', 'Unknown error')
                return False, {'error': self.last_error}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def cancel_order(self, order_id: str) -> Tuple[bool, Dict]:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            import requests
            url = f'{self.api.base_url}/v3/accounts/{self.api.account_id}/orders/{order_id}/cancel'
            response = requests.put(url, headers=self.api.headers)
            response.raise_for_status()
            result = response.json()

            if result and 'orderCancelTransaction' in result:
                return True, {
                    'order_id': order_id,
                    'cancelled': True,
                    'time': result['orderCancelTransaction'].get('time', datetime.now().isoformat())
                }
            else:
                self.last_error = result.get('errorMessage', 'Failed to cancel order')
                return False, {'error': self.last_error}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def get_pending_orders(self, symbol: str = None) -> Tuple[bool, list]:
        """
        Get all pending orders, optionally filtered by symbol.

        Args:
            symbol: Filter by trading pair (optional)

        Returns:
            Tuple of (success, list of pending orders)
        """
        try:
            import requests
            url = f'{self.api.base_url}/v3/accounts/{self.api.account_id}/pendingOrders'
            response = requests.get(url, headers=self.api.headers)
            response.raise_for_status()
            result = response.json()

            if result and 'orders' in result:
                orders = result['orders']
                if symbol:
                    orders = [o for o in orders if o.get('instrument') == symbol]
                return True, orders
            else:
                return True, []

        except Exception as e:
            self.last_error = str(e)
            return False, []

    def cancel_all_orders_for_symbol(self, symbol: str) -> Tuple[int, int]:
        """
        Cancel all pending orders for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Tuple of (cancelled_count, failed_count)
        """
        success, orders = self.get_pending_orders(symbol)
        if not success or not orders:
            return 0, 0

        cancelled = 0
        failed = 0

        for order in orders:
            order_id = order.get('id', '')
            if order_id:
                ok, _ = self.cancel_order(order_id)
                if ok:
                    cancelled += 1
                else:
                    failed += 1

        return cancelled, failed

    def set_trailing_stop(self, trade_id: str, distance_pips: float, symbol: str) -> Tuple[bool, Dict]:
        """
        Set an actual OANDA Trailing Stop order for a trade.

        Args:
            trade_id: Trade ID to add trailing stop to
            distance_pips: Trailing stop distance in pips (minimum 5 for OANDA)
            symbol: Trading pair for pip calculation

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            import requests

            # Convert pips to price distance
            # JPY pairs have 2 decimal places (1 pip = 0.01)
            # Other pairs have 4-5 decimal places (1 pip = 0.0001)
            if 'JPY' in symbol:
                distance = distance_pips * 0.01
            else:
                distance = distance_pips * 0.0001

            # OANDA requires minimum 5 pips trailing stop
            min_pips = 5.0
            if distance_pips < min_pips:
                distance_pips = min_pips
                if 'JPY' in symbol:
                    distance = min_pips * 0.01
                else:
                    distance = min_pips * 0.0001

            # Build trailing stop order data (format distance correctly for JPY pairs)
            if 'JPY' in symbol:
                distance_str = f"{distance:.3f}"
            else:
                distance_str = f"{distance:.5f}"

            modify_data = {
                'trailingStopLoss': {
                    'distance': distance_str
                }
            }

            url = f'{self.api.base_url}/v3/accounts/{self.api.account_id}/trades/{trade_id}/orders'
            response = requests.put(url, headers=self.api.headers, json=modify_data)
            response.raise_for_status()
            result = response.json()

            if result:
                return True, {
                    'trade_id': trade_id,
                    'trailing_distance_pips': distance_pips,
                    'trailing_distance_price': distance,
                    'trailing_stop_set': True,
                    'time': datetime.now().isoformat()
                }
            else:
                self.last_error = 'Failed to set trailing stop'
                return False, {'error': self.last_error}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}

    def get_trade_details(self, trade_id: str) -> Tuple[bool, Dict]:
        """
        Get details for a specific trade including trailing stop info.

        Args:
            trade_id: Trade ID to query

        Returns:
            Tuple of (success, trade_dict)
        """
        try:
            import requests
            url = f'{self.api.base_url}/v3/accounts/{self.api.account_id}/trades/{trade_id}'
            response = requests.get(url, headers=self.api.headers)
            response.raise_for_status()
            result = response.json()

            if result and 'trade' in result:
                return True, result['trade']
            else:
                return False, {}

        except Exception as e:
            self.last_error = str(e)
            return False, {'error': str(e)}
