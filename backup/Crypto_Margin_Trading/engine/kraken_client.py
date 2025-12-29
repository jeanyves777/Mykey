"""
Kraken API Client for Margin Trading
=====================================
Handles all API interactions with Kraken REST API for margin trading.

Features:
- Account balance and margin info
- Current prices and OHLC data
- Margin order placement with TP/SL
- Position management
- Trade history

Requires:
- KRAKEN_API_KEY in .env
- KRAKEN_API_SECRET in .env
"""

import os
import time
import hmac
import base64
import hashlib
import urllib.parse
from datetime import datetime
from typing import Optional, Dict, List, Any
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


class KrakenClient:
    """Kraken API client for margin trading"""

    BASE_URL = "https://api.kraken.com"

    # Symbol mapping (standard -> Kraken) - Updated Dec 19, 2025
    SYMBOL_MAP = {
        "BTC/USD": "XXBTZUSD",
        "ETH/USD": "XETHZUSD",
        "SOL/USD": "SOLUSD",
        "XRP/USD": "XXRPZUSD",
        "DOGE/USD": "XDGUSD",
        "LTC/USD": "XLTCZUSD",
        "ADA/USD": "ADAUSD",
        "LINK/USD": "LINKUSD",
        "AVAX/USD": "AVAXUSD",
        "DOT/USD": "DOTUSD",
        "SUI/USD": "SUIUSD",
        "ZEC/USD": "XZECZUSD",
        "BCH/USD": "BCHUSD",
        "PEPE/USD": "PEPEUSD",
        "XLM/USD": "XXLMZUSD",
        "UNI/USD": "UNIUSD",
        "XMR/USD": "XXMRZUSD",
    }

    # Reverse mapping (Kraken -> standard)
    KRAKEN_TO_STANDARD = {v: k for k, v in SYMBOL_MAP.items()}

    # Margin leverage available per pair (Kraken actual limits - Dec 2025)
    LEVERAGE_LIMITS = {
        "XXBTZUSD": 10,   # BTC - 10x (was 5x)
        "XETHZUSD": 10,   # ETH - 10x
        "SOLUSD": 10,     # SOL - 10x
        "XXRPZUSD": 10,   # XRP - 10x
        "XDGUSD": 10,     # DOGE - 10x
        "XLTCZUSD": 10,   # LTC - 10x
        "ADAUSD": 10,     # ADA - 10x
        "LINKUSD": 10,    # LINK - 10x
        "AVAXUSD": 10,    # AVAX - 10x
        "DOTUSD": 3,      # DOT - 3x
        "SUIUSD": 10,     # SUI - 10x
        "XZECZUSD": 3,    # ZEC - 3x
        "BCHUSD": 3,      # BCH - 3x
        "PEPEUSD": 3,     # PEPE - 3x
        "XXLMZUSD": 2,    # XLM - 2x
        "UNIUSD": 3,      # UNI - 3x
        "XXMRZUSD": 2,    # XMR - 2x
    }

    # Price precision (decimal places)
    PRICE_PRECISION = {
        "XXBTZUSD": 1,   # BTC: $0.1
        "XETHZUSD": 2,   # ETH: $0.01
        "SOLUSD": 2,     # SOL: $0.01
        "XXRPZUSD": 5,   # XRP: $0.00001
        "XDGUSD": 7,     # DOGE: $0.0000001
        "XLTCZUSD": 2,   # LTC: $0.01
        "ADAUSD": 6,     # ADA: $0.000001
        "LINKUSD": 3,    # LINK: $0.001
        "AVAXUSD": 2,    # AVAX: $0.01
        "DOTUSD": 4,     # DOT: $0.0001
        "SUIUSD": 4,     # SUI: $0.0001
        "XZECZUSD": 2,   # ZEC: $0.01
        "BCHUSD": 2,     # BCH: $0.01
        "PEPEUSD": 9,    # PEPE: $0.000000001
        "XXLMZUSD": 6,   # XLM: $0.000001
        "UNIUSD": 3,     # UNI: $0.001
        "XXMRZUSD": 2,   # XMR: $0.01
    }

    # Volume precision (decimal places for order volume)
    VOLUME_PRECISION = {
        "XXBTZUSD": 8,   # BTC: 0.00000001
        "XETHZUSD": 8,   # ETH
        "SOLUSD": 8,     # SOL
        "XXRPZUSD": 0,   # XRP: whole units
        "XDGUSD": 0,     # DOGE: whole units
        "XLTCZUSD": 8,   # LTC
        "ADAUSD": 0,     # ADA: whole units
        "LINKUSD": 8,    # LINK
        "AVAXUSD": 8,    # AVAX
        "DOTUSD": 8,     # DOT
    }

    def __init__(self):
        """Initialize Kraken client"""
        self.api_key = os.getenv("KRAKEN_API_KEY")
        self.api_secret = os.getenv("KRAKEN_API_SECRET")

        if not self.api_key or not self.api_secret:
            print("[Kraken] WARNING: API credentials not found in .env")
            print("[Kraken] Public endpoints will work, private endpoints will fail")

        self.session = create_retry_session(retries=3, backoff_factor=1.0)

        print("[Kraken] Client initialized")

    def _get_kraken_signature(self, urlpath: str, data: Dict) -> str:
        """Generate Kraken API signature"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512
        )
        return base64.b64encode(mac.digest()).decode()

    def _public_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make public API request (no auth required)"""
        url = f"{self.BASE_URL}/0/public/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("error") and len(data["error"]) > 0:
                print(f"[Kraken] API Error: {data['error']}")
                return {"error": data["error"]}

            return data.get("result", {})

        except Exception as e:
            print(f"[Kraken] Request error: {e}")
            return {"error": str(e)}

    def _private_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make private API request (auth required)"""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not configured"}

        urlpath = f"/0/private/{endpoint}"
        url = f"{self.BASE_URL}{urlpath}"

        if data is None:
            data = {}

        data["nonce"] = str(int(time.time() * 1000))

        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._get_kraken_signature(urlpath, data)
        }

        try:
            response = self.session.post(url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get("error") and len(result["error"]) > 0:
                print(f"[Kraken] API Error: {result['error']}")
                return {"error": result["error"]}

            return result.get("result", {})

        except Exception as e:
            print(f"[Kraken] Request error: {e}")
            return {"error": str(e)}

    def _get_precision(self, pair: str, price_or_volume: str = "price") -> int:
        """Get decimal precision for a pair"""
        if price_or_volume == "price":
            return self.PRICE_PRECISION.get(pair, 2)
        else:
            return self.VOLUME_PRECISION.get(pair, 8)

    def _round_price(self, price: float, pair: str) -> float:
        """Round price to correct precision"""
        precision = self._get_precision(pair, "price")
        return round(price, precision)

    def _round_volume(self, volume: float, pair: str) -> float:
        """Round volume to correct precision"""
        precision = self._get_precision(pair, "volume")
        return round(volume, precision)

    def to_kraken_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Kraken format"""
        return self.SYMBOL_MAP.get(symbol, symbol)

    def to_standard_symbol(self, kraken_symbol: str) -> str:
        """Convert Kraken symbol to standard format"""
        return self.KRAKEN_TO_STANDARD.get(kraken_symbol, kraken_symbol)

    # ==================== PUBLIC ENDPOINTS ====================

    def get_server_time(self) -> Optional[datetime]:
        """Get Kraken server time"""
        result = self._public_request("Time")
        if "unixtime" in result:
            return datetime.utcfromtimestamp(result["unixtime"])
        return None

    def get_ticker(self, pair: str) -> Optional[Dict]:
        """
        Get current ticker info for a pair

        Args:
            pair: Kraken pair symbol (e.g., "XXBTZUSD")

        Returns:
            Dict with bid, ask, last price, volume
        """
        result = self._public_request("Ticker", {"pair": pair})

        if "error" in result:
            return None

        # Kraken returns nested dict with pair as key
        if pair in result:
            ticker = result[pair]
            return {
                "pair": pair,
                "bid": float(ticker["b"][0]),
                "ask": float(ticker["a"][0]),
                "last": float(ticker["c"][0]),
                "volume_24h": float(ticker["v"][1]),
                "vwap_24h": float(ticker["p"][1]),
                "trades_24h": int(ticker["t"][1]),
                "high_24h": float(ticker["h"][1]),
                "low_24h": float(ticker["l"][1]),
                "open_24h": float(ticker["o"]),
            }

        return None

    def get_current_price(self, pair: str) -> Optional[Dict]:
        """
        Get current bid/ask prices (same as get_ticker but simplified)

        Args:
            pair: Kraken pair symbol

        Returns:
            Dict with bid, ask, spread, mid price
        """
        ticker = self.get_ticker(pair)
        if ticker:
            bid = ticker["bid"]
            ask = ticker["ask"]
            return {
                "pair": pair,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "spread": ask - bid,
                "spread_pct": ((ask - bid) / bid) * 100
            }
        return None

    def get_ohlc(
        self,
        pair: str,
        interval: int = 1,
        since: Optional[int] = None
    ) -> List[Dict]:
        """
        Get OHLC candlestick data

        Args:
            pair: Kraken pair symbol
            interval: Timeframe in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Unix timestamp to get data since

        Returns:
            List of candle dictionaries
        """
        params = {"pair": pair, "interval": interval}
        if since:
            params["since"] = since

        result = self._public_request("OHLC", params)

        if "error" in result:
            return []

        candles = []
        if pair in result:
            for candle in result[pair]:
                candles.append({
                    "timestamp": int(candle[0]),
                    "datetime": datetime.utcfromtimestamp(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "vwap": float(candle[5]),
                    "volume": float(candle[6]),
                    "count": int(candle[7])
                })

        return candles

    # ==================== PRIVATE ENDPOINTS ====================

    def get_balance(self) -> Dict[str, float]:
        """
        Get account balance

        Returns:
            Dict of {currency: balance}
        """
        result = self._private_request("Balance")

        if "error" in result:
            return {}

        balances = {}
        for currency, balance in result.items():
            balances[currency] = float(balance)

        return balances

    def get_trade_balance(self, asset: str = "ZUSD") -> Optional[Dict]:
        """
        Get trade balance (margin info)

        Args:
            asset: Base asset for balance calculation

        Returns:
            Dict with equity, margin, unrealized P/L, etc.
        """
        result = self._private_request("TradeBalance", {"asset": asset})

        if "error" in result:
            return None

        return {
            "equity": float(result.get("e", 0)),           # Total equity
            "trade_balance": float(result.get("tb", 0)),   # Trade balance
            "margin": float(result.get("m", 0)),           # Used margin
            "unrealized_pl": float(result.get("n", 0)),    # Unrealized P/L
            "cost_basis": float(result.get("c", 0)),       # Cost basis
            "floating_valuation": float(result.get("v", 0)), # Floating valuation
            "free_margin": float(result.get("mf", 0)),     # Free margin
            "margin_level": float(result.get("ml", 0)) if result.get("ml") else None,  # Margin level %
        }

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open margin positions

        Returns:
            List of position dictionaries
        """
        result = self._private_request("OpenPositions", {"docalcs": "true"})

        if "error" in result or not result:
            return []

        positions = []
        for pos_id, pos in result.items():
            positions.append({
                "id": pos_id,
                "pair": pos["pair"],
                "type": pos["type"],  # "buy" or "sell"
                "volume": float(pos["vol"]),
                "volume_closed": float(pos["vol_closed"]),
                "cost": float(pos["cost"]),
                "fee": float(pos["fee"]),
                "margin": float(pos["margin"]),
                "value": float(pos.get("value", 0)),
                "unrealized_pl": float(pos.get("net", 0)),
                "open_time": datetime.utcfromtimestamp(float(pos["otime"])),
            })

        return positions

    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders

        Returns:
            List of order dictionaries
        """
        result = self._private_request("OpenOrders")

        if "error" in result:
            return []

        orders = []
        open_orders = result.get("open", {})

        for order_id, order in open_orders.items():
            descr = order.get("descr", {})
            orders.append({
                "id": order_id,
                "pair": descr.get("pair"),
                "type": descr.get("type"),  # "buy" or "sell"
                "ordertype": descr.get("ordertype"),  # "market", "limit", etc.
                "price": float(descr.get("price", 0)),
                "volume": float(order.get("vol", 0)),
                "volume_executed": float(order.get("vol_exec", 0)),
                "status": order.get("status"),
                "open_time": datetime.utcfromtimestamp(float(order["opentm"])),
                "leverage": descr.get("leverage"),
            })

        return orders

    def place_margin_order(
        self,
        pair: str,
        side: str,
        volume: float,
        leverage: int = 2,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: str = "market"
    ) -> Dict:
        """
        Place a margin order

        Args:
            pair: Kraken pair symbol
            side: "buy" or "sell"
            volume: Order volume (in base currency)
            leverage: Leverage ratio (2-5)
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_type: "market" or "limit"

        Returns:
            Order result dict
        """
        # Validate leverage
        max_leverage = self.LEVERAGE_LIMITS.get(pair, 2)
        leverage = min(leverage, max_leverage)

        # Round volume
        volume = self._round_volume(volume, pair)

        data = {
            "pair": pair,
            "type": side.lower(),
            "ordertype": order_type,
            "volume": str(volume),
            "leverage": f"{leverage}:1",
        }

        # Add conditional close orders (TP/SL)
        if stop_loss:
            sl_price = self._round_price(stop_loss, pair)
            data["close[ordertype]"] = "stop-loss"
            data["close[price]"] = str(sl_price)

        # Note: Kraken only allows one close order per position
        # For TP, we'd need to use a limit close order
        # This is a limitation - we'll handle TP manually or use a different approach

        result = self._private_request("AddOrder", data)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "order_ids": result.get("txid", []),
            "description": result.get("descr", {}),
        }

    def place_market_order(
        self,
        pair: str,
        side: str,
        volume: float,
        leverage: int = 2,
        stop_loss: Optional[float] = None
    ) -> Dict:
        """
        Place a market margin order (convenience method)

        Args:
            pair: Kraken pair symbol
            side: "buy" or "sell"
            volume: Order volume
            leverage: Leverage ratio
            stop_loss: Stop loss price

        Returns:
            Order result
        """
        return self.place_margin_order(
            pair=pair,
            side=side,
            volume=volume,
            leverage=leverage,
            stop_loss=stop_loss,
            order_type="market"
        )

    def close_position(self, pair: str, volume: Optional[float] = None) -> Dict:
        """
        Close an open position

        Args:
            pair: Kraken pair symbol
            volume: Volume to close (None = close all)

        Returns:
            Close result
        """
        # Get current position to determine direction
        positions = self.get_open_positions()
        position = None

        for pos in positions:
            if pos["pair"] == pair:
                position = pos
                break

        if not position:
            return {"success": False, "error": f"No open position for {pair}"}

        # Determine close direction (opposite of position)
        close_side = "sell" if position["type"] == "buy" else "buy"
        close_volume = volume or (position["volume"] - position["volume_closed"])

        # Place closing order
        data = {
            "pair": pair,
            "type": close_side,
            "ordertype": "market",
            "volume": str(self._round_volume(close_volume, pair)),
            "leverage": "2:1",  # Use same leverage
            "reduce_only": "true",  # Only reduce position
        }

        result = self._private_request("AddOrder", data)

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {
            "success": True,
            "order_ids": result.get("txid", []),
            "closed_volume": close_volume,
        }

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order"""
        result = self._private_request("CancelOrder", {"txid": order_id})

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {"success": True, "count": result.get("count", 0)}

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        result = self._private_request("CancelAll")

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {"success": True, "count": result.get("count", 0)}

    def get_trades_history(self, count: int = 50) -> List[Dict]:
        """
        Get closed trades history

        Args:
            count: Number of trades to retrieve

        Returns:
            List of trade dictionaries
        """
        result = self._private_request("TradesHistory", {"trades": "true"})

        if "error" in result:
            return []

        trades = []
        trade_data = result.get("trades", {})

        for trade_id, trade in list(trade_data.items())[:count]:
            trades.append({
                "id": trade_id,
                "pair": trade["pair"],
                "type": trade["type"],
                "ordertype": trade["ordertype"],
                "price": float(trade["price"]),
                "volume": float(trade["vol"]),
                "cost": float(trade["cost"]),
                "fee": float(trade["fee"]),
                "margin": float(trade.get("margin", 0)),
                "time": datetime.utcfromtimestamp(float(trade["time"])),
            })

        return trades

    def get_account_info(self) -> Dict:
        """
        Get comprehensive account information

        Returns:
            Dict with balance, margin info, positions
        """
        balance = self.get_balance()
        trade_balance = self.get_trade_balance()
        positions = self.get_open_positions()

        return {
            "balances": balance,
            "usd_balance": balance.get("ZUSD", 0),
            "trade_balance": trade_balance,
            "equity": trade_balance.get("equity", 0) if trade_balance else 0,
            "free_margin": trade_balance.get("free_margin", 0) if trade_balance else 0,
            "unrealized_pl": trade_balance.get("unrealized_pl", 0) if trade_balance else 0,
            "open_positions": len(positions),
            "positions": positions,
        }


if __name__ == "__main__":
    # Test connection
    print("Testing Kraken connection...")
    client = KrakenClient()

    # Test public endpoints (no auth needed)
    print("\n[Public] Server time:")
    server_time = client.get_server_time()
    print(f"  {server_time}")

    print("\n[Public] BTC/USD Ticker:")
    ticker = client.get_ticker("XXBTZUSD")
    if ticker:
        print(f"  Bid: ${ticker['bid']:,.2f}")
        print(f"  Ask: ${ticker['ask']:,.2f}")
        print(f"  Last: ${ticker['last']:,.2f}")
        print(f"  Volume 24h: {ticker['volume_24h']:,.2f} BTC")

    print("\n[Public] ETH/USD Price:")
    price = client.get_current_price("XETHZUSD")
    if price:
        print(f"  Bid: ${price['bid']:,.2f}")
        print(f"  Ask: ${price['ask']:,.2f}")
        print(f"  Spread: ${price['spread']:.2f} ({price['spread_pct']:.3f}%)")

    print("\n[Public] OHLC (last 5 candles):")
    candles = client.get_ohlc("XXBTZUSD", interval=1)[-5:]
    for c in candles:
        print(f"  {c['datetime']}: O={c['open']:.0f} H={c['high']:.0f} L={c['low']:.0f} C={c['close']:.0f}")

    # Test private endpoints (requires API key)
    print("\n[Private] Account Info:")
    account = client.get_account_info()
    if account.get("balances"):
        print(f"  USD Balance: ${account.get('usd_balance', 0):,.2f}")
        print(f"  Equity: ${account.get('equity', 0):,.2f}")
        print(f"  Free Margin: ${account.get('free_margin', 0):,.2f}")
        print(f"  Open Positions: {account.get('open_positions', 0)}")
    else:
        print("  (API credentials not configured)")
