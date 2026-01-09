#!/usr/bin/env python3
"""
BTCC Data Fetcher
=================
Fetch real-time and historical price data for BTCC Futures.

Since BTCC doesn't have a public market data API, we use Binance
for price data (prices are very similar for same pairs).
"""

import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BTCCDataFetcher:
    """Fetch price data for BTCC trading"""

    # Use Binance.US for US-based data (or Binance.com for others)
    BINANCE_BASE_URL = "https://api.binance.us"
    BINANCE_COM_URL = "https://api.binance.com"

    # Symbol mapping from BTCC to Binance format
    SYMBOL_MAP = {
        'BTCUSDT150x': 'BTCUSDT',
        'BTCUSDT100x': 'BTCUSDT',
        'BTCUSDT50x': 'BTCUSDT',
        'BTCUSDT': 'BTCUSDT',
        'ETHUSDT100x': 'ETHUSDT',
        'ETHUSDT50x': 'ETHUSDT',
        'ETHUSDT': 'ETHUSDT',
        'SOLUSDT50x': 'SOLUSDT',
        'SOLUSDT': 'SOLUSDT',
        'XRPUSDT50x': 'XRPUSDT',
        'XRPUSDT': 'XRPUSDT',
        'DOGEUSDT50x': 'DOGEUSDT',
        'DOGEUSDT': 'DOGEUSDT',
        'LTCUSDT': 'LTCUSDT',
        'LINKUSDT': 'LINKUSDT',
        'AVAXUSDT': 'AVAXUSDT',
        'ADAUSDT': 'ADAUSDT',
    }

    def __init__(self, use_binance_com: bool = False):
        """
        Initialize data fetcher.

        Args:
            use_binance_com: Use Binance.com (True) or Binance.US (False)
                             Default is False (Binance.US) to avoid geo-blocking
        """
        self.base_url = self.BINANCE_COM_URL if use_binance_com else self.BINANCE_BASE_URL
        self.session = requests.Session()
        self._price_cache: Dict[str, tuple] = {}  # symbol -> (price, timestamp)
        self._cache_ttl = 1.0  # Cache TTL in seconds

    def _get_binance_symbol(self, btcc_symbol: str) -> str:
        """Convert BTCC symbol to Binance format."""
        return self.SYMBOL_MAP.get(btcc_symbol, btcc_symbol.replace('x', '').replace('150', '').replace('100', '').replace('50', ''))

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: BTCC symbol name

        Returns:
            Current price or None if failed
        """
        # Check cache
        if symbol in self._price_cache:
            price, timestamp = self._price_cache[symbol]
            if time.time() - timestamp < self._cache_ttl:
                return price

        binance_symbol = self._get_binance_symbol(symbol)

        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = self.session.get(url, params={'symbol': binance_symbol}, timeout=5)
            response.raise_for_status()
            data = response.json()

            price = float(data.get('price', 0))
            self._price_cache[symbol] = (price, time.time())

            return price

        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            return None

    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of BTCC symbol names

        Returns:
            Dict of symbol -> price
        """
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Create lookup dict
            binance_prices = {item['symbol']: float(item['price']) for item in data}

            # Map to BTCC symbols
            result = {}
            for symbol in symbols:
                binance_symbol = self._get_binance_symbol(symbol)
                if binance_symbol in binance_prices:
                    result[symbol] = binance_prices[binance_symbol]
                    self._price_cache[symbol] = (binance_prices[binance_symbol], time.time())

            return result

        except Exception as e:
            logger.warning(f"Failed to get multiple prices: {e}")
            return {}

    def get_klines(self, symbol: str, interval: str = '1m',
                   limit: int = 100) -> List[OHLCV]:
        """
        Get candlestick data.

        Args:
            symbol: BTCC symbol name
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch (max 1000)

        Returns:
            List of OHLCV objects
        """
        binance_symbol = self._get_binance_symbol(symbol)

        try:
            url = f"{self.base_url}/api/v3/klines"
            response = self.session.get(url, params={
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit,
            }, timeout=10)
            response.raise_for_status()
            data = response.json()

            candles = []
            for kline in data:
                candles.append(OHLCV(
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                ))

            return candles

        except Exception as e:
            logger.warning(f"Failed to get klines for {symbol}: {e}")
            return []

    def get_historical_klines(self, symbol: str, interval: str,
                              start_time: datetime, end_time: datetime = None) -> List[OHLCV]:
        """
        Get historical candlestick data.

        Args:
            symbol: BTCC symbol name
            interval: Candle interval
            start_time: Start datetime
            end_time: End datetime (default: now)

        Returns:
            List of OHLCV objects
        """
        binance_symbol = self._get_binance_symbol(symbol)
        end_time = end_time or datetime.now()

        all_candles = []
        current_start = start_time

        while current_start < end_time:
            try:
                url = f"{self.base_url}/api/v3/klines"
                response = self.session.get(url, params={
                    'symbol': binance_symbol,
                    'interval': interval,
                    'startTime': int(current_start.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000),
                    'limit': 1000,
                }, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                for kline in data:
                    all_candles.append(OHLCV(
                        timestamp=datetime.fromtimestamp(kline[0] / 1000),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5]),
                    ))

                # Move start time forward
                current_start = all_candles[-1].timestamp + timedelta(minutes=1)

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to get historical klines: {e}")
                break

        return all_candles

    def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """
        Get 24-hour statistics for a symbol.

        Returns:
            Dict with price change, high, low, volume, etc.
        """
        binance_symbol = self._get_binance_symbol(symbol)

        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            response = self.session.get(url, params={'symbol': binance_symbol}, timeout=5)
            response.raise_for_status()
            data = response.json()

            return {
                'price': float(data.get('lastPrice', 0)),
                'price_change': float(data.get('priceChange', 0)),
                'price_change_pct': float(data.get('priceChangePercent', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'volume_24h': float(data.get('volume', 0)),
                'quote_volume_24h': float(data.get('quoteVolume', 0)),
            }

        except Exception as e:
            logger.warning(f"Failed to get 24h stats for {symbol}: {e}")
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Get order book for a symbol.

        Returns:
            Dict with 'bids' and 'asks' lists
        """
        binance_symbol = self._get_binance_symbol(symbol)

        try:
            url = f"{self.base_url}/api/v3/depth"
            response = self.session.get(url, params={
                'symbol': binance_symbol,
                'limit': limit,
            }, timeout=5)
            response.raise_for_status()
            data = response.json()

            return {
                'bids': [[float(price), float(qty)] for price, qty in data.get('bids', [])],
                'asks': [[float(price), float(qty)] for price, qty in data.get('asks', [])],
            }

        except Exception as e:
            logger.warning(f"Failed to get orderbook for {symbol}: {e}")
            return None


# Singleton instance
_data_fetcher: Optional[BTCCDataFetcher] = None


def get_data_fetcher() -> BTCCDataFetcher:
    """Get or create data fetcher singleton."""
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = BTCCDataFetcher()
    return _data_fetcher


# Convenience functions
def get_price(symbol: str) -> Optional[float]:
    """Get current price for symbol."""
    return get_data_fetcher().get_current_price(symbol)


def get_candles(symbol: str, interval: str = '1m', limit: int = 100) -> List[OHLCV]:
    """Get candlestick data for symbol."""
    return get_data_fetcher().get_klines(symbol, interval, limit)


if __name__ == "__main__":
    # Test the data fetcher
    print("Testing BTCC Data Fetcher...")
    print("=" * 50)

    fetcher = BTCCDataFetcher()

    # Test price fetching
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    print("\nCurrent Prices:")
    for symbol in symbols:
        price = fetcher.get_current_price(symbol)
        print(f"  {symbol}: ${price:,.2f}" if price else f"  {symbol}: Failed")

    # Test multiple prices
    print("\nBatch Price Fetch:")
    prices = fetcher.get_multiple_prices(symbols)
    for symbol, price in prices.items():
        print(f"  {symbol}: ${price:,.2f}")

    # Test klines
    print("\nRecent Candles (BTCUSDT, 1m):")
    candles = fetcher.get_klines('BTCUSDT', '1m', 5)
    for c in candles:
        print(f"  {c.timestamp}: O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f}")

    # Test 24h stats
    print("\n24h Stats (BTCUSDT):")
    stats = fetcher.get_24h_stats('BTCUSDT')
    if stats:
        print(f"  Price: ${stats['price']:,.2f}")
        print(f"  Change: {stats['price_change_pct']:+.2f}%")
        print(f"  High: ${stats['high_24h']:,.2f}")
        print(f"  Low: ${stats['low_24h']:,.2f}")
        print(f"  Volume: {stats['volume_24h']:,.2f}")

    print("\n" + "=" * 50)
    print("Data fetcher test complete!")
