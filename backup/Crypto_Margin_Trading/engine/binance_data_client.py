"""
Binance Historical Data Downloader
===================================
Downloads unlimited historical candlestick data from Binance.
No API key required for public market data.

Features:
- Unlimited candle download (pagination supported)
- Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- Saves to CSV for backtesting
- Resume capability (appends new data)
- All major crypto pairs supported

Usage:
    python binance_data_client.py

    # Or programmatically:
    client = BinanceDataClient()
    client.download_all_pairs(candle_count=100000)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


class BinanceDataClient:
    """Binance public API client for historical data"""

    # Use Binance.US for US-based users (no 451 geo-block)
    # Falls back to international Binance if US fails
    BASE_URL = "https://api.binance.us"
    FALLBACK_URL = "https://api.binance.com"

    # Alternative: CryptoCompare API (no geo restrictions, but limited free tier)
    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com"

    # Trading pairs to download (Binance symbols)
    # These match Kraken's margin trading pairs - Updated Dec 19, 2025
    PAIRS = [
        # Top 10x leverage pairs on Kraken
        "BTCUSDT",   # Bitcoin - 10x
        "ETHUSDT",   # Ethereum - 10x
        "SOLUSDT",   # Solana - 10x
        "XRPUSDT",   # Ripple - 10x
        "DOGEUSDT",  # Dogecoin - 10x
        "LTCUSDT",   # Litecoin - 10x
        "ADAUSDT",   # Cardano - 10x
        "LINKUSDT",  # Chainlink - 10x
        "AVAXUSDT",  # Avalanche - 10x
        "SUIUSDT",   # SUI - 10x (NEW)
        # 3x leverage pairs
        "DOTUSDT",   # Polkadot - 3x
        "ZECUSDT",   # Zcash - 3x (NEW)
        "BCHUSDT",   # Bitcoin Cash - 3x (NEW)
        "PEPEUSDT",  # PEPE - 3x (NEW)
        "UNIUSDT",   # Uniswap - 3x (NEW)
        # 2x leverage pairs
        "XLMUSDT",   # Stellar - 2x (NEW)
        "XMRUSDT",   # Monero - 2x (NEW)
    ]

    # Mapping from Binance to Kraken symbols
    BINANCE_TO_KRAKEN = {
        "BTCUSDT": "XXBTZUSD",
        "ETHUSDT": "XETHZUSD",
        "SOLUSDT": "SOLUSD",
        "XRPUSDT": "XXRPZUSD",
        "DOGEUSDT": "XDGUSD",
        "LTCUSDT": "XLTCZUSD",
        "ADAUSDT": "ADAUSD",
        "LINKUSDT": "LINKUSD",
        "AVAXUSDT": "AVAXUSD",
        "DOTUSDT": "DOTUSD",
        "SUIUSDT": "SUIUSD",
        "ZECUSDT": "XZECZUSD",
        "BCHUSDT": "BCHUSD",
        "PEPEUSDT": "PEPEUSD",
        "XLMUSDT": "XXLMZUSD",
        "UNIUSDT": "UNIUSD",
        "XMRUSDT": "XXMRZUSD",
    }

    # Timeframe mapping
    TIMEFRAMES = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize Binance data client

        Args:
            data_dir: Directory to save CSV files
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / "Crypto_Data_from_Binance"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

        print(f"[Binance] Data directory: {self.data_dir}")

    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Get candlestick data from Binance (tries US first, then fallback)

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            List of candle dictionaries
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)  # Binance max is 1000
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        # Try Binance.US first, then international Binance
        urls_to_try = [
            f"{self.BASE_URL}/api/v3/klines",  # Binance.US
            f"{self.FALLBACK_URL}/api/v3/klines",  # International Binance
        ]

        for url in urls_to_try:
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Check for geo-block (451) or other client errors
                if response.status_code == 451:
                    print(f"[Binance] Geo-blocked at {url.split('/api')[0]}, trying next...")
                    continue

                response.raise_for_status()
                data = response.json()

                candles = []
                for kline in data:
                    candles.append({
                        "timestamp": kline[0],
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "close_time": kline[6],
                        "quote_volume": float(kline[7]),
                        "trades": int(kline[8]),
                        "taker_buy_base": float(kline[9]),
                        "taker_buy_quote": float(kline[10]),
                    })

                return candles

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 451:
                    continue  # Try next URL
                print(f"[Binance] HTTP Error at {url.split('/api')[0]}: {e}")
                continue
            except Exception as e:
                print(f"[Binance] Error fetching {symbol} from {url.split('/api')[0]}: {e}")
                continue

        # If all Binance sources fail, try CryptoCompare
        return self._get_klines_from_cryptocompare(symbol, interval, limit, start_time, end_time)

    def _get_klines_from_cryptocompare(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Fallback: Get candle data from CryptoCompare (no geo restrictions)
        """
        # Convert Binance symbol to CryptoCompare format (BTCUSDT -> BTC, USDT)
        if symbol.endswith("USDT"):
            fsym = symbol[:-4]  # BTC
            tsym = "USDT"
        elif symbol.endswith("USD"):
            fsym = symbol[:-3]
            tsym = "USD"
        else:
            print(f"[CryptoCompare] Unsupported symbol format: {symbol}")
            return []

        # CryptoCompare uses different endpoint per timeframe
        if interval == "1m":
            endpoint = "histominute"
        elif interval in ["5m", "15m", "30m"]:
            # CryptoCompare only has minute/hour/day - use minute for sub-hour
            endpoint = "histominute"
        elif interval in ["1h", "4h"]:
            endpoint = "histohour"
        else:
            endpoint = "histoday"

        url = f"{self.CRYPTOCOMPARE_URL}/data/v2/{endpoint}"

        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": min(limit, 2000),  # CryptoCompare allows up to 2000
        }

        if end_time:
            params["toTs"] = end_time // 1000  # Convert ms to seconds

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("Response") != "Success":
                print(f"[CryptoCompare] API Error: {data.get('Message', 'Unknown error')}")
                return []

            candles = []
            for kline in data.get("Data", {}).get("Data", []):
                # CryptoCompare uses seconds, we need milliseconds
                ts_ms = kline["time"] * 1000
                candles.append({
                    "timestamp": ts_ms,
                    "open": float(kline["open"]),
                    "high": float(kline["high"]),
                    "low": float(kline["low"]),
                    "close": float(kline["close"]),
                    "volume": float(kline.get("volumefrom", 0)),
                    "close_time": ts_ms + 60000,  # Approximate
                    "quote_volume": float(kline.get("volumeto", 0)),
                    "trades": 0,  # Not provided
                    "taker_buy_base": 0,
                    "taker_buy_quote": 0,
                })

            print(f"[CryptoCompare] Retrieved {len(candles)} candles for {symbol}")
            return candles

        except Exception as e:
            print(f"[CryptoCompare] Error fetching {symbol}: {e}")
            return []

    def download_historical_data(
        self,
        symbol: str,
        interval: str = "1m",
        candle_count: int = 100000,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Download large amount of historical data with pagination

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe
            candle_count: Total candles to download
            end_time: End datetime (defaults to now)

        Returns:
            DataFrame with all candles
        """
        print(f"\n[Binance] Downloading {candle_count:,} {interval} candles for {symbol}...")

        all_candles = []
        end_ts = int((end_time or datetime.utcnow()).timestamp() * 1000)

        # Calculate time step per candle in ms
        interval_ms = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }[interval]

        remaining = candle_count
        current_end = end_ts
        batch_num = 0

        while remaining > 0:
            batch_size = min(remaining, 1000)
            batch_num += 1

            # Calculate start time for this batch
            start_ts = current_end - (batch_size * interval_ms)

            candles = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=batch_size,
                start_time=start_ts,
                end_time=current_end
            )

            if not candles:
                print(f"[Binance] No more data available for {symbol}")
                break

            all_candles = candles + all_candles  # Prepend (oldest first)
            remaining -= len(candles)
            current_end = candles[0]["timestamp"] - 1  # Move end to before oldest candle

            # Progress update
            downloaded = candle_count - remaining
            pct = (downloaded / candle_count) * 100
            print(f"  Batch {batch_num}: {downloaded:,}/{candle_count:,} ({pct:.1f}%)")

            # Rate limiting (1200 requests/min = 20/sec)
            time.sleep(0.1)

        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Remove duplicates
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

            print(f"[Binance] Downloaded {len(df):,} candles for {symbol}")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

            return df

        return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str = "1m"):
        """Save DataFrame to CSV file"""
        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath, index=False)
        print(f"[Binance] Saved to {filepath}")

        return filepath

    def load_from_csv(self, symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
        """Load DataFrame from CSV file"""
        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / filename

        if filepath.exists():
            df = pd.read_csv(filepath)
            df["datetime"] = pd.to_datetime(df["datetime"])
            print(f"[Binance] Loaded {len(df):,} candles from {filepath}")
            return df

        return None

    def update_existing_data(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """
        Update existing CSV with new data

        Args:
            symbol: Trading pair
            interval: Timeframe

        Returns:
            Updated DataFrame
        """
        existing_df = self.load_from_csv(symbol, interval)

        if existing_df is not None and len(existing_df) > 0:
            # Get last timestamp and download new data
            last_ts = existing_df["timestamp"].max()
            print(f"[Binance] Updating {symbol} from {pd.to_datetime(last_ts, unit='ms')}")

            # Download new candles
            new_candles = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=last_ts + 1
            )

            if new_candles:
                new_df = pd.DataFrame(new_candles)
                new_df["datetime"] = pd.to_datetime(new_df["timestamp"], unit="ms")

                # Combine and remove duplicates
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)

                self.save_to_csv(combined, symbol, interval)
                print(f"[Binance] Added {len(new_candles)} new candles to {symbol}")

                return combined
            else:
                print(f"[Binance] No new data for {symbol}")
                return existing_df
        else:
            # No existing data, download fresh
            return self.download_historical_data(symbol, interval, candle_count=100000)

    def download_all_pairs(
        self,
        interval: str = "1m",
        candle_count: int = 100000,
        pairs: Optional[List[str]] = None
    ):
        """
        Download historical data for all trading pairs

        Args:
            interval: Timeframe
            candle_count: Candles per pair
            pairs: Optional list of pairs (defaults to all)
        """
        pairs_to_download = pairs or self.PAIRS

        print("=" * 70)
        print(f"BINANCE HISTORICAL DATA DOWNLOAD")
        print("=" * 70)
        print(f"Pairs: {len(pairs_to_download)}")
        print(f"Interval: {interval}")
        print(f"Candles per pair: {candle_count:,}")
        print(f"Data directory: {self.data_dir}")
        print("=" * 70)

        results = []

        for i, symbol in enumerate(pairs_to_download, 1):
            print(f"\n[{i}/{len(pairs_to_download)}] {symbol}")

            try:
                df = self.download_historical_data(
                    symbol=symbol,
                    interval=interval,
                    candle_count=candle_count
                )

                if len(df) > 0:
                    self.save_to_csv(df, symbol, interval)
                    results.append({
                        "symbol": symbol,
                        "candles": len(df),
                        "start": df["datetime"].min(),
                        "end": df["datetime"].max(),
                        "status": "OK"
                    })
                else:
                    results.append({
                        "symbol": symbol,
                        "candles": 0,
                        "status": "FAILED"
                    })

            except Exception as e:
                print(f"[Binance] Error downloading {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "candles": 0,
                    "status": f"ERROR: {e}"
                })

            # Rate limiting between pairs
            time.sleep(1)

        # Summary
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)

        total_candles = 0
        for r in results:
            status = r.get("status", "UNKNOWN")
            candles = r.get("candles", 0)
            total_candles += candles

            if status == "OK":
                print(f"  {r['symbol']}: {candles:,} candles ({r['start']} to {r['end']})")
            else:
                print(f"  {r['symbol']}: {status}")

        print("-" * 70)
        print(f"TOTAL: {total_candles:,} candles downloaded")
        print("=" * 70)

        return results

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for a symbol"""
        url = f"{self.BASE_URL}/api/v3/ticker/price"
        params = {"symbol": symbol}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": data["symbol"],
                "price": float(data["price"])
            }
        except Exception as e:
            print(f"[Binance] Error getting price for {symbol}: {e}")
            return None

    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker statistics"""
        url = f"{self.BASE_URL}/api/v3/ticker/24hr"
        params = {"symbol": symbol}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": data["symbol"],
                "price_change": float(data["priceChange"]),
                "price_change_percent": float(data["priceChangePercent"]),
                "high_24h": float(data["highPrice"]),
                "low_24h": float(data["lowPrice"]),
                "volume_24h": float(data["volume"]),
                "quote_volume_24h": float(data["quoteVolume"]),
            }
        except Exception as e:
            print(f"[Binance] Error getting 24h ticker for {symbol}: {e}")
            return None


def main():
    """Download historical data for all pairs"""
    client = BinanceDataClient()

    # Download 100K 1-minute candles for each pair
    # This gives ~69 days of data per pair
    client.download_all_pairs(
        interval="1m",
        candle_count=100000
    )


if __name__ == "__main__":
    main()
