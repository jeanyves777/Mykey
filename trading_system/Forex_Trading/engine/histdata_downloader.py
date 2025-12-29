"""
HistData.com Free Forex Data Downloader
Downloads 1-minute historical forex data for backtesting

Alternative to OANDA API for historical data
Source: https://www.histdata.com/
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from zipfile import ZipFile
from io import BytesIO
import pytz
from typing import List, Dict


class HistDataDownloader:
    """
    Download free 1-minute forex data from HistData.com
    """

    def __init__(self, data_dir: str = "trading_system/Forex_Trading/data"):
        """
        Initialize downloader

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.base_url = "https://www.histdata.com/download-free-forex-historical-data"

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)

        # Mapping of instruments to HistData format
        self.instrument_map = {
            "EUR_USD": "EURUSD",
            "GBP_USD": "GBPUSD",
            "USD_JPY": "USDJPY",
            "USD_CHF": "USDCHF",
            "AUD_USD": "AUDUSD",
            "USD_CAD": "USDCAD",
            "NZD_USD": "NZDUSD"
        }

        print(f"[HISTDATA] Initialized data downloader")
        print(f"[HISTDATA] Data directory: {data_dir}")

    def download_month_data(self, instrument: str, year: int, month: int) -> pd.DataFrame:
        """
        Download 1-minute data for a specific month

        NOTE: HistData.com requires manual download via browser.
        This function provides instructions and processes local CSV files.

        Args:
            instrument: Forex pair (e.g., "EUR_USD")
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            DataFrame with OHLC data
        """
        histdata_pair = self.instrument_map.get(instrument, instrument.replace("_", ""))

        filename = f"{histdata_pair}_{year}_{month:02d}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath):
            print(f"[HISTDATA] Loading cached data: {filename}")
            return self._load_csv(filepath)

        print(f"\n[HISTDATA] Data not found locally for {instrument} {year}-{month:02d}")
        print(f"[HISTDATA] To download:")
        print(f"  1. Go to: https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/{histdata_pair.lower()}/{year}/{month}")
        print(f"  2. Download the ZIP file")
        print(f"  3. Extract the CSV file")
        print(f"  4. Place it in: {self.data_dir}/{filename}")
        print(f"  5. Run backtest again\n")

        return pd.DataFrame()

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load and parse HistData CSV format"""
        try:
            # HistData format: DateTime,Open,High,Low,Close,Volume
            # Example: 20240101 000000,1.10450,1.10455,1.10448,1.10452,100
            df = pd.read_csv(filepath, sep=',', names=['datetime', 'open', 'high', 'low', 'close', 'volume'])

            # Parse datetime
            df['time'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
            df['time'] = df['time'].dt.tz_localize('UTC')

            # Drop original datetime column
            df = df.drop('datetime', axis=1)

            # Reorder columns
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

            print(f"[HISTDATA] Loaded {len(df)} bars from {filepath}")
            return df

        except Exception as e:
            print(f"[HISTDATA] Error loading {filepath}: {e}")
            return pd.DataFrame()

    def get_data_range(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get 1-minute data for a date range

        Args:
            instrument: Forex pair (e.g., "EUR_USD")
            start_date: Start date (datetime with timezone)
            end_date: End date (datetime with timezone)

        Returns:
            DataFrame with OHLC data for the entire range
        """
        print(f"[HISTDATA] Fetching {instrument} data from {start_date.date()} to {end_date.date()}")

        all_data = []

        # Iterate through each month in the range
        current = start_date.replace(day=1)
        end = end_date.replace(day=1)

        while current <= end:
            month_data = self.download_month_data(instrument, current.year, current.month)

            if not month_data.empty:
                # Filter to date range
                mask = (month_data['time'] >= start_date) & (month_data['time'] <= end_date)
                filtered = month_data[mask]
                all_data.append(filtered)

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if len(all_data) == 0:
            print(f"[HISTDATA] No data available for {instrument}")
            return pd.DataFrame()

        # Concatenate all months
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('time').reset_index(drop=True)

        print(f"[HISTDATA] Total bars loaded: {len(df)}")
        return df

    def generate_sample_data(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        base_price: float = 1.0850
    ) -> pd.DataFrame:
        """
        Generate synthetic sample data for testing

        Args:
            instrument: Forex pair
            start_date: Start datetime
            end_date: End datetime
            base_price: Starting price

        Returns:
            DataFrame with synthetic OHLC data
        """
        print(f"[HISTDATA] Generating sample data for {instrument}")

        import numpy as np

        # Generate 1-minute timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min', tz='UTC')

        # Generate realistic price movements
        np.random.seed(42)
        returns = np.random.normal(0, 0.0001, len(timestamps))  # Small random returns
        prices = base_price * (1 + returns).cumprod()

        # Generate OHLC from prices
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            # Add some volatility
            volatility = close * 0.0002
            high = close + abs(np.random.normal(0, volatility))
            low = close - abs(np.random.normal(0, volatility))
            open_price = prices[i-1] if i > 0 else close

            data.append({
                'time': ts,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': int(np.random.uniform(50, 200))
            })

        df = pd.DataFrame(data)
        print(f"[HISTDATA] Generated {len(df)} synthetic bars")

        return df


# Alternative: Use yfinance for forex data (limited history)
def get_yfinance_forex_data(instrument: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download forex data using yfinance (alternative source)

    Args:
        instrument: Forex pair (e.g., "EUR_USD")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLC data
    """
    try:
        import yfinance as yf

        # Convert to yfinance format
        yf_symbol = instrument.replace("_", "") + "=X"  # EURUSD=X

        print(f"[YFINANCE] Downloading {yf_symbol} from {start_date} to {end_date}")

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1m")

        if df.empty:
            print(f"[YFINANCE] No data returned for {yf_symbol}")
            return pd.DataFrame()

        # Convert to our format
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Ensure timezone
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        else:
            df['time'] = df['time'].dt.tz_convert('UTC')

        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

        print(f"[YFINANCE] Downloaded {len(df)} bars")
        return df

    except ImportError:
        print("[YFINANCE] yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"[YFINANCE] Error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the downloader
    downloader = HistDataDownloader()

    # Try to load November 2024 data
    start = datetime(2024, 11, 1, tzinfo=pytz.UTC)
    end = datetime(2024, 11, 30, tzinfo=pytz.UTC)

    # Try EUR/USD
    data = downloader.get_data_range("EUR_USD", start, end)

    if data.empty:
        print("\n[TEST] No data found. Generating sample data for testing...")
        data = downloader.generate_sample_data("EUR_USD", start, end, base_price=1.0850)

    print(f"\n[TEST] Data shape: {data.shape}")
    print(f"\n[TEST] First 5 rows:")
    print(data.head())
    print(f"\n[TEST] Last 5 rows:")
    print(data.tail())
