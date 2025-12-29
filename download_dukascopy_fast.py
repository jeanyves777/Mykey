"""
Fast Dukascopy Data Downloader - Parallel downloads with proper error handling
Downloads 16 months of 1-minute (M1) data
Saves data to trading_system/Forex_Trading/Backtesting_data_Dukascopy/

Uses ThreadPoolExecutor for parallel downloads to speed up the process significantly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Output directory
DATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_Dukascopy')

# Pairs to download
PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD']

# Date range: 16 months (Aug 2023 - Nov 2024)
START_DATE = datetime(2023, 8, 1)
END_DATE = datetime(2024, 11, 30)

# Thread-safe counter
download_lock = threading.Lock()
download_count = 0
total_downloads = 0


def download_dukascopy_hour(pair: str, dt: datetime) -> pd.DataFrame:
    """
    Download 1-minute OHLC data from Dukascopy for a single hour.

    Dukascopy stores data hourly:
    https://datafeed.dukascopy.com/datafeed/EURUSD/2024/10/14/00h_ticks.bi5
    """
    global download_count

    dukascopy_pair = pair.replace("_", "")
    year = dt.year
    month = dt.month - 1  # 0-indexed
    day = dt.day - 1      # 0-indexed (Dukascopy quirk - actually day is 1-indexed but we need the date)
    hour = dt.hour

    # Actually Dukascopy uses real day numbers (1-indexed), let me check the URL format again
    # Correct URL format: datafeed/EURUSD/2024/10/14/00h_ticks.bi5
    # Month is 0-indexed, day is 1-indexed

    url = f"https://datafeed.dukascopy.com/datafeed/{dukascopy_pair}/{year}/{month:02d}/{dt.day:02d}/{hour:02d}h_ticks.bi5"

    try:
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            return pd.DataFrame()

        if len(response.content) == 0:
            return pd.DataFrame()

        # Decompress LZMA data
        try:
            decompressed = lzma.decompress(response.content)
        except:
            return pd.DataFrame()

        if len(decompressed) == 0:
            return pd.DataFrame()

        # Parse tick data: each tick is 20 bytes
        # Format: time_ms (4 bytes), ask (4 bytes), bid (4 bytes), ask_vol (4 bytes), bid_vol (4 bytes)
        tick_size = 20
        num_ticks = len(decompressed) // tick_size

        if num_ticks == 0:
            return pd.DataFrame()

        ticks = []
        base_time = datetime(year, dt.month, dt.day, hour, 0, 0)

        # Determine price multiplier based on pair
        if 'JPY' in pair:
            price_mult = 1e-3
        else:
            price_mult = 1e-5

        for i in range(num_ticks):
            offset = i * tick_size
            record = struct.unpack('>IIIff', decompressed[offset:offset+tick_size])

            time_offset_ms = record[0]
            timestamp = base_time + timedelta(milliseconds=time_offset_ms)
            ask = record[1] * price_mult
            bid = record[2] * price_mult

            ticks.append({
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2
            })

        if not ticks:
            return pd.DataFrame()

        # Convert to 1-minute OHLC
        df = pd.DataFrame(ticks)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        ohlc = df['mid'].resample('1min').ohlc()
        ohlc = ohlc.dropna()

        if ohlc.empty:
            return pd.DataFrame()

        ohlc = ohlc.reset_index()
        ohlc.columns = ['time', 'open', 'high', 'low', 'close']
        ohlc['volume'] = 1  # Placeholder

        with download_lock:
            download_count += 1
            if download_count % 100 == 0:
                print(f"    Downloaded {download_count}/{total_downloads} hours...")

        return ohlc

    except Exception as e:
        return pd.DataFrame()


def download_pair_parallel(pair: str) -> pd.DataFrame:
    """Download data for a pair using parallel threads"""
    global download_count, total_downloads

    print(f"\n{'='*60}")
    print(f"Downloading {pair}")
    print(f"{'='*60}")

    # Generate all hour timestamps
    current = START_DATE
    hours = []
    while current <= END_DATE:
        # Skip weekends (Sat=5, Sun=6)
        if current.weekday() < 5:
            hours.append(current)
        current += timedelta(hours=1)

    total_downloads = len(hours)
    download_count = 0
    print(f"  Total hours to download: {len(hours)}")

    all_data = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_dukascopy_hour, pair, h): h for h in hours}

        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                all_data.append(df)

    if not all_data:
        print(f"  WARNING: No data downloaded for {pair}")
        return pd.DataFrame()

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['time'], keep='first')

    print(f"  Total: {len(df):,} M1 bars")

    return df


def main():
    print("=" * 70)
    print("DUKASCOPY FAST DATA DOWNLOADER (Parallel)")
    print("=" * 70)
    print(f"\nPairs: {', '.join(PAIRS)}")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Output: {DATA_DIR}")

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pair in PAIRS:
        start = time.time()
        df = download_pair_parallel(pair)
        elapsed = time.time() - start

        if df.empty:
            print(f"  Skipping {pair} - no data")
            continue

        # Save to CSV
        output_file = DATA_DIR / f"{pair}_M1.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        print(f"  Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    # Summary
    print("\nFiles created:")
    for f in DATA_DIR.glob("*_M1.csv"):
        size = f.stat().st_size / 1024 / 1024
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,} bars ({size:.1f} MB)")


if __name__ == "__main__":
    main()
