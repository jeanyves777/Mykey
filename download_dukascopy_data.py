"""
Download 16 months of 1-minute (M1) data from Dukascopy
Saves data to trading_system/Forex_Trading/Backtesting_data_Dukascopy/

Pairs: EUR_USD, GBP_USD, USD_JPY, USD_CHF, USD_CAD
Period: Aug 2023 - Nov 2024 (16 months)
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

# Output directory
DATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_Dukascopy')

# Pairs to download
PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD']

# Date range: 16 months (Aug 2023 - Nov 2024)
START_YEAR = 2023
START_MONTH = 8
END_YEAR = 2024
END_MONTH = 11


def download_dukascopy_day(pair: str, year: int, month: int, day: int) -> pd.DataFrame:
    """
    Download 1-minute OHLC data from Dukascopy for a single day.

    Dukascopy URL format:
    https://datafeed.dukascopy.com/datafeed/EURUSD/2024/10/14/BID_candles_min_1.bi5

    Note: Month is 0-indexed, day is 0-indexed in URL
    """
    dukascopy_pair = pair.replace("_", "")

    # Dukascopy uses 0-indexed months and days
    url = f"https://datafeed.dukascopy.com/datafeed/{dukascopy_pair}/{year}/{month-1:02d}/{day-1:02d}/BID_candles_min_1.bi5"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return pd.DataFrame()

        if len(response.content) == 0:
            return pd.DataFrame()

        # Decompress LZMA data
        try:
            decompressed = lzma.decompress(response.content)
        except:
            return pd.DataFrame()

        # Parse binary data
        # Each record: 5 floats (time_offset_ms, open, high, low, close, volume)
        # Dukascopy stores data as: time_delta (ms from day start), open, high, low, close, volume
        # Format: >IIIIIf (5 unsigned ints + 1 float)
        record_size = 20  # 5 * 4 bytes
        num_records = len(decompressed) // record_size

        if num_records == 0:
            return pd.DataFrame()

        data = []
        base_time = datetime(year, month, day, 0, 0, 0)

        for i in range(num_records):
            offset = i * record_size
            record = struct.unpack('>IIIIf', decompressed[offset:offset+record_size])

            # time_offset is in milliseconds from start of day
            time_offset_ms = record[0]
            timestamp = base_time + timedelta(milliseconds=time_offset_ms)

            # Prices need different multipliers for JPY pairs vs others
            if 'JPY' in pair:
                price_mult = 1e-3  # JPY pairs use 3 decimal places
            else:
                price_mult = 1e-5  # Other pairs use 5 decimal places

            data.append({
                'time': timestamp,
                'open': record[1] * price_mult,
                'high': record[2] * price_mult,
                'low': record[3] * price_mult,
                'close': record[4] * price_mult,
                'volume': record[4]  # Volume is stored as float
            })

        return pd.DataFrame(data)

    except Exception as e:
        return pd.DataFrame()


def download_month(pair: str, year: int, month: int) -> pd.DataFrame:
    """Download entire month of 1-minute data"""

    # Get number of days in month
    if month == 12:
        next_month_start = datetime(year + 1, 1, 1)
    else:
        next_month_start = datetime(year, month + 1, 1)

    days_in_month = (next_month_start - datetime(year, month, 1)).days

    all_days = []

    for day in range(1, days_in_month + 1):
        df_day = download_dukascopy_day(pair, year, month, day)
        if not df_day.empty:
            all_days.append(df_day)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    if not all_days:
        return pd.DataFrame()

    return pd.concat(all_days, ignore_index=True)


def download_pair_data(pair: str) -> pd.DataFrame:
    """Download full 16 months of data for a pair"""

    print(f"\n{'='*60}")
    print(f"Downloading {pair}")
    print(f"{'='*60}")

    all_months = []

    year = START_YEAR
    month = START_MONTH

    while (year < END_YEAR) or (year == END_YEAR and month <= END_MONTH):
        print(f"  {year}-{month:02d}...", end=" ", flush=True)

        df_month = download_month(pair, year, month)

        if not df_month.empty:
            all_months.append(df_month)
            print(f"{len(df_month):,} bars")
        else:
            print("no data (weekend/holiday)")

        # Next month
        month += 1
        if month > 12:
            month = 1
            year += 1

    if not all_months:
        print(f"  ERROR: No data downloaded for {pair}")
        return pd.DataFrame()

    df = pd.concat(all_months, ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates(subset=['time'], keep='first')

    print(f"\n  Total: {len(df):,} bars ({len(df)/1440:.1f} trading days)")

    return df


def main():
    print("=" * 70)
    print("DUKASCOPY DATA DOWNLOADER")
    print("=" * 70)
    print(f"\nPairs: {', '.join(PAIRS)}")
    print(f"Period: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d} (16 months)")
    print(f"Output: {DATA_DIR}")

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pair in PAIRS:
        df = download_pair_data(pair)

        if df.empty:
            print(f"  Skipping {pair} - no data")
            continue

        # Save to CSV
        output_file = DATA_DIR / f"{pair}_M1.csv"
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    # Summary
    print("\nFiles created:")
    for f in DATA_DIR.glob("*_M1.csv"):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
