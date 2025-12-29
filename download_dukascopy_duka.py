"""
Download 16 months of 1-minute (M1) data from Dukascopy using the duka library
Saves data to trading_system/Forex_Trading/Backtesting_data_Dukascopy/

Pairs: EUR_USD, GBP_USD, USD_JPY, USD_CHF, USD_CAD
Period: Aug 2023 - Nov 2024 (16 months)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime, date
from pathlib import Path
import pandas as pd

# Try to import duka
try:
    from duka.app import app
    from duka.core.utils import TimeFrame
    DUKA_AVAILABLE = True
except ImportError:
    DUKA_AVAILABLE = False
    print("duka library not available, using direct download method")

# Output directory
DATA_DIR = Path(r'C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data_Dukascopy')

# Pairs to download (Dukascopy format)
PAIRS_DUKA = {
    'EUR_USD': 'EURUSD',
    'GBP_USD': 'GBPUSD',
    'USD_JPY': 'USDJPY',
    'USD_CHF': 'USDCHF',
    'USD_CAD': 'USDCAD'
}

# Date range
START_DATE = date(2023, 8, 1)
END_DATE = date(2024, 11, 30)


def download_with_duka():
    """Download using duka library"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pair_name, duka_pair in PAIRS_DUKA.items():
        print(f"\n{'='*60}")
        print(f"Downloading {pair_name} ({duka_pair})")
        print(f"{'='*60}")

        output_file = DATA_DIR / f"{pair_name}_M1.csv"

        try:
            # Use duka's app function with all required args
            app(
                symbols=[duka_pair],
                start=START_DATE,
                end=END_DATE,
                timeframe=TimeFrame.M1,
                folder=str(DATA_DIR),
                threads=10,
                header=True
            )

            # Find and rename file to our format
            for f in DATA_DIR.glob(f"{duka_pair}*.csv"):
                if output_file.name not in str(f):
                    f.rename(output_file)
                    print(f"  Saved: {output_file}")
                    break

        except Exception as e:
            print(f"  Error with duka: {e}")
            print("  Falling back to direct download...")


def download_direct():
    """Direct download method as fallback"""
    import requests
    import lzma
    import struct
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def download_hour(pair, duka_pair, dt):
        """Download data for a single hour"""
        year = dt.year
        month = dt.month - 1  # 0-indexed
        day = dt.day
        hour = dt.hour

        url = f"https://datafeed.dukascopy.com/datafeed/{duka_pair}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200 or len(response.content) == 0:
                return pd.DataFrame()

            decompressed = lzma.decompress(response.content)
            if len(decompressed) == 0:
                return pd.DataFrame()

            tick_size = 20
            num_ticks = len(decompressed) // tick_size
            if num_ticks == 0:
                return pd.DataFrame()

            ticks = []
            base_time = datetime(year, dt.month, day, hour)
            price_mult = 1e-3 if 'JPY' in pair else 1e-5

            for i in range(num_ticks):
                offset = i * tick_size
                record = struct.unpack('>IIIff', decompressed[offset:offset+tick_size])
                from datetime import timedelta
                timestamp = base_time + timedelta(milliseconds=record[0])
                bid = record[2] * price_mult
                ask = record[1] * price_mult
                ticks.append({'timestamp': timestamp, 'mid': (bid + ask) / 2})

            df = pd.DataFrame(ticks)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            ohlc = df['mid'].resample('1min').ohlc().dropna().reset_index()
            ohlc.columns = ['time', 'open', 'high', 'low', 'close']
            return ohlc

        except:
            return pd.DataFrame()

    for pair_name, duka_pair in PAIRS_DUKA.items():
        print(f"\n{'='*60}")
        print(f"Downloading {pair_name}")
        print(f"{'='*60}")

        # Generate all hours
        from datetime import timedelta
        hours = []
        current = datetime.combine(START_DATE, datetime.min.time())
        end = datetime.combine(END_DATE, datetime.min.time())

        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                hours.append(current)
            current += timedelta(hours=1)

        print(f"  Total hours: {len(hours)}")

        all_data = []
        completed = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_hour, pair_name, duka_pair, h): h for h in hours}

            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    all_data.append(df)
                completed += 1
                if completed % 500 == 0:
                    print(f"    Progress: {completed}/{len(hours)} hours ({len(all_data)} with data)")

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('time').drop_duplicates(subset=['time'])
            df['volume'] = 1

            output_file = DATA_DIR / f"{pair_name}_M1.csv"
            df.to_csv(output_file, index=False)
            print(f"  Saved: {output_file} ({len(df):,} bars)")
        else:
            print(f"  No data for {pair_name}")


def main():
    print("=" * 70)
    print("DUKASCOPY DATA DOWNLOADER")
    print("=" * 70)
    print(f"\nPairs: {', '.join(PAIRS_DUKA.keys())}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output: {DATA_DIR}")

    if DUKA_AVAILABLE:
        print("\nUsing duka library for downloads...")
        download_with_duka()
    else:
        print("\nUsing direct download method...")
        download_direct()

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    # Summary
    print("\nFiles created:")
    for f in DATA_DIR.glob("*_M1.csv"):
        df = pd.read_csv(f)
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {len(df):,} bars ({size:.1f} MB)")


if __name__ == "__main__":
    main()
