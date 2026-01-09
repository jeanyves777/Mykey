#!/usr/bin/env python3
"""Patch backtest to use Binance Futures API instead of Yahoo Finance"""

import re

filepath = 'backtest_htf_confluence.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the get_historical_data method
old_pattern = r'(    def get_historical_data\(self, days: int = 90, ltf_interval: str = "15m"\):.*?return None, None)'

new_method = '''    def get_historical_data(self, days: int = 90, ltf_interval: str = "15m"):
        """
        Fetch historical data using Binance Futures API (real futures data).

        Args:
            days: Number of days of history
            ltf_interval: LTF interval (15m or 1h)

        Returns:
            (ltf_df, htf_df) DataFrames
        """
        print(f"\\nFetching {days} days of data for {self.symbol}...")
        print("Using Binance Futures API for historical data...")

        def fetch_binance_futures_klines(symbol, interval, num_days):
            import requests
            from datetime import datetime, timedelta

            base_url = "https://fapi.binance.com"
            endpoint = "/fapi/v1/klines"

            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=num_days)).timestamp() * 1000)

            all_klines = []
            limit = 1500

            while start_time < end_time:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_time,
                    "limit": limit
                }

                response = requests.get(base_url + endpoint, params=params)
                data = response.json()

                if not data or isinstance(data, dict):
                    break

                all_klines.extend(data)
                start_time = data[-1][0] + 1

                if len(data) < limit:
                    break

            if not all_klines:
                return None

            df = pd.DataFrame(all_klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        try:
            ltf_days = min(days, 60)
            print(f"  Fetching {ltf_interval} data (last {ltf_days} days)...")
            ltf_df = fetch_binance_futures_klines(self.symbol, ltf_interval, ltf_days)

            if ltf_df is None or len(ltf_df) == 0:
                print(f"ERROR: No LTF data returned for {self.symbol}")
                return None, None

            htf_days = days + 50
            print(f"  Fetching 4h data (last {htf_days} days)...")
            htf_df = fetch_binance_futures_klines(self.symbol, "4h", htf_days)

            if htf_df is None or len(htf_df) == 0:
                print(f"ERROR: No HTF data returned for {self.symbol}")
                return None, None

            print(f"  LTF ({ltf_interval}): {len(ltf_df)} candles")
            print(f"  HTF (4h): {len(htf_df)} candles")

            return ltf_df, htf_df

        except Exception as e:
            print(f"[ERROR] Binance Futures API failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None'''

# Use regex with DOTALL to match across newlines
new_content = re.sub(old_pattern, new_method, content, flags=re.DOTALL)

if new_content != content:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS: Patched backtest to use Binance Futures API")
else:
    print("FAILED: Pattern not found")
