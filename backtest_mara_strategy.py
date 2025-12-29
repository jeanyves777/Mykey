"""
MARA 0DTE Momentum Strategy Backtest

Downloads MARA 1-minute data from Yahoo Finance and runs backtest
using the mara_0dte_momentum strategy.

Usage:
    python backtest_mara_strategy.py
"""

import asyncio
import io
import json
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from trading_system.core.models import (
    Bar, Instrument, OptionContract, InstrumentType, OptionType
)
from trading_system.engine.backtest_engine import BacktestEngine, BacktestConfig
from trading_system.strategies.mara_0dte_momentum import (
    MARADaily0DTEMomentumStrategy, MARADaily0DTEMomentumConfig
)
from trading_system.analytics.performance import PerformanceAnalyzer, DataSourceInfo


def download_mara_data(days_back: int = 30) -> pd.DataFrame:
    """
    Download MARA 1-minute data from Yahoo Finance.
    Yahoo limits 1-min data to 7 days per request, so we fetch in chunks.
    """
    print(f"\n{'='*60}")
    print("DOWNLOADING MARA DATA FROM YAHOO FINANCE")
    print(f"{'='*60}")

    ticker = yf.Ticker("MARA")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Target period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Fetching 1-minute bars in 7-day chunks...")

    # Fetch in 7-day chunks (Yahoo limit)
    all_data = []
    chunk_end = end_date
    chunk_num = 0

    while chunk_end > start_date:
        chunk_start = max(start_date, chunk_end - timedelta(days=7))
        chunk_num += 1

        print(f"   Chunk {chunk_num}: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...", end=" ")

        try:
            hist = ticker.history(
                start=chunk_start.strftime('%Y-%m-%d'),
                end=chunk_end.strftime('%Y-%m-%d'),
                interval='1m'
            )

            if len(hist) > 0:
                print(f"{len(hist):,} bars")
                all_data.append(hist)
            else:
                print("no data")

        except Exception as e:
            print(f"error: {e}")

        # Move to previous chunk
        chunk_end = chunk_start - timedelta(days=1)

        # Small delay to avoid rate limiting
        import time
        time.sleep(0.5)

    if not all_data:
        print("\nERROR: No data returned from Yahoo Finance")
        return None

    # Combine all chunks
    combined = pd.concat(all_data)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    # Convert to our format
    df = pd.DataFrame({
        'timestamp': combined.index.tz_localize('UTC') if combined.index.tz is None else combined.index.tz_convert('UTC'),
        'open': combined['Open'].values,
        'high': combined['High'].values,
        'low': combined['Low'].values,
        'close': combined['Close'].values,
        'volume': combined['Volume'].values.astype(int),
    }).reset_index(drop=True)

    print(f"\nTotal downloaded: {len(df):,} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # Get unique trading days
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    trading_days = df['date'].nunique()
    print(f"Trading days: {trading_days}")

    return df


def save_mara_data(df: pd.DataFrame, filename: str = "mara_1min_data.json"):
    """Save MARA data to JSON file for future use."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    filepath = data_dir / filename

    # Convert to serializable format
    data = {
        'symbol': 'MARA',
        'interval': '1m',
        'downloaded_at': datetime.now().isoformat(),
        'bars': len(df),
        'start_date': str(df['timestamp'].min()),
        'end_date': str(df['timestamp'].max()),
        'data': df.to_dict(orient='records')
    }

    # Convert timestamps to strings
    for record in data['data']:
        record['timestamp'] = str(record['timestamp'])
        if 'date' in record:
            record['date'] = str(record['date'])

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nData saved to: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.1f} KB")

    return filepath


def create_mara_option_contracts(df: pd.DataFrame, max_contracts: int = 100) -> list:
    """
    Create MARA option contracts for the backtest period.
    MARA has $0.50 strike increments (low-priced stock).

    Parameters:
        max_contracts: Maximum number of contracts to create (for faster backtesting)
    """
    contracts = []
    created_symbols = set()

    # Get unique dates
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    trading_days = sorted(df['date'].unique())

    min_price = float(df['low'].min())
    max_price = float(df['high'].max())

    print(f"\nCreating MARA options:")
    print(f"   Price range: ${min_price:.2f} - ${max_price:.2f}")
    print(f"   Max contracts: {max_contracts}")

    # MARA uses $0.50 strike increments
    strike_increment = 0.5

    for day in trading_days:
        day_data = df[df['date'] == day]
        if len(day_data) == 0:
            continue

        day_open = float(day_data['open'].iloc[0])
        day_high = float(day_data['high'].max())
        day_low = float(day_data['low'].min())

        # Create strikes within 10% of day's price (MARA is volatile)
        price_buffer = day_open * 0.10
        strike_low = int((day_low - price_buffer) / strike_increment) * strike_increment
        strike_high = int((day_high + price_buffer) / strike_increment) * strike_increment + strike_increment

        # Weekly expiry (Friday)
        weekday = day.weekday()
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = (4 - weekday) % 7
        expiry_date = day + timedelta(days=days_to_friday)

        expiry_dt = datetime.combine(expiry_date, datetime.min.time())
        expiry_str = expiry_dt.strftime('%y%m%d')

        # Create strikes
        strike = strike_low
        while strike <= strike_high:
            for opt_type in [OptionType.CALL, OptionType.PUT]:
                type_char = 'C' if opt_type == OptionType.CALL else 'P'
                # MARA option symbol format
                symbol = f"MARA{expiry_str}{type_char}{int(strike*1000):08d}"

                if symbol in created_symbols:
                    strike += strike_increment
                    continue
                created_symbols.add(symbol)

                contract = OptionContract(
                    symbol=symbol,
                    instrument_type=InstrumentType.OPTION,
                    currency="USD",
                    multiplier=100,
                    tick_size=0.01,
                    exchange="CBOE",
                    underlying_symbol="MARA",
                    option_type=opt_type,
                    strike_price=float(strike),
                    expiration=expiry_dt,
                )
                contracts.append(contract)

                # Check if we've hit the limit
                if len(contracts) >= max_contracts:
                    print(f"   Created {len(contracts)} option contracts (limit reached)")
                    return contracts

            strike += strike_increment

    print(f"   Created {len(contracts)} option contracts")
    return contracts


def generate_mara_options_bars(underlying_df: pd.DataFrame, contract: OptionContract) -> list:
    """
    Generate synthetic MARA options bars.
    MARA has HIGH IV (~100-150%) as a Bitcoin mining stock.
    """
    import pytz

    # MARA-specific IV parameters (Bitcoin miner = very high vol)
    base_iv = 1.20  # 120% base IV for MARA
    skew_factor = 0.20  # Higher skew
    smile_factor = 0.10

    n = len(underlying_df)
    S_close = underlying_df['close'].values.astype(float)
    S_open = underlying_df['open'].values.astype(float)
    S_high = underlying_df['high'].values.astype(float)
    S_low = underlying_df['low'].values.astype(float)
    volumes = underlying_df.get('volume', pd.Series([1000]*n)).values.astype(float)

    K = contract.strike_price
    is_call = contract.is_call
    r = 0.05

    # Parse timestamps
    expiry_dt = pd.Timestamp(contract.expiration, tz='UTC')
    est_tz = pytz.timezone('America/New_York')

    timestamps = pd.to_datetime(underlying_df['timestamp'])
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize('UTC')

    # Time to expiry
    T_seconds = (expiry_dt - timestamps).dt.total_seconds().values
    T = np.maximum(0.0001, T_seconds / (365.0 * 24 * 3600))

    # Hour of day
    est_hours = timestamps.dt.tz_convert(est_tz).dt.hour.values

    # Intraday IV adjustment
    intraday_adj = np.zeros(n)
    intraday_adj[est_hours == 9] = 0.05
    intraday_adj[est_hours == 10] = 0.03
    intraday_adj[est_hours == 11] = 0.01
    intraday_adj[(est_hours == 12) | (est_hours == 13)] = 0.0
    intraday_adj[est_hours == 14] = 0.01
    intraday_adj[est_hours == 15] = 0.03
    intraday_adj[est_hours == 16] = 0.05

    # 0DTE IV boost
    days_to_expiry = T * 365
    term_adj = np.where(days_to_expiry <= 1, 0.10, np.where(days_to_expiry <= 7, 0.04 * (7 - days_to_expiry) / 6, 0.0))

    # Moneyness
    moneyness = np.log(K / S_close)

    # IV skew
    if is_call:
        skew_adj = skew_factor * np.maximum(0, moneyness) * 0.5
    else:
        skew_adj = skew_factor * np.maximum(0, -moneyness)

    # IV smile
    smile_adj = smile_factor * (np.abs(moneyness) ** 2)

    # Final IV
    sigma = base_iv + skew_adj + smile_adj + term_adj + intraday_adj
    sigma = np.clip(sigma, 0.20, 2.5)

    # Black-Scholes pricing
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_close / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        prices = S_close * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S_close * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    prices = np.maximum(0.01, prices)
    delta_abs = np.abs(delta)

    # Realistic price movement
    underlying_pct_change = np.zeros(n)
    underlying_pct_change[1:] = (S_close[1:] - S_close[:-1]) / S_close[:-1]

    leverage = np.clip(1.0 / np.maximum(delta_abs, 0.1), 1.0, 10.0)

    if is_call:
        option_pct_change = underlying_pct_change * leverage * delta_abs
    else:
        option_pct_change = -underlying_pct_change * leverage * delta_abs

    noise = np.random.normal(0, 0.002, n)
    option_pct_change = option_pct_change + noise

    # Build close prices
    close_prices = np.zeros(n)
    close_prices[0] = prices[0]
    for i in range(1, n):
        theoretical = prices[i]
        previous_close = close_prices[i-1]
        moved_price = previous_close * (1 + option_pct_change[i])
        close_prices[i] = 0.8 * moved_price + 0.2 * theoretical

    close_prices = np.maximum(0.01, close_prices)

    # OHLC
    underlying_bar_range_pct = (S_high - S_low) / S_close
    option_bar_range_pct = np.clip(underlying_bar_range_pct * leverage * delta_abs, 0.001, 0.04)

    open_prices = np.zeros(n)
    open_prices[0] = close_prices[0] * 0.998
    for i in range(1, n):
        underlying_gap = (S_open[i] - S_close[i-1]) / S_close[i-1]
        if is_call:
            gap_effect = underlying_gap * delta_abs[i]
        else:
            gap_effect = -underlying_gap * delta_abs[i]
        open_prices[i] = close_prices[i-1] * (1 + gap_effect)
    open_prices = np.maximum(0.01, open_prices)

    half_range = close_prices * option_bar_range_pct * 0.5
    high_prices = np.maximum(open_prices, close_prices) + half_range * 0.3
    low_prices = np.minimum(open_prices, close_prices) - half_range * 0.3
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    low_prices = np.maximum(0.01, low_prices)

    option_volumes = np.maximum(1, (volumes * 0.01 * np.random.uniform(0.5, 1.5, n)).astype(int))

    # Build bars
    bars = []
    for i in range(n):
        ts = timestamps.iloc[i]
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()

        bar = Bar(
            symbol=contract.symbol,
            timestamp=ts,
            open=round(open_prices[i], 2),
            high=round(high_prices[i], 2),
            low=round(low_prices[i], 2),
            close=round(close_prices[i], 2),
            volume=int(option_volumes[i]),
        )
        bars.append(bar)

    return bars


async def run_mara_backtest(df: pd.DataFrame, position_size: float = 200.0):
    """Run MARA 0DTE momentum backtest."""

    print(f"\n{'='*80}")
    print("MARA 0DTE MOMENTUM STRATEGY BACKTEST")
    print(f"{'='*80}")
    print(f"Data period:      {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Bars:             {len(df):,}")
    print(f"Position size:    ${position_size:.2f}")
    print(f"{'='*80}\n")

    data_info = DataSourceInfo()
    data_info.underlying_source = "REAL (Yahoo Finance API)"
    data_info.underlying_bars = len(df)

    # Create backtest engine
    config = BacktestConfig(
        initial_capital=10000.0,  # Smaller capital for MARA
        commission_per_contract=0.65,
        slippage_pct=0.1,
    )
    engine = BacktestEngine(config)

    # Add underlying instrument
    underlying_inst = Instrument(
        symbol="MARA",
        instrument_type=InstrumentType.STOCK,
        currency="USD",
        exchange="NASDAQ",
    )
    engine.add_instrument(underlying_inst)

    # Convert to bars
    underlying_bars = []
    for _, row in df.iterrows():
        bar = Bar(
            symbol="MARA",
            timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
        )
        underlying_bars.append(bar)
    engine.add_data(underlying_bars)

    # Create option contracts
    print("Creating MARA option contracts...")
    contracts = create_mara_option_contracts(df)

    # Generate options data
    print("Generating options data...")
    total_options_bars = 0
    for i, contract in enumerate(contracts):
        engine.add_instrument(contract)
        options_bars = generate_mara_options_bars(df, contract)
        engine.add_data(options_bars)
        total_options_bars += len(options_bars)

        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(contracts)} contracts...")

    data_info.options_bars = total_options_bars
    print(f"   Total options bars: {total_options_bars:,}")

    # Create MARA strategy using proper MARA strategy class
    print("\nConfiguring MARA 0DTE Momentum strategy...")
    strategy_config = MARADaily0DTEMomentumConfig(
        underlying_symbol="MARA",
        instrument_id="MARA",
        fixed_position_value=position_size,
        target_profit_pct=Decimal("7.5"),
        stop_loss_pct=Decimal("25.0"),
        max_hold_minutes=30,
        max_trades_per_day=3,
        daily_profit_target_pct=15.0,
        trailing_stop_enabled=True,
        trailing_trigger_pct=Decimal("10.0"),
        trailing_distance_pct=Decimal("15.0"),
        entry_time_start="09:30:00",
        entry_time_end="15:45:00",  # Extended for MARA
        force_exit_time="15:50:00",
    )
    strategy = MARADaily0DTEMomentumStrategy(strategy_config)
    engine.add_strategy(strategy)

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run()

    # Generate report
    print("\n" + "=" * 100)
    print("MARA 0DTE MOMENTUM STRATEGY - PERFORMANCE ANALYSIS")
    print("=" * 100)

    analyzer = PerformanceAnalyzer(
        results,
        strategy_name="MARA Daily 0DTE Momentum",
        data_source_info=data_info
    )
    report = analyzer.analyze()
    report_text = analyzer.print_report(report)
    print(report_text)

    # Save results
    output_dir = Path(__file__).parent

    results_file = output_dir / "mara_backtest_results.json"
    with open(results_file, 'w') as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    report_file = output_dir / "mara_backtest_results.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Report saved to: {report_file}")

    return results


def main():
    print("\n" + "=" * 80)
    print("MARA 0DTE MOMENTUM STRATEGY - BACKTEST RUNNER")
    print("=" * 80)

    # Step 1: Download MARA data
    df = download_mara_data(days_back=30)

    if df is None or len(df) == 0:
        print("\nFailed to download data. Exiting.")
        return

    # Step 2: Save data for future use
    save_mara_data(df)

    # Step 3: Run backtest
    asyncio.run(run_mara_backtest(df, position_size=200.0))


if __name__ == '__main__':
    main()
