"""
Backtest Always Hold + Hybrid DCA Strategy
============================================

Tests the strategy on historical data to validate performance.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from trading_system.High_frequency_crypto_tradin.dca_config import load_always_hold_config, DCAConfig


@dataclass
class BacktestPosition:
    """Track a position during backtest."""
    symbol: str
    entry_price: float
    avg_entry_price: float
    quantity: float
    entry_time: datetime
    dca_level: int = 0
    dca_quantities: List[float] = field(default_factory=list)
    dca_prices: List[float] = field(default_factory=list)
    last_dca_time: Optional[datetime] = None
    is_hold: bool = False  # True for HOLD, False for TRADE


@dataclass
class BacktestTrade:
    """Record a completed trade."""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    dca_level: int
    exit_reason: str


class AlwaysHoldBacktest:
    """Backtest engine for Always Hold + Hybrid DCA."""

    def __init__(self, config: DCAConfig):
        self.config = config
        self.commission_pct = config.commission_pct

        # Position tracking
        self.hold_positions: Dict[str, BacktestPosition] = {}
        self.trade_positions: Dict[str, BacktestPosition] = {}

        # Results
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.daily_dca_count: Dict[str, Dict[str, int]] = {}  # date -> symbol -> count

        # Stats
        self.stats = {
            'initial_capital': config.initial_capital,
            'final_equity': config.initial_capital,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'hold_pnl': 0.0,
            'trade_pnl': 0.0,
            'max_drawdown': 0.0,
            'dca_entries': 0,
            'tp_hits': 0,
            'fees_paid': 0.0
        }

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()

    def is_reversal_candle(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """Check for bullish reversal candle."""
        body = abs(row['close'] - row['open'])
        candle_range = row['high'] - row['low']

        if candle_range == 0:
            return False

        body_ratio = body / candle_range
        is_bullish = row['close'] > row['open']

        return is_bullish and body_ratio >= self.config.min_reversal_body_ratio

    def get_volume_ratio(self, df: pd.DataFrame, idx: int, period: int = 20) -> float:
        """Get volume ratio vs average."""
        if idx < period:
            return 1.0

        avg_vol = df['volume'].iloc[idx-period:idx].mean()
        if avg_vol == 0:
            return 1.0

        return df['volume'].iloc[idx] / avg_vol

    def check_dca_filter(self, dca_level: int, rsi: float, adx: float,
                         is_reversal: bool, volume_ratio: float) -> Tuple[bool, str]:
        """Apply Hybrid DCA filter."""
        if self.config.is_easy_dca(dca_level):
            # EASY DCA (1-2)
            if rsi > self.config.easy_dca_rsi_threshold:
                return False, f"RSI {rsi:.1f} > {self.config.easy_dca_rsi_threshold}"
            if adx > self.config.easy_dca_adx_max:
                return False, f"ADX {adx:.1f} > {self.config.easy_dca_adx_max}"
            return True, "EASY"
        else:
            # STRICT DCA (3-4)
            if rsi > self.config.strict_dca_rsi_threshold:
                return False, f"RSI {rsi:.1f} > {self.config.strict_dca_rsi_threshold}"
            if adx > self.config.strict_dca_adx_max:
                return False, f"ADX {adx:.1f} > {self.config.strict_dca_adx_max}"
            if self.config.require_reversal_candle and not is_reversal:
                return False, "No reversal candle"
            if self.config.require_volume_spike and volume_ratio < self.config.volume_spike_multiplier:
                return False, f"Volume {volume_ratio:.1f}x < {self.config.volume_spike_multiplier}x"
            return True, "STRICT"

    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data

        Returns:
            Dict with backtest results
        """
        print("\n" + "=" * 70)
        print("ALWAYS HOLD + HYBRID DCA BACKTEST")
        print("=" * 70)
        print(f"Capital: ${self.config.initial_capital:,.2f}")
        print(f"Per Symbol: ${self.config.allocation_per_symbol:,.2f}")
        print(f"Commission: {self.config.commission_pct*100:.2f}% per trade")
        print("=" * 70)

        # Prepare data with indicators
        prepared_data = {}
        min_len = float('inf')

        for symbol, df in data.items():
            df = df.copy()
            df['rsi'] = self.calculate_rsi(df['close'], self.config.rsi_period)
            df['adx'] = self.calculate_adx(df, self.config.adx_period)
            # Fill NaN in indicators with neutral values instead of dropping rows
            df['rsi'] = df['rsi'].fillna(50.0)
            df['adx'] = df['adx'].fillna(20.0)
            # Only drop if OHLCV has NaN (shouldn't happen after data loading)
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            prepared_data[symbol] = df
            min_len = min(min_len, len(df))
            print(f"{symbol}: {len(df):,} bars")

        if min_len < 100:
            print("ERROR: Not enough data for backtest")
            return {}

        # Initialize positions at first bar
        equity = self.config.initial_capital
        peak_equity = equity

        # Get first prices and enter positions
        first_prices = {}
        for symbol, df in prepared_data.items():
            first_prices[symbol] = df['close'].iloc[0]

        # Enter initial HOLD + TRADE positions
        print("\nInitializing positions...")
        for symbol, price in first_prices.items():
            hold_qty = self.config.get_hold_quantity(symbol, price)
            trade_qty = self.config.get_trade_quantity(symbol, price, dca_level=0)

            # Deduct commission for entry
            entry_cost = (hold_qty + trade_qty) * price * self.commission_pct
            equity -= entry_cost
            self.stats['fees_paid'] += entry_cost

            self.hold_positions[symbol] = BacktestPosition(
                symbol=symbol,
                entry_price=price,
                avg_entry_price=price,
                quantity=hold_qty,
                entry_time=df.index[0] if hasattr(df.index[0], 'strftime') else datetime.now(),
                is_hold=True
            )

            self.trade_positions[symbol] = BacktestPosition(
                symbol=symbol,
                entry_price=price,
                avg_entry_price=price,
                quantity=trade_qty,
                entry_time=df.index[0] if hasattr(df.index[0], 'strftime') else datetime.now(),
                dca_quantities=[trade_qty],
                dca_prices=[price],
                is_hold=False
            )

            print(f"  {symbol} @ ${price:,.2f}: HOLD {hold_qty:.6f} + TRADE {trade_qty:.6f}")

        # Main backtest loop
        print("\nRunning backtest...")
        self.equity_curve = [equity]

        # Use first symbol's index for iteration
        main_symbol = list(prepared_data.keys())[0]
        main_df = prepared_data[main_symbol]

        for i in range(1, min_len):
            current_date = str(main_df.index[i])[:10] if hasattr(main_df.index[i], 'strftime') else f"day_{i}"

            for symbol, df in prepared_data.items():
                if i >= len(df):
                    continue

                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                price = row['close']
                rsi = row['rsi']
                adx = row['adx']
                timestamp = df.index[i] if hasattr(df.index, '__getitem__') else datetime.now()

                # Check reversal and volume
                is_reversal = self.is_reversal_candle(row, prev_row)
                volume_ratio = self.get_volume_ratio(df, i)

                # Update HOLD position (never closes)
                if symbol in self.hold_positions:
                    hold = self.hold_positions[symbol]
                    hold_pnl = (price - hold.entry_price) * hold.quantity

                # Check TRADE position
                if symbol in self.trade_positions:
                    trade = self.trade_positions[symbol]

                    # Calculate current profit
                    profit_pct = (price - trade.avg_entry_price) / trade.avg_entry_price

                    # Dynamic TP based on DCA level
                    tp_pct = self.config.take_profit_pct - (trade.dca_level * self.config.tp_reduction_per_dca)
                    tp_pct = max(tp_pct, self.config.min_tp_pct)

                    # Emergency exit for DCA 4
                    if trade.dca_level >= self.config.emergency_exit_dca_level:
                        if profit_pct >= self.config.emergency_exit_profit_pct:
                            self._close_trade(symbol, price, timestamp, "EMERGENCY_TP")
                            self._reenter_trade(symbol, price, timestamp)
                            continue

                    # Normal TP check
                    if profit_pct >= tp_pct:
                        self._close_trade(symbol, price, timestamp, "TP")
                        self._reenter_trade(symbol, price, timestamp)
                        continue

                    # Check DCA conditions
                    if trade.dca_level < self.config.max_dca_stages:
                        # Check daily limit
                        if current_date not in self.daily_dca_count:
                            self.daily_dca_count[current_date] = {}
                        if self.daily_dca_count[current_date].get(symbol, 0) >= self.config.max_dca_per_day:
                            continue

                        # Check price drop
                        dca_levels = self.config.get_dca_levels(trade.entry_price)
                        next_level = trade.dca_level
                        if next_level < len(dca_levels):
                            target_price = dca_levels[next_level]

                            if price <= target_price:
                                # Check DCA filter
                                can_dca, reason = self.check_dca_filter(
                                    next_level + 1, rsi, adx, is_reversal, volume_ratio
                                )

                                if can_dca:
                                    self._execute_dca(symbol, price, timestamp, next_level + 1)
                                    self.daily_dca_count[current_date][symbol] = \
                                        self.daily_dca_count[current_date].get(symbol, 0) + 1

            # Calculate equity
            equity = self.config.initial_capital + self.stats['trade_pnl']

            # Add unrealized P&L from HOLD positions
            for symbol, hold in self.hold_positions.items():
                if symbol in prepared_data and i < len(prepared_data[symbol]):
                    current_price = prepared_data[symbol]['close'].iloc[i]
                    equity += (current_price - hold.entry_price) * hold.quantity

            # Add unrealized P&L from TRADE positions
            for symbol, trade in self.trade_positions.items():
                if symbol in prepared_data and i < len(prepared_data[symbol]):
                    current_price = prepared_data[symbol]['close'].iloc[i]
                    equity += (current_price - trade.avg_entry_price) * trade.quantity

            self.equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)

            # Track drawdown
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)

        # Final equity calculation
        final_equity = self.config.initial_capital + self.stats['trade_pnl']

        # Add final HOLD P&L
        for symbol, hold in self.hold_positions.items():
            if symbol in prepared_data:
                final_price = prepared_data[symbol]['close'].iloc[-1]
                hold_pnl = (final_price - hold.entry_price) * hold.quantity
                # Subtract commission if we were to close
                hold_pnl -= hold.quantity * final_price * self.commission_pct
                self.stats['hold_pnl'] += hold_pnl
                final_equity += hold_pnl

        # Add final TRADE P&L (unrealized)
        for symbol, trade in self.trade_positions.items():
            if symbol in prepared_data:
                final_price = prepared_data[symbol]['close'].iloc[-1]
                trade_unrealized = (final_price - trade.avg_entry_price) * trade.quantity
                final_equity += trade_unrealized

        self.stats['final_equity'] = final_equity
        self.stats['total_pnl'] = final_equity - self.config.initial_capital

        return self._generate_report()

    def _close_trade(self, symbol: str, price: float, timestamp, reason: str):
        """Close TRADE position."""
        if symbol not in self.trade_positions:
            return

        trade = self.trade_positions[symbol]

        # Calculate P&L
        gross_pnl = (price - trade.avg_entry_price) * trade.quantity
        commission = trade.quantity * price * self.commission_pct
        net_pnl = gross_pnl - commission

        self.stats['fees_paid'] += commission
        self.stats['trade_pnl'] += net_pnl
        self.stats['total_trades'] += 1
        self.stats['tp_hits'] += 1

        if net_pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1

        # Record trade
        self.trades.append(BacktestTrade(
            symbol=symbol,
            entry_price=trade.avg_entry_price,
            exit_price=price,
            quantity=trade.quantity,
            entry_time=trade.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            pnl_pct=(price - trade.avg_entry_price) / trade.avg_entry_price * 100,
            dca_level=trade.dca_level,
            exit_reason=reason
        ))

        del self.trade_positions[symbol]

    def _reenter_trade(self, symbol: str, price: float, timestamp):
        """Re-enter TRADE position after TP."""
        trade_qty = self.config.get_trade_quantity(symbol, price, dca_level=0)

        # Deduct commission
        commission = trade_qty * price * self.commission_pct
        self.stats['fees_paid'] += commission

        self.trade_positions[symbol] = BacktestPosition(
            symbol=symbol,
            entry_price=price,
            avg_entry_price=price,
            quantity=trade_qty,
            entry_time=timestamp,
            dca_quantities=[trade_qty],
            dca_prices=[price],
            is_hold=False
        )

    def _execute_dca(self, symbol: str, price: float, timestamp, dca_level: int):
        """Execute DCA entry."""
        if symbol not in self.trade_positions:
            return

        trade = self.trade_positions[symbol]
        dca_qty = self.config.get_trade_quantity(symbol, price, dca_level=dca_level)

        # Deduct commission
        commission = dca_qty * price * self.commission_pct
        self.stats['fees_paid'] += commission

        # Update position
        trade.dca_level = dca_level
        trade.dca_quantities.append(dca_qty)
        trade.dca_prices.append(price)
        trade.quantity += dca_qty
        trade.last_dca_time = timestamp

        # Recalculate average entry
        total_cost = sum(q * p for q, p in zip(trade.dca_quantities, trade.dca_prices))
        trade.avg_entry_price = total_cost / trade.quantity

        self.stats['dca_entries'] += 1

    def _generate_report(self) -> Dict:
        """Generate backtest report."""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        total_return = (self.stats['final_equity'] - self.stats['initial_capital']) / self.stats['initial_capital'] * 100
        win_rate = self.stats['winning_trades'] / max(1, self.stats['total_trades']) * 100

        print(f"\nCapital: ${self.stats['initial_capital']:,.2f} -> ${self.stats['final_equity']:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Max Drawdown: {self.stats['max_drawdown']*100:.2f}%")

        print(f"\nTrades:")
        print(f"  Total: {self.stats['total_trades']}")
        print(f"  Winners: {self.stats['winning_trades']} ({win_rate:.1f}%)")
        print(f"  Losers: {self.stats['losing_trades']}")
        print(f"  TP Hits: {self.stats['tp_hits']}")
        print(f"  DCA Entries: {self.stats['dca_entries']}")

        print(f"\nP&L Breakdown:")
        print(f"  HOLD P&L: ${self.stats['hold_pnl']:,.2f}")
        print(f"  TRADE P&L: ${self.stats['trade_pnl']:,.2f}")
        print(f"  Fees Paid: ${self.stats['fees_paid']:,.2f}")
        print(f"  Net P&L: ${self.stats['total_pnl']:,.2f}")

        # Trade details
        if self.trades:
            print(f"\nRecent Trades:")
            for trade in self.trades[-10:]:
                print(f"  {trade.symbol}: {trade.pnl_pct:+.2f}% (${trade.pnl:+.2f}) DCA{trade.dca_level} [{trade.exit_reason}]")

        return self.stats


def load_local_csv_data(symbols: List[str], data_dir: str = None) -> Dict[str, pd.DataFrame]:
    """Load historical crypto data from local CSV files."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "Crypto_Data_Fresh"
    else:
        data_dir = Path(data_dir)

    print(f"Loading data from: {data_dir}")
    data = {}

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_1m.csv"

        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found")
            continue

        try:
            df = pd.read_csv(csv_path)

            # Handle different column naming conventions
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Timestamp': 'timestamp',
                'Date': 'timestamp',
                'time': 'timestamp'
            }

            df.rename(columns=column_mapping, inplace=True)

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"  WARNING: {symbol} missing columns: {missing_cols}")
                continue

            # Convert price columns to float, handling any NaN
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaN in required columns
            df = df.dropna(subset=required_cols)

            # Set timestamp as index - use datetime column if available
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'timestamp' in df.columns:
                # Check if timestamp is Unix ms
                if df['timestamp'].iloc[0] > 1e12:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif df.index.name is None:
                # Create a timestamp index
                df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')

            # Sort by index to ensure chronological order
            df = df.sort_index()

            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars, ${df['close'].iloc[0]:,.2f} -> ${df['close'].iloc[-1]:,.2f}")

        except Exception as e:
            print(f"  ERROR loading {symbol}: {e}")
            import traceback
            traceback.print_exc()

    return data


def fetch_crypto_data(symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
    """Fetch historical crypto data from Alpaca (fallback if no local data)."""
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key = os.getenv('ALPACA_CRYPTO_KEY') or os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_CRYPTO_SECRET') or os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not api_secret:
            print("ERROR: Alpaca credentials not found")
            return {}

        client = CryptoHistoricalDataClient(api_key, api_secret)
        data = {}

        end = datetime.now()
        start = end - timedelta(days=days)

        for symbol in symbols:
            alpaca_symbol = symbol.replace('USD', '/USD')
            print(f"Fetching {symbol}...")

            try:
                bars = client.get_crypto_bars(
                    CryptoBarsRequest(
                        symbol_or_symbols=alpaca_symbol,
                        timeframe=TimeFrame.Minute,
                        start=start,
                        end=end
                    )
                )

                if bars and alpaca_symbol in bars:
                    records = []
                    for bar in bars[alpaca_symbol]:
                        records.append({
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': float(bar.volume)
                        })

                    df = pd.DataFrame(records)
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
                    print(f"  {len(df)} bars loaded")

            except Exception as e:
                print(f"  Error: {e}")

        return data

    except ImportError:
        print("ERROR: Alpaca SDK not installed")
        return {}


def generate_synthetic_data(symbols: List[str], bars: int = 10000) -> Dict[str, pd.DataFrame]:
    """Generate synthetic crypto data for testing."""
    print("Generating synthetic data for testing...")
    np.random.seed(42)

    base_prices = {
        'BTCUSD': 98000,
        'ETHUSD': 3400,
        'SOLUSD': 190
    }

    data = {}

    for symbol in symbols:
        base_price = base_prices.get(symbol, 1000)
        volatility = 0.0015  # 0.15% per bar typical crypto volatility

        prices = [base_price]
        for i in range(1, bars):
            # Random walk with slight upward drift
            change = np.random.normal(0.00002, volatility)
            # Add some trending behavior
            if i % 500 < 250:
                change += 0.00005  # Uptrend
            else:
                change -= 0.00003  # Downtrend

            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # Create OHLCV data
        records = []
        start_time = datetime.now() - timedelta(minutes=bars)

        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, volatility/2)))
            low = close * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(100, 1000) * base_price / 100

            records.append({
                'timestamp': start_time + timedelta(minutes=i),
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        data[symbol] = df
        print(f"  {symbol}: {len(df)} bars, ${df['close'].iloc[0]:,.2f} -> ${df['close'].iloc[-1]:,.2f}")

    return data


def main():
    """Run backtest."""
    from dotenv import load_dotenv
    load_dotenv()

    config = load_always_hold_config()
    print(config)

    # Try to load local CSV data first
    print("\nLoading historical data from local CSV files...")
    data = load_local_csv_data(config.symbols, config.data_dir)

    # If no local data, try Alpaca API
    if not data:
        print("\nNo local data, trying Alpaca API...")
        data = fetch_crypto_data(config.symbols, days=7)

    # If still no data, use synthetic
    if not data:
        print("\nNo real data available, using synthetic data...")
        data = generate_synthetic_data(config.symbols, bars=10000)  # ~7 days of 1-min data

    if not data:
        print("ERROR: No data available")
        return

    # Run backtest
    backtest = AlwaysHoldBacktest(config)
    results = backtest.run_backtest(data)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
