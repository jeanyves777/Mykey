"""
Run Always Hold Trading Strategy - Comprehensive Logging
=========================================================

ALWAYS HOLD + HYBRID DCA Strategy:
- Maintains permanent HOLD position (never closed)
- TRADE position uses DCA and closes at TP
- When TP hit: close TRADE, keep HOLD, re-enter new TRADE
- Hybrid DCA: Easy filters for DCA 1-2, Strict for DCA 3-4

Usage:
    python -m trading_system.High_frequency_crypto_tradin.run_always_hold
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from trading_system.High_frequency_crypto_tradin.dca_config import load_always_hold_config
from trading_system.High_frequency_crypto_tradin.always_hold_engine import AlwaysHoldEngine

# Try to import Alpaca for data
try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI from price series."""
    if len(prices) < period + 1:
        return 50.0

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX from OHLC dataframe."""
    if len(df) < period + 1:
        return 20.0

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
    adx = dx.rolling(period).mean()

    return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 20.0


def is_reversal_candle(df: pd.DataFrame, direction: str = 'bullish') -> bool:
    """Check if last candle is a reversal pattern."""
    if len(df) < 2:
        return False

    current = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(current['close'] - current['open'])
    candle_range = current['high'] - current['low']

    if candle_range == 0:
        return False

    body_ratio = body / candle_range

    if direction == 'bullish':
        is_bullish = current['close'] > current['open']
        return is_bullish and body_ratio >= 0.5
    else:
        is_bearish = current['close'] < current['open']
        return is_bearish and body_ratio >= 0.5


def get_volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """Get current volume relative to average."""
    if len(df) < period:
        return 1.0

    avg_volume = df['volume'].tail(period).mean()
    current_volume = df['volume'].iloc[-1]

    if avg_volume == 0:
        return 1.0

    return current_volume / avg_volume


def get_market_data(data_client, symbol: str, limit: int = 100) -> pd.DataFrame:
    """Get recent market data for a symbol using multiple methods."""
    alpaca_symbol = symbol.replace('USD', '/USD')

    # Method 1: Try to get bars with explicit start/end
    try:
        from datetime import timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=limit + 50)

        bars = data_client.get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )
        )

        if bars and alpaca_symbol in bars and len(bars[alpaca_symbol]) > 0:
            data = []
            for bar in bars[alpaca_symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume)
                })
            return pd.DataFrame(data)

    except Exception as e:
        pass  # Try next method

    # Method 2: Try with limit only (no start/end)
    try:
        bars = data_client.get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=TimeFrame.Minute,
                limit=limit
            )
        )

        if bars and alpaca_symbol in bars and len(bars[alpaca_symbol]) > 0:
            data = []
            for bar in bars[alpaca_symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume)
                })
            return pd.DataFrame(data)

    except Exception as e:
        pass  # Try next method

    # Method 3: Use latest quote to create minimal data for indicators
    try:
        from alpaca.data.requests import CryptoLatestQuoteRequest

        quotes = data_client.get_crypto_latest_quote(
            CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbol)
        )

        if quotes and alpaca_symbol in quotes:
            quote = quotes[alpaca_symbol]
            mid_price = (float(quote.bid_price) + float(quote.ask_price)) / 2

            # Create minimal dataframe with current price
            # Use neutral RSI/ADX values since we don't have history
            now = datetime.now()
            data = []
            for i in range(30):  # Create 30 bars of "fake" history at current price
                data.append({
                    'timestamp': now - timedelta(minutes=30-i),
                    'open': mid_price,
                    'high': mid_price * 1.001,
                    'low': mid_price * 0.999,
                    'close': mid_price,
                    'volume': 100.0
                })
            return pd.DataFrame(data)

    except Exception as e:
        pass

    return pd.DataFrame()


class AlwaysHoldRunner:
    """Runner with comprehensive logging for Always Hold strategy."""

    def __init__(self, config, engine, data_client):
        self.config = config
        self.engine = engine
        self.data_client = data_client

        # Session tracking
        self.session_start = datetime.now()
        self.iteration = 0

        # Session stats
        self.stats = {
            'signals_detected': 0,
            'entries': 0,
            'entries_blocked': 0,
            'exits_tp': 0,
            'exits_sl': 0,
            'exits_manual': 0,
            'dca_entries': 0,
            'dca_blocked': 0,
            'total_pnl': 0.0
        }

    def get_account_equity(self) -> float:
        """Get current account equity."""
        if not self.engine.trading_client:
            return self.config.initial_capital
        try:
            account = self.engine.trading_client.get_account()
            return float(account.equity)
        except:
            return self.config.initial_capital

    def display_comprehensive_status(self):
        """Display comprehensive status header."""
        now = datetime.now()
        equity = self.get_account_equity()

        # Get actual positions from Alpaca
        try:
            positions = self.engine.trading_client.get_all_positions()
            total_positions = len(positions)
        except:
            total_positions = 0

        print("\n" + "=" * 70)
        print(f"[{now.strftime('%H:%M:%S')}] ALWAYS HOLD SCANNING {len(self.config.symbols)} PAIRS | "
              f"Equity: ${equity:,.2f} | Positions: {total_positions}")
        print("=" * 70)

    def display_position_status(self):
        """Display detailed position status from Alpaca directly."""
        try:
            positions = self.engine.trading_client.get_all_positions()
        except Exception as e:
            print(f"\nError fetching positions: {e}")
            return

        if not positions:
            print("\nNo open positions on Alpaca")
            return

        total_pnl = 0.0

        print(f"\n--- ALPACA POSITIONS ({len(positions)}) ---")

        for pos in positions:
            symbol = pos.symbol.replace('/', '')  # Convert BTC/USD to BTCUSD
            qty = float(pos.qty)
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            market_value = float(pos.market_value)
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100

            total_pnl += pnl

            # Get DCA level from local tracking (if available)
            dca_level = 0
            if symbol in self.engine.trade_positions:
                dca_level = self.engine.trade_positions[symbol].dca_level

            # Calculate TP and DCA targets
            tp_pct = self.config.take_profit_pct - (dca_level * self.config.tp_reduction_per_dca)
            tp_pct = max(tp_pct, self.config.min_tp_pct)
            tp_price = entry_price * (1 + tp_pct)
            tp_away = tp_price - current_price
            tp_away_pct = (tp_away / current_price) * 100 if current_price > 0 else 0

            # Next DCA level
            dca_levels = self.config.get_dca_levels(entry_price)
            next_dca_price = dca_levels[dca_level] if dca_level < len(dca_levels) else 0
            filter_type = "EASY" if self.config.is_easy_dca(dca_level + 1) else "STRICT"

            print(f"  {symbol}: {qty:.6f} @ ${entry_price:,.2f}")
            print(f"    Current: ${current_price:,.2f} | Value: ${market_value:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            print(f"    TP: ${tp_price:,.2f} ({tp_pct*100:.2f}%) (+${tp_away:,.2f} / {tp_away_pct:+.2f}% away)")
            print(f"    DCA Level: {dca_level}/{self.config.max_dca_stages} | Filter: {filter_type}")

            if next_dca_price > 0 and dca_level < self.config.max_dca_stages:
                dca_away = current_price - next_dca_price
                print(f"    Next DCA: ${next_dca_price:,.2f} (${dca_away:,.2f} away)")

        print(f"\n  TOTAL Unrealized P&L: ${total_pnl:,.2f}")

    def display_session_stats(self):
        """Display session statistics."""
        runtime = datetime.now() - self.session_start
        engine_stats = self.engine.stats

        print(f"\n--- SESSION STATS (Runtime: {runtime}) ---")
        print(f"  TP Hits: {engine_stats['tp_hits']}")
        print(f"  DCA Entries: {engine_stats['dca_entries']}")
        print(f"  Trades Closed: {engine_stats['trades_closed']}")
        print(f"  Realized TRADE P&L: ${engine_stats['trade_pnl']:,.2f}")

    def _handle_shutdown(self):
        """Handle shutdown - ask user about positions."""
        if not self.engine.hold_positions and not self.engine.trade_positions:
            print("\nNo open positions. Clean shutdown.")
            return

        print("\n" + "=" * 70)
        print("SHUTDOWN OPTIONS - OPEN POSITIONS")
        print("=" * 70)

        # Show current positions summary
        total_hold_pnl = 0.0
        total_trade_pnl = 0.0

        for symbol, hold in self.engine.hold_positions.items():
            if hold.current_price > 0:
                pnl = (hold.current_price - hold.entry_price) * hold.quantity
                total_hold_pnl += pnl
                print(f"  HOLD {symbol}: {hold.quantity:.6f} @ ${hold.entry_price:,.2f} | P&L: ${pnl:,.2f}")

        for symbol, trade in self.engine.trade_positions.items():
            if trade.current_price > 0:
                pnl = (trade.current_price - trade.avg_entry_price) * trade.total_quantity
                total_trade_pnl += pnl
                print(f"  TRADE {symbol}: {trade.total_quantity:.6f} @ ${trade.avg_entry_price:,.2f} | "
                      f"P&L: ${pnl:,.2f} | DCA: {trade.dca_level}")

        print(f"\n  HOLD Unrealized: ${total_hold_pnl:,.2f}")
        print(f"  TRADE Unrealized: ${total_trade_pnl:,.2f}")
        print(f"  TOTAL Unrealized: ${total_hold_pnl + total_trade_pnl:,.2f}")

        print("\nWhat would you like to do?")
        print("  [1] LEAVE OPEN - Keep all positions open (recommended for Always Hold)")
        print("  [2] CLOSE ALL - Close ALL positions (including HOLD)")
        print("  [3] CLOSE TRADE ONLY - Close TRADE positions, keep HOLD")
        print("  [4] CLOSE LOSERS - Close only losing TRADE positions")

        try:
            choice = input("\nEnter choice (1-4) [default: 1]: ").strip()

            if choice == '2':
                self._close_all_positions()
            elif choice == '3':
                self._close_trade_positions()
            elif choice == '4':
                self._close_losing_trade_positions()
            else:
                print("\nLeaving all positions open.")
                print("Positions will remain active on your Alpaca account.")
                print("Run this script again to resume monitoring.")

        except (KeyboardInterrupt, EOFError):
            print("\nLeaving positions open.")

    def _close_all_positions(self):
        """Close all positions including HOLD."""
        if not self.engine.trading_client:
            return

        print("\nClosing ALL positions (including HOLD)...")

        try:
            # Use Alpaca's close all positions
            self.engine.trading_client.close_all_positions()
            print("All positions closed.")

            self.engine.hold_positions.clear()
            self.engine.trade_positions.clear()

        except Exception as e:
            print(f"Error closing positions: {e}")

    def _close_trade_positions(self):
        """Close only TRADE positions, keep HOLD."""
        if not self.engine.trading_client:
            return

        print("\nClosing TRADE positions only (keeping HOLD)...")

        for symbol in list(self.engine.trade_positions.keys()):
            try:
                trade = self.engine.trade_positions[symbol]
                alpaca_symbol = symbol.replace('USD', '/USD')

                # Get actual position from Alpaca
                actual_pos = self.engine.trading_client.get_open_position(alpaca_symbol)
                total_qty = float(actual_pos.qty)

                # Calculate TRADE qty (total - HOLD)
                hold_qty = self.engine.hold_positions[symbol].quantity if symbol in self.engine.hold_positions else 0
                sell_qty = total_qty - hold_qty

                if sell_qty > 0:
                    from alpaca.trading.requests import MarketOrderRequest
                    from alpaca.trading.enums import OrderSide, TimeInForce

                    order_data = MarketOrderRequest(
                        symbol=alpaca_symbol,
                        qty=sell_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )

                    order = self.engine.trading_client.submit_order(order_data)
                    time.sleep(1)

                    filled_order = self.engine.trading_client.get_order_by_id(order.id)
                    if filled_order.filled_avg_price:
                        exit_price = float(filled_order.filled_avg_price)
                        pnl = sell_qty * (exit_price - trade.avg_entry_price)
                        print(f"  {symbol}: TRADE CLOSED @ ${exit_price:,.2f} | P&L: ${pnl:,.2f}")

                del self.engine.trade_positions[symbol]

            except Exception as e:
                print(f"  {symbol}: Error closing TRADE: {e}")

        print("\nTRADE positions closed. HOLD positions remain active.")

    def _close_losing_trade_positions(self):
        """Close only losing TRADE positions."""
        if not self.engine.trading_client:
            return

        print("\nClosing losing TRADE positions...")

        for symbol in list(self.engine.trade_positions.keys()):
            trade = self.engine.trade_positions[symbol]

            if trade.current_price <= 0:
                continue

            pnl = (trade.current_price - trade.avg_entry_price) * trade.total_quantity

            if pnl < 0:
                try:
                    alpaca_symbol = symbol.replace('USD', '/USD')

                    # Get actual position
                    actual_pos = self.engine.trading_client.get_open_position(alpaca_symbol)
                    total_qty = float(actual_pos.qty)

                    # Calculate TRADE qty
                    hold_qty = self.engine.hold_positions[symbol].quantity if symbol in self.engine.hold_positions else 0
                    sell_qty = total_qty - hold_qty

                    if sell_qty > 0:
                        from alpaca.trading.requests import MarketOrderRequest
                        from alpaca.trading.enums import OrderSide, TimeInForce

                        order_data = MarketOrderRequest(
                            symbol=alpaca_symbol,
                            qty=sell_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )

                        order = self.engine.trading_client.submit_order(order_data)
                        time.sleep(1)

                        filled_order = self.engine.trading_client.get_order_by_id(order.id)
                        if filled_order.filled_avg_price:
                            exit_price = float(filled_order.filled_avg_price)
                            actual_pnl = sell_qty * (exit_price - trade.avg_entry_price)
                            print(f"  {symbol}: LOSER CLOSED @ ${exit_price:,.2f} | P&L: ${actual_pnl:,.2f}")

                    del self.engine.trade_positions[symbol]

                except Exception as e:
                    print(f"  {symbol}: Error closing: {e}")
            else:
                print(f"  {symbol}: Keeping (in profit: ${pnl:,.2f})")

    def run(self, interval_seconds: int = 30, max_hours: float = 24):
        """Run Always Hold trading loop with comprehensive logging."""
        print("\n" + "=" * 70)
        print("STARTING ALWAYS HOLD TRADING")
        print("=" * 70)
        print(f"Symbols: {self.config.symbols}")
        print(f"Scan Interval: {interval_seconds} seconds")
        print(f"Max Runtime: {max_hours} hours")

        print(f"\nStrategy Settings:")
        print(f"  HOLD Position: {self.config.hold_position_pct*100:.0f}% of allocation (NEVER closed)")
        print(f"  TRADE Position: {self.config.trade_position_pct*100:.0f}% of allocation (uses DCA)")
        print(f"  TP: {self.config.take_profit_pct*100:.2f}% (net ~{(self.config.take_profit_pct - 0.003)*100:.2f}% after fees)")
        print(f"  DCA Spacing: {self.config.dca_spacing_pct*100:.2f}%")
        print(f"  Max DCA Stages: {self.config.max_dca_stages}")

        print(f"\nHybrid DCA Filters:")
        print(f"  Easy DCA (1-2): RSI < {self.config.easy_dca_rsi_threshold}, ADX < {self.config.easy_dca_adx_max}")
        print(f"  Strict DCA (3-4): RSI < {self.config.strict_dca_rsi_threshold}, ADX < {self.config.strict_dca_adx_max}")
        print(f"  Strict requires: Reversal candle = {self.config.require_reversal_candle}, Volume spike = {self.config.require_volume_spike}")

        print(f"\nSmart Exit Settings:")
        print(f"  Dynamic TP: -{self.config.tp_reduction_per_dca*100:.1f}% per DCA (min {self.config.min_tp_pct*100:.2f}%)")
        print(f"  Breakeven Lock: DCA {self.config.breakeven_dca_threshold}+ at +{self.config.breakeven_buffer_pct*100:.1f}%")
        print(f"  Emergency Exit: DCA {self.config.emergency_exit_dca_level}+ at +{self.config.emergency_exit_profit_pct*100:.2f}%")

        print(f"\nPosition Sizing (${self.config.allocation_per_symbol:,.2f} per symbol):")
        print(f"  HOLD: ${self.config.allocation_per_symbol * self.config.hold_position_pct:,.2f}")
        print(f"  TRADE: ${self.config.allocation_per_symbol * self.config.trade_position_pct:,.2f}")
        print(f"  DCA Multipliers: {self.config.dca_multipliers}")

        max_runtime = timedelta(hours=max_hours)

        try:
            while datetime.now() - self.session_start < max_runtime:
                self.iteration += 1

                # Display header
                self.display_comprehensive_status()

                # Get all positions from Alpaca for current prices
                alpaca_positions = {}
                try:
                    positions = self.engine.trading_client.get_all_positions()
                    for pos in positions:
                        sym = pos.symbol.replace('/', '')
                        alpaca_positions[sym] = {
                            'price': float(pos.current_price),
                            'qty': float(pos.qty),
                            'entry': float(pos.avg_entry_price),
                            'pnl': float(pos.unrealized_pl),
                            'pnl_pct': float(pos.unrealized_plpc) * 100
                        }
                except Exception as e:
                    print(f"  Error fetching positions: {e}")

                for symbol in self.config.symbols:
                    try:
                        # Get current price from Alpaca position first
                        current_price = 0.0
                        if symbol in alpaca_positions:
                            current_price = alpaca_positions[symbol]['price']

                        # If no position, try market data
                        if current_price == 0:
                            df = get_market_data(self.data_client, symbol)
                            if not df.empty:
                                current_price = float(df['close'].iloc[-1])

                        if current_price == 0:
                            # Last resort: get quote
                            try:
                                from alpaca.data.requests import CryptoLatestQuoteRequest
                                alpaca_symbol = symbol.replace('USD', '/USD')
                                quotes = self.data_client.get_crypto_latest_quote(
                                    CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbol)
                                )
                                if quotes and alpaca_symbol in quotes:
                                    q = quotes[alpaca_symbol]
                                    current_price = (float(q.bid_price) + float(q.ask_price)) / 2
                            except:
                                pass

                        if current_price == 0:
                            print(f"  {symbol}: No price available")
                            continue

                        # Get market data for indicators (ok if empty, use defaults)
                        df = get_market_data(self.data_client, symbol)

                        # Calculate indicators (use neutral defaults if no data)
                        if df.empty or len(df) < 15:
                            rsi = 50.0  # Neutral RSI
                            adx = 20.0  # Neutral ADX
                            reversal = False
                            volume_ratio = 1.0
                        else:
                            rsi = calculate_rsi(df['close'], self.config.rsi_period)
                            adx = calculate_adx(df, self.config.adx_period)
                            reversal = is_reversal_candle(df, 'bullish')
                            volume_ratio = get_volume_ratio(df)

                        # Update HOLD position prices
                        if symbol in self.engine.hold_positions:
                            hold = self.engine.hold_positions[symbol]
                            hold.current_price = current_price
                            hold.unrealized_pnl = (current_price - hold.entry_price) * hold.quantity

                        # Process TRADE position
                        if symbol in self.engine.trade_positions:
                            trade = self.engine.trade_positions[symbol]
                            trade.current_price = current_price
                            trade.unrealized_pnl = (current_price - trade.avg_entry_price) * trade.total_quantity

                            pnl = trade.unrealized_pnl
                            pnl_pct = (current_price / trade.avg_entry_price - 1) * 100

                            # Check TP
                            should_tp, profit_pct = self.engine.check_tp_conditions(symbol, current_price)
                            if should_tp:
                                print(f"  {symbol}: TRADE POSITION - TP HIT! ({profit_pct*100:+.2f}%)")
                                self.engine.close_trade_position(symbol, current_price)
                                continue

                            # Check DCA
                            should_dca, reason = self.engine.check_dca_conditions(
                                symbol, current_price, rsi, adx, reversal, volume_ratio
                            )

                            if should_dca:
                                print(f"  {symbol}: TRADE POSITION - DCA TRIGGER!")
                                print(f"    Reason: {reason}")
                                self.engine.execute_dca(symbol, current_price)
                            else:
                                # Normal status
                                pnl_str = f"${pnl:+,.2f}" if pnl >= 0 else f"${pnl:,.2f}"
                                filter_type = "EASY" if self.config.is_easy_dca(trade.dca_level + 1) else "STRICT"
                                print(f"  {symbol}: TRADE @ ${trade.avg_entry_price:,.2f} | "
                                      f"P&L: {pnl_str} ({pnl_pct:+.2f}%) | DCA: {trade.dca_level} ({filter_type})")

                                # Show why DCA not triggered
                                if trade.dca_level < self.config.max_dca_stages:
                                    print(f"    RSI: {rsi:.1f} | ADX: {adx:.1f} | Vol: {volume_ratio:.1f}x | {reason}")

                        else:
                            # No TRADE position - show status
                            print(f"  {symbol}: @ ${current_price:,.2f} | RSI: {rsi:.1f} | ADX: {adx:.1f}")

                    except Exception as e:
                        print(f"  {symbol}: Error - {e}")

                # Display detailed position status
                self.display_position_status()

                # Display session stats every 5 iterations
                if self.iteration % 5 == 0:
                    self.display_session_stats()

                print(f"\nNext scan in {interval_seconds} seconds... (Ctrl+C to stop)")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("STOPPED BY USER")
            print("=" * 70)

        finally:
            self.display_position_status()
            self.display_session_stats()

            # Handle shutdown
            self._handle_shutdown()


def main():
    """Main entry point."""
    print("=" * 70)
    print("ALWAYS HOLD + HYBRID DCA TRADING")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load configuration
    config = load_always_hold_config()
    print(config)

    # Initialize engine
    engine = AlwaysHoldEngine(config)

    if not engine.trading_client:
        print("\nERROR: Could not connect to Alpaca. Check credentials.")
        print("\nRequired environment variables:")
        print("  ALPACA_CRYPTO_KEY=your_key")
        print("  ALPACA_CRYPTO_SECRET=your_secret")
        return

    # Initialize data client
    api_key = os.getenv('ALPACA_CRYPTO_KEY') or os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_CRYPTO_SECRET') or os.getenv('ALPACA_SECRET_KEY')
    data_client = CryptoHistoricalDataClient(api_key, api_secret)

    # Initialize positions
    print("\nInitializing positions...")
    engine.initialize_positions()

    # Show position sizing breakdown
    print(f"\nPosition Sizing per Symbol (${config.allocation_per_symbol:,.2f}):")
    print(f"  HOLD: ${config.allocation_per_symbol * config.hold_position_pct:,.2f} (never closed)")
    print(f"  TRADE: ${config.allocation_per_symbol * config.trade_position_pct:,.2f} (uses DCA)")
    print(f"\n  DCA Breakdown (TRADE portion):")
    trade_allocation = config.allocation_per_symbol * config.trade_position_pct
    for i, mult in enumerate(config.dca_multipliers):
        stage = "Entry" if i == 0 else f"DCA {i}"
        val = trade_allocation * mult
        print(f"    {stage}: ${val:,.2f} ({mult*100:.0f}% mult)")

    # Show resumed positions
    if engine.hold_positions or engine.trade_positions:
        print(f"\n*** POSITIONS ACTIVE ***")
        for symbol in config.symbols:
            if symbol in engine.hold_positions:
                hold = engine.hold_positions[symbol]
                print(f"  HOLD {symbol}: {hold.quantity:.6f} @ ${hold.entry_price:,.2f}")
            if symbol in engine.trade_positions:
                trade = engine.trade_positions[symbol]
                print(f"  TRADE {symbol}: {trade.total_quantity:.6f} @ ${trade.avg_entry_price:,.2f} | DCA: {trade.dca_level}")

    # Confirmation
    print("\n" + "=" * 70)
    print("Ready to start ALWAYS HOLD trading.")
    print("This will place REAL orders on your PAPER account.")
    print("Press Ctrl+C at any time to stop.")
    print("\nOn shutdown, you can choose to:")
    print("  - Leave all positions open (recommended)")
    print("  - Close all positions")
    print("  - Close only TRADE positions (keep HOLD)")
    print("=" * 70)

    try:
        input("\nPress Enter to start (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\nCancelled")
        return

    # Create runner and start
    runner = AlwaysHoldRunner(config, engine, data_client)
    runner.run(interval_seconds=30, max_hours=24)

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
