"""
Always Hold Trading Engine for Crypto DCA
==========================================

ALWAYS HOLD STRATEGY:
- Maintains a permanent HOLD position that is NEVER closed
- TRADE position uses DCA logic and closes at TP
- When TP hit: close TRADE positions only, immediately re-enter new TRADE
- No entry signals required - always in the market

HYBRID DCA FILTER:
- DCA 1-2: Easy triggers (just price drop + weak RSI filter)
- DCA 3-4: Strict triggers (require reversal confirmation)
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
import time

from dotenv import load_dotenv
load_dotenv()

from trading_system.High_frequency_crypto_tradin.dca_config import DCAConfig, load_always_hold_config

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("WARNING: Alpaca SDK not installed. Run: pip install alpaca-py")


@dataclass
class HoldPosition:
    """Tracks the permanent HOLD position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class TradePosition:
    """Tracks the TRADE position with DCA."""
    symbol: str
    entry_price: float           # Original entry price
    avg_entry_price: float       # Average entry after DCA
    total_quantity: float        # Total quantity including DCA
    entry_time: datetime
    dca_level: int = 0           # Current DCA level (0-4)
    last_dca_time: Optional[datetime] = None
    dca_quantities: List[float] = field(default_factory=list)
    dca_prices: List[float] = field(default_factory=list)
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    trailing_stop_price: Optional[float] = None
    breakeven_locked: bool = False


class AlwaysHoldEngine:
    """Trading engine for Always Hold + Hybrid DCA strategy."""

    def __init__(self, config: DCAConfig):
        self.config = config

        # Alpaca clients
        self.trading_client = None
        self.data_client = None

        # Position tracking
        self.hold_positions: Dict[str, HoldPosition] = {}  # NEVER closed
        self.trade_positions: Dict[str, TradePosition] = {}  # Uses DCA, closes at TP

        # Daily tracking
        self.daily_dca_count: Dict[str, int] = {}  # symbol -> count
        self.last_reset_date: str = ""

        # Session stats
        self.stats = {
            'session_start': datetime.now(),
            'hold_pnl': 0.0,
            'trade_pnl': 0.0,
            'trades_closed': 0,
            'dca_entries': 0,
            'tp_hits': 0
        }

        # Initialize Alpaca
        self._init_alpaca()

    def _init_alpaca(self):
        """Initialize Alpaca API clients."""
        if not ALPACA_AVAILABLE:
            print("ERROR: Alpaca SDK not available")
            return

        api_key = os.getenv('ALPACA_CRYPTO_KEY') or os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_CRYPTO_SECRET') or os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not api_secret:
            print("ERROR: Alpaca API credentials not found!")
            return

        try:
            self.trading_client = TradingClient(api_key, api_secret, paper=True)
            self.data_client = CryptoHistoricalDataClient(api_key, api_secret)

            account = self.trading_client.get_account()
            print(f"Connected to Alpaca Paper Trading")
            print(f"  Account: ${float(account.equity):,.2f}")
            print(f"  Buying Power: ${float(account.buying_power):,.2f}")

        except Exception as e:
            print(f"ERROR connecting to Alpaca: {e}")
            self.trading_client = None

    def initialize_positions(self):
        """
        Initialize HOLD and TRADE positions for all symbols.
        Called once at startup - enters both positions immediately.
        """
        if not self.trading_client:
            print("ERROR: Not connected to Alpaca")
            return False

        print("\n" + "=" * 70)
        print("INITIALIZING ALWAYS HOLD POSITIONS")
        print("=" * 70)

        for symbol in self.config.symbols:
            alpaca_symbol = symbol.replace('USD', '/USD')

            try:
                # Get current price
                price = self._get_current_price(symbol)
                if price <= 0:
                    print(f"ERROR: Could not get price for {symbol}")
                    continue

                print(f"\n{symbol} @ ${price:,.2f}")

                # Check for existing position
                existing_qty = self._get_existing_position(alpaca_symbol)

                if existing_qty > 0:
                    print(f"  Found existing position: {existing_qty:.6f}")
                    self._resume_existing_position(symbol, existing_qty, price)
                else:
                    # Enter new HOLD + TRADE positions
                    self._enter_initial_positions(symbol, price)

            except Exception as e:
                print(f"ERROR initializing {symbol}: {e}")

        return True

    def _enter_initial_positions(self, symbol: str, price: float):
        """Enter both HOLD and TRADE positions for a symbol."""
        alpaca_symbol = symbol.replace('USD', '/USD')

        # Calculate quantities
        hold_qty = self.config.get_hold_quantity(symbol, price)
        trade_qty = self.config.get_trade_quantity(symbol, price, dca_level=0)
        total_qty = hold_qty + trade_qty

        print(f"  Entering initial positions:")
        print(f"    HOLD: {hold_qty:.6f} (${hold_qty * price:.2f})")
        print(f"    TRADE: {trade_qty:.6f} (${trade_qty * price:.2f})")
        print(f"    TOTAL: {total_qty:.6f} (${total_qty * price:.2f})")

        # Execute single buy order for combined quantity
        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=total_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            result = self.trading_client.submit_order(order)

            # Wait for fill
            time.sleep(2)
            filled_order = self.trading_client.get_order_by_id(result.id)
            fill_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else price

            # Track HOLD position
            self.hold_positions[symbol] = HoldPosition(
                symbol=symbol,
                quantity=hold_qty,
                entry_price=fill_price,
                entry_time=datetime.now(),
                current_price=fill_price
            )

            # Track TRADE position
            self.trade_positions[symbol] = TradePosition(
                symbol=symbol,
                entry_price=fill_price,
                avg_entry_price=fill_price,
                total_quantity=trade_qty,
                entry_time=datetime.now(),
                dca_level=0,
                dca_quantities=[trade_qty],
                dca_prices=[fill_price],
                current_price=fill_price
            )

            print(f"  [OK] Positions opened at ${fill_price:,.2f}")

        except Exception as e:
            print(f"  [ERROR] Failed to open positions: {e}")

    def _resume_existing_position(self, symbol: str, existing_qty: float, current_price: float):
        """Resume tracking of existing position from Alpaca."""
        # Get actual entry price from Alpaca
        symbol_noslash = symbol.replace('USD', 'USD')  # Keep as-is for lookup
        alpaca_symbol = symbol.replace('USD', '/USD')

        entry_price = current_price  # Default fallback
        try:
            pos = self.trading_client.get_open_position(symbol)
            entry_price = float(pos.avg_entry_price)
            pnl = float(pos.unrealized_pl)
            print(f"  Found on Alpaca: {existing_qty:.6f} @ ${entry_price:,.2f} (P&L: ${pnl:,.2f})")
        except:
            try:
                pos = self.trading_client.get_open_position(alpaca_symbol)
                entry_price = float(pos.avg_entry_price)
            except:
                pass

        # Since HOLD is 0%, all existing quantity is TRADE
        hold_qty = 0.0  # No separate HOLD allocation
        trade_qty = existing_qty

        print(f"  Resuming as TRADE position:")
        print(f"    Quantity: {trade_qty:.6f}")
        print(f"    Entry: ${entry_price:,.2f}")
        print(f"    Current: ${current_price:,.2f}")

        # Track HOLD position (empty since hold_pct = 0)
        self.hold_positions[symbol] = HoldPosition(
            symbol=symbol,
            quantity=0.0,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=current_price
        )

        # Track TRADE position (entire position)
        if trade_qty > 0:
            self.trade_positions[symbol] = TradePosition(
                symbol=symbol,
                entry_price=entry_price,
                avg_entry_price=entry_price,
                total_quantity=trade_qty,
                entry_time=datetime.now(),
                dca_level=0,  # Assume DCA 0 on resume
                dca_quantities=[trade_qty],
                dca_prices=[entry_price],
                current_price=current_price
            )

    def check_dca_conditions(self, symbol: str, current_price: float,
                             rsi: float, adx: float,
                             is_reversal_candle: bool, volume_ratio: float) -> Tuple[bool, str]:
        """
        Check if DCA should trigger using Hybrid filter.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            rsi: Current RSI value
            adx: Current ADX value
            is_reversal_candle: Whether current candle is reversal pattern
            volume_ratio: Current volume / average volume

        Returns:
            Tuple of (should_dca, reason)
        """
        if symbol not in self.trade_positions:
            return False, "No trade position"

        pos = self.trade_positions[symbol]

        # Check max DCA level
        if pos.dca_level >= self.config.max_dca_stages:
            return False, "Max DCA reached"

        # Check daily DCA limit
        self._reset_daily_counts_if_needed()
        if self.daily_dca_count.get(symbol, 0) >= self.config.max_dca_per_day:
            return False, "Daily DCA limit reached"

        # Check cooldown
        next_dca_level = pos.dca_level + 1
        cooldown_mins = self.config.get_dca_cooldown(next_dca_level)
        if pos.last_dca_time:
            elapsed = (datetime.now() - pos.last_dca_time).total_seconds() / 60
            if elapsed < cooldown_mins:
                return False, f"Cooldown: {cooldown_mins - elapsed:.0f}m remaining"

        # Check price drop
        dca_levels = self.config.get_dca_levels(pos.entry_price)
        target_price = dca_levels[pos.dca_level] if pos.dca_level < len(dca_levels) else 0

        if current_price > target_price:
            drop_needed = (pos.entry_price - target_price) / pos.entry_price * 100
            current_drop = (pos.entry_price - current_price) / pos.entry_price * 100
            return False, f"Need {drop_needed:.2f}% drop, only {current_drop:.2f}%"

        # Apply Hybrid filter
        if self.config.is_easy_dca(next_dca_level):
            # EASY DCA (1-2): Just RSI + ADX check
            if rsi > self.config.easy_dca_rsi_threshold:
                return False, f"RSI {rsi:.1f} > {self.config.easy_dca_rsi_threshold} (easy)"
            if adx > self.config.easy_dca_adx_max:
                return False, f"ADX {adx:.1f} > {self.config.easy_dca_adx_max} (trend too strong)"
            return True, f"EASY DCA {next_dca_level}: RSI={rsi:.1f}, ADX={adx:.1f}"

        else:
            # STRICT DCA (3-4): Full confirmation required
            if rsi > self.config.strict_dca_rsi_threshold:
                return False, f"RSI {rsi:.1f} > {self.config.strict_dca_rsi_threshold} (strict)"
            if adx > self.config.strict_dca_adx_max:
                return False, f"ADX {adx:.1f} > {self.config.strict_dca_adx_max} (strict)"
            if self.config.require_reversal_candle and not is_reversal_candle:
                return False, "No reversal candle (strict)"
            if self.config.require_volume_spike and volume_ratio < self.config.volume_spike_multiplier:
                return False, f"Volume {volume_ratio:.1f}x < {self.config.volume_spike_multiplier}x (strict)"
            return True, f"STRICT DCA {next_dca_level}: RSI={rsi:.1f}, ADX={adx:.1f}, reversal confirmed"

    def execute_dca(self, symbol: str, current_price: float) -> bool:
        """Execute DCA entry for TRADE position."""
        if symbol not in self.trade_positions:
            return False

        pos = self.trade_positions[symbol]
        next_level = pos.dca_level + 1

        # Calculate DCA quantity
        dca_qty = self.config.get_trade_quantity(symbol, current_price, dca_level=next_level)

        alpaca_symbol = symbol.replace('USD', '/USD')

        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=dca_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            result = self.trading_client.submit_order(order)

            # Wait for fill
            time.sleep(1)
            filled_order = self.trading_client.get_order_by_id(result.id)
            fill_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else current_price

            # Update position
            pos.dca_level = next_level
            pos.dca_quantities.append(dca_qty)
            pos.dca_prices.append(fill_price)
            pos.total_quantity += dca_qty
            pos.last_dca_time = datetime.now()

            # Recalculate average entry
            total_cost = sum(q * p for q, p in zip(pos.dca_quantities, pos.dca_prices))
            pos.avg_entry_price = total_cost / pos.total_quantity

            # Update daily count
            self.daily_dca_count[symbol] = self.daily_dca_count.get(symbol, 0) + 1
            self.stats['dca_entries'] += 1

            print(f"  [DCA {next_level}] {symbol}: +{dca_qty:.6f} @ ${fill_price:,.2f}")
            print(f"           Total: {pos.total_quantity:.6f}, Avg: ${pos.avg_entry_price:,.2f}")

            return True

        except Exception as e:
            print(f"  [ERROR] DCA failed for {symbol}: {e}")
            return False

    def check_tp_conditions(self, symbol: str, current_price: float) -> Tuple[bool, float]:
        """
        Check if TRADE position should close at TP.

        Returns:
            Tuple of (should_close, profit_pct)
        """
        if symbol not in self.trade_positions:
            return False, 0.0

        pos = self.trade_positions[symbol]

        # Calculate profit from average entry
        profit_pct = (current_price - pos.avg_entry_price) / pos.avg_entry_price

        # Dynamic TP based on DCA level
        tp_pct = self.config.take_profit_pct - (pos.dca_level * self.config.tp_reduction_per_dca)
        tp_pct = max(tp_pct, self.config.min_tp_pct)

        # Emergency exit for DCA 4
        if pos.dca_level >= self.config.emergency_exit_dca_level:
            if profit_pct >= self.config.emergency_exit_profit_pct:
                return True, profit_pct

        # Normal TP check
        if profit_pct >= tp_pct:
            return True, profit_pct

        return False, profit_pct

    def close_trade_position(self, symbol: str, current_price: float) -> bool:
        """
        Close TRADE position (HOLD stays open).
        Then immediately re-enter new TRADE position.
        """
        if symbol not in self.trade_positions:
            return False

        pos = self.trade_positions[symbol]
        alpaca_symbol = symbol.replace('USD', '/USD')

        try:
            # Get actual quantity from Alpaca
            actual_pos = self.trading_client.get_open_position(alpaca_symbol)
            total_qty = float(actual_pos.qty)

            # Calculate how much to sell (total - HOLD)
            hold_qty = self.hold_positions[symbol].quantity if symbol in self.hold_positions else 0
            sell_qty = total_qty - hold_qty

            if sell_qty <= 0:
                print(f"  [WARN] No TRADE quantity to sell for {symbol}")
                return False

            # Calculate profit
            profit = (current_price - pos.avg_entry_price) * sell_qty
            profit_pct = (current_price - pos.avg_entry_price) / pos.avg_entry_price * 100

            print(f"\n  [TP HIT] {symbol}")
            print(f"    Selling TRADE: {sell_qty:.6f} @ ${current_price:,.2f}")
            print(f"    Profit: ${profit:,.2f} ({profit_pct:.2f}%)")
            print(f"    HOLD remains: {hold_qty:.6f}")

            # Sell TRADE position
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=sell_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            self.trading_client.submit_order(order)
            time.sleep(1)

            # Update stats
            self.stats['trade_pnl'] += profit
            self.stats['trades_closed'] += 1
            self.stats['tp_hits'] += 1

            # Remove old TRADE position
            del self.trade_positions[symbol]

            # Auto re-enter new TRADE position
            if self.config.auto_reentry_on_tp:
                self._reenter_trade_position(symbol, current_price)

            return True

        except Exception as e:
            print(f"  [ERROR] Failed to close TRADE for {symbol}: {e}")
            return False

    def _reenter_trade_position(self, symbol: str, price: float):
        """Re-enter TRADE position after TP hit."""
        trade_qty = self.config.get_trade_quantity(symbol, price, dca_level=0)
        alpaca_symbol = symbol.replace('USD', '/USD')

        print(f"  [RE-ENTRY] New TRADE: {trade_qty:.6f} @ ${price:,.2f}")

        try:
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=trade_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            result = self.trading_client.submit_order(order)
            time.sleep(1)

            filled_order = self.trading_client.get_order_by_id(result.id)
            fill_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else price

            # Create new TRADE position
            self.trade_positions[symbol] = TradePosition(
                symbol=symbol,
                entry_price=fill_price,
                avg_entry_price=fill_price,
                total_quantity=trade_qty,
                entry_time=datetime.now(),
                dca_level=0,
                dca_quantities=[trade_qty],
                dca_prices=[fill_price],
                current_price=fill_price
            )

            print(f"  [OK] New TRADE opened at ${fill_price:,.2f}")

        except Exception as e:
            print(f"  [ERROR] Re-entry failed: {e}")

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol using multiple methods."""
        alpaca_symbol = symbol.replace('USD', '/USD')

        # Method 1: Try to get latest quote
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoLatestQuoteRequest

            if self.data_client:
                quotes = self.data_client.get_crypto_latest_quote(
                    CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_symbol)
                )
                if quotes and alpaca_symbol in quotes:
                    quote = quotes[alpaca_symbol]
                    # Use mid price between bid and ask
                    if quote.bid_price and quote.ask_price:
                        price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                        return price
        except Exception as e:
            if self.config.verbose:
                print(f"  Quote method failed for {symbol}: {e}")

        # Method 2: Try to get latest bar
        try:
            if self.data_client:
                bars = self.data_client.get_crypto_bars(
                    CryptoBarsRequest(
                        symbol_or_symbols=alpaca_symbol,
                        timeframe=TimeFrame.Minute,
                        limit=1
                    )
                )
                if bars and alpaca_symbol in bars and len(bars[alpaca_symbol]) > 0:
                    return float(bars[alpaca_symbol][0].close)
        except Exception as e:
            if self.config.verbose:
                print(f"  Bars method failed for {symbol}: {e}")

        # Method 3: Try getting from existing position's market value
        try:
            if self.trading_client:
                pos = self.trading_client.get_open_position(alpaca_symbol)
                if pos and float(pos.qty) > 0:
                    return float(pos.current_price)
        except Exception as e:
            pass  # No position, that's okay

        # Method 4: Use latest trade
        try:
            from alpaca.data.requests import CryptoLatestTradeRequest

            if self.data_client:
                trades = self.data_client.get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=alpaca_symbol)
                )
                if trades and alpaca_symbol in trades:
                    return float(trades[alpaca_symbol].price)
        except Exception as e:
            if self.config.verbose:
                print(f"  Trade method failed for {symbol}: {e}")

        return 0.0

    def _get_existing_position(self, alpaca_symbol: str) -> float:
        """Get existing position quantity from Alpaca."""
        # Alpaca positions use BTCUSD format, not BTC/USD
        symbol_noslash = alpaca_symbol.replace('/', '')

        try:
            pos = self.trading_client.get_open_position(symbol_noslash)
            return float(pos.qty)
        except:
            pass

        # Also try with slash (in case API changes)
        try:
            pos = self.trading_client.get_open_position(alpaca_symbol)
            return float(pos.qty)
        except:
            pass

        return 0.0

    def _reset_daily_counts_if_needed(self):
        """Reset daily DCA counts at midnight."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.daily_dca_count = {}
            self.last_reset_date = today

    def get_status(self) -> str:
        """Get current status string."""
        status = []
        status.append("\n" + "=" * 70)
        status.append("ALWAYS HOLD STATUS")
        status.append("=" * 70)

        for symbol in self.config.symbols:
            status.append(f"\n{symbol}:")

            if symbol in self.hold_positions:
                hold = self.hold_positions[symbol]
                status.append(f"  HOLD: {hold.quantity:.6f} @ ${hold.entry_price:,.2f}")

            if symbol in self.trade_positions:
                trade = self.trade_positions[symbol]
                status.append(f"  TRADE: {trade.total_quantity:.6f} @ avg ${trade.avg_entry_price:,.2f}")
                status.append(f"    DCA Level: {trade.dca_level}/{self.config.max_dca_stages}")
                if trade.unrealized_pnl != 0:
                    status.append(f"    Unrealized: ${trade.unrealized_pnl:,.2f}")

        status.append(f"\nSession Stats:")
        status.append(f"  Trade P&L: ${self.stats['trade_pnl']:,.2f}")
        status.append(f"  Trades Closed: {self.stats['trades_closed']}")
        status.append(f"  DCA Entries: {self.stats['dca_entries']}")
        status.append(f"  TP Hits: {self.stats['tp_hits']}")

        return "\n".join(status)


if __name__ == "__main__":
    # Test the engine
    config = load_always_hold_config()
    print(config)

    engine = AlwaysHoldEngine(config)
    if engine.trading_client:
        engine.initialize_positions()
        print(engine.get_status())
