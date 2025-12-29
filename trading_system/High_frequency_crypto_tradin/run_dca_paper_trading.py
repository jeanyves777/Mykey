"""
Run DCA Paper Trading - Comprehensive Version
==============================================

Paper trading for the DCA momentum strategy on crypto.
Uses Alpaca Paper Trading API with proper order handling:
- No bracket orders (Alpaca limitation)
- Send SL order AFTER main order fills
- Monitor TP manually

Usage:
    python -m trading_system.High_frequency_crypto_tradin.run_dca_paper_trading
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

from trading_system.High_frequency_crypto_tradin.dca_config import DCAConfig, load_dca_config
from trading_system.High_frequency_crypto_tradin.dca_engine import DCAEngine
from trading_system.High_frequency_crypto_tradin.trend_filter import TrendFilter, TrendFilterConfig

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("WARNING: Alpaca SDK not installed. Run: pip install alpaca-py")


class DCALivePaperEngine:
    """Live paper trading engine for DCA strategy with comprehensive logging."""

    def __init__(self, config: DCAConfig):
        self.config = config
        self.dca_engine = DCAEngine(config)

        # Initialize trend filter for Smart DCA
        self.trend_filter = None
        if getattr(config, 'use_smart_dca_filter', True):
            trend_config = TrendFilterConfig(
                adx_period=getattr(config, 'adx_period', 14),
                adx_strong_trend=getattr(config, 'adx_strong_trend', 30.0),
                adx_weak_trend=getattr(config, 'adx_weak_trend', 20.0),
                ema_fast_period=getattr(config, 'ema_fast_period', 8),
                ema_slow_period=getattr(config, 'ema_slow_period', 21),
                require_reversal_candle=getattr(config, 'require_reversal_candle', True),
                min_reversal_body_ratio=getattr(config, 'min_reversal_body_ratio', 0.6),
                high_dca_threshold=getattr(config, 'high_dca_confirmation_level', 3)
            )
            self.trend_filter = TrendFilter(trend_config)

        # Alpaca clients
        self.trading_client = None
        self.data_client = None

        # Trading state
        self.positions = {}  # symbol -> position info
        self.stop_orders = {}  # symbol -> stop order id
        self.last_signal_time = {}
        self.trade_count = 0
        self.session_start = datetime.now()

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

        # Initialize Alpaca
        self._init_alpaca()

        # Check for existing positions on startup
        if self.trading_client:
            self._check_existing_positions()

    def _init_alpaca(self):
        """Initialize Alpaca API clients."""
        if not ALPACA_AVAILABLE:
            print("ERROR: Alpaca SDK not available")
            return

        api_key = os.getenv('ALPACA_CRYPTO_KEY') or os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_CRYPTO_SECRET') or os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not api_secret:
            print("ERROR: Alpaca API credentials not found!")
            print("Set ALPACA_CRYPTO_KEY and ALPACA_CRYPTO_SECRET in .env")
            return

        try:
            # Paper trading client
            self.trading_client = TradingClient(api_key, api_secret, paper=True)

            # Crypto data client (with API keys for authenticated access)
            self.data_client = CryptoHistoricalDataClient(api_key, api_secret)

            # Get account info
            account = self.trading_client.get_account()
            print(f"Connected to Alpaca Paper Trading")
            print(f"  Account: ${float(account.equity):,.2f}")
            print(f"  Buying Power: ${float(account.buying_power):,.2f}")

        except Exception as e:
            print(f"ERROR connecting to Alpaca: {e}")
            self.trading_client = None

    def _check_existing_positions(self):
        """Check for existing positions on startup and ensure stop losses are in place."""
        if not self.trading_client:
            return

        print("\n" + "=" * 70)
        print("CHECKING EXISTING POSITIONS & ORDERS")
        print("=" * 70)

        try:
            # Get all open positions
            positions = self.trading_client.get_all_positions()

            # Get all open orders
            open_orders = self.trading_client.get_orders()

            # Build map of existing stop orders by symbol (stop or stop_limit)
            existing_stops = {}
            for order in open_orders:
                if order.side == OrderSide.SELL and order.type.value in ['stop', 'stop_limit']:
                    symbol = order.symbol.replace('/', '')  # BTC/USD -> BTCUSD
                    existing_stops[symbol] = {
                        'order_id': str(order.id),  # Convert UUID to string
                        'stop_price': float(order.stop_price) if order.stop_price else 0,
                        'qty': float(order.qty)
                    }

            if not positions:
                print("  No existing positions found.")
                return

            print(f"\nFound {len(positions)} existing position(s):")

            for pos in positions:
                symbol = pos.symbol.replace('/', '')  # BTC/USD -> BTCUSD
                qty = float(pos.qty)
                avg_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100

                print(f"\n  {symbol}:")
                print(f"    Qty: {qty:.6f} @ Avg ${avg_price:,.2f}")
                print(f"    Current: ${current_price:,.2f} | P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%)")

                # Check if this is one of our symbols
                if symbol not in self.config.symbols:
                    print(f"    WARNING: Not in config symbols, will not manage this position")
                    continue

                # Calculate TP/SL levels based on avg entry
                stop_loss = avg_price * (1 - self.config.stop_loss_pct)
                take_profit = avg_price * (1 + self.config.take_profit_pct)

                # Add to our tracked positions
                self.positions[symbol] = {
                    'entry_price': avg_price,
                    'quantity': qty,
                    'avg_price': avg_price,
                    'total_qty': qty,
                    'dca_stage': 0,  # Unknown, assume 0
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now(),  # Unknown actual entry time
                    'entry_momentum': 0,
                    'entry_rsi': 0,
                    'resumed': True,  # Mark as resumed position
                    'be_locked': False  # Breakeven lock flag
                }

                print(f"    TP: ${take_profit:,.2f} (+{self.config.take_profit_pct*100:.2f}%)")
                print(f"    SL: ${stop_loss:,.2f} (-{self.config.stop_loss_pct*100:.2f}%)")

                # Check for existing stop order
                if symbol in existing_stops:
                    stop_info = existing_stops[symbol]
                    self.stop_orders[symbol] = stop_info['order_id']
                    print(f"    STOP ORDER EXISTS: ${stop_info['stop_price']:,.2f} for {stop_info['qty']:.6f}")

                    # Check if stop quantity matches position
                    if abs(stop_info['qty'] - qty) > 0.000001:
                        print(f"    WARNING: Stop qty ({stop_info['qty']:.6f}) doesn't match position ({qty:.6f})")
                        print(f"    Canceling old stop and placing new one...")
                        self.cancel_stop_order(symbol)
                        sl_order_id = self.place_stop_loss_order(symbol, qty, stop_loss)
                        if sl_order_id:
                            self.stop_orders[symbol] = sl_order_id
                            print(f"    NEW STOP PLACED @ ${stop_loss:,.2f}")
                else:
                    # No stop order - MUST place one
                    print(f"    NO STOP ORDER! Placing stop @ ${stop_loss:,.2f}...")
                    sl_order_id = self.place_stop_loss_order(symbol, qty, stop_loss)
                    if sl_order_id:
                        self.stop_orders[symbol] = sl_order_id
                        print(f"    STOP ORDER PLACED @ ${stop_loss:,.2f}")
                    else:
                        print(f"    ERROR: Failed to place stop order!")

            print(f"\n  Resumed {len(self.positions)} position(s) with stop losses verified.")

        except Exception as e:
            print(f"  ERROR checking positions: {e}")

    def get_account_equity(self) -> float:
        """Get current account equity."""
        if not self.trading_client:
            return self.config.initial_capital
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except:
            return self.config.initial_capital

    def get_live_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """Fetch live 1-minute bars from Alpaca."""
        if not self.data_client:
            return None

        try:
            # Convert symbol format (BTCUSD -> BTC/USD)
            alpaca_symbol = symbol.replace('USD', '/USD')

            # Use timezone-aware datetime
            from datetime import timezone
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=bars + 50)

            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time
            )

            bars_data = self.data_client.get_crypto_bars(request)

            # Handle different response formats
            if hasattr(bars_data, 'df'):
                df = bars_data.df
                if len(df) > 0:
                    df = df.reset_index()
                    # Rename columns to match expected format
                    col_map = {'symbol': 'symbol', 'timestamp': 'timestamp',
                               'open': 'open', 'high': 'high', 'low': 'low',
                               'close': 'close', 'volume': 'volume'}
                    df = df.rename(columns={c: c.lower() for c in df.columns})
                    return df.tail(bars).reset_index(drop=True)
            elif alpaca_symbol in bars_data:
                df = bars_data[alpaca_symbol].df.reset_index()
                return df.tail(bars).reset_index(drop=True)

        except Exception as e:
            if self.config.verbose:
                print(f"    Data error {symbol}: {e}")

        return None

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        df = self.get_live_data(symbol, bars=2)
        if df is not None and len(df) > 0:
            return df['close'].iloc[-1]
        return 0.0

    def check_signal(self, symbol: str) -> tuple:
        """Check if we have a valid entry signal. Returns (has_signal, momentum, rsi, price)"""
        df = self.get_live_data(symbol, bars=100)
        if df is None or len(df) < 50:
            return False, 0, 0, 0

        # Calculate signals using DCA engine
        df = self.dca_engine.calculate_signals(df)

        latest_signal = df['signal'].iloc[-1]
        momentum = df['momentum'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        price = df['close'].iloc[-1]

        # Cooldown check
        last_time = self.last_signal_time.get(symbol, datetime.min)
        cooldown_ok = (datetime.now() - last_time).total_seconds() > (self.config.cooldown_bars * 60)

        if latest_signal == 1 and cooldown_ok:
            self.stats['signals_detected'] += 1
            return True, momentum, rsi, price

        return False, momentum, rsi, price

    def wait_for_fill(self, order_id: str, timeout: int = 30) -> dict:
        """Wait for an order to fill. Returns fill info or None."""
        if not self.trading_client:
            return None

        start = time.time()
        while time.time() - start < timeout:
            try:
                order = self.trading_client.get_order_by_id(order_id)
                if order.status == OrderStatus.FILLED:
                    return {
                        'filled_qty': float(order.filled_qty),
                        'filled_avg_price': float(order.filled_avg_price)
                    }
                elif order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    return None
            except:
                pass
            time.sleep(0.5)
        return None

    def place_stop_loss_order(self, symbol: str, quantity: float, stop_price: float) -> str:
        """
        Place a stop-limit order for crypto (Alpaca crypto supports: market, limit, stop_limit).
        Uses stop_limit with limit price slightly below stop to ensure fill.
        """
        if not self.trading_client:
            return None

        try:
            from alpaca.trading.requests import StopLimitOrderRequest

            alpaca_symbol = symbol.replace('USD', '/USD')

            # For stop-limit, set limit price 0.5% below stop to ensure fill on a fast move
            limit_price = stop_price * 0.995

            order_data = StopLimitOrderRequest(
                symbol=alpaca_symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
                limit_price=round(limit_price, 2)
            )

            order = self.trading_client.submit_order(order_data)
            print(f"    STOP-LIMIT ORDER PLACED @ ${stop_price:,.2f} (limit ${limit_price:,.2f})")
            return str(order.id)  # Convert UUID to string

        except Exception as e:
            # If stop-limit fails, fall back to manual monitoring
            print(f"    STOP-LIMIT ORDER FAILED: {e}")
            print(f"    SL LEVEL SET @ ${stop_price:,.2f} (will be monitored manually)")
            fake_sl_id = f"manual_sl_{symbol}_{stop_price:.2f}"
            return fake_sl_id

    def cancel_stop_order(self, symbol: str):
        """Cancel existing stop order for a symbol."""
        if symbol in self.stop_orders:
            order_id = self.stop_orders[symbol]
            # Only try to cancel if it's a real order (not manual_sl_*)
            if self.trading_client and not order_id.startswith('manual_sl_'):
                try:
                    self.trading_client.cancel_order_by_id(order_id)
                    print(f"    Canceled stop order for {symbol}")
                except Exception as e:
                    print(f"    Note: Could not cancel stop order: {e}")
            del self.stop_orders[symbol]

    def enter_position(self, symbol: str, momentum: float, rsi: float, current_price: float):
        """Enter a new position with proper order handling and trend validation."""
        if not self.trading_client:
            print(f"  {symbol}: Would enter @ ${current_price:,.2f} (no API)")
            return

        if symbol in self.positions:
            return

        # Trend validation before entry (Smart DCA filter)
        if self.trend_filter and getattr(self.config, 'validate_entry_trend', True):
            df = self.get_live_data(symbol, bars=100)
            if df is not None and len(df) >= 30:
                can_enter, reason = self.trend_filter.validate_entry(df, 'long')
                if not can_enter:
                    print(f"  {symbol}: ENTRY BLOCKED - {reason}")
                    self.stats['entries_blocked'] += 1
                    return
                else:
                    print(f"    Trend validation: {reason}")

        try:
            # PROPER POSITION SIZING - DISTRIBUTED TO AFFORD ALL DCA STAGES:
            # Example with $1,000 / 3 symbols = $333 per symbol
            # DCA multipliers: [0.25, 0.125, 0.25, 0.375, 0.5] = 1.5 total
            # Base unit = $333 / 1.5 = $222
            # Entry: $222 * 0.25 = $55.50
            # DCA1: $222 * 0.125 = $27.75
            # DCA2: $222 * 0.25 = $55.50
            # DCA3: $222 * 0.375 = $83.25
            # DCA4: $222 * 0.5 = $111
            # Total: $333 ✓

            equity = self.get_account_equity()
            num_symbols = len(self.config.symbols)

            # Per-symbol allocation (e.g., $1000 / 3 symbols = $333 each)
            symbol_allocation = equity / num_symbols

            # Normalize by total DCA multipliers to afford ALL stages
            total_dca_multipliers = sum(self.config.dca_multipliers)  # 1.5
            base_unit = symbol_allocation / total_dca_multipliers  # $333 / 1.5 = $222

            # Entry uses first DCA multiplier
            entry_multiplier = self.config.dca_multipliers[0]  # 0.25
            position_value = base_unit * entry_multiplier  # $222 * 0.25 = $55.50

            # Ensure we have enough buying power
            try:
                account = self.trading_client.get_account()
                buying_power = float(account.buying_power)
                if position_value > buying_power * 0.95:
                    position_value = buying_power * 0.95
                    print(f"    Position adjusted to ${position_value:,.2f} (buying power limit)")
            except:
                pass

            quantity = position_value / current_price
            quantity = round(quantity, 6)

            # Place market order
            alpaca_symbol = symbol.replace('USD', '/USD')

            order_data = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )

            print(f"  {symbol}: ENTERING @ ~${current_price:,.2f}")
            print(f"    Momentum: {momentum:.2f}% | RSI: {rsi:.1f}")
            print(f"    Position Value: ${position_value:,.2f} (Entry = {entry_multiplier*100:.1f}% of ${base_unit:,.2f} base)")

            order = self.trading_client.submit_order(order_data)

            # Wait for fill
            fill_info = self.wait_for_fill(order.id)

            if fill_info:
                filled_price = fill_info['filled_avg_price']
                filled_qty = fill_info['filled_qty']

                # Calculate TP/SL levels
                stop_loss = filled_price * (1 - self.config.stop_loss_pct)
                take_profit = filled_price * (1 + self.config.take_profit_pct)

                # Place stop loss order
                sl_order_id = self.place_stop_loss_order(symbol, filled_qty, stop_loss)

                # Track position
                self.positions[symbol] = {
                    'entry_price': filled_price,
                    'quantity': filled_qty,
                    'avg_price': filled_price,
                    'total_qty': filled_qty,
                    'dca_stage': 0,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now(),
                    'entry_momentum': momentum,
                    'entry_rsi': rsi,
                    'be_locked': False  # Breakeven lock flag
                }

                if sl_order_id:
                    self.stop_orders[symbol] = sl_order_id

                self.last_signal_time[symbol] = datetime.now()
                self.trade_count += 1
                self.stats['entries'] += 1

                print(f"    FILLED @ ${filled_price:,.2f} | Qty: {filled_qty:.6f}")
                print(f"    TP: ${take_profit:,.2f} (+{self.config.take_profit_pct*100:.2f}%)")
                print(f"    SL: ${stop_loss:,.2f} (-{self.config.stop_loss_pct*100:.2f}%)")

            else:
                print(f"    ORDER NOT FILLED (timeout)")

        except Exception as e:
            print(f"  {symbol}: Order error: {e}")

    def get_dynamic_tp(self, dca_stage: int) -> float:
        """Get dynamic TP percentage based on DCA level (reduces as DCA increases)."""
        # Base TP reduced by tp_reduction_per_dca for each DCA level
        # DCA 0: 0.5%, DCA 1: 0.4%, DCA 2: 0.35%, DCA 3: 0.3%, DCA 4: 0.25%
        tp_reduction = getattr(self.config, 'tp_reduction_per_dca', 0.001)
        min_tp = getattr(self.config, 'min_tp_pct', 0.0025)

        dynamic_tp = self.config.take_profit_pct - (dca_stage * tp_reduction)
        return max(dynamic_tp, min_tp)

    def check_tp_and_dca(self, symbol: str) -> tuple:
        """Check for TP hit, DCA opportunities, breakeven lock, and emergency exit."""
        if symbol not in self.positions:
            return None, {}

        pos = self.positions[symbol]
        current_price = self.get_current_price(symbol)

        if current_price <= 0:
            return None, {}

        # Calculate P&L
        pnl = pos['total_qty'] * (current_price - pos['avg_price'])
        pnl_pct = (current_price / pos['avg_price'] - 1) * 100
        dca_stage = pos['dca_stage']

        # Get dynamic TP based on DCA level
        dynamic_tp_pct = self.get_dynamic_tp(dca_stage)
        dynamic_tp_price = pos['avg_price'] * (1 + dynamic_tp_pct)

        # Update position's TP to dynamic value
        pos['take_profit'] = dynamic_tp_price

        details = {
            'current_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'avg_price': pos['avg_price'],
            'tp': dynamic_tp_price,
            'tp_pct': dynamic_tp_pct * 100,
            'sl': pos['stop_loss'],
            'dca_stage': dca_stage,
            'total_qty': pos['total_qty']
        }

        # ================================================================
        # EMERGENCY EXIT: DCA 4 positions - exit at small profit
        # ================================================================
        emergency_dca = getattr(self.config, 'emergency_exit_dca_level', 4)
        emergency_profit = getattr(self.config, 'emergency_exit_profit_pct', 0.0015)

        if dca_stage >= emergency_dca:
            if pnl_pct / 100 >= emergency_profit:
                details['exit_reason'] = f'EMERGENCY EXIT (DCA {dca_stage}, +{pnl_pct:.2f}%)'
                return 'EMERGENCY_EXIT', details

        # ================================================================
        # BREAKEVEN LOCK: DCA 3+ positions - lock in no-loss
        # ================================================================
        be_threshold = getattr(self.config, 'breakeven_dca_threshold', 3)
        be_buffer = getattr(self.config, 'breakeven_buffer_pct', 0.001)

        if dca_stage >= be_threshold and not pos.get('be_locked', False):
            breakeven_trigger = pos['avg_price'] * (1 + be_buffer)
            if current_price >= breakeven_trigger:
                # Move SL to breakeven (avg_price)
                if pos['stop_loss'] < pos['avg_price']:
                    details['new_sl'] = pos['avg_price']
                    details['exit_reason'] = f'BE LOCK (DCA {dca_stage})'
                    return 'BREAKEVEN_LOCK', details

        # ================================================================
        # REGULAR TP CHECK (with dynamic TP)
        # ================================================================
        if current_price >= dynamic_tp_price:
            details['exit_reason'] = f'TP HIT ({dynamic_tp_pct*100:.2f}%)'
            return 'TP', details

        # ================================================================
        # DCA TRIGGER
        # ================================================================
        if dca_stage < self.config.max_dca_stages:
            next_dca_price = pos['entry_price'] * (1 - self.config.dca_spacing_pct * (dca_stage + 1))
            details['next_dca'] = next_dca_price

            if current_price <= next_dca_price:
                return 'DCA', details

        return 'HOLD', details

    def execute_tp_exit(self, symbol: str, current_price: float):
        """Execute take profit exit."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        try:
            # Cancel stop order
            self.cancel_stop_order(symbol)

            if self.trading_client:
                # Get ACTUAL position quantity from Alpaca (may differ from tracked qty)
                alpaca_symbol = symbol.replace('USD', '/USD')
                try:
                    actual_pos = self.trading_client.get_open_position(alpaca_symbol)
                    actual_qty = float(actual_pos.qty)
                except:
                    actual_qty = pos['total_qty']  # Fallback to tracked qty

                order_data = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=actual_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )

                order = self.trading_client.submit_order(order_data)
                fill_info = self.wait_for_fill(order.id)

                if fill_info:
                    exit_price = fill_info['filled_avg_price']
                    pnl = pos['total_qty'] * (exit_price - pos['avg_price'])
                    pnl_pct = (exit_price / pos['avg_price'] - 1) * 100

                    print(f"  {symbol}: TP HIT @ ${exit_price:,.2f}")
                    print(f"    P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                    print(f"    DCA Stages Used: {pos['dca_stage']}")

                    self.stats['exits_tp'] += 1
                    self.stats['total_pnl'] += pnl

            del self.positions[symbol]
            self.last_signal_time[symbol] = datetime.now()

        except Exception as e:
            print(f"  {symbol}: TP exit error: {e}")

    def execute_emergency_exit(self, symbol: str, current_price: float, reason: str):
        """Execute emergency exit for max DCA positions at small profit."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        try:
            # Cancel stop order
            self.cancel_stop_order(symbol)

            if self.trading_client:
                # Get ACTUAL position quantity from Alpaca (may differ from tracked qty)
                alpaca_symbol = symbol.replace('USD', '/USD')
                try:
                    actual_pos = self.trading_client.get_open_position(alpaca_symbol)
                    actual_qty = float(actual_pos.qty)
                except:
                    actual_qty = pos['total_qty']  # Fallback to tracked qty

                order_data = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=actual_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )

                order = self.trading_client.submit_order(order_data)
                fill_info = self.wait_for_fill(order.id)

                if fill_info:
                    exit_price = fill_info['filled_avg_price']
                    pnl = pos['total_qty'] * (exit_price - pos['avg_price'])
                    pnl_pct = (exit_price / pos['avg_price'] - 1) * 100

                    print(f"  {symbol}: {reason} @ ${exit_price:,.2f}")
                    print(f"    P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                    print(f"    DCA Stages Used: {pos['dca_stage']}")

                    self.stats['exits_tp'] += 1  # Count as TP since it's a profit exit
                    self.stats['total_pnl'] += pnl

            del self.positions[symbol]
            self.last_signal_time[symbol] = datetime.now()

        except Exception as e:
            print(f"  {symbol}: Emergency exit error: {e}")

    def execute_breakeven_lock(self, symbol: str, new_sl_price: float):
        """Move stop loss to breakeven for high DCA positions."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        try:
            # Cancel old stop order
            self.cancel_stop_order(symbol)

            # Update position's stop loss to breakeven
            old_sl = pos['stop_loss']
            pos['stop_loss'] = new_sl_price
            pos['be_locked'] = True  # Mark as locked to prevent re-triggering

            # Place new stop order at breakeven
            sl_order_id = self.place_stop_loss_order(symbol, pos['total_qty'], new_sl_price)

            if sl_order_id:
                self.stop_orders[symbol] = sl_order_id
                print(f"  {symbol}: BREAKEVEN LOCK ACTIVATED!")
                print(f"    SL moved from ${old_sl:,.2f} to ${new_sl_price:,.2f} (breakeven)")
                print(f"    DCA Level: {pos['dca_stage']} | Position now protected!")
            else:
                print(f"  {symbol}: BE lock failed - SL at ${new_sl_price:,.2f} will be monitored manually")

        except Exception as e:
            print(f"  {symbol}: BE lock error: {e}")

    def execute_dca(self, symbol: str, dca_price: float):
        """Execute a DCA entry with trend filtering."""
        if not self.trading_client or symbol not in self.positions:
            return

        pos = self.positions[symbol]
        stage_idx = pos['dca_stage'] + 1

        if stage_idx >= len(self.config.dca_multipliers):
            return

        # Smart DCA trend filter - check if safe to add to position
        if self.trend_filter and getattr(self.config, 'use_smart_dca_filter', True):
            df = self.get_live_data(symbol, bars=100)
            if df is not None and len(df) >= 30:
                can_dca, reason = self.trend_filter.can_dca(df, 'long', stage_idx)
                if not can_dca:
                    print(f"  {symbol}: DCA {stage_idx} BLOCKED - {reason}")
                    print(f"    Waiting for trend reversal before adding...")
                    self.stats['dca_blocked'] += 1
                    return
                else:
                    print(f"    Trend check: {reason}")

        try:
            # Use SAME sizing logic as entry - normalized to afford all DCA stages
            equity = self.get_account_equity()
            num_symbols = len(self.config.symbols)
            symbol_allocation = equity / num_symbols

            # Normalize by total DCA multipliers
            total_dca_multipliers = sum(self.config.dca_multipliers)
            base_unit = symbol_allocation / total_dca_multipliers

            # DCA position value based on stage multiplier
            dca_multiplier = self.config.dca_multipliers[stage_idx]
            position_value = base_unit * dca_multiplier
            dca_quantity = position_value / dca_price
            dca_quantity = round(dca_quantity, 6)

            # Place DCA order
            alpaca_symbol = symbol.replace('USD', '/USD')

            order_data = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=dca_quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )

            print(f"  {symbol}: DCA {stage_idx} ENTRY @ ~${dca_price:,.2f}")

            order = self.trading_client.submit_order(order_data)
            fill_info = self.wait_for_fill(order.id)

            if fill_info:
                filled_price = fill_info['filled_avg_price']
                filled_qty = fill_info['filled_qty']

                # Update position
                old_cost = pos['avg_price'] * pos['total_qty']
                new_cost = filled_price * filled_qty
                pos['total_qty'] += filled_qty
                pos['avg_price'] = (old_cost + new_cost) / pos['total_qty']
                pos['dca_stage'] += 1
                pos['take_profit'] = pos['avg_price'] * (1 + self.config.take_profit_pct)

                # Cancel old stop order and place new one
                self.cancel_stop_order(symbol)
                sl_order_id = self.place_stop_loss_order(symbol, pos['total_qty'], pos['stop_loss'])
                if sl_order_id:
                    self.stop_orders[symbol] = sl_order_id

                self.stats['dca_entries'] += 1

                print(f"    FILLED @ ${filled_price:,.2f} | +{filled_qty:.6f}")
                print(f"    New Avg: ${pos['avg_price']:,.2f}")
                print(f"    New TP: ${pos['take_profit']:,.2f}")
                print(f"    Total Qty: {pos['total_qty']:.6f}")

        except Exception as e:
            print(f"  {symbol}: DCA error: {e}")

    def check_sl_triggered(self, symbol: str) -> bool:
        """Check if stop loss was triggered (order filled OR price hit manual SL level)."""
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]

        # Check if we have a real stop-limit order
        if symbol in self.stop_orders and not self.stop_orders[symbol].startswith('manual_sl_'):
            try:
                order = self.trading_client.get_order_by_id(self.stop_orders[symbol])
                if order.status == OrderStatus.FILLED:
                    exit_price = float(order.filled_avg_price)
                    pnl = pos['total_qty'] * (exit_price - pos['avg_price'])

                    print(f"  {symbol}: SL ORDER FILLED @ ${exit_price:,.2f}")
                    print(f"    P&L: ${pnl:,.2f}")
                    print(f"    DCA Stages Used: {pos.get('dca_stage', 0)}")

                    self.stats['exits_sl'] += 1
                    self.stats['total_pnl'] += pnl

                    del self.positions[symbol]
                    del self.stop_orders[symbol]

                    self.last_signal_time[symbol] = datetime.now()
                    return True
            except:
                pass

        # Also check manual SL monitoring (backup in case stop-limit didn't fill)
        current_price = self.get_current_price(symbol)
        if current_price > 0 and current_price <= pos['stop_loss']:
            # Price hit our SL level - execute market sell
            print(f"  {symbol}: SL LEVEL HIT @ ${current_price:,.2f} (SL was ${pos['stop_loss']:,.2f})")
            print(f"    Executing market sell...")

            try:
                # Cancel any existing stop order first
                if symbol in self.stop_orders and not self.stop_orders[symbol].startswith('manual_sl_'):
                    try:
                        self.trading_client.cancel_order_by_id(self.stop_orders[symbol])
                    except:
                        pass

                # Market sell - get ACTUAL quantity from Alpaca
                alpaca_symbol = symbol.replace('USD', '/USD')
                try:
                    actual_pos = self.trading_client.get_open_position(alpaca_symbol)
                    actual_qty = float(actual_pos.qty)
                except:
                    actual_qty = pos['total_qty']

                order_data = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=actual_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )

                order = self.trading_client.submit_order(order_data)
                fill_info = self.wait_for_fill(order.id)

                if fill_info:
                    exit_price = fill_info['filled_avg_price']
                    pnl = actual_qty * (exit_price - pos['avg_price'])

                    print(f"    SL EXIT FILLED @ ${exit_price:,.2f}")
                    print(f"    P&L: ${pnl:,.2f}")
                    print(f"    DCA Stages Used: {pos.get('dca_stage', 0)}")

                    self.stats['exits_sl'] += 1
                    self.stats['total_pnl'] += pnl

                del self.positions[symbol]
                if symbol in self.stop_orders:
                    del self.stop_orders[symbol]

                self.last_signal_time[symbol] = datetime.now()
                return True

            except Exception as e:
                print(f"    ERROR executing SL: {e}")

        return False

    def display_comprehensive_status(self):
        """Display comprehensive status like the Forex runner."""
        now = datetime.now()
        equity = self.get_account_equity()

        print("\n" + "=" * 70)
        print(f"[{now.strftime('%H:%M:%S')}] DCA SCANNING {len(self.config.symbols)} PAIRS | "
              f"Equity: ${equity:,.2f} | Open: {len(self.positions)}")
        print("=" * 70)

    def display_position_status(self):
        """Display detailed position status with Smart DCA Exit info."""
        if not self.positions:
            print("\nNo open positions")
            return

        total_pnl = 0.0

        print(f"\n--- OPEN POSITIONS ({len(self.positions)}) ---")

        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price <= 0:
                continue

            pnl = pos['total_qty'] * (current_price - pos['avg_price'])
            pnl_pct = (current_price / pos['avg_price'] - 1) * 100
            total_pnl += pnl

            # Get dynamic TP based on DCA level
            dynamic_tp_pct = self.get_dynamic_tp(pos['dca_stage'])
            dynamic_tp_price = pos['avg_price'] * (1 + dynamic_tp_pct)

            tp_away = dynamic_tp_price - current_price
            tp_away_pct = (tp_away / current_price) * 100
            sl_away = current_price - pos['stop_loss']
            sl_away_pct = (sl_away / current_price) * 100

            # Check if breakeven is locked
            be_locked = pos.get('be_locked', False)
            sl_status = " [BE LOCKED]" if be_locked else ""

            # Check if emergency exit is active
            emergency_dca = getattr(self.config, 'emergency_exit_dca_level', 4)
            emergency_active = pos['dca_stage'] >= emergency_dca
            emergency_status = " [EMERGENCY MODE]" if emergency_active else ""

            print(f"  {symbol}: BUY {pos['total_qty']:.6f} @ ${pos['avg_price']:,.2f}{emergency_status}")
            print(f"    Current: ${current_price:,.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            print(f"    TP: ${dynamic_tp_price:,.2f} ({dynamic_tp_pct*100:.2f}%) (+${tp_away:,.2f} / {tp_away_pct:+.2f}% away)")
            print(f"    SL: ${pos['stop_loss']:,.2f} (${sl_away:,.2f} / {sl_away_pct:.2f}% away){sl_status}")
            print(f"    DCA Level: {pos['dca_stage']} | Entry: {pos['entry_time'].strftime('%H:%M:%S')}")

            # Show next DCA level if available
            if pos['dca_stage'] < self.config.max_dca_stages:
                next_dca = pos['entry_price'] * (1 - self.config.dca_spacing_pct * (pos['dca_stage'] + 1))
                dca_away = current_price - next_dca
                print(f"    Next DCA: ${next_dca:,.2f} (${dca_away:,.2f} away)")

        print(f"\n  Total Unrealized P&L: ${total_pnl:,.2f}")

    def display_session_stats(self):
        """Display session statistics."""
        runtime = datetime.now() - self.session_start
        hours = runtime.total_seconds() / 3600

        print(f"\n--- SESSION STATS (Runtime: {runtime}) ---")
        print(f"  Signals Detected: {self.stats['signals_detected']}")
        print(f"  Entries: {self.stats['entries']} (Blocked: {self.stats.get('entries_blocked', 0)})")
        print(f"  DCA Entries: {self.stats['dca_entries']} (Blocked: {self.stats.get('dca_blocked', 0)})")
        print(f"  TP Exits: {self.stats['exits_tp']}")
        print(f"  SL Exits: {self.stats['exits_sl']}")
        print(f"  Realized P&L: ${self.stats['total_pnl']:,.2f}")

    def _handle_shutdown(self):
        """Handle shutdown - ask user about open positions."""
        if not self.positions:
            print("\nNo open positions. Clean shutdown.")
            return

        print("\n" + "=" * 70)
        print("SHUTDOWN OPTIONS - OPEN POSITIONS")
        print("=" * 70)

        # Show current positions summary
        total_unrealized = 0.0
        for symbol, pos in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price > 0:
                pnl = pos['total_qty'] * (current_price - pos['avg_price'])
                total_unrealized += pnl
                sl_status = "SL ACTIVE" if symbol in self.stop_orders else "NO STOP!"
                print(f"  {symbol}: {pos['total_qty']:.6f} @ ${pos['avg_price']:,.2f} | "
                      f"P&L: ${pnl:,.2f} | {sl_status}")

        print(f"\n  Total Unrealized P&L: ${total_unrealized:,.2f}")

        print("\nWhat would you like to do?")
        print("  [1] LEAVE OPEN - Keep positions open with stop losses active")
        print("  [2] CLOSE ALL - Close all positions now at market")
        print("  [3] CLOSE LOSERS - Close only losing positions")
        print("  [4] CANCEL STOPS - Leave positions but cancel stop orders")

        try:
            choice = input("\nEnter choice (1-4) [default: 1]: ").strip()

            if choice == '2':
                self._close_all_positions()
            elif choice == '3':
                self._close_losing_positions()
            elif choice == '4':
                self._cancel_all_stops()
            else:
                # Default: leave open
                self._verify_all_stops()

        except (KeyboardInterrupt, EOFError):
            print("\nLeaving positions open with stops active.")
            self._verify_all_stops()

    def _close_all_positions(self):
        """Close all open positions at market."""
        if not self.trading_client:
            return

        print("\nClosing all positions...")

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                # Cancel stop order first
                self.cancel_stop_order(symbol)

                # Sell position - get ACTUAL quantity from Alpaca
                alpaca_symbol = symbol.replace('USD', '/USD')
                try:
                    actual_pos = self.trading_client.get_open_position(alpaca_symbol)
                    actual_qty = float(actual_pos.qty)
                except:
                    actual_qty = pos['total_qty']

                order_data = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    qty=actual_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )

                order = self.trading_client.submit_order(order_data)
                fill_info = self.wait_for_fill(order.id)

                if fill_info:
                    exit_price = fill_info['filled_avg_price']
                    pnl = actual_qty * (exit_price - pos['avg_price'])
                    print(f"  {symbol}: CLOSED @ ${exit_price:,.2f} | P&L: ${pnl:,.2f}")
                    self.stats['exits_manual'] += 1
                    self.stats['total_pnl'] += pnl
                else:
                    print(f"  {symbol}: CLOSE FAILED - check manually!")

                del self.positions[symbol]

            except Exception as e:
                print(f"  {symbol}: Error closing: {e}")

        print("\nAll positions closed.")

    def _close_losing_positions(self):
        """Close only positions that are currently in loss."""
        if not self.trading_client:
            return

        print("\nClosing losing positions...")

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self.get_current_price(symbol)
            if current_price <= 0:
                continue

            pnl = pos['total_qty'] * (current_price - pos['avg_price'])

            if pnl < 0:
                try:
                    # Cancel stop order first
                    self.cancel_stop_order(symbol)

                    # Sell position - get ACTUAL quantity from Alpaca
                    alpaca_symbol = symbol.replace('USD', '/USD')
                    try:
                        actual_pos = self.trading_client.get_open_position(alpaca_symbol)
                        actual_qty = float(actual_pos.qty)
                    except:
                        actual_qty = pos['total_qty']

                    order_data = MarketOrderRequest(
                        symbol=alpaca_symbol,
                        qty=actual_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )

                    order = self.trading_client.submit_order(order_data)
                    fill_info = self.wait_for_fill(order.id)

                    if fill_info:
                        exit_price = fill_info['filled_avg_price']
                        actual_pnl = actual_qty * (exit_price - pos['avg_price'])
                        print(f"  {symbol}: CLOSED LOSER @ ${exit_price:,.2f} | P&L: ${actual_pnl:,.2f}")
                        self.stats['exits_manual'] += 1
                        self.stats['total_pnl'] += actual_pnl

                    del self.positions[symbol]

                except Exception as e:
                    print(f"  {symbol}: Error closing: {e}")
            else:
                print(f"  {symbol}: Keeping (in profit: ${pnl:,.2f})")

        # Verify stops on remaining positions
        self._verify_all_stops()

    def _cancel_all_stops(self):
        """Cancel all stop orders but keep positions open."""
        print("\nCanceling all stop orders...")

        for symbol in list(self.stop_orders.keys()):
            try:
                self.cancel_stop_order(symbol)
                print(f"  {symbol}: Stop order canceled")
            except Exception as e:
                print(f"  {symbol}: Error canceling stop: {e}")

        print("\nWARNING: Positions are now UNPROTECTED!")
        print("         Manage manually in Alpaca dashboard.")

    def _verify_all_stops(self):
        """Verify all positions have active stop orders."""
        print("\nVerifying stop orders...")

        for symbol, pos in self.positions.items():
            if symbol in self.stop_orders:
                print(f"  {symbol}: Stop order active @ ${pos['stop_loss']:,.2f}")
            else:
                # Try to place stop
                print(f"  {symbol}: No stop! Placing @ ${pos['stop_loss']:,.2f}...")
                sl_order_id = self.place_stop_loss_order(symbol, pos['total_qty'], pos['stop_loss'])
                if sl_order_id:
                    self.stop_orders[symbol] = sl_order_id
                    print(f"  {symbol}: Stop placed successfully")
                else:
                    print(f"  {symbol}: WARNING - Failed to place stop!")

        print("\nPositions will be managed by stop orders.")
        print("Bot can be restarted to resume management.")

    def run(self, interval_seconds: int = 60, max_hours: float = 24):
        """Run paper trading loop with comprehensive logging."""
        print("\n" + "=" * 70)
        print("STARTING DCA PAPER TRADING")
        print("=" * 70)
        print(f"Symbols: {self.config.symbols}")
        print(f"Scan Interval: {interval_seconds} seconds")
        print(f"Max Runtime: {max_hours} hours")
        print(f"\nStrategy Settings:")
        print(f"  Momentum Threshold: >{self.config.momentum_threshold}%")
        print(f"  RSI Max: <{self.config.rsi_max_for_entry}")
        print(f"  TP: {self.config.take_profit_pct*100:.2f}%")
        print(f"  SL: {self.config.stop_loss_pct*100:.2f}%")
        print(f"  DCA Spacing: {self.config.dca_spacing_pct*100:.2f}%")
        print(f"  Max DCA Stages: {self.config.max_dca_stages}")
        print(f"  Cooldown: {self.config.cooldown_bars} bars ({self.config.cooldown_bars} min)")

        # Smart DCA Exit Settings
        tp_reduction = getattr(self.config, 'tp_reduction_per_dca', 0.001)
        min_tp = getattr(self.config, 'min_tp_pct', 0.0025)
        be_threshold = getattr(self.config, 'breakeven_dca_threshold', 3)
        emergency_dca = getattr(self.config, 'emergency_exit_dca_level', 4)
        emergency_profit = getattr(self.config, 'emergency_exit_profit_pct', 0.0015)

        print(f"\nSmart DCA Exit Settings:")
        print(f"  Dynamic TP: -{tp_reduction*100:.1f}% per DCA (min {min_tp*100:.2f}%)")
        print(f"  Breakeven Lock: DCA {be_threshold}+ at +{getattr(self.config, 'breakeven_buffer_pct', 0.001)*100:.1f}%")
        print(f"  Emergency Exit: DCA {emergency_dca}+ at +{emergency_profit*100:.2f}% profit")

        # Smart DCA Trend Filter Settings
        if self.trend_filter:
            print(f"\nSmart DCA Trend Filter:")
            print(f"  ADX Strong Trend: >{getattr(self.config, 'adx_strong_trend', 30.0)} (blocks DCA)")
            print(f"  ADX Weak Trend: <{getattr(self.config, 'adx_weak_trend', 20.0)} (safe to DCA)")
            print(f"  EMA: {getattr(self.config, 'ema_fast_period', 8)}/{getattr(self.config, 'ema_slow_period', 21)} (reversal detection)")
            print(f"  Require Reversal Candle: {getattr(self.config, 'require_reversal_candle', True)}")
            print(f"  High DCA Confirmation: Level {getattr(self.config, 'high_dca_confirmation_level', 3)}+")
            print(f"  Entry Trend Validation: {getattr(self.config, 'validate_entry_trend', True)}")
        else:
            print(f"\nSmart DCA Trend Filter: DISABLED")

        max_runtime = timedelta(hours=max_hours)
        iteration = 0

        try:
            while datetime.now() - self.session_start < max_runtime:
                iteration += 1

                # Display header
                self.display_comprehensive_status()

                for symbol in self.config.symbols:
                    # First check if SL was triggered
                    if symbol in self.positions:
                        if self.check_sl_triggered(symbol):
                            continue

                    # Check existing positions for TP/DCA
                    if symbol in self.positions:
                        action, details = self.check_tp_and_dca(symbol)

                        if action == 'EMERGENCY_EXIT':
                            print(f"  {symbol}: {details.get('exit_reason', 'EMERGENCY EXIT')}")
                            self.execute_emergency_exit(symbol, details['current_price'], details.get('exit_reason', 'EMERGENCY EXIT'))

                        elif action == 'BREAKEVEN_LOCK':
                            print(f"  {symbol}: {details.get('exit_reason', 'BREAKEVEN LOCK')}")
                            self.execute_breakeven_lock(symbol, details['new_sl'])

                        elif action == 'TP':
                            print(f"  {symbol}: POSITION OPEN (BUY) - TP HIT! ({details['tp_pct']:.2f}%)")
                            self.execute_tp_exit(symbol, details['current_price'])

                        elif action == 'DCA':
                            print(f"  {symbol}: POSITION OPEN (BUY) - DCA TRIGGER!")
                            self.execute_dca(symbol, details['next_dca'])

                        elif action == 'HOLD':
                            pnl_str = f"${details['pnl']:+,.2f}" if details['pnl'] >= 0 else f"${details['pnl']:,.2f}"
                            tp_pct = details.get('tp_pct', self.config.take_profit_pct * 100)
                            print(f"  {symbol}: POSITION OPEN (BUY) @ ${details['avg_price']:,.2f} | "
                                  f"P&L: {pnl_str} ({details['pnl_pct']:+.2f}%) | DCA: {details['dca_stage']} | TP: {tp_pct:.2f}%")

                    else:
                        # Check for new entry signals
                        if len(self.positions) < self.config.max_open_trades:
                            has_signal, momentum, rsi, price = self.check_signal(symbol)

                            if has_signal:
                                print(f"  {symbol}: SIGNAL! Mom={momentum:.2f}% RSI={rsi:.1f} @ ${price:,.2f}")
                                self.enter_position(symbol, momentum, rsi, price)
                            else:
                                # Show current status
                                if price > 0:
                                    status = "Below threshold" if momentum <= self.config.momentum_threshold else "RSI too high"
                                    print(f"  {symbol}: Mom={momentum:.2f}% RSI={rsi:.1f} @ ${price:,.2f} ({status})")
                                else:
                                    print(f"  {symbol}: No data")
                        else:
                            print(f"  {symbol}: Max positions reached ({self.config.max_open_trades})")

                # Display detailed position status
                self.display_position_status()

                # Display session stats every 5 iterations
                if iteration % 5 == 0:
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

            # Handle shutdown with position options
            self._handle_shutdown()


def main():
    print("=" * 70)
    print("DCA MOMENTUM STRATEGY - PAPER TRADING")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load configuration
    config = load_dca_config()

    print(f"\nConfiguration:")
    print(f"  Symbols: {config.symbols}")
    print(f"  Capital: ${config.initial_capital:,.2f}")
    print(f"  Risk per trade: {config.risk_per_trade_pct*100:.1f}%")
    print(f"  TP: {config.take_profit_pct*100:.2f}%")
    print(f"  SL: {config.stop_loss_pct*100:.2f}%")
    print(f"  DCA spacing: {config.dca_spacing_pct*100:.2f}%")
    print(f"  Momentum threshold: >{config.momentum_threshold}%")
    print(f"  RSI max: <{config.rsi_max_for_entry}")

    # Show position sizing breakdown
    print(f"\nPosition Sizing (per symbol):")
    num_symbols = len(config.symbols)
    total_mult = sum(config.dca_multipliers)
    symbol_alloc = config.initial_capital / num_symbols
    base_unit = symbol_alloc / total_mult
    print(f"  Symbol Allocation: ${symbol_alloc:,.2f} ({num_symbols} symbols)")
    print(f"  Total DCA Multipliers: {total_mult:.2f}x")
    print(f"  Base Unit: ${base_unit:,.2f}")
    print(f"  ---")
    stage_names = ['Entry'] + [f'DCA {i}' for i in range(1, len(config.dca_multipliers))]
    cumulative = 0
    for i, (name, mult) in enumerate(zip(stage_names, config.dca_multipliers)):
        pos_val = base_unit * mult
        cumulative += pos_val
        print(f"  {name}: ${pos_val:,.2f} ({mult*100:.1f}% mult) | Cumulative: ${cumulative:,.2f}")

    # Create engine
    engine = DCALivePaperEngine(config)

    if not engine.trading_client:
        print("\nERROR: Could not connect to Alpaca. Check credentials.")
        print("\nRequired environment variables:")
        print("  ALPACA_CRYPTO_KEY=your_key")
        print("  ALPACA_CRYPTO_SECRET=your_secret")
        return

    # Show any resumed positions
    if engine.positions:
        print(f"\n*** RESUMED {len(engine.positions)} EXISTING POSITION(S) ***")
        print("Stop losses have been verified/placed for all positions.")

    # Confirmation
    print("\n" + "=" * 70)
    print("Ready to start PAPER trading.")
    print("This will place REAL orders on your PAPER account.")
    print("Press Ctrl+C at any time to stop.")
    print("\nOn shutdown, you can choose to:")
    print("  - Leave positions open (stops remain active)")
    print("  - Close all positions")
    print("  - Close only losing positions")
    print("=" * 70)

    try:
        input("\nPress Enter to start (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\nCancelled")
        return

    # Run paper trading
    engine.run(interval_seconds=60, max_hours=24)

    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
