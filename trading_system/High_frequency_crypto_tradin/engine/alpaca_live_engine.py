"""
Alpaca Live Trading Engine for High-Frequency Crypto Trading
==============================================================

Connects to Alpaca Paper Trading API for real-time crypto trading.
Integrates with the ML ensemble for signal generation.
USES CENTRALIZED RISK MANAGEMENT for consistent risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import threading
import logging
import json
from pathlib import Path

# Import centralized risk management
try:
    from ..risk_management import (
        RiskManager, RiskConfig, PositionManager,
        TradeExecutor, ExecutionMode, PortfolioRiskAnalyzer
    )
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    crypto_symbols: List[str] = None

    def __post_init__(self):
        if self.crypto_symbols is None:
            self.crypto_symbols = [
                "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"
            ]


class AlpacaLiveEngine:
    """
    Live trading engine using Alpaca API for SPOT TRADING (BUY-only with DCA).

    Features:
    - BUY-ONLY trading (no short selling for spot crypto)
    - DCA (Dollar Cost Averaging) for trades going against us
    - CENTRALIZED RISK MANAGEMENT (same as backtest)
    - Real-time crypto data streaming
    - Order execution with error handling
    - Position management with SL on exchange, TP monitored locally
    - P&L tracking
    - Integration with ML ensemble

    Strategy:
    - Enter on BUY signal
    - If price drops, average down with DCA at defined intervals
    - Exit when averaged position is in profit
    - Stop Loss placed on Alpaca exchange for protection
    - Take Profit monitored locally and executed when hit
    """

    def __init__(self,
                 config: AlpacaConfig,
                 strategy=None,
                 ensemble=None,
                 feature_engineer=None,
                 risk_config: 'RiskConfig' = None):
        """
        Initialize the Alpaca Live Engine.

        Args:
            config: AlpacaConfig with API credentials
            strategy: HFTradingStrategy instance
            ensemble: EnsembleVotingSystem instance
            feature_engineer: FeatureEngineer instance
            risk_config: RiskConfig for centralized risk management
        """
        self.config = config
        self.strategy = strategy
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer

        self.api = None
        self.crypto_api = None
        self.data_stream = None

        self.is_running = False
        self.positions: Dict[str, dict] = {}
        self.orders: Dict[str, dict] = {}

        # Initialize centralized risk management
        if HAS_RISK_MANAGEMENT:
            self.risk_config = risk_config or RiskConfig()
            self.risk_manager = RiskManager(self.risk_config)
            self.position_manager = PositionManager(self.risk_manager)
            self.portfolio_analyzer = PortfolioRiskAnalyzer(
                self.risk_manager, self.position_manager
            )
            logger.info("Centralized risk management initialized")
        else:
            self.risk_manager = None
            self.position_manager = None
            self.portfolio_analyzer = None
            logger.warning("Centralized risk management not available")
        self.account: dict = {}
        self.bars_buffer: Dict[str, List[dict]] = {s: [] for s in config.crypto_symbols}

        self.trade_log: List[dict] = []
        self.pnl_history: List[dict] = []

        # Local TP/SL tracking (Alpaca doesn't support bracket orders for crypto)
        # For DCA: tracks average entry price and DCA levels
        self.position_targets: Dict[str, dict] = {}  # {symbol: {entry, tp, sl, direction, dca_info}}

        # DCA tracking for live trading
        self.dca_positions: Dict[str, dict] = {}  # {symbol: {dca_count, original_entry, avg_price, total_qty, dca_levels}}

        self._init_alpaca()

    def _init_alpaca(self):
        """Initialize Alpaca API client."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.crypto import CryptoHistoricalDataClient
            from alpaca.data.live.crypto import CryptoDataStream

            # Trading client
            self.api = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret,
                paper=True
            )

            # Data client
            self.crypto_api = CryptoHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret
            )

            # Data stream for real-time
            self.data_stream = CryptoDataStream(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret
            )

            logger.info("Alpaca API initialized successfully")

        except ImportError:
            logger.warning("alpaca-py not installed. Run: pip install alpaca-py")
            self.api = None

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            self.api = None

    def get_account(self) -> dict:
        """Get current account information."""
        if not self.api:
            return {}

        try:
            account = self.api.get_account()
            self.account = {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'currency': account.currency,
                'status': account.status
            }
            return self.account

        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}

    def get_positions(self) -> Dict[str, dict]:
        """Get current positions."""
        if not self.api:
            return {}

        try:
            positions = self.api.get_all_positions()
            self.positions = {}

            for pos in positions:
                self.positions[pos.symbol] = {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }

            return self.positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_historical_bars(self,
                            symbol: str,
                            timeframe: str = "1Min",
                            limit: int = 200) -> pd.DataFrame:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Bar timeframe
            limit: Number of bars

        Returns:
            DataFrame with OHLCV data
        """
        if not self.crypto_api:
            logger.error(f"[{symbol}] Crypto API not initialized")
            return pd.DataFrame()

        try:
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta

            # Calculate start time (need enough bars for features)
            # Request extra minutes to ensure we get enough data
            start_time = datetime.now() - timedelta(minutes=limit + 60)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                limit=limit
            )

            logger.debug(f"[{symbol}] Fetching bars from {start_time}...")
            bars = self.crypto_api.get_crypto_bars(request)

            # Check if we got data - bars is a dict-like object
            if bars and len(bars.data) > 0:
                # Access data properly
                if symbol in bars.data:
                    data = bars.data[symbol]
                elif symbol.replace("/", "") in bars.data:
                    data = bars.data[symbol.replace("/", "")]
                else:
                    # Try to get first available key
                    available_keys = list(bars.data.keys())
                    if available_keys:
                        data = bars.data[available_keys[0]]
                        logger.debug(f"[{symbol}] Using key: {available_keys[0]}")
                    else:
                        logger.warning(f"[{symbol}] No data in response")
                        return pd.DataFrame()

                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap if hasattr(bar, 'vwap') else 0
                } for bar in data])

                df['datetime'] = pd.to_datetime(df['timestamp'])
                logger.info(f"[{symbol}] Fetched {len(df)} bars")
                return df
            else:
                logger.warning(f"[{symbol}] Empty response from Alpaca API")

        except Exception as e:
            logger.error(f"[{symbol}] Failed to get historical bars: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return pd.DataFrame()

    def submit_order(self,
                     symbol: str,
                     qty: float,
                     side: str,
                     order_type: str = "market",
                     time_in_force: str = "gtc",
                     limit_price: float = None,
                     stop_price: float = None,
                     take_profit: float = None,
                     stop_loss: float = None) -> Optional[dict]:
        """
        Submit an order with optional TP/SL (bracket order).

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'gtc', 'day', 'ioc', 'fok'
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            take_profit: Take profit price (creates bracket order)
            stop_loss: Stop loss price (creates bracket order)

        Returns:
            Order details dictionary
        """
        if not self.api:
            logger.error("Alpaca API not initialized")
            return None

        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
            from alpaca.trading.requests import TakeProfitRequest, StopLossRequest

            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce.GTC if time_in_force.lower() == 'gtc' else TimeInForce.DAY

            # NOTE: Alpaca does NOT support bracket orders (OTO/OTOCO/OCO) for crypto
            # TP/SL must be managed manually through price monitoring
            # We will store TP/SL locally and close position when levels are hit

            if take_profit or stop_loss:
                logger.info(f"  [ORDER] TP/SL will be monitored locally (Alpaca crypto limitation)")
                if take_profit:
                    logger.info(f"  [ORDER]   Take Profit: ${take_profit:,.2f}")
                if stop_loss:
                    logger.info(f"  [ORDER]   Stop Loss: ${stop_loss:,.2f}")

            # Submit simple market/limit order (no bracket for crypto)
            if order_type.lower() == 'market':
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            elif order_type.lower() == 'limit':
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            else:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )

            order = self.api.submit_order(request)

            order_info = {
                'id': str(order.id),
                'symbol': order.symbol,
                'qty': float(order.qty) if order.qty else qty,
                'side': order.side.value,
                'type': order.type.value,
                'status': order.status.value,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'submitted_at': str(order.submitted_at),
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }

            self.orders[order_info['id']] = order_info

            logger.info(f"[ORDER] Submitted: {side.upper()} {qty:.6f} {symbol}")
            if take_profit:
                logger.info(f"[ORDER]   Take Profit: ${take_profit:.2f}")
            if stop_loss:
                logger.info(f"[ORDER]   Stop Loss: ${stop_loss:.2f}")

            return order_info

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def place_tp_sl_orders(self,
                           symbol: str,
                           qty: float,
                           entry_side: str,
                           take_profit: float,
                           stop_loss: float) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Place Stop Loss order on Alpaca, monitor Take Profit locally.

        STRATEGY:
        - STOP LOSS: Place as STOP_LIMIT order on Alpaca (CRITICAL for protection)
        - TAKE PROFIT: Monitor locally and close position when price hits TP

        For crypto on Alpaca:
        - Stop Loss: STOP_LIMIT order on opposite side (placed on exchange)
        - Take Profit: Monitored locally by trading engine

        Args:
            symbol: Trading symbol
            qty: Quantity (same as entry position)
            entry_side: 'buy' or 'sell' (the entry side)
            take_profit: Take profit price
            stop_loss: Stop loss price

        Returns:
            Tuple of (tp_order, sl_order)
        """
        if not self.api:
            logger.error("Alpaca API not initialized")
            return None, None

        try:
            from alpaca.trading.requests import StopLimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Exit side is opposite of entry
            exit_side = OrderSide.SELL if entry_side.lower() == 'buy' else OrderSide.BUY
            tif = TimeInForce.GTC

            tp_order = None
            sl_order = None

            # STOP LOSS ORDER - Place on Alpaca (MOST IMPORTANT)
            if stop_loss:
                try:
                    # For stop_limit: stop_price triggers, limit_price is execution price
                    # Use small offset to ensure fill
                    if exit_side == OrderSide.SELL:
                        # For SELL stop: limit slightly below stop to ensure fill
                        sl_limit = stop_loss * 0.998  # 0.2% below stop
                    else:
                        # For BUY stop: limit slightly above stop to ensure fill
                        sl_limit = stop_loss * 1.002  # 0.2% above stop

                    sl_request = StopLimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=exit_side,
                        time_in_force=tif,
                        stop_price=stop_loss,
                        limit_price=sl_limit
                    )
                    sl_result = self.api.submit_order(sl_request)
                    sl_order = {
                        'id': str(sl_result.id),
                        'symbol': sl_result.symbol,
                        'type': 'stop_loss',
                        'price': stop_loss,
                        'status': sl_result.status.value
                    }
                    logger.info(f"[SL ORDER] PLACED on Alpaca: STOP_LIMIT {exit_side.value} @ ${stop_loss:,.2f}")
                except Exception as e:
                    logger.error(f"[SL ORDER] Failed to place: {e}")
                    # If SL order fails, we'll monitor locally
                    sl_order = {
                        'type': 'stop_loss_local',
                        'price': stop_loss,
                        'status': 'monitored_fallback'
                    }
                    logger.info(f"[SL ORDER] Will monitor locally as fallback")

            # TAKE PROFIT - Monitor locally (don't place order)
            if take_profit:
                logger.info(f"[TP] Take Profit @ ${take_profit:,.2f} will be MONITORED LOCALLY")
                tp_order = {
                    'type': 'take_profit_local',
                    'price': take_profit,
                    'status': 'monitored'
                }

            return tp_order, sl_order

        except Exception as e:
            logger.error(f"Failed to place TP/SL orders: {e}")
            return None, None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.api:
            return False

        try:
            self.api.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def close_position(self, symbol: str) -> Optional[dict]:
        """Close an open position."""
        if not self.api:
            return None

        try:
            # Convert symbol format if needed
            symbol_clean = symbol.replace("/", "")

            order = self.api.close_position(symbol_clean)

            order_info = {
                'id': str(order.id),
                'symbol': order.symbol,
                'side': order.side.value,
                'status': order.status.value,
            }

            logger.info(f"Position closed: {symbol}")

            return order_info

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None

    def close_all_positions(self):
        """Close all open positions."""
        if not self.api:
            return

        try:
            self.api.close_all_positions()
            logger.info("All positions closed")

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")

    def start_streaming(self, on_bar: Callable = None, on_trade: Callable = None):
        """
        Start real-time data streaming.

        Args:
            on_bar: Callback for new bars
            on_trade: Callback for trade updates
        """
        if not self.data_stream:
            logger.error("Data stream not initialized")
            return

        async def bar_handler(bar):
            """Handle incoming bars."""
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }

            # Add to buffer
            if bar.symbol in self.bars_buffer:
                self.bars_buffer[bar.symbol].append(bar_data)

                # Keep buffer manageable
                if len(self.bars_buffer[bar.symbol]) > 500:
                    self.bars_buffer[bar.symbol] = self.bars_buffer[bar.symbol][-200:]

            if on_bar:
                on_bar(bar_data)

        async def trade_handler(trade):
            """Handle trade updates."""
            if on_trade:
                trade_data = {
                    'symbol': trade.symbol,
                    'price': trade.price,
                    'size': trade.size,
                    'timestamp': trade.timestamp
                }
                on_trade(trade_data)

        try:
            # Subscribe to bars
            for symbol in self.config.crypto_symbols:
                self.data_stream.subscribe_bars(bar_handler, symbol)
                logger.info(f"Subscribed to bars: {symbol}")

            # Run in background thread
            def run_stream():
                self.data_stream.run()

            stream_thread = threading.Thread(target=run_stream, daemon=True)
            stream_thread.start()

            logger.info("Data streaming started")

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")

    def run_live_trading(self,
                         symbols: List[str] = None,
                         interval_seconds: int = 60,
                         max_runtime_hours: float = 24):
        """
        Run live trading loop.

        Args:
            symbols: Symbols to trade (uses config default if None)
            interval_seconds: Seconds between trading cycles
            max_runtime_hours: Maximum runtime in hours
        """
        symbols = symbols or self.config.crypto_symbols
        self.is_running = True
        start_time = datetime.now()
        max_runtime = timedelta(hours=max_runtime_hours)

        logger.info("=" * 60)
        logger.info("Starting Live Trading")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info("=" * 60)

        # Get initial account info
        account = self.get_account()
        if account:
            logger.info(f"Account Equity: ${account['equity']:,.2f}")

        try:
            iteration = 0
            while self.is_running:
                cycle_start = datetime.now()
                iteration += 1

                # Check runtime limit
                if datetime.now() - start_time > max_runtime:
                    logger.info("Max runtime reached, stopping")
                    break

                # Get LIVE account and positions from Alpaca
                account = self.get_account()
                positions = self.get_positions()
                equity = account.get('equity', 0)
                open_count = len(positions)

                # ML SCANNING HEADER (like Forex system)
                now = datetime.now().strftime('%H:%M:%S')
                print(f"\n{'='*60}")
                print(f"[{now}] ML SCANNING {len(symbols)} PAIRS | Equity: ${equity:,.2f} | Open: {open_count}")
                print(f"{'='*60}")

                for symbol in symbols:
                    try:
                        self._process_symbol(symbol, account, positions)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # SESSION STATUS (like Forex system)
                if len(positions) > 0:
                    total_unrealized = sum(p['unrealized_pl'] for p in positions.values())
                    print(f"\n--- SESSION STATUS (LIVE ALPACA DATA) ---")
                    print(f"OPEN POSITIONS ({len(positions)}) | Unrealized P&L: ${total_unrealized:+,.2f}")
                    for symbol_clean, pos in positions.items():
                        symbol = symbol_clean[:3] + "/" + symbol_clean[3:]  # BTCUSD -> BTC/USD
                        side = pos['side'].upper()
                        qty = pos['qty']
                        entry = pos['avg_entry_price']
                        current = pos['current_price']
                        pnl = pos['unrealized_pl']
                        pnl_pct = pos['unrealized_plpc'] * 100

                        # Get local TP/SL
                        targets = self.position_targets.get(symbol, {})
                        tp = targets.get('tp', 0)
                        sl = targets.get('sl', 0)

                        print(f"  {symbol}: {side} {qty:.8f} @ ${entry:,.2f}")
                        print(f"    Current: ${current:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                        if tp > 0 and sl > 0:
                            tp_pct = abs(tp - current) / current * 100
                            sl_pct = abs(current - sl) / current * 100
                            print(f"    TP: ${tp:,.2f} ({tp_pct:.2f}% away) | SL: ${sl:,.2f} ({sl_pct:.2f}% away)")

                # Wait for next cycle
                elapsed = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, interval_seconds - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Trading stopped by user")

        finally:
            self.is_running = False
            self._save_trading_log()

    def _process_symbol(self, symbol: str, account: dict, positions: dict):
        """Process a single symbol for trading signals - matches Forex ML display format."""
        symbol_clean = symbol.replace("/", "")

        # Get historical data
        df = self.get_historical_bars(symbol, limit=200)
        if df.empty:
            print(f"  {symbol}: No historical data available")
            return

        # Update strategy data buffer
        if self.strategy:
            for _, row in df.iterrows():
                self.strategy.update_data(row.to_dict())

        # Get current price
        current_price = df['close'].iloc[-1]
        equity = account.get('equity', 0)

        # Check existing position
        has_position = symbol_clean in positions

        if has_position:
            pos = positions[symbol_clean]
            pnl_pct = pos['unrealized_plpc'] * 100
            entry_price = pos['avg_entry_price']
            unrealized_pnl = pos['unrealized_pl']

            # Get local TP/SL targets
            targets = self.position_targets.get(symbol, {})
            tp = targets.get('tp', 0)
            sl = targets.get('sl', 0)
            direction = targets.get('direction', pos['side'].upper())

            # Calculate TP/SL distances
            if tp > 0:
                tp_pct = abs(tp - current_price) / current_price * 100
            else:
                tp_pct = 0
            if sl > 0:
                sl_pct = abs(current_price - sl) / current_price * 100
            else:
                sl_pct = 0

            print(f"  {symbol}: POSITION OPEN ({direction}) - Checking TP/SL...")
            print(f"    Entry: ${entry_price:,.2f} | Current: ${current_price:,.2f} | P&L: ${unrealized_pnl:+,.2f} ({pnl_pct:+.2f}%)")
            if tp > 0 and sl > 0:
                print(f"    TP: ${tp:,.2f} ({tp_pct:.2f}% away) | SL: ${sl:,.2f} ({sl_pct:.2f}% away)")

            # Check TP/SL (manual monitoring since Alpaca doesn't support bracket for crypto)
            should_exit = False
            reason = ""

            if tp > 0 and direction == 'BUY' and current_price >= tp:
                should_exit = True
                reason = f"Take Profit hit (${tp:,.2f})"
            elif tp > 0 and direction == 'SELL' and current_price <= tp:
                should_exit = True
                reason = f"Take Profit hit (${tp:,.2f})"
            elif sl > 0 and direction == 'BUY' and current_price <= sl:
                should_exit = True
                reason = f"Stop Loss hit (${sl:,.2f})"
            elif sl > 0 and direction == 'SELL' and current_price >= sl:
                should_exit = True
                reason = f"Stop Loss hit (${sl:,.2f})"

            # Also check strategy exit conditions
            if not should_exit and self.strategy:
                should_exit, reason = self.strategy.check_exit_conditions(current_price)

            if should_exit:
                print(f"  {symbol}: >>> EXIT SIGNAL - {reason} <<<")
                self.close_position(symbol)

                # Cancel any pending SL orders
                if symbol in self.position_targets:
                    sl_order_id = self.position_targets[symbol].get('sl_order_id')
                    if sl_order_id:
                        self.cancel_order(sl_order_id)
                    del self.position_targets[symbol]

                # Clean up DCA tracking
                if symbol in self.dca_positions:
                    del self.dca_positions[symbol]

                # Update strategy
                pnl = pos['unrealized_pl']
                self.strategy.on_trade_closed(pnl)

                # Log trade
                self._log_trade(symbol, 'close', current_price, pnl, reason)
            else:
                # Check for DCA opportunity (if enabled and position is losing)
                if self.risk_manager and hasattr(self.risk_manager.config, 'dca_enabled') and self.risk_manager.config.dca_enabled:
                    should_dca, dca_reason = self._check_dca_opportunity(symbol, current_price, pos)
                    if should_dca:
                        print(f"  {symbol}: >>> DCA OPPORTUNITY - {dca_reason} <<<")
                        self._execute_dca(symbol, current_price, pos)
                    else:
                        print(f"  {symbol}: Hold position (within TP/SL range)")
                else:
                    print(f"  {symbol}: Hold position (within TP/SL range)")

        else:
            # Check for entry signals - ML SCANNING
            if self.strategy:
                signal = self.strategy.generate_signal(
                    current_price=current_price,
                    account_equity=equity,
                    symbol=symbol
                )

                if signal:
                    # Show ML decision
                    direction = 'BUY' if signal.signal.value == 1 else 'SELL'
                    print(f"  {symbol}: ML={direction} Conf={signal.confidence:.0%} Agree={signal.model_agreement}/5 @ ${current_price:,.2f}")

                    # Execute trade with TP/SL
                    side = 'buy' if signal.signal.value == 1 else 'sell'

                    print(f"  {symbol}: >>> EXECUTING {direction} TRADE <<<")

                    order = self.submit_order(
                        symbol=symbol,
                        qty=signal.position_size,
                        side=side,
                        order_type='market',
                        take_profit=signal.take_profit,
                        stop_loss=signal.stop_loss
                    )

                    if order:
                        # Calculate TP/SL percentages
                        tp_pct = abs(signal.take_profit - current_price) / current_price * 100
                        sl_pct = abs(signal.stop_loss - current_price) / current_price * 100

                        print(f"  {symbol}: Entry: ${current_price:,.2f} | TP: ${signal.take_profit:,.2f} ({tp_pct:.2f}%) | SL: ${signal.stop_loss:,.2f} ({sl_pct:.2f}%)")

                        # Place TP and SL orders separately (Alpaca doesn't support bracket for crypto)
                        print(f"  {symbol}: Placing TP/SL orders...")
                        tp_order, sl_order = self.place_tp_sl_orders(
                            symbol=symbol,
                            qty=signal.position_size,
                            entry_side=side,
                            take_profit=signal.take_profit,
                            stop_loss=signal.stop_loss
                        )

                        # Store TP/SL order IDs locally for tracking
                        self.position_targets[symbol] = {
                            'entry': current_price,
                            'tp': signal.take_profit,
                            'sl': signal.stop_loss,
                            'direction': direction,
                            'tp_order_id': tp_order.get('id') if tp_order else None,
                            'sl_order_id': sl_order.get('id') if sl_order else None,
                            'opened_at': datetime.now().isoformat()
                        }

                        if tp_order and sl_order:
                            print(f"  {symbol}: TP/SL orders placed successfully!")
                        elif tp_order or sl_order:
                            print(f"  {symbol}: Partial TP/SL orders placed (one failed)")
                        else:
                            print(f"  {symbol}: TP/SL orders failed - will monitor locally")

                        # Update strategy state
                        from .hf_trading_strategy import SignalType
                        position_type = SignalType.BUY if side == 'buy' else SignalType.SELL

                        self.strategy.update_position_state(
                            current_price=current_price,
                            position=position_type,
                            entry_price=current_price,
                            entry_time=datetime.now(),
                            position_size=signal.position_size
                        )

                        # Log trade
                        self._log_trade(symbol, side, current_price, 0, signal.reason)
                    else:
                        print(f"  {symbol}: Order FAILED")
                else:
                    # No signal or below threshold
                    if hasattr(self.strategy, 'state'):
                        conf = getattr(self.strategy.state, 'last_confidence', 0) or 0
                        agree = getattr(self.strategy.state, 'last_agreement', 0) or 0
                        print(f"  {symbol}: ML=HOLD Conf={conf:.0%} Agree={agree}/5 @ ${current_price:,.2f}")
                    else:
                        print(f"  {symbol}: ML=HOLD @ ${current_price:,.2f}")

    def _log_trade(self, symbol: str, action: str, price: float, pnl: float, reason: str):
        """Log a trade."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.trade_log.append(trade)
        logger.info(f"Trade: {action} {symbol} @ ${price:.2f} (P&L: ${pnl:.2f}) - {reason}")

    def _check_dca_opportunity(self, symbol: str, current_price: float, position: dict) -> Tuple[bool, str]:
        """
        Check if we should add to position via DCA (Dollar Cost Averaging).

        Args:
            symbol: Trading symbol
            current_price: Current market price
            position: Current position data from Alpaca

        Returns:
            Tuple of (should_dca, reason)
        """
        if not self.risk_manager or not hasattr(self.risk_manager.config, 'dca_enabled'):
            return False, ""

        if not self.risk_manager.config.dca_enabled:
            return False, ""

        # Initialize DCA tracking if not exists
        if symbol not in self.dca_positions:
            self.dca_positions[symbol] = {
                'dca_count': 0,
                'original_entry': position['avg_entry_price'],
                'avg_price': position['avg_entry_price'],
                'total_qty': position['qty'],
                'dca_levels': [position['avg_entry_price']]
            }

        dca_info = self.dca_positions[symbol]
        config = self.risk_manager.config

        # Check if max DCA reached
        if dca_info['dca_count'] >= config.max_dca_entries:
            return False, f"Max DCA ({config.max_dca_entries}) reached"

        # Calculate next DCA trigger price
        if dca_info['dca_count'] == 0:
            # First DCA trigger
            dca_trigger_price = dca_info['original_entry'] * (1 - config.dca_trigger_pct)
        else:
            # Subsequent DCA triggers
            last_dca_price = dca_info['dca_levels'][-1]
            dca_trigger_price = last_dca_price * (1 - config.dca_spacing_pct)

        # Check if price has dropped to trigger
        if current_price <= dca_trigger_price:
            drop_pct = (dca_info['original_entry'] - current_price) / dca_info['original_entry'] * 100
            return True, f"Price dropped {drop_pct:.1f}% from entry - DCA #{dca_info['dca_count'] + 1}"

        return False, ""

    def _execute_dca(self, symbol: str, current_price: float, position: dict):
        """
        Execute a DCA (Dollar Cost Averaging) buy order.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            position: Current position data from Alpaca
        """
        if symbol not in self.dca_positions:
            return

        dca_info = self.dca_positions[symbol]
        config = self.risk_manager.config
        equity = self.account.get('equity', 0)

        # Calculate DCA position size
        base_size = equity * config.base_position_size_pct
        dca_size = base_size * (config.dca_multiplier ** (dca_info['dca_count'] + 1))
        dca_size = min(dca_size, config.max_position_value)

        # Calculate quantity
        dca_qty = dca_size / current_price

        print(f"  {symbol}: DCA #{dca_info['dca_count'] + 1} - Adding ${dca_size:.2f} ({dca_qty:.8f})")

        # Submit DCA order
        order = self.submit_order(
            symbol=symbol,
            qty=dca_qty,
            side='buy',  # Always BUY for DCA in spot trading
            order_type='market'
        )

        if order:
            # Update DCA tracking
            old_qty = dca_info['total_qty']
            old_cost = dca_info['avg_price'] * old_qty

            new_qty = old_qty + dca_qty
            new_cost = old_cost + dca_size
            new_avg_price = new_cost / new_qty

            dca_info['dca_count'] += 1
            dca_info['total_qty'] = new_qty
            dca_info['avg_price'] = new_avg_price
            dca_info['dca_levels'].append(current_price)

            # Update local targets with new average price for TP
            if symbol in self.position_targets:
                new_tp = new_avg_price * (1 + config.dca_profit_target_pct)
                self.position_targets[symbol]['entry'] = new_avg_price
                self.position_targets[symbol]['tp'] = new_tp
                # Keep SL wide for DCA
                self.position_targets[symbol]['sl'] = new_avg_price * (1 - config.stop_loss_pct)

                print(f"  {symbol}: New avg: ${new_avg_price:.2f} | New TP: ${new_tp:.2f}")

            # Cancel old SL and place new one at adjusted level
            old_sl_id = self.position_targets.get(symbol, {}).get('sl_order_id')
            if old_sl_id:
                self.cancel_order(old_sl_id)

            # Place new SL order with updated quantity
            new_sl = new_avg_price * (1 - config.stop_loss_pct)
            _, sl_order = self.place_tp_sl_orders(
                symbol=symbol,
                qty=new_qty,
                entry_side='buy',
                take_profit=0,  # TP monitored locally
                stop_loss=new_sl
            )

            if sl_order:
                self.position_targets[symbol]['sl_order_id'] = sl_order.get('id')

            # Log DCA trade
            self._log_trade(symbol, f'dca_{dca_info["dca_count"]}', current_price, 0,
                           f"DCA #{dca_info['dca_count']} - New avg: ${new_avg_price:.2f}")

            print(f"  {symbol}: DCA #{dca_info['dca_count']} executed successfully!")
        else:
            print(f"  {symbol}: DCA order FAILED")

    def _save_trading_log(self):
        """Save trading log to file."""
        if not self.trade_log:
            return

        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"trading_log_{timestamp}.json"

        with open(log_file, 'w') as f:
            json.dump({
                'trades': self.trade_log,
                'summary': {
                    'total_trades': len(self.trade_log),
                    'start_time': self.trade_log[0]['timestamp'] if self.trade_log else None,
                    'end_time': self.trade_log[-1]['timestamp'] if self.trade_log else None
                }
            }, f, indent=2)

        logger.info(f"Trading log saved to {log_file}")

    def stop(self):
        """Stop the trading engine."""
        self.is_running = False
        logger.info("Trading engine stopped")


def create_engine_from_env():
    """Create engine from environment variables using centralized config."""
    import os
    from dotenv import load_dotenv
    from trading_system.High_frequency_crypto_tradin.config import load_config

    load_dotenv()

    # Load centralized config for symbols
    trading_config = load_config()

    config = AlpacaConfig(
        api_key=os.getenv('ALPACA_CRYPTO_KEY', ''),
        api_secret=os.getenv('ALPACA_CRYPTO_SECRET', ''),
        base_url=os.getenv('ALPACA_CRYPTO_BASE_URL', 'https://paper-api.alpaca.markets'),
        crypto_symbols=trading_config.symbols  # Use centralized config symbols
    )

    return AlpacaLiveEngine(config)


if __name__ == "__main__":
    # Test the engine
    print("Testing Alpaca Live Engine...")

    engine = create_engine_from_env()

    if engine.api:
        account = engine.get_account()
        print(f"\nAccount Info:")
        print(f"  Equity: ${account.get('equity', 0):,.2f}")
        print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")

        positions = engine.get_positions()
        print(f"\nPositions: {len(positions)}")

        # Get some historical data
        df = engine.get_historical_bars("BTC/USD", limit=10)
        print(f"\nRecent BTC bars:")
        print(df.tail())
    else:
        print("Alpaca API not available")
