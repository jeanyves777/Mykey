"""
HTF Confluence Live Trading Engine for Forex (OANDA)
====================================================
Live trading engine for the HTF Trend + Confluence Strategy, adapted for Forex markets.

Strategy (IDENTICAL to Binance version):
- HTF (1H): 21/50 EMA crossover trend filter
- LTF (15m): MACD + RSI + EMA (9/21) confluence entry
- 5m confirmation + momentum + volume filters
- 8/8 confluence requirement for entry

Features:
- Multi-symbol Forex trading (EUR_USD, GBP_USD, etc.)
- OANDA API v20 integration
- Pip-based TP/SL management (80 pips TP, 200 pips SL default)
- Smart profit lock (close early if trend reverses in profit)
- Trailing profit lock (never give back too much profit)
- Fakeout protection (exit early on suspected fakeouts)
- ML data logging for training
"""

import os
import sys
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.oanda_client import OANDAClient
from engine.htf_confluence_strategy import (
    HTFConfluenceForexStrategy,
    TrendDirection,
    SignalStrength,
    ConfluenceSignal
)
from engine.trade_tracker import TradeTracker
from config.trading_config import (
    FOREX_SYMBOLS,
    SYMBOL_CONFIG,
    get_symbol_config,
    OANDA_CONFIG,
    SESSION_TRADING
)

# Configure logging with file output
log_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
logger.handlers.clear()

# Console handler (for screen output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler with rotation (main trading log)
file_handler = RotatingFileHandler(
    'logs/trading_system.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Detailed analysis log (for debugging)
analysis_handler = RotatingFileHandler(
    'logs/analysis_detailed.log',
    maxBytes=50*1024*1024,  # 50MB
    backupCount=3
)
analysis_handler.setFormatter(log_formatter)

# Create separate logger for detailed analysis
analysis_logger = logging.getLogger('analysis')
analysis_logger.setLevel(logging.INFO)
analysis_logger.addHandler(analysis_handler)
analysis_logger.addHandler(console_handler)  # Also show on console


class HTFConfluenceForexEngine:
    """
    Live Trading Engine for HTF Confluence Strategy on Forex markets.
    
    Adapted from Binance version with identical strategy logic:
    - Same 8/8 confluence conditions
    - Same smart entry filters
    - Same smart exit system
    - Pip-based TP/SL instead of ROI-based
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        testnet: bool = True,
        total_capital: float = None,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize live trading engine.
        
        Args:
            symbols: List of forex pairs to trade (default: from config)
            testnet: Use practice account (default: True for safety)
            total_capital: Total capital to use (None = use account balance)
            risk_per_trade: Risk per trade as fraction (default: 2%)
        """
        self.symbols = symbols or FOREX_SYMBOLS
        self.testnet = testnet
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade
        
        # Get symbol-specific configs
        self.symbol_configs = {}
        for symbol in self.symbols:
            self.symbol_configs[symbol] = get_symbol_config(symbol)
        
        # Initialize OANDA client
        self.client = OANDAClient(
            practice=OANDA_CONFIG["practice"]
        )
        
        # Initialize strategy for each symbol
        self.strategies = {}
        for symbol in self.symbols:
            cfg = self.symbol_configs[symbol]
            self.strategies[symbol] = HTFConfluenceForexStrategy(
                leverage=cfg["leverage"],
                tp_pips=cfg["tp_pips"],
                sl_pips=cfg["sl_pips"],
                min_confluence_score=cfg.get("min_confluence_score", 4)  # Use config value
            )
        
        # Position tracking
        self.positions = {}  # symbol -> position info
        self.open_trades = {}  # symbol -> trade info
        
        # Cooldown tracking (prevent over-trading)
        self.last_trade_time = {}  # symbol -> datetime
        self.cooldown_minutes = 0  # No cooldown - trade whenever signal appears
        
        # Daily loss protection - Per-Symbol (from backtest optimization)
        self.per_symbol_daily_losses = {}  # {symbol: {'date': str, 'losses': int}}
        self.max_daily_losses_per_symbol = 1  # Max 1 loss per symbol per day
        # Each symbol trades independently
        
        # Statistics
        self.stats_file = os.path.join(os.path.dirname(__file__), "session_stats.json")
        self._load_stats()
        
        # Running state
        self.running = False
        self.check_interval = 60  # Check every 60 seconds

        # Initialize trade tracker for comprehensive logging
        self.tracker = TradeTracker(data_dir="data")
        logger.info("Trade Tracker initialized for comprehensive logging")
        
        # SMART EXIT SETTINGS (identical to Binance)
        
        # SMART PROFIT LOCK - Close early if trend reverses while in profit
        self.use_profit_lock = True
        self.profit_lock_min_pips = 40  # Minimum 40 pips profit to consider profit lock
        
        # SMART FAKEOUT PROTECTION - Exit early on suspected fakeout entries
        self.use_fakeout_protection = True
        self.reversal_cycles = {}  # symbol -> count of cycles HTF has been reversed
        self.reversal_cycle_threshold = 5  # Wait 5 cycles before acting
        self.breakeven_pips_threshold = 30  # Need 30 pips for breakeven
        self.small_profit_exit_pips = 10  # Only exit at 10 pips - be patient
        self.damage_control_pips = -40  # Wider tolerance -40 pips
        
        # TRAILING PROFIT LOCK - Never give back too much profit
        self.use_trailing_profit_lock = True
        self.peak_pips = {}  # symbol -> highest pips reached
        self.trailing_lock_activation = 60  # Start trailing after 60 pips
        self.trailing_lock_distance = 30  # Close if drops 30 pips from peak
        self.trailing_lock_min_floor = 30  # Never let floor go below 30 pips
        
        logger.info("=" * 60)
        logger.info("HTF CONFLUENCE FOREX LIVE TRADING ENGINE")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Mode: {'PRACTICE' if testnet else 'LIVE'}")
        logger.info(f"Risk: {risk_per_trade*100}% per trade")
        logger.info(f"Daily Loss Protection: Max {self.max_daily_losses_per_symbol} loss per symbol per day")
        logger.info(f"Symbol Independence: Each symbol trades independently")
        logger.info(f"Risk per trade: {self.risk_per_trade * 100:.0f}%")
        logger.info("-" * 60)
        if SESSION_TRADING["enabled"]:
            sessions = ", ".join(SESSION_TRADING["allowed_sessions"])
            logger.info(f"TRADING SESSIONS: {sessions} ONLY")
            for session_name in SESSION_TRADING["allowed_sessions"]:
                start, end = SESSION_TRADING["session_hours"][session_name]
                logger.info(f"  {session_name}: {start:02d}:00 - {end:02d}:00 UTC")
            logger.info("-" * 60)
        logger.info("DAILY LOSS PROTECTION:")
        logger.info(f"  Max losses per symbol: {self.max_daily_losses_per_symbol} per day")
        logger.info(f"  Each symbol blocked for rest of day after {self.max_daily_losses_per_symbol} loss")
        logger.info(f"  Other symbols continue trading independently")
        logger.info("-" * 60)
        logger.info("SYMBOL-SPECIFIC SETTINGS:")
        for symbol in self.symbols:
            cfg = self.symbol_configs[symbol]
            tp = cfg["tp_pips"]
            sl = cfg["sl_pips"]
            rr = tp / sl
            logger.info(f"  {symbol}: TP {tp:.0f} pips / SL {sl:.0f} pips ({rr:.1f}:1 R:R)")
        logger.info("-" * 60)
        logger.info("SMART PROFIT LOCK:")
        logger.info(f"  Profit Lock: {'ON' if self.use_profit_lock else 'OFF'} (min: {self.profit_lock_min_pips} pips)")
        logger.info(f"  Action: Close if profit >= {self.profit_lock_min_pips} pips AND HTF trend reverses")
        logger.info("-" * 60)
        logger.info("SMART FAKEOUT PROTECTION:")
        logger.info(f"  Fakeout Protection: {'ON' if self.use_fakeout_protection else 'OFF'}")
        logger.info(f"  Reversal cycles to act: {self.reversal_cycle_threshold}")
        logger.info(f"  Pips >= {self.breakeven_pips_threshold}: Move SL to breakeven")
        logger.info(f"  Pips 0-{self.small_profit_exit_pips} + weak: Small profit exit")
        logger.info(f"  Pips < {self.damage_control_pips}: Damage control (immediate close)")
        logger.info("-" * 60)
        logger.info("TRAILING PROFIT LOCK:")
        logger.info(f"  Trailing Lock: {'ON' if self.use_trailing_profit_lock else 'OFF'}")
        logger.info(f"  Activates at: {self.trailing_lock_activation} pips")
        logger.info(f"  Floor distance: {self.trailing_lock_distance} pips below peak")
        logger.info(f"  Minimum floor: {self.trailing_lock_min_floor} pips")
        logger.info("=" * 60)
    
    def _load_stats(self):
        """Load session stats from file."""
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.system_realized_pnl = 0.0  # Track only OUR trades
        self.symbol_stats = {}
        for symbol in self.symbols:
            self.symbol_stats[symbol] = {"wins": 0, "losses": 0, "pnl": 0.0}
        
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    # Load overall system realized PnL (persists across days)
                    self.system_realized_pnl = data.get("system_realized_pnl", 0.0)
                    
                    # Check if stats are from today
                    if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        self.trades_today = data.get("trades_today", 0)
                        self.wins_today = data.get("wins_today", 0)
                        self.losses_today = data.get("losses_today", 0)
                        self.pnl_today = data.get("pnl_today", 0.0)
                        saved_stats = data.get("symbol_stats", {})
                        for symbol in self.symbols:
                            if symbol in saved_stats:
                                self.symbol_stats[symbol] = saved_stats[symbol]
                        logger.info(f"Loaded session stats: {self.wins_today}W/{self.losses_today}L, PnL: ${self.pnl_today:+.2f}")
                        logger.info(f"System Total Realized PnL: ${self.system_realized_pnl:+,.2f}")
        except Exception as e:
            logger.warning(f"Could not load stats: {e}")
    
    def _save_stats(self):
        """Save session stats to file."""
        try:
            data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "trades_today": self.trades_today,
                "wins_today": self.wins_today,
                "losses_today": self.losses_today,
                "pnl_today": self.pnl_today,
                "system_realized_pnl": self.system_realized_pnl,  # Persists across days
                "symbol_stats": self.symbol_stats
            }
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save stats: {e}")
    
    def get_candles(self, symbol: str, granularity: str, count: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candle data from OANDA.
        
        Args:
            symbol: Forex pair (e.g., EUR_USD)
            granularity: Timeframe (M1, M5, M15, H1, H4)
            count: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data, or None if error
        """
        try:
            candles = self.client.get_candles(symbol, granularity, count)
            if candles is None or (isinstance(candles, pd.DataFrame) and candles.empty):
                return None
            
            # Convert to DataFrame if not already
            if not isinstance(candles, pd.DataFrame):
                df = pd.DataFrame(candles)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
                df = df.astype(float)
            else:
                df = candles
            
            return df
        
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch {granularity} candles: {e}")
            return None
    
    def get_pip_location(self, symbol: str) -> int:
        """
        Get pip decimal location for symbol.
        
        Most pairs: -4 (0.0001)
        JPY pairs: -2 (0.01)
        
        Args:
            symbol: Forex pair
            
        Returns:
            Pip location (-2 or -4)
        """
        if 'JPY' in symbol:
            return -2
        return -4
    
    def calculate_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """
        Calculate position size in units based on risk management.
        
        Args:
            symbol: Forex pair
            stop_loss_pips: Stop loss in pips
            
        Returns:
            Position size in units
        """
        try:
            # Get account info
            account_info = self.client.get_account_summary()
            if not account_info:
                logger.error("Failed to get account info")
                return 0
            
            # Use specified capital or account balance
            capital = self.total_capital or float(account_info.get('balance', 0))
            if capital <= 0:
                logger.error("No capital available")
                return 0
            
            # Calculate position size based on MARGIN USAGE
            # OANDA fixed 50:1 leverage
            # Example: 2% of $5,250 = $105 margin → $5,250 notional → 4,500 units for EUR/USD

            # Get current price
            current_price = 1.0
            try:
                pricing = self.client.get_current_price(symbol)
                if pricing and 'mid' in pricing:
                    current_price = float(pricing['mid'])
            except:
                logger.warning(f"[{symbol}] Could not get current price, using 1.0")

            # Calculate target margin to use
            target_margin = capital * self.risk_per_trade

            # Get OANDA's actual margin rate for this specific pair
            # Different pairs have different margin rates (EUR: 2%, GBP: 5%, etc.)
            margin_rate = self.client.get_margin_rate(symbol)

            # Calculate target notional: Notional = Margin / Margin Rate
            # Example: $102 margin / 0.05 (5%) = $2,040 notional for GBP_USD
            target_notional = target_margin / margin_rate

            # Calculate units based on pair type
            # For XXX_USD pairs (EUR/USD, GBP/USD, NZD/USD): notional = units × price
            # For USD_XXX pairs (USD/JPY, USD/CAD): notional = units (already in USD!)
            if symbol.endswith('_USD'):
                # Quote currency is USD: units × price = notional in USD
                position_size = target_notional / current_price
            else:
                # Base currency is USD: units = notional in USD directly
                position_size = target_notional

            # Round to nearest 100 units (OANDA minimum)
            position_size = round(position_size / 100) * 100

            # Ensure minimum position size
            position_size = max(position_size, 100)

            # Calculate actual values for logging
            if symbol.endswith('_USD'):
                final_notional = position_size * current_price
            else:
                final_notional = position_size  # Units already in USD
            final_margin = final_notional * margin_rate

            # Calculate pip value for risk estimation
            pip_location = self.get_pip_location(symbol)
            pip_size = 10 ** pip_location
            if symbol.endswith('_USD'):
                pip_value_per_unit = pip_size
            else:
                pip_value_per_unit = pip_size / current_price

            actual_risk_per_pip = position_size * pip_value_per_unit
            estimated_risk = actual_risk_per_pip * stop_loss_pips

            logger.info(f"[{symbol}] Position sizing: {position_size:,.0f} units")
            logger.info(f"   Price: {current_price:.5f}")
            logger.info(f"   Target margin: ${target_margin:.2f} ({self.risk_per_trade*100}% of capital)")
            logger.info(f"   Actual margin (50:1): ${final_margin:.2f}")
            logger.info(f"   Notional value: ${final_notional:.2f}")
            logger.info(f"   Estimated SL risk: ${estimated_risk:.2f} ({stop_loss_pips:.1f} pips)")

            return position_size
        
        except Exception as e:
            logger.error(f"[{symbol}] Failed to calculate position size: {e}")
            return 0
    
    def check_for_signal(self, symbol: str) -> Optional[ConfluenceSignal]:
        """
        Check for entry signal on a symbol.

        Args:
            symbol: Forex pair to check

        Returns:
            ConfluenceSignal if valid signal, None otherwise
        """
        try:
            # Fetch required timeframes
            ltf_df = self.get_candles(symbol, "M15", 200)  # 15m for entry signals
            htf_df = self.get_candles(symbol, "H1", 100)   # 1H for trend
            m5_df = self.get_candles(symbol, "M5", 100)    # 5m for confirmation

            if ltf_df is None or htf_df is None or m5_df is None:
                logger.warning(f"[{symbol}] Missing candle data")
                return None

            # Add 5m confirmation indicators to ltf indicators
            strategy = self.strategies[symbol]
            m5_indicators = strategy.calculate_indicators(m5_df)
            ltf_indicators = strategy.calculate_indicators(ltf_df)
            htf_indicators = strategy.calculate_indicators(htf_df)

            # Get HTF trend for logging
            htf_trend, htf_strength = strategy.get_htf_trend(htf_df)

            # Get strategy instance
            signal = strategy.should_enter(
                ltf_df=ltf_df,
                htf_df=htf_df,
                pip_location=self.get_pip_location(symbol)
            )

            # Add 5m confirmation to signal indicators
            if m5_indicators:
                signal.indicators["5m_ema_bullish"] = m5_indicators.get("ema_bullish", False)
                signal.indicators["5m_ema_bearish"] = m5_indicators.get("ema_bearish", False)
                signal.indicators["volume_confirmed"] = m5_indicators.get("volume_confirmed", False)
                signal.indicators["volume_ratio"] = m5_indicators.get("volume_ratio", 0)

            # Get current price for market snapshot
            current_price = ltf_df['close'].iloc[-1] if not ltf_df.empty else 0

            # Helper to convert numpy types to Python types
            def to_python(val):
                if hasattr(val, 'item'):
                    return val.item()
                return val

            # Prepare comprehensive analysis data
            htf_trend_str = htf_trend.value if hasattr(htf_trend, 'value') else str(htf_trend)
            htf_strength_str = htf_strength.value if htf_strength and hasattr(htf_strength, 'value') else str(htf_strength) if htf_strength else "unknown"

            all_indicators = {
                "htf_trend": htf_trend_str,
                "htf_strength": htf_strength_str,
                "htf_ema_21": to_python(htf_indicators.get("ema_21", 0)),
                "htf_ema_50": to_python(htf_indicators.get("ema_50", 0)),
                "ltf_macd": to_python(ltf_indicators.get("macd", 0)),
                "ltf_macd_signal": to_python(ltf_indicators.get("macd_signal", 0)),
                "ltf_rsi": to_python(ltf_indicators.get("rsi", 0)),
                "ltf_ema_9": to_python(ltf_indicators.get("ema_9", 0)),
                "ltf_ema_21": to_python(ltf_indicators.get("ema_21", 0)),
                "5m_volume_ratio": to_python(m5_indicators.get("volume_ratio", 0)) if m5_indicators else 0,
                **signal.indicators
            }

            market_state = {
                "price": to_python(current_price),
                "htf_open": to_python(htf_df['open'].iloc[-1]) if not htf_df.empty else 0,
                "htf_high": to_python(htf_df['high'].iloc[-1]) if not htf_df.empty else 0,
                "htf_low": to_python(htf_df['low'].iloc[-1]) if not htf_df.empty else 0,
                "htf_close": to_python(htf_df['close'].iloc[-1]) if not htf_df.empty else 0,
                "ltf_open": to_python(ltf_df['open'].iloc[-1]) if not ltf_df.empty else 0,
                "ltf_high": to_python(ltf_df['high'].iloc[-1]) if not ltf_df.empty else 0,
                "ltf_low": to_python(ltf_df['low'].iloc[-1]) if not ltf_df.empty else 0,
                "ltf_close": to_python(ltf_df['close'].iloc[-1]) if not ltf_df.empty else 0
            }

            signal_data = {
                "action": signal.action,
                "confluence_score": signal.confluence_score,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "entry_price": signal.entry_price,
                "tp_pips": signal.take_profit_pips,
                "sl_pips": signal.stop_loss_pips
            }

            # Log to tracker
            self.tracker.log_signal_analysis(
                symbol=symbol,
                signal_data=signal_data,
                indicators=all_indicators,
                market_state=market_state
            )

            # Record market snapshot for data collection
            price_data = {
                "open": to_python(ltf_df['open'].iloc[-1]) if not ltf_df.empty else 0,
                "high": to_python(ltf_df['high'].iloc[-1]) if not ltf_df.empty else 0,
                "low": to_python(ltf_df['low'].iloc[-1]) if not ltf_df.empty else 0,
                "close": to_python(ltf_df['close'].iloc[-1]) if not ltf_df.empty else 0,
                "volume": to_python(ltf_df['volume'].iloc[-1]) if not ltf_df.empty else 0
            }
            self.tracker.record_market_snapshot(
                symbol=symbol,
                price_data=price_data,
                indicators=all_indicators,
                htf_trend=htf_trend_str,
                ltf_trend="bullish" if ltf_indicators.get("ema_bullish") else "bearish" if ltf_indicators.get("ema_bearish") else "neutral"
            )

            return signal

        except Exception as e:
            logger.error(f"[{symbol}] Error checking for signal: {e}")
            analysis_logger.error(f"[{symbol}] Detailed error in signal check: {e}")
            return None
    
    def execute_trade(self, symbol: str, signal: ConfluenceSignal) -> bool:
        """
        Execute a trade based on signal.
        
        Args:
            symbol: Forex pair
            signal: Entry signal with TP/SL levels
            
        Returns:
            True if trade executed successfully
        """
        try:
            # Calculate position size
            position_size = self.calculate_position_size(symbol, signal.stop_loss_pips)
            if position_size <= 0:
                logger.error(f"[{symbol}] Invalid position size")
                return False
            
            # Log position size calculation
            try:
                account_info = self.client.get_account_summary()
                balance = float(account_info.get('balance', 0)) if account_info else 0
                risk_amount = balance * self.risk_per_trade
                potential_gain = risk_amount * (signal.take_profit_pips / signal.stop_loss_pips)
                
                logger.info(f"[{symbol}] 💼 Position Size Calculation:")
                logger.info(f"   Balance: ${balance:,.2f} | Risk: {self.risk_per_trade*100}% = ${risk_amount:,.2f}")
                logger.info(f"   SL: {signal.stop_loss_pips}p | TP: {signal.take_profit_pips}p | R:R = 1:{signal.take_profit_pips/signal.stop_loss_pips:.1f}")
                logger.info(f"   Position: {abs(position_size):,.0f} units")
                logger.info(f"   Max Loss: ${risk_amount:,.2f} | Max Gain: ${potential_gain:,.2f}")
            except Exception as e:
                logger.debug(f"Could not log position calculation: {e}")
            
            # Determine trade side
            if signal.action == "BUY":
                units = position_size
            elif signal.action == "SELL":
                units = -position_size
            else:
                logger.warning(f"[{symbol}] Invalid signal action: {signal.action}")
                return False
            
            # Calculate TP/SL prices
            pip_location = self.get_pip_location(symbol)
            pip_value = 10 ** pip_location
            
            if signal.action == "BUY":
                tp_price = signal.entry_price + (signal.take_profit_pips * pip_value)
                sl_price = signal.entry_price - (signal.stop_loss_pips * pip_value)
            else:
                tp_price = signal.entry_price - (signal.take_profit_pips * pip_value)
                sl_price = signal.entry_price + (signal.stop_loss_pips * pip_value)
            
            # Place market order with TP/SL
            logger.info(f"[{symbol}] Executing {signal.action} | Size: {abs(units)} | Entry: {signal.entry_price:.5f} | TP: {tp_price:.5f} | SL: {sl_price:.5f}")
            
            order_result = self.client.place_market_order(
                instrument=symbol,
                units=units,
                take_profit=tp_price,
                stop_loss=sl_price
            )
            
            if not order_result:
                logger.error(f"[{symbol}] Failed to place order")
                return False
            
            # Extract OANDA trade ID from order result
            oanda_trade_id = None
            if order_result and "orderFillTransaction" in order_result:
                fill = order_result["orderFillTransaction"]
                trade_ids = fill.get("tradeOpened", {}).get("tradeID")
                if trade_ids:
                    oanda_trade_id = trade_ids

            # Track position
            self.positions[symbol] = {
                "entry_time": datetime.now(),
                "entry_price": signal.entry_price,
                "side": signal.action,
                "units": abs(units),
                "tp_price": tp_price,
                "sl_price": sl_price,
                "tp_pips": signal.take_profit_pips,
                "sl_pips": signal.stop_loss_pips,
                "confluence_score": signal.confluence_score,
                "htf_trend": signal.trend.value,
                "oanda_trade_id": oanda_trade_id
            }

            # Initialize tracking for smart exits
            self.reversal_cycles[symbol] = 0
            self.peak_pips[symbol] = 0

            # Record trade entry in tracker
            self.tracker.record_trade_entry(
                symbol=symbol,
                side=signal.action,
                entry_price=signal.entry_price,
                units=abs(units),
                tp_price=tp_price,
                sl_price=sl_price,
                tp_pips=signal.take_profit_pips,
                sl_pips=signal.stop_loss_pips,
                confluence_score=signal.confluence_score,
                signal_reason=signal.reason,
                indicators=signal.indicators,
                oanda_trade_id=oanda_trade_id
            )

            logger.info(f"[{symbol}] ✓ Trade executed | {signal.reason}")
            logger.info(f"[{symbol}] OANDA Trade ID: {oanda_trade_id}")
            return True
        
        except Exception as e:
            logger.error(f"[{symbol}] Failed to execute trade: {e}")
            return False
    
    def monitor_position(self, symbol: str):
        """
        Monitor open position and handle smart exits.
        
        Args:
            symbol: Forex pair to monitor
        """
        if symbol not in self.positions:
            return
        
        try:
            position_info = self.positions[symbol]
            
            # Get current price
            pricing = self.client.get_current_price(symbol)
            if not pricing:
                return
            
            current_price = float(pricing.get('mid', 0))
            if current_price == 0:
                return
            
            # Calculate current P&L in pips
            pip_location = self.get_pip_location(symbol)
            pip_value = 10 ** pip_location
            
            if position_info["side"] == "BUY":
                pnl_pips = (current_price - position_info["entry_price"]) / pip_value
            else:
                pnl_pips = (position_info["entry_price"] - current_price) / pip_value
            
            # Update peak pips for trailing lock
            if pnl_pips > self.peak_pips.get(symbol, 0):
                self.peak_pips[symbol] = pnl_pips
            
            # Get current HTF trend
            htf_df = self.get_candles(symbol, "H1", 100)
            if htf_df is not None:
                strategy = self.strategies[symbol]
                current_htf_trend, _ = strategy.get_htf_trend(htf_df)
                
                # Check if HTF trend has reversed
                original_trend = TrendDirection[position_info["htf_trend"]]
                if current_htf_trend != original_trend and current_htf_trend != TrendDirection.NEUTRAL:
                    self.reversal_cycles[symbol] = self.reversal_cycles.get(symbol, 0) + 1
                else:
                    self.reversal_cycles[symbol] = 0  # Reset if trend aligns again
                
                # SMART PROFIT LOCK - Close if in profit and trend reverses
                if self.use_profit_lock and pnl_pips >= self.profit_lock_min_pips:
                    if current_htf_trend != original_trend:
                        logger.info(f"[{symbol}] 🔒 PROFIT LOCK triggered | Pips: {pnl_pips:+.1f} | HTF reversed")
                        self.close_position(symbol, "profit_lock")
                        return
                
                # SMART FAKEOUT PROTECTION
                if self.use_fakeout_protection:
                    reversal_count = self.reversal_cycles.get(symbol, 0)
                    
                    if reversal_count >= self.reversal_cycle_threshold:
                        # Damage control - close immediately if deep loss
                        if pnl_pips < self.damage_control_pips:
                            logger.info(f"[{symbol}] 🚨 DAMAGE CONTROL | Pips: {pnl_pips:+.1f} | Reversals: {reversal_count}")
                            self.close_position(symbol, "damage_control")
                            return
                        
                        # Breakeven protection - move SL to breakeven if good profit
                        if pnl_pips >= self.breakeven_pips_threshold:
                            logger.info(f"[{symbol}] 🛡️ Moving SL to breakeven | Pips: {pnl_pips:+.1f}")
                            self.modify_stop_loss(symbol, position_info["entry_price"])
                            return
                        
                        # Small profit exit - take small profit if weak confirmation
                        if 0 < pnl_pips < self.small_profit_exit_pips:
                            logger.info(f"[{symbol}] 💰 Small profit exit | Pips: {pnl_pips:+.1f} | Reversals: {reversal_count}")
                            self.close_position(symbol, "small_profit")
                            return
            
            # TRAILING PROFIT LOCK
            if self.use_trailing_profit_lock:
                peak = self.peak_pips.get(symbol, 0)
                
                if peak >= self.trailing_lock_activation:
                    # Calculate floor (never give back more than trailing_lock_distance from peak)
                    floor = max(peak - self.trailing_lock_distance, self.trailing_lock_min_floor)
                    
                    if pnl_pips < floor:
                        logger.info(f"[{symbol}] 📉 TRAILING LOCK triggered | Pips: {pnl_pips:+.1f} | Peak: {peak:+.1f} | Floor: {floor:+.1f}")
                        self.close_position(symbol, "trailing_lock")
                        return
        
        except Exception as e:
            logger.error(f"[{symbol}] Error monitoring position: {e}")
    
    def close_position(self, symbol: str, reason: str = "manual"):
        """
        Close a position.

        Args:
            symbol: Forex pair
            reason: Reason for closing
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"[{symbol}] No position to close")
                return

            position_info = self.positions[symbol]
            side = "LONG" if position_info["side"] == "BUY" else "SHORT"

            # Close position via OANDA API
            result = self.client.close_position(symbol, side)

            if result and "error" not in result:
                # Calculate final PnL to track daily losses per symbol
                cfg = self.symbol_configs[symbol]
                pip_value = 10 ** cfg["pip_location"]
                pricing = self.client.get_current_price(symbol)
                if not pricing:
                    logger.error(f"[{symbol}] Could not get current price for PnL calculation")
                    return
                current_price = float(pricing.get('mid', 0))
                if current_price == 0:
                    logger.error(f"[{symbol}] Invalid price for PnL calculation")
                    return

                if position_info["side"] == "BUY":
                    pnl_pips = (current_price - position_info["entry_price"]) / pip_value
                else:
                    pnl_pips = (position_info["entry_price"] - current_price) / pip_value

                # Calculate PnL in USD
                pnl_usd = pnl_pips * pip_value * position_info["units"]

                # Update system realized PnL (only OUR trades)
                self.system_realized_pnl += pnl_usd
                self.pnl_today += pnl_usd

                # Update daily loss tracking per symbol
                current_date = datetime.now().strftime('%Y-%m-%d')

                if pnl_pips < 0:
                    # Track daily loss for this symbol
                    if symbol not in self.per_symbol_daily_losses:
                        self.per_symbol_daily_losses[symbol] = {'date': current_date, 'losses': 0}

                    # If new day, reset
                    if self.per_symbol_daily_losses[symbol]['date'] != current_date:
                        self.per_symbol_daily_losses[symbol] = {'date': current_date, 'losses': 0}

                    self.per_symbol_daily_losses[symbol]['losses'] += 1
                    daily_losses = self.per_symbol_daily_losses[symbol]['losses']

                    self.losses_today += 1
                    logger.info(f"[{symbol}] ✓ Position closed | Reason: {reason} | PnL: {pnl_pips:+.1f}p (${pnl_usd:+,.2f}) | Daily losses: {daily_losses}/{self.max_daily_losses_per_symbol}")

                    if daily_losses >= self.max_daily_losses_per_symbol:
                        logger.warning(f"⚠️  [{symbol}] DAILY LIMIT REACHED: {daily_losses} loss today - BLOCKED until tomorrow")
                else:
                    self.wins_today += 1
                    logger.info(f"[{symbol}] ✓ Position closed | Reason: {reason} | PnL: {pnl_pips:+.1f}p (${pnl_usd:+,.2f})")

                # Record exit in tracker
                self.tracker.record_trade_exit(
                    symbol=symbol,
                    exit_price=current_price,
                    exit_reason=reason,
                    realized_pips=pnl_pips,
                    realized_pl=pnl_usd
                )

                # Update stats
                self.trades_today += 1
                self._save_stats()

                # Remove from tracking
                del self.positions[symbol]
                if symbol in self.reversal_cycles:
                    del self.reversal_cycles[symbol]
                if symbol in self.peak_pips:
                    del self.peak_pips[symbol]

        except Exception as e:
            logger.error(f"[{symbol}] Failed to close position: {e}")
    
    def modify_stop_loss(self, symbol: str, new_sl_price: float):
        """
        Modify stop loss for a position.
        
        Args:
            symbol: Forex pair
            new_sl_price: New stop loss price
        """
        try:
            # Get open trades for symbol
            trades = self.client.get_trades(symbol)
            if not trades:
                return
            
            # Modify SL for first trade (assuming single position per symbol)
            trade_id = trades[0]['id']
            result = self.client.modify_trade(trade_id, stop_loss_price=new_sl_price)
            
            if result:
                logger.info(f"[{symbol}] ✓ SL modified to {new_sl_price:.5f}")
        
        except Exception as e:
            logger.error(f"[{symbol}] Failed to modify SL: {e}")
    
    def run(self):
        """Main trading loop."""
        logger.info("\n🚀 Starting trading engine...")
        
        # Test connection
        if not self.client.test_connection():
            logger.error("Failed to connect to OANDA API")
            return
        
        logger.info("✓ Connected to OANDA API")
        
        self.running = True
        
        try:
            while self.running:
                logger.info("\n" + "=" * 60)
                logger.info(f"Checking signals... | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Session filtering
                current_time = datetime.now()
                if SESSION_TRADING["enabled"]:
                    hour = current_time.hour
                    current_session = None
                    
                    for session_name, (start_hour, end_hour) in SESSION_TRADING["session_hours"].items():
                        if start_hour <= hour < end_hour:
                            current_session = session_name
                            break
                    
                    if current_session not in SESSION_TRADING["allowed_sessions"]:
                        logger.info(f"⏸️  Outside trading hours ({current_session} session) - Waiting...")
                        logger.info("=" * 60)
                        time.sleep(self.check_interval)
                        continue
                    else:
                        logger.info(f"✅ {current_session} session - Trading active")
                
                logger.info("=" * 60)
                
                # Sync with OANDA to detect trades closed by SL/TP
                try:
                    closed_by_oanda = self.tracker.sync_with_oanda(self.client, self.symbols)
                    for closed_trade in closed_by_oanda:
                        symbol = closed_trade.get("symbol")
                        pnl_pips = closed_trade.get("realized_pips", 0)
                        pnl_usd = closed_trade.get("realized_pl", 0)
                        exit_reason = closed_trade.get("exit_reason", "UNKNOWN")

                        logger.info(f"\n🔔 [{symbol}] TRADE CLOSED BY OANDA: {exit_reason}")
                        logger.info(f"   Result: {closed_trade.get('outcome')} | {pnl_pips:+.1f} pips (${pnl_usd:+.2f})")

                        # Update daily stats
                        if pnl_pips > 0:
                            self.wins_today += 1
                        elif pnl_pips < 0:
                            self.losses_today += 1
                            # Track daily loss for this symbol
                            current_date_str = datetime.now().strftime('%Y-%m-%d')
                            if symbol not in self.per_symbol_daily_losses:
                                self.per_symbol_daily_losses[symbol] = {'date': current_date_str, 'losses': 0}
                            if self.per_symbol_daily_losses[symbol]['date'] != current_date_str:
                                self.per_symbol_daily_losses[symbol] = {'date': current_date_str, 'losses': 0}
                            self.per_symbol_daily_losses[symbol]['losses'] += 1

                        self.system_realized_pnl += pnl_usd
                        self.pnl_today += pnl_usd
                        self.trades_today += 1

                        # Remove from internal tracking if present
                        if symbol in self.positions:
                            del self.positions[symbol]
                        if symbol in self.reversal_cycles:
                            del self.reversal_cycles[symbol]
                        if symbol in self.peak_pips:
                            del self.peak_pips[symbol]

                    if closed_by_oanda:
                        self._save_stats()
                except Exception as e:
                    logger.error(f"Error syncing with OANDA: {e}")

                # Check each symbol for signals
                current_date = datetime.now().strftime('%Y-%m-%d')

                for symbol in self.symbols:
                    try:
                        # Skip if already in position
                        if symbol in self.positions:
                            self.monitor_position(symbol)
                            continue
                        
                        # Check daily loss limit for this symbol
                        if symbol in self.per_symbol_daily_losses:
                            symbol_data = self.per_symbol_daily_losses[symbol]
                            # Check if it's the same day
                            if symbol_data['date'] == current_date:
                                if symbol_data['losses'] >= self.max_daily_losses_per_symbol:
                                    logger.info(f"[{symbol}] 🚫 BLOCKED - {symbol_data['losses']}/{self.max_daily_losses_per_symbol} losses today - Try tomorrow")
                                    continue
                            else:
                                # New day - reset
                                self.per_symbol_daily_losses[symbol] = {'date': current_date, 'losses': 0}
                        
                        # Check cooldown
                        if symbol in self.last_trade_time:
                            time_since = (datetime.now() - self.last_trade_time[symbol]).total_seconds() / 60
                            if time_since < self.cooldown_minutes:
                                logger.debug(f"[{symbol}] Cooldown: {self.cooldown_minutes - time_since:.0f}m remaining")
                                continue
                        
                        # Check for signal
                        signal = self.check_for_signal(symbol)
                        
                        # Log detailed analysis to file
                        if signal:
                            analysis_logger.info(f"\n[{symbol}] DETAILED ANALYSIS:")
                            analysis_logger.info(f"  Action: {signal.action or 'NONE'}")
                            analysis_logger.info(f"  Confluence Score: {signal.confluence_score}/8")
                            analysis_logger.info(f"  Confidence: {signal.confidence:.1%}")
                            analysis_logger.info(f"  Reason: {signal.reason}")
                            analysis_logger.info(f"  Entry Price: {signal.entry_price:.5f}")
                            if signal.indicators:
                                analysis_logger.info(f"  Indicators: {signal.indicators}")
                        
                        if signal and signal.action:
                            logger.info(f"\n[{symbol}] 📊 Signal: {signal.action} | Score: {signal.confluence_score}/8 | {signal.reason}")
                            analysis_logger.info(f"[{symbol}] EXECUTING TRADE: {signal.action} at {signal.entry_price:.5f}")
                            
                            # Execute trade
                            if self.execute_trade(symbol, signal):
                                self.last_trade_time[symbol] = datetime.now()
                        elif signal:
                            # Show all checks even when no signal
                            logger.info(f"[{symbol}]  Checked | {signal.reason}")
                            analysis_logger.info(f"[{symbol}] NO SIGNAL: {signal.reason} (Score: {signal.confluence_score}/8)")
                    
                    except Exception as e:
                        logger.error(f"[{symbol}] Error in trading loop: {e}")
                        continue
                
                # Display stats with account information
                logger.info("\n" + "-" * 60)
                
                # Get account summary
                try:
                    account_info = self.client.get_account_summary()
                    if account_info:
                        balance = float(account_info.get('balance', 0))
                        unrealized_pl = float(account_info.get('unrealizedPL', 0))
                        
                        logger.info(f"💰 Account Balance: ${balance:,.2f}")
                        logger.info(f"📊 System Realized PnL: ${self.system_realized_pnl:+,.2f} (our trades only)")
                        logger.info(f"📈 Unrealized PnL: ${unrealized_pl:+,.2f}")
                        logger.info(f"🎯 Total Equity: ${balance + unrealized_pl:,.2f}")
                        
                        # Detailed logging to file
                        analysis_logger.info(f"ACCOUNT SUMMARY:")
                        analysis_logger.info(f"  Balance: ${balance:,.2f}")
                        analysis_logger.info(f"  System Realized PnL: ${self.system_realized_pnl:+,.2f}")
                        analysis_logger.info(f"  Unrealized PnL: ${unrealized_pl:+,.2f}")
                        analysis_logger.info(f"  Total Equity: ${balance + unrealized_pl:,.2f}")
                        analysis_logger.info(f"  Daily Stats: {self.wins_today}W/{self.losses_today}L | PnL: ${self.pnl_today:+.2f}")
                        if hasattr(self, 'per_symbol_daily_losses'):
                            analysis_logger.info(f"  Daily Losses by Symbol: {dict(self.per_symbol_daily_losses)}")
                except Exception as e:
                    logger.error(f"Failed to get account info: {e}")
                    analysis_logger.error(f"Account info error: {e}")
                
                logger.info(f"📋 Today's Stats: {self.wins_today}W/{self.losses_today}L | PnL: ${self.pnl_today:+.2f}")

                # Fetch REAL positions from OANDA API (not internal tracking)
                real_positions = self.client.get_open_positions()
                logger.info(f"🔄 Active positions: {len(real_positions)}")

                # Show detailed position information if any
                if real_positions:
                    logger.info("\n📍 OPEN POSITIONS (LIVE FROM OANDA):")
                    for pos in real_positions:
                        symbol = pos["instrument"]
                        try:
                            # OANDA position data only has basic info
                            # Use internal tracking for entry price and other metadata if available
                            internal_pos = self.positions.get(symbol)

                            if internal_pos:
                                # We have internal tracking - use full display
                                pricing = self.client.get_current_price(symbol)
                                if not pricing:
                                    continue
                                current_price = float(pricing.get('mid', 0))
                                if current_price == 0:
                                    continue

                                cfg = self.symbol_configs[symbol]
                                pip_value = 10 ** cfg["pip_location"]

                                if internal_pos["side"] == "BUY":
                                    pnl_pips = (current_price - internal_pos["entry_price"]) / pip_value
                                else:
                                    pnl_pips = (internal_pos["entry_price"] - current_price) / pip_value

                                pnl_usd = pnl_pips * pip_value * pos["units"]
                                time_in_trade = datetime.now() - internal_pos["entry_time"]
                                hours = time_in_trade.seconds // 3600
                                minutes = (time_in_trade.seconds % 3600) // 60
                                status = "🟢" if pnl_pips > 0 else "🔴" if pnl_pips < 0 else "⚪"

                                logger.info(f"\n  {status} {symbol} - {pos['side']}")
                                logger.info(f"     Entry: {internal_pos['entry_price']:.5f} | Current: {current_price:.5f}")
                                logger.info(f"     TP: {internal_pos['tp_price']:.5f} ({internal_pos['tp_pips']:+.1f}p) | SL: {internal_pos['sl_price']:.5f} ({-internal_pos['sl_pips']:+.1f}p)")
                                logger.info(f"     Size: {pos['units']:,.0f} units | Risk: {self.risk_per_trade*100}%")
                                logger.info(f"     PnL: {pnl_pips:+.1f} pips (${pnl_usd:+,.2f})")
                                logger.info(f"     Duration: {hours}h {minutes}m | Score: {internal_pos['confluence_score']}/8")
                            else:
                                # No internal tracking - show OANDA data with average price
                                pricing = self.client.get_current_price(symbol)
                                if pricing:
                                    current_price = float(pricing.get('mid', 0))
                                    avg_price = pos.get('average_price', 0)
                                    unrealized_pl = pos.get('unrealized_pl', 0)

                                    status = "🟢" if unrealized_pl > 0 else "🔴" if unrealized_pl < 0 else "⚪"
                                    logger.info(f"\n  {status} {symbol} - {pos['side']}")
                                    logger.info(f"     Avg Entry: {avg_price:.5f} | Current: {current_price:.5f}")
                                    logger.info(f"     Size: {pos['units']:,.0f} units")
                                    logger.info(f"     Unrealized P&L: ${unrealized_pl:+.2f}")
                                    logger.info(f"     (Not tracked - opened before restart)")
                                else:
                                    logger.info(f"\n  ⚪ {symbol} - {pos['side']}")
                                    logger.info(f"     Size: {pos['units']:,.0f} units")

                        except Exception as e:
                            logger.error(f"     Error monitoring {symbol}: {str(e)}")
                else:
                    # Show position sizing info when no positions
                    try:
                        account_info = self.client.get_account_summary()
                        if account_info:
                            balance = float(account_info.get('balance', 0))
                            risk_amount = balance * self.risk_per_trade
                            logger.info(f"\n💼 Position Sizing (when signal appears):")
                            logger.info(f"   Risk per trade: {self.risk_per_trade*100}% = ${risk_amount:,.2f}")
                            logger.info(f"   Example: EUR_USD with 12p SL = ~{int(risk_amount / (0.0001 * 12) / 100) * 100:,} units")
                    except Exception as e:
                        logger.debug(f"Could not calculate position size: {e}")
                
                # Show performance summary from tracker
                try:
                    perf = self.tracker.get_performance_summary()
                    if perf.get("total_trades", 0) > 0:
                        logger.info("\n📊 LIFETIME PERFORMANCE (from tracker):")
                        logger.info(f"   Total trades: {perf['total_trades']} | Win rate: {perf['win_rate']:.1f}%")
                        logger.info(f"   Total PnL: {perf['total_pips']:+.1f} pips (${perf['total_pl']:+.2f})")
                        logger.info(f"   Avg Win: {perf['avg_win_pips']:+.1f}p | Avg Loss: {perf['avg_loss_pips']:+.1f}p")
                except Exception as e:
                    logger.debug(f"Could not get performance summary: {e}")

                logger.info("-" * 60)

                # Wait before next check
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Stopping trading engine...")
            self.running = False
        
        except Exception as e:
            logger.error(f"\n\n❌ Fatal error: {e}")
            self.running = False
        
        logger.info("✓ Trading engine stopped")


if __name__ == "__main__":
    # Initialize and run engine
    engine = HTFConfluenceForexEngine(testnet=True)
    engine.run()
