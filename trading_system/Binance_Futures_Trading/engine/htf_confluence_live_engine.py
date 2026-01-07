"""
HTF Confluence Live Trading Engine
==================================
Live trading engine for the HTF Trend + Confluence Strategy.

Strategy:
- HTF (15m): 21/50 EMA crossover trend filter (fast for scalping)
- LTF (5m): MACD + RSI + EMA (9/21) confluence entry (precise entries)
- Single direction trading (follow the trend)
- MODERATE Config: 20x leverage, 30% ROI TP, 10% ROI SL (3:1 R:R)

SMART ENTRY FILTERS (added for better entries):
- 1m Pullback: Wait for dip to 21 EMA on 1-minute timeframe
- Candle Confirmation: Require bullish/bearish candle pattern
- Volatility Filter: Skip if ATR > 2% (too choppy)
- Trend Filter: Skip if ADX < 20 (no clear trend)

ML DATA LOGGING:
- Logs every signal (traded and skipped) with full market context
- Captures OHLCV data, indicators, and market conditions at signal time
- Records trade results for supervised learning
- Saved to CSV files for easy ML training
"""

import os
import sys
import time
import json
import logging
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.binance_client import BinanceClient
from strategies.htf_confluence_strategy import (
    HTFConfluenceStrategy,
    MODERATE_CONFIG,
    ASSET_SPECIFIC_CONFIG,
    get_config_for_symbol,
    TrendDirection,
    SignalStrength
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MLTradeLogger:
    """
    ML Training Data Logger

    Logs comprehensive data for every signal and trade for future ML training:
    - Signal data: action, strength, confluence score, reason
    - Market data: OHLCV, volume, indicators at multiple timeframes
    - Trade results: entry, exit, PnL, duration, outcome

    Data saved to CSV for easy ML model training.
    """

    def __init__(self, log_dir: str = None):
        """Initialize ML logger."""
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), "ml_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # File paths
        self.signals_file = os.path.join(self.log_dir, "signals_log.csv")
        self.trades_file = os.path.join(self.log_dir, "trades_log.csv")
        self.market_data_file = os.path.join(self.log_dir, "market_snapshots.csv")

        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()

        logger.info(f"ML Logger initialized - logs at: {self.log_dir}")

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Signals log headers
        if not os.path.exists(self.signals_file):
            with open(self.signals_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Timestamp & ID
                    'timestamp', 'signal_id', 'symbol',
                    # Signal info
                    'action', 'strength', 'confluence_score', 'reason',
                    # Smart filter results
                    'filters_passed', 'filter_reason',
                    # Was trade executed?
                    'trade_executed', 'skip_reason',
                    # 5m (LTF) Price Data
                    'ltf_open', 'ltf_high', 'ltf_low', 'ltf_close', 'ltf_volume',
                    # 5m Indicators
                    'ltf_ema9', 'ltf_ema21', 'ltf_ema50', 'ltf_rsi', 'ltf_macd', 'ltf_macd_signal', 'ltf_macd_hist',
                    # 15m (HTF) Price Data
                    'htf_open', 'htf_high', 'htf_low', 'htf_close', 'htf_volume',
                    # 15m Indicators
                    'htf_ema21', 'htf_ema50', 'htf_rsi',
                    # 1m Data (for smart filters)
                    'm1_close', 'm1_ema21', 'm1_distance_from_ema',
                    # Volatility/Trend
                    'atr_pct', 'adx',
                    # Market structure
                    'higher_high', 'higher_low', 'trend_direction',
                    # Recent candle patterns
                    'candle_body_ratio', 'candle_is_bullish',
                    # Volume analysis
                    'volume_sma20', 'volume_ratio',
                    # Price levels
                    'price_vs_daily_high', 'price_vs_daily_low',
                ])

        # Trades log headers
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # Trade ID (links to signal)
                    'trade_id', 'signal_id', 'symbol',
                    # Entry
                    'entry_time', 'entry_price', 'position_side', 'quantity', 'margin',
                    # TP/SL levels
                    'tp_price', 'sl_price', 'tp_roi_target', 'sl_roi_target',
                    # Exit
                    'exit_time', 'exit_price', 'exit_type',
                    # Results
                    'pnl', 'roi_pct', 'price_move_pct', 'duration_minutes',
                    # Outcome for ML
                    'outcome',  # WIN/LOSS
                    # Market conditions at entry
                    'entry_atr_pct', 'entry_adx', 'entry_rsi', 'entry_volume_ratio',
                    # Market conditions at exit
                    'exit_atr_pct', 'exit_adx', 'exit_rsi',
                ])

        # Market snapshots headers
        if not os.path.exists(self.market_data_file):
            with open(self.market_data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'signal_id', 'symbol', 'timeframe',
                    # Last 10 candles OHLCV
                    'c0_open', 'c0_high', 'c0_low', 'c0_close', 'c0_volume',
                    'c1_open', 'c1_high', 'c1_low', 'c1_close', 'c1_volume',
                    'c2_open', 'c2_high', 'c2_low', 'c2_close', 'c2_volume',
                    'c3_open', 'c3_high', 'c3_low', 'c3_close', 'c3_volume',
                    'c4_open', 'c4_high', 'c4_low', 'c4_close', 'c4_volume',
                    'c5_open', 'c5_high', 'c5_low', 'c5_close', 'c5_volume',
                    'c6_open', 'c6_high', 'c6_low', 'c6_close', 'c6_volume',
                    'c7_open', 'c7_high', 'c7_low', 'c7_close', 'c7_volume',
                    'c8_open', 'c8_high', 'c8_low', 'c8_close', 'c8_volume',
                    'c9_open', 'c9_high', 'c9_low', 'c9_close', 'c9_volume',
                ])

    def _generate_signal_id(self, symbol: str) -> str:
        """Generate unique signal ID."""
        return f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate indicators from OHLCV dataframe."""
        if df is None or len(df) < 50:
            return {}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # EMAs
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Volume SMA
        vol_sma20 = volume.rolling(window=20).mean()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_smooth = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 0.0001))
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 0.0001))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.ewm(span=14, adjust=False).mean()

        # Market structure
        recent_highs = high.tail(20)
        recent_lows = low.tail(20)
        higher_high = high.iloc[-1] > high.iloc[-2] > high.iloc[-3]
        higher_low = low.iloc[-1] > low.iloc[-2] > low.iloc[-3]

        # Candle analysis
        current = df.iloc[-1]
        body = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        body_ratio = body / (total_range + 0.0001)
        is_bullish = current['close'] > current['open']

        return {
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'volume': current['volume'],
            'ema9': ema9.iloc[-1],
            'ema21': ema21.iloc[-1],
            'ema50': ema50.iloc[-1],
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'macd_hist': macd_hist.iloc[-1],
            'atr': atr.iloc[-1],
            'atr_pct': (atr.iloc[-1] / close.iloc[-1]) * 100,
            'adx': adx.iloc[-1],
            'vol_sma20': vol_sma20.iloc[-1],
            'volume_ratio': current['volume'] / (vol_sma20.iloc[-1] + 0.0001),
            'higher_high': higher_high,
            'higher_low': higher_low,
            'body_ratio': body_ratio,
            'is_bullish': is_bullish,
            'daily_high': high.tail(288).max(),  # ~24h of 5m candles
            'daily_low': low.tail(288).min(),
        }

    def log_signal(self, symbol: str, signal, ltf_df: pd.DataFrame, htf_df: pd.DataFrame,
                   m1_df: pd.DataFrame, filters_passed: bool, filter_reason: str,
                   trade_executed: bool, skip_reason: str = "") -> str:
        """
        Log a signal with all market context for ML training.

        Returns:
            signal_id for linking to trade result
        """
        try:
            signal_id = self._generate_signal_id(symbol)
            timestamp = datetime.now().isoformat()

            # Calculate indicators for each timeframe
            ltf_ind = self._calculate_indicators(ltf_df)
            htf_ind = self._calculate_indicators(htf_df)
            m1_ind = self._calculate_indicators(m1_df) if m1_df is not None else {}

            # Determine trend direction
            if ltf_ind.get('ema21', 0) > ltf_ind.get('ema50', 0):
                trend_direction = 'BULLISH'
            elif ltf_ind.get('ema21', 0) < ltf_ind.get('ema50', 0):
                trend_direction = 'BEARISH'
            else:
                trend_direction = 'NEUTRAL'

            # Price vs daily range
            current_price = ltf_ind.get('close', 0)
            daily_high = ltf_ind.get('daily_high', current_price)
            daily_low = ltf_ind.get('daily_low', current_price)
            daily_range = daily_high - daily_low
            price_vs_high = (daily_high - current_price) / (daily_range + 0.0001) if daily_range > 0 else 0.5
            price_vs_low = (current_price - daily_low) / (daily_range + 0.0001) if daily_range > 0 else 0.5

            # M1 distance from EMA
            m1_distance = 0
            if m1_ind:
                m1_close = m1_ind.get('close', 0)
                m1_ema21 = m1_ind.get('ema21', m1_close)
                m1_distance = (m1_close - m1_ema21) / (m1_ema21 + 0.0001) * 100

            # Write signal row
            with open(self.signals_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, signal_id, symbol,
                    signal.action if signal else '',
                    signal.strength.value if signal and hasattr(signal, 'strength') else '',
                    signal.confluence_score if signal else 0,
                    signal.reason if signal else '',
                    filters_passed, filter_reason,
                    trade_executed, skip_reason,
                    # LTF data
                    ltf_ind.get('open', ''), ltf_ind.get('high', ''), ltf_ind.get('low', ''),
                    ltf_ind.get('close', ''), ltf_ind.get('volume', ''),
                    ltf_ind.get('ema9', ''), ltf_ind.get('ema21', ''), ltf_ind.get('ema50', ''),
                    ltf_ind.get('rsi', ''), ltf_ind.get('macd', ''),
                    ltf_ind.get('macd_signal', ''), ltf_ind.get('macd_hist', ''),
                    # HTF data
                    htf_ind.get('open', ''), htf_ind.get('high', ''), htf_ind.get('low', ''),
                    htf_ind.get('close', ''), htf_ind.get('volume', ''),
                    htf_ind.get('ema21', ''), htf_ind.get('ema50', ''), htf_ind.get('rsi', ''),
                    # M1 data
                    m1_ind.get('close', ''), m1_ind.get('ema21', ''), m1_distance,
                    # Volatility/Trend
                    ltf_ind.get('atr_pct', ''), ltf_ind.get('adx', ''),
                    # Market structure
                    ltf_ind.get('higher_high', ''), ltf_ind.get('higher_low', ''), trend_direction,
                    # Candle patterns
                    ltf_ind.get('body_ratio', ''), ltf_ind.get('is_bullish', ''),
                    # Volume
                    ltf_ind.get('vol_sma20', ''), ltf_ind.get('volume_ratio', ''),
                    # Price levels
                    price_vs_high, price_vs_low,
                ])

            # Also save market snapshot (last 10 candles)
            self._log_market_snapshot(signal_id, symbol, '5m', ltf_df)
            self._log_market_snapshot(signal_id, symbol, '15m', htf_df)
            if m1_df is not None:
                self._log_market_snapshot(signal_id, symbol, '1m', m1_df)

            logger.info(f"[ML] Signal logged: {signal_id}")
            return signal_id

        except Exception as e:
            logger.error(f"[ML] Failed to log signal: {e}")
            return ""

    def _log_market_snapshot(self, signal_id: str, symbol: str, timeframe: str, df: pd.DataFrame):
        """Log last 10 candles of market data."""
        try:
            if df is None or len(df) < 10:
                return

            timestamp = datetime.now().isoformat()
            last_10 = df.tail(10)

            row = [timestamp, signal_id, symbol, timeframe]

            # Add last 10 candles (most recent first)
            for i in range(10):
                idx = -(i + 1)
                if abs(idx) <= len(last_10):
                    candle = last_10.iloc[idx]
                    row.extend([
                        candle['open'], candle['high'], candle['low'],
                        candle['close'], candle['volume']
                    ])
                else:
                    row.extend(['', '', '', '', ''])

            with open(self.market_data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            logger.error(f"[ML] Failed to log market snapshot: {e}")

    def log_trade_entry(self, signal_id: str, symbol: str, position_side: str,
                        entry_price: float, quantity: float, margin: float,
                        tp_price: float, sl_price: float, tp_roi: float, sl_roi: float,
                        atr_pct: float, adx: float, rsi: float, volume_ratio: float) -> str:
        """Log trade entry for later result matching."""
        try:
            trade_id = f"T_{signal_id}"
            entry_time = datetime.now().isoformat()

            # Write partial trade row (will be updated on exit)
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_id, signal_id, symbol,
                    entry_time, entry_price, position_side, quantity, margin,
                    tp_price, sl_price, tp_roi, sl_roi,
                    '', '', '',  # exit_time, exit_price, exit_type (filled on exit)
                    '', '', '', '',  # pnl, roi, price_move, duration (filled on exit)
                    '',  # outcome (filled on exit)
                    atr_pct, adx, rsi, volume_ratio,
                    '', '', '',  # exit conditions (filled on exit)
                ])

            logger.info(f"[ML] Trade entry logged: {trade_id}")
            return trade_id

        except Exception as e:
            logger.error(f"[ML] Failed to log trade entry: {e}")
            return ""

    def log_trade_exit(self, trade_id: str, exit_price: float, exit_type: str,
                       pnl: float, roi_pct: float, price_move_pct: float,
                       duration_minutes: float, atr_pct: float = 0, adx: float = 0, rsi: float = 0):
        """Update trade log with exit details."""
        try:
            # Read all trades
            trades = []
            with open(self.trades_file, 'r', newline='') as f:
                reader = csv.reader(f)
                trades = list(reader)

            # Find and update the trade
            for i, row in enumerate(trades):
                if len(row) > 0 and row[0] == trade_id:
                    # Update exit fields
                    row[12] = datetime.now().isoformat()  # exit_time
                    row[13] = exit_price  # exit_price
                    row[14] = exit_type  # exit_type
                    row[15] = pnl  # pnl
                    row[16] = roi_pct  # roi_pct
                    row[17] = price_move_pct  # price_move_pct
                    row[18] = duration_minutes  # duration_minutes
                    row[19] = 'WIN' if pnl > 0 else 'LOSS'  # outcome
                    row[24] = atr_pct  # exit_atr_pct
                    row[25] = adx  # exit_adx
                    row[26] = rsi  # exit_rsi
                    trades[i] = row
                    break

            # Write back all trades
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(trades)

            outcome = 'WIN' if pnl > 0 else 'LOSS'
            logger.info(f"[ML] Trade exit logged: {trade_id} - {outcome} (${pnl:+.2f})")

        except Exception as e:
            logger.error(f"[ML] Failed to log trade exit: {e}")


class HTFConfluenceLiveEngine:
    """
    Live Trading Engine for HTF Confluence Strategy

    Features:
    - Multi-symbol support (DOT, BNB, etc.)
    - HTF trend detection on 4H timeframe
    - LTF entry signals on 15m timeframe
    - Automatic TP/SL order placement
    - Position tracking and management
    - Cooldown between trades
    """

    def __init__(
        self,
        symbols: List[str] = None,
        config: dict = None,
        testnet: bool = True,
        total_capital: float = None,
        risk_per_trade: float = 0.02
    ):
        """
        Initialize live trading engine.

        Args:
            symbols: List of symbols to trade (default: DOTUSDT, BNBUSDT)
            config: Strategy config (default: MODERATE_CONFIG)
            testnet: Use demo/testnet mode (default: True for safety)
            total_capital: Total capital to use (None = use account balance)
            risk_per_trade: Risk per trade as fraction (default: 2%)
        """
        self.symbols = symbols or ["DOTUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
        self.default_config = config or MODERATE_CONFIG
        self.testnet = testnet
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade

        # Use ASSET-SPECIFIC configs for each symbol
        # Each symbol gets its own optimized TP/SL settings based on backtest results
        self.symbol_configs = {}
        for symbol in self.symbols:
            self.symbol_configs[symbol] = get_config_for_symbol(symbol)

        # Extract default config values (for display purposes)
        self.leverage = self.default_config["leverage"]
        self.tp_roi = self.default_config["tp_roi"]
        self.sl_roi = self.default_config["sl_roi"]

        # Initialize Binance client
        self.client = BinanceClient(testnet=testnet, use_demo=testnet)

        # Initialize strategy for each symbol with ASSET-SPECIFIC config
        self.strategies = {}
        for symbol in self.symbols:
            sym_config = self.symbol_configs[symbol]
            self.strategies[symbol] = HTFConfluenceStrategy(**sym_config)

        # Position tracking
        self.positions = {}  # symbol -> position info
        self.pending_orders = {}  # symbol -> {tp_order_id, sl_order_id}

        # Cooldown tracking (prevent over-trading)
        self.last_trade_time = {}  # symbol -> datetime
        self.cooldown_minutes = 0  # No cooldown - trade whenever signal appears

        # Statistics - load from file if exists (persist across restarts)
        self.stats_file = os.path.join(os.path.dirname(__file__), "session_stats.json")
        self._load_stats()

        # Tracked positions file (for detecting closed positions after restart)
        self.positions_file = os.path.join(os.path.dirname(__file__), "tracked_positions.json")

        # Running state
        self.running = False
        self.check_interval = 60  # Check every 60 seconds (on 15m TF, no need for faster)

        # ML TRAINING DATA LOGGER
        self.ml_logger = MLTradeLogger()
        self.active_trade_ids = {}  # symbol -> trade_id for linking entry to exit

        # SMART ENTRY FILTER SETTINGS
        self.use_1m_pullback = True        # Require pullback to 21 EMA on 1m
        self.use_candle_confirm = True     # Require bullish/bearish candle
        self.use_atr_filter = True         # Skip if ATR > max_atr_pct
        self.use_adx_filter = True         # Skip if ADX < min_adx
        self.max_atr_pct = 2.0             # Max ATR as % of price (skip if higher)
        self.min_adx = 20                  # Min ADX for trending market
        self.pullback_tolerance = 0.003    # 0.3% tolerance from 21 EMA

        # SMART PROFIT LOCK - Close early if trend reverses while in profit
        self.use_profit_lock = True        # Enable smart profit lock
        self.profit_lock_min_roi = 30.0    # Minimum ROI% to consider profit lock (30%)

        # SMART FAKEOUT PROTECTION - Exit early on suspected fakeout entries (STRENGTHENED)
        self.use_fakeout_protection = True
        self.reversal_cycles = {}          # symbol -> count of cycles HTF has been reversed
        self.reversal_cycle_threshold = 5  # STRENGTHENED: Wait 5 cycles (was 3) before acting
        self.breakeven_roi_threshold = 15.0   # STRENGTHENED: Need 15% ROI (was 10%) for breakeven
        self.small_profit_exit_roi = 5.0      # STRENGTHENED: Only exit at 5% (was 10%) - be more patient
        self.damage_control_roi = -20.0       # STRENGTHENED: Wider tolerance -20% (was -15%)

        # TRAILING PROFIT LOCK - Never give back too much profit
        self.use_trailing_profit_lock = True
        self.peak_roi = {}                    # symbol -> highest ROI reached
        self.trailing_lock_activation = 30.0  # Start trailing after 30% ROI
        self.trailing_lock_distance = 15.0    # Close if ROI drops 15% from peak
        self.trailing_lock_min_floor = 15.0   # Never let floor go below 15% ROI

        logger.info("=" * 60)
        logger.info("HTF CONFLUENCE LIVE TRADING ENGINE")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Mode: {'DEMO/TESTNET' if testnet else 'LIVE MAINNET'}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"Risk per trade: {self.risk_per_trade * 100:.0f}%")
        logger.info("-" * 60)
        logger.info("ASSET-SPECIFIC OPTIMIZED SETTINGS:")
        for symbol in self.symbols:
            cfg = self.symbol_configs[symbol]
            tp = cfg["tp_roi"] * 100
            sl = cfg["sl_roi"] * 100
            rr = cfg["tp_roi"] / cfg["sl_roi"]
            tp_price = cfg["tp_roi"] / cfg["leverage"] * 100
            sl_price = cfg["sl_roi"] / cfg["leverage"] * 100
            logger.info(f"  {symbol}: TP {tp:.0f}% / SL {sl:.0f}% ROI ({rr:.1f}:1 R:R) | Price: TP {tp_price:.2f}% / SL {sl_price:.2f}%")
        logger.info("-" * 60)
        logger.info("SMART ENTRY FILTERS:")
        logger.info(f"  1m Pullback to 21 EMA: {'ON' if self.use_1m_pullback else 'OFF'} (tolerance: {self.pullback_tolerance*100:.1f}%)")
        logger.info(f"  Candle Confirmation: {'ON' if self.use_candle_confirm else 'OFF'}")
        logger.info(f"  ATR Volatility Filter: {'ON' if self.use_atr_filter else 'OFF'} (max: {self.max_atr_pct}%)")
        logger.info(f"  ADX Trend Filter: {'ON' if self.use_adx_filter else 'OFF'} (min: {self.min_adx})")
        logger.info("-" * 60)
        logger.info("SMART PROFIT LOCK:")
        logger.info(f"  Profit Lock: {'ON' if self.use_profit_lock else 'OFF'} (min ROI: {self.profit_lock_min_roi}%)")
        logger.info(f"  Action: Close position if ROI >= {self.profit_lock_min_roi}% AND HTF trend reverses")
        logger.info("-" * 60)
        logger.info("SMART FAKEOUT PROTECTION:")
        logger.info(f"  Fakeout Protection: {'ON' if self.use_fakeout_protection else 'OFF'}")
        logger.info(f"  Reversal cycles to act: {self.reversal_cycle_threshold}")
        logger.info(f"  ROI >= {self.breakeven_roi_threshold}%: Move SL to breakeven")
        logger.info(f"  ROI 0-{self.small_profit_exit_roi}% + confirms ≤1: Small profit exit")
        logger.info(f"  ROI -10-0% + confirms ≤1: Cut loss early")
        logger.info(f"  ROI < {self.damage_control_roi}%: Damage control (immediate close)")
        logger.info("-" * 60)
        logger.info("TRAILING PROFIT LOCK:")
        logger.info(f"  Trailing Lock: {'ON' if self.use_trailing_profit_lock else 'OFF'}")
        logger.info(f"  Activates at: {self.trailing_lock_activation}% ROI")
        logger.info(f"  Floor distance: {self.trailing_lock_distance}% below peak")
        logger.info(f"  Minimum floor: {self.trailing_lock_min_floor}% ROI")
        logger.info(f"  Example: Peak 34% → Floor 19% → Close if drops to 19%")
        logger.info("=" * 60)

    def _load_stats(self):
        """Load session stats from file (persist across restarts)."""
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.symbol_stats = {}
        for symbol in self.symbols:
            self.symbol_stats[symbol] = {"wins": 0, "losses": 0, "pnl": 0.0}

        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
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
                    else:
                        logger.info("Stats file is from a previous day, syncing from Binance...")
                        self._sync_stats_from_binance()
            else:
                logger.info("No stats file, syncing from Binance...")
                self._sync_stats_from_binance()
        except Exception as e:
            logger.warning(f"Could not load stats: {e}")

    def _sync_stats_from_binance(self):
        """Sync today's stats from Binance income history."""
        try:
            # Get today's start timestamp
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_ms = int(today_start.timestamp() * 1000)

            # Get income history (last 100 records)
            income = self.client.get_income_history(income_type="REALIZED_PNL", limit=100)

            if not income:
                logger.info("No income history found")
                return

            # Filter for today's records and count wins/losses
            for record in income:
                # Check if record is from today
                record_time = int(record.get("time", 0))
                if record_time < today_start_ms:
                    continue  # Skip records from before today

                symbol = record.get("symbol", "")
                pnl = float(record.get("income", 0))

                if symbol not in self.symbols:
                    continue

                self.trades_today += 1
                self.pnl_today += pnl

                if pnl > 0:
                    self.wins_today += 1
                    self.symbol_stats[symbol]["wins"] += 1
                else:
                    self.losses_today += 1
                    self.symbol_stats[symbol]["losses"] += 1

                self.symbol_stats[symbol]["pnl"] += pnl

            logger.info(f"Synced from Binance: {self.wins_today}W/{self.losses_today}L, PnL: ${self.pnl_today:+.2f}")
            self._save_stats()

        except Exception as e:
            logger.warning(f"Could not sync stats from Binance: {e}")

    def _save_stats(self):
        """Save session stats to file."""
        try:
            data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "trades_today": self.trades_today,
                "wins_today": self.wins_today,
                "losses_today": self.losses_today,
                "pnl_today": self.pnl_today,
                "symbol_stats": self.symbol_stats
            }
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save stats: {e}")

    def _save_tracked_positions(self):
        """Save tracked positions to file for resume after restart."""
        try:
            data = {}
            for symbol, pos in self.positions.items():
                data[symbol] = {
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "quantity": pos["quantity"],
                    "tp_price": pos["tp_price"],
                    "sl_price": pos["sl_price"],
                    "entry_time": pos["entry_time"].isoformat() if isinstance(pos["entry_time"], datetime) else str(pos["entry_time"])
                }
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save positions: {e}")

    def _load_tracked_positions(self) -> Dict:
        """Load tracked positions from file."""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load positions: {e}")
        return {}

    def _check_closed_positions_on_startup(self):
        """Check if any tracked positions were closed while engine was down."""
        saved_positions = self._load_tracked_positions()
        if not saved_positions:
            return

        for symbol, saved_pos in saved_positions.items():
            # Check if position still exists on Binance
            position = self.client.get_position(symbol, saved_pos["side"])

            if not position or position["quantity"] == 0:
                # Position was closed while engine was down
                entry_price = saved_pos["entry_price"]
                tp_price = saved_pos["tp_price"]
                sl_price = saved_pos["sl_price"]
                quantity = saved_pos["quantity"]

                # Get current price to estimate exit type
                try:
                    price_data = self.client.get_current_price(symbol)
                    current_price = price_data["price"]

                    # Determine exit type
                    if saved_pos["side"] == "LONG":
                        if current_price >= tp_price * 0.995:
                            exit_type = "TP"
                            pnl = quantity * (tp_price - entry_price)
                        else:
                            exit_type = "SL"
                            pnl = quantity * (sl_price - entry_price)
                    else:
                        if current_price <= tp_price * 1.005:
                            exit_type = "TP"
                            pnl = quantity * (entry_price - tp_price)
                        else:
                            exit_type = "SL"
                            pnl = quantity * (entry_price - sl_price)

                    roi = (pnl / (quantity * entry_price)) * self.leverage * 100

                    # Update stats
                    if exit_type == "TP":
                        self.wins_today += 1
                        self.symbol_stats[symbol]["wins"] += 1
                        logger.info(f"[{symbol}] Position closed while offline - TP HIT! ROI: +{roi:.1f}%")
                    else:
                        self.losses_today += 1
                        self.symbol_stats[symbol]["losses"] += 1
                        logger.info(f"[{symbol}] Position closed while offline - SL HIT. ROI: {roi:.1f}%")

                    self.trades_today += 1
                    self.pnl_today += pnl
                    self.symbol_stats[symbol]["pnl"] += pnl
                    self._save_stats()

                except Exception as e:
                    logger.error(f"[{symbol}] Failed to detect closed position: {e}")

    def initialize(self) -> bool:
        """
        Initialize the trading engine.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing trading engine...")

        # Check for positions that closed while offline
        self._check_closed_positions_on_startup()

        # Test API connection
        if not self.client.test_connection():
            logger.error("Failed to connect to Binance API")
            return False

        # Get account balance
        try:
            balance = self.client.get_balance()
            available = self.client.get_available_balance()
            logger.info(f"Account Balance: ${balance:,.2f} USDT")
            logger.info(f"Available Balance: ${available:,.2f} USDT")

            # Store initial capital for reference (if specified)
            if self.total_capital is None:
                self.total_capital = available

            # Dynamic allocation: margin is calculated per-trade based on available balance
            # Each symbol gets equal share of available margin when opening
            logger.info(f"Margin Mode: DYNAMIC (Available balance split equally among symbols)")
            logger.info(f"Initial per-symbol margin: ~${available / len(self.symbols):.2f} (will adjust based on availability)")

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return False

        # Set up each symbol
        for symbol in self.symbols:
            try:
                # Set leverage
                result = self.client.set_leverage(symbol, self.leverage)
                if "code" in result and result["code"] != 0:
                    logger.warning(f"[{symbol}] Leverage may already be set: {result}")
                else:
                    logger.info(f"[{symbol}] Leverage set to {self.leverage}x")

                # Set isolated margin
                try:
                    self.client.set_margin_type(symbol, "ISOLATED")
                    logger.info(f"[{symbol}] Margin type set to ISOLATED")
                except:
                    logger.info(f"[{symbol}] Margin type already ISOLATED")

                # Check for existing positions and resume management
                position = self.client.get_position(symbol)
                if position and position.get("quantity", 0) > 0:
                    logger.info(f"[{symbol}] RESUMING existing position: {position['side']} {position['quantity']} @ ${position['entry_price']:,.4f}")

                    # Calculate TP/SL for existing position
                    strategy = self.strategies[symbol]
                    sl_price, tp_price = strategy.calculate_exit_levels(
                        position["entry_price"],
                        position["side"]
                    )

                    # Track position with calculated levels
                    self.positions[symbol] = {
                        "side": position["side"],
                        "entry_price": position["entry_price"],
                        "quantity": position["quantity"],
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "entry_time": datetime.now(),
                        "signal_strength": "RESUMED",
                        "confluence_score": 0
                    }
                    logger.info(f"[{symbol}] TP: ${tp_price:,.4f} | SL: ${sl_price:,.4f}")

                    # Verify TP/SL orders exist for this position
                    self._verify_tp_sl_orders(symbol, position)

            except Exception as e:
                logger.error(f"[{symbol}] Setup failed: {e}")
                return False

        # Save tracked positions for restart detection
        self._save_tracked_positions()

        logger.info("Initialization complete!")
        return True

    def _verify_tp_sl_orders(self, symbol: str, position: dict):
        """
        Verify TP/SL orders exist for a position, create if missing.

        Args:
            symbol: Trading symbol
            position: Position dict with side, quantity, entry_price
        """
        try:
            position_side = position["side"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]

            # Get open orders for this symbol
            open_orders = self.client.get_open_orders(symbol=symbol)

            # Check for TP and SL orders
            has_tp = False
            has_sl = False

            for order in open_orders:
                order_position_side = order.get("positionSide", "")
                if order_position_side != position_side:
                    continue

                order_type = order.get("type", "")
                if order_type == "TAKE_PROFIT_MARKET":
                    has_tp = True
                    logger.info(f"[{symbol}] Found existing TP order: {order.get('orderId')}")
                elif order_type == "STOP_MARKET":
                    has_sl = True
                    logger.info(f"[{symbol}] Found existing SL order: {order.get('orderId')}")

            # Calculate expected TP/SL prices
            strategy = self.strategies[symbol]
            sl_price, tp_price = strategy.calculate_exit_levels(entry_price, position_side)

            close_side = "SELL" if position_side == "LONG" else "BUY"

            # Create missing TP order
            if not has_tp:
                logger.info(f"[{symbol}] Creating missing TP order at ${tp_price:,.4f}")
                tp_result = self.client.place_take_profit(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    take_profit_price=tp_price,
                    position_side=position_side
                )
                if "code" not in tp_result or tp_result.get("code") == 0:
                    logger.info(f"[{symbol}] TP order created: {tp_result.get('orderId')}")
                else:
                    logger.warning(f"[{symbol}] TP order failed: {tp_result}")

            # Create missing SL order
            if not has_sl:
                logger.info(f"[{symbol}] Creating missing SL order at ${sl_price:,.4f}")
                sl_result = self.client.place_stop_loss(
                    symbol=symbol,
                    side=close_side,
                    quantity=quantity,
                    stop_price=sl_price,
                    position_side=position_side
                )
                if "code" not in sl_result or sl_result.get("code") == 0:
                    logger.info(f"[{symbol}] SL order created: {sl_result.get('orderId')}")
                else:
                    logger.warning(f"[{symbol}] SL order failed: {sl_result}")

        except Exception as e:
            logger.error(f"[{symbol}] Failed to verify TP/SL orders: {e}")
            import traceback
            traceback.print_exc()

    def get_market_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch market data for strategy analysis.

        Args:
            symbol: Trading symbol

        Returns:
            (ltf_df, htf_df, m5_df) - 15m, 1H, and 5m DataFrames
        """
        try:
            # Get 15m data (LTF) - entry signals with EMA/RSI/MACD
            ltf_df = self.client.get_klines(symbol, "15m", 150)

            # Get 1H data (HTF) - trend detection with 21/50 EMA
            htf_df = self.client.get_klines(symbol, "1h", 150)

            # Get 5m data for confirmation
            m5_df = self.client.get_klines(symbol, "5m", 50)

            if ltf_df.empty or htf_df.empty:
                logger.warning(f"[{symbol}] Empty data received")
                return None, None, None

            return ltf_df, htf_df, m5_df

        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch market data: {e}")
            return None, None, None

    def get_1m_data(self, symbol: str) -> pd.DataFrame:
        """Fetch 1-minute data for precise pullback detection."""
        try:
            df = self.client.get_klines(symbol, "1m", 50)
            return df
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch 1m data: {e}")
            return None

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range as % of price."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        current_price = close.iloc[-1]
        atr_pct = (atr / current_price) * 100
        return atr_pct

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index for trend strength."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

    def check_1m_pullback(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Check if price has pulled back to 21 EMA on 1-minute timeframe.

        For LONG: Price should be near/touching 21 EMA from above (dip buy)
        For SHORT: Price should be near/touching 21 EMA from below (rally sell)

        Returns:
            (passed, reason)
        """
        df = self.get_1m_data(symbol)
        if df is None or len(df) < 25:
            return True, "No 1m data (skipped)"

        close = df["close"]
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]

        ema21 = self.calculate_ema(close, 21)
        ema_value = ema21.iloc[-1]

        distance_from_ema = (current_price - ema_value) / ema_value

        if direction in ["LONG", "BUY"]:
            # For LONG: Price should be slightly above EMA (just bounced) or touching it
            # Good: Price dipped to EMA and bouncing (distance -0.3% to +0.5%)
            if -self.pullback_tolerance <= distance_from_ema <= 0.005:
                if current_price >= prev_price:  # Bouncing up
                    return True, f"1m pullback OK ({distance_from_ema*100:.2f}% from EMA, bouncing)"
            return False, f"No 1m pullback ({distance_from_ema*100:.2f}% from EMA) - wait for dip"

        elif direction in ["SHORT", "SELL"]:
            # For SHORT: Price should be slightly below EMA (just rejected) or touching it
            if -0.005 <= distance_from_ema <= self.pullback_tolerance:
                if current_price <= prev_price:  # Rejecting down
                    return True, f"1m rally OK ({distance_from_ema*100:.2f}% from EMA, rejecting)"
            return False, f"No 1m rally ({distance_from_ema*100:.2f}% from EMA) - wait for rally"

        return True, "Direction unknown"

    def check_candle_confirmation(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Check if current 1m candle confirms the direction.

        For LONG: Current candle should be bullish (close > open)
        For SHORT: Current candle should be bearish (close < open)

        Returns:
            (passed, reason)
        """
        df = self.get_1m_data(symbol)
        if df is None or len(df) < 3:
            return True, "No 1m data (skipped)"

        current = df.iloc[-1]
        body = abs(current["close"] - current["open"])
        total_range = current["high"] - current["low"]

        if total_range == 0:
            return False, "Doji candle (indecision)"

        body_ratio = body / total_range

        if direction in ["LONG", "BUY"]:
            is_bullish = current["close"] > current["open"]
            if is_bullish and body_ratio > 0.3:  # At least 30% body
                lower_wick = min(current["close"], current["open"]) - current["low"]
                upper_wick = current["high"] - max(current["close"], current["open"])
                if lower_wick >= upper_wick:
                    return True, "Bullish candle with rejection wick"
                return True, "Bullish candle confirmed"
            elif is_bullish:
                return False, "Weak bullish candle (small body)"
            return False, "Bearish candle - waiting for bullish"

        elif direction in ["SHORT", "SELL"]:
            is_bearish = current["close"] < current["open"]
            if is_bearish and body_ratio > 0.3:
                upper_wick = current["high"] - max(current["close"], current["open"])
                lower_wick = min(current["close"], current["open"]) - current["low"]
                if upper_wick >= lower_wick:
                    return True, "Bearish candle with rejection wick"
                return True, "Bearish candle confirmed"
            elif is_bearish:
                return False, "Weak bearish candle (small body)"
            return False, "Bullish candle - waiting for bearish"

        return True, "Direction unknown"

    def check_smart_entry_filters(self, symbol: str, signal) -> Tuple[bool, str]:
        """
        Apply all smart entry filters to a signal.

        Returns:
            (passed, reason)
        """
        if not signal or not signal.action:
            return True, "No signal"

        direction = signal.action
        failed_filters = []

        # Get 5m data for ATR/ADX calculation
        ltf_df, _, _ = self.get_market_data(symbol)
        if ltf_df is None:
            return True, "No data for filters"

        # Filter 1: ATR volatility check
        if self.use_atr_filter:
            atr_pct = self.calculate_atr(ltf_df, 14)
            if atr_pct > self.max_atr_pct:
                failed_filters.append(f"ATR {atr_pct:.2f}% > {self.max_atr_pct}% (too choppy)")

        # Filter 2: ADX trend check
        if self.use_adx_filter:
            adx = self.calculate_adx(ltf_df, 14)
            if adx < self.min_adx:
                failed_filters.append(f"ADX {adx:.1f} < {self.min_adx} (no trend)")

        # Filter 3: 1m pullback check
        if self.use_1m_pullback:
            passed, reason = self.check_1m_pullback(symbol, direction)
            if not passed:
                failed_filters.append(reason)

        # Filter 4: Candle confirmation
        if self.use_candle_confirm:
            passed, reason = self.check_candle_confirmation(symbol, direction)
            if not passed:
                failed_filters.append(reason)

        if failed_filters:
            return False, " | ".join(failed_filters)

        return True, "All filters passed"

    def check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period.

        Returns:
            True if can trade (not in cooldown)
        """
        if symbol not in self.last_trade_time:
            return True

        elapsed = datetime.now() - self.last_trade_time[symbol]
        if elapsed.total_seconds() < self.cooldown_minutes * 60:
            remaining = self.cooldown_minutes - (elapsed.total_seconds() / 60)
            logger.debug(f"[{symbol}] Cooldown: {remaining:.1f} minutes remaining")
            return False

        return True

    def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """
        Calculate position size using dynamic available margin split equally among symbols.

        Uses CURRENT available balance divided by number of symbols without positions.
        This ensures each symbol gets equal margin based on what's actually available.

        Args:
            symbol: Trading symbol
            entry_price: Entry price

        Returns:
            Position quantity
        """
        # Get CURRENT available balance (dynamic, not fixed)
        available_balance = self.client.get_available_balance()

        # Count how many symbols DON'T have positions (need margin)
        symbols_needing_margin = [s for s in self.symbols if s not in self.positions]
        num_symbols_needing_margin = len(symbols_needing_margin)

        if num_symbols_needing_margin == 0:
            logger.warning(f"[{symbol}] All symbols have positions, cannot calculate margin")
            return 0.0

        # Split available balance equally among symbols needing margin
        margin_per_symbol = available_balance / num_symbols_needing_margin

        # Apply a small buffer (95%) to avoid "insufficient margin" errors
        margin = margin_per_symbol * 0.95

        logger.info(f"[{symbol}] Dynamic margin: ${margin:.2f} (${available_balance:.2f} available / {num_symbols_needing_margin} symbols * 0.95)")

        # Position value = margin * leverage
        position_value = margin * self.leverage

        # Quantity = position_value / price
        quantity = position_value / entry_price

        # Get symbol precision
        from config.trading_config import SYMBOL_SETTINGS
        settings = SYMBOL_SETTINGS.get(symbol, {})
        qty_precision = settings.get("qty_precision", 3)

        quantity = round(quantity, qty_precision)

        logger.info(f"[{symbol}] Position size: {quantity} (margin ${margin:.2f} x {self.leverage}x leverage = ${position_value:.2f})")

        return quantity

    def open_position(self, symbol: str, signal, signal_id: str = "") -> bool:
        """
        Open a new position based on signal.

        Args:
            symbol: Trading symbol
            signal: ConfluenceSignal from strategy
            signal_id: ML signal ID for linking trade to signal log

        Returns:
            True if position opened successfully
        """
        try:
            # Get current price
            price_data = self.client.get_current_price(symbol)
            current_price = price_data["price"]

            # Determine position side
            if signal.action == "BUY":
                side = "BUY"
                position_side = "LONG"
            else:
                side = "SELL"
                position_side = "SHORT"

            # Calculate position size
            quantity = self.calculate_position_size(symbol, current_price)

            if quantity <= 0:
                logger.warning(f"[{symbol}] Invalid position size: {quantity}")
                return False

            # Place market order
            logger.info(f"[{symbol}] Opening {position_side} position...")
            logger.info(f"[{symbol}] Entry: ${current_price:,.4f} | Qty: {quantity}")

            order_result = self.client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                position_side=position_side
            )

            if "code" in order_result:
                logger.error(f"[{symbol}] Order failed: {order_result}")
                return False

            logger.info(f"[{symbol}] Market order placed: {order_result.get('orderId')}")

            # Wait for fill
            time.sleep(1)

            # Get actual entry price from position
            position = self.client.get_position(symbol, position_side)
            if not position:
                logger.error(f"[{symbol}] Position not found after order")
                return False

            actual_entry = position["entry_price"]
            actual_qty = position["quantity"]

            logger.info(f"[{symbol}] Position opened: {position_side} {actual_qty} @ ${actual_entry:,.4f}")

            # Calculate TP and SL prices using ASSET-SPECIFIC config
            strategy = self.strategies[symbol]
            sym_config = self.symbol_configs[symbol]
            sl_price, tp_price = strategy.calculate_exit_levels(actual_entry, position_side)

            logger.info(f"[{symbol}] TP: ${tp_price:,.4f} ({sym_config['tp_roi']*100:.0f}% ROI)")
            logger.info(f"[{symbol}] SL: ${sl_price:,.4f} ({sym_config['sl_roi']*100:.0f}% ROI)")

            # Place TP order
            close_side = "SELL" if position_side == "LONG" else "BUY"

            tp_result = self.client.place_take_profit(
                symbol=symbol,
                side=close_side,
                quantity=actual_qty,
                take_profit_price=tp_price,
                position_side=position_side
            )

            if "code" in tp_result and tp_result.get("code") != 0:
                logger.warning(f"[{symbol}] TP order warning: {tp_result}")
            else:
                tp_order_id = tp_result.get("orderId") or tp_result.get("algoId")
                logger.info(f"[{symbol}] TP order placed: {tp_order_id}")

            # Place SL order
            sl_result = self.client.place_stop_loss(
                symbol=symbol,
                side=close_side,
                quantity=actual_qty,
                stop_price=sl_price,
                position_side=position_side
            )

            if "code" in sl_result and sl_result.get("code") != 0:
                logger.warning(f"[{symbol}] SL order warning: {sl_result}")
            else:
                sl_order_id = sl_result.get("orderId") or sl_result.get("algoId")
                logger.info(f"[{symbol}] SL order placed: {sl_order_id}")

            # Calculate margin for ML logging
            margin = (actual_qty * actual_entry) / self.leverage

            # Get market conditions for ML logging
            ltf_df, _, _ = self.get_market_data(symbol)
            atr_pct = self.calculate_atr(ltf_df, 14) if ltf_df is not None else 0
            adx = self.calculate_adx(ltf_df, 14) if ltf_df is not None else 0

            # Calculate RSI for ML logging
            rsi = 50  # default
            if ltf_df is not None and len(ltf_df) >= 14:
                delta = ltf_df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 0.0001)
                rsi_series = 100 - (100 / (1 + rs))
                rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50

            # Calculate volume ratio for ML logging
            volume_ratio = 1.0
            if ltf_df is not None and len(ltf_df) >= 20:
                vol_sma = ltf_df['volume'].rolling(window=20).mean().iloc[-1]
                volume_ratio = ltf_df['volume'].iloc[-1] / (vol_sma + 0.0001)

            # LOG TRADE ENTRY FOR ML
            if signal_id:
                trade_id = self.ml_logger.log_trade_entry(
                    signal_id=signal_id,
                    symbol=symbol,
                    position_side=position_side,
                    entry_price=actual_entry,
                    quantity=actual_qty,
                    margin=margin,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    tp_roi=sym_config['tp_roi'] * 100,
                    sl_roi=sym_config['sl_roi'] * 100,
                    atr_pct=atr_pct,
                    adx=adx,
                    rsi=rsi,
                    volume_ratio=volume_ratio
                )
                # Store trade_id for linking to exit
                self.active_trade_ids[symbol] = trade_id

            # Track position
            self.positions[symbol] = {
                "side": position_side,
                "entry_price": actual_entry,
                "quantity": actual_qty,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_time": datetime.now(),
                "signal_strength": signal.strength.value,
                "confluence_score": signal.confluence_score,
                "signal_id": signal_id  # Store for ML linking
            }

            # Save tracked positions for restart detection
            self._save_tracked_positions()

            # Set cooldown
            self.last_trade_time[symbol] = datetime.now()

            return True

        except Exception as e:
            logger.error(f"[{symbol}] Failed to open position: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_profit_lock(self, symbol: str) -> bool:
        """
        Check if we should lock profit by closing early.

        Conditions:
        1. Position is at 30%+ ROI profit
        2. HTF trend has reversed (EMA crossover against position)

        Returns:
            True if position was closed to lock profit
        """
        if not self.use_profit_lock:
            return False

        if symbol not in self.positions:
            return False

        try:
            pos = self.positions[symbol]

            # Get current price
            price_data = self.client.get_current_price(symbol)
            current_price = price_data["price"]
            entry = pos["entry_price"]

            # Calculate current ROI
            if pos["side"] == "LONG":
                price_move = (current_price - entry) / entry * 100
            else:
                price_move = (entry - current_price) / entry * 100
            roi = price_move * self.leverage

            # Check if we're at minimum profit level
            if roi < self.profit_lock_min_roi:
                return False

            # Get HTF data to check trend
            _, htf_df, _ = self.get_market_data(symbol)
            if htf_df is None or len(htf_df) < 50:
                return False

            # Calculate HTF EMAs
            htf_close = htf_df['close']
            htf_ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            htf_ema50 = htf_close.ewm(span=50, adjust=False).mean().iloc[-1]

            # STRENGTHENED: Check for STRONG trend reversal (not just EMA cross)
            # Require: EMA cross + Price on wrong side + Clear EMA separation
            current_price = close.iloc[-1]
            ema_diff_pct = abs(htf_ema21 - htf_ema50) / htf_ema50 * 100
            
            trend_reversed = False
            if pos["side"] == "LONG":
                # LONG position - check for STRONG bearish reversal
                # Need: EMA21 < EMA50 AND price below EMA21 AND clear separation
                if htf_ema21 < htf_ema50 and current_price < htf_ema21 and ema_diff_pct > 0.3:
                    trend_reversed = True
            elif pos["side"] == "SHORT":
                # SHORT position - check for STRONG bullish reversal
                # Need: EMA21 > EMA50 AND price above EMA21 AND clear separation
                if htf_ema21 > htf_ema50 and current_price > htf_ema21 and ema_diff_pct > 0.3:
                    trend_reversed = True

            if not trend_reversed:
                return False

            # Close the position to lock profit!
            logger.info(f"[{symbol}] PROFIT LOCK triggered at {roi:+.1f}% ROI - Trend reversed!")

            # Close position using proper client method
            position_side = pos["side"]  # "LONG" or "SHORT"
            close_side = "SELL" if position_side == "LONG" else "BUY"

            result = self.client.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=pos["quantity"],
                position_side=position_side  # Required for hedge mode
            )

            if result:
                logger.info(f"[{symbol}] Position closed to lock ${roi * pos['quantity'] * entry / self.leverage / 100:+.2f} profit")

                # Get actual PnL
                try:
                    time.sleep(1)  # Wait for order to fill
                    income = self.client.get_income_history(symbol=symbol, income_type="REALIZED_PNL", limit=1)
                    pnl = float(income[0].get("income", 0)) if income else 0
                except:
                    pnl = (roi / 100) * (pos["quantity"] * entry / self.leverage)

                # Update stats
                self.wins_today += 1
                self.symbol_stats[symbol]["wins"] += 1
                self.pnl_today += pnl
                self.symbol_stats[symbol]["pnl"] += pnl
                self.trades_today += 1

                logger.info(f"[{symbol}] PROFIT LOCK WIN! PnL: ${pnl:+.2f} | ROI: {roi:+.1f}%")

                # Log to ML
                if symbol in self.active_trade_ids:
                    trade_id = self.active_trade_ids[symbol]
                    entry_time = pos.get("entry_time", datetime.now())
                    duration = (datetime.now() - entry_time).total_seconds() / 60 if isinstance(entry_time, datetime) else 0
                    self.ml_logger.log_trade_exit(
                        trade_id=trade_id,
                        exit_price=current_price,
                        exit_type="PROFIT_LOCK",
                        pnl=pnl,
                        roi_pct=roi,
                        price_move_pct=price_move,
                        duration_minutes=duration
                    )
                    del self.active_trade_ids[symbol]

                # Clean up
                del self.positions[symbol]
                self._save_tracked_positions()
                self._save_stats()

                # Cancel any remaining TP/SL orders
                self.client.cancel_all_orders(symbol)

                return True

        except Exception as e:
            logger.error(f"[{symbol}] Profit lock check failed: {e}")

        return False

    def check_fakeout_protection(self, symbol: str, roi: float, htf_trend: str, ltf_trend: str, confirmations: int) -> Optional[str]:
        """
        Smart fakeout protection - exit early on suspected fakeout entries.

        IMPORTANT: Both HTF and LTF must be against us to count as a real reversal.
        If only one timeframe is against us, we wait for more confirmation.

        Cases:
        1. ROI >= +10% and BOTH reversed: Move SL to breakeven
        2. ROI 0-10% and confirms <= 1 after 3 cycles: Close (small profit)
        3. ROI -10% to 0% and confirms <= 1: Close (cut loss early)
        4. ROI < -15% and BOTH reversed: Close immediately (damage control)

        Returns:
            Action taken: "BREAKEVEN", "SMALL_PROFIT_EXIT", "CUT_LOSS", "DAMAGE_CONTROL", or None
        """
        if not self.use_fakeout_protection:
            return None

        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        original_trend = "BULLISH" if pos["side"] == "LONG" else "BEARISH"
        
        # STRENGTHENED REVERSAL DETECTION:
        # Simple EMA cross is NOT enough - require STRONG reversal confirmation
        # htf_trend and ltf_trend coming from the main loop are already calculated
        # But we need to check for STRONG reversal, not just EMA cross
        
        # For fakeout protection, we need BOTH to be strongly against us
        # The trend strings are "BULLISH", "BEARISH", or "MIXED"
        htf_reversed = htf_trend != original_trend and htf_trend != "MIXED"
        ltf_reversed = ltf_trend != original_trend and ltf_trend != "MIXED"

        # BOTH timeframes must be STRONGLY against us (not MIXED)
        is_reversed = htf_reversed and ltf_reversed

        # Track reversal cycles
        if symbol not in self.reversal_cycles:
            self.reversal_cycles[symbol] = 0

        if is_reversed:
            self.reversal_cycles[symbol] += 1
        else:
            # At least one timeframe still in our favor - reset counter
            self.reversal_cycles[symbol] = 0
            return None

        cycles_reversed = self.reversal_cycles[symbol]

        try:
            # Case 4: DAMAGE CONTROL - ROI < -15% and reversed
            if roi < self.damage_control_roi:
                logger.info(f"[{symbol}] ⚠️ DAMAGE CONTROL: ROI {roi:.1f}% < {self.damage_control_roi}% with reversed trend!")
                self._close_position_early(symbol, "DAMAGE_CONTROL", roi)
                return "DAMAGE_CONTROL"

            # Case 1: Move SL to breakeven at +10% if reversed
            if roi >= self.breakeven_roi_threshold and cycles_reversed >= self.reversal_cycle_threshold:
                # Move SL to entry price (breakeven)
                entry_price = pos["entry_price"]
                current_sl = pos.get("sl_price", 0)

                # Only move if SL is worse than breakeven
                if pos["side"] == "LONG" and current_sl < entry_price:
                    logger.info(f"[{symbol}] 🛡️ BREAKEVEN: Moving SL to entry ${entry_price:.4f} (was ${current_sl:.4f})")
                    self._move_sl_to_breakeven(symbol, entry_price)
                    return "BREAKEVEN"
                elif pos["side"] == "SHORT" and current_sl > entry_price:
                    logger.info(f"[{symbol}] 🛡️ BREAKEVEN: Moving SL to entry ${entry_price:.4f} (was ${current_sl:.4f})")
                    self._move_sl_to_breakeven(symbol, entry_price)
                    return "BREAKEVEN"

            # Case 2 & 3: Small profit exit or cut loss early
            if cycles_reversed >= self.reversal_cycle_threshold and confirmations <= 1:
                if 0 <= roi < self.small_profit_exit_roi:
                    logger.info(f"[{symbol}] 💰 SMALL PROFIT EXIT: ROI +{roi:.1f}%, confirms {confirmations}/4, reversed {cycles_reversed} cycles")
                    self._close_position_early(symbol, "SMALL_PROFIT_EXIT", roi)
                    return "SMALL_PROFIT_EXIT"
                elif -10 <= roi < 0:
                    logger.info(f"[{symbol}] ✂️ CUT LOSS EARLY: ROI {roi:.1f}%, confirms {confirmations}/4, reversed {cycles_reversed} cycles")
                    self._close_position_early(symbol, "CUT_LOSS_EARLY", roi)
                    return "CUT_LOSS_EARLY"

        except Exception as e:
            logger.error(f"[{symbol}] Fakeout protection error: {e}")

        return None

    def _move_sl_to_breakeven(self, symbol: str, breakeven_price: float):
        """Move stop loss to breakeven price."""
        try:
            pos = self.positions[symbol]

            # Cancel existing orders
            self.client.cancel_all_orders(symbol)

            # Determine sides
            if pos["side"] == "LONG":
                sl_side = "SELL"
                tp_side = "SELL"
            else:
                sl_side = "BUY"
                tp_side = "BUY"

            # Place new SL at breakeven using the proper client method
            sl_order = self.client.place_stop_loss(
                symbol=symbol,
                side=sl_side,
                quantity=pos["quantity"],
                stop_price=breakeven_price
            )

            # Also place TP order again
            tp_order = self.client.place_take_profit(
                symbol=symbol,
                side=tp_side,
                quantity=pos["quantity"],
                take_profit_price=pos["tp_price"]
            )

            # Update tracked position
            pos["sl_price"] = breakeven_price
            pos["sl_moved_to_breakeven"] = True
            self._save_tracked_positions()

            logger.info(f"[{symbol}] SL moved to breakeven: ${breakeven_price:.4f}")

        except Exception as e:
            logger.error(f"[{symbol}] Failed to move SL to breakeven: {e}")

    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol."""
        precisions = {
            "DOTUSDT": 4,
            "BNBUSDT": 2,
            "XRPUSDT": 4,
            "ADAUSDT": 5
        }
        return precisions.get(symbol, 4)

    def _close_position_early(self, symbol: str, reason: str, roi: float):
        """Close position early for fakeout protection."""
        try:
            pos = self.positions[symbol]
            position_side = pos["side"]  # "LONG" or "SHORT"
            close_side = "SELL" if position_side == "LONG" else "BUY"

            # Close position using proper client method
            result = self.client.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=pos["quantity"],
                position_side=position_side  # Required for hedge mode
            )

            if result:
                # Get actual PnL
                time.sleep(1)
                try:
                    income = self.client.get_income_history(symbol=symbol, income_type="REALIZED_PNL", limit=1)
                    pnl = float(income[0].get("income", 0)) if income else 0
                except:
                    margin = (pos["quantity"] * pos["entry_price"]) / self.leverage
                    pnl = margin * (roi / 100)

                # Update stats
                if pnl >= 0:
                    self.wins_today += 1
                    self.symbol_stats[symbol]["wins"] += 1
                else:
                    self.losses_today += 1
                    self.symbol_stats[symbol]["losses"] += 1

                self.pnl_today += pnl
                self.symbol_stats[symbol]["pnl"] += pnl
                self.trades_today += 1

                logger.info(f"[{symbol}] {reason}: Closed at ROI {roi:+.1f}% | PnL: ${pnl:+.2f}")

                # Log to ML
                if symbol in self.active_trade_ids:
                    trade_id = self.active_trade_ids[symbol]
                    entry_time = pos.get("entry_time", datetime.now())
                    duration = (datetime.now() - entry_time).total_seconds() / 60 if isinstance(entry_time, datetime) else 0
                    price_data = self.client.get_current_price(symbol)
                    self.ml_logger.log_trade_exit(
                        trade_id=trade_id,
                        exit_price=price_data["price"],
                        exit_type=reason,
                        pnl=pnl,
                        roi_pct=roi,
                        price_move_pct=roi / self.leverage,
                        duration_minutes=duration
                    )
                    del self.active_trade_ids[symbol]

                # Clean up
                del self.positions[symbol]
                if symbol in self.reversal_cycles:
                    del self.reversal_cycles[symbol]
                self._save_tracked_positions()
                self._save_stats()

                # Cancel any remaining orders
                self.client.cancel_all_orders(symbol)

        except Exception as e:
            logger.error(f"[{symbol}] Failed to close position early: {e}")

    def check_trailing_profit_lock(self, symbol: str, roi: float) -> bool:
        """
        Trailing profit lock - never give back too much profit.

        Once ROI hits 30%, we track the peak and close if it drops too much.

        Logic:
        - Track peak ROI for each position
        - Once peak >= 30%, set a floor at (peak - 15%) but minimum 15%
        - If ROI drops to floor, close to lock profit

        Example:
        - Peak hits 34% → floor = 34 - 15 = 19%
        - If ROI drops to 19% → close and lock +19% profit

        Returns:
            True if position was closed
        """
        if not self.use_trailing_profit_lock:
            return False

        if symbol not in self.positions:
            return False

        # Initialize peak tracking if not exists
        if symbol not in self.peak_roi:
            self.peak_roi[symbol] = roi

        # Update peak if current ROI is higher
        if roi > self.peak_roi[symbol]:
            self.peak_roi[symbol] = roi

        peak = self.peak_roi[symbol]

        # Only activate trailing lock once peak hits threshold
        if peak < self.trailing_lock_activation:
            return False

        # Calculate floor: peak - distance, but minimum floor
        floor = max(peak - self.trailing_lock_distance, self.trailing_lock_min_floor)

        # Check if ROI dropped to floor
        if roi <= floor:
            logger.info(f"[{symbol}] 📉 TRAILING PROFIT LOCK: ROI dropped from peak {peak:.1f}% to {roi:.1f}% (floor: {floor:.1f}%)")

            try:
                pos = self.positions[symbol]
                position_side = pos["side"]  # "LONG" or "SHORT"
                close_side = "SELL" if position_side == "LONG" else "BUY"

                # Close position using proper client method
                result = self.client.place_market_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=pos["quantity"],
                    position_side=position_side  # Required for hedge mode
                )

                if result:
                    # Get actual PnL
                    time.sleep(1)
                    try:
                        income = self.client.get_income_history(symbol=symbol, income_type="REALIZED_PNL", limit=1)
                        pnl = float(income[0].get("income", 0)) if income else 0
                    except:
                        margin = (pos["quantity"] * pos["entry_price"]) / self.leverage
                        pnl = margin * (roi / 100)

                    # Update stats (this is a WIN - we locked profit)
                    self.wins_today += 1
                    self.symbol_stats[symbol]["wins"] += 1
                    self.pnl_today += pnl
                    self.symbol_stats[symbol]["pnl"] += pnl
                    self.trades_today += 1

                    logger.info(f"[{symbol}] TRAILING LOCK WIN! Peak: {peak:.1f}% → Locked: {roi:.1f}% | PnL: ${pnl:+.2f}")

                    # Log to ML
                    if symbol in self.active_trade_ids:
                        trade_id = self.active_trade_ids[symbol]
                        entry_time = pos.get("entry_time", datetime.now())
                        duration = (datetime.now() - entry_time).total_seconds() / 60 if isinstance(entry_time, datetime) else 0
                        price_data = self.client.get_current_price(symbol)
                        self.ml_logger.log_trade_exit(
                            trade_id=trade_id,
                            exit_price=price_data["price"],
                            exit_type="TRAILING_PROFIT_LOCK",
                            pnl=pnl,
                            roi_pct=roi,
                            price_move_pct=roi / self.leverage,
                            duration_minutes=duration
                        )
                        del self.active_trade_ids[symbol]

                    # Clean up
                    del self.positions[symbol]
                    if symbol in self.peak_roi:
                        del self.peak_roi[symbol]
                    if symbol in self.reversal_cycles:
                        del self.reversal_cycles[symbol]
                    self._save_tracked_positions()
                    self._save_stats()

                    # Cancel any remaining orders
                    self.client.cancel_all_orders(symbol)

                    return True

            except Exception as e:
                logger.error(f"[{symbol}] Failed to execute trailing profit lock: {e}")

        return False

    def check_position(self, symbol: str) -> Optional[str]:
        """
        Check if position was closed (TP or SL hit).

        Args:
            symbol: Trading symbol

        Returns:
            "TP", "SL", or None if still open
        """
        if symbol not in self.positions:
            return None

        try:
            # Check if position still exists
            tracked = self.positions[symbol]
            position_side = tracked["side"]

            position = self.client.get_position(symbol, position_side)

            if not position or position["quantity"] == 0:
                # Position closed - get actual realized PnL from API
                entry_price = tracked["entry_price"]
                entry_time = tracked.get("entry_time", datetime.now())

                # Query income history to get actual realized PnL
                try:
                    income_history = self.client.get_income_history(symbol=symbol, income_type="REALIZED_PNL", limit=5)
                    if income_history:
                        # Get the most recent realized PnL for this symbol
                        pnl = float(income_history[0].get("income", 0))
                    else:
                        # Fallback: estimate from recent trades
                        recent_trades = self.client.get_recent_trades(symbol, limit=5)
                        if recent_trades:
                            pnl = sum(float(t.get("realizedPnl", 0)) for t in recent_trades)
                        else:
                            pnl = 0
                except Exception as e:
                    logger.warning(f"[{symbol}] Could not get realized PnL: {e}")
                    pnl = 0

                # Determine win/loss based on actual PnL (positive = win, negative = loss)
                if pnl > 0:
                    exit_type = "TP"
                else:
                    exit_type = "SL"

                # Calculate ROI from actual PnL
                margin = (tracked["quantity"] * entry_price) / self.leverage
                roi = (pnl / margin) * 100 if margin > 0 else 0

                # Calculate price move %
                exit_price = tracked["tp_price"] if exit_type == "TP" else tracked["sl_price"]
                if tracked["side"] == "LONG":
                    price_move_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    price_move_pct = (entry_price - exit_price) / entry_price * 100

                # Calculate duration
                duration_minutes = 0
                if isinstance(entry_time, datetime):
                    duration_minutes = (datetime.now() - entry_time).total_seconds() / 60

                if exit_type == "TP":
                    self.wins_today += 1
                    self.symbol_stats[symbol]["wins"] += 1
                    logger.info(f"[{symbol}] Position closed - TP HIT! PnL: ${pnl:+.2f} | ROI: {roi:+.1f}%")
                else:
                    self.losses_today += 1
                    self.symbol_stats[symbol]["losses"] += 1
                    logger.info(f"[{symbol}] Position closed - SL HIT. PnL: ${pnl:.2f} | ROI: {roi:.1f}%")

                self.pnl_today += pnl
                self.symbol_stats[symbol]["pnl"] += pnl
                self.trades_today += 1

                # LOG TRADE EXIT FOR ML
                if symbol in self.active_trade_ids:
                    trade_id = self.active_trade_ids[symbol]

                    # Get current market conditions for exit logging
                    ltf_df, _, _ = self.get_market_data(symbol)
                    exit_atr = self.calculate_atr(ltf_df, 14) if ltf_df is not None else 0
                    exit_adx = self.calculate_adx(ltf_df, 14) if ltf_df is not None else 0
                    exit_rsi = 50
                    if ltf_df is not None and len(ltf_df) >= 14:
                        delta = ltf_df['close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / (loss + 0.0001)
                        rsi_series = 100 - (100 / (1 + rs))
                        exit_rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50

                    self.ml_logger.log_trade_exit(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        exit_type=exit_type,
                        pnl=pnl,
                        roi_pct=roi,
                        price_move_pct=price_move_pct,
                        duration_minutes=duration_minutes,
                        atr_pct=exit_atr,
                        adx=exit_adx,
                        rsi=exit_rsi
                    )
                    del self.active_trade_ids[symbol]

                # Save stats and update tracked positions
                self._save_stats()

                # Clean up
                del self.positions[symbol]
                self._save_tracked_positions()

                # Cancel any remaining orders
                self.client.cancel_orders_for_side(symbol, position_side)

                # Scan for new signal immediately after close
                logger.info(f"[{symbol}] Scanning for new signal...")
                signal = self.analyze_symbol(symbol)
                if signal:
                    trend_str = signal.trend.value if hasattr(signal, 'trend') else "?"
                    if signal.action:
                        logger.info(f"[{symbol}] Signal: {signal.action} | Trend: {trend_str} | Score: {signal.confluence_score}/4")
                        if signal.confluence_score < 3:
                            logger.info(f"[{symbol}] Waiting for stronger confluence (need 3+)")
                    else:
                        logger.info(f"[{symbol}] No signal | Trend: {trend_str} | Waiting for entry conditions")
                else:
                    logger.info(f"[{symbol}] Could not analyze - will retry next cycle")

                return exit_type

            return None

        except Exception as e:
            logger.error(f"[{symbol}] Failed to check position: {e}")
            return None

    def analyze_symbol(self, symbol: str, verbose: bool = False) -> Optional[dict]:
        """
        Analyze a symbol for trading signals.

        Args:
            symbol: Trading symbol
            verbose: Show detailed indicator breakdown

        Returns:
            Signal dict or None
        """
        try:
            # Get market data (15m LTF, 1H HTF, 5m confirmation)
            ltf_df, htf_df, m5_df = self.get_market_data(symbol)

            if ltf_df is None or htf_df is None:
                return None

            # Calculate 5m EMA confirmation
            m5_ema_bullish = True  # Default to True if no 5m data
            m5_ema_bearish = True
            if m5_df is not None and len(m5_df) >= 25:
                m5_close = m5_df['close']
                m5_ema9 = m5_close.ewm(span=9, adjust=False).mean().iloc[-1]
                m5_ema21 = m5_close.ewm(span=21, adjust=False).mean().iloc[-1]
                m5_ema_bullish = m5_ema9 > m5_ema21
                m5_ema_bearish = m5_ema9 < m5_ema21

            # Get strategy and calculate indicators
            strategy = self.strategies[symbol]
            indicators = strategy.calculate_indicators(ltf_df)
            
            if indicators:
                # Inject 5m EMA confirmation into indicators
                indicators["5m_ema_bullish"] = m5_ema_bullish
                indicators["5m_ema_bearish"] = m5_ema_bearish
            
            # Get HTF trend
            htf_trend, htf_distance = strategy.get_htf_trend(htf_df)
            
            if indicators:
                indicators["htf_trend"] = htf_trend.value
                indicators["htf_distance_pct"] = htf_distance
            
            # Now get signal with injected 5m data
            # The strategy.should_enter recalculates indicators, so we need a workaround
            # Temporarily patch the strategy's calculate_indicators to include our 5m data
            original_calc = strategy.calculate_indicators
            def patched_calc(df):
                result = original_calc(df)
                if result:
                    result["5m_ema_bullish"] = m5_ema_bullish
                    result["5m_ema_bearish"] = m5_ema_bearish
                return result
            
            strategy.calculate_indicators = patched_calc
            signal = strategy.should_enter(ltf_df, htf_df)
            strategy.calculate_indicators = original_calc  # Restore

            if signal.action:
                logger.info(f"[{symbol}] Signal: {signal.action} | Strength: {signal.strength.value} | Score: {signal.confluence_score}/6")
                logger.info(f"[{symbol}] Reason: {signal.reason}")
            elif verbose:
                # Show detailed indicator breakdown when no signal
                self._log_signal_diagnostics(symbol, ltf_df, htf_df, signal)

            return signal

        except Exception as e:
            logger.error(f"[{symbol}] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_signal_diagnostics(self, symbol: str, ltf_df: pd.DataFrame, htf_df: pd.DataFrame, signal):
        """Log detailed indicator breakdown for signal diagnostics."""
        try:
            # Calculate LTF (5m) indicators
            ltf_close = ltf_df['close']
            ltf_ema9 = ltf_close.ewm(span=9, adjust=False).mean().iloc[-1]
            ltf_ema21 = ltf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            ltf_ema50 = ltf_close.ewm(span=50, adjust=False).mean().iloc[-1]
            ltf_price = ltf_close.iloc[-1]

            # RSI
            delta = ltf_close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 0.0001)
            ltf_rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # MACD
            ema12 = ltf_close.ewm(span=12, adjust=False).mean()
            ema26 = ltf_close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            ltf_macd = macd.iloc[-1]
            ltf_macd_sig = macd_signal.iloc[-1]
            macd_cross = "BULL" if ltf_macd > ltf_macd_sig else "BEAR"

            # HTF (15m) indicators
            htf_close = htf_df['close']
            htf_ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            htf_ema50 = htf_close.ewm(span=50, adjust=False).mean().iloc[-1]

            # Trend direction
            trend = signal.trend.value if hasattr(signal, 'trend') else "?"

            # EMA alignment check
            ltf_ema_bullish = ltf_ema9 > ltf_ema21 > ltf_ema50
            ltf_ema_bearish = ltf_ema9 < ltf_ema21 < ltf_ema50
            htf_ema_bullish = htf_ema21 > htf_ema50
            htf_ema_bearish = htf_ema21 < htf_ema50

            # Build status indicators
            ema_status = "✓" if (ltf_ema_bullish or ltf_ema_bearish) else "✗"
            rsi_status = "✓" if (30 < ltf_rsi < 70) else "✗"  # Not overbought/oversold
            macd_status = "✓" if ((trend == "BULLISH" and ltf_macd > ltf_macd_sig) or (trend == "BEARISH" and ltf_macd < ltf_macd_sig)) else "✗"

            # Log compact diagnostics
            logger.info(f"┌─ {symbol} SIGNAL CHECK ──────────────────────────")
            logger.info(f"│ Trend: {trend} | Price: ${ltf_price:.4f}")
            logger.info(f"│ 5m EMA: 9={ltf_ema9:.4f} | 21={ltf_ema21:.4f} | 50={ltf_ema50:.4f}")
            logger.info(f"│ 5m RSI: {ltf_rsi:.1f} | MACD: {ltf_macd:.6f} vs Sig: {ltf_macd_sig:.6f} ({macd_cross})")
            logger.info(f"│ 15m EMA: 21={htf_ema21:.4f} | 50={htf_ema50:.4f}")
            logger.info(f"│ Checks: EMA-align:{ema_status} RSI:{rsi_status} MACD:{macd_status}")
            logger.info(f"│ Score: {signal.confluence_score}/4 | Reason: {signal.reason if signal.reason else 'Waiting for confluence'}")
            logger.info(f"└────────────────────────────────────────────────")

        except Exception as e:
            logger.debug(f"[{symbol}] Diagnostics error: {e}")

    def run_cycle(self):
        """Run one trading cycle."""
        logger.info("-" * 40)
        logger.info(f"Cycle: {datetime.now().strftime('%H:%M:%S')}")

        for symbol in self.symbols:
            try:
                # Check if we have an existing position
                if symbol in self.positions:
                    # First check for profit lock (close early if trend reverses while in profit)
                    if self.check_profit_lock(symbol):
                        continue  # Position was closed, move to next symbol

                    # Check if position was closed (TP/SL hit)
                    exit_type = self.check_position(symbol)
                    if exit_type:
                        logger.info(f"[{symbol}] Position exit: {exit_type}")
                    else:
                        # Position still open - show detailed status with trend info
                        pos = self.positions[symbol]
                        try:
                            price_data = self.client.get_current_price(symbol)
                            current_price = price_data["price"]
                            entry = pos["entry_price"]
                            qty = pos["quantity"]

                            # Calculate price move %
                            if pos["side"] == "LONG":
                                price_move = (current_price - entry) / entry * 100
                                roi = price_move * self.leverage
                                to_tp_price = (pos["tp_price"] - current_price) / current_price * 100
                                to_sl_price = (current_price - pos["sl_price"]) / current_price * 100
                            else:
                                price_move = (entry - current_price) / entry * 100
                                roi = price_move * self.leverage
                                to_tp_price = (current_price - pos["tp_price"]) / current_price * 100
                                to_sl_price = (pos["sl_price"] - current_price) / current_price * 100

                            # Check trailing profit lock FIRST (before anything else)
                            if self.check_trailing_profit_lock(symbol, roi):
                                continue  # Position was closed, move to next symbol

                            # Calculate margin and unrealized PnL
                            position_value = qty * entry
                            margin = position_value / self.leverage
                            unrealized_pnl = margin * (roi / 100)

                            # ROI to TP/SL
                            to_tp_roi = to_tp_price * self.leverage
                            to_sl_roi = to_sl_price * self.leverage

                            # Get HTF trend info for reversal check
                            try:
                                _, htf_df, _ = self.get_market_data(symbol)
                                ltf_df, _, _ = self.get_market_data(symbol)

                                # HTF EMAs (1H) - STRENGTHENED reversal detection
                                htf_close = htf_df['close']
                                htf_price = htf_close.iloc[-1]
                                htf_ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
                                htf_ema50 = htf_close.ewm(span=50, adjust=False).mean().iloc[-1]
                                
                                # STRONG trend requires: EMA alignment AND price confirmation
                                # Not just EMA cross, but price must be on the right side
                                ema_diff_pct = abs(htf_ema21 - htf_ema50) / htf_ema50 * 100
                                
                                if htf_ema21 > htf_ema50 and htf_price > htf_ema21:
                                    htf_trend = "BULLISH"  # Strong bullish
                                elif htf_ema21 < htf_ema50 and htf_price < htf_ema21:
                                    htf_trend = "BEARISH"  # Strong bearish
                                elif htf_ema21 > htf_ema50:
                                    htf_trend = "BULLISH" if ema_diff_pct > 0.3 else "MIXED"  # Weak bullish needs clear separation
                                elif htf_ema21 < htf_ema50:
                                    htf_trend = "BEARISH" if ema_diff_pct > 0.3 else "MIXED"  # Weak bearish needs clear separation
                                else:
                                    htf_trend = "MIXED"

                                # LTF indicators (5m)
                                ltf_close = ltf_df['close']
                                ltf_ema9 = ltf_close.ewm(span=9, adjust=False).mean().iloc[-1]
                                ltf_ema21 = ltf_close.ewm(span=21, adjust=False).mean().iloc[-1]
                                ltf_ema50 = ltf_close.ewm(span=50, adjust=False).mean().iloc[-1]

                                # RSI
                                delta = ltf_close.diff()
                                gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
                                loss = (-delta).where(delta < 0, 0).ewm(span=14, adjust=False).mean()
                                rs = gain / loss
                                rsi = (100 - (100 / (1 + rs))).iloc[-1]

                                # MACD
                                ema12 = ltf_close.ewm(span=12, adjust=False).mean()
                                ema26 = ltf_close.ewm(span=26, adjust=False).mean()
                                macd = ema12 - ema26
                                signal_line = macd.ewm(span=9, adjust=False).mean()
                                macd_val = macd.iloc[-1]
                                signal_val = signal_line.iloc[-1]
                                macd_trend = "BULL" if macd_val > signal_val else "BEAR"

                                # Check reversal conditions
                                original_trend = "BULLISH" if pos["side"] == "LONG" else "BEARISH"

                                # LTF trend based on EMA alignment
                                if ltf_ema9 > ltf_ema21 > ltf_ema50:
                                    ltf_trend = "BULLISH"
                                elif ltf_ema9 < ltf_ema21 < ltf_ema50:
                                    ltf_trend = "BEARISH"
                                else:
                                    ltf_trend = "MIXED"

                                # Both must be against us for real reversal
                                htf_reversed = htf_trend != original_trend
                                ltf_reversed = ltf_trend != original_trend and ltf_trend != "MIXED"
                                both_reversed = htf_reversed and ltf_reversed

                                if both_reversed:
                                    trend_match = "✗ BOTH REVERSED"
                                elif htf_reversed:
                                    trend_match = f"⚡ HTF reversed, 5m {ltf_trend}"
                                elif ltf_reversed:
                                    trend_match = f"⚡ 5m reversed, HTF OK"
                                else:
                                    trend_match = "✓"

                                # EMA alignment for exit
                                if pos["side"] == "LONG":
                                    ema_aligned = ltf_ema9 > ltf_ema21 > ltf_ema50
                                    rsi_ok = rsi > 30  # Not oversold
                                    macd_ok = macd_val > signal_val
                                else:
                                    ema_aligned = ltf_ema9 < ltf_ema21 < ltf_ema50
                                    rsi_ok = rsi < 70  # Not overbought
                                    macd_ok = macd_val < signal_val

                                # Count confirmations still valid
                                confirmations = sum([ema_aligned, rsi_ok, macd_ok, htf_trend == original_trend])

                                # Trailing profit lock status
                                peak = self.peak_roi.get(symbol, roi)
                                if peak >= self.trailing_lock_activation:
                                    floor = max(peak - self.trailing_lock_distance, self.trailing_lock_min_floor)
                                    trailing_status = f"📈 Peak: {peak:.1f}% | Floor: {floor:.1f}%"
                                else:
                                    trailing_status = f"Peak: {peak:.1f}% (need {self.trailing_lock_activation}% to activate)"

                                # Profit lock status (only triggers if BOTH reversed)
                                if roi >= self.profit_lock_min_roi:
                                    if both_reversed:
                                        lock_status = "⚠️ PROFIT LOCK TRIGGERED!"
                                    else:
                                        lock_status = f"🔓 Ready (ROI≥{self.profit_lock_min_roi}%, trend OK)"
                                else:
                                    lock_status = f"ROI needs {self.profit_lock_min_roi - roi:.1f}% more"

                                # Fakeout protection status (only counts if BOTH reversed)
                                cycles_reversed = self.reversal_cycles.get(symbol, 0)
                                sl_at_breakeven = pos.get("sl_moved_to_breakeven", False)

                                if not both_reversed:
                                    if htf_reversed or ltf_reversed:
                                        fakeout_status = "👀 One TF against, watching"
                                    else:
                                        fakeout_status = "✓ Both TF valid"
                                elif sl_at_breakeven:
                                    fakeout_status = "🛡️ SL at breakeven (risk-free)"
                                elif cycles_reversed < self.reversal_cycle_threshold:
                                    fakeout_status = f"⏳ BOTH reversed ({cycles_reversed}/{self.reversal_cycle_threshold} cycles)"
                                elif roi >= self.breakeven_roi_threshold:
                                    fakeout_status = "🛡️ Will move SL to breakeven"
                                elif confirmations <= 1:
                                    if roi >= 0:
                                        fakeout_status = f"⚠️ SMALL PROFIT EXIT (confirms {confirmations}/4)"
                                    else:
                                        fakeout_status = f"⚠️ CUT LOSS EARLY (confirms {confirmations}/4)"
                                else:
                                    fakeout_status = f"👀 Monitoring (confirms {confirmations}/4)"

                                # Check fakeout protection (pass both trends)
                                fakeout_action = self.check_fakeout_protection(symbol, roi, htf_trend, ltf_trend, confirmations)
                                if fakeout_action:
                                    continue  # Position was handled

                            except Exception as trend_e:
                                htf_trend = "?"
                                ltf_trend = "?"
                                original_trend = "BULLISH" if pos["side"] == "LONG" else "BEARISH"
                                trend_match = "?"
                                confirmations = 0
                                rsi = 0
                                macd_trend = "?"
                                lock_status = "Error checking trend"
                                fakeout_status = "Error"
                                cycles_reversed = 0
                                trailing_status = "Error"

                            logger.info(f"┌─ {symbol} {pos['side']} ─────────────────────────────────")
                            logger.info(f"│ Entry: ${entry:,.4f} | Now: ${current_price:,.4f} | Qty: {qty}")
                            logger.info(f"│ Price Move: {price_move:+.3f}% | ROI: {roi:+.1f}% | PnL: ${unrealized_pnl:+.2f}")
                            logger.info(f"│ Margin: ${margin:.2f} | Position: ${position_value:.2f}")
                            logger.info(f"│ To TP: {to_tp_price:.3f}% ({to_tp_roi:+.1f}% ROI) | To SL: {to_sl_price:.3f}% ({to_sl_roi:.1f}% ROI)")
                            logger.info(f"│ ── TREND CHECK ──")
                            logger.info(f"│ Entry: {original_trend} | 15m: {htf_trend} | 5m: {ltf_trend} {trend_match}")
                            logger.info(f"│ 5m RSI: {rsi:.1f} | MACD: {macd_trend} | Confirms: {confirmations}/4")
                            logger.info(f"│ Trailing: {trailing_status}")
                            logger.info(f"│ Profit Lock: {lock_status}")
                            logger.info(f"│ Fakeout: {fakeout_status}")
                            logger.info(f"└────────────────────────────────────────────────")
                        except Exception as e:
                            logger.info(f"[{symbol}] {pos['side']} @ ${pos['entry_price']:,.4f} (monitoring) - {e}")
                    continue

                # Check cooldown
                if not self.check_cooldown(symbol):
                    continue

                # Get market data for ML logging
                ltf_df, htf_df, _ = self.get_market_data(symbol)
                m1_df = self.get_1m_data(symbol)

                # Analyze for new signal (verbose=True shows diagnostics when no signal)
                signal = self.analyze_symbol(symbol, verbose=True)

                if signal and signal.action:
                    # We have a signal - check if strong enough
                    if signal.confluence_score >= 3:
                        logger.info(f"[{symbol}] SIGNAL: {signal.action} (Score: {signal.confluence_score}/4)")

                        # Apply SMART ENTRY FILTERS before opening
                        filters_passed, filter_reason = self.check_smart_entry_filters(symbol, signal)

                        if filters_passed:
                            logger.info(f"[{symbol}] Smart filters: {filter_reason}")

                            # LOG SIGNAL FOR ML (will be traded)
                            signal_id = self.ml_logger.log_signal(
                                symbol=symbol,
                                signal=signal,
                                ltf_df=ltf_df,
                                htf_df=htf_df,
                                m1_df=m1_df,
                                filters_passed=True,
                                filter_reason=filter_reason,
                                trade_executed=True,
                                skip_reason=""
                            )

                            # Open position
                            if self.open_position(symbol, signal, signal_id):
                                logger.info(f"[{symbol}] Position opened successfully")
                            else:
                                logger.warning(f"[{symbol}] Failed to open position")
                        else:
                            # Filters not passed - wait for better entry
                            logger.info(f"[{symbol}] WAITING - Filters not passed:")
                            logger.info(f"[{symbol}]   {filter_reason}")

                            # LOG SKIPPED SIGNAL FOR ML (filters failed)
                            self.ml_logger.log_signal(
                                symbol=symbol,
                                signal=signal,
                                ltf_df=ltf_df,
                                htf_df=htf_df,
                                m1_df=m1_df,
                                filters_passed=False,
                                filter_reason=filter_reason,
                                trade_executed=False,
                                skip_reason="Smart filters not passed"
                            )
                    else:
                        # Show signal status when not strong enough
                        trend_str = signal.trend.value if hasattr(signal, 'trend') else "?"
                        logger.info(f"[{symbol}] Watching | Trend: {trend_str} | Signal: {signal.action} ({signal.confluence_score}/4) - waiting for stronger confluence")

                        # LOG WEAK SIGNAL FOR ML (confluence too low)
                        self.ml_logger.log_signal(
                            symbol=symbol,
                            signal=signal,
                            ltf_df=ltf_df,
                            htf_df=htf_df,
                            m1_df=m1_df,
                            filters_passed=False,
                            filter_reason="N/A",
                            trade_executed=False,
                            skip_reason=f"Low confluence ({signal.confluence_score}/4)"
                        )
                else:
                    # No signal - diagnostics already shown by analyze_symbol(verbose=True)
                    pass

            except Exception as e:
                logger.error(f"[{symbol}] Cycle error: {e}")
                import traceback
                traceback.print_exc()

        # Get account info for balance and realized PnL
        try:
            account = self.client.get_account_info()
            total_balance = float(account.get('totalWalletBalance', 0))
            available_balance = float(account.get('availableBalance', 0))

            # Calculate total unrealized PnL across all positions
            total_unrealized_pnl = 0.0
            positions_data = self.client.get_positions()
            for pos in positions_data:
                # get_positions() returns 'unrealized_pnl' (processed field name)
                unrealized = float(pos.get('unrealized_pnl', 0))
                total_unrealized_pnl += unrealized

            logger.info(f"┌─ ACCOUNT SUMMARY ──────────────────────────────")
            logger.info(f"│ Balance: ${total_balance:,.2f} | Available: ${available_balance:,.2f}")
            logger.info(f"│ Total Unrealized PnL: ${total_unrealized_pnl:+,.2f}")
            logger.info(f"│ Session Realized PnL: ${self.pnl_today:+,.2f}")
            logger.info(f"└────────────────────────────────────────────────")
        except Exception as e:
            logger.warning(f"Could not fetch account info: {e}")

        # Log daily stats with per-symbol breakdown
        win_rate = (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0
        logger.info(f"┌─ SESSION STATS ─────────────────────────────────")
        logger.info(f"│ Total: {self.trades_today} trades | {self.wins_today}W/{self.losses_today}L | WR: {win_rate:.0f}%")
        for symbol in self.symbols:
            stats = self.symbol_stats[symbol]
            sym_trades = stats["wins"] + stats["losses"]
            sym_wr = (stats["wins"] / sym_trades * 100) if sym_trades > 0 else 0
            logger.info(f"│ {symbol}: {stats['wins']}W/{stats['losses']}L | WR: {sym_wr:.0f}% | PnL: ${stats['pnl']:+,.2f}")
        logger.info(f"└────────────────────────────────────────────────")

    def run(self):
        """Main trading loop."""
        if not self.initialize():
            logger.error("Initialization failed. Exiting.")
            return

        self.running = True
        logger.info("=" * 60)
        logger.info("LIVE TRADING STARTED")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        try:
            while self.running:
                self.run_cycle()

                # Wait for next cycle
                logger.info(f"Next check in {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nStopping trading engine...")
            self.running = False

        self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        logger.info("=" * 60)
        logger.info("SHUTDOWN")
        logger.info("=" * 60)

        # Log final stats
        logger.info(f"Total trades today: {self.trades_today}")
        logger.info(f"Wins: {self.wins_today} | Losses: {self.losses_today}")
        if self.trades_today > 0:
            logger.info(f"Win rate: {self.wins_today/self.trades_today*100:.1f}%")
        logger.info(f"P&L today: ${self.pnl_today:+,.2f}")

        # List open positions
        if self.positions:
            logger.info("\nOpen positions (will remain open):")
            for symbol, pos in self.positions.items():
                logger.info(f"  {symbol}: {pos['side']} @ ${pos['entry_price']:,.4f}")

        logger.info("=" * 60)
        logger.info("Engine stopped.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HTF Confluence Live Trading Engine")
    parser.add_argument("--symbols", nargs="+", default=["DOTUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"],
                        help="Symbols to trade")
    parser.add_argument("--live", action="store_true",
                        help="Use live mainnet (default: demo/testnet)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Total capital to use (default: account balance)")
    parser.add_argument("--risk", type=float, default=0.02,
                        help="Risk per trade (default: 0.02 = 2%%)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds (default: 60)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip confirmation prompt for live trading")

    args = parser.parse_args()

    # Create engine
    engine = HTFConfluenceLiveEngine(
        symbols=args.symbols,
        config=MODERATE_CONFIG,
        testnet=not args.live,
        total_capital=args.capital,
        risk_per_trade=args.risk
    )

    engine.check_interval = args.interval

    # Safety confirmation for live trading
    if args.live and not args.no_confirm:
        print("\n" + "=" * 60)
        print("WARNING: LIVE MAINNET TRADING MODE")
        print("=" * 60)
        print("You are about to trade with REAL MONEY!")
        print(f"Symbols: {', '.join(args.symbols)}")
        print(f"Leverage: {engine.leverage}x")
        print("Asset-specific TP/SL settings:")
        for symbol in engine.symbols:
            cfg = engine.symbol_configs[symbol]
            print(f"  {symbol}: TP {cfg['tp_roi']*100:.0f}% / SL {cfg['sl_roi']*100:.0f}% ROI")
        print("=" * 60)

        confirm = input("\nType 'CONFIRM' to start live trading: ")
        if confirm != "CONFIRM":
            print("Live trading cancelled.")
            return

    # Run engine
    engine.run()


if __name__ == "__main__":
    main()
