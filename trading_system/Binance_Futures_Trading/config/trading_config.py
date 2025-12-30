"""
Binance Futures Trading Configuration
=====================================
Pure Momentum Strategy (NO ML)
Based on Forex Master Momentum System
"""

import os

# =============================================================================
# 1. BINANCE API CONFIGURATION
# =============================================================================
BINANCE_CONFIG = {
    # Demo API Keys (from demo.binance.com) - Futures Enabled
    "api_key": os.getenv("BINANCE_DEMO_API_KEY", "7fCaP1ndvOzyEdDZd7DEjVRNIrQ8Jm8kTgID9CCV3IQ9VqGaSfQV1KjpvOiVZ4vI"),
    "api_secret": os.getenv("BINANCE_DEMO_API_SECRET", "ld4YmedQ5VgIFLIzyhDgZCykmwQTjsMjKtZl0T3RooFIzF6lBRfExxEeN8CZCJGw"),

    # LIVE API Keys (from binance.com) - REAL MONEY!
    "live_api_key": os.getenv("BINANCE_LIVE_API_KEY", "28VMu8a8Ho9GIlPSZZQnAyAPOMU2WRncPDOWdgNXORunOVMbB4ilsWtJbX8NwwGP"),
    "live_api_secret": os.getenv("BINANCE_LIVE_API_SECRET", "bl9qn0mq99BJfeH0MwrRRCdyu8OpXMclVsVUbtmOxY48SzNS1DvmPaXMq4853e7R"),

    # Use demo mode
    "testnet": True,
    "demo_mode": True,

    # Binance Futures URLs
    "futures_demo_url": "https://demo-fapi.binance.com",               # Demo futures API (demo.binance.com)
    "futures_testnet_url": "https://testnet.binancefuture.com",        # Old testnet
    "futures_mainnet_url": "https://fapi.binance.com",                 # Production

    # Demo trading URL (web interface)
    "demo_url": "https://demo.binance.com/en/futures",

    # Fees (Binance Futures)
    "maker_fee": 0.0002,   # 0.02% maker
    "taker_fee": 0.0004,   # 0.04% taker
}

# =============================================================================
# 2. TRADING SYMBOLS - OPTIMIZED FOR SCALPING
# =============================================================================
# LIVE MODE: Top performers with Enhanced Boost Mode
FUTURES_SYMBOLS_LIVE = [
    "DOTUSDT",   # Polkadot - Best performer: +93.1% in 90-day backtest
    "AVAXUSDT",  # Avalanche - Stable: -2.1% during -43% crash
    "BTCUSDT",   # Bitcoin - With tighter settings: +40.6% in 90-day backtest
]

# DEMO MODE: All symbols for testing
# Tier 1: Highest Volume & Liquidity (Best for scalping)
FUTURES_SYMBOLS_TIER1 = [
    "BTCUSDT",   # Bitcoin - Deepest liquidity, $30B+ daily volume
    "ETHUSDT",   # Ethereum - $29B+ daily volume, predictable volatility
]

# Tier 2: High Volume Altcoins (Great for momentum scalping)
FUTURES_SYMBOLS_TIER2 = [
    "SOLUSDT",   # Solana - High volatility, great for breakouts
    "BNBUSDT",   # Binance Coin - Top-notch exchange liquidity
    # REMOVED: XRPUSDT, DOGEUSDT - poor backtest performance
]

# Tier 3: Volatile Altcoins (Higher risk, higher reward)
FUTURES_SYMBOLS_TIER3 = [
    "ADAUSDT",   # Cardano - Good liquidity, moderate volatility
    "AVAXUSDT",  # Avalanche - Strong volatility, DeFi catalyst
    "DOTUSDT",   # Polkadot - Parachain events
    "LTCUSDT",   # Litecoin - BTC correlation, stable liquidity
    # REMOVED: LINKUSDT, POLUSDT - poor backtest performance
]

# Active trading symbols - ALL TIERS for DEMO mode
FUTURES_SYMBOLS_DEMO = FUTURES_SYMBOLS_TIER1 + FUTURES_SYMBOLS_TIER2 + FUTURES_SYMBOLS_TIER3

# Default symbols (used by imports) - DEMO symbols
# Live trading engine will override with FUTURES_SYMBOLS_LIVE
FUTURES_SYMBOLS = FUTURES_SYMBOLS_DEMO

# Symbol-specific settings (pip value adjustments + VOLATILITY MULTIPLIER for DCA)
# dca_volatility_mult: Multiplier for DCA trigger levels
#   1.0 = Default DCA triggers (BTC/ETH - stable)
#   1.5 = 50% wider DCA triggers (moderately volatile altcoins)
#   2.0 = 100% wider DCA triggers (highly volatile altcoins)
# Example: If base DCA L1 is -20% ROI, with 1.5x mult it becomes -30% ROI
#
# ENHANCED BOOST SETTINGS (symbol-specific):
#   - tp_roi: Take profit ROI (lower for less volatile coins)
#   - boost_trigger_dca: DCA level to trigger boost (earlier for less volatile)
#   - dca_levels: Custom DCA trigger levels (tighter for less volatile)
SYMBOL_SETTINGS = {
    # Tier 1: BTC/ETH - Less volatile, need TIGHTER parameters
    "BTCUSDT": {
        "min_qty": 0.001,
        "min_notional": 5.0,          # Binance min notional $5 for BTC
        "price_precision": 2,
        "qty_precision": 3,
        "tick_size": 0.01,
        "dca_volatility_mult": 1.0,   # Default - BTC is stable
        # ENHANCED BOOST - Tighter settings for BTC
        "tp_roi": 0.05,               # 5% TP (vs 8% default) - faster exits
        "boost_trigger_dca": 2,       # Trigger boost at DCA 2 (vs 3 default) - earlier
        "dca_levels": [               # Tighter DCA triggers for BTC
            {"trigger_roi": -0.03, "multiplier": 1.50, "tp_roi": 0.04},   # L1: -3% ROI
            {"trigger_roi": -0.10, "multiplier": 1.75, "tp_roi": 0.04},   # L2: -10% ROI
            {"trigger_roi": -0.18, "multiplier": 2.00, "tp_roi": 0.03},   # L3: -18% ROI
            {"trigger_roi": -0.25, "multiplier": 2.25, "tp_roi": 0.03},   # L4: -25% ROI
        ],
    },
    "ETHUSDT": {
        "min_qty": 0.01,
        "price_precision": 2,
        "qty_precision": 3,
        "tick_size": 0.01,
        "dca_volatility_mult": 1.0,   # Default - ETH is stable
        # ENHANCED BOOST - Same as DOT (which works well, 93% return)
        "tp_roi": 0.08,               # 8% TP (same as DOT) - proven to work
        "boost_trigger_dca": 3,       # Trigger boost at DCA 3 (same as DOT)
        # Uses default DCA levels from DCA_CONFIG (same as DOT)
    },
    # Tier 2: Moderate volatility
    "BNBUSDT": {
        "min_qty": 0.01,
        "price_precision": 2,
        "qty_precision": 2,
        "tick_size": 0.01,
        "dca_volatility_mult": 1.2,   # 20% wider DCA triggers
    },
    "SOLUSDT": {
        "min_qty": 0.1,
        "price_precision": 3,
        "qty_precision": 1,
        "tick_size": 0.001,
        "dca_volatility_mult": 1.5,   # 50% wider - SOL is volatile
    },
    "XRPUSDT": {
        "min_qty": 1.0,
        "price_precision": 4,
        "qty_precision": 1,
        "tick_size": 0.0001,
        "dca_volatility_mult": 1.3,   # 30% wider
    },
    "DOGEUSDT": {
        "min_qty": 1.0,
        "price_precision": 5,
        "qty_precision": 0,
        "tick_size": 0.00001,
        "dca_volatility_mult": 1.8,   # 80% wider - DOGE is very volatile
    },
    # Tier 3: High volatility altcoins - WIDER DCA triggers
    "ADAUSDT": {
        "min_qty": 1.0,
        "price_precision": 4,
        "qty_precision": 0,
        "tick_size": 0.0001,
        "dca_volatility_mult": 1.5,   # 50% wider
    },
    "AVAXUSDT": {
        "min_qty": 1,
        "price_precision": 4,
        "qty_precision": 0,           # Integer only - Binance requires stepSize=1
        "tick_size": 0.001,
        "dca_volatility_mult": 1.6,   # 60% wider - AVAX is volatile
        # ENHANCED BOOST - Default settings (AVAX is volatile, works well)
        "tp_roi": 0.08,               # 8% TP (default) - works well for AVAX
        "boost_trigger_dca": 3,       # Trigger boost at DCA 3 (default)
        # Uses default DCA levels from DCA_CONFIG
    },
    "LINKUSDT": {
        "min_qty": 0.1,
        "price_precision": 3,
        "qty_precision": 1,
        "tick_size": 0.001,
        "dca_volatility_mult": 1.4,   # 40% wider
    },
    "POLUSDT": {
        "min_qty": 1.0,
        "price_precision": 4,
        "qty_precision": 0,
        "tick_size": 0.0001,
        "dca_volatility_mult": 1.5,   # 50% wider
    },
    "DOTUSDT": {
        "min_qty": 0.1,
        "price_precision": 3,
        "qty_precision": 1,
        "tick_size": 0.001,
        "dca_volatility_mult": 1.2,   # 20% wider - tightened for faster DCA recovery
        # ENHANCED BOOST - Default settings (DOT is volatile, works well)
        "tp_roi": 0.08,               # 8% TP (default) - works well for DOT
        "boost_trigger_dca": 3,       # Trigger boost at DCA 3 (default)
        # Uses default DCA levels from DCA_CONFIG
    },
    "LTCUSDT": {
        "min_qty": 0.01,
        "price_precision": 2,
        "qty_precision": 3,
        "tick_size": 0.01,
        "dca_volatility_mult": 1.2,   # 20% wider - LTC follows BTC
    },
}

# =============================================================================
# 3. MASTER MOMENTUM SETTINGS (NO ML)
# =============================================================================
MOMENTUM_CONFIG = {
    # Core momentum detection
    "enabled": True,
    "momentum_period": 3,           # Look back 3 candles for momentum
    "momentum_threshold": 0.08,     # 0.08% move required (matching Forex system)

    # Trend detection (EMA)
    "ema_fast_period": 8,           # Fast EMA
    "ema_slow_period": 21,          # Slow EMA

    # RSI filter (avoid overbought/oversold)
    "rsi_period": 14,
    "rsi_max_for_buy": 70.0,        # Max RSI to allow BUY
    "rsi_min_for_sell": 30.0,       # Min RSI to allow SELL

    # ADX trend strength
    "adx_period": 14,
    "min_adx": 15.0,                # Minimum ADX for trend confirmation

    # Cooldown between signals
    "cooldown_bars": 5,             # Bars to wait between signals
    "cooldown_seconds": 300,        # 5 min between trades
}

# =============================================================================
# 4. STRATEGY CONFIGURATION
# =============================================================================
STRATEGY_CONFIG = {
    # Trade management
    "max_trades_per_day": 50,       # Max trades per day (increased for 12 symbols)
    "daily_profit_target": 0.03,    # 3% daily target

    # Take Profit / Stop Loss
    "take_profit_pct": 0.02,        # 2% take profit
    "stop_loss_pct": 0.01,          # 1% stop loss

    # Trailing stop
    "trailing_stop_enabled": True,
    "trailing_stop_trigger": 0.008, # Activate at 0.8% profit
    "trailing_stop_distance": 0.005,# Trail 0.5% behind peak

    # Leverage & Margin Settings
    "leverage": 20,                  # Fixed 20x leverage for all symbols
    "margin_type": "ISOLATED",       # ISOLATED margin (risk contained per position)
}

# =============================================================================
# 5. DCA (Dollar Cost Averaging) SETTINGS
# =============================================================================
DCA_CONFIG = {
    "enabled": True,

    # ROI-BASED TP/SL (For 20x Leverage Scalping)
    # Formula: price_move = roi / leverage
    # IMPORTANT: With 20x leverage, liquidation happens at ~91% ROI loss (~4.5% price move)
    #
    # FEES CALCULATION (per round trip):
    # - Taker fee: 0.04% x 2 = 0.08% of position value
    # - With 20x leverage: 0.08% * 20 = 1.6% ROI just to cover fees!
    # - Need at least 5% ROI to make meaningful profit after fees
    #
    # DCA STRATEGY: Wide SL to let DCA recover positions
    # TP: 8% ROI = 0.4% price move (smaller target, faster exits)
    # SL: 80% ROI = 4.0% price move (wide - trust DCA to recover)
    "take_profit_roi": 0.08,        # 8% ROI target (= 0.4% price move with 20x) - SMALLER TP
    "stop_loss_roi": 0.90,          # 90% ROI loss (= 4.5% price move) - VERY WIDE for all DCA levels
    "liquidation_buffer_pct": 0.01, # 1% buffer from liquidation price for safety

    "trailing_stop_pct": 0.004,     # 0.4% price trailing (= 8% ROI with 20x)
    "position_divisor": 4,          # Initial size = 25% of normal

    # DCA Levels: ROI-BASED triggers (for 20x leverage)
    # L1: Reduced trigger (-5% ROI instead of -10%) for faster averaging
    # L2-L4: Keep original triggers
    "levels": [
        {"trigger_roi": -0.05, "multiplier": 1.50, "tp_roi": 0.06},   # Level 1: -5% ROI (0.25% price drop) - REDUCED
        {"trigger_roi": -0.20, "multiplier": 1.75, "tp_roi": 0.06},   # Level 2: -20% ROI (1% price drop)
        {"trigger_roi": -0.30, "multiplier": 2.00, "tp_roi": 0.05},   # Level 3: -30% ROI (1.5% price drop)
        {"trigger_roi": -0.40, "multiplier": 2.25, "tp_roi": 0.04},   # Level 4: -40% ROI (2% price drop)
    ],
    # Note: L1 triggers faster at -5% ROI, all DCA TPs also smaller

    "max_exposure_multiplier": 4.00,  # Max 4x normal position with all DCAs
    "sl_after_dca_roi": 0.90,         # 90% ROI SL from avg entry after DCA (same as initial - WIDE)

    # ==========================================================================
    # TRAILING TAKE PROFIT (Software-based - we monitor and close manually)
    # ==========================================================================
    # When position reaches activation_roi, we start trailing
    # We track peak ROI and close when it drops by trail_distance
    # Example: 82% ROI peak, 15% trail = close at 67% ROI
    "trailing_tp": {
        "enabled": True,
        "activation_roi": 0.20,       # Activate trailing at 20% ROI profit
        "trail_distance_roi": 0.15,   # Trail 15% ROI behind peak
        "min_profit_roi": 0.10,       # Minimum 10% ROI profit to close (don't give back all profits)
    },

    # ==========================================================================
    # STALE POSITION EXIT (Close positions held too long once profitable)
    # ==========================================================================
    # If position held > max_hold_hours, close immediately at min_exit_roi
    # Avoids holding positions forever waiting for full TP
    "stale_exit": {
        "enabled": True,
        "max_hold_hours": 4,          # Position held > 4 hours = stale
        "min_exit_roi": 0.10,         # Close at 10% ROI profit
    },

    # ==========================================================================
    # HEDGE MODE - SIMULTANEOUS LONG + SHORT POSITIONS
    # ==========================================================================
    # Strategy: Hold BOTH LONG and SHORT positions on the same symbol
    # - Enter BOTH directions on startup
    # - Each side has independent DCA, TP, SL, Trailing
    # - When one side hits TP, re-enter that side immediately
    # - Profit from BOTH directions (whichever moves in your favor)
    # - Budget split: 50% LONG, 50% SHORT per symbol
    "hedge_mode": {
        "enabled": True,                    # ENABLE HEDGE MODE (both LONG + SHORT)
        "budget_split": 0.5,                # 50% of symbol budget to each side
        "reenter_on_tp": True,              # Re-enter same side after TP hit
        "independent_dca": True,            # Each side has independent DCA levels
        "independent_trailing": True,       # Each side has independent trailing TP
        "max_dca_per_side": 4,              # Max 4 DCA levels per side (8 total possible)
    },

    # ==========================================================================
    # HYBRID HOLD + TRADE SYSTEM (used when hedge_mode is disabled)
    # ==========================================================================
    # Strategy: Always in the market, following the trend
    # - Auto-enter on startup (detect trend, enter immediately)
    # - On TP: Close and re-enter same direction (take profit, stay in trend)
    # - On Trend Change: Close position, flip to opposite direction
    # - DCA: Hybrid filters (easy for L1-2, strict for L3-4)
    "hybrid_hold": {
        "enabled": False,                 # DISABLED when hedge_mode is enabled
        "auto_enter_on_start": True,      # Auto-detect trend and enter on startup
        "reenter_on_tp": True,            # After TP hit, re-enter same direction
        "flip_on_trend_change": False,    # Don't close on trend change, ride it out with DCA

        # STRATEGY: Stay in position until TP or SL
        # - Don't flip on trend change (too many premature exits)
        # - Trust DCA to recover adverse moves
        # - Only follow trend for NEW entries after position closes naturally

        # HEDGE SETTINGS (disabled since we don't flip anymore)
        "hedge_on_dca_flip": False,          # Disabled - no flipping
        "hedge_min_dca_level": 1,            # (unused)
        "hedge_tighten_sl_roi": 0.15,        # (unused)
        "hedge_breakeven_buffer": 0.005,     # (unused)
        "hedge_max_wait_hours": 4,           # (unused)
    },

    # ==========================================================================
    # TREND DETECTION SYSTEM (STRICT - Avoid Flip-Flopping)
    # ==========================================================================
    # Uses EMA crossover + Price position + RSI + ADX for trend determination
    # Requires MULTIPLE confirmations to avoid false signals
    "trend_detection": {
        # EMA Settings (use slower EMAs for stability)
        "ema_fast": 12,               # Fast EMA period (was 8, now 12 for stability)
        "ema_slow": 26,               # Slow EMA period (was 21, now 26 for stability)

        # RSI Thresholds (wider neutral zone to avoid flip-flop)
        "bullish_rsi_min": 50,        # RSI must be above 50 for bullish (was 45)
        "bearish_rsi_max": 50,        # RSI must be below 50 for bearish (was 55)

        # Confirmation Requirements (STRICT)
        "confirmation_candles": 5,    # Need 5 candles confirming trend (was 2)
        "ema_separation_pct": 0.002,  # EMAs must be 0.2% apart (avoid crossover noise)

        # Trend Change Cooldown
        "flip_cooldown_minutes": 30,  # Minimum 30 min between trend flips

        # ADX Filter (trend strength)
        "use_adx_filter": True,       # Require ADX confirmation
        "adx_min_trend": 20,          # ADX must be > 20 to confirm trend

        # Volume confirmation
        "use_volume_confirm": True,   # Require volume confirmation
        "volume_multiplier": 1.5,     # Volume must be 1.5x average (was 1.2)
    },

    # ==========================================================================
    # HYBRID DCA FILTERS
    # ==========================================================================
    # DCA 1-2: Easy filters (faster averaging, accept mild adverse momentum)
    # DCA 3-4: Strict filters (require clear reversal signals)
    "hybrid_dca_filters": {
        "easy_levels": [1, 2],        # DCA levels with easy filters
        "strict_levels": [3, 4],      # DCA levels with strict filters

        # Easy filter settings (DCA 1-2)
        "easy_momentum_threshold": 0.3,      # Accept DCA if momentum weakening by 30%
        "easy_require_reversal": False,      # Don't require reversal candle

        # Strict filter settings (DCA 3-4)
        "strict_momentum_threshold": 0.5,    # Require momentum weakening by 50%
        "strict_require_reversal": True,     # Require reversal candle pattern
        "strict_require_rsi_extreme": True,  # Require RSI at extreme (oversold/overbought)
        "strict_rsi_oversold": 25,           # RSI below this for long DCA
        "strict_rsi_overbought": 75,         # RSI above this for short DCA
    },
}

# =============================================================================
# 6. RISK MANAGEMENT - DYNAMIC ALLOCATION (PROGRESSIVE DCA)
# =============================================================================
RISK_CONFIG = {
    # Dynamic Fund Allocation
    # On start: balance / num_symbols = budget per symbol
    # Each symbol's budget covers: initial entry + ALL 3 DCA levels
    "dynamic_allocation": True,       # Enable dynamic fund allocation
    "allocation_buffer_pct": 0.05,    # 5% buffer for fees/safety

    # Per-Symbol Budget Distribution (PROGRESSIVE - start small, increase with each DCA)
    # Initial entry + 4 DCA levels share this budget
    # Strategy: Start small, add MORE as price drops to better levels
    "initial_entry_pct": 0.10,        # 10% of symbol budget for initial entry (smallest)
    "dca1_pct": 0.15,                 # 15% for DCA level 1
    "dca2_pct": 0.20,                 # 20% for DCA level 2
    "dca3_pct": 0.25,                 # 25% for DCA level 3
    "dca4_pct": 0.30,                 # 30% for DCA level 4 (largest - best price)
    # Total: 0.10 + 0.15 + 0.20 + 0.25 + 0.30 = 1.00 (100%)

    # Safety Limits
    "max_daily_loss_pct": 0.10,       # Stop trading at -10% daily loss
    "max_total_positions": 12,        # Max concurrent positions (all symbols)
    "max_drawdown_pct": 0.10,         # Max 10% drawdown from peak
}

# =============================================================================
# 7. TIMEFRAMES
# =============================================================================
TIMEFRAMES = {
    "entry": "1m",           # Entry signal generation
    "price_action": "5m",    # Pattern recognition
    "htf_short": "15m",      # Higher timeframe filter 1
    "htf_long": "1h",        # Higher timeframe filter 2
}

# Candle counts per timeframe
DATA_CONFIG = {
    "candles_1m": 500,       # 1-minute candles
    "candles_5m": 100,       # 5-minute candles
    "candles_15m": 50,       # 15-minute candles
    "candles_1h": 50,        # 1-hour candles
}

# =============================================================================
# 8. PAPER TRADING CONFIGURATION
# =============================================================================
PAPER_TRADING_CONFIG = {
    "initial_balance": 10000.0,      # Start with $10,000 USDT
    "commission_per_trade": 0.0004,  # 0.04% taker fee
    "slippage_pct": 0.0002,          # 0.02% slippage estimate
    "check_interval": 30,            # Check every 30 seconds
    "symbol_check_interval": 60,     # Check each symbol every 60 seconds
}

# =============================================================================
# 9. BACKTEST CONFIGURATION
# =============================================================================
BACKTEST_CONFIG = {
    "initial_balance": 10000.0,
    "commission_per_trade": 0.0004,
    "slippage_pct": 0.0002,
    "default_days": 30,              # Default backtest period
}

# =============================================================================
# 10. LOGGING CONFIGURATION
# =============================================================================
LOGGING_CONFIG = {
    "log_dir": "logs",
    "trade_log_file": "binance_trades.json",
    "performance_log_file": "binance_performance.json",
    "console_output": True,
    "file_output": True,
}
