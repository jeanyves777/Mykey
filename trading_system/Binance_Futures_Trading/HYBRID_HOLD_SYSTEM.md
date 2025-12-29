# Hybrid Hold + Trade System
## Binance Futures Trading Engine - Internal Documentation

---

## Overview

The **Hybrid Hold + Trade System** is a trading strategy that ensures we are ALWAYS in the market following the detected trend. Instead of waiting for entry signals, the system:

1. **Auto-enters on startup** based on trend detection
2. **Re-enters after TP** in the same direction (stay with the trend)
3. **Flips position on trend change** (close and enter opposite direction)
4. Uses **Hybrid DCA filters** (easy for L1-2, strict for L3-4)
5. **NEW: Hedge on DCA Flip** - For DCA positions, uses hedge strategy instead of closing at loss

---

## System Architecture

### Configuration Location
- **Config File**: `config/trading_config.py`
- **Engine File**: `engine/live_trading_engine.py`

### Key Configuration Blocks

```python
# In DCA_CONFIG:
"hybrid_hold": {
    "enabled": True,                # Master switch for hybrid system
    "auto_enter_on_start": True,    # Enter positions on startup
    "reenter_on_tp": True,          # Re-enter after taking profit
    "flip_on_trend_change": True,   # Close and flip when trend reverses

    # HEDGE ON DCA FLIP (minimize loss on DCA positions)
    "hedge_on_dca_flip": True,      # Use hedge strategy for DCA positions
    "hedge_min_dca_level": 1,       # Min DCA level to use hedge (1 = any DCA)
    "hedge_tighten_sl_roi": 0.15,   # Tighten SL to -15% ROI
    "hedge_breakeven_buffer": 0.005, # Exit at breakeven + 0.5%
    "hedge_max_wait_hours": 4,      # Max hours to wait for breakeven
},

"trend_detection": {
    "ema_fast": 8,                  # Fast EMA period
    "ema_slow": 21,                 # Slow EMA period
    "bullish_rsi_min": 45,          # Min RSI for bullish confirmation
    "bearish_rsi_max": 55,          # Max RSI for bearish confirmation
    "confirmation_candles": 2,      # Candles above/below EMA required
    "use_volume_confirm": True,     # Require volume confirmation
    "volume_multiplier": 1.2,       # Volume must be 1.2x average
},

"hybrid_dca_filters": {
    # Easy levels (L1-2): Allow DCA with basic momentum check
    "easy_levels": [1, 2],
    "easy_momentum_threshold": 0.3,
    "easy_require_reversal": False,

    # Strict levels (L3-4): Require strong reversal signals
    "strict_levels": [3, 4],
    "strict_momentum_threshold": 0.5,
    "strict_require_reversal": True,
    "strict_require_rsi_extreme": True,
    "strict_rsi_oversold": 25,
    "strict_rsi_overbought": 75,
},
```

---

## Core Methods

### 1. `detect_trend(symbol, df)` -> str
**Location**: `live_trading_engine.py:352`

**Purpose**: Detect current market trend for a symbol

**Returns**: `"BULLISH"`, `"BEARISH"`, or `"NEUTRAL"`

**Logic**:
1. Calculate EMA 8 (fast) and EMA 21 (slow)
2. Calculate RSI 14
3. **BULLISH** if:
   - EMA fast > EMA slow (uptrend)
   - Price > EMA slow (above trend)
   - RSI >= 45 (not oversold)
   - Optionally: Volume > 1.2x average
4. **BEARISH** if:
   - EMA fast < EMA slow (downtrend)
   - Price < EMA slow (below trend)
   - RSI <= 55 (not overbought)
   - Optionally: Volume > 1.2x average
5. **NEUTRAL** otherwise

---

### 2. `auto_enter_on_trend(symbol)` -> bool
**Location**: `live_trading_engine.py:459`

**Purpose**: Called on startup to enter positions for symbols without existing positions

**Workflow**:
1. Fetch 1-minute kline data for symbol
2. Call `detect_trend()` to determine market direction
3. If trend is BULLISH -> Enter LONG position
4. If trend is BEARISH -> Enter SHORT position
5. If NEUTRAL -> Wait (do not enter)

**Called From**: `run()` method at startup

---

### 3. `check_trend_change(symbol, position)` -> bool
**Location**: `live_trading_engine.py:420`

**Purpose**: Check if the trend has reversed while in a position

**Returns**: `True` if trend changed and flip should occur

**Logic**:
1. Throttle checks to every 60 seconds to avoid noise
2. Call `detect_trend()` to get current trend
3. Compare with position side:
   - LONG position + BEARISH trend = TREND CHANGE
   - SHORT position + BULLISH trend = TREND CHANGE
4. Update `self.current_trend[symbol]` for flip direction

**Called From**: `manage_positions()` at start of each position check

---

### 4. `close_and_flip(symbol, position, new_trend)`
**Location**: `live_trading_engine.py:528`

**Purpose**: Close current position and enter the opposite direction (for NON-DCA positions)

**Workflow**:
1. Close current position with market order
2. Cancel all pending SL/TP orders
3. Log the trade with exit type "TREND_FLIP"
4. Update daily stats
5. Wait 2 seconds for settlement
6. Determine new side (BULLISH -> LONG, BEARISH -> SHORT)
7. Enter new position with fresh SL/TP orders

**Called From**: `manage_positions()` when trend change detected AND position has NO DCA

---

### 4b. `hedge_on_trend_flip(symbol, position, new_trend)` - NEW
**Location**: `live_trading_engine.py:622`

**Purpose**: Hedge strategy for DCA positions when trend flips (minimize loss)

**Why Hedge?**: When you have a DCA position (L1-L4), closing immediately on trend flip could result in a large loss. Instead, we:
- Keep the losing position open with tight SL
- Open a hedge (opposite direction) to follow the new trend
- Wait for the losing position to recover to breakeven
- Then close the loser and keep the hedge running

**Workflow**:
1. Cancel all existing SL/TP orders
2. Place TIGHT SL on losing position (-15% ROI instead of -80%)
3. Place BREAKEVEN TP on losing position (entry + 0.5% buffer)
4. Mark position as hedged and waiting for breakeven
5. Open HEDGE position in new trend direction (initial size ~$2)
6. Wait for breakeven TP to hit OR max wait time (4 hours)

**Example**:
```
BEFORE: DOTUSDT LONG @ $1.90, DCA L2, margin $6.50, losing -30% ROI
TREND FLIPS TO BEARISH

AFTER:
- DOTUSDT LONG: SL @ $1.87 (-15% ROI), TP @ $1.91 (breakeven)
- DOTUSDT SHORT [HEDGE]: Entry @ $1.85, margin $2 (initial size)

OUTCOME 1: Price goes to $1.91 → LONG exits at breakeven, SHORT keeps running
OUTCOME 2: Price drops, SL hits @ $1.87 → LONG exits at -15% ROI, SHORT profits
OUTCOME 3: 4 hours pass → LONG market-closed at current price, SHORT continues
```

**Called From**: `manage_positions()` when trend change detected AND position HAS DCA

---

### 4c. `manage_hedged_position(symbol, position, current_price)` - NEW
**Location**: `live_trading_engine.py:738`

**Purpose**: Manage hedged positions - check for timeout and force close if needed

**Called From**: `manage_positions()` for positions with `is_hedged=True`

---

### 5. `reenter_after_tp(symbol, closed_side)`
**Location**: `live_trading_engine.py:613`

**Purpose**: Re-enter position after TP is hit to stay in the trend

**Workflow**:
1. Wait 3 seconds after TP hit
2. Call `detect_trend()` to confirm trend still valid
3. If trend confirms same direction -> Re-enter same side
4. If trend is now opposite -> Enter new direction
5. If trend is NEUTRAL -> Do not re-enter

**Called From**: `sync_positions()` when TP exit is detected

---

### 6. `check_hybrid_dca_filter(symbol, position, dca_level, df)` -> tuple
**Location**: `live_trading_engine.py:662`

**Purpose**: Apply easy or strict filter based on DCA level

**Returns**: `(can_dca: bool, reason: str)`

**Logic**:
- **Easy Levels (L1-2)**:
  - Check momentum weakening (RSI leaving extreme zone)
  - No reversal candle required
  - Purpose: Allow early averaging down

- **Strict Levels (L3-4)**:
  - Require RSI at extreme (< 25 for LONG, > 75 for SHORT)
  - Require reversal candle pattern
  - Require momentum threshold met
  - Purpose: Only add when very high probability reversal

**Called From**: `check_dca()` when DCA trigger level is reached

---

## Execution Flow

### Startup Flow
```
run()
  |-> sync_existing_positions()    # Load any existing Binance positions
  |-> [HYBRID] auto_enter_on_trend()  # For each symbol without position:
      |-> detect_trend()
      |-> enter_position() if trend is not NEUTRAL
```

### Main Loop Flow
```
while running:
  |-> check_daily_reset()          # Reset daily stats at midnight
  |-> check_entry_signals()        # Check for new entry signals
  |-> manage_positions()           # For each position:
      |-> [HYBRID] check_trend_change()
          |-> close_and_flip() if trend reversed
      |-> _check_stale_exit()      # Close stale profitable positions
      |-> _check_trailing_tp()     # Trail and close on pullback
      |-> _check_auto_close_tp()   # Market close if past TP
      |-> check_dca()              # Check DCA levels
          |-> check_hybrid_dca_filter()  # Easy or strict filter
```

### Position Close Flow
```
sync_positions()  # Called from manage_positions()
  |-> Detect closed position (from Binance)
  |-> Get realized PNL from Binance
  |-> Determine exit type (TP/SL)
  |-> Log trade
  |-> [HYBRID] reenter_after_tp() if TP hit
      |-> detect_trend()
      |-> enter_position() if trend valid
```

---

## Key Behaviors

### 1. Always In Market
- System enters on startup if no position exists
- After TP hit, immediately re-enters same direction
- After trend flip, immediately enters opposite direction
- Only time NOT in market: during order execution or NEUTRAL trend

### 2. Trend Following
- Trend detection uses EMA crossover + price position + RSI
- Conservative thresholds to avoid whipsaws (RSI 45-55 neutral zone)
- Volume confirmation optional but recommended

### 3. Smart DCA
- Early DCA levels (L1-2) use easy filters for faster averaging
- Late DCA levels (L3-4) use strict filters requiring reversal signals
- Prevents adding to losing positions without clear reversal

### 4. Risk Management
- 20x leverage with isolated margin
- ROI-based SL/TP (not price-based)
- Liquidation protection buffer (1% above liquidation price)
- Daily loss limit (10% of starting balance)

---

## Configuration Reference

### Hybrid Hold Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `True` | Enable hybrid hold system |
| `auto_enter_on_start` | `True` | Enter positions on startup |
| `reenter_on_tp` | `True` | Re-enter after TP hit |
| `flip_on_trend_change` | `True` | Close and flip on trend reversal |

### Trend Detection Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `ema_fast` | `8` | Fast EMA period |
| `ema_slow` | `21` | Slow EMA period |
| `bullish_rsi_min` | `45` | Min RSI for bullish |
| `bearish_rsi_max` | `55` | Max RSI for bearish |
| `confirmation_candles` | `2` | Candles above/below EMA |
| `use_volume_confirm` | `True` | Require volume confirmation |
| `volume_multiplier` | `1.2` | Volume threshold |

### Hybrid DCA Filter Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `easy_levels` | `[1, 2]` | Levels using easy filter |
| `easy_momentum_threshold` | `0.3` | Min momentum for easy |
| `easy_require_reversal` | `False` | Require reversal candle |
| `strict_levels` | `[3, 4]` | Levels using strict filter |
| `strict_momentum_threshold` | `0.5` | Min momentum for strict |
| `strict_require_reversal` | `True` | Require reversal candle |
| `strict_require_rsi_extreme` | `True` | Require RSI extreme |
| `strict_rsi_oversold` | `25` | RSI oversold threshold |
| `strict_rsi_overbought` | `75` | RSI overbought threshold |

---

## Logging

The system logs all hybrid actions with appropriate levels:

- `[TRADE]` - Position entries, exits, flips
- `[DCA]` - DCA filter decisions (allowed/blocked)
- `[INFO]` - Trend detection results

Example log output:
```
[2024-12-28 10:00:00] [LIVE] [TRADE] HYBRID MODE: Auto-entering positions based on trend detection...
[2024-12-28 10:00:01] [LIVE] [TRADE] HYBRID AUTO-ENTER: DOTUSDT trend is BULLISH - entering LONG
[2024-12-28 10:15:00] [LIVE] [TRADE] HYBRID TREND FLIP: DOTUSDT trend changed to BEARISH
[2024-12-28 10:15:02] [LIVE] [TRADE] HYBRID FLIPPED: DOTUSDT LONG -> SHORT @ $6.85
[2024-12-28 11:30:00] [LIVE] [TRADE] HYBRID: TP hit for DOTUSDT, scheduling re-entry SHORT
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2024-12-28 | 1.0 | Initial Hybrid Hold + Trade System implementation |

---

*Last Updated: 2024-12-28*
