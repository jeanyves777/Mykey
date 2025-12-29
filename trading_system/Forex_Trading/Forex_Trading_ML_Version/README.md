# Forex ML Trading System

## Overview

A production-ready ML-powered Forex trading system using a 5-model ensemble for signal generation with OANDA broker integration. The system fetches **LIVE market data** and **LIVE position data** from OANDA on every iteration - no caching.

## Key Features

- **5-Model ML Ensemble**: RandomForest, XGBoost, LightGBM, CatBoost, Neural Network
- **3-Class Classification**: SELL (0), HOLD (1), BUY (2)
- **Live OANDA Integration**: Real-time market data, positions, and order execution
- **229 Technical Features**: Including lagged indicators for temporal patterns
- **Risk Management**: Position sizing, SL/TP, daily loss limits, cooldowns

---

## Architecture

```
Forex_Trading_ML_Version/
├── config/
│   └── trading_config.py      # All configuration settings
├── data/
│   ├── oanda_client.py        # OANDA API wrapper
│   └── data_loader.py         # Historical data fetching
├── features/
│   └── feature_engineer.py    # 229 technical features
├── ensemble/
│   └── ensemble_voting.py     # 5-model ensemble with soft voting
├── risk_management/
│   ├── risk_manager.py        # Position sizing, daily limits
│   ├── position_manager.py    # Track open positions
│   └── trade_executor.py      # OANDA order execution
├── engine/
│   ├── trading_strategy.py    # ML signal generation
│   └── paper_trading_engine.py # Main trading loop
├── saved_models/
│   └── ensemble/              # Trained model files
└── run_paper_trading.py       # Entry point
```

---

## Configuration (trading_config.py)

### OANDA Settings
```python
@dataclass
class OandaConfig:
    api_key: str = ""           # From environment: OANDA_API_KEY
    account_id: str = ""        # From environment: OANDA_ACCOUNT_ID
    environment: str = "practice"  # "practice" or "live"
```

### ML Settings
```python
@dataclass
class MLConfig:
    confidence_threshold: float = 0.60   # Minimum 60% confidence to trade
    min_model_agreement: int = 3         # At least 3/5 models must agree
    voting_method: str = "soft"          # "soft" (probability) or "hard" (majority)
```

### Risk Settings
```python
@dataclass
class RiskConfig:
    position_size_pct: float = 0.02      # 2% of equity per trade
    max_daily_loss_pct: float = 0.05     # Stop trading at 5% daily loss
    max_trades_per_day: int = 20
    cooldown_seconds: int = 300          # 5 min between trades
    stop_loss_pips: float = 20.0
    take_profit_pips: float = 30.0
    use_trailing_stop: bool = False
```

### Symbols
```python
symbols: List[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"
]
```

---

## Core Components

### 1. Paper Trading Engine (paper_trading_engine.py)

**Main Loop:**
```python
def _trading_iteration(self):
    """Execute one trading iteration - ML SCANNING IS PRIORITY."""

    # 1. Fetch LIVE equity from OANDA
    account_info = self.client.get_account_summary()

    # 2. Fetch LIVE positions from OANDA (cache for this iteration only)
    success, self._live_oanda_trades = self.trade_executor.get_open_trades()

    # 3. ML SCANNING - Process each symbol
    for symbol in self.symbols:
        self._process_symbol(symbol)

    # 4. Update positions with LIVE OANDA data
    self._update_positions()
```

**Symbol Processing:**
```python
def _process_symbol(self, symbol: str):
    # ALWAYS fetch fresh market data - NO CACHING
    df = self.data_loader.get_latest_bars(symbol, count=500, granularity='M5')

    # Check position from LIVE OANDA trades
    has_position = any(t.get('instrument') == symbol for t in self._live_oanda_trades)

    if has_position:
        # Check for ML exit signal
        should_close, reason = self.strategy.should_close_position(symbol, direction, df)
        if should_close:
            self._close_position(symbol, reason)
    else:
        # ML SCANNING: Generate signal
        signal, confidence, agreement, reason = self.strategy.generate_signal(symbol, df)

        if signal:  # BUY or SELL
            if self.risk_manager.can_trade():
                self._open_position(symbol, signal, df, confidence, reason)
```

**Position Display (LIVE OANDA Data):**
```python
def _print_compact_status(self):
    # Show LIVE positions from OANDA with TP/SL distances
    for trade in oanda_trades:
        # Get TP/SL from OANDA trade object
        stop_loss = float(trade['stopLossOrder']['price'])
        take_profit = float(trade['takeProfitOrder']['price'])

        # Get LIVE current price
        prices = self.client.get_pricing([symbol])
        current_price = prices[symbol]['bid']  # or 'ask' for SELL

        # Calculate pip distances
        tp_pips = (take_profit - current_price) / pip_value
        sl_pips = (current_price - stop_loss) / pip_value

        print(f"  {symbol}: {direction} {units} @ {entry_price}")
        print(f"    Current: {current_price} | P&L: {pnl_pips} pips (${unrealized_pnl})")
        print(f"    TP: {take_profit} ({tp_pips} pips away) | SL: {stop_loss} ({sl_pips} pips away)")
```

### 2. ML Strategy (trading_strategy.py)

**Signal Generation:**
```python
class MLTradingStrategy:
    # Signal constants
    SELL = 0
    HOLD = 1
    BUY = 2

    def generate_signal(self, symbol: str, df: pd.DataFrame):
        # 1. Engineer features (229 features)
        featured_df = self.feature_engineer.engineer_features(df.copy(), fit=False)
        X = self.feature_engineer.get_feature_matrix(featured_df)

        # 2. Get ensemble prediction
        prediction, confidence, agreement, details = self.ensemble.predict_single(X[-1])

        # 3. Check thresholds
        if not self.ensemble.should_trade(confidence, agreement):
            return None, confidence, agreement, "Below threshold"

        # 4. Convert to signal
        if prediction == self.BUY:
            return 'BUY', confidence, agreement, reason
        elif prediction == self.SELL:
            return 'SELL', confidence, agreement, reason
        else:
            return None, confidence, agreement, "HOLD signal"
```

### 3. Ensemble Voting (ensemble_voting.py)

**Soft Voting with Binary/3-Class Support:**
```python
def predict_with_confidence(self, X):
    all_probas = {}
    all_predictions = {}

    for name, model in self.models.items():
        all_probas[name] = model.predict_proba(X)
        raw_preds = model.predict(X)

        # Map predictions to signal format
        n_classes = all_probas[name].shape[1]
        if n_classes == 2:
            # Binary: 0=SELL -> -1, 1=BUY -> 1
            all_predictions[name] = np.where(raw_preds == 0, -1, 1)
        else:
            # 3-class: already -1, 0, 1
            all_predictions[name] = raw_preds

    # Weighted average probabilities
    weights = self.weight_manager.get_weights()
    weighted_proba = sum(weights[name] * proba for name, proba in all_probas.items())
    weighted_proba /= sum(weights.values())

    # Final prediction
    raw_indices = np.argmax(weighted_proba, axis=1)
    confidences = np.max(weighted_proba, axis=1)

    if n_classes == 2:
        predictions = np.where(raw_indices == 0, -1, 1)  # SELL=-1, BUY=1
    else:
        predictions = raw_indices - 1  # 0,1,2 -> -1,0,1

    return predictions, confidences, agreement_details
```

### 4. Position Sync (LIVE OANDA Data)

**Auto-Remove Closed Positions:**
```python
def _update_positions(self):
    oanda_trades = self._live_oanda_trades  # Already fetched this iteration

    # Build map of currently open trades on OANDA
    oanda_trade_map = {t['instrument']: t for t in oanda_trades}

    # Remove locally tracked positions that are closed on OANDA
    for symbol in list(self.position_manager.positions.keys()):
        if symbol not in oanda_trade_map:
            print(f"[SYNC] {symbol}: Position closed on OANDA - removing")
            del self.position_manager.positions[symbol]

    # Update remaining positions with fresh OANDA data
    for symbol, position in self.position_manager.positions.items():
        if symbol in oanda_trade_map:
            trade = oanda_trade_map[symbol]
            position.unrealized_pnl = float(trade['unrealizedPL'])
            position.units = abs(float(trade['currentUnits']))
            position.entry_price = float(trade['price'])
```

---

## Output Format

Every iteration displays:

```
============================================================
[14:02:16] ML SCANNING 8 PAIRS | Equity: $4,529.14 | Open: 1
============================================================
  EUR_USD: ML=HOLD Conf=48% Agree=3/5 @ 1.17529 (Below threshold)
  GBP_USD: ML=HOLD Conf=56% Agree=5/5 @ 1.34620 (Below threshold)
  USD_JPY: ML=HOLD Conf=43% Agree=4/5 @ 156.94900 (Below threshold)
  USD_CHF: ML=HOLD Conf=44% Agree=4/5 @ 0.79206 (Below threshold)
  AUD_USD: ML=HOLD Conf=46% Agree=2/5 @ 0.66539 (Below threshold)
  USD_CAD: ML=HOLD Conf=49% Agree=5/5 @ 1.37434 (Below threshold)
  NZD_USD: POSITION OPEN (BUY) - Checking for exit signal...
  NZD_USD: Hold position (no exit signal)
  EUR_GBP: ML=HOLD Conf=54% Agree=5/5 @ 0.87306 (Below threshold)

--- SESSION STATUS (LIVE OANDA DATA) ---
OPEN POSITIONS (1) | Unrealized P&L: $-1.17
  NZD_USD: BUY 4512 @ 0.57945
    Current: 0.57919 | P&L: -2.6 pips ($-1.17)
    TP: 0.58025 (+10.6 pips away) | SL: 0.57745 (17.4 pips away)
```

**When ML Signal Detected:**
```
  USD_JPY: ML=BUY Conf=67% Agree=4/5 @ 149.234
  USD_JPY: >>> EXECUTING BUY TRADE <<<
```

**When Trade Blocked:**
```
  USD_CAD: ML=SELL Conf=61% Agree=3/5 @ 1.35678
  USD_CAD: Trade BLOCKED - Max positions reached
```

---

## Key Design Principles

### 1. LIVE Data Only - No Caching
```python
# WRONG - Using cached data
if current_bar_time <= self.last_bar_time[symbol]:
    return  # Skip

# CORRECT - Always fetch fresh
df = self.data_loader.get_latest_bars(symbol, count=500)
self.data_buffer[symbol] = df  # Store for ML features only
```

### 2. Position Check from OANDA, Not Local Cache
```python
# WRONG - Only checking local tracking
has_position = self.position_manager.has_position(symbol)

# CORRECT - Check LIVE OANDA trades first
oanda_trades = self._live_oanda_trades
has_position = any(t.get('instrument') == symbol for t in oanda_trades)
```

### 3. Binary Classification Mapping
```python
# Models trained for binary (0=SELL, 1=BUY)
# Must map to signals (-1=SELL, 0=HOLD, 1=BUY)

if n_classes == 2:
    predictions = np.where(raw_indices == 0, -1, 1)  # 0->-1, 1->1
else:
    predictions = raw_indices - 1  # 0,1,2 -> -1,0,1
```

### 4. One API Call Per Resource Per Iteration
```python
def _trading_iteration(self):
    # Fetch ONCE at start of iteration
    account_info = self.client.get_account_summary()
    success, self._live_oanda_trades = self.trade_executor.get_open_trades()

    # Use cached data throughout iteration
    for symbol in self.symbols:
        self._process_symbol(symbol)  # Uses self._live_oanda_trades

    self._update_positions()  # Uses self._live_oanda_trades
    self._print_compact_status()  # Uses self._live_oanda_trades
```

---

## Running the System

```bash
# Set environment variables
export OANDA_API_KEY="your-api-key"
export OANDA_ACCOUNT_ID="your-account-id"

# Run paper trading
python -m trading_system.Forex_Trading.Forex_Trading_ML_Version.run_paper_trading
```

---

## Files Modified for LIVE Data

1. **paper_trading_engine.py**
   - `_trading_iteration()`: Fetch OANDA data ONCE per iteration
   - `_process_symbol()`: Always fetch fresh market data, check OANDA positions
   - `_update_positions()`: Auto-remove closed positions, update from OANDA
   - `_print_compact_status()`: Display LIVE TP/SL with pip distances
   - `_sync_existing_positions()`: Clear stale positions before sync

2. **ensemble_voting.py**
   - `predict_with_confidence()`: Map binary to signal format
   - `_soft_vote()`: Handle 2-class and 3-class models
   - `_hard_vote()`: Same binary mapping
   - `get_trade_signal()`: Map individual predictions for logging

---

## Common Issues & Fixes

### Issue: "Old positions still showing"
**Fix**: Clear local positions before sync, check OANDA trades in `_process_symbol()`

### Issue: "ML shows HOLD when all models vote BUY"
**Fix**: Binary classification mapping - 0=SELL->-1, 1=BUY->1

### Issue: "Agreement shows 0/5 when 4 models agree"
**Fix**: Map individual predictions before storing in `all_predictions`

### Issue: "Position display not showing TP/SL"
**Fix**: Get TP/SL from OANDA trade object: `trade['stopLossOrder']['price']`
