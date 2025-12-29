# Multi-Timeframe Momentum Strategy - Decision Flow

## Complete Trade Entry Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   NEW TRADING OPPORTUNITY                        │
│                  (Check every 60 seconds)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: MARKET TIMING CHECK                                    │
│  ✓ Forex market open (24/5)                                     │
│  ✓ Preferred: London/NY overlap (13:00-17:00 UTC)               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: POSITION & RISK CHECK                                  │
│  ✓ No existing position for this instrument                     │
│  ✓ Total positions < 3                                          │
│  ✓ Trades today < 3                                             │
│  ✓ Daily P&L < +2% (profit target)                              │
│  ✓ Daily P&L > -3% (loss limit)                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: TECHNICAL SCORING (1-MIN BARS)                         │
│  📊 Analyzes 17+ indicators:                                    │
│     • EMA Stack (9, 20, 50)                                      │
│     • RSI (14)                                                   │
│     • MACD + Histogram                                           │
│     • Bollinger Bands                                            │
│     • ATR                                                        │
│     • Price Momentum                                             │
│  ➜ Output: BULLISH / BEARISH / NEUTRAL + confidence             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: PRICE ACTION ANALYSIS (5-MIN BARS)                     │
│  📈 Pattern recognition:                                        │
│     • Candle color patterns (green vs red)                       │
│     • Higher highs / Lower lows                                  │
│     • 5-bar moving average                                       │
│     • Momentum direction                                         │
│  ➜ Output: BULLISH / BEARISH / NEUTRAL + strength               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: REAL-TIME MOMENTUM (LAST 5 1-MIN BARS) ⭐ HIGHEST!    │
│  🔥 Most recent price action (2x WEIGHT):                       │
│     • Green vs red count (4/5 = STRONG)                          │
│     • 5-bar price change                                         │
│     • Last bar direction                                         │
│     • Higher highs / Lower lows trend                            │
│  ➜ Output: BULLISH / BEARISH / NEUTRAL + momentum               │
│  ➜ Special: STRONG override possible                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: V3 WEIGHTED DECISION                                   │
│  ⚖️  Combines all 3 methods:                                    │
│     • Technical:      1x weight                                  │
│     • Price Action:   1x weight                                  │
│     • Momentum:       2x weight (most important!)                │
│                                                                  │
│  Decision Logic:                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ✅ All 3 agree         → HIGH CONFIDENCE trade           │   │
│  │ ✅ Strong momentum     → APPROVED (override)             │   │
│  │ ✅ Momentum + 1 other  → APPROVED                        │   │
│  │ ✅ Weighted score ≥2   → MEDIUM CONFIDENCE               │   │
│  │ ❌ No consensus        → SKIP TRADE                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ➜ Output: BUY / SELL / SKIP                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │  SKIP?  │
                    └────┬────┘
                         │ No (BUY or SELL)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: HTF TREND FILTER (30-MIN + 1-HOUR) ⭐ STRICT          │
│  🔍 Analyzes both timeframes:                                   │
│     30-MIN:  Price vs EMA9, HH/LL, Candle colors                │
│     1-HOUR:  Price vs EMA9, HH/LL, Candle colors                │
│                                                                  │
│  HTF Trend Determination:                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ BULLISH:  Both 30-min AND 1-hour bullish                │   │
│  │ BEARISH:  Both 30-min AND 1-hour bearish                │   │
│  │ NEUTRAL:  Conflicting timeframes                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Filter Decision:                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ For BUY signal:                                          │   │
│  │   ✅ APPROVED if HTF = BULLISH                           │   │
│  │   🛑 BLOCKED if HTF = BEARISH                            │   │
│  │   🛑 BLOCKED if HTF = NEUTRAL (conflicting) ⭐ STRICT    │   │
│  │                                                          │   │
│  │ For SELL signal:                                         │   │
│  │   ✅ APPROVED if HTF = BEARISH                           │   │
│  │   🛑 BLOCKED if HTF = BULLISH                            │   │
│  │   🛑 BLOCKED if HTF = NEUTRAL (conflicting) ⭐ STRICT    │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────┴─────┐
                    │ BLOCKED? │
                    └────┬─────┘
                         │ No (APPROVED)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: PULLBACK DETECTION (5-MIN HTF) ⭐ QUALITY FILTER      │
│  ⏳ Wait for better entry timing:                              │
│                                                                  │
│  For BULLISH (BUY) Trades:                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ✓ Wait for dip (3+ red 5-min candles)                   │   │
│  │ ✓ Then recovery signal (green 5-min candle)             │   │
│  │ ✓ RSI < 65 (not overbought)                             │   │
│  │ ✓ Price pullback from recent high                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  For BEARISH (SELL) Trades:                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ✓ Wait for bounce (3+ green 5-min candles)              │   │
│  │ ✓ Then rejection signal (red 5-min candle)              │   │
│  │ ✓ RSI > 35 (not oversold)                               │   │
│  │ ✓ Price bounce from recent low                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Status Messages:                                                │
│    ⏳ "WAITING FOR BETTER ENTRY..." (not ready)                 │
│    ✅ "PULLBACK CONDITIONS MET - Entering" (ready)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    ┌────┴─────┐
                    │  READY?  │
                    └────┬─────┘
                         │ Yes
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9: POSITION SIZING                                        │
│  💰 Calculate trade size:                                       │
│     • Target: 10% of account balance                             │
│     • Units = (Balance × 0.10) / Current Price                   │
│     • Round to nearest 1,000 (micro lot)                         │
│     • Minimum: 1,000 units (1 micro lot)                         │
│     • Maximum: 20% of account                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 10: CALCULATE STOP LOSS & TAKE PROFIT                     │
│  🎯 Risk/Reward setup:                                          │
│                                                                  │
│  For BUY (LONG):                                                 │
│    Stop Loss:    Entry × (1 - 0.01) = -1.0%  (100 pips)         │
│    Take Profit:  Entry × (1 + 0.015) = +1.5% (150 pips)         │
│                                                                  │
│  For SELL (SHORT):                                               │
│    Stop Loss:    Entry × (1 + 0.01) = -1.0%  (100 pips)         │
│    Take Profit:  Entry × (1 - 0.015) = +1.5% (150 pips)         │
│                                                                  │
│  Risk/Reward Ratio: 1:1.5                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 11: ORDER EXECUTION                                       │
│  📝 Execute trade:                                              │
│     1. Apply slippage (0.5 pips)                                 │
│     2. Calculate commission (0.2 pips)                           │
│     3. Submit MARKET order to OANDA                              │
│     4. Attach Stop Loss order                                    │
│     5. Attach Take Profit order                                  │
│     6. Initialize trailing stop tracking                         │
│        - Triggers at +0.6% profit                                │
│        - Trails 0.4% behind high/low                             │
│                                                                  │
│  ✅ TRADE ENTERED!                                              │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  POSITION MONITORING (Every 30 seconds)                         │
│  👁️  Track open position:                                      │
│     • Check if TP hit → Close at profit                          │
│     • Check if SL hit → Close at loss                            │
│     • Update highest/lowest price                                │
│     • Check trailing stop trigger (+0.6%)                        │
│     • If triggered, trail 0.4% behind                            │
│     • Check if trailing stop hit → Close position               │
│                                                                  │
│  Continue until position closed                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Summary of Validation Layers

| Layer | Component | Weight | Can Block? |
|-------|-----------|--------|------------|
| 1 | Market Timing | - | ✅ Yes |
| 2 | Position & Risk Limits | - | ✅ Yes |
| 3 | Technical Indicators | 1x | ⚠️ Partial |
| 4 | Price Action | 1x | ⚠️ Partial |
| 5 | Real-Time Momentum | 2x | ⚠️ Partial |
| 6 | Weighted Decision | - | ✅ Yes |
| 7 | HTF Trend Filter (STRICT) | - | ✅ Yes |
| 8 | Pullback Detection | - | ⏳ Delay |
| 9-11 | Execution | - | - |

**Total Checkpoints**: ~40+ individual conditions across 11 major steps

## Decision Examples

### Example 1: ALL APPROVED - High Confidence Trade

```
Technical:      BULLISH (score: 12/17, confidence: 70%)
Price Action:   BULLISH (4/5 green candles)
Momentum:       BULLISH (STRONG - 4 green bars, +0.4%)
                ↓
Weighted Decision: BUY (All agree = HIGH CONFIDENCE)
                ↓
HTF Filter:     30-min BULLISH, 1-hour BULLISH → APPROVED
                ↓
Pullback:       3 red bars + 1 green recovery → READY
                ↓
✅ ENTER LONG - All checks passed
```

### Example 2: BLOCKED by HTF Filter

```
Technical:      BULLISH
Price Action:   BULLISH
Momentum:       BULLISH (STRONG)
                ↓
Weighted Decision: BUY (All agree = HIGH CONFIDENCE)
                ↓
HTF Filter:     30-min BEARISH, 1-hour BEARISH → BLOCKED
                ↓
🛑 SKIP TRADE - HTF trend is opposite
```

### Example 3: BLOCKED by Conflicting HTF (STRICT Mode)

```
Technical:      BULLISH
Price Action:   NEUTRAL
Momentum:       BULLISH (STRONG override)
                ↓
Weighted Decision: BUY (Strong momentum override)
                ↓
HTF Filter:     30-min BULLISH, 1-hour BEARISH → NEUTRAL
                ↓
🛑 SKIP TRADE - HTF timeframes conflicting (STRICT mode blocks)
```

### Example 4: WAITING for Pullback

```
Technical:      BULLISH
Price Action:   BULLISH
Momentum:       BULLISH (STRONG)
                ↓
Weighted Decision: BUY (All agree = HIGH CONFIDENCE)
                ↓
HTF Filter:     30-min BULLISH, 1-hour BULLISH → APPROVED
                ↓
Pullback:       Only 1 red bar, no recovery yet → NOT READY
                ↓
⏳ WAIT - "WAITING FOR BETTER ENTRY..."
(Will check again in 60 seconds)
```

### Example 5: No Consensus - Skip

```
Technical:      NEUTRAL (score: 0/17)
Price Action:   BULLISH (3 green)
Momentum:       BEARISH (3 red)
                ↓
Weighted Score: (0×1) + (1×1) + (-1×2) = -1
                ↓
Weighted Decision: SKIP (No consensus, score < 2)
                ↓
🛑 SKIP TRADE - Methods disagree
```

## Why This Strategy Works

1. **Multi-Layer Validation** - Trade must pass ALL checkpoints
2. **Momentum Priority** - Recent price action weighted 2x
3. **Strict HTF Filter** - Prevents counter-trend trades
4. **Pullback Timing** - Waits for better entry, not chasing
5. **Risk Management** - Automatic stops protect capital
6. **Quality Over Quantity** - Strict filters = fewer but better trades

This is the same proven logic from your successful MARA options strategy! 🎯
