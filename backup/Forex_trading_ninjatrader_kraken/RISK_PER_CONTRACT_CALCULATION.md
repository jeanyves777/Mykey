# Risk Per Contract - EXACT Calculation for $25K Account

## What Does "1 Contract" Risk Mean?

When you trade **1 contract**, your risk = Stop Loss distance × Tick Value

---

## All 5 Pairs - Risk Per 1 Contract

### 1. EUR/USD → M6E (Micro Euro)

**Your Settings:**
- Stop Loss: 16 pips
- Tick Size: 0.00005 (half-pip)
- 1 pip = 2 ticks
- Tick Value: $6.25 per tick

**Risk Calculation:**
```
SL in ticks = 16 pips × 2 = 32 ticks
Risk per contract = 32 ticks × $6.25 = $200
Risk as % of $25K = $200 / $25,000 = 0.8%
```

**✓ 1 M6E contract = $200 risk (0.8% of $25K)**

---

### 2. GBP/USD → M6B (Micro British Pound)

**Your Settings:**
- Stop Loss: 25 pips
- Tick Size: 0.0001 (1 pip)
- 1 pip = 1 tick
- Tick Value: $6.25 per tick

**Risk Calculation:**
```
SL in ticks = 25 pips × 1 = 25 ticks
Risk per contract = 25 ticks × $6.25 = $156.25
Risk as % of $25K = $156.25 / $25,000 = 0.625%
```

**✓ 1 M6B contract = $156.25 risk (0.625% of $25K)**

---

### 3. USD/JPY → MJY (Micro Japanese Yen)

**Your Settings:**
- Stop Loss: 15 pips
- Tick Size: 0.0000005
- 1 pip (0.01) = 100 ticks
- Tick Value: $6.25 per tick

**Risk Calculation:**
```
SL in ticks = 15 pips × 100 = 1,500 ticks
Risk per contract = 1,500 ticks × $6.25 = $9,375
```

**❌ WAIT - THIS IS WRONG!**

Let me recalculate MJY correctly:

**MJY Contract Specs (CORRECT):**
- Contract Size: ¥1,250,000 (NOT ¥12,500,000)
- Tick Size: 0.000001 ($1.25 per tick for MJY)
- 1 pip = 0.01 in JPY pairs
- For MJY: 1 pip = 10 ticks (not 100!)

**CORRECTED Calculation:**
```
SL in ticks = 15 pips × 10 = 150 ticks
Tick Value for MJY = $1.25 (not $6.25)
Risk per contract = 150 ticks × $1.25 = $187.50
Risk as % of $25K = $187.50 / $25,000 = 0.75%
```

**✓ 1 MJY contract = $187.50 risk (0.75% of $25K)**

---

### 4. USD/CAD → MCD (Micro Canadian Dollar)

**Your Settings:**
- Stop Loss: 16 pips
- Tick Size: 0.00005 (half-pip)
- 1 pip = 2 ticks
- Tick Value: $5.00 per tick (CAD is $5, not $6.25)

**Risk Calculation:**
```
SL in ticks = 16 pips × 2 = 32 ticks
Risk per contract = 32 ticks × $5.00 = $160
Risk as % of $25K = $160 / $25,000 = 0.64%
```

**✓ 1 MCD contract = $160 risk (0.64% of $25K)**

---

### 5. USD/CHF → MSF (Micro Swiss Franc)

**Your Settings:**
- Stop Loss: 12 pips
- Tick Size: 0.00005 (half-pip)
- 1 pip = 2 ticks
- Tick Value: $6.25 per tick

**Risk Calculation:**
```
SL in ticks = 12 pips × 2 = 24 ticks
Risk per contract = 24 ticks × $6.25 = $150
Risk as % of $25K = $150 / $25,000 = 0.6%
```

**✓ 1 MSF contract = $150 risk (0.6% of $25K)**

---

## Summary Table: Risk Per 1 Contract on $25K Account

| Pair | Symbol | SL (pips) | SL (ticks) | Tick Value | **Risk per Contract** | **% of $25K** |
|------|--------|-----------|------------|------------|---------------------|---------------|
| **EUR/USD** | M6E | 16 | 32 | $6.25 | **$200.00** | 0.80% |
| **GBP/USD** | M6B | 25 | 25 | $6.25 | **$156.25** | 0.63% |
| **USD/JPY** | MJY | 15 | 150 | $1.25 | **$187.50** | 0.75% |
| **USD/CAD** | MCD | 16 | 32 | $5.00 | **$160.00** | 0.64% |
| **USD/CHF** | MSF | 12 | 24 | $6.25 | **$150.00** | 0.60% |

---

## What This Means for FundedNext

### Max Loss Limit: $1,000 (4% of $25K)

**Scenario 1: All 5 positions open at once (1 contract each)**
```
Total risk = $200 + $156 + $187 + $160 + $150 = $853
% of account = $853 / $25,000 = 3.4%
```
✓ **Under the $1,000 max loss limit** (with $147 buffer)

**Scenario 2: Worst case - all 5 hit stop loss**
```
Account balance = $25,000 - $853 = $24,147
Still above $24,000 ✓
```

**Scenario 3: Can you trade 2 contracts per position?**
```
2 contracts × $853 = $1,706 total risk
```
❌ **Would EXCEED $1,000 max loss limit**

**Recommendation: Start with 1 contract per trade**

---

## Position Sizing Options for $25K Account

### Option 1: Conservative (1 contract per trade)
- **Risk per trade:** $150-200 (0.6-0.8%)
- **Max concurrent:** 5 positions
- **Total risk:** $853 (3.4%)
- **Safety buffer:** $147 to max loss limit
- **✓ RECOMMENDED for Challenge Phase**

### Option 2: Moderate (2 contracts on best pairs only)
- **M6E (EUR/USD):** 2 contracts = $400 risk
- **MCD (USD/CAD):** 2 contracts = $320 risk
- **Others:** 1 contract each = $493 risk
- **Total risk:** $1,213 (4.9%)
- **❌ EXCEEDS max loss limit - NOT SAFE**

### Option 3: Aggressive (Scale after proving profitable)
- **Start with 1 contract per trade**
- **After 2 weeks profitable:** Scale to 2 contracts on EUR/USD only
- **M6E:** 2 contracts = $400 risk
- **Others:** 1 contract each = $653 risk
- **Total risk:** $1,053 (4.2%)
- **⚠️ Slightly over limit, but manageable if careful**

---

## Daily Risk Analysis

### Daily Loss Limit: -$200 (self-imposed)

**How many losing trades = -$200?**

| Pair | Risk/Contract | Losses Needed for -$200 |
|------|---------------|-------------------------|
| M6E | $200 | **1 loss** |
| M6B | $156 | 1.3 losses (2 losses = -$312) |
| MJY | $187 | 1.1 losses (2 losses = -$375) |
| MCD | $160 | 1.25 losses (2 losses = -$320) |
| MSF | $150 | 1.3 losses (2 losses = -$300) |

**Average:** ~1-2 losing trades = hit daily loss limit

**With 48% win rate (52% loss rate):**
- 10 trades/day × 52% = ~5 losers expected
- 5 losers × $170 avg = **-$850 daily loss** (WITHOUT STOP)

**⚠️ WARNING: You MUST enforce -$200 daily stop or you'll blow account!**

---

## Weekly Risk Analysis

**Week 1 Example (10 trades/day × 5 days = 50 trades):**

**At 48% WR:**
- Winners: 24 trades
- Losers: 26 trades

**Profit Calculation (M6E example):**
- Winners: 24 × $250 (20 pip TP) = +$6,000
- Losers: 26 × $200 (16 pip SL) = -$5,200
- **Net: +$800** ✓ On track to $1,250 target

**But if you have a bad week (40% WR):**
- Winners: 20 trades × $250 = +$5,000
- Losers: 30 trades × $200 = -$6,000
- **Net: -$1,000** ❌ ACCOUNT BLOWN

**Protection: Daily -$200 stop prevents weekly blowup**
- Max weekly loss with daily stops: 5 days × -$200 = **-$1,000**
- Exactly at max loss limit (you'd stop trading)

---

## Profit Potential Per Contract

### Take Profit Values (1 contract):

| Pair | Symbol | TP (pips) | TP (ticks) | Tick Value | **Profit per Winner** |
|------|--------|-----------|------------|------------|--------------------|
| **EUR/USD** | M6E | 20 | 40 | $6.25 | **$250.00** |
| **GBP/USD** | M6B | 30 | 30 | $6.25 | **$187.50** |
| **USD/JPY** | MJY | 18 | 180 | $1.25 | **$225.00** |
| **USD/CAD** | MCD | 20 | 40 | $5.00 | **$200.00** |
| **USD/CHF** | MSF | 15 | 30 | $6.25 | **$187.50** |

**Average profit per winner: ~$210**
**Average loss per loser: ~$171**
**Net per trade at 48% WR:** (0.48 × $210) - (0.52 × $171) = $100.80 - $88.92 = **+$11.88/trade**

---

## Path to $1,250 Target

**Using average $11.88 per trade:**
```
Trades needed = $1,250 / $11.88 = 105 trades
Days needed = 105 trades / 10 trades per day = 10.5 days
Timeline = 2-3 weeks (with buffer)
```

**Best case (EUR/USD only, 80% WR):**
```
50 trades: 40 winners × $250 = $10,000
           10 losers × $200 = -$2,000
           Net = +$8,000 (pass in 1 week!)
```

**Worst case (40% WR across all pairs):**
```
50 trades: 20 winners × $210 = $4,200
           30 losers × $171 = -$5,130
           Net = -$930 (close to max loss!)
```

**With daily -$200 stops, worst case prevented:**
```
Day 1: -$200 (stop)
Day 2: -$200 (stop)
Day 3: -$200 (stop)
Day 4: -$200 (stop)
Day 5: -$200 (stop)
Total: -$1,000 (hit max loss, account protection triggered)
```

---

## FINAL RECOMMENDATION

### For $25K FundedNext Challenge:

**✓ Position Size: 1 contract per trade**
- M6E: $200 risk (0.8%)
- M6B: $156 risk (0.6%)
- MJY: $187 risk (0.75%)
- MCD: $160 risk (0.64%)
- MSF: $150 risk (0.6%)

**✓ Max Concurrent: 5 positions (1 per pair)**
- Total risk: $853 (3.4% of account)
- Safety buffer: $147 to max loss

**✓ Daily Loss Limit: -$200 (HARD STOP)**
- Prevents weekly blowup
- Protects from hitting $1,000 max loss too quickly

**✓ Daily Profit Cap: +$400**
- Ensures consistency rule compliance
- No single day >40% of total profit

**✓ Expected Timeline:**
- 105 trades at $11.88 avg = $1,250 target
- 10-12 trading days
- 2-3 weeks with buffer

**This gives you the BEST chance to pass the FundedNext $25K challenge while protecting against catastrophic loss.**

---

## Answer to Your Question:

**"1 contract is equal to how much risk on that 25K?"**

**ANSWER:**
- **M6E (EUR/USD): $200 (0.8%)**
- **M6B (GBP/USD): $156 (0.63%)**
- **MJY (USD/JPY): $187 (0.75%)**
- **MCD (USD/CAD): $160 (0.64%)**
- **MSF (USD/CHF): $150 (0.6%)**

**All 5 positions open = $853 total risk (3.4% of $25K)**

**This is SAFE for FundedNext challenge (under $1,000 max loss limit).**
