# FundedNext $25K Futures Challenge - Complete Compliance Plan

## Your Goal: Pass Evaluation → Get Funded → Scale to Profit

---

## FundedNext $25K Account Rules (STRICT COMPLIANCE REQUIRED)

### Challenge Phase Rules:
| Rule | Limit | Your Strategy Impact |
|------|-------|---------------------|
| **Account Size** | $25,000 | Starting balance |
| **Profit Target** | $1,250 (5%) | MUST hit to pass challenge |
| **Max Loss Limit** | $1,000 (4%) | Based on EOD balance - CANNOT EXCEED |
| **Daily Loss Limit** | NO LIMIT | No daily limit in challenge phase ✓ |
| **Max Positions (Challenge)** | 20 Micros | You can trade up to 20 micro contracts |
| **Consistency Rule** | 40% | No single day can be >40% of total profit |
| **Activation Fee** | $79.99 | One-time fee (already paid or budgeted) |

### Funded Account Rules (After Passing):
| Rule | Limit | Notes |
|------|-------|-------|
| **Max Positions (Funded)** | 30 Micros | Increased to 30 after passing |
| **Max Loss Limit** | Still $1,000 EOD | Same 4% trailing max loss |
| **Daily Loss Limit** | Still NONE | No daily limit ✓ |
| **Profit Split** | 80% yours | You keep 80% of profits |

---

## CRITICAL SUCCESS FACTORS

### 1. Profit Target: $1,250 (5% return)
**Path to $1,250:**
- Current OANDA performance: +0.55% per session
- Need: ~9-10 profitable sessions to hit $1,250
- Timeline: 2-3 weeks of consistent trading

**Your Recent Performance:**
- Dec 11 session: +$27.22 on $4,937 account = +0.55%
- If scaled to $25K: +$137.50 per session
- 10 sessions × $137.50 = **$1,375** ✓ Exceeds target

### 2. Max Loss Limit: $1,000 (4% of $25K)
**ABSOLUTE HARD STOP - THIS IS THE KILL SWITCH**

**Protection Strategy:**
- Account drops to $24,000 = STOP ALL TRADING
- Build in safety buffer: Stop at $24,200 (-$800 loss)
- Never risk more than $200/day during challenge

**Your Current Risk Per Trade:**
- OANDA: 15% position size = ~$740 per trade
- **CANNOT USE 15% ON $25K ACCOUNT!**
- $25K × 15% = $3,750 position = WAY TOO HIGH
- One bad trade could lose $500+ = half your max loss

**NEW Position Sizing for FundedNext:**
- **Max risk per trade: $100** (0.4% of $25K)
- **Max concurrent positions: 5** (total risk: $500 = 2% account)
- **Daily loss limit (self-imposed): $200** (0.8% account)

### 3. Consistency Rule: 40% Max Per Day
**What This Means:**
- If total profit = $1,250, no single day can exceed $500
- If you make $600 in one day, you MUST make at least $900 more on other days
- **Strategy: Spread profits across multiple days**

**Your Current Pattern:**
- Dec 11: +$27.22 (single session)
- If scaled to $25K: +$137.50/session
- $137.50 / $1,250 target = 11% of total ✓ Well under 40%

**Protection:**
- Stop trading for the day at +$400 profit
- Resume next day to spread profits

### 4. Max Positions: 20 Micros
**Your Current Setup:**
- Trade 5 forex pairs (EUR/USD, GBP/USD, USD/JPY, USD/CAD, USD/CHF)
- Max 5 concurrent positions
- Max 10 trades/day per pair = 50 total entries/day

**FundedNext Translation:**
- 5 pairs × 2 contracts each = **10 micros max** ✓ Under 20 limit
- Plenty of room to scale

---

## Position Sizing for $25K Account

### Current OANDA Logic (15% margin):
```python
position_size = account_balance * 0.15  # $4,937 × 15% = $740
units = position_size / price           # 740 / 1.05 = ~700 EUR
```

### NEW FundedNext Logic (Risk-Based):
```python
# Risk per trade: 0.4% of account
risk_per_trade = 25000 * 0.004  # $100 max loss per trade

# Calculate contracts based on stop loss
# Example: M6E with 30 tick SL (15 pips)
sl_ticks = 30
tick_value = 6.25
risk_per_contract = sl_ticks * tick_value  # 30 × $6.25 = $187.50

# Max contracts
max_contracts = risk_per_trade / risk_per_contract  # $100 / $187.50 = 0.53
# Round DOWN to: 1 contract (conservative)
```

### Position Sizing Table (All 5 Pairs):

| Pair | Symbol | SL (pips) | SL (ticks) | Tick Value | Risk/Contract | Max Contracts |
|------|--------|-----------|------------|------------|---------------|---------------|
| **EUR/USD** | M6E | 16 | 32 | $6.25 | $200 | **1 contract** |
| **GBP/USD** | M6B | 25 | 25 | $6.25 | $156 | **1 contract** |
| **USD/JPY** | MJY | 15 | 1500 | $6.25 | $93.75 | **1 contract** |
| **USD/CAD** | MCD | 16 | 32 | $5.00 | $160 | **1 contract** |
| **USD/CHF** | MSF | 12 | 24 | $6.25 | $150 | **1 contract** |

**CONSERVATIVE START: 1 contract per pair**

### Why 1 Contract Per Trade?
- M6E SL risk: $200 (16 pips × 2 ticks × $6.25)
- This is 0.8% of $25K account
- **Fits within our $100 target, but gives room for slippage**
- With 5 max concurrent: 5 × $200 = $1,000 total risk = 4% ✓

**After gaining confidence (Week 2+):**
- Scale to 2 contracts on best performers (EUR/USD, USD/CAD)
- Keep 1 contract on others

---

## Daily Trading Limits (FundedNext Compliant)

### Daily Profit Target:
- **Target:** +$125/day (0.5% of $25K)
- **Stop trading at:** +$400/day (consistency rule protection)
- **Monthly target:** $125/day × 20 days = $2,500/month

### Daily Loss Limit (Self-Imposed):
- **Hard stop:** -$200/day (0.8% of account)
- **Warning at:** -$150/day (review trades, reduce size)
- **Max consecutive losses:** 3 trades (stop and review)

### Per-Trade Limits:
- **Max loss per trade:** $200 (worst case with 1 contract)
- **Max concurrent positions:** 5 (1 per pair)
- **Max trades per day per pair:** 10
- **Max total trades per day:** 30 (across all 5 pairs)

### Weekly Limits:
- **Max weekly loss:** -$400 (1.6% of account)
- **If hit:** Reduce to 50% size or stop for week

---

## Strategy Adaptation for FundedNext

### KEEP EXACTLY AS IS:
✓ Entry signals (2 of 3 consensus: RSI, BB, Range)
✓ TP/SL levels (pair-specific settings)
✓ Trailing stop logic
✓ HTF trend filter
✓ Pullback requirement
✓ Max 10 trades/day per pair

### MODIFY FOR FUNDEDNEXT:
❌ **Position size:** Change from 15% margin to 1 contract per trade
❌ **Daily stop:** Add -$200 daily loss limit
❌ **Daily target:** Stop at +$400 profit (consistency rule)
❌ **Account stop:** Stop all trading if account drops to $24,200
❌ **Max concurrent:** Keep at 5 positions (but now 1 contract each)

### NEW Risk Management Code:
```python
# FundedNext Challenge Limits
ACCOUNT_SIZE = 25000
MAX_LOSS_LIMIT = 1000          # -$1,000 EOD balance = FAIL
PROFIT_TARGET = 1250           # +$1,250 = PASS
CONSISTENCY_LIMIT = 0.40       # 40% max per day

# Daily limits
DAILY_PROFIT_TARGET = 125      # Stop at +$400 for consistency
DAILY_PROFIT_STOP = 400        # Consistency rule protection
DAILY_LOSS_LIMIT = 200         # Self-imposed hard stop

# Position limits
CONTRACTS_PER_TRADE = 1        # Start conservative
MAX_CONCURRENT_POSITIONS = 5   # 1 per pair
MAX_DAILY_TRADES_PER_PAIR = 10

# Account protection
STOP_TRADING_BALANCE = 24200   # -$800 loss = stop (before hitting -$1,000)
```

---

## Expected Performance (Based on Current Strategy)

### Your Current OANDA Results:
- **Sample:** 25 trades, 48% WR, 1.27 PF, +$27.22 (+0.55%)
- **Best pairs:** EUR/USD (80% WR), USD/CAD (100% WR), GBP/USD (50% WR)
- **Average trade:** +$1.09 per trade

### Scaled to $25K with 1 Contract Per Trade:

**Profit Per Trade (Conservative Estimate):**
- M6E (EUR/USD): 20 pip TP × 2 ticks × $6.25 = **+$250** (winner)
- M6E (EUR/USD): 16 pip SL × 2 ticks × $6.25 = **-$200** (loser)
- Net per trade at 48% WR: (0.48 × $250) - (0.52 × $200) = $120 - $104 = **+$16/trade**

**Path to $1,250 Target:**
- Need: $1,250 / $16 avg = **78 trades**
- At 10 trades/day (across all 5 pairs): **8 trading days**
- With 48% WR: ~2 weeks to be safe

**Realistic Timeline:**
- **Week 1:** 50 trades, +$800 profit
- **Week 2:** 50 trades, +$800 profit
- **Total:** 100 trades, **+$1,600 profit** ✓ Exceeds $1,250 target

### Best Case (80% WR on EUR/USD only):
- EUR/USD only: 100 trades, 80 winners, 20 losers
- Gross: (80 × $250) - (20 × $200) = $20,000 - $4,000 = **+$16,000**
- Costs: 100 trades × 1 contract × $0.85 × 2 = **-$170**
- **Net: +$15,830** (Would pass in 2 days!)

### Worst Case (40% WR across all pairs):
- 100 trades, 40 winners, 60 losers
- Gross: (40 × $250) - (60 × $200) = $10,000 - $12,000 = **-$2,000**
- **FAIL - hit max loss limit**

**Protection:** With daily -$200 stop, you'd stop after 10 losing trades (5 days max)

---

## Consistency Rule Strategy

### Problem:
- If you make $600 in one day, you need $900+ more to avoid consistency violation
- $600 / total profit must be ≤ 40%
- $600 / X ≤ 0.40 → X ≥ $1,500 total profit needed

### Solution: Daily Profit Cap
**Stop trading for the day when you hit:**
- **+$400 profit** (32% of $1,250 target)
- This ensures no single day exceeds 40% of final profit

**Example Path to $1,250:**
| Day | Trades | Profit | Cumulative | % of Total |
|-----|--------|--------|------------|------------|
| Day 1 | 10 | +$350 | $350 | 28% |
| Day 2 | 8 | +$280 | $630 | 22% (Day 2) |
| Day 3 | 6 | +$200 | $830 | 16% |
| Day 4 | 10 | +$400 | $1,230 | 32% |
| Day 5 | 2 | +$50 | $1,280 | 4% |

All days under 40% ✓

---

## Risk Management Rules (HARD CODED)

### Account-Level Stops:
```python
# Check before EVERY trade
current_balance = get_account_balance()

# KILL SWITCH: Account drops to $24,200
if current_balance <= 24200:
    STOP_ALL_TRADING()
    CLOSE_ALL_POSITIONS()
    EXIT_STRATEGY()

# Daily profit cap (consistency rule)
if daily_profit >= 400:
    STOP_TRADING_FOR_DAY()

# Daily loss limit
if daily_loss <= -200:
    STOP_TRADING_FOR_DAY()
    CLOSE_ALL_POSITIONS()
```

### Trade-Level Stops:
```python
# Max concurrent positions
if open_positions >= 5:
    SKIP_TRADE()

# Max trades per day per pair
if trades_today[pair] >= 10:
    SKIP_TRADE()

# Max contracts per trade
contracts_to_trade = 1  # Start conservative

# Position sizing check
risk_per_trade = contracts_to_trade * sl_ticks * tick_value
if risk_per_trade > 200:
    REDUCE_CONTRACTS()
```

---

## NinjaTrader Sim Backtest Plan

### Phase 1: Historical Backtest (6 Months Data)
**Goal:** Validate strategy translates correctly from Python to NinjaScript

**Steps:**
1. Download 6 months M6E, M6B, MJY, MCD, MSF data
2. Run backtest with FundedNext rules (1 contract, $25K account)
3. Compare to OANDA backtest results

**Success Criteria:**
- Win rate: 45-55% (close to 48% current)
- Profit Factor: 1.2-1.4 (close to 1.27 current)
- Avg trade: +$10-20 per trade
- Max drawdown: <$800 (within $1,000 limit)
- Total profit: $2,000+ in 6 months

### Phase 2: Forward Test (2 Weeks Market Replay)
**Goal:** Validate real-time execution matches backtest

**Steps:**
1. Run NinjaTrader Market Replay with recent data
2. Execute strategy as if live (1 contract per trade)
3. Monitor slippage, commission costs, execution speed

**Success Criteria:**
- Results within 10% of backtest
- No execution errors
- Slippage <1 tick average
- Can execute 10+ trades/day without issues

### Phase 3: Live Sim Account (1 Week)
**Goal:** Final validation before attempting $25K challenge

**Steps:**
1. Connect to live Sim account (NinjaTrader)
2. Trade 1 week with real market data
3. Track daily P&L, consistency, max loss

**Success Criteria:**
- Profitable week (+$200-500)
- No single day >40% of weekly profit
- Max drawdown <$300
- All FundedNext rules respected

---

## Timeline to Pass $25K Challenge

### Week 1: Setup + Backtest
- [ ] Translate Python strategy to NinjaScript C#
- [ ] Configure all 5 pairs (M6E, M6B, MJY, MCD, MSF)
- [ ] Add FundedNext risk management rules
- [ ] Run 6-month historical backtest
- [ ] Validate results match OANDA performance

### Week 2: Market Replay + Sim
- [ ] Run 2-week Market Replay test
- [ ] Monitor execution quality
- [ ] Run 1-week Live Sim test
- [ ] Final adjustments based on Sim results

### Week 3: Apply for FundedNext Challenge
- [ ] Pay $79.99 activation fee
- [ ] Start $25K challenge
- [ ] Trade conservatively (1 contract per trade)
- [ ] Target +$125/day, stop at +$400/day

### Week 4-5: Pass Challenge
- [ ] Accumulate $1,250 profit
- [ ] Ensure consistency rule compliance
- [ ] Maintain account above $24,000
- [ ] Submit for funded account

### Week 6+: Funded Account
- [ ] Receive funded $25K account (30 micro limit)
- [ ] Scale to 2 contracts on best pairs
- [ ] Target $2,000+/month (80% yours = $1,600)

---

## Expected Monthly Income (After Funded)

### Conservative (1 contract per trade):
- 10 trades/day × 20 days = 200 trades/month
- Avg profit: $16/trade × 200 = **$3,200/month gross**
- Costs: 200 × $0.85 × 2 = -$340
- **Net: $2,860/month**
- **Your 80% split: $2,288/month**

### Moderate (2 contracts on best pairs):
- EUR/USD + USD/CAD: 2 contracts
- Others: 1 contract
- Avg profit: $25/trade × 200 = **$5,000/month gross**
- Costs: -$510
- **Net: $4,490/month**
- **Your 80% split: $3,592/month**

### Aggressive (After proven - 3-4 contracts):
- Scale up after 3 months of consistent profits
- **Potential: $10,000+/month** (80% = $8,000)

---

## Final Recommendation

### ✓ APPROVED PLAN:

**Phase 1: NinjaTrader Setup + Backtest (Week 1-2)**
1. Translate forex_scalping.py → NinjaScript C#
2. Configure all 5 pairs with FundedNext risk rules
3. Backtest 6 months + Market Replay validation
4. Target: Match current 48% WR, 1.27 PF performance

**Phase 2: Live Sim Validation (Week 2)**
1. Run 1 week on NinjaTrader Live Sim
2. Ensure all FundedNext rules are enforced
3. Validate +$200-500/week profit potential

**Phase 3: FundedNext $25K Challenge (Week 3-5)**
1. Pay $79.99, start challenge
2. Trade 1 contract per trade (conservative)
3. Target +$125/day, cap at +$400/day
4. Pass in 2-3 weeks with $1,250+ profit

**Phase 4: Funded Account (Week 6+)**
1. Scale to 2 contracts on EUR/USD, USD/CAD
2. Target $3,000-5,000/month
3. Keep 80% = $2,400-4,000/month income

---

## APPROVAL REQUIRED

**Review and approve:**
1. ✓ FundedNext $25K challenge rules understood?
2. ✓ Position sizing (1 contract per trade) acceptable?
3. ✓ Daily limits (+$400 profit cap, -$200 loss limit) acceptable?
4. ✓ Timeline (2 weeks backtest + 2-3 weeks challenge) realistic?
5. ✓ All 5 pairs (M6E, M6B, MJY, MCD, MSF) included?

**Once approved, I will:**
- Create complete NinjaScript C# strategy file
- Configure all 5 pairs with exact settings
- Add FundedNext risk management rules
- Prepare for NinjaTrader Sim backtest

**Ready to proceed with NinjaScript implementation after your approval?**
