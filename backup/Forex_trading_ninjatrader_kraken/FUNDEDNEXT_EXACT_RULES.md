# FundedNext $25K Challenge - EXACT Rules & Compliance Strategy

## Official FundedNext $25K Challenge Rules

---

## 1. Maximum Loss Limit: $1,000 (EOD Balance Trailing Method)

### What is EOD Balance Trailing?

**Definition:** Your account equity (balance + open P&L) cannot drop more than $1,000 below the **highest EOD (End of Day) balance** you've achieved.

**How it works:**
```
Starting balance: $25,000
Max Loss Limit: $1,000

Day 1 EOD Balance: $25,200 (made $200)
  → New max loss threshold: $25,200 - $1,000 = $24,200

Day 2 EOD Balance: $25,500 (made another $300)
  → New max loss threshold: $25,500 - $1,000 = $24,500

Day 3 during trading: Account drops to $24,450
  → Still SAFE ($24,500 threshold not breached)

Day 3 during trading: Account drops to $24,400
  → ACCOUNT FAILED (below $24,500 threshold)
```

**CRITICAL INSIGHT:**
- The max loss limit **TRAILS UP** as you make profit
- Once you make $500, your kill switch moves from $24,000 to $24,500
- **You can NEVER give back more than $1,000 from your highest EOD balance**

---

## 2. Profit Target: $1,250 (5% of $25K)

**What you need:**
- Total profit: $1,250
- Starting from: $25,000
- Target balance: $26,250

**Timeline:**
- No time limit (can take as long as needed)
- Most traders: 2-4 weeks

**Once achieved:**
- Account moves to Funded phase
- You get 80% profit split

---

## 3. Consistency Rule: 40% Max Per Day (Challenge ONLY)

**What it means:**
- No single day can contribute more than 40% of your **total profit**
- Only applies during Challenge phase
- Does NOT apply once you're funded

**Example:**
```
Total profit to pass: $1,250

Max profit on any single day: $1,250 × 40% = $500

If Day 1 profit = $600:
  → Need at least $900 more profit on other days
  → $600 / ($600 + $900) = 40% ✓

If Day 1 profit = $700:
  → Need at least $1,050 more profit
  → But total would be $1,750 (exceeds $1,250 target)
  → Would pass challenge, but $700 is 40% of $1,750 ✓
```

**Strategy:**
- Stop trading at +$400/day during challenge
- Ensures you stay well under 40% threshold

---

## 4. Max Positions: 20 Micros (Challenge) → 30 Micros (Funded)

**Challenge Phase:**
- Max concurrent: 20 Micro contracts
- OR: 2 Mini contracts
- Cannot mix (must choose one or the other)

**Your Strategy:**
- 5 pairs × 1 contract each = **5 Micros** ✓ Well under limit
- Room to scale to 4 contracts per pair (20 total)

**Funded Phase:**
- Increases to 30 Micros or 3 Minis

---

## 5. No Daily Loss Limit

**Good news:**
- FundedNext has NO daily loss limit during challenge
- You can lose as much as you want in a single day
- **BUT:** Must not breach the $1,000 max loss limit

**Risk:**
- Without self-imposed daily limit, you could blow account in 1 day
- Example: 6 losing trades × $200 = -$1,200 (account failed)

**Protection:**
- Self-impose -$200/day limit
- Prevents catastrophic single-day loss

---

## Critical Risk Analysis with EOD Trailing

### Scenario 1: Steady Growth (IDEAL)

```
Day 1: Start $25,000 → End $25,150 (+$150)
  Max loss threshold: $25,150 - $1,000 = $24,150

Day 2: Start $25,150 → End $25,300 (+$150)
  Max loss threshold: $25,300 - $1,000 = $24,300

Day 3: Start $25,300 → End $25,450 (+$150)
  Max loss threshold: $25,450 - $1,000 = $24,450

...continue until $26,250 (pass challenge)

Final threshold: $26,250 - $1,000 = $25,250
```

**Safety:** Even after passing, you have $1,000 buffer from original start

---

### Scenario 2: Big Win Day Then Drawdown (DANGEROUS)

```
Day 1: Start $25,000 → End $26,000 (+$1,000) 🎉
  Max loss threshold: $26,000 - $1,000 = $25,000

Day 2: Start $26,000 → Lose $500 → End $25,500
  Current equity: $25,500
  Threshold: $25,000 ✓ SAFE (but very close!)

Day 3: Start $25,500 → Lose $600 → Account at $24,900
  Current equity: $24,900
  Threshold: $25,000 ❌ FAILED!
```

**DANGER:** After a big win day, your threshold moves UP. Any drawdown can fail you!

---

### Scenario 3: What YOUR Strategy Faces

**With 1 contract per trade, max 5 concurrent:**

```
Day 1: Start $25,000
  Trade 1: M6E loss -$200 → $24,800
  Trade 2: M6B loss -$156 → $24,644
  Trade 3: MJY loss -$187 → $24,457
  Trade 4: MCD loss -$160 → $24,297
  Trade 5: MSF loss -$150 → $24,147

  5 consecutive losses = -$853
  Account: $24,147
  Threshold: $24,000 (from $25,000 start)
  Status: ✓ SAFE (but only $147 buffer left!)
```

**What happens if you have 6th loss:**
```
  Trade 6: M6E loss -$200 → $23,947
  Threshold: $24,000
  Status: ❌ FAILED (below threshold by $53)
```

**CRITICAL INSIGHT: You can only afford 5-6 consecutive losses before account failure!**

---

## Position Sizing Strategy for EOD Trailing Protection

### Current Plan: 1 Contract Per Trade

**Risk per trade:**
- M6E: $200 (0.8% of $25K)
- M6B: $156 (0.63%)
- MJY: $187 (0.75%)
- MCD: $160 (0.64%)
- MSF: $150 (0.6%)

**Max concurrent risk:** 5 positions = $853 (3.4%)

**Problem:**
- With 48% WR (52% loss rate), you could easily hit 5-6 losers in a row
- 10 trades/day × 52% = 5-6 losers expected DAILY
- Without daily stop, you'd hit max loss in 1-2 bad days

---

### SOLUTION: Tiered Risk Management

#### Level 1: Fresh Account ($25,000)
```
Max loss buffer: $1,000 (to $24,000 threshold)
Position size: 1 contract per trade
Max concurrent: 5 positions
Daily loss limit: -$300 (self-imposed)

Safe zone: Account between $25,000 - $25,700
```

#### Level 2: Small Profit ($25,700+)
```
Max loss buffer: Still $1,000 (threshold now $24,700+)
Position size: 1 contract per trade
Max concurrent: 5 positions
Daily loss limit: -$300

Caution zone: Any profit raises threshold
Risk: Larger drawdown from higher peak
```

#### Level 3: Near Target ($26,000+)
```
Max loss buffer: $1,000 (threshold now $25,000+)
Position size: REDUCE to 1 contract MAX 3 concurrent
Daily loss limit: -$200
Daily profit target: +$50-100 (slow and steady)

Critical zone: Protect profits, avoid giving back gains
Strategy: Trade only highest WR pairs (EUR/USD, USD/CAD)
```

#### Level 4: Passed Challenge ($26,250+)
```
STOP TRADING!
Submit for funded account
Don't risk giving back profits
```

---

## Daily Trading Limits (REVISED for EOD Trailing)

### Day 1-5: Building Profit (Account: $25,000 - $25,800)

**Daily Profit Target:** +$150-200
**Daily Loss Limit:** -$300 (HARD STOP)
**Position Size:** 1 contract per trade
**Max Concurrent:** 5 positions
**Max Trades/Day:** 30 total (10 per pair on 3 pairs)

**Why -$300 daily limit:**
- Gives you 3 bad days buffer before hitting max loss
- 3 days × -$300 = -$900 (still $100 from $24,000 threshold)

---

### Day 6-10: Approaching Target (Account: $25,800 - $26,200)

**Daily Profit Target:** +$100-150
**Daily Loss Limit:** -$200 (TIGHTER STOP)
**Daily Profit Cap:** +$400 (consistency rule)
**Position Size:** 1 contract per trade
**Max Concurrent:** 5 positions

**Why tighter -$200 limit:**
- Your threshold has moved UP (now $24,800+ instead of $24,000)
- Less room for drawdown
- Protect accumulated profits

---

### Day 11+: Final Push (Account: $26,200 - $26,250)

**Daily Profit Target:** +$50
**Daily Loss Limit:** -$150 (VERY TIGHT)
**Position Size:** 1 contract (only trade 2-3 best pairs)
**Max Concurrent:** 3 positions MAX

**Why ultra-conservative:**
- You're only $50 away from passing
- Threshold is now ~$25,200 (only $800 buffer from current $26,200)
- One bad day could give back $500+ and risk failure
- SLOW AND STEADY WINS

---

## Expected Path to $1,250 Target

### Realistic Scenario (48% WR, $11.88 avg/trade):

```
Week 1: 50 trades
  Expected: 24 winners, 26 losers
  Winners: 24 × $210 = +$5,040
  Losers: 26 × $171 = -$4,446
  Net: +$594 (account: $25,594)
  EOD threshold: $24,594

Week 2: 50 trades
  Expected: 24 winners, 26 losers
  Net: +$594 (account: $26,188)
  EOD threshold: $25,188

Week 3: 10 trades (final push)
  Need: $62 more
  Expected: 5 winners, 5 losers
  Winners: 5 × $210 = +$1,050
  Losers: 5 × $171 = -$855
  Net: +$195 (account: $26,383)

PASS CHALLENGE! ✓
Final threshold: $25,383 ($1,383 above start)
```

**Timeline:** 3 weeks, 110 trades

---

### Optimistic Scenario (Trade only EUR/USD + USD/CAD at 80%+ WR):

```
Week 1: 50 trades (EUR/USD only)
  Expected: 40 winners, 10 losers
  Winners: 40 × $250 = +$10,000
  Losers: 10 × $200 = -$2,000
  Net: +$8,000 (account: $33,000) 🎉

PROBLEM: Consistency Rule Violation!
  Day 1 profit: $1,000
  Total profit: $8,000
  Day 1 %: $1,000 / $8,000 = 12.5% ✓ Under 40%

BUT: You only needed $1,250, not $8,000!

BETTER STRATEGY:
  Stop at +$400/day
  10 trades/day × 80% WR = 8 winners, 2 losers
  Net: (8 × $250) - (2 × $200) = +$1,600/day
  Hit +$400 after 3-4 trades
  STOP TRADING

  Days needed: $1,250 / $400 = 3.1 days
```

**Timeline:** 4-5 days (with consistency rule buffer)

---

### Conservative Scenario (40% WR - worst case):

```
Week 1: 50 trades
  Expected: 20 winners, 30 losers
  Winners: 20 × $210 = +$4,200
  Losers: 30 × $171 = -$5,130
  Net: -$930 (account: $24,070)
  EOD threshold: $24,000
  Status: DANGER - only $70 buffer!

Day 6: Hit -$300 daily limit early
  Account: $23,770
  Threshold: $24,000
  Status: FAILED ❌
```

**Protection:** Daily -$300 limit would stop you on Day 1 after losing -$300
- Prevents full -$930 weekly loss
- Gives you time to review strategy before continuing

---

## NinjaScript Implementation Requirements

### Account Protection (CRITICAL - MUST IMPLEMENT):

```csharp
// FundedNext Challenge Settings
private double INITIAL_BALANCE = 25000;
private double MAX_LOSS_LIMIT = 1000;
private double PROFIT_TARGET = 1250;

// Track highest EOD balance
private double highestEODBalance = 25000;
private double currentThreshold = 24000; // $25,000 - $1,000

// Daily limits (self-imposed)
private double dailyLossLimit = -300;
private double dailyProfitCap = 400;

// Position limits
private int maxConcurrentPositions = 5;
private int contractsPerTrade = 1;

protected override void OnBarUpdate()
{
    // 1. Check if account is below threshold (KILL SWITCH)
    double currentEquity = Account.Get(AccountItem.CashValue, Currency.UsDollar);

    if (currentEquity <= currentThreshold)
    {
        Print("ACCOUNT FAILED: Below max loss threshold!");
        CloseAllPositions();
        DisableStrategy();
        return;
    }

    // 2. Update EOD balance and threshold (run at market close)
    if (ToTime(Time[0]) == 160000) // 4:00 PM EST
    {
        if (currentEquity > highestEODBalance)
        {
            highestEODBalance = currentEquity;
            currentThreshold = highestEODBalance - MAX_LOSS_LIMIT;
            Print($"New EOD Balance: {highestEODBalance}, New Threshold: {currentThreshold}");
        }
    }

    // 3. Check daily profit cap (consistency rule)
    double dailyPnL = GetDailyPnL();

    if (dailyPnL >= dailyProfitCap)
    {
        Print("Daily profit cap reached. Stopping for the day.");
        return;
    }

    // 4. Check daily loss limit (self-imposed protection)
    if (dailyPnL <= dailyLossLimit)
    {
        Print("Daily loss limit hit. Stopping for the day.");
        CloseAllPositions();
        return;
    }

    // 5. Check if profit target reached
    double totalProfit = currentEquity - INITIAL_BALANCE;

    if (totalProfit >= PROFIT_TARGET)
    {
        Print("PROFIT TARGET REACHED! Stop trading and submit for funding.");
        CloseAllPositions();
        DisableStrategy();
        return;
    }

    // 6. Proceed with normal strategy logic...
    CheckEntrySignals();
}
```

---

## Risk Management Per Trading Phase

### Phase 1: First $500 Profit (Days 1-5)

**Account Range:** $25,000 - $25,500
**Threshold Range:** $24,000 - $24,500

**Settings:**
- Contracts: 1 per trade
- Max concurrent: 5 positions
- Daily loss limit: -$300
- Daily profit cap: +$400
- Trade all 5 pairs

**Risk:** Moderate (largest buffer from threshold)

---

### Phase 2: Middle $500 Profit (Days 6-10)

**Account Range:** $25,500 - $26,000
**Threshold Range:** $24,500 - $25,000

**Settings:**
- Contracts: 1 per trade
- Max concurrent: 5 positions
- Daily loss limit: -$250 (TIGHTER)
- Daily profit cap: +$350
- Trade all 5 pairs

**Risk:** Increasing (threshold rising with account)

---

### Phase 3: Final $250 Profit (Days 11-15)

**Account Range:** $26,000 - $26,250
**Threshold Range:** $25,000 - $25,250

**Settings:**
- Contracts: 1 per trade
- Max concurrent: 3 positions (REDUCE)
- Daily loss limit: -$150 (VERY TIGHT)
- Daily profit cap: +$100
- Trade ONLY EUR/USD + USD/CAD (best performers)

**Risk:** HIGH (close to passing, protect profits)

---

## Final Recommendation

### Position Sizing: ✓ APPROVED
- **1 contract per trade** on all 5 pairs
- Risk: $150-200 per trade (0.6-0.8%)
- Max concurrent: 5 positions ($853 total risk = 3.4%)

### Daily Limits: ✓ APPROVED
- **Daily loss limit:** -$300 (Phase 1), -$250 (Phase 2), -$150 (Phase 3)
- **Daily profit cap:** +$400 (consistency rule protection)

### Account Protection: ✓ CRITICAL
- **EOD Trailing Threshold:** Track highest EOD balance, stop if below $1,000 from peak
- **Profit Target:** Stop trading at $26,250 (PASS!)

### Expected Timeline: ✓ REALISTIC
- **Best case:** 4-5 days (80% WR on EUR/USD)
- **Realistic:** 2-3 weeks (48% WR across all pairs)
- **Conservative:** 4 weeks (with bad luck)

---

## APPROVAL CHECKLIST

Please confirm you understand and approve:

- ✓ Max Loss Limit ($1,000) uses EOD Balance Trailing method?
- ✓ Threshold moves UP as you make profit?
- ✓ After making $500, you can only draw down $1,000 from that peak (not from start)?
- ✓ Position sizing (1 contract = $150-200 risk)?
- ✓ Daily loss limits (-$300 initially, tighter as you approach target)?
- ✓ Daily profit cap (+$400 for consistency rule)?
- ✓ Tiered risk management (reduce positions as you approach $26,250)?

**Once approved, I will create the complete NinjaScript C# strategy with ALL FundedNext protection rules built in.**

**Ready to proceed?**
