# NinjaTrader Futures Trading - Conversion Plan

## Project Overview

Convert the proven **Combined V2 Forex Scalping Strategy** to NinjaTrader 8 for futures trading.

**Current Performance (Forex):**
- Win Rate: 48% (close to 51.9% backtest)
- Profit Factor: 1.27
- Return: +0.55% per session (~16% monthly)
- Position Size: 15% per trade
- Best Pairs: EUR/USD (80% WR), USD/CAD (100% WR), GBP/USD (50% WR)

---

## Phase 1: EXACT Strategy Conversion - Forex to Forex Futures

### 1.1 Current Strategy Components (Combined V2 - KEEP EXACTLY AS IS)

**Entry Signals (Need ANY 2 of 3):**
1. **Momentum Signal:**
   - MACD crossover + ADX > 25
   - Price breaking above/below Bollinger Bands

2. **Bollinger Band Signal:**
   - Price touching outer bands
   - RSI confirmation (oversold for buy, overbought for sell)

3. **Range Scalping:**
   - Price near support/resistance
   - Quick reversal setup

**Filters:**
- Higher Timeframe (HTF) trend alignment (15min, 30min)
- Pullback required (better entries)
- Volume/volatility confirmation

**Risk Management (KEEP EXACTLY AS IS):**
- TP: 15-30 pips (varies by pair)
- SL: 10-20 pips (varies by pair)
- Trailing Stop: Triggers at 20 pips, trails by 10 pips
- Position Size: 15% of account per trade
- Max Concurrent: 5 positions
- Max Daily Trades: 10 per instrument

### 1.2 Forex Futures Contracts (Direct Translation)

**EXACT MAPPING - Your Forex Pairs → CME Forex Futures:**

| Your Forex Pair | Futures Symbol | Micro Symbol | Contract Size | Tick Size | Tick Value | Margin (approx) |
|-----------------|----------------|--------------|---------------|-----------|------------|-----------------|
| **EUR/USD** | 6E | M6E | €125,000 | 0.00005 | $6.25 | $2,400 (6E) / $240 (M6E) |
| **GBP/USD** | 6B | M6B | £62,500 | 0.0001 | $6.25 | $3,300 (6B) / $330 (M6B) |
| **USD/JPY** | 6J | MJY | ¥12,500,000 | 0.0000005 | $6.25 | $2,200 (6J) / $220 (MJY) |
| **USD/CAD** | 6C | MCD | C$100,000 | 0.00005 | $5.00 | $1,800 (6C) / $180 (MCD) |
| **AUD/USD** | 6A | M6A | A$100,000 | 0.0001 | $10.00 | $2,000 (6A) / $200 (M6A) |
| **NZD/USD** | 6N | MNE | NZ$100,000 | 0.0001 | $10.00 | $1,500 (6N) / $150 (MNE) |
| **USD/CHF** | 6S | MSF | CHF125,000 | 0.00005 | $6.25 | $2,600 (6S) / $260 (MSF) |

**RECOMMENDATION: Use MICRO Forex Futures (M6E, M6B, etc.)**
- 10x smaller than standard contracts
- Margin: $150-$330 per contract (perfect for $5K account)
- Same tick structure as your current forex pairs
- **NO STRATEGY CHANGES NEEDED** - just translate pips to ticks

**Your Top 3 Forex Pairs → Micro Futures:**
1. **EUR/USD (80% WR) → M6E (Micro Euro)** - BEST PERFORMER
2. **USD/CAD (100% WR) → MCD (Micro Canadian Dollar)** - PERFECT WIN RATE
3. **GBP/USD (50% WR) → M6B (Micro British Pound)** - SOLID

---

## Phase 2: Strategy Conversion Mapping

### 2.1 DIRECT Translation (NO Changes to Strategy Logic)

**FOREX SPOT (OANDA) → FOREX FUTURES (CME):**

| Forex Concept | Forex Futures | Translation |
|---------------|---------------|-------------|
| **EUR/USD pair** | **M6E contract** | EXACT SAME MARKET |
| **1 pip (0.0001)** | **2 ticks (0.00005 × 2)** | 1 pip = 2 ticks = $12.50 |
| **18 pip TP** | **36 ticks TP** | M6E: 36 ticks = $225 |
| **15 pip SL** | **30 ticks SL** | M6E: 30 ticks = $187.50 |
| **15% position** | **2-3 contracts** | $5K × 15% = $750 / $240 margin = 3 contracts |
| **Spread (0.1 pip)** | **Commission** | $0.85/contract (NinjaTrader) |
| **5 max positions** | **5 max positions** | SAME |
| **10 trades/day** | **10 trades/day** | SAME |

**KEY POINT: Your strategy settings stay EXACTLY the same - just multiply pips × 2 for ticks!**

### 2.2 Position Sizing for Futures

**Account: $5,000**

**Per-Trade Risk: 2% = $100 max loss**

| Contract | Margin Req | Stop Loss | Max Contracts | Risk per Contract |
|----------|------------|-----------|---------------|-------------------|
| MES | $1,250 | 6 ticks ($7.50) | 3 contracts | $7.50 x 3 = $22.50 |
| MNQ | $1,400 | 12 ticks ($6.00) | 3 contracts | $6.00 x 3 = $18.00 |
| MYM | $1,100 | 12 ticks ($6.00) | 4 contracts | $6.00 x 4 = $24.00 |

**Recommended:** Start with **1-2 contracts** per trade until validated.

### 2.3 Timeframe Adjustments

**Forex (Current):**
- Entry: 1-minute chart
- Confirmation: 5-minute, 15-minute, 30-minute

**Futures (Proposed):**
- Entry: 1-minute chart (same)
- Confirmation: 5-minute, 15-minute (same)
- HTF Trend: 30-minute, 1-hour (futures trend longer)

**Session Focus:**
- **US Regular Session:** 9:30 AM - 4:00 PM EST (highest volume)
- **Extended Hours:** 6:00 PM - 9:30 AM EST (lower volume, wider spreads)

---

## Phase 3: NinjaTrader Implementation Architecture

### 3.1 Project Structure

```
trading_system/Forex_trading_ninjatrader/
├── strategies/
│   ├── CombinedV2FuturesStrategy.cs       # Main NinjaScript strategy
│   ├── FuturesScalpingIndicators.cs       # Custom indicators
│   └── FuturesRiskManager.cs              # Position sizing & risk
│
├── config/
│   ├── futures_instruments.json           # Instrument-specific settings
│   ├── trading_hours.json                 # Session definitions
│   └── risk_parameters.json               # Risk management rules
│
├── backtesting/
│   ├── backtest_runner.py                 # Python backtest wrapper
│   ├── historical_data/                   # Downloaded futures data
│   └── results/                           # Backtest results
│
├── utils/
│   ├── data_downloader.py                 # Download NinjaTrader data
│   ├── performance_analyzer.py            # Analyze NT8 results
│   └── symbol_mapper.py                   # Forex → Futures mapping
│
└── live/
    ├── live_trading_config.xml            # NT8 live config
    └── account_management.cs              # Account monitoring
```

### 3.2 NinjaScript Strategy Components

**File: CombinedV2FuturesStrategy.cs**

```csharp
namespace NinjaTrader.NinjaScript.Strategies
{
    public class CombinedV2FuturesStrategy : Strategy
    {
        // Strategy Parameters
        private int adxPeriod = 14;
        private int macdFast = 12;
        private int macdSlow = 26;
        private int macdSignal = 9;
        private int rsiPeriod = 14;
        private int bbPeriod = 20;
        private double bbStdDev = 2.0;

        // Risk Parameters
        private int stopLossTicks = 6;          // 6 ticks for MES
        private int takeProfitTicks = 8;        // 8 ticks for MES
        private int trailingTriggerTicks = 8;   // Start trailing after 8 ticks
        private int trailingStopTicks = 4;      // Trail by 4 ticks

        private int maxContracts = 2;
        private int maxDailyTrades = 10;

        // Main Components
        1. Initialize() - Setup indicators
        2. OnBarUpdate() - Check signals every bar
        3. CheckEntrySignals() - Combined V2 logic
        4. CheckExitSignals() - TP/SL/Trailing
        5. CalculatePositionSize() - Contracts to trade
        6. ValidateHTFTrend() - Higher timeframe filter
    }
}
```

### 3.3 Indicator Requirements

**Built-in NinjaTrader Indicators:**
- ADX (Average Directional Index)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- EMA (Exponential Moving Avergie)
- Volume

**Custom Indicators Needed:**
- HTF Trend Filter (multi-timeframe)
- Pullback Detector
- Momentum Score Calculator

---

## Phase 4: Backtesting Plan

### 4.1 Data Requirements

**Historical Data:**
- Source: NinjaTrader Market Replay or Rithmic
- Timeframe: Last 6 months minimum (for validation)
- Resolution: 1-minute bars
- Instruments: MES, MNQ, MYM

**Data Quality Checks:**
- No gaps during regular session
- Accurate bid/ask spreads
- Volume data included

### 4.2 Backtest Scenarios

**Test 1: Single Instrument (MES)**
- Period: 3 months
- Session: Regular hours only (9:30 AM - 4:00 PM)
- Expected: 50+ trades, 45%+ WR, 1.2+ PF

**Test 2: Multi-Instrument (MES + MNQ)**
- Period: 3 months
- Session: Regular hours
- Expected: 100+ trades, 45%+ WR, 1.2+ PF

**Test 3: Extended Hours**
- Period: 1 month
- Session: Full 23-hour session
- Compare: Regular vs Extended performance

**Test 4: Different Market Conditions**
- Trending Month (e.g., October 2024)
- Ranging Month (e.g., August 2024)
- Volatile Month (e.g., Election week)

### 4.3 Performance Metrics

**Target Metrics (Minimum Acceptable):**
- Win Rate: ≥ 45%
- Profit Factor: ≥ 1.2
- Max Drawdown: ≤ 15%
- Sharpe Ratio: ≥ 1.0
- Average Win/Loss Ratio: ≥ 1.3

**Expected Monthly Return:**
- Conservative: 5-10%
- Realistic: 10-15%
- Optimistic: 15-20%

---

## Phase 5: Implementation Steps

### Step 1: Environment Setup (Week 1)
- [ ] Install NinjaTrader 8
- [ ] Setup Rithmic/CQG data feed (or use free Kinetick)
- [ ] Download 6 months historical data for MES, MNQ, MYM
- [ ] Create project folder structure

### Step 2: Strategy Translation (Week 1-2)
- [ ] Translate momentum signal to C#
- [ ] Translate Bollinger Band signal to C#
- [ ] Translate range scalping signal to C#
- [ ] Implement HTF trend filter
- [ ] Implement pullback detector

### Step 3: Risk Management (Week 2)
- [ ] Position sizing logic (contracts based on account)
- [ ] Stop loss placement (tick-based)
- [ ] Take profit placement (tick-based)
- [ ] Trailing stop logic

### Step 4: Backtesting (Week 3)
- [ ] Run backtest on MES (3 months)
- [ ] Analyze results (win rate, PF, drawdown)
- [ ] Optimize parameters if needed
- [ ] Run multi-instrument backtest
- [ ] Compare to Forex results

### Step 5: Paper Trading (Week 4)
- [ ] Setup NinjaTrader Sim account
- [ ] Run strategy live on Sim for 1 week
- [ ] Monitor real-time performance
- [ ] Fix any live execution issues
- [ ] Compare Sim vs Backtest results

### Step 6: Live Trading (Week 5+)
- [ ] Start with 1 contract only
- [ ] Trade MES only for first 2 weeks
- [ ] Monitor daily performance
- [ ] Scale to 2 contracts if successful
- [ ] Add MNQ after 1 month

---

## Phase 6: Risk Management & Safeguards

### 6.1 Daily Risk Limits

**Max Daily Loss:** $100 (2% of $5,000)
- If hit, stop trading for the day
- Auto-disable strategy in NinjaTrader

**Max Daily Profit Target:** $250 (5% of $5,000)
- Optional: Stop at profit target to lock in gains
- Or let it run with trailing stops

**Max Consecutive Losses:** 5 trades
- If hit, stop trading and review
- Check for market regime change

### 6.2 Position Limits

**Max Contracts per Trade:**
- MES: 2 contracts max
- MNQ: 2 contracts max
- MYM: 2 contracts max

**Max Total Contracts:** 4 contracts across all instruments

**Max Trades per Day:** 10 per instrument (30 total)

### 6.3 Trading Hours Restrictions

**Allowed Hours:**
- Regular Session: 9:30 AM - 3:30 PM EST (avoid last 30 min volatility)
- Extended Hours: 6:00 PM - 8:00 PM EST (optional, lower volume)

**Forbidden Hours:**
- Market Open: 9:30 - 10:00 AM (initial volatility)
- Market Close: 3:30 - 4:00 PM (closing imbalances)
- Asian Session: 8:00 PM - 6:00 AM (low volume, wide spreads)

### 6.4 News/Event Blackout

**Avoid Trading 30 Minutes Before/After:**
- FOMC Announcements
- NFP (Non-Farm Payroll)
- CPI (Consumer Price Index)
- GDP Releases
- Fed Chair Speeches

---

## Phase 7: Cost Analysis

### 7.1 Software Costs

| Item | Cost | Frequency |
|------|------|-----------|
| NinjaTrader 8 License | Free (for backtesting) | One-time |
| NinjaTrader Lifetime License | $1,395 | One-time (optional) |
| NinjaTrader Lease | $60/month | Monthly (alternative) |
| Market Data (Kinetick) | Free (delayed) | N/A |
| Market Data (Rithmic) | $90/month | Monthly (real-time) |

**Recommended for Testing:** Use free NT8 + Kinetick delayed data

**Recommended for Live:** NT8 Lease ($60/mo) + Rithmic ($90/mo) = $150/mo

### 7.2 Trading Costs

**Per Contract (Round Trip):**
- Commission: $0.50 - $1.50 (varies by broker)
- Exchange Fees: $1.50 - $2.50
- **Total: ~$2.00 - $4.00 per round trip**

**Monthly Cost (Estimated):**
- 200 trades/month × 2 contracts × $3.00 = **$1,200/month**

**Profit Required to Break Even:**
- $1,200 / $5,000 = **24% just to cover costs**
- Need **30%+ gross returns** for **6% net profit**

**Note:** This is HIGH for a $5K account. Consider:
- Starting with $10K account
- Or using volume discounts from broker

---

## Phase 8: Expected Performance

### 8.1 Conservative Projections

**Account: $5,000**
**Trades/Month: 200** (10/day × 20 days)
**Win Rate: 45%**
**Profit Factor: 1.2**
**Avg Win: $15** (12 ticks MES × $1.25)
**Avg Loss: $10** (8 ticks MES × $1.25)

**Monthly Performance:**
- Wins: 90 trades × $15 = $1,350
- Losses: 110 trades × $10 = -$1,100
- Gross Profit: $250
- Costs: -$1,200 (commissions)
- **Net Profit: -$950 (LOSS)**

**PROBLEM:** Costs eat all profits on $5K account!

### 8.2 Realistic Projections (Need Higher Win Rate or Bigger Account)

**Option 1: Increase Win Rate to 50%**
- Wins: 100 trades × $15 = $1,500
- Losses: 100 trades × $10 = -$1,000
- Gross Profit: $500
- Costs: -$1,200
- **Net Profit: -$700 (STILL LOSS)**

**Option 2: Use $10K Account (2x contracts)**
- Gross Profit: $500 × 2 = $1,000
- Costs: -$1,200 (same)
- **Net Profit: -$200 (STILL LOSS)**

**Option 3: Reduce Trades to 100/month (5/day) + 50% WR**
- Wins: 50 × $30 = $1,500 (2 contracts)
- Losses: 50 × $20 = -$1,000
- Gross Profit: $500
- Costs: -$600 (100 trades × 2 contracts × $3)
- **Net Profit: -$100 (CLOSE!)**

**SOLUTION:** Need **15-20 point TP** and **50%+ win rate** OR **$10K account**

---

## Phase 9: Key Differences: Forex vs Futures

| Aspect | Forex (OANDA) | Futures (NinjaTrader) |
|--------|---------------|----------------------|
| **Leverage** | 20:1 (5% margin) | ~4:1 (25% margin) |
| **Costs** | Spread (~1 pip) | Commission (~$3/RT) |
| **Session** | 24/5 | 23/5 (1hr break) |
| **Liquidity** | Very high (EUR/USD) | Very high (MES) |
| **Contract Size** | Variable (any amount) | Fixed (1 contract = $5/point MES) |
| **Slippage** | Low (0.1-0.5 pips) | Low (0.25-0.5 ticks) |
| **Min Account** | $1,000 | $5,000+ |
| **Scalability** | High | Medium |

---

## Phase 10: Success Criteria

### Before Going Live:

**Backtest Results Must Show:**
- [ ] Win Rate ≥ 48%
- [ ] Profit Factor ≥ 1.25
- [ ] Max Drawdown ≤ 15%
- [ ] Positive returns over 3+ months
- [ ] Consistent performance across different market conditions

**Sim Trading Results Must Show:**
- [ ] 1 week of profitable trading
- [ ] Win rate within 5% of backtest
- [ ] No execution errors
- [ ] Slippage < 1 tick per trade

**Account Requirements:**
- [ ] At least $5,000 in trading account
- [ ] Additional $2,000 buffer for margin calls
- [ ] Can afford $150/month for software/data

---

## Summary & Recommendation

### Recommended Plan:

**Phase 1 (Week 1-2): Build & Test**
- Translate strategy to NinjaScript
- Backtest on MES with 6 months data
- Target: 48% WR, 1.25 PF minimum

**Phase 2 (Week 3-4): Optimize & Validate**
- Optimize TP/SL for futures tick structure
- Test on MNQ and MYM
- Run walk-forward analysis

**Phase 3 (Week 5-6): Sim Trading**
- Paper trade live market for 2 weeks
- Validate backtest results
- Identify execution issues

**Phase 4 (Week 7+): Live with 1 Contract**
- Start MES only, 1 contract
- Scale to 2 contracts after 2 profitable weeks
- Add MNQ after 1 profitable month

### Critical Success Factors:

1. **Account Size:** $5K minimum, $10K recommended
2. **Win Rate:** Must achieve 50%+ (higher than Forex due to costs)
3. **Trade Frequency:** 5-10 trades/day max (lower costs)
4. **Risk Management:** 2% max risk per trade, 10% daily stop
5. **Session Selection:** Trade regular hours only (9:30 AM - 3:30 PM)

---

## APPROVAL NEEDED

Please review and approve:

1. **Instrument Selection:** MES, MNQ, MYM (Micro E-mini futures) - Approved?
2. **Position Sizing:** 1-2 contracts per trade, 15% account risk - Approved?
3. **TP/SL Levels:** 8 tick TP, 6 tick SL (MES) - Approved?
4. **Implementation:** NinjaTrader 8 C# strategy - Approved?
5. **Timeline:** 4-6 weeks development + testing - Approved?
6. **Budget:** $150/month software + $1,200/month trading costs - Approved?

Once approved, I will begin Phase 1: Strategy Translation to NinjaScript.
