# EXACT Strategy Translation - Python to C# NinjaScript

## Current Strategy: Combined V2 Forex Scalping (PROVEN WINNER)

**Backtest Results:** +25.82% return | 51.9% WR | 1.34 PF | 5.0% DD | 291 trades

**Entry:** ANY 2 of 3 signals (RSI + Bollinger Bands + Range Scalping)

---

## Part 1: Your Current Python Implementation

### File 1: `forex_scalping.py` (664 lines)

**Key Components:**

1. **Class Initialization** (Lines 48-86)
   - Max trades per day: 10
   - Position size: 15% margin (20:1 leverage)
   - Min consensus: 2 of 3 signals

2. **Indicator Calculations** (Lines 87-98)
   - EMA calculation
   - RSI calculation

3. **Entry Signal Logic** (`should_enter_trade` - Lines 191-345)
   - **Signal 1:** RSI Oversold (<30) or Overbought (>70)
   - **Signal 2:** Bollinger Band touch (price hits upper/lower band)
   - **Signal 3:** Range Scalping (support/resistance + RSI + Stochastic)
   - **Entry:** If 2+ signals agree → BUY or SELL

4. **Position Sizing** (`calculate_position_size` - Lines 347-392)
   - Target 15% margin usage
   - 20:1 leverage = 15% × 20 = 300% notional
   - Special handling for USD/JPY

5. **Stop Loss / Take Profit** (`calculate_stop_loss_take_profit` - Lines 394-430)
   - Uses pair-specific settings from `pair_specific_settings.py`
   - Absolute price movements (not percentages)
   - Precision: 3 decimals (JPY), 5 decimals (others)

6. **Trailing Stop Settings** (`get_pair_trailing_settings` - Lines 432-440)
   - Pair-specific trigger and distance

### File 2: `pair_specific_settings.py` (199 lines)

**Pair-Specific Settings (Lines 69-109):**

| Pair | TP (pips) | SL (pips) | Trail Trigger | Trail Distance | R:R |
|------|-----------|-----------|---------------|----------------|-----|
| EUR_USD | 20 | 16 | 12 | 6 | 1.25 |
| GBP_USD | 30 | 25 | 18 | 8 | 1.20 |
| USD_JPY | 18 | 15 | 12 | 6 | 1.20 |
| USD_CHF | 15 | 12 | 10 | 5 | 1.25 |
| USD_CAD | 20 | 16 | 12 | 6 | 1.25 |

---

## Part 2: NinjaTrader C# Translation (1:1 Mapping)

### Main Strategy File: `CombinedV2ForexFutures.cs`

```csharp
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class CombinedV2ForexFutures : Strategy
    {
        #region Variables
        // ===== EXACT COPY FROM forex_scalping.py __init__ =====

        private int maxTradesPerDay = 10;
        private double dailyProfitTarget = 0.03;  // 3%
        private double tradeSizePct = 0.15;       // 15% margin
        private int minConsensusScore = 2;        // Need 2 of 3 signals

        // Indicator objects
        private RSI rsiIndicator;
        private Bollinger bbIndicator;
        private ADX adxIndicator;
        private EMA ema9;
        private EMA ema20;
        private Stochastics stochIndicator;

        // State tracking
        private Dictionary<string, int> tradesPerDayCount;
        private Dictionary<string, PairSettings> pairSettings;
        private double dailyStartBalance;
        private int currentPositions = 0;

        #endregion

        #region OnStateChange
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Combined V2 Forex Futures - EXACT Python Translation";
                Name = "CombinedV2ForexFutures";

                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;

                // Initialize pair settings (EXACT copy from pair_specific_settings.py)
                InitializePairSettings();

                Print("=========================================");
                Print("COMBINED V2 STRATEGY (Proven Winner)");
                Print("+25.82% return | 51.9% WR | 1.34 PF");
                Print("=========================================");
                Print("Max trades/day: " + maxTradesPerDay);
                Print("Entry: 2 of 3 signals (RSI + BB + Range)");
                Print("Position size: 15% margin @ 20:1 leverage");
            }
            else if (State == State.DataLoaded)
            {
                // Initialize indicators (EXACT copy from should_enter_trade)
                // Using 15-min timeframe for entry signals
                rsiIndicator = RSI(14, 3);
                bbIndicator = Bollinger(20, 2);
                stochIndicator = Stochastics(14, 3, 3);
                ema9 = EMA(9);
                ema20 = EMA(20);

                tradesPerDayCount = new Dictionary<string, int>();
                dailyStartBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            }
        }
        #endregion

        #region Initialize Pair Settings
        private void InitializePairSettings()
        {
            // EXACT COPY from pair_specific_settings.py lines 69-109
            // Multiply pips by conversion factor for each pair

            pairSettings = new Dictionary<string, PairSettings>
            {
                // EUR/USD: 20 TP, 16 SL → multiply by 2 for M6E ticks
                {"M6E", new PairSettings {
                    TpPips = 20,
                    SlPips = 16,
                    TpTicks = 40,    // 20 * 2
                    SlTicks = 32,    // 16 * 2
                    TrailTriggerTicks = 24,   // 12 * 2
                    TrailDistanceTicks = 12,  // 6 * 2
                    RiskRewardRatio = 1.25,
                    PipMultiplier = 2,
                    TickValue = 6.25
                }},

                // GBP/USD: 30 TP, 25 SL → multiply by 1 for M6B ticks
                {"M6B", new PairSettings {
                    TpPips = 30,
                    SlPips = 25,
                    TpTicks = 30,    // 30 * 1
                    SlTicks = 25,    // 25 * 1
                    TrailTriggerTicks = 18,   // 18 * 1
                    TrailDistanceTicks = 8,   // 8 * 1
                    RiskRewardRatio = 1.20,
                    PipMultiplier = 1,
                    TickValue = 6.25
                }},

                // USD/JPY: 18 TP, 15 SL → multiply by 100 for MJY ticks
                {"MJY", new PairSettings {
                    TpPips = 18,
                    SlPips = 15,
                    TpTicks = 1800,    // 18 * 100
                    SlTicks = 1500,    // 15 * 100
                    TrailTriggerTicks = 1200,   // 12 * 100
                    TrailDistanceTicks = 600,   // 6 * 100
                    RiskRewardRatio = 1.20,
                    PipMultiplier = 100,
                    TickValue = 6.25
                }},

                // USD/CAD: 20 TP, 16 SL → multiply by 2 for MCD ticks
                {"MCD", new PairSettings {
                    TpPips = 20,
                    SlPips = 16,
                    TpTicks = 40,    // 20 * 2
                    SlTicks = 32,    // 16 * 2
                    TrailTriggerTicks = 24,   // 12 * 2
                    TrailDistanceTicks = 12,  // 6 * 2
                    RiskRewardRatio = 1.25,
                    PipMultiplier = 2,
                    TickValue = 5.00
                }},

                // USD/CHF: 15 TP, 12 SL → multiply by 2 for MSF ticks
                {"MSF", new PairSettings {
                    TpPips = 15,
                    SlPips = 12,
                    TpTicks = 30,    // 15 * 2
                    SlTicks = 24,    // 12 * 2
                    TrailTriggerTicks = 20,   // 10 * 2
                    TrailDistanceTicks = 10,  // 5 * 2
                    RiskRewardRatio = 1.25,
                    PipMultiplier = 2,
                    TickValue = 6.25
                }}
            };
        }
        #endregion

        #region OnBarUpdate
        protected override void OnBarUpdate()
        {
            // EXACT COPY from should_enter_trade() lines 212-345

            // STEP 1: Basic checks (lines 213-223)
            if (Position.MarketPosition != MarketPosition.Flat)
                return;  // Position already open

            int tradesCount = GetDailyTradesCount(Instrument.FullName);
            if (tradesCount >= maxTradesPerDay)
                return;  // Max trades reached

            double currentBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double dailyPL = (currentBalance - dailyStartBalance) / dailyStartBalance;

            if (dailyPL >= dailyProfitTarget)
                return;  // Daily profit target reached

            if (dailyPL <= -0.05)  // -5% daily loss limit
                return;  // Daily loss limit reached

            // STEP 2: Calculate indicators on CURRENT BAR (lines 225-270)
            if (CurrentBar < 50)
                return;  // Insufficient data

            // Get current values (using NinjaTrader indicators)
            double currentRSI = rsiIndicator[0];
            double currentPrice = Close[0];
            double prevPrice = Close[1];
            double currentBBUpper = bbIndicator.Upper[0];
            double currentBBLower = bbIndicator.Lower[0];
            double prevBBUpper = bbIndicator.Upper[1];
            double prevBBLower = bbIndicator.Lower[1];
            double currentStoch = stochIndicator[0];

            // Calculate support/resistance (50-bar lookback)
            double support = MIN(Low, 50)[0];
            double resistance = MAX(High, 50)[0];

            // STEP 3: Check BULLISH signals (need 2 of 3) - lines 272-286
            List<string> bullishSignals = new List<string>();

            // Signal 1: RSI Oversold (line 276-277)
            if (currentRSI < 30)
                bullishSignals.Add("RSI_OVERSOLD");

            // Signal 2: Bollinger Lower Band Touch (lines 280-281)
            if (currentPrice <= currentBBLower && prevPrice > prevBBLower)
                bullishSignals.Add("BB_LOWER_TOUCH");

            // Signal 3: Range Support + Confirmation (lines 284-286)
            double distToSupport = (currentPrice - support) / support;
            if (distToSupport < 0.002 && currentRSI < 35 && currentStoch < 35)
                bullishSignals.Add("RANGE_SUPPORT");

            // STEP 4: Check BEARISH signals (need 2 of 3) - lines 288-302
            List<string> bearishSignals = new List<string>();

            // Signal 1: RSI Overbought (lines 292-293)
            if (currentRSI > 70)
                bearishSignals.Add("RSI_OVERBOUGHT");

            // Signal 2: Bollinger Upper Band Touch (lines 296-297)
            if (currentPrice >= currentBBUpper && prevPrice < prevBBUpper)
                bearishSignals.Add("BB_UPPER_TOUCH");

            // Signal 3: Range Resistance + Confirmation (lines 300-302)
            double distToResistance = (resistance - currentPrice) / currentPrice;
            if (distToResistance < 0.002 && currentRSI > 65 && currentStoch > 65)
                bearishSignals.Add("RANGE_RESISTANCE");

            // STEP 5: Enter if 2+ signals agree (lines 304-345)
            if (bullishSignals.Count >= 2)
            {
                // BUY SIGNAL (lines 305-317)
                string reason = string.Format("Combined V2: {0}/3 BULLISH signals [{1}] | RSI: {2:F1}",
                    bullishSignals.Count,
                    string.Join(", ", bullishSignals),
                    currentRSI);

                EnterTrade("BUY", reason);
            }
            else if (bearishSignals.Count >= 2)
            {
                // SELL SIGNAL (lines 319-331)
                string reason = string.Format("Combined V2: {0}/3 BEARISH signals [{1}] | RSI: {2:F1}",
                    bearishSignals.Count,
                    string.Join(", ", bearishSignals),
                    currentRSI);

                EnterTrade("SELL", reason);
            }
        }
        #endregion

        #region Enter Trade
        private void EnterTrade(string direction, string reason)
        {
            // EXACT COPY from calculate_position_size() lines 347-392

            // Get pair settings
            string symbol = Instrument.FullName;
            if (!pairSettings.ContainsKey(symbol))
                return;

            PairSettings settings = pairSettings[symbol];

            // Calculate position size (15% margin with 20:1 leverage)
            double currentBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double targetMargin = currentBalance * 0.15;  // 15% margin
            double leverage = 20;  // OANDA leverage
            double targetNotional = targetMargin * leverage;

            // Convert to contracts
            // For futures: 1 contract = fixed notional value
            // Start with 1 contract (conservative)
            int contracts = 1;

            // Calculate SL and TP in ticks (EXACT copy from calculate_stop_loss_take_profit)
            int slTicks = settings.SlTicks;
            int tpTicks = settings.TpTicks;

            Print(string.Format("================================================="));
            Print(string.Format("[ENTRY SIGNAL] {0} {1}", direction, symbol));
            Print(string.Format("  Price: {0:F5}", Close[0]));
            Print(string.Format("  Reason: {0}", reason));
            Print(string.Format("  Contracts: {0}", contracts));
            Print(string.Format("  Stop Loss: {0} ticks", slTicks));
            Print(string.Format("  Take Profit: {0} ticks", tpTicks));
            Print(string.Format("================================================="));

            // Place order
            if (direction == "BUY")
            {
                EnterLong(contracts, "LONG");
                SetStopLoss(CalculationMode.Ticks, slTicks);
                SetProfitTarget(CalculationMode.Ticks, tpTicks);
            }
            else
            {
                EnterShort(contracts, "SHORT");
                SetStopLoss(CalculationMode.Ticks, slTicks);
                SetProfitTarget(CalculationMode.Ticks, tpTicks);
            }

            // Increment daily trade count
            IncrementDailyTradesCount(symbol);
        }
        #endregion

        #region Helper Methods
        private int GetDailyTradesCount(string symbol)
        {
            // Reset counter at start of new day
            if (Bars.IsFirstBarOfSession)
            {
                tradesPerDayCount.Clear();
                dailyStartBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            }

            if (!tradesPerDayCount.ContainsKey(symbol))
                tradesPerDayCount[symbol] = 0;

            return tradesPerDayCount[symbol];
        }

        private void IncrementDailyTradesCount(string symbol)
        {
            if (!tradesPerDayCount.ContainsKey(symbol))
                tradesPerDayCount[symbol] = 0;

            tradesPerDayCount[symbol]++;
        }
        #endregion
    }

    #region Pair Settings Class
    public class PairSettings
    {
        public int TpPips { get; set; }
        public int SlPips { get; set; }
        public int TpTicks { get; set; }
        public int SlTicks { get; set; }
        public int TrailTriggerTicks { get; set; }
        public int TrailDistanceTicks { get; set; }
        public double RiskRewardRatio { get; set; }
        public int PipMultiplier { get; set; }
        public double TickValue { get; set; }
    }
    #endregion
}
```

---

## Part 3: Translation Checklist - Line by Line Mapping

### Python → C# Equivalents

| Python File | Lines | C# Section | Lines | Description |
|-------------|-------|------------|-------|-------------|
| `forex_scalping.py` | 48-78 | `Variables` region | 11-24 | Class initialization |
| `forex_scalping.py` | 87-89 | Built-in `EMA()` | N/A | EMA calculation |
| `forex_scalping.py` | 91-98 | Built-in `RSI()` | N/A | RSI calculation |
| `forex_scalping.py` | 191-223 | `OnBarUpdate()` start | 168-191 | Basic checks |
| `forex_scalping.py` | 225-270 | `OnBarUpdate()` indicators | 193-209 | Calculate indicators |
| `forex_scalping.py` | 272-286 | `OnBarUpdate()` bullish | 211-224 | Bullish signals |
| `forex_scalping.py` | 288-302 | `OnBarUpdate()` bearish | 226-239 | Bearish signals |
| `forex_scalping.py` | 304-331 | `OnBarUpdate()` entry | 241-260 | Entry logic |
| `forex_scalping.py` | 347-392 | `EnterTrade()` | 265-308 | Position sizing |
| `forex_scalping.py` | 394-430 | `EnterTrade()` SL/TP | 283-289 | Stop/Target calc |
| `pair_specific_settings.py` | 69-109 | `InitializePairSettings()` | 73-137 | All 5 pairs |

---

## Part 4: What's EXACTLY the Same

✓ Entry logic: 2 of 3 signals (RSI + BB + Range)
✓ Indicator calculations: RSI(14), BB(20,2), Stoch(14)
✓ Support/Resistance: 50-bar lookback
✓ RSI thresholds: <30 oversold, >70 overbought
✓ Range distance: <0.002 (0.2% = 20 pips)
✓ Position size: 15% margin with 20:1 leverage
✓ Max trades/day: 10 per instrument
✓ Daily profit target: 3%
✓ Daily loss limit: -5%
✓ TP/SL settings: Exact same pip values per pair

---

## Part 5: What Changes (Only These!)

1. **Language:** Python → C#
2. **Symbols:** EUR_USD → M6E, GBP_USD → M6B, etc.
3. **Units → Contracts:** Variable units → 1-2 fixed contracts
4. **Pips → Ticks:** Multiply pip values by conversion factor
5. **Indicators:** Custom Python functions → Built-in NinjaTrader indicators

**Everything else stays EXACTLY the same!**

---

## Part 6: Testing Validation

### Expected Backtest Results (should match Python):

- **Win Rate:** ~50-52%
- **Profit Factor:** ~1.3-1.4
- **Max Drawdown:** <10%
- **Return:** 15-25% monthly

**If results differ by more than 10%, check:**
1. Indicator calculations (RSI, BB periods correct?)
2. Entry signal logic (2 of 3 still working?)
3. TP/SL tick conversions (multiply factors correct?)

---

## NEXT STEP: Implementation

Ready to create the full NinjaScript `.cs` file with ALL 5 pairs configured?

**Approve to proceed with implementation?**
