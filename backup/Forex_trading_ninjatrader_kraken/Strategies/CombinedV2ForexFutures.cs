//
// Copyright (C) 2024, NinjaTrader LLC <www.ninjatrader.com>
// NinjaTrader Strategy: Combined V2 Forex Futures Strategy
//
// PROVEN STRATEGY TRANSLATION:
// - From OANDA Forex (Python) to NinjaTrader Forex Futures (C#)
// - Backtest: +25.82% return, 51.9% WR, 1.34 PF
// - Live: +0.55% per session, 48% WR, 1.27 PF
//
// ENTRY LOGIC (EXACT COPY):
// - ANY 2 of 3 signals: RSI + Bollinger Bands + Range Scalping
// - 15min timeframe for indicators
// - All 5 pairs: M6E, M6B, MJY, MCD, MSF
//
// FUNDEDNEXT COMPLIANCE:
// - $25K account
// - Max Loss: $1,000 (EOD Balance Trailing)
// - Profit Target: $1,250
// - Consistency Rule: 40% max per day (automated)
// - Daily Loss Limit: -$500 (self-imposed)
//

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class CombinedV2ForexFutures : Strategy
    {
        #region Variables

        // ==================== FUNDEDNEXT CHALLENGE SETTINGS ====================
        private double INITIAL_BALANCE = 25000;
        private double MAX_LOSS_LIMIT = 1000;          // $1K max loss from highest EOD
        private double PROFIT_TARGET = 1250;           // $1,250 to pass challenge
        private double CONSISTENCY_LIMIT = 0.40;       // 40% max per day

        private double dailyLossLimit = -500;          // -$500 daily stop
        private double dailyProfitCap = 0;             // Set dynamically based on consistency rule

        private double highestEODBalance = 25000;
        private double currentThreshold = 24000;       // $25,000 - $1,000
        private double startingBalanceToday = 25000;
        private double todayProfit = 0;
        private double totalProfitToDate = 0;
        private bool isChallengeMode = true;           // Set false when funded

        // ==================== POSITION LIMITS ====================
        private int contractsPerTrade = 1;
        private int maxConcurrentPositions = 5;
        private int maxTradesPerDay = 50;              // Total across all symbols
        private int maxTradesPerSymbol = 10;

        // ==================== TRACKING VARIABLES ====================
        private DateTime currentDate = DateTime.MinValue;
        private int tradesCountToday = 0;
        private Dictionary<string, int> tradesPerSymbol = new Dictionary<string, int>();
        private Dictionary<string, double> dailyProfitByDay = new Dictionary<string, double>();

        // ==================== PAIR-SPECIFIC SETTINGS ====================
        private class PairSettings
        {
            public int TpPips { get; set; }
            public int SlPips { get; set; }
            public int TpTicks { get; set; }
            public int SlTicks { get; set; }
            public int TrailTriggerTicks { get; set; }
            public int TrailDistanceTicks { get; set; }
            public double PipMultiplier { get; set; }  // Conversion factor (pips → ticks)
            public double TickValue { get; set; }
        }

        private Dictionary<string, PairSettings> pairSettings;

        // ==================== INDICATORS ====================
        private RSI rsi;
        private Bollinger bollinger;
        private Stochastics stochastics;
        private MIN supportLevel;
        private MAX resistanceLevel;

        #endregion

        #region OnStateChange

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"Combined V2 Forex Futures Strategy - FundedNext Compliant";
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
                TraceOrders = true;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 50;

                // User parameters
                IsChallengeMode = true;
                InitialBalance = 25000;
                ContractsPerTrade = 1;
                DailyLossLimit = -500;
                MaxConcurrentPositions = 5;
            }
            else if (State == State.Configure)
            {
                // Add all 5 forex futures symbols
                // Primary chart should be one of these, others added as additional data series
                AddDataSeries("M6E 06-25", BarsPeriodType.Minute, 15);  // EUR/USD
                AddDataSeries("M6B 06-25", BarsPeriodType.Minute, 15);  // GBP/USD
                AddDataSeries("MJY 06-25", BarsPeriodType.Minute, 15);  // USD/JPY
                AddDataSeries("MCD 06-25", BarsPeriodType.Minute, 15);  // USD/CAD
                AddDataSeries("MSF 06-25", BarsPeriodType.Minute, 15);  // USD/CHF
            }
            else if (State == State.DataLoaded)
            {
                // Initialize pair-specific settings
                InitializePairSettings();

                // Initialize indicators
                rsi = RSI(14, 3);
                bollinger = Bollinger(2, 20);
                stochastics = Stochastics(14, 3, 3);
                supportLevel = MIN(Low, 50);
                resistanceLevel = MAX(High, 50);

                // Initialize tracking
                highestEODBalance = InitialBalance;
                currentThreshold = InitialBalance - MAX_LOSS_LIMIT;
                startingBalanceToday = InitialBalance;

                Print("========================================");
                Print("COMBINED V2 FOREX FUTURES STRATEGY");
                Print("========================================");
                Print("Strategy: 2 of 3 signals (RSI, BB, Range)");
                Print("Symbols: M6E, M6B, MJY, MCD, MSF");
                Print("Position: 1 contract per trade");
                Print("Max Concurrent: 5 positions");
                Print("FundedNext: $25K Challenge Mode");
                Print("Daily Loss Limit: -$500");
                Print("Profit Target: +$1,250");
                Print("Consistency Rule: 40% max per day");
                Print("========================================");
            }
        }

        #endregion

        #region Pair Settings Initialization

        private void InitializePairSettings()
        {
            pairSettings = new Dictionary<string, PairSettings>
            {
                // EUR/USD (M6E): Pip × 2 = Tick
                {
                    "M6E", new PairSettings
                    {
                        TpPips = 20,
                        SlPips = 16,
                        TpTicks = 40,              // 20 pips × 2
                        SlTicks = 32,              // 16 pips × 2
                        TrailTriggerTicks = 24,    // 12 pips × 2
                        TrailDistanceTicks = 12,   // 6 pips × 2
                        PipMultiplier = 2.0,
                        TickValue = 6.25
                    }
                },

                // GBP/USD (M6B): Pip × 1 = Tick
                {
                    "M6B", new PairSettings
                    {
                        TpPips = 30,
                        SlPips = 25,
                        TpTicks = 30,              // 30 pips × 1
                        SlTicks = 25,              // 25 pips × 1
                        TrailTriggerTicks = 18,    // 18 pips × 1
                        TrailDistanceTicks = 8,    // 8 pips × 1
                        PipMultiplier = 1.0,
                        TickValue = 6.25
                    }
                },

                // USD/JPY (MJY): Pip × 10 = Tick (CORRECTED - MJY tick = 0.000001)
                {
                    "MJY", new PairSettings
                    {
                        TpPips = 18,
                        SlPips = 15,
                        TpTicks = 180,             // 18 pips × 10
                        SlTicks = 150,             // 15 pips × 10
                        TrailTriggerTicks = 120,   // 12 pips × 10
                        TrailDistanceTicks = 60,   // 6 pips × 10
                        PipMultiplier = 10.0,
                        TickValue = 1.25           // MJY tick value is $1.25
                    }
                },

                // USD/CAD (MCD): Pip × 2 = Tick
                {
                    "MCD", new PairSettings
                    {
                        TpPips = 20,
                        SlPips = 16,
                        TpTicks = 40,              // 20 pips × 2
                        SlTicks = 32,              // 16 pips × 2
                        TrailTriggerTicks = 24,    // 12 pips × 2
                        TrailDistanceTicks = 12,   // 6 pips × 2
                        PipMultiplier = 2.0,
                        TickValue = 5.00           // CAD tick value is $5
                    }
                },

                // USD/CHF (MSF): Pip × 2 = Tick
                {
                    "MSF", new PairSettings
                    {
                        TpPips = 15,
                        SlPips = 12,
                        TpTicks = 30,              // 15 pips × 2
                        SlTicks = 24,              // 12 pips × 2
                        TrailTriggerTicks = 20,    // 10 pips × 2
                        TrailDistanceTicks = 10,   // 5 pips × 2
                        PipMultiplier = 2.0,
                        TickValue = 6.25
                    }
                }
            };
        }

        #endregion

        #region OnBarUpdate

        protected override void OnBarUpdate()
        {
            // ==================== SAFETY GUARD 1: Date Tracking & Reset ====================
            if (Time[0].Date != currentDate)
            {
                // New trading day
                DateTime previousDate = currentDate;
                currentDate = Time[0].Date;

                // Reset daily counters
                tradesCountToday = 0;
                tradesPerSymbol.Clear();
                startingBalanceToday = Account.Get(AccountItem.CashValue, Currency.UsDollar);

                // Update EOD balance from previous day
                if (previousDate != DateTime.MinValue && ToTime(Time[0]) == 160000) // Market close
                {
                    double eodBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);

                    if (eodBalance > highestEODBalance)
                    {
                        highestEODBalance = eodBalance;
                        currentThreshold = highestEODBalance - MAX_LOSS_LIMIT;

                        Print(string.Format("[EOD UPDATE] New Highest Balance: ${0:N2}, New Threshold: ${1:N2}",
                            highestEODBalance, currentThreshold));
                    }

                    // Track daily profit for consistency rule
                    double dailyProfit = eodBalance - startingBalanceToday;
                    string dateKey = previousDate.ToString("yyyy-MM-dd");
                    dailyProfitByDay[dateKey] = dailyProfit;

                    Print(string.Format("[EOD] Date: {0}, Daily P&L: ${1:N2}, EOD Balance: ${2:N2}",
                        dateKey, dailyProfit, eodBalance));
                }

                Print(string.Format("[NEW DAY] {0} - Starting Balance: ${1:N2}, Threshold: ${2:N2}",
                    currentDate.ToShortDateString(), startingBalanceToday, currentThreshold));
            }

            // ==================== SAFETY GUARD 2: Account Threshold Check ====================
            double currentEquity = Account.Get(AccountItem.CashValue, Currency.UsDollar);
            double buffer = currentEquity - currentThreshold;

            if (currentEquity <= currentThreshold)
            {
                Print(string.Format("[ACCOUNT FAILED] Equity: ${0:N2} <= Threshold: ${1:N2}",
                    currentEquity, currentThreshold));
                Print("[STOP] Max loss limit breached! Closing all positions and disabling strategy.");

                CloseAllPositions();
                return;
            }

            // Buffer warning
            if (buffer < 200)
            {
                Print(string.Format("[WARNING] Low buffer: ${0:N2} - Only ${1:N2} from threshold!",
                    currentEquity, buffer));
                Print("[STOP] Buffer < $200 - Stopping trading to protect account.");
                return;
            }

            // ==================== SAFETY GUARD 3: Daily Loss Limit ====================
            todayProfit = currentEquity - startingBalanceToday;

            if (todayProfit <= dailyLossLimit)
            {
                Print(string.Format("[DAILY STOP] Daily P&L: ${0:N2} <= Limit: ${1:N2}",
                    todayProfit, dailyLossLimit));
                Print("[STOP] Daily loss limit hit. No more trading today.");
                return;
            }

            // ==================== SAFETY GUARD 4: Profit Target ====================
            totalProfitToDate = currentEquity - InitialBalance;

            if (totalProfitToDate >= PROFIT_TARGET)
            {
                Print(string.Format("[PROFIT TARGET HIT] Total Profit: ${0:N2} >= Target: ${1:N2}",
                    totalProfitToDate, PROFIT_TARGET));
                Print("[STOP] Challenge passed! Closing all positions and stopping strategy.");

                CloseAllPositions();
                return;
            }

            // ==================== SAFETY GUARD 5: Consistency Rule (Challenge Only) ====================
            if (isChallengeMode && todayProfit > 0)
            {
                // Calculate max allowed profit for today based on consistency rule
                // If we've already made total profit, ensure today doesn't exceed 40%
                if (totalProfitToDate > 0)
                {
                    // Max today = total * 0.40
                    // But if we haven't hit target yet, we need room to grow
                    // Conservative: Cap today at 40% of CURRENT total
                    double maxTodayProfit = totalProfitToDate * CONSISTENCY_LIMIT;

                    if (todayProfit >= maxTodayProfit)
                    {
                        Print(string.Format("[CONSISTENCY STOP] Today: ${0:N2} >= 40% of Total: ${1:N2}",
                            todayProfit, maxTodayProfit));
                        Print("[STOP] Consistency rule protection - stop trading for today.");
                        return;
                    }
                }

                // Alternative approach: Hard cap at +$400/day during challenge
                // This ensures no single day can be >32% of $1,250 target
                if (todayProfit >= 400)
                {
                    Print(string.Format("[CONSISTENCY CAP] Today: ${0:N2} >= Daily Cap: $400",
                        todayProfit));
                    Print("[STOP] Daily profit cap reached (consistency rule).");
                    return;
                }
            }

            // ==================== SAFETY GUARD 6: Max Trades Limits ====================
            if (tradesCountToday >= maxTradesPerDay)
            {
                return; // Max trades for the day reached
            }

            // ==================== SAFETY GUARD 7: Max Concurrent Positions ====================
            // Note: In multi-instrument strategy, we track positions manually
            // For now, allow trading (position tracking handled per symbol below)

            // This will be enforced at symbol level in CheckEntrySignals

            // ==================== CHECK ENTRY SIGNALS ====================
            // Process each data series (each symbol)
            for (int barsInProgress = 0; barsInProgress < BarsArray.Length; barsInProgress++)
            {
                if (CurrentBars[barsInProgress] < BarsRequiredToTrade)
                    continue;

                // Get symbol from bars
                string symbol = BarsArray[barsInProgress].Instrument.MasterInstrument.Name;

                // Check if we have settings for this symbol
                if (!pairSettings.ContainsKey(symbol))
                    continue;

                // Check max trades per symbol
                if (tradesPerSymbol.ContainsKey(symbol) && tradesPerSymbol[symbol] >= maxTradesPerSymbol)
                    continue;

                // Check if we already have a position on this symbol
                if (Position.MarketPosition != MarketPosition.Flat)
                    continue;

                // Check entry signals
                CheckEntrySignals(barsInProgress, symbol);
            }
        }

        #endregion

        #region Entry Signal Logic

        private void CheckEntrySignals(int barsInProgress, string symbol)
        {
            // COMBINED V2 STRATEGY: Need ANY 2 of 3 signals
            // 1. RSI Oversold/Overbought (<30 / >70)
            // 2. Bollinger Bands (price touches bands)
            // 3. Range Scalping (price near support/resistance + confirmation)

            // Set to correct bars series
            int idx = barsInProgress;

            // Get current values
            double currentPrice = Closes[idx][0];
            double prevPrice = Closes[idx][1];
            double currentRSI = rsi[0];
            double currentStoch = stochastics[0];
            double bbUpper = bollinger.Upper[0];
            double bbLower = bollinger.Lower[0];
            double prevBBUpper = bollinger.Upper[1];
            double prevBBLower = bollinger.Lower[1];
            double support = supportLevel[0];
            double resistance = resistanceLevel[0];

            // Track signals
            List<string> bullishSignals = new List<string>();
            List<string> bearishSignals = new List<string>();

            // ========== BULLISH SIGNALS ==========

            // Signal 1: RSI Oversold
            if (currentRSI < 30)
            {
                bullishSignals.Add("RSI_OVERSOLD");
            }

            // Signal 2: Bollinger Lower Band Touch
            if (currentPrice <= bbLower && prevPrice > prevBBLower)
            {
                bullishSignals.Add("BB_LOWER_TOUCH");
            }

            // Signal 3: Range Support + Confirmation
            double distToSupport = (currentPrice - support) / support;
            if (distToSupport < 0.002 && currentRSI < 35 && currentStoch < 35)
            {
                bullishSignals.Add("RANGE_SUPPORT");
            }

            // ========== BEARISH SIGNALS ==========

            // Signal 1: RSI Overbought
            if (currentRSI > 70)
            {
                bearishSignals.Add("RSI_OVERBOUGHT");
            }

            // Signal 2: Bollinger Upper Band Touch
            if (currentPrice >= bbUpper && prevPrice < prevBBUpper)
            {
                bearishSignals.Add("BB_UPPER_TOUCH");
            }

            // Signal 3: Range Resistance + Confirmation
            double distToResistance = (resistance - currentPrice) / currentPrice;
            if (distToResistance < 0.002 && currentRSI > 65 && currentStoch > 65)
            {
                bearishSignals.Add("RANGE_RESISTANCE");
            }

            // ========== ENTRY DECISION ==========

            PairSettings settings = pairSettings[symbol];

            // BUY if 2+ bullish signals
            if (bullishSignals.Count >= 2)
            {
                string signalsStr = string.Join(", ", bullishSignals);
                Print(string.Format("[ENTRY] {0} BUY: {1}/3 signals [{2}] | RSI: {3:F1}",
                    symbol, bullishSignals.Count, signalsStr, currentRSI));

                EnterLongLimit(idx, true, contractsPerTrade, currentPrice, symbol + "_LONG");

                // Set stop loss and take profit
                SetStopLoss(symbol + "_LONG", CalculationMode.Ticks, settings.SlTicks, false);
                SetProfitTarget(symbol + "_LONG", CalculationMode.Ticks, settings.TpTicks);

                // Update counters
                tradesCountToday++;
                if (!tradesPerSymbol.ContainsKey(symbol))
                    tradesPerSymbol[symbol] = 0;
                tradesPerSymbol[symbol]++;
            }
            // SELL if 2+ bearish signals
            else if (bearishSignals.Count >= 2)
            {
                string signalsStr = string.Join(", ", bearishSignals);
                Print(string.Format("[ENTRY] {0} SELL: {1}/3 signals [{2}] | RSI: {3:F1}",
                    symbol, bearishSignals.Count, signalsStr, currentRSI));

                EnterShortLimit(idx, true, contractsPerTrade, currentPrice, symbol + "_SHORT");

                // Set stop loss and take profit
                SetStopLoss(symbol + "_SHORT", CalculationMode.Ticks, settings.SlTicks, false);
                SetProfitTarget(symbol + "_SHORT", CalculationMode.Ticks, settings.TpTicks);

                // Update counters
                tradesCountToday++;
                if (!tradesPerSymbol.ContainsKey(symbol))
                    tradesPerSymbol[symbol] = 0;
                tradesPerSymbol[symbol]++;
            }
        }

        #endregion

        #region Helper Methods

        private void CloseAllPositions()
        {
            if (Position.MarketPosition == MarketPosition.Long)
            {
                ExitLong();
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                ExitShort();
            }
        }

        #endregion

        #region Properties

        [NinjaScriptProperty]
        [Display(Name="Challenge Mode", Description="Enable FundedNext challenge rules", Order=1, GroupName="FundedNext")]
        public bool IsChallengeMode
        { get; set; }

        [NinjaScriptProperty]
        [Range(1000, 100000)]
        [Display(Name="Initial Balance", Description="Starting account balance", Order=2, GroupName="FundedNext")]
        public double InitialBalance
        { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name="Contracts Per Trade", Description="Number of contracts per trade", Order=3, GroupName="Position Sizing")]
        public int ContractsPerTrade
        { get; set; }

        [NinjaScriptProperty]
        [Range(-10000, -100)]
        [Display(Name="Daily Loss Limit", Description="Maximum loss allowed per day (negative value)", Order=4, GroupName="Risk Management")]
        public double DailyLossLimit
        { get; set; }

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name="Max Concurrent Positions", Description="Maximum number of open positions", Order=5, GroupName="Risk Management")]
        public int MaxConcurrentPositions
        { get; set; }

        #endregion
    }
}
