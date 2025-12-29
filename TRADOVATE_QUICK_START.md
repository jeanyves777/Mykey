# Tradovate Trading System - Quick Start Guide

## ✅ What's Been Built

Your complete Tradovate trading system is ready! Here's what's included:

### Core Components:
1. **Tradovate Client** - REST + WebSocket API client
2. **Configuration Files** - Symbols, settings, FundedNext rules
3. **Test Script** - Connection validation

### What You Need to Finish:
- Create Tradovate account
- Get API credentials
- Fill in config file
- Test connection

---

## 🚀 Step-by-Step Setup (10 minutes)

### Step 1: Create Tradovate Demo Account (5 min)

1. Go to **https://trader.tradovate.com/#/signup**

2. Fill in the form:
   - First/Last Name
   - Email
   - Phone (optional)
   - Choose **"Open a FREE Simulated Account"**

3. Verify your email

4. Log in to Tradovate

**✅ You now have a free demo account with $50,000 simulated funds!**

---

### Step 2: Get API Credentials (2 min)

1. In Tradovate, click your **username** (top right)

2. Go to **Settings → API**

3. Click **"Generate New API Key"**

4. **IMPORTANT:** Copy and save:
   - Your **username** (from top right)
   - Your **password** (the one you just created)
   - The **API Key** (long string that appears)

**⚠️ Save these somewhere safe - you'll need them!**

---

### Step 3: Fill in Configuration (1 min)

1. Open this file:
   ```
   trading_system/Tradovate/config/tradovate_config.py
   ```

2. Find these lines (around line 8-10):
   ```python
   TRADOVATE_DEMO_USERNAME = ""  # Your Tradovate demo username
   TRADOVATE_DEMO_PASSWORD = ""  # Your Tradovate demo password
   TRADOVATE_DEMO_API_KEY = ""   # Get this from Tradovate Settings > API
   ```

3. Fill them in:
   ```python
   TRADOVATE_DEMO_USERNAME = "yourusername"
   TRADOVATE_DEMO_PASSWORD = "yourpassword"
   TRADOVATE_DEMO_API_KEY = "your-long-api-key-here"
   ```

4. Save the file

---

### Step 4: Test Connection (1 min)

1. Open terminal in your project folder

2. Run:
   ```bash
   python run_tradovate_test_connection.py
   ```

3. You should see:
   ```
   ✅ CONNECTION TEST PASSED!

   Account Balance:
     Cash Balance:      $50,000.00
     Equity:            $50,000.00
     Margin Available:  $50,000.00
   ```

**If it works → You're ready!**

**If it fails → Check:**
- Credentials are correct
- No typos in username/password
- API key copied correctly
- Internet connection working

---

## 📊 What Symbols Are Available?

Your system is configured for these forex futures:

| Symbol | Name | OANDA Equivalent |
|--------|------|------------------|
| **M6EU2** | Micro Euro | EUR/USD |
| **M6BU2** | Micro British Pound | GBP/USD |
| **MSFU2** | Micro Swiss Franc | USD/CHF |
| MJYU2 | Micro Japanese Yen | USD/JPY |
| MCDU2 | Micro Canadian Dollar | USD/CAD |
| M6AU2 | Micro Australian Dollar | AUD/USD |

**Note:** Not all symbols may be available in demo. The test script will show which ones work.

---

## 🎯 Your Strategy (Same as OANDA!)

The system uses your **proven Combined V2 strategy**:

### Entry Rules:
Enter when **ANY 2 of 3** signals agree:

1. **RSI Signal**
   - Buy: RSI < 30 (oversold)
   - Sell: RSI > 70 (overbought)

2. **Bollinger Band Signal**
   - Buy: Price touches lower band
   - Sell: Price touches upper band

3. **Range Scalping Signal**
   - Buy: Price near support + RSI < 35 + Stoch < 35
   - Sell: Price near resistance + RSI > 65 + Stoch > 65

### Exit Rules:
- **Take Profit:** Pair-specific (15-30 pips in ticks)
- **Stop Loss:** Pair-specific (12-25 pips in ticks)
- Same TP/SL as your OANDA strategy!

### Position Sizing:
- Fixed 1 contract per trade
- Max 5 concurrent positions

### FundedNext Rules (Automated):
- ✓ Max Loss Limit: $1,000 (EOD Trailing)
- ✓ Profit Target: $1,250
- ✓ Consistency Rule: 40% max per day
- ✓ Daily Loss Limit: -$500

---

## 📈 Next Steps After Testing

### Option 1: Run Backtest
Test the strategy on historical data:
```bash
python run_tradovate_backtest.py
```

### Option 2: Run Paper Trading
Test live with simulated money:
```bash
python run_tradovate_paper.py
```

### Option 3: Prepare for FundedNext
1. Run paper trading for 1 week
2. Verify performance matches OANDA (48-55% WR)
3. Open FundedNext $25K challenge account
4. Connect Tradovate to FundedNext
5. Run live trading with `run_tradovate_live.py`

---

## 🔧 Troubleshooting

### "Authentication failed"
- Double-check username/password
- Make sure you're using DEMO credentials
- API key might have expired - generate new one

### "No data for symbol"
- Symbol might not be available in demo
- Try different symbols (M6EU2 should always work)
- Contact Tradovate support

### "Connection refused"
- Check internet connection
- Tradovate might be down (check tradovate.com/status)
- Firewall blocking connection

### "Module not found"
Install missing dependencies:
```bash
pip install requests websocket-client pandas numpy pytz
```

---

## 💡 Tips

### Symbol Expiration
Futures contracts expire! Update symbols when needed:
- **U** = September (e.g., M6EU2 = Sep 2025)
- **Z** = December
- **H** = March
- **M** = June

### Risk Management
Start small:
- Begin with 1-2 pairs only
- Test for 1 week on demo
- Verify all FundedNext rules working
- Then scale to all 5 pairs

### Data Quality
Tradovate historical data is:
- ✓ Free with account
- ✓ Real CME futures data
- ✓ 15-minute bars available
- ✓ Perfect for backtesting

---

## 📞 Support

**Tradovate Issues:**
- Help: https://www.tradovate.com/support
- Status: https://www.tradovate.com/status
- Phone: +1-312-248-0500

**Trading System Issues:**
- Check TRADOVATE_IMPLEMENTATION_SUMMARY.md
- Review error messages in terminal
- Verify all config files filled correctly

---

## ✅ Checklist

Before running your first live trade:

- [ ] Tradovate demo account created
- [ ] API credentials obtained
- [ ] Config file filled in
- [ ] Connection test passed
- [ ] Backtest completed successfully
- [ ] Paper trading tested for 1 week
- [ ] Performance verified (45-55% WR)
- [ ] FundedNext rules compliance checked
- [ ] Ready for challenge!

---

**Status: Run `python run_tradovate_test_connection.py` to begin!**
