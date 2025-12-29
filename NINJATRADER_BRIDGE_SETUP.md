# NinjaTrader Bridge - Complete Setup Guide

## 🎯 What This Does

**OANDA (Python) → NinjaTrader (Futures)**

Your proven OANDA strategy generates signals → Bridge sends them to NinjaTrader → NinjaTrader executes on futures!

**Why This is PERFECT:**
- ✅ Keep your proven OANDA strategy (48% WR, working!)
- ✅ Execute on real futures (works with FundedNext)
- ✅ ALL FundedNext rules automated
- ✅ No monthly fees
- ✅ Works with ANY prop firm that accepts NinjaTrader

---

## 📦 What's Been Built

### 1. C# NinjaTrader Bridge (`NinjaTraderBridge.cs`)
- Listens for signals from Python (port 8888)
- Executes orders on NinjaTrader via API
- Handles TP/SL placement
- Position tracking

### 2. Python Signal Generator (`oanda_to_ninjatrader_bridge.py`)
- Runs your OANDA strategy
- ALL FundedNext rules automated:
  - ✅ EOD Balance Trailing
  - ✅ Buffer Protection ($200 min)
  - ✅ Daily Loss Limit (-$500)
  - ✅ Profit Target (+$1,250)
  - ✅ Consistency Rule (40%/day, $400 cap)
  - ✅ Position Limits (5 max, 1 contract each)
  - ✅ Trade Limits (50/day, 10/symbol)
- Exact TP/SL per symbol (in ticks)
- Sends signals to C# bridge

---

## 🚀 Setup Instructions

### STEP 1: Install NinjaTrader (5 minutes)

1. Download: https://ninjatrader.com/GetStarted
2. Install (free)
3. Create account (free)
4. Start 14-day free trial of market data

**✅ You now have NinjaTrader with 14 days of free forex futures data!**

---

### STEP 2: Compile C# Bridge (10 minutes)

**Option A: Using Visual Studio (Recommended)**

1. Download Visual Studio Community (free): https://visualstudio.microsoft.com/downloads/

2. Install with ".NET desktop development" workload

3. Create new project:
   - File → New → Project
   - Choose "Console App (.NET Framework)"
   - Target Framework: **.NET Framework 4.8**
   - Name: NinjaTraderBridge

4. Add NinjaTrader.Client.dll reference:
   - Right-click "References" in Solution Explorer
   - Add Reference → Browse
   - Navigate to: `C:\Program Files\NinjaTrader 8\bin\NinjaTrader.Client.dll`
   - Click Add

5. Install Newtonsoft.Json:
   - Tools → NuGet Package Manager → Package Manager Console
   - Run: `Install-Package Newtonsoft.Json`

6. Copy code:
   - Open `trading_system/NinjaTrader_Bridge/NinjaTraderBridge.cs`
   - Copy all code
   - Paste into Program.cs in Visual Studio

7. Build:
   - Build → Build Solution (or press F6)
   - Find exe: `bin\Debug\NinjaTraderBridge.exe`

**Option B: Using VS Code (Alternative)**

1. Install VS Code + C# extension
2. Install .NET Framework 4.8 SDK
3. Same steps as above

---

### STEP 3: Test the Bridge (5 minutes)

1. **Start NinjaTrader 8**
   - Open NinjaTrader
   - Connect to Sim101 account (demo)

2. **Run the C# Bridge**
   ```
   NinjaTraderBridge.exe
   ```

   You should see:
   ```
   ✓ Connected to NinjaTrader 8
   ✓ Listening on port 8888 for signals
   Status: ACTIVE
   ```

3. **Test with Python**
   ```bash
   python
   ```

   ```python
   import socket
   import json

   # Test signal
   signal = {
       "Action": "ENTRY",
       "Symbol": "M6E",
       "Side": "BUY",
       "Quantity": 1,
       "EntryPrice": 1.05000,
       "StopLoss": 1.04800,
       "TakeProfit": 1.05200,
       "Timestamp": "2025-01-13T10:00:00"
   }

   # Send to bridge
   sock = socket.socket()
   sock.connect(('localhost', 8888))
   sock.sendall(json.dumps(signal).encode())
   response = sock.recv(1024)
   print(response)  # Should print: b'OK'
   sock.close()
   ```

4. **Check NinjaTrader**
   - You should see an order placed!
   - Check Orders tab in NinjaTrader

**✅ If you see the order, the bridge is working!**

---

## 🎯 Using with Your OANDA Strategy

### Current State

The Python bridge (`oanda_to_ninjatrader_bridge.py`) has:
- ✅ ALL FundedNext rules
- ✅ Exact TP/SL settings
- ✅ Signal sending to NinjaTrader
- ⏳ Needs integration with your OANDA strategy

### Integration (I can do this!)

I need to connect the bridge to your existing OANDA strategy.

**Two options:**

**Option 1: Modify Existing OANDA Live Script**

Your file: `trading_system/run_crypto_live_trading.py` (or forex equivalent)

I'll modify it to:
1. Keep running OANDA strategy
2. When signal generated → Send to NinjaTrader bridge
3. Bridge executes on futures

**Option 2: Create New Hybrid Script**

New file that:
1. Uses OANDA for market data + signals
2. Uses NinjaTrader for execution only
3. All FundedNext rules in Python (already done!)

---

## 📊 How It Works (End-to-End)

```
[OANDA Strategy]
      ↓
  (Python detects signal: RSI + BB + Range = BUY EUR/USD)
      ↓
[Python Bridge]
  - Check FundedNext rules ✓
  - Calculate TP/SL in ticks ✓
  - Convert EUR_USD → M6E
      ↓
  (Send JSON signal via TCP)
      ↓
[C# Bridge (port 8888)]
  - Receive signal
  - Parse JSON
      ↓
[NinjaTrader API]
  - Place market order: BUY 1 M6E
  - Place stop loss order
  - Place take profit order
      ↓
[Futures Exchange (CME)]
  - Order filled!
  - Real futures position
```

---

## ✅ FundedNext Rules (ALL AUTOMATED)

### Rule 1: EOD Balance Trailing
```python
if self.current_balance <= self.current_threshold:
    print("Account failed!")
    return False
```

### Rule 2: Buffer Protection
```python
buffer = self.current_balance - self.current_threshold
if buffer < 200:
    print("Buffer too low, stop trading")
    return False
```

### Rule 3: Daily Loss Limit
```python
today_profit = self.current_balance - self.starting_balance_today
if today_profit <= -500:
    print("Daily loss limit hit")
    return False
```

### Rule 4: Profit Target
```python
if self.total_profit >= 1250:
    print("Challenge passed!")
    return False
```

### Rule 5: Consistency Rule
```python
if self.is_challenge_mode and today_profit > 0:
    max_today = total_profit * 0.40
    if today_profit >= max_today or today_profit >= 400:
        print("Consistency rule: stop for today")
        return False
```

### Rule 6: Position Limits
```python
if len(self.open_positions) >= 5:
    return False

if self.trades_per_symbol[symbol] >= 10:
    return False
```

All rules checked BEFORE sending signal to NinjaTrader!

---

## 📋 Exact TP/SL Settings

```python
PAIR_SETTINGS = {
    'M6E': {  # EUR/USD
        'tp_pips': 20, 'sl_pips': 16,
        'tp_ticks': 40, 'sl_ticks': 32,
        'tick_size': 0.00005, 'tick_value': 6.25
    },
    'M6B': {  # GBP/USD
        'tp_pips': 30, 'sl_pips': 25,
        'tp_ticks': 30, 'sl_ticks': 25,
        'tick_size': 0.0001, 'tick_value': 6.25
    },
    'MJY': {  # USD/JPY
        'tp_pips': 18, 'sl_pips': 15,
        'tp_ticks': 180, 'sl_ticks': 150,
        'tick_size': 0.000001, 'tick_value': 1.25
    },
    'MCD': {  # USD/CAD
        'tp_pips': 20, 'sl_pips': 16,
        'tp_ticks': 40, 'sl_ticks': 32,
        'tick_size': 0.00005, 'tick_value': 5.00
    },
    'MSF': {  # USD/CHF
        'tp_pips': 15, 'sl_pips': 12,
        'tp_ticks': 30, 'sl_ticks': 24,
        'tick_size': 0.00005, 'tick_value': 6.25
    }
}
```

**All calculations automated!**

---

## 🎯 Next Steps

### OPTION 1: I Finish the Integration (Recommended)

Tell me which OANDA script you want to use as the signal generator, and I'll:

1. Modify it to send signals to NinjaTrader
2. Keep all your existing strategy logic
3. Add FundedNext rules
4. Ready to test!

**Takes ~30 minutes**

### OPTION 2: You Compile and Test First

1. Compile the C# bridge following Step 2
2. Test with the Python test script (Step 3)
3. Once working, let me know
4. I'll integrate with your OANDA strategy

---

## 💰 Costs

- NinjaTrader: **FREE**
- Market Data Trial: **FREE** (14 days)
- C# Bridge: **FREE** (I built it)
- Python Integration: **FREE** (I'll do it)

**After 14-day trial:**
- Kinetick Data: $60/month (or use free sim data)
- OR: Use with FundedNext (they provide data)

---

## ✅ Checklist

- [ ] Download & install NinjaTrader
- [ ] Download & install Visual Studio
- [ ] Compile C# bridge
- [ ] Test bridge connection
- [ ] Let me know to integrate with OANDA strategy
- [ ] Test on sim for 1 week
- [ ] Apply for FundedNext challenge
- [ ] PROFIT! 🚀

---

**Ready to proceed? Let me know if you want me to finish the integration!**
