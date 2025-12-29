# Quick Test Instructions - NinjaTrader Bridge

## ✅ What's Ready

The bridge is **compiled and ready** to test with NinjaTrader!

---

## 🚀 Test in 3 Steps (5 minutes)

### STEP 1: Start NinjaTrader (1 min)

1. Go to https://account.ninjatrader.com/ (the link you sent)
2. Download and install **NinjaTrader 8** if you haven't already
3. Open NinjaTrader 8
4. It will automatically connect to **Sim101** account (free simulator)

### STEP 2: Start the Bridge (30 sec)

Open PowerShell and run:

```powershell
cd C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge
.\NinjaTraderBridge.exe
```

**Expected output:**
```
================================================================================
NINJATRADER API BRIDGE - SIGNAL RECEIVER
================================================================================
✓ Connected to NinjaTrader 8
✓ Listening on port 8888 for signals
Status: ACTIVE
================================================================================
```

**Leave this window open!**

### STEP 3: Test Connection (30 sec)

In a **new** PowerShell window:

```powershell
cd C:\Users\Jean-Yves\thevolumeainative
python trading_system\NinjaTrader_Bridge\run_strategy_with_bridge.py
```

This will send a test signal to verify everything works!

---

## ✅ What to Expect

### In the Bridge Window:
```
[18:00:00] Received signal:
  Action: ENTRY
  Symbol: M6E
  Side: BUY
✓ Order placed: BUY 1 M6E @ Market
  Order ID: 12345
  Stop Loss: 1.04800 (Order ID: 12346)
  Take Profit: 1.05200 (Order ID: 12347)
✓ Position opened: M6E BUY
```

### In Python Window:
```
================================================================================
✓ SUCCESS - Bridge is working!
================================================================================
```

### In NinjaTrader:
- Control Panel → **Orders** tab
- You should see 3 orders:
  1. BUY 1 M6E @ Market (filled)
  2. SELL 1 M6E @ 1.04800 STOP (stop loss)
  3. SELL 1 M6E @ 1.05200 LIMIT (take profit)

---

## 📝 About DEMO6108607

The account in your screenshot (DEMO6108607) appears to be a demo account. The NinjaTrader bridge will work with:

- ✅ **Sim101** (NinjaTrader's built-in simulator) - FREE
- ✅ **Any broker demo account** connected to NinjaTrader
- ✅ **FundedNext account** (when you start the challenge)
- ✅ **Live trading accounts**

The bridge doesn't care which account NinjaTrader is connected to - it just sends orders to whatever account is active!

---

## 🎯 Next Steps After Test Passes

Once you confirm the test works, I'll integrate your proven OANDA strategy so it:

1. Monitors OANDA for market data
2. Generates signals using your Combined V2 strategy
3. Sends signals to NinjaTrader bridge
4. Bridge executes on NinjaTrader futures
5. **All FundedNext rules automated!**

---

## 🛠 Troubleshooting

### "Could not connect to NinjaTrader"
- Make sure NinjaTrader 8 is open before running the bridge

### "Connection refused on port 8888"
- The bridge isn't running - start `NinjaTraderBridge.exe` first

### "No orders in NinjaTrader"
- Check the bridge window for errors
- Make sure NinjaTrader is connected (not disconnected)

---

**Ready to test? Follow the 3 steps above!** 🚀
