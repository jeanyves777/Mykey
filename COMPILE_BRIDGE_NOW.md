# Compile NinjaTrader Bridge - Quick Start

## ✅ What You Have
- Visual Studio: INSTALLED ✓
- NinjaTrader 8: Should be installed
- Bridge code: `trading_system/NinjaTrader_Bridge/NinjaTraderBridge.cs` ✓

## 🚀 Compile in 5 Minutes

### Step 1: Open Visual Studio (1 min)
1. Start Visual Studio
2. File → New → Project
3. Search for "Console App (.NET Framework)"
4. **IMPORTANT**: Select **.NET Framework 4.8** (NOT .NET Core)
5. Name: `NinjaTraderBridge`
6. Click Create

### Step 2: Add NinjaTrader Reference (2 min)
1. In Solution Explorer (right side), right-click "References"
2. Click "Add Reference..."
3. Click "Browse..." button at bottom
4. Navigate to: `C:\Program Files\NinjaTrader 8\bin\`
5. Select: `NinjaTrader.Client.dll`
6. Click "Add" then "OK"

### Step 3: Install JSON Package (1 min)
1. Tools → NuGet Package Manager → Package Manager Console
2. In console at bottom, type:
   ```
   Install-Package Newtonsoft.Json
   ```
3. Press Enter, wait for install

### Step 4: Copy Code (30 sec)
1. Open file: `C:\Users\Jean-Yves\thevolumeainative\trading_system\NinjaTrader_Bridge\NinjaTraderBridge.cs`
2. Copy ALL code (Ctrl+A, Ctrl+C)
3. In Visual Studio, open `Program.cs`
4. Delete everything in Program.cs
5. Paste the bridge code (Ctrl+V)

### Step 5: Build (30 sec)
1. Build → Build Solution (or press F6)
2. Check Output window - should say "Build succeeded"
3. Find your exe: `bin\Debug\NinjaTraderBridge.exe`

## 🧪 Test the Bridge (2 min)

### Start the Bridge:
1. Open NinjaTrader 8
2. Connect to Sim101 account
3. Run: `bin\Debug\NinjaTraderBridge.exe`

You should see:
```
================================================================================
NINJATRADER API BRIDGE - SIGNAL RECEIVER
================================================================================
✓ Connected to NinjaTrader 8
✓ Listening on port 8888 for signals
Status: ACTIVE
Press Ctrl+C to stop
================================================================================
```

### Send Test Signal:
Open PowerShell in your project folder and run:
```powershell
python -c "
import socket, json
signal = {
    'Action': 'ENTRY',
    'Symbol': 'M6E',
    'Side': 'BUY',
    'Quantity': 1,
    'EntryPrice': 1.05000,
    'StopLoss': 1.04800,
    'TakeProfit': 1.05200,
    'Timestamp': '2025-01-13T10:00:00'
}
sock = socket.socket()
sock.connect(('localhost', 8888))
sock.sendall(json.dumps(signal).encode())
print(sock.recv(1024))
sock.close()
"
```

**Expected**:
- Bridge shows: "Received signal: ENTRY M6E BUY"
- NinjaTrader shows order in Orders tab
- Python prints: `b'OK'`

## ✅ If Test Works

**YOU'RE READY!** The bridge is working.

Next step: Tell me which OANDA script to integrate:
- `run_forex_live_trading.py`?
- `run_crypto_live_trading.py`?
- Other?

I'll add the signal sending code to connect your proven strategy to NinjaTrader!

## ❌ If You Get Errors

**Error: "Could not load NinjaTrader.Client.dll"**
- Solution: Make sure NinjaTrader 8 is installed at `C:\Program Files\NinjaTrader 8\`

**Error: "Newtonsoft.Json not found"**
- Solution: Retry Step 3, make sure NuGet installed successfully

**Error: "Connection refused on port 8888"**
- Solution: Make sure NinjaTraderBridge.exe is running first

**Error: "No open position for M6E"**
- Solution: This is expected for test - just confirms bridge is receiving signals!

---

## 📁 File Locations Reference

```
Your Project:
C:\Users\Jean-Yves\thevolumeainative\
├── trading_system\
│   └── NinjaTrader_Bridge\
│       ├── NinjaTraderBridge.cs          ← Copy this code
│       └── oanda_to_ninjatrader_bridge.py ← Python bridge (ready)
│
Visual Studio Project (after creation):
C:\Users\Jean-Yves\source\repos\NinjaTraderBridge\
├── NinjaTraderBridge\
│   ├── Program.cs                        ← Paste code here
│   └── bin\Debug\
│       └── NinjaTraderBridge.exe         ← Run this!
```

---

**Time to compile: ~5 minutes**
**Time to test: ~2 minutes**
**Total: 7 minutes to working bridge!**

Then I'll finish the integration and you'll be ready for FundedNext! 🚀
