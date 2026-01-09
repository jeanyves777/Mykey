# CRITICAL BUGS FOUND - January 4, 2026

## 🚨 BUG 1: Fake Breakeven Trades (Lines 3035-3041)
**Issue**: When realized PNL = $0 (Binance API timeout), it:
- Counts it as "BE" (breakeven)  
- **Still counts it as a WIN** (+1 to daily_wins)
- Adds $0 to daily_pnl
- Shows "+$1.82 profit, 3 wins" but balance doesn't increase

**Fix**: Don't count $0 PNL trades as wins. Mark as "UNKNOWN" and require manual verification.

---

## 🚨 BUG 2: Trailing TP Broken After Half-Close (Line 1688)
**Issue**: After boost half-close:
```python
position.peak_roi = 0.0  # Reset peak for trailing
```
This **RESETS peak_roi to 0** every half-close! So:
- First half-close @ +20% ROI → peak_roi = 0
- Position drops to +15% → trailing should trigger at +5%
- But peak_roi = 0, so it thinks ROI is +15% (above trail threshold)
- **Trailing never triggers!**

**Fix**: Don't reset peak_roi. Only update it when current ROI > peak_roi.

---

## 🚨 BUG 3: Liquidation Price = $0.00 (Always)
**Issue**: Line 4274 reads from Binance:
```python
liq_price = float(binance_pos.get("liquidation_price", 0))
```

Binance returns `liquidationPrice: "0"` for isolated positions with cross-margin enabled or when position is small.

**Why it's dangerous**:
- You think liquidation is impossible ($0.00 = can't liquidate)
- Real liquidation price might be $85,000 for BTC LONG
- If BTC drops to $85k → LIQUIDATED without warning

**Fix**: Calculate liquidation price manually using formula:
```
Liq Price = Entry Price × (1 ± (1 / Leverage))
LONG: Entry × (1 - 1/20) = Entry × 0.95
SHORT: Entry × (1 + 1/20) = Entry × 1.05
```

---

## 🚨 BUG 4: PNL Not Verified Against Binance
**Issue**: Lines 1640-1650 calculate PNL from order fill price:
```python
if position.side == "LONG":
    half_pnl = (fill_price - position.avg_entry_price) * half_qty
```

But doesn't verify with Binance realized PNL! So:
- Fill price might be wrong (slippage)
- Avg entry price corrupted → fake PNL
- Shows +$1.82 profit but Binance shows $0

**Fix**: After close, fetch realized PNL from Binance income API (wait up to 120 seconds).

---

## 🚨 BUG 5: Half-Close Uses avg_entry_price (Can Be Corrupted)
**Issue**: Line 1643 uses `position.avg_entry_price` to calculate PNL:
```python
half_pnl = (fill_price - position.avg_entry_price) * half_qty
```

We already know avg_entry_price can become $0.00 (previous bug). So:
- avg_entry_price = $0.00
- fill_price = $91,000
- half_pnl = ($91,000 - $0) × 0.001 = +$91 (FAKE PROFIT!)

**Fix**: Use entry_price as fallback if avg_entry_price invalid.

---

## 📊 IMPACT ON YOUR $59 BALANCE

Current state:
- Balance: $59.10
- Claims: "+$1.82 profit, 3 wins"
- Reality: Balance hasn't moved from starting value

**What happened:**
1. 3 positions hit TP
2. System attempted to close
3. Calculated PNL = +$1.82
4. Binance actually closed at breakeven or loss
5. Realized PNL = $0 or negative
6. System couldn't find PNL in income API (timeout)
7. Defaulted to $0, counted as "win"
8. Stats show "+$1.82" but balance shows truth

**Real outcome**: Those 3 "winning" trades were actually breakevens or small losses.

---

## ✅ FIXES NEEDED (In Priority Order)

1. **FIX PNL VERIFICATION** (Highest priority)
   - Always verify realized PNL from Binance income API
   - Wait up to 120 seconds
   - If still $0, mark as "UNKNOWN" not "WIN"

2. **FIX TRAILING PEAK ROI RESET**
   - Don't reset peak_roi after half-close
   - Only update when current_roi > peak_roi

3. **FIX LIQUIDATION PRICE CALCULATION**
   - Calculate manually: Entry × (1 ± 1/leverage)
   - Display as warning if Binance returns $0

4. **FIX HALF-CLOSE PNL CALCULATION**
   - Use entry_price if avg_entry_price corrupted
   - Validate before calculating PNL

5. **ADD BINANCE SYNC CHECK BEFORE CLOSE**
   - Before closing, fetch live position from Binance
   - Verify quantity, entry price, current PNL
   - Only close if data matches

---

## 🔧 DEPLOYMENT PLAN

1. Apply all fixes to local file
2. SCP to server
3. Stop live trading
4. Clear corrupted stats
5. Restart with --live flag
6. Monitor for 30 minutes to verify fixes work

---

## ⚠️ IMMEDIATE ACTION REQUIRED

**DO NOT TRUST CURRENT STATS:**
- "3 wins, +$1.82" is likely FAKE
- Real balance: $59.10 (unchanged)
- You may have actually lost money on those trades

**RECOMMENDATION:**
1. Stop trading NOW
2. Check Binance trade history manually
3. Calculate real P&L from Binance
4. Fix bugs before continuing
5. Restart with clean state
