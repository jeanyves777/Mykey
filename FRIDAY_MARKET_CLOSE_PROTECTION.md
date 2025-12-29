# FRIDAY MARKET CLOSE PROTECTION

## Overview

Added automatic position management for Friday market close to avoid holding positions over the weekend and exposure to weekend gaps.

## Forex Market Hours

- **Market Close**: Friday 5:00 PM EST (22:00 UTC)
- **Market Open**: Sunday 5:00 PM EST (22:00 UTC)
- **Weekend Gap Risk**: 48 hours without ability to manage positions

## Protection Rules

### 1. No New Trades in Last Hour (21:00-22:00 UTC Friday)
- **Trigger**: Friday 21:00 UTC (4:00 PM EST)
- **Action**: Stop scanning for new entry signals
- **Reason**: Avoid opening positions right before weekend

**Log Message:**
```
[FRIDAY] No new trades in last hour before market close
```

### 2. Close All Trades 10 Minutes Before Market Close (21:50 UTC Friday)
- **Trigger**: Friday 21:50 UTC (4:50 PM EST)
- **Action**: Automatically close ALL open positions
- **Reason**: Exit all trades before market close

**Log Output:**
```
================================================================================
[FRIDAY MARKET CLOSE] Closing all 5 positions before market close
  Current Time: 2025-12-13 21:50:00 UTC (Friday 21:50 UTC)
  Market Closes: Friday 22:00 UTC (5:00 PM EST)
================================================================================

  Closing: EUR_USD (Trade #192) | P&L: $+12.50
    ✓ Closed successfully
  Closing: GBP_USD (Trade #193) | P&L: $-5.30
    ✓ Closed successfully
  ...

================================================================================
[WEEKEND MODE] All positions closed. No new trades until Monday.
  Next trading starts: Monday ~00:00 UTC
================================================================================
```

## Implementation Details

### Code Location
`run_forex_multi_symbol_live.py` - Lines 144-185

### Variables
```python
is_friday = now.weekday() == 4  # Friday = 4
hour_utc = now.hour
minute_utc = now.minute

friday_no_new_trades = is_friday and hour_utc >= 21
friday_close_all = is_friday and (hour_utc == 21 and minute_utc >= 50)
```

### Logic Flow

1. **Every Loop Iteration** (5 seconds):
   - Check current UTC time
   - Check if Friday
   - Set protection flags

2. **If Friday 21:50+ UTC AND positions open**:
   - Loop through all open trades
   - Close each trade individually
   - Log close results
   - Enter "weekend mode"

3. **If Friday 21:00+ UTC**:
   - Skip entry signal checks
   - Display Friday protection message
   - Continue monitoring open positions

## Time Zones Reference

| Location | Market Close (Friday) | No New Trades | Close All Trades |
|----------|----------------------|---------------|------------------|
| UTC      | 22:00                | 21:00         | 21:50            |
| EST      | 5:00 PM              | 4:00 PM       | 4:50 PM          |
| PST      | 2:00 PM              | 1:00 PM       | 1:50 PM          |
| London   | 10:00 PM             | 9:00 PM       | 9:50 PM          |
| Tokyo    | 7:00 AM (Sat)        | 6:00 AM (Sat) | 6:50 AM (Sat)    |

## Weekend Mode

After closing all trades on Friday:
- System continues running
- Monitors balance and logs
- No new trades until Monday
- Sunday evening market open (22:00 UTC) resumes normal trading

## Benefits

1. ✅ **No Weekend Gap Risk**: All positions closed before weekend
2. ✅ **Controlled Exits**: Closes at market, not forced by broker
3. ✅ **10-Minute Buffer**: Time to handle any close failures
4. ✅ **Automatic**: No manual intervention required
5. ✅ **Logged**: All Friday closes tracked in trade logs

## Testing

### Manual Test (Non-Friday)
```python
# In run_forex_multi_symbol_live.py, temporarily change:
is_friday = True  # Force Friday mode for testing
```

### Verify Friday Protection
1. Run system on Thursday evening
2. Confirm no new trades after 21:00 UTC
3. Confirm all positions close at 21:50 UTC
4. Verify logs show Friday close messages

## Notes

- Uses `client.close_trade()` for each position
- Logs exit reason as "MANUAL" (Friday close)
- Does NOT affect stop loss or take profit orders
- System continues running over weekend (monitoring only)

---

**Status**: ✅ Implemented
**Date**: December 12, 2025
**File**: `run_forex_multi_symbol_live.py`
