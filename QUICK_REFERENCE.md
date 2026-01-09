# 🚀 HTF Confluence System - Quick Reference

## Current Status
✅ **LIVE & PROFITABLE** - 75% WR | +$2.86 PnL  
🖥️ **VPS**: 72.62.3.184 | Screen: htf_binance  
💰 **Balance**: $24.66 | Available: $4.85

---

## Essential Commands

### Deploy Updates
```bash
./deploy_to_vps.sh
```

### Monitor Live
```bash
# Real-time log
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -f /tmp/htf_engine.log"

# Recent positions
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -100 /tmp/htf_engine.log | grep 'Entry: '"

# Session stats
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -200 /tmp/htf_engine.log | grep 'Total:'"
```

### Restart Engine
```bash
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "screen -S htf_binance -X quit && cd /root/thevolumeainative/trading_system/Binance_Futures_Trading && screen -dmS htf_binance bash -c 'echo CONFIRM | python3 engine/htf_confluence_live_engine.py --live 2>&1 | tee /tmp/htf_engine.log'"
```

---

## Strategy at a Glance

**Entry**: 4H + 1H trends aligned + 15m confluence 5-8/8 + smart filters  
**Exit**: TP/SL per symbol OR fakeout protection  
**Risk**: 20x leverage, $5 min, isolated margin

**Key Protection**:
- 4H+1H must align (no counter-trend)
- Fakeout: Breakeven at +15%, cut loss at -10%
- Trailing: Locks profit at +30% ROI

---

## Files

📁 **trading_system/** - Production code  
📄 **README.md** - Full documentation  
📄 **CLAUDE.MD** - Development notes  
📦 **backup_archive/** - All old files (201 files backed up)

---

## Performance

**Today**: 4 trades | 3W/1L (75%)  
**Open**: 4 SHORT positions (DOT, BNB, XRP, ADA)  
**ROI**: +8.7%, +1.8%, -1.7%, +2.0%

---

**Last Updated**: Jan 9, 2026
