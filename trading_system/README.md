# 🎯 HTF Confluence Futures Trading System
## Ultra-Clean Production Structure

```
trading_system/
├── __init__.py
├── requirements.txt
└── Binance_Futures_Trading/
    ├── config/
    │   ├── __init__.py
    │   └── trading_config.py          ⚙️ API keys, symbols, settings
    └── engine/
        ├── __init__.py
        ├── binance_client.py            🔌 Binance API wrapper
        ├── htf_confluence_strategy.py   🎯 Core strategy logic
        └── htf_confluence_live_engine.py 🚀 Live trading engine
```

---

## 📁 File Purposes

### `trading_config.py` (540 lines)
- **Binance API Configuration**: Demo & Live keys, URLs
- **Symbol Settings**: All 10 trading pairs with TP/SL/confluence
- **Risk Parameters**: Leverage (20x), position sizing ($5 min)
- **Trading Settings**: Timeframes, filters, protection settings

### `binance_client.py` (1,231 lines)
- **API Wrapper**: All Binance Futures API calls
- **Position Management**: Open, close, modify positions
- **Order Management**: Market orders, TP/SL orders
- **Data Fetching**: Klines, account info, trade history
- **Error Handling**: Rate limits, connection issues

### `htf_confluence_strategy.py` (870 lines)
- **Strategy Logic**: 8-point confluence system
- **Indicator Calculations**: MACD, RSI, EMA, ADX
- **Signal Generation**: Entry/exit conditions
- **Asset-Specific Rules**: Different settings per symbol
- **Smart Filters**: Pullback detection, trend alignment

### `htf_confluence_live_engine.py` (2,783 lines)
- **Main Trading Loop**: 60-second cycles
- **Position Monitoring**: Track open positions, ROI, PnL
- **Risk Management**: Fakeout protection, trailing stops
- **4H+1H Trend Filter**: Prevents counter-trend entries
- **ML Data Logging**: Captures all signals and trades
- **Session Management**: Stats tracking, position resume

---

## 🔥 What's Different?

**BEFORE**: 131+ files in main workspace, 80+ scripts in Binance folder  
**AFTER**: 4 essential files in workspace, 4 core files in engine

**ALL OTHER FILES**: Safely backed up in `backup_archive/` (742 files, 68 MB)

---

## ✅ Verified Working

- ✅ Import paths correct
- ✅ Config properly loaded
- ✅ No missing dependencies in code structure
- ✅ Deployed successfully to VPS
- ✅ Currently running live and profitable

**Production Status**: 
- 🟢 LIVE on VPS 72.62.3.184
- 💰 4 trades: 3W/1L (75% WR)
- 📈 +$2.86 net PnL (+12% growth)
- 📊 4 open SHORT positions

---

## 🚀 Quick Actions

### Deploy Updates
```bash
cd /workspaces/Mykey
./deploy_to_vps.sh
```

### Check System
```bash
# VPS status
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "screen -ls"

# Live log
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -f /tmp/htf_engine.log"
```

### Edit Strategy
- **Settings**: `config/trading_config.py`
- **Logic**: `engine/htf_confluence_strategy.py`
- **Engine**: `engine/htf_confluence_live_engine.py`

---

## 📦 Backup Location

All 742+ archived files organized in `/backup_archive/`:
- Old strategies & platforms
- Backtest scripts  
- Analysis tools
- Documentation
- Historical results

**Nothing lost - everything preserved!**

---

**Last Cleaned**: January 9, 2026  
**System**: Production HTF Confluence Futures Trader  
**Status**: ✅ Clean, Focused, Profitable
