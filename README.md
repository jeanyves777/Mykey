# HTF Confluence Futures Trading System

**Live Production System** - Binance Futures 20x Leverage  
**Status**: ✅ Active and Profitable  
**Performance**: 75% Win Rate | +$2.86 Net PnL (from $23.75 starting balance)

---

## 🚀 Quick Start

### Deploy to VPS
```bash
./deploy_to_vps.sh
```

### Monitor Live Trading
```bash
# Real-time log monitoring
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -f /tmp/htf_engine.log"

# Check recent positions
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -100 /tmp/htf_engine.log | grep -E '(Balance|PnL|ROI)'"

# View session stats
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "tail -200 /tmp/htf_engine.log | grep 'SESSION STATS' -A 20"
```

---

## 📁 System Structure

```
trading_system/Binance_Futures_Trading/
├── engine/
│   ├── htf_confluence_strategy.py      # Core strategy logic
│   ├── htf_confluence_live_engine.py   # Live trading engine
│   ├── binance_client.py               # Binance API wrapper
│   └── ml_logs/                        # ML data collection
│       ├── signals_log.csv             # All signals
│       ├── trades_log.csv              # Completed trades
│       └── market_snapshots.csv        # Candle data
└── config/
    └── (configuration files)
```

---

## ⚙️ Strategy Configuration

### Trading Symbols (10)
BTC, ETH, SOL, DOT, BNB, XRP, ADA, AVAX, LINK, LTC

### Asset-Specific Settings
| Symbol | TP ROI | SL ROI | Confluence |
|--------|--------|--------|------------|
| BTC    | 40%    | 20%    | 8/8        |
| ETH    | 45%    | 20%    | 7/8        |
| SOL    | 50%    | 25%    | 7/8        |
| DOT    | 75%    | 35%    | 5/8        |
| BNB    | 30%    | 15%    | 5/8        |
| XRP    | 50%    | 35%    | 6/8        |
| ADA    | 60%    | 25%    | 5/8        |
| AVAX   | 75%    | 35%    | 7/8        |
| LINK   | 45%    | 20%    | 7/8        |
| LTC    | 40%    | 20%    | 8/8        |

### Core Parameters
- **Leverage**: 20x (isolated margin)
- **Minimum Trade**: $5 USDT
- **Max Concurrent**: Based on available balance / $5
- **Cycle Time**: 60 seconds

---

## 🎯 Strategy Features

### 1. Multi-Timeframe Trend Alignment
- **4H**: Major trend direction
- **1H**: Confirmation layer
- **15m**: Entry signals
- **5m**: Additional confirmation
- **1m**: Pullback detection

**Entry Rule**: 4H and 1H trends MUST align (both BULLISH or both BEARISH)

### 2. 8-Point Confluence System
1. ✅ HTF (1H) trend alignment
2. ✅ 15m EMA alignment (9 > 21 > 50 for LONG)
3. ✅ 15m RSI in range (35-75 LONG, 25-65 SHORT)
4. ✅ 15m MACD confirmation (line > signal)
5. ✅ 5m EMA alignment
6. ✅ ADX > 20 (trend strength)
7. ✅ MACD momentum building
8. ✅ Volume > 1.2x average

### 3. Smart Entry Filters
- **1m Pullback**: Wait for price within 0.3% of 1m EMA21
- **Candle Confirmation**: Entry candle must match direction
- **Trend Filter**: Skip if 4H ≠ 1H (prevents counter-trend entries)

### 4. Risk Management

#### Fakeout Protection (5-Cycle Confirmation)
- **Breakeven**: ROI ≥ +15% + trends reversed → Move SL to entry
- **Small Profit Exit**: ROI 0-5% + reversed + weak confirms → Close
- **Cut Loss Early**: ROI -10 to 0% + reversed + weak confirms → Close
- **Damage Control**: ROI < -15% + reversed → Close immediately

#### Trailing Profit Lock
- **Activation**: ROI ≥ +30%
- **Distance**: Trail 5% below peak
- **Floor**: Minimum 20% ROI guaranteed

### 5. Indicator Configuration
- **MACD (12, 26, 9)**: Line > signal (more responsive than histogram)
- **RSI (14)**: Widened ranges for trending markets
- **EMA**: 9, 21, 50 (multiple timeframes)
- **ADX (14)**: Minimum 20 for trend strength

---

## 📊 Current Performance

**Session Stats** (Jan 9, 2026):
- Total Trades: 4
- Wins: 3 (75% WR)
- Losses: 1
- Realized PnL: +$1.86
- Unrealized PnL: ~$1.00
- Total: +$2.86 (+12% from start)

**Open Positions** (4 SHORT):
- DOT: +8.7% ROI
- BNB: +1.8% ROI
- XRP: -1.7% ROI
- ADA: +2.0% ROI

---

## 🔍 ML Data Collection

All signals and trades are logged with 45+ features for future ML training:

**Current Data** (as of Jan 9):
- 383 signals logged
- 27 completed trades
- 1,149 market snapshots
- 3 days of data (need 200+ trades for ML)

**Status**: Continue collecting diverse market data. ML training recommended after 2-4 weeks when dataset includes bull + bear + ranging conditions.

---

## 🛠️ Maintenance

### Update Strategy
1. Edit `htf_confluence_strategy.py` or `htf_confluence_live_engine.py`
2. Run `./deploy_to_vps.sh`
3. Restart engine on VPS (script prompts)

### Check System Health
```bash
# Screen session status
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "screen -ls"

# Process running
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "ps aux | grep htf_confluence"

# Recent errors
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "grep ERROR /tmp/htf_engine.log | tail -20"
```

### Restart Engine
```bash
ssh -i ~/.ssh/id_rsa_vps root@72.62.3.184 "screen -S htf_binance -X quit && cd /root/thevolumeainative/trading_system/Binance_Futures_Trading && screen -dmS htf_binance bash -c 'echo CONFIRM | python3 engine/htf_confluence_live_engine.py --live 2>&1 | tee /tmp/htf_engine.log'"
```

---

## 📚 Documentation

- **CLAUDE.MD**: Development notes and conversations
- **backup_archive/**: All old strategies, backtests, and analysis scripts

---

## ⚠️ Risk Disclaimer

This is a live trading system using real funds with 20x leverage. Losses can exceed deposits. Monitor positions regularly and understand all risk management features before trading.

**VPS**: 72.62.3.184  
**Platform**: Binance Futures Mainnet  
**Last Updated**: January 9, 2026
