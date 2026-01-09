# BACKUP ARCHIVE

**Created**: January 9, 2026  
**Reason**: Cleaning up workspace to focus on production HTF Confluence Futures Trading System

## What's Here

### `/backtest_scripts/`
All historical backtest scripts used for strategy development:
- backtest_*.py files
- Various timeframe and data source backtests
- Optimization and testing scripts

### `/analysis_scripts/`
Analysis and data processing scripts:
- analyze_*.py files
- Performance analysis tools
- Pattern recognition scripts

### `/old_strategies/`
Previous strategy implementations and utility scripts:
- Other trading strategies (MARA, scalping, 0DTE, etc.)
- Utility scripts (check_*, debug_*, download_*, etc.)
- Experimental approaches
- Other platform implementations (Forex, NinjaTrader, Tradovate, etc.)

### `/binance_futures_old/`
Old Binance Futures scripts and strategies:
- Previous backtest scripts (54 files)
- Old trading engines (momentum, paper, backtest)
- Utility scripts (analyze, optimize, fix, etc.)
- Old strategy implementations
- Historical results and summaries

### `/old_logs_results/`
Historical backtest results and logs:
- backtest_results_*.txt files
- Session logs
- Performance reports

### `/documentation/`
Old documentation and strategy notes:
- BACKTEST_*.md
- BRIDGE_*.md
- Strategy planning docs
- Bug reports and fixes

---

## Active Production System

The main workspace now contains ONLY the production system:

**Core Strategy Files:**
- `trading_system/Binance_Futures_Trading/engine/htf_confluence_strategy.py`
- `trading_system/Binance_Futures_Trading/engine/htf_confluence_live_engine.py`
- `trading_system/Binance_Futures_Trading/engine/binance_client.py`

**Configuration:**
- Asset-specific TP/SL settings
- 4H + 1H dual trend alignment filter
- Smart entry filters (1m pullback, candle confirmation)
- Fakeout protection with breakeven SL

**Performance (as of Jan 9, 2026):**
- 4 trades: 3W/1L (75% WR)
- PnL: +$1.86 realized, ~$1.00 unrealized
- 10 symbols: BTC, ETH, SOL, DOT, BNB, XRP, ADA, AVAX, LINK, LTC
- 20x leverage, isolated margin, $5 minimum per trade

**Key Features:**
- ✅ 4H + 1H trend alignment (prevents counter-trend entries)
- ✅ Fixed MACD (line > signal) and RSI (35-75 LONG, 25-65 SHORT)
- ✅ Fakeout protection (breakeven at +15%, damage control at -15%)
- ✅ ML logging (signals, trades, market snapshots)
- ✅ Trailing profit lock (activates at +30% ROI)

---

## Restoring Files

If you need to reference or restore any backed up files:

```bash
# Copy a specific backtest back to workspace
cp backup_archive/backtest_scripts/backtest_name.py ./

# View old results
cat backup_archive/old_logs_results/backtest_results_v6.txt

# Check old strategy
less backup_archive/old_strategies/mara_strategy.py
```

All files are preserved and can be restored at any time!
