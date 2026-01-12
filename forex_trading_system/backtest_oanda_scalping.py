"""
OANDA Forex Scalping Backtest
==============================
Backtest HTF Confluence Strategy with REAL OANDA data
Tests all 9 forex pairs with asset-specific scalping settings

Features:
- Real OANDA historical data
- Realistic spread costs
- Asset-specific TP/SL (scalping optimized)
- 8/8 confluence requirement
- Multi-timeframe analysis (4H, 1H, 15m, 5m)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.oanda_client import OANDAClient
from engine.htf_confluence_strategy import HTFConfluenceForexStrategy, TrendDirection
from config.trading_config import (
    FOREX_SYMBOLS,
    get_symbol_config,
    OANDA_CONFIG,
    SMART_EXIT_CONFIG,
    SESSION_TRADING
)


class OANDABacktester:
    """
    Backtest HTF Confluence Strategy using real OANDA data.
    """
    
    def __init__(self, initial_balance: float = 5000.0):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting capital in USD
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        
        # Risk management
        self.min_balance_threshold = initial_balance * 0.20  # Stop if balance drops below 20%
        self.max_drawdown_pct = 0.50  # Stop if drawdown > 50%
        self.is_stopped = False
        self.stop_reason = None
        
        # Daily loss protection - Per-Symbol
        self.per_symbol_daily_losses = {}  # {symbol: {'date': str, 'losses': int}}
        self.max_daily_losses_per_symbol = 1
        self.reduce_size_after_losses = 0  # No reduction needed with 1 loss limit
        
        # Connect to OANDA
        self.client = OANDAClient(practice=OANDA_CONFIG["practice"])
        
        # Statistics
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        
        print("\n" + "="*70)
        print("OANDA FOREX SCALPING BACKTEST - REALISTIC")
        print("="*70)
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Min Balance Threshold: ${self.min_balance_threshold:,.2f} (20%)")
        print(f"Max Drawdown: {self.max_drawdown_pct*100}%")
        print(f"Daily Loss Limit: {self.max_daily_losses_per_symbol} loss per symbol per day")
        print(f"Protection: Each symbol blocked for rest of day after {self.max_daily_losses_per_symbol} loss")
        print(f"Independence: Other symbols continue trading normally")
        print(f"Symbols: {len(FOREX_SYMBOLS)} pairs")
        if SESSION_TRADING["enabled"]:
            sessions = ", ".join(SESSION_TRADING["allowed_sessions"])
            print(f"Trading Sessions: {sessions} only")
        print("="*70)
    
    def fetch_data(
        self,
        symbol: str,
        granularity: str,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical data from OANDA.
        
        Args:
            symbol: Forex pair
            granularity: Timeframe (M15, H1, H4)
            days_back: How many days of history
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"  Fetching {symbol} {granularity} data ({days_back} days)...", end=" ")
        
        # Calculate candle count needed
        if granularity == "M15":
            candles_per_day = 96  # 24h * 4 candles/hour
        elif granularity == "H1":
            candles_per_day = 24
        elif granularity == "H4":
            candles_per_day = 6
        elif granularity == "M5":
            candles_per_day = 288  # 24h * 12 candles/hour
        else:
            candles_per_day = 100
        
        count = min(candles_per_day * days_back, 5000)  # OANDA limit
        
        df = self.client.get_candles(symbol, granularity, count)
        
        if df is None or df.empty:
            print("❌ Failed")
            return None
        
        print(f"✓ {len(df)} candles")
        return df
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        sl_pips: float,
        risk_pct: float = 0.02
    ) -> float:
        """
        Calculate position size based on CURRENT balance.
        
        Args:
            symbol: Forex pair
            entry_price: Entry price
            sl_pips: Stop loss in pips
            risk_pct: Risk percentage (default 2%)
            
        Returns:
            Position size in units (0 if insufficient balance)
        """
        # Check if we have enough balance
        if self.balance <= self.min_balance_threshold:
            return 0
        
        config = get_symbol_config(symbol)
        pip_location = config["pip_location"]
        pip_value = 10 ** pip_location
        
        # Risk amount in USD - BASED ON CURRENT BALANCE
        risk_amount = self.balance * risk_pct
        
        # Minimum risk check
        if risk_amount < 10:  # Minimum $10 risk
            return 0
        
        # Calculate position size
        # Risk = Units * Pip Value * SL Pips
        # Units = Risk / (Pip Value * SL Pips)
        position_size = risk_amount / (pip_value * sl_pips)
        
        # Round to nearest 100 units
        position_size = round(position_size / 100) * 100
        
        # Ensure minimum size
        if position_size < 1000:
            return 0
        
        # Check margin requirement (simplified - 50:1 leverage)
        margin_required = (position_size * entry_price) / 50
        
        if margin_required > self.balance * 0.80:  # Don't use more than 80% as margin
            return 0
        
        return position_size
    
    def simulate_trade(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        tp_pips: float,
        sl_pips: float,
        entry_time: datetime,
        data_m15: pd.DataFrame,
        data_h1: pd.DataFrame
    ) -> Dict:
        """
        Simulate a trade with smart exits.
        
        Args:
            symbol: Forex pair
            entry_price: Entry price
            side: "BUY" or "SELL"
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
            entry_time: Entry timestamp
            data_m15: 15m data for monitoring
            data_h1: 1H data for HTF trend monitoring
            
        Returns:
            Trade result dictionary
        """
        config = get_symbol_config(symbol)
        pip_location = config["pip_location"]
        pip_value = 10 ** pip_location
        spread = config.get("typical_spread", 0.5)
        
        # Apply spread cost
        if side == "BUY":
            entry_price += spread * pip_value  # Pay the spread
            tp_price = entry_price + (tp_pips * pip_value)
            sl_price = entry_price - (sl_pips * pip_value)
        else:
            entry_price -= spread * pip_value  # Pay the spread
            tp_price = entry_price - (tp_pips * pip_value)
            sl_price = entry_price + (sl_pips * pip_value)
        
        # Calculate position size
        position_size = self.calculate_position_size(symbol, entry_price, sl_pips)
        
        # Track peak for trailing stop
        peak_pips = 0
        
        # Get data after entry
        future_data = data_m15[data_m15.index > entry_time]
        
        if future_data.empty:
            return None
        
        # Simulate tick by tick
        for timestamp, row in future_data.iterrows():
            current_price = row["close"]
            
            # Calculate current P&L in pips
            if side == "BUY":
                pnl_pips = (current_price - entry_price) / pip_value
            else:
                pnl_pips = (entry_price - current_price) / pip_value
            
            # Update peak
            if pnl_pips > peak_pips:
                peak_pips = pnl_pips
            
            # Check SL
            if side == "BUY" and current_price <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                duration = (timestamp - entry_time).total_seconds() / 60
                break
            elif side == "SELL" and current_price >= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                duration = (timestamp - entry_time).total_seconds() / 60
                break
            
            # Check TP
            if side == "BUY" and current_price >= tp_price:
                exit_price = tp_price
                exit_reason = "TP"
                duration = (timestamp - entry_time).total_seconds() / 60
                break
            elif side == "SELL" and current_price <= tp_price:
                exit_price = tp_price
                exit_reason = "TP"
                duration = (timestamp - entry_time).total_seconds() / 60
                break
            
            # SMART EXITS
            
            # Trailing profit lock
            if SMART_EXIT_CONFIG["use_trailing_profit_lock"]:
                activation = SMART_EXIT_CONFIG["trailing_lock_activation_pips"]
                distance = SMART_EXIT_CONFIG["trailing_lock_distance_pips"]
                min_floor = SMART_EXIT_CONFIG["trailing_lock_min_floor_pips"]
                
                if peak_pips >= activation:
                    floor = max(peak_pips - distance, min_floor)
                    
                    if pnl_pips < floor:
                        exit_price = current_price
                        exit_reason = "TRAILING_LOCK"
                        duration = (timestamp - entry_time).total_seconds() / 60
                        break
            
            # Profit lock (HTF reversal check - simplified for backtest)
            if SMART_EXIT_CONFIG["use_profit_lock"]:
                min_profit = SMART_EXIT_CONFIG["profit_lock_min_pips"]
                
                if pnl_pips >= min_profit:
                    # Check if we should exit (simplified - would check HTF trend in live)
                    # For backtest, just take the profit at min threshold
                    pass
            
            # Damage control
            if SMART_EXIT_CONFIG["use_fakeout_protection"]:
                damage_limit = SMART_EXIT_CONFIG["damage_control_pips"]
                
                if pnl_pips < damage_limit:
                    exit_price = current_price
                    exit_reason = "DAMAGE_CONTROL"
                    duration = (timestamp - entry_time).total_seconds() / 60
                    break
        else:
            # No exit found in data
            exit_price = future_data.iloc[-1]["close"]
            exit_reason = "END_OF_DATA"
            duration = (future_data.index[-1] - entry_time).total_seconds() / 60
        
        # Calculate final P&L
        if side == "BUY":
            pnl_pips = (exit_price - entry_price) / pip_value
            pnl_usd = pnl_pips * pip_value * position_size
        else:
            pnl_pips = (entry_price - exit_price) / pip_value
            pnl_usd = pnl_pips * pip_value * position_size
        
        # Apply spread cost on exit
        pnl_usd -= spread * pip_value * position_size
        
        return {
            "symbol": symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "side": side,
            "position_size": position_size,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "exit_reason": exit_reason,
            "pnl_pips": pnl_pips,
            "pnl_usd": pnl_usd,
            "duration_minutes": duration,
            "peak_pips": peak_pips,
            "outcome": "WIN" if pnl_usd > 0 else "LOSS"
        }
    
    def backtest_symbol(
        self,
        symbol: str,
        days_back: int = 30
    ) -> List[Dict]:
        """
        Backtest a single symbol.
        
        Args:
            symbol: Forex pair
            days_back: Days of history to test
            
        Returns:
            List of trade results
        """
        print(f"\n{'='*70}")
        print(f"Backtesting {symbol}")
        print(f"{'='*70}")
        
        # Fetch data for all timeframes
        data_h4 = self.fetch_data(symbol, "H4", days_back)
        data_h1 = self.fetch_data(symbol, "H1", days_back)
        data_m15 = self.fetch_data(symbol, "M15", days_back)
        data_m5 = self.fetch_data(symbol, "M5", days_back)
        
        if any(df is None or df.empty for df in [data_h4, data_h1, data_m15, data_m5]):
            print(f"❌ Failed to fetch data for {symbol}")
            return []
        
        # Initialize strategy
        config = get_symbol_config(symbol)
        strategy = HTFConfluenceForexStrategy(
            leverage=config["leverage"],
            tp_pips=config["tp_pips"],
            sl_pips=config["sl_pips"],
            min_confluence_score=config["min_confluence_score"]
        )
        
        symbol_trades = []
        
        # Iterate through 15m candles looking for signals
        print(f"\nScanning for signals...")
        
        for i in range(100, len(data_m15) - 50):  # Leave buffer
            current_time = data_m15.index[i]
            
            # Session filtering
            if SESSION_TRADING["enabled"]:
                hour = current_time.hour
                current_session = None
                
                for session_name, (start_hour, end_hour) in SESSION_TRADING["session_hours"].items():
                    if start_hour <= hour < end_hour:
                        current_session = session_name
                        break
                
                # Skip if not in allowed sessions
                if current_session not in SESSION_TRADING["allowed_sessions"]:
                    continue
            
            # Get historical data up to this point
            ltf_history = data_m15.iloc[:i+1]
            htf_history = data_h1[data_h1.index <= current_time]
            m5_history = data_m5[data_m5.index <= current_time]
            
            if len(htf_history) < 60 or len(m5_history) < 30:
                continue
            
            # Check for signal
            signal = strategy.should_enter(
                ltf_df=ltf_history,
                htf_df=htf_history,
                pip_location=config["pip_location"]
            )
            
            # Add 5m confirmation
            if signal and signal.action:
                m5_indicators = strategy.calculate_indicators(m5_history)
                signal.indicators["5m_ema_bullish"] = m5_indicators.get("ema_bullish", False)
                signal.indicators["5m_ema_bearish"] = m5_indicators.get("ema_bearish", False)
                signal.indicators["volume_confirmed"] = m5_indicators.get("volume_confirmed", False)
                
                # Re-check confluence with 5m data
                if signal.action == "BUY":
                    if not (m5_indicators.get("ema_bullish") and m5_indicators.get("volume_confirmed")):
                        continue
                elif signal.action == "SELL":
                    if not (m5_indicators.get("ema_bearish") and m5_indicators.get("volume_confirmed")):
                        continue
                
                print(f"\n  📊 {signal.action} Signal | {current_time.strftime('%Y-%m-%d %H:%M')} | {signal.confluence_score}/8")
                print(f"     {signal.reason}")
                
                # Check if THIS SYMBOL already lost today
                current_date = current_time.strftime('%Y-%m-%d')
                if symbol in self.per_symbol_daily_losses:
                    symbol_data = self.per_symbol_daily_losses[symbol]
                    # Check if it's the same day
                    if symbol_data['date'] == current_date:
                        if symbol_data['losses'] >= self.max_daily_losses_per_symbol:
                            print(f"     🚫 BLOCKED: {symbol} already lost {symbol_data['losses']}x today - Try tomorrow")
                            continue
                    else:
                        # New day - reset
                        self.per_symbol_daily_losses[symbol] = {'date': current_date, 'losses': 0}
                
                # Simulate the trade
                trade_result = self.simulate_trade(
                    symbol=symbol,
                    entry_price=signal.entry_price,
                    side=signal.action,
                    tp_pips=config["tp_pips"],
                    sl_pips=config["sl_pips"],
                    entry_time=current_time,
                    data_m15=data_m15,
                    data_h1=data_h1
                )
                
                if trade_result:
                    symbol_trades.append(trade_result)
                    
                    # Update balance
                    self.balance += trade_result["pnl_usd"]
                    
                    # Update daily loss tracking per symbol
                    current_date = current_time.strftime('%Y-%m-%d')
                    
                    if trade_result["outcome"] == "LOSS":
                        # Track daily loss for this symbol
                        if symbol not in self.per_symbol_daily_losses:
                            self.per_symbol_daily_losses[symbol] = {'date': current_date, 'losses': 0}
                        
                        # If new day, reset
                        if self.per_symbol_daily_losses[symbol]['date'] != current_date:
                            self.per_symbol_daily_losses[symbol] = {'date': current_date, 'losses': 0}
                        
                        self.per_symbol_daily_losses[symbol]['losses'] += 1
                        daily_losses = self.per_symbol_daily_losses[symbol]['losses']
                        
                        if daily_losses >= self.max_daily_losses_per_symbol:
                            print(f"     ⚠️  {symbol} DAILY LIMIT: {daily_losses} loss today - BLOCKED until tomorrow")
                    else:
                        # Win - symbol can continue trading today
                        print(f"     ✅ Win! {symbol} can continue trading today")
                    
                    # Update peak balance
                    if self.balance > self.peak_balance:
                        self.peak_balance = self.balance
                    
                    # Calculate current drawdown
                    current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
                    
                    # Log result
                    outcome_icon = "✅" if trade_result["outcome"] == "WIN" else "❌"
                    # Show if symbol lost today
                    symbol_data = self.per_symbol_daily_losses.get(symbol, {'losses': 0})
                    daily_loss_indicator = f" (⚠️ {symbol_data['losses']}/1 today)" if symbol_data['losses'] >= 1 else ""
                    print(f"     {outcome_icon} {trade_result['exit_reason']} | "
                          f"P&L: {trade_result['pnl_pips']:+.1f}p (${trade_result['pnl_usd']:+.2f}) | "
                          f"Duration: {trade_result['duration_minutes']:.0f}min | "
                          f"Balance: ${self.balance:,.2f} ({current_drawdown*100:.1f}% DD){daily_loss_indicator}")
                    
                    # STOP CONDITIONS - Realistic account protection
                    if self.balance <= self.min_balance_threshold:
                        self.is_stopped = True
                        self.stop_reason = f"Balance dropped below ${self.min_balance_threshold:,.2f}"
                        print(f"\n⛔ TRADING STOPPED: {self.stop_reason}")
                        break
                    
                    if current_drawdown >= self.max_drawdown_pct:
                        self.is_stopped = True
                        self.stop_reason = f"Max drawdown {self.max_drawdown_pct*100}% reached"
                        print(f"\n⛔ TRADING STOPPED: {self.stop_reason}")
                        break
                    
                    if self.balance <= 0:
                        self.is_stopped = True
                        self.stop_reason = "Account balance reached zero"
                        print(f"\n⛔ TRADING STOPPED: {self.stop_reason}")
                        break
            
            # Check if stopped
            if self.is_stopped:
                break
        
        print(f"\n{symbol} Summary: {len(symbol_trades)} trades")
        
        return symbol_trades
    
    def run_backtest(self, days_back: int = 30):
        """
        Run backtest on all symbols.
        
        Args:
            days_back: Days of history to test
        """
        print(f"\nBacktest Period: Last {days_back} days")
        print(f"Testing {len(FOREX_SYMBOLS)} forex pairs")
        
        # Test connection first
        if not self.client.test_connection():
            print("❌ Failed to connect to OANDA")
            return
        
        all_trades = []
        
        # Backtest each symbol
        for symbol in FOREX_SYMBOLS:
            # Check if trading is stopped
            if self.is_stopped:
                print(f"\n⛔ Skipping {symbol} - Trading stopped: {self.stop_reason}")
                continue
            
            symbol_trades = self.backtest_symbol(symbol, days_back)
            all_trades.extend(symbol_trades)
            self.trades.extend(symbol_trades)
            
            # Check stop conditions after each symbol
            if self.is_stopped:
                break
        
        # Display results
        self.display_results()
    
    def display_results(self):
        """Display comprehensive backtest results."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        if not self.trades:
            print("No trades executed")
            return
        
        # Overall statistics
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["outcome"] == "WIN")
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(t["pnl_usd"] for t in self.trades)
        final_balance = self.initial_balance + total_pnl
        roi = (total_pnl / self.initial_balance * 100)
        
        winning_trades = [t for t in self.trades if t["outcome"] == "WIN"]
        losing_trades = [t for t in self.trades if t["outcome"] == "LOSS"]
        
        avg_win = np.mean([t["pnl_usd"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl_usd"] for t in losing_trades]) if losing_trades else 0
        
        avg_win_pips = np.mean([t["pnl_pips"] for t in winning_trades]) if winning_trades else 0
        avg_loss_pips = np.mean([t["pnl_pips"] for t in losing_trades]) if losing_trades else 0
        
        avg_duration = np.mean([t["duration_minutes"] for t in self.trades])
        
        # Calculate max drawdown
        max_dd = ((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
        
        print(f"\n📊 Overall Performance")
        print(f"{'─'*70}")
        
        if self.is_stopped:
            print(f"⚠️  STOPPED: {self.stop_reason}")
            print(f"{'─'*70}")
        
        # Calculate max losses in a single day (overall and per-symbol)
        from collections import defaultdict
        daily_losses_overall = defaultdict(int)
        daily_losses_per_symbol = defaultdict(lambda: defaultdict(int))
        
        for trade in self.trades:
            trade_date = trade["entry_time"].split()[0] if isinstance(trade["entry_time"], str) else trade["entry_time"].strftime('%Y-%m-%d')
            if trade["outcome"] == "LOSS":
                daily_losses_overall[trade_date] += 1
                daily_losses_per_symbol[trade["symbol"]][trade_date] += 1
        
        max_daily_losses_overall = max(daily_losses_overall.values()) if daily_losses_overall else 0
        
        # Find worst symbol (most losses in a single day)
        worst_symbol_daily = ("N/A", 0)
        for symbol, daily_losses in daily_losses_per_symbol.items():
            max_losses = max(daily_losses.values()) if daily_losses else 0
            if max_losses > worst_symbol_daily[1]:
                worst_symbol_daily = (symbol, max_losses)
        
        print(f"Total Trades:        {total_trades}")
        print(f"Wins:                {wins} ({win_rate:.1f}%)")
        print(f"Losses:              {losses} ({100-win_rate:.1f}%)")
        print(f"Max Daily Losses:    {max_daily_losses_overall} losses/day (overall) {'✅' if max_daily_losses_overall <= 6 else '⚠️'}")
        print(f"Worst Symbol Daily:  {worst_symbol_daily[0]} {worst_symbol_daily[1]} losses/day {'✅' if worst_symbol_daily[1] <= self.max_daily_losses_per_symbol else '⚠️'}")
        print(f"\nInitial Balance:     ${self.initial_balance:,.2f}")
        print(f"Peak Balance:        ${self.peak_balance:,.2f}")
        print(f"Final Balance:       ${final_balance:,.2f}")
        print(f"Total P&L:           ${total_pnl:+,.2f}")
        print(f"ROI:                 {roi:+.2f}%")
        print(f"Max Drawdown:        {max_dd:.2f}%")
        print(f"\nAvg Win:             ${avg_win:.2f} ({avg_win_pips:+.1f} pips)")
        print(f"Avg Loss:            ${avg_loss:.2f} ({avg_loss_pips:+.1f} pips)")
        print(f"Avg Duration:        {avg_duration:.0f} minutes")
        
        # Per-symbol breakdown
        print(f"\n📈 Per-Symbol Performance")
        print(f"{'─'*70}")
        print(f"{'Symbol':<10} {'Trades':<8} {'Wins':<6} {'W%':<7} {'P&L':>12} {'Avg'}")
        print(f"{'─'*70}")
        
        for symbol in FOREX_SYMBOLS:
            symbol_trades = [t for t in self.trades if t["symbol"] == symbol]
            
            if not symbol_trades:
                continue
            
            s_total = len(symbol_trades)
            s_wins = sum(1 for t in symbol_trades if t["outcome"] == "WIN")
            s_wr = (s_wins / s_total * 100) if s_total > 0 else 0
            s_pnl = sum(t["pnl_usd"] for t in symbol_trades)
            s_avg = s_pnl / s_total if s_total > 0 else 0
            
            print(f"{symbol:<10} {s_total:<8} {s_wins:<6} {s_wr:>5.1f}%  ${s_pnl:>+10.2f}  ${s_avg:>+6.2f}")
        
        print(f"{'─'*70}")
        print(f"{'TOTAL':<10} {total_trades:<8} {wins:<6} {win_rate:>5.1f}%  ${total_pnl:>+10.2f}")
        
        # Exit reasons breakdown
        print(f"\n🚪 Exit Reasons")
        print(f"{'─'*70}")
        
        exit_reasons = {}
        for trade in self.trades:
            reason = trade["exit_reason"]
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "wins": 0, "pnl": 0}
            exit_reasons[reason]["count"] += 1
            if trade["outcome"] == "WIN":
                exit_reasons[reason]["wins"] += 1
            exit_reasons[reason]["pnl"] += trade["pnl_usd"]
        
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
            count = stats["count"]
            wins = stats["wins"]
            wr = (wins / count * 100) if count > 0 else 0
            pnl = stats["pnl"]
            print(f"  {reason}: {count} trades ({wr:.1f}% WR, PNL: ${pnl:.2f})")
        
        # Save results to file
        results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "initial_balance": self.initial_balance,
                    "final_balance": final_balance,
                    "peak_balance": self.peak_balance,
                    "max_drawdown": max_dd,
                    "total_pnl": total_pnl,
                    "roi": roi,
                    "total_trades": total_trades,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": win_rate,
                    "stop_reason": self.stop_reason
                },
                "trades": self.trades
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    # Run backtest
    backtester = OANDABacktester(initial_balance=5000.0)
    backtester.run_backtest(days_back=30)
