"""
Optimized Forex Trading Engine
Uses pair-specific strategies with 70%+ win rate from optimization results
"""

import time
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
import json
import os

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.optimized_strategy import OptimizedForexStrategy


# Optimized instruments
OPTIMIZED_INSTRUMENTS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD']

# Risk config for optimized strategy
RISK_CONFIG = {
    "max_total_positions": 3,
    "max_daily_loss_pct": 0.03,  # 3% daily loss limit
    "max_trades_per_day": 15
}

# Paper trading config
PAPER_TRADING_CONFIG = {
    "initial_balance": 10000,
    "commission_per_trade": 0.00003,  # ~3 pips spread equivalent
    "slippage_pips": 0.3
}


class OptimizedForexEngine:
    """Trading engine using optimized pair-specific strategies"""

    def __init__(self, account_type: str = "practice", initial_balance: float = None):
        """
        Initialize optimized forex engine

        Args:
            account_type: 'practice' for demo, 'live' for real trading
            initial_balance: Starting balance (only for paper trading)
        """
        self.account_type = account_type
        self.oanda = OandaClient(account_type=account_type)

        self.strategy = OptimizedForexStrategy(
            instruments=OPTIMIZED_INSTRUMENTS,
            max_trades_per_day=RISK_CONFIG["max_trades_per_day"]
        )

        # Trading state
        self.balance = initial_balance or PAPER_TRADING_CONFIG["initial_balance"]
        self.initial_balance = self.balance
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.daily_trades = 0
        self.daily_start_balance = self.balance

        # Risk management
        self.max_total_positions = RISK_CONFIG["max_total_positions"]
        self.max_daily_loss_pct = RISK_CONFIG["max_daily_loss_pct"]

        # State
        self.is_running = False
        self.last_check_time = {}

        print(f"[ENGINE] Optimized Forex Engine initialized ({account_type})")
        print(f"[ENGINE] Initial Balance: ${self.balance:,.2f}")
        print(f"[ENGINE] Trading {len(OPTIMIZED_INSTRUMENTS)} optimized pairs")

    def get_market_data(self, instrument: str) -> Optional[pd.DataFrame]:
        """Fetch 5-minute candles for strategy"""
        try:
            candles = self.oanda.get_candles(
                instrument,
                granularity="M5",
                count=100  # Need at least 50 for indicators
            )

            df = pd.DataFrame(candles)
            if len(df) < 50:
                return None
            return df

        except Exception as e:
            print(f"[ENGINE] Error fetching data for {instrument}: {e}")
            return None

    def check_daily_reset(self):
        """Reset daily counters at start of new day"""
        now = datetime.now(pytz.UTC)

        if hasattr(self, 'last_reset_date'):
            if now.date() > self.last_reset_date:
                print(f"\n[ENGINE] ===== NEW TRADING DAY: {now.date()} =====")
                self.daily_trades = 0
                self.daily_start_balance = self.balance
                self.last_reset_date = now.date()
                self.strategy.reset_daily_cooldowns()
        else:
            self.last_reset_date = now.date()

    def get_daily_pl_pct(self) -> float:
        """Calculate daily P&L percentage"""
        if self.daily_start_balance == 0:
            return 0
        return (self.balance - self.daily_start_balance) / self.daily_start_balance

    def check_entry_signals(self):
        """Check all instruments for entry signals"""
        self.check_daily_reset()

        # Check daily loss limit
        daily_pl_pct = self.get_daily_pl_pct()
        if daily_pl_pct <= -self.max_daily_loss_pct:
            print(f"[ENGINE] Daily loss limit reached ({daily_pl_pct*100:.2f}%). Stopping for today.")
            return

        # Check max positions
        if len(self.positions) >= self.max_total_positions:
            return

        for instrument in OPTIMIZED_INSTRUMENTS:
            # Skip if already have position
            if instrument in self.positions:
                continue

            # Rate limiting: Check each pair every 60 seconds
            now = datetime.now(pytz.UTC)
            if instrument in self.last_check_time:
                if (now - self.last_check_time[instrument]).seconds < 60:
                    continue

            self.last_check_time[instrument] = now

            # Get market data
            df_5min = self.get_market_data(instrument)
            if df_5min is None:
                continue

            # Check for entry signal using optimized strategy
            signal = self.strategy.should_enter_trade(
                instrument=instrument,
                df_5min=df_5min,
                current_positions=len(self.positions),
                trades_today=self.daily_trades,
                now=now
            )

            action = signal['action']
            reason = signal['reason']

            if action in ['BUY', 'SELL']:
                print(f"\n[SIGNAL] {instrument}: {action} - {reason}")
                self.enter_position(instrument, signal)
            else:
                # Only print skip messages occasionally
                if now.second < 10:
                    print(f"[SCAN] {instrument}: {reason}")

    def enter_position(self, instrument: str, signal: Dict):
        """Enter a new position"""
        # Get current price
        price_data = self.oanda.get_current_price(instrument)

        if signal["action"] == "BUY":
            entry_price = price_data["ask"]
            direction = "LONG"
        else:
            entry_price = price_data["bid"]
            direction = "SHORT"

        # Calculate position size ($1 per pip)
        units = self.strategy.calculate_position_size(instrument, entry_price)

        # Get TP/SL from optimized settings
        settings = signal['settings']
        tp_sl = self.strategy.calculate_tp_sl_prices(
            instrument,
            entry_price,
            signal["action"],
            settings
        )

        # Apply slippage
        pip_value = 0.0001 if "JPY" not in instrument else 0.01
        slippage = PAPER_TRADING_CONFIG["slippage_pips"] * pip_value

        if signal["action"] == "BUY":
            entry_price += slippage
        else:
            entry_price -= slippage

        # Store position
        position = {
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "units": units,
            "stop_loss": tp_sl['sl_price'],
            "take_profit": tp_sl['tp_price'],
            "entry_time": datetime.now(pytz.UTC),
            "signal": signal,
            "strategy": settings['strategy'],
            "tp_pips": tp_sl['tp_pips'],
            "sl_pips": tp_sl['sl_pips']
        }

        self.positions[instrument] = position
        self.daily_trades += 1

        print(f"\n{'='*70}")
        print(f"[TRADE] ENTERED {direction} on {instrument}")
        print(f"[TRADE] Strategy: {settings['strategy']}")
        print(f"[TRADE] Entry: {entry_price:.5f}")
        print(f"[TRADE] Units: {units:,} (~$1/pip)")
        print(f"[TRADE] TP: {tp_sl['tp_price']:.5f} (+{tp_sl['tp_pips']} pips)")
        print(f"[TRADE] SL: {tp_sl['sl_price']:.5f} (-{tp_sl['sl_pips']} pips)")
        if 'analysis' in signal:
            print(f"[TRADE] RSI: {signal['analysis']['rsi']:.1f}")
        print(f"{'='*70}\n")

    def manage_positions(self):
        """Manage open positions (check SL, TP)"""
        positions_to_close = []

        for instrument, position in self.positions.items():
            # Get current price
            price_data = self.oanda.get_current_price(instrument)

            if position["direction"] == "LONG":
                current_price = price_data["bid"]
            else:
                current_price = price_data["ask"]

            # Calculate P&L in pips
            pip_value = 0.0001 if "JPY" not in instrument else 0.01
            if position["direction"] == "LONG":
                pips = (current_price - position["entry_price"]) / pip_value
                pl = pips  # $1 per pip
            else:
                pips = (position["entry_price"] - current_price) / pip_value
                pl = pips

            # Check exit conditions
            exit_reason = None

            # Take Profit
            if position["direction"] == "LONG" and current_price >= position["take_profit"]:
                exit_reason = f"TAKE PROFIT (+{position['tp_pips']} pips)"
            elif position["direction"] == "SHORT" and current_price <= position["take_profit"]:
                exit_reason = f"TAKE PROFIT (+{position['tp_pips']} pips)"

            # Stop Loss
            elif position["direction"] == "LONG" and current_price <= position["stop_loss"]:
                exit_reason = f"STOP LOSS (-{position['sl_pips']} pips)"
            elif position["direction"] == "SHORT" and current_price >= position["stop_loss"]:
                exit_reason = f"STOP LOSS (-{position['sl_pips']} pips)"

            if exit_reason:
                positions_to_close.append((instrument, current_price, exit_reason, pl, pips))
            else:
                # Status update
                print(f"[POS] {instrument} {position['direction']}: {pips:+.1f} pips | "
                      f"Price: {current_price:.5f} | TP: {position['take_profit']:.5f} | SL: {position['stop_loss']:.5f}")

        # Close positions
        for instrument, exit_price, exit_reason, pl, pips in positions_to_close:
            self.close_position(instrument, exit_price, exit_reason, pl, pips)

    def close_position(self, instrument: str, exit_price: float, exit_reason: str, pl: float, pips: float):
        """Close a position"""
        position = self.positions[instrument]

        # Update balance
        self.balance += pl

        # Record trade
        trade_record = {
            "instrument": instrument,
            "direction": position["direction"],
            "strategy": position["strategy"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "units": position["units"],
            "entry_time": position["entry_time"].isoformat(),
            "exit_time": datetime.now(pytz.UTC).isoformat(),
            "pips": pips,
            "pl": pl,
            "exit_reason": exit_reason,
            "balance": self.balance
        }

        self.trades.append(trade_record)

        result = "WIN" if pl > 0 else "LOSS"

        print(f"\n{'='*70}")
        print(f"[TRADE] CLOSED {position['direction']} on {instrument} - {result}")
        print(f"[TRADE] Strategy: {position['strategy']}")
        print(f"[TRADE] Entry: {position['entry_price']:.5f} -> Exit: {exit_price:.5f}")
        print(f"[TRADE] Result: {pips:+.1f} pips (${pl:+.2f})")
        print(f"[TRADE] Reason: {exit_reason}")
        print(f"[TRADE] Balance: ${self.balance:,.2f}")
        print(f"{'='*70}\n")

        del self.positions[instrument]

    def save_trades(self):
        """Save trade history to file"""
        os.makedirs("trading_system/Forex_Trading/logs", exist_ok=True)

        filename = f"trading_system/Forex_Trading/logs/optimized_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump({
                "strategy": "OptimizedForexStrategy",
                "initial_balance": self.initial_balance,
                "final_balance": self.balance,
                "total_pips": sum(t['pips'] for t in self.trades),
                "trades": self.trades
            }, f, indent=2)

        print(f"[ENGINE] Trades saved to {filename}")

    def print_summary(self):
        """Print trading summary"""
        if len(self.trades) == 0:
            print("[SUMMARY] No trades executed")
            return

        winning_trades = [t for t in self.trades if t["pl"] > 0]
        losing_trades = [t for t in self.trades if t["pl"] < 0]

        total_pips = sum(t["pips"] for t in self.trades)
        win_rate = len(winning_trades) / len(self.trades) * 100

        avg_win_pips = sum(t["pips"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss_pips = sum(t["pips"] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # By strategy
        by_strategy = {}
        for t in self.trades:
            s = t['strategy']
            if s not in by_strategy:
                by_strategy[s] = {'trades': 0, 'wins': 0, 'pips': 0}
            by_strategy[s]['trades'] += 1
            by_strategy[s]['pips'] += t['pips']
            if t['pl'] > 0:
                by_strategy[s]['wins'] += 1

        print(f"\n{'='*70}")
        print("[SUMMARY] OPTIMIZED FOREX STRATEGY RESULTS")
        print(f"{'='*70}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Avg Win: +{avg_win_pips:.1f} pips")
        print(f"Avg Loss: {avg_loss_pips:.1f} pips")
        print(f"Total P&L: {total_pips:+.1f} pips (${total_pips:+.2f})")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"\nBy Strategy:")
        for s, data in by_strategy.items():
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            print(f"  {s}: {data['trades']} trades, {wr:.0f}% WR, {data['pips']:+.1f} pips")
        print(f"{'='*70}\n")

    def run(self, duration_hours: float = 24):
        """
        Run trading engine

        Args:
            duration_hours: How long to run (default 24 hours)
        """
        print(f"\n[ENGINE] Starting Optimized Forex Trading for {duration_hours} hours...")
        print(f"[ENGINE] Monitoring {len(OPTIMIZED_INSTRUMENTS)} optimized pairs")

        self.is_running = True
        start_time = datetime.now(pytz.UTC)
        end_time = start_time + timedelta(hours=duration_hours)

        try:
            while self.is_running and datetime.now(pytz.UTC) < end_time:
                # Check for new entry signals
                if len(self.positions) < self.max_total_positions:
                    self.check_entry_signals()

                # Manage existing positions
                if len(self.positions) > 0:
                    self.manage_positions()

                # Sleep before next iteration
                time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("\n[ENGINE] Stopping trading...")

        finally:
            self.is_running = False

            # Close any remaining positions
            if len(self.positions) > 0:
                print("\n[ENGINE] Closing remaining positions...")
                for instrument in list(self.positions.keys()):
                    price_data = self.oanda.get_current_price(instrument)
                    position = self.positions[instrument]

                    pip_value = 0.0001 if "JPY" not in instrument else 0.01
                    if position["direction"] == "LONG":
                        exit_price = price_data["bid"]
                        pips = (exit_price - position["entry_price"]) / pip_value
                    else:
                        exit_price = price_data["ask"]
                        pips = (position["entry_price"] - exit_price) / pip_value

                    self.close_position(instrument, exit_price, "Manual close at end", pips, pips)

            # Print summary and save trades
            self.print_summary()
            self.save_trades()


if __name__ == "__main__":
    print("Starting Optimized Forex Trading Engine...")
    engine = OptimizedForexEngine(account_type="practice")
    engine.run(duration_hours=8)
