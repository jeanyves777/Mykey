"""
Forex Paper Trading Engine
Simulates live trading with OANDA data but without real money
"""

import time
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
import json
import os

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.multi_timeframe_momentum import MultiTimeframeMomentumStrategy
from trading_system.Forex_Trading.config.forex_trading_config import (
    FOREX_INSTRUMENTS,
    STRATEGY_CONFIG,
    RISK_CONFIG,
    DATA_CONFIG,
    PAPER_TRADING_CONFIG
)


class ForexPaperTradingEngine:
    """Paper trading engine for forex with OANDA data"""

    def __init__(self, strategy_config: Dict = None):
        """Initialize paper trading engine"""
        self.oanda = OandaClient(account_type="practice")

        config = strategy_config or STRATEGY_CONFIG
        self.strategy = MultiTimeframeMomentumStrategy(
            instruments=FOREX_INSTRUMENTS,
            **config
        )

        # Paper trading state
        self.balance = PAPER_TRADING_CONFIG["initial_balance"]
        self.initial_balance = self.balance
        self.positions: Dict[str, Dict] = {}  # instrument -> position info
        self.trades: List[Dict] = []
        self.daily_trades = 0
        self.daily_start_balance = self.balance

        # Risk management
        self.max_total_positions = RISK_CONFIG["max_total_positions"]
        self.max_daily_loss_pct = RISK_CONFIG["max_daily_loss_pct"]

        # Trading state
        self.is_running = False
        self.last_check_time = {}  # instrument -> last check time

        print("[ENGINE] Forex Paper Trading Engine initialized")
        print(f"[ENGINE] Initial Balance: ${self.balance:,.2f}")
        print(f"[ENGINE] Trading {len(FOREX_INSTRUMENTS)} pairs")

    def get_market_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        """Fetch all required timeframes for an instrument"""
        try:
            # Fetch different timeframes
            candles_1min = self.oanda.get_candles(
                instrument,
                granularity="M1",
                count=DATA_CONFIG["candles_1min"]
            )
            candles_5min = self.oanda.get_candles(
                instrument,
                granularity="M5",
                count=DATA_CONFIG["candles_5min"]
            )
            candles_30min = self.oanda.get_candles(
                instrument,
                granularity="M30",
                count=DATA_CONFIG["candles_30min"]
            )
            candles_1hour = self.oanda.get_candles(
                instrument,
                granularity="H1",
                count=DATA_CONFIG["candles_1hour"]
            )

            # Convert to DataFrames
            df_1min = pd.DataFrame(candles_1min)
            df_5min = pd.DataFrame(candles_5min)
            df_30min = pd.DataFrame(candles_30min)
            df_1hour = pd.DataFrame(candles_1hour)

            if len(df_1min) == 0:
                return None

            return {
                "1min": df_1min,
                "5min": df_5min,
                "30min": df_30min,
                "1hour": df_1hour
            }

        except Exception as e:
            print(f"[ENGINE] Error fetching data for {instrument}: {e}")
            return None

    def check_daily_reset(self):
        """Reset daily counters at start of new day"""
        now = datetime.now(pytz.UTC)

        # Reset at 00:00 UTC
        if hasattr(self, 'last_reset_date'):
            if now.date() > self.last_reset_date:
                print(f"\n[ENGINE] ===== NEW TRADING DAY: {now.date()} =====")
                self.daily_trades = 0
                self.daily_start_balance = self.balance
                self.last_reset_date = now.date()
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

        for instrument in FOREX_INSTRUMENTS:
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
            print(f"\n[ENGINE] Checking {instrument}...")
            market_data = self.get_market_data(instrument)

            if not market_data:
                continue

            # Check for entry signal
            signal = self.strategy.should_enter_trade(
                instrument=instrument,
                df_1min=market_data["1min"],
                df_5min=market_data["5min"],
                df_30min=market_data["30min"],
                df_1hour=market_data["1hour"],
                current_positions=len(self.positions),
                trades_today=self.daily_trades,
                daily_pl_pct=daily_pl_pct
            )

            print(f"[ENGINE] {instrument} Signal: {signal['action']} - {signal['reason']}")

            if signal["action"] in ["BUY", "SELL"]:
                self.enter_position(instrument, signal, market_data)

    def enter_position(self, instrument: str, signal: Dict, market_data: Dict):
        """Enter a new position"""
        # Get current price
        price_data = self.oanda.get_current_price(instrument)

        if signal["action"] == "BUY":
            entry_price = price_data["ask"]  # Buy at ask
            direction = "LONG"
        else:
            entry_price = price_data["bid"]  # Sell at bid
            direction = "SHORT"

        # Calculate position size
        units = self.strategy.calculate_position_size(
            self.balance,
            entry_price,
            instrument
        )

        # Calculate SL and TP
        stop_loss, take_profit = self.strategy.calculate_stop_loss_take_profit(
            entry_price,
            signal["action"]
        )

        # Apply commission and slippage
        commission = abs(units) * entry_price * PAPER_TRADING_CONFIG["commission_per_trade"]

        # Slippage (0.5 pips = 0.00005 for most pairs)
        pip_value = 0.0001 if "JPY" not in instrument else 0.01
        slippage = PAPER_TRADING_CONFIG["slippage_pips"] * pip_value

        if signal["action"] == "BUY":
            entry_price += slippage
        else:
            entry_price -= slippage

        # Deduct commission
        self.balance -= commission

        # Store position
        position = {
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "units": units,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.now(pytz.UTC),
            "commission": commission,
            "signal": signal,
            "trailing_stop_triggered": False,
            "highest_price": entry_price if direction == "LONG" else None,
            "lowest_price": entry_price if direction == "SHORT" else None
        }

        self.positions[instrument] = position
        self.daily_trades += 1

        print(f"\n{'='*80}")
        print(f"[TRADE] ENTERED {direction} position on {instrument}")
        print(f"[TRADE] Entry Price: {entry_price:.5f}")
        print(f"[TRADE] Units: {units:,}")
        print(f"[TRADE] Position Size: ${abs(units * entry_price):,.2f}")
        print(f"[TRADE] Stop Loss: {stop_loss:.5f}")
        print(f"[TRADE] Take Profit: {take_profit:.5f}")
        print(f"[TRADE] Commission: ${commission:.2f}")
        print(f"[TRADE] Confidence: {signal['confidence']}")
        print(f"[TRADE] Reason: {signal['reason']}")
        print(f"[TRADE] Balance: ${self.balance:,.2f}")
        print(f"{'='*80}\n")

    def manage_positions(self):
        """Manage open positions (check SL, TP, trailing stops)"""
        positions_to_close = []

        for instrument, position in self.positions.items():
            # Get current price
            price_data = self.oanda.get_current_price(instrument)

            if position["direction"] == "LONG":
                current_price = price_data["bid"]  # Exit at bid
            else:
                current_price = price_data["ask"]  # Cover at ask

            # Calculate unrealized P&L
            if position["direction"] == "LONG":
                pl = (current_price - position["entry_price"]) * position["units"]
            else:
                pl = (position["entry_price"] - current_price) * abs(position["units"])

            pl_pct = pl / (abs(position["units"]) * position["entry_price"])

            # Update highest/lowest for trailing stop
            if position["direction"] == "LONG":
                if position["highest_price"] is None or current_price > position["highest_price"]:
                    position["highest_price"] = current_price
            else:
                if position["lowest_price"] is None or current_price < position["lowest_price"]:
                    position["lowest_price"] = current_price

            # Check trailing stop trigger
            if not position["trailing_stop_triggered"]:
                if pl_pct >= self.strategy.trailing_stop_trigger:
                    position["trailing_stop_triggered"] = True
                    print(f"[TRADE] {instrument} Trailing stop ACTIVATED at +{pl_pct*100:.2f}%")

            # Check exit conditions
            exit_reason = None

            # Take Profit
            if position["direction"] == "LONG" and current_price >= position["take_profit"]:
                exit_reason = f"TAKE PROFIT at {current_price:.5f}"
            elif position["direction"] == "SHORT" and current_price <= position["take_profit"]:
                exit_reason = f"TAKE PROFIT at {current_price:.5f}"

            # Stop Loss
            elif position["direction"] == "LONG" and current_price <= position["stop_loss"]:
                exit_reason = f"STOP LOSS at {current_price:.5f}"
            elif position["direction"] == "SHORT" and current_price >= position["stop_loss"]:
                exit_reason = f"STOP LOSS at {current_price:.5f}"

            # Trailing Stop
            elif position["trailing_stop_triggered"]:
                if position["direction"] == "LONG":
                    trailing_stop_price = position["highest_price"] * (1 - self.strategy.trailing_stop_distance)
                    if current_price <= trailing_stop_price:
                        exit_reason = f"TRAILING STOP at {current_price:.5f} (was at {position['highest_price']:.5f})"
                else:
                    trailing_stop_price = position["lowest_price"] * (1 + self.strategy.trailing_stop_distance)
                    if current_price >= trailing_stop_price:
                        exit_reason = f"TRAILING STOP at {current_price:.5f} (was at {position['lowest_price']:.5f})"

            if exit_reason:
                positions_to_close.append((instrument, current_price, exit_reason, pl, pl_pct))
            else:
                # Print status update
                print(f"[POSITION] {instrument} {position['direction']}: "
                      f"P&L: ${pl:,.2f} ({pl_pct*100:+.2f}%) | "
                      f"Price: {current_price:.5f}")

        # Close positions
        for instrument, exit_price, exit_reason, pl, pl_pct in positions_to_close:
            self.close_position(instrument, exit_price, exit_reason, pl, pl_pct)

    def close_position(self, instrument: str, exit_price: float, exit_reason: str, pl: float, pl_pct: float):
        """Close a position"""
        position = self.positions[instrument]

        # Update balance
        self.balance += pl

        # Record trade
        trade_record = {
            "instrument": instrument,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "units": position["units"],
            "entry_time": position["entry_time"].isoformat(),
            "exit_time": datetime.now(pytz.UTC).isoformat(),
            "pl": pl,
            "pl_pct": pl_pct,
            "commission": position["commission"],
            "exit_reason": exit_reason,
            "balance": self.balance
        }

        self.trades.append(trade_record)

        print(f"\n{'='*80}")
        print(f"[TRADE] CLOSED {position['direction']} position on {instrument}")
        print(f"[TRADE] Entry: {position['entry_price']:.5f} -> Exit: {exit_price:.5f}")
        print(f"[TRADE] P&L: ${pl:,.2f} ({pl_pct*100:+.2f}%)")
        print(f"[TRADE] Reason: {exit_reason}")
        print(f"[TRADE] New Balance: ${self.balance:,.2f}")
        print(f"[TRADE] Total Return: {((self.balance/self.initial_balance)-1)*100:+.2f}%")
        print(f"{'='*80}\n")

        # Remove position
        del self.positions[instrument]

    def save_trades(self):
        """Save trade history to file"""
        os.makedirs("trading_system/Forex_Trading/logs", exist_ok=True)

        filename = f"trading_system/Forex_Trading/logs/forex_paper_trades_{datetime.now().strftime('%Y%m%d')}.json"

        with open(filename, 'w') as f:
            json.dump({
                "initial_balance": self.initial_balance,
                "final_balance": self.balance,
                "total_return_pct": ((self.balance / self.initial_balance) - 1) * 100,
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

        total_pl = sum(t["pl"] for t in self.trades)
        win_rate = len(winning_trades) / len(self.trades) * 100

        avg_win = sum(t["pl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t["pl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        print(f"\n{'='*80}")
        print("[SUMMARY] TRADING SUMMARY")
        print(f"{'='*80}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Total P&L: ${total_pl:,.2f}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total Return: {((self.balance/self.initial_balance)-1)*100:+.2f}%")
        print(f"{'='*80}\n")

    def run(self, duration_hours: float = 24):
        """
        Run paper trading engine

        Args:
            duration_hours: How long to run (default 24 hours)
        """
        print(f"\n[ENGINE] Starting paper trading for {duration_hours} hours...")
        print(f"[ENGINE] Monitoring {len(FOREX_INSTRUMENTS)} instruments")

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
            print("\n[ENGINE] Stopping paper trading...")

        finally:
            self.is_running = False

            # Close any remaining positions
            if len(self.positions) > 0:
                print("\n[ENGINE] Closing remaining positions...")
                for instrument in list(self.positions.keys()):
                    price_data = self.oanda.get_current_price(instrument)
                    position = self.positions[instrument]

                    if position["direction"] == "LONG":
                        exit_price = price_data["bid"]
                        pl = (exit_price - position["entry_price"]) * position["units"]
                    else:
                        exit_price = price_data["ask"]
                        pl = (position["entry_price"] - exit_price) * abs(position["units"])

                    pl_pct = pl / (abs(position["units"]) * position["entry_price"])
                    self.close_position(instrument, exit_price, "Manual close at end", pl, pl_pct)

            # Print summary and save trades
            self.print_summary()
            self.save_trades()


if __name__ == "__main__":
    print("Starting Forex Paper Trading Engine...")

    engine = ForexPaperTradingEngine()
    engine.run(duration_hours=8)  # Run for 8 hours
