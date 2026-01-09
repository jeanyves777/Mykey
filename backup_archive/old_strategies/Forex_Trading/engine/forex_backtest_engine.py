"""
Forex Backtest Engine
Historical simulation of multi-timeframe momentum strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
import json

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.multi_timeframe_momentum import MultiTimeframeMomentumStrategy
from trading_system.Forex_Trading.config.forex_trading_config import (
    FOREX_INSTRUMENTS,
    STRATEGY_CONFIG,
    RISK_CONFIG,
    BACKTEST_CONFIG
)


class ForexBacktestEngine:
    """Backtest engine for forex strategies"""

    def __init__(
        self,
        instruments: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        strategy_config: Dict = None
    ):
        """
        Initialize backtest engine

        Args:
            instruments: List of forex pairs to trade
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            strategy_config: Strategy configuration dict
        """
        self.instruments = instruments
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        self.initial_capital = initial_capital

        # Initialize OANDA client for data
        self.oanda = OandaClient(account_type="practice")

        # Initialize strategy
        config = strategy_config or STRATEGY_CONFIG
        self.strategy = MultiTimeframeMomentumStrategy(
            instruments=instruments,
            **config
        )

        # Backtest state
        self.balance = initial_capital
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        self.daily_trades = 0
        self.daily_start_balance = initial_capital
        self.current_date = None

        # Risk management
        self.max_total_positions = RISK_CONFIG["max_total_positions"]
        self.max_daily_loss_pct = RISK_CONFIG["max_daily_loss_pct"]

        # Commission and slippage
        self.commission_rate = BACKTEST_CONFIG["commission"]
        self.slippage_pips = BACKTEST_CONFIG["slippage"]

        print("[BACKTEST] Forex Backtest Engine initialized")
        print(f"[BACKTEST] Period: {start_date} to {end_date}")
        print(f"[BACKTEST] Instruments: {', '.join(instruments)}")
        print(f"[BACKTEST] Initial Capital: ${initial_capital:,.2f}")

    def fetch_historical_data(self, instrument: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all historical data for backtesting

        Returns dict with all timeframes
        """
        print(f"[BACKTEST] Fetching historical data for {instrument}...")

        try:
            # Calculate how many days of data we need
            days = (self.end_date - self.start_date).days + 10  # Extra buffer

            # Fetch 1-minute data in chunks (max 5000 candles per request)
            candles_1min = []
            current_time = self.start_date

            while current_time < self.end_date:
                chunk = self.oanda.get_candles(
                    instrument,
                    granularity="M1",
                    from_time=current_time,
                    to_time=min(current_time + timedelta(days=3), self.end_date),
                    count=5000
                )
                candles_1min.extend(chunk)

                if len(chunk) > 0:
                    current_time = chunk[-1]["time"] + timedelta(minutes=1)
                else:
                    break

            # Fetch other timeframes
            candles_5min = self.oanda.get_candles(
                instrument,
                granularity="M5",
                from_time=self.start_date - timedelta(days=2),
                to_time=self.end_date
            )

            candles_30min = self.oanda.get_candles(
                instrument,
                granularity="M30",
                from_time=self.start_date - timedelta(days=10),
                to_time=self.end_date
            )

            candles_1hour = self.oanda.get_candles(
                instrument,
                granularity="H1",
                from_time=self.start_date - timedelta(days=20),
                to_time=self.end_date
            )

            # Convert to DataFrames
            df_1min = pd.DataFrame(candles_1min)
            df_5min = pd.DataFrame(candles_5min)
            df_30min = pd.DataFrame(candles_30min)
            df_1hour = pd.DataFrame(candles_1hour)

            print(f"[BACKTEST] {instrument}: {len(df_1min)} 1-min bars, "
                  f"{len(df_5min)} 5-min bars, "
                  f"{len(df_30min)} 30-min bars, "
                  f"{len(df_1hour)} 1-hour bars")

            return {
                "1min": df_1min,
                "5min": df_5min,
                "30min": df_30min,
                "1hour": df_1hour
            }

        except Exception as e:
            print(f"[BACKTEST] Error fetching data for {instrument}: {e}")
            return None

    def resample_to_timeframe(self, df: pd.DataFrame, current_time: datetime, lookback_bars: int) -> pd.DataFrame:
        """Get data up to current_time for strategy analysis"""
        if len(df) == 0:
            return df

        # Filter data up to current time
        mask = df['time'] <= current_time
        filtered = df[mask].copy()

        # Return last N bars
        return filtered.tail(lookback_bars)

    def check_daily_reset(self, current_time: datetime):
        """Reset daily counters"""
        if self.current_date is None or current_time.date() > self.current_date:
            if self.current_date is not None:
                print(f"\n[BACKTEST] Day {self.current_date} completed. "
                      f"Trades: {self.daily_trades}, Balance: ${self.balance:,.2f}")

            self.current_date = current_time.date()
            self.daily_trades = 0
            self.daily_start_balance = self.balance

    def get_daily_pl_pct(self) -> float:
        """Calculate daily P&L percentage"""
        if self.daily_start_balance == 0:
            return 0
        return (self.balance - self.daily_start_balance) / self.daily_start_balance

    def simulate_trade_execution(self, instrument: str, signal: Dict, current_price: float):
        """Simulate entering a trade"""
        # Apply slippage
        pip_value = 0.0001 if "JPY" not in instrument else 0.01
        slippage = self.slippage_pips * pip_value

        if signal["action"] == "BUY":
            entry_price = current_price + slippage
            direction = "LONG"
        else:
            entry_price = current_price - slippage
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

        # Deduct commission
        commission = abs(units) * entry_price * self.commission_rate
        self.balance -= commission

        # Store position
        position = {
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": self.current_time,
            "units": units,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "commission": commission,
            "trailing_stop_triggered": False,
            "highest_price": entry_price if direction == "LONG" else None,
            "lowest_price": entry_price if direction == "SHORT" else None
        }

        self.positions[instrument] = position
        self.daily_trades += 1

        return position

    def update_position(self, instrument: str, current_bar: Dict) -> Optional[Dict]:
        """
        Update position and check for exit

        Returns exit info if position should be closed
        """
        position = self.positions[instrument]

        high = current_bar["high"]
        low = current_bar["low"]
        close = current_bar["close"]

        # Update highest/lowest
        if position["direction"] == "LONG":
            if position["highest_price"] is None or high > position["highest_price"]:
                position["highest_price"] = high
        else:
            if position["lowest_price"] is None or low < position["lowest_price"]:
                position["lowest_price"] = low

        # Calculate current P&L
        if position["direction"] == "LONG":
            current_pl = (close - position["entry_price"]) * position["units"]
        else:
            current_pl = (position["entry_price"] - close) * abs(position["units"])

        pl_pct = current_pl / (abs(position["units"]) * position["entry_price"])

        # Check trailing stop trigger
        if not position["trailing_stop_triggered"]:
            if pl_pct >= self.strategy.trailing_stop_trigger:
                position["trailing_stop_triggered"] = True

        # Check exit conditions
        exit_price = None
        exit_reason = None

        if position["direction"] == "LONG":
            # Check TP hit during bar
            if high >= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_reason = "TAKE_PROFIT"

            # Check SL hit during bar
            elif low <= position["stop_loss"]:
                exit_price = position["stop_loss"]
                exit_reason = "STOP_LOSS"

            # Check trailing stop
            elif position["trailing_stop_triggered"]:
                trailing_stop = position["highest_price"] * (1 - self.strategy.trailing_stop_distance)
                if low <= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "TRAILING_STOP"

        else:  # SHORT
            # Check TP hit during bar
            if low <= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_reason = "TAKE_PROFIT"

            # Check SL hit during bar
            elif high >= position["stop_loss"]:
                exit_price = position["stop_loss"]
                exit_reason = "STOP_LOSS"

            # Check trailing stop
            elif position["trailing_stop_triggered"]:
                trailing_stop = position["lowest_price"] * (1 + self.strategy.trailing_stop_distance)
                if high >= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = "TRAILING_STOP"

        if exit_price:
            # Calculate final P&L
            if position["direction"] == "LONG":
                final_pl = (exit_price - position["entry_price"]) * position["units"]
            else:
                final_pl = (position["entry_price"] - exit_price) * abs(position["units"])

            final_pl_pct = final_pl / (abs(position["units"]) * position["entry_price"])

            return {
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pl": final_pl,
                "pl_pct": final_pl_pct
            }

        return None

    def run_backtest(self):
        """Run the backtest"""
        print(f"\n[BACKTEST] Starting backtest from {self.start_date.date()} to {self.end_date.date()}")

        # Fetch all historical data
        all_data = {}
        for instrument in self.instruments:
            data = self.fetch_historical_data(instrument)
            if data:
                all_data[instrument] = data

        if len(all_data) == 0:
            print("[BACKTEST] No data available for backtest")
            return

        # Get all unique 1-min timestamps across all instruments
        all_timestamps = set()
        for instrument, data in all_data.items():
            if len(data["1min"]) > 0:
                all_timestamps.update(data["1min"]["time"].tolist())

        timestamps = sorted(list(all_timestamps))
        timestamps = [t for t in timestamps if self.start_date <= t <= self.end_date]

        print(f"[BACKTEST] Processing {len(timestamps)} time bars...")

        # Simulate tick-by-tick
        for i, current_time in enumerate(timestamps):
            self.current_time = current_time
            self.check_daily_reset(current_time)

            # Update existing positions
            for instrument in list(self.positions.keys()):
                if instrument not in all_data:
                    continue

                # Find current bar
                df_1min = all_data[instrument]["1min"]
                current_bars = df_1min[df_1min["time"] == current_time]

                if len(current_bars) > 0:
                    current_bar = current_bars.iloc[0].to_dict()

                    # Check for exit
                    exit_info = self.update_position(instrument, current_bar)

                    if exit_info:
                        position = self.positions[instrument]
                        self.balance += exit_info["pl"]

                        # Record trade
                        trade = {
                            "instrument": instrument,
                            "direction": position["direction"],
                            "entry_price": position["entry_price"],
                            "entry_time": position["entry_time"].isoformat(),
                            "exit_price": exit_info["exit_price"],
                            "exit_time": current_time.isoformat(),
                            "units": position["units"],
                            "pl": exit_info["pl"],
                            "pl_pct": exit_info["pl_pct"],
                            "exit_reason": exit_info["exit_reason"],
                            "commission": position["commission"],
                            "balance": self.balance
                        }

                        self.trades.append(trade)
                        del self.positions[instrument]

            # Check for new entries (only on 1-min bar closes)
            if len(self.positions) < self.max_total_positions:
                daily_pl_pct = self.get_daily_pl_pct()

                # Check daily loss limit
                if daily_pl_pct > -self.max_daily_loss_pct:
                    for instrument in self.instruments:
                        if instrument in self.positions or instrument not in all_data:
                            continue

                        # Get historical data up to current time
                        df_1min = self.resample_to_timeframe(
                            all_data[instrument]["1min"],
                            current_time,
                            500
                        )
                        df_5min = self.resample_to_timeframe(
                            all_data[instrument]["5min"],
                            current_time,
                            100
                        )
                        df_30min = self.resample_to_timeframe(
                            all_data[instrument]["30min"],
                            current_time,
                            50
                        )
                        df_1hour = self.resample_to_timeframe(
                            all_data[instrument]["1hour"],
                            current_time,
                            50
                        )

                        if len(df_1min) < 50:
                            continue

                        # Check for entry signal
                        signal = self.strategy.should_enter_trade(
                            instrument=instrument,
                            df_1min=df_1min,
                            df_5min=df_5min,
                            df_30min=df_30min,
                            df_1hour=df_1hour,
                            current_positions=len(self.positions),
                            trades_today=self.daily_trades,
                            daily_pl_pct=daily_pl_pct
                        )

                        if signal["action"] in ["BUY", "SELL"]:
                            current_price = df_1min.iloc[-1]["close"]
                            self.simulate_trade_execution(instrument, signal, current_price)

            # Record equity every hour
            if i % 60 == 0:
                unrealized_pl = 0
                for instrument, position in self.positions.items():
                    if instrument in all_data:
                        df_1min = all_data[instrument]["1min"]
                        current_bars = df_1min[df_1min["time"] == current_time]
                        if len(current_bars) > 0:
                            current_price = current_bars.iloc[0]["close"]

                            if position["direction"] == "LONG":
                                pl = (current_price - position["entry_price"]) * position["units"]
                            else:
                                pl = (position["entry_price"] - current_price) * abs(position["units"])

                            unrealized_pl += pl

                self.equity_curve.append({
                    "time": current_time,
                    "balance": self.balance,
                    "equity": self.balance + unrealized_pl,
                    "positions": len(self.positions)
                })

            # Progress update
            if i % 1000 == 0:
                progress = i / len(timestamps) * 100
                print(f"[BACKTEST] Progress: {progress:.1f}% | "
                      f"Balance: ${self.balance:,.2f} | "
                      f"Trades: {len(self.trades)}")

        # Close any remaining positions
        print("\n[BACKTEST] Closing remaining positions...")
        for instrument in list(self.positions.keys()):
            if instrument in all_data:
                df_1min = all_data[instrument]["1min"]
                last_price = df_1min.iloc[-1]["close"]

                position = self.positions[instrument]

                if position["direction"] == "LONG":
                    pl = (last_price - position["entry_price"]) * position["units"]
                else:
                    pl = (position["entry_price"] - last_price) * abs(position["units"])

                pl_pct = pl / (abs(position["units"]) * position["entry_price"])
                self.balance += pl

                trade = {
                    "instrument": instrument,
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "entry_time": position["entry_time"].isoformat(),
                    "exit_price": last_price,
                    "exit_time": self.end_date.isoformat(),
                    "units": position["units"],
                    "pl": pl,
                    "pl_pct": pl_pct,
                    "exit_reason": "BACKTEST_END",
                    "commission": position["commission"],
                    "balance": self.balance
                }

                self.trades.append(trade)

        print("[BACKTEST] Backtest completed!")
        self.print_results()
        self.save_results()

    def print_results(self):
        """Print backtest results"""
        if len(self.trades) == 0:
            print("\n[RESULTS] No trades executed during backtest period")
            return

        winning_trades = [t for t in self.trades if t["pl"] > 0]
        losing_trades = [t for t in self.trades if t["pl"] < 0]

        total_pl = self.balance - self.initial_capital
        total_return = (self.balance / self.initial_capital - 1) * 100

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t["pl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pl"] for t in losing_trades]) if losing_trades else 0

        profit_factor = abs(sum(t["pl"] for t in winning_trades) / sum(t["pl"] for t in losing_trades)) if losing_trades else float('inf')

        # Calculate max drawdown
        equity_values = [self.initial_capital] + [t["balance"] for t in self.trades]
        peak = equity_values[0]
        max_dd = 0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        print(f"\n{'='*80}")
        print("[RESULTS] BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print(f"Total P&L: ${total_pl:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"\nTotal Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Average Win: ${avg_win:,.2f}")
        print(f"Average Loss: ${avg_loss:,.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"{'='*80}\n")

    def save_results(self):
        """Save backtest results to file"""
        import os
        os.makedirs("trading_system/Forex_Trading/logs", exist_ok=True)

        filename = f"trading_system/Forex_Trading/logs/forex_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = {
            "config": {
                "instruments": self.instruments,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "initial_capital": self.initial_capital
            },
            "summary": {
                "final_balance": self.balance,
                "total_pl": self.balance - self.initial_capital,
                "total_return_pct": (self.balance / self.initial_capital - 1) * 100,
                "total_trades": len(self.trades),
                "winning_trades": len([t for t in self.trades if t["pl"] > 0]),
                "losing_trades": len([t for t in self.trades if t["pl"] < 0])
            },
            "trades": self.trades,
            "equity_curve": [
                {"time": e["time"].isoformat(), "balance": e["balance"], "equity": e["equity"]}
                for e in self.equity_curve
            ]
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[RESULTS] Results saved to {filename}")


if __name__ == "__main__":
    # Run backtest for last 30 days
    end_date = datetime.now(pytz.UTC).date()
    start_date = end_date - timedelta(days=30)

    engine = ForexBacktestEngine(
        instruments=["EUR_USD", "GBP_USD"],  # Start with 2 pairs
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=10000
    )

    engine.run_backtest()
