#!/usr/bin/env python3
"""
Backtest Hedge + DCA Strategy During Market Crash
==================================================
Tests how our strategy performs during heavy downtrends.
Uses historical data from Binance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG

class CrashBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Strategy params from config
        self.tp_roi = DCA_CONFIG["take_profit_roi"]  # 8%
        self.sl_roi = DCA_CONFIG["stop_loss_roi"]    # 90%
        self.dca_levels = DCA_CONFIG["levels"]
        self.budget_split = DCA_CONFIG["hedge_mode"]["budget_split"]  # 50%

        # Positions
        self.long_position = None
        self.short_position = None

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance

    def get_historical_data(self, days: int = 30, interval: str = "1h"):
        """Fetch historical klines from Binance"""
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")

        client = BinanceClient(testnet=True, use_demo=True)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Get klines
        klines = client.get_klines(
            self.symbol,
            interval=interval,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000),
            limit=1000
        )

        if not klines:
            print("No data returned!")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)

        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        # Calculate price change
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        print(f"Price change over period: {price_change:+.2f}%")

        return df

    def calculate_roi(self, entry_price: float, current_price: float, side: str) -> float:
        """Calculate ROI based on position side"""
        if side == "LONG":
            price_pct = (current_price - entry_price) / entry_price
        else:
            price_pct = (entry_price - current_price) / entry_price
        return price_pct * self.leverage

    def get_dca_level(self, position: dict) -> int:
        """Get current DCA level from position"""
        return position.get("dca_level", 0)

    def get_tp_price(self, entry_price: float, side: str, dca_level: int) -> float:
        """Calculate TP price based on DCA level"""
        if dca_level > 0 and dca_level <= len(self.dca_levels):
            tp_roi = self.dca_levels[dca_level - 1].get("tp_roi", self.tp_roi)
        else:
            tp_roi = self.tp_roi

        tp_pct = tp_roi / self.leverage

        if side == "LONG":
            return entry_price * (1 + tp_pct)
        else:
            return entry_price * (1 - tp_pct)

    def get_sl_price(self, entry_price: float, side: str) -> float:
        """Calculate SL price (always 90% ROI)"""
        sl_pct = self.sl_roi / self.leverage

        if side == "LONG":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def check_dca_trigger(self, position: dict, current_price: float) -> bool:
        """Check if DCA should trigger"""
        dca_level = self.get_dca_level(position)
        if dca_level >= len(self.dca_levels):
            return False  # Max DCA reached

        trigger_roi = self.dca_levels[dca_level]["trigger_roi"]
        current_roi = self.calculate_roi(position["entry_price"], current_price, position["side"])

        return current_roi <= trigger_roi

    def execute_dca(self, position: dict, current_price: float) -> dict:
        """Execute DCA - add to position"""
        dca_level = self.get_dca_level(position)
        add_pct = self.dca_levels[dca_level]["add_pct"]

        # Calculate new position
        old_qty = position["quantity"]
        old_margin = position["margin"]
        add_margin = (self.start_balance * self.budget_split) * add_pct
        add_qty = (add_margin * self.leverage) / current_price

        new_qty = old_qty + add_qty
        new_margin = old_margin + add_margin

        # New average entry
        old_value = old_qty * position["entry_price"]
        add_value = add_qty * current_price
        new_entry = (old_value + add_value) / new_qty

        position["quantity"] = new_qty
        position["margin"] = new_margin
        position["entry_price"] = new_entry
        position["dca_level"] = dca_level + 1
        position["tp_price"] = self.get_tp_price(new_entry, position["side"], dca_level + 1)
        position["sl_price"] = self.get_sl_price(new_entry, position["side"])

        return position

    def open_position(self, side: str, price: float) -> dict:
        """Open new position"""
        budget = self.start_balance * self.budget_split
        initial_margin = budget * 0.20  # 20% initial entry
        quantity = (initial_margin * self.leverage) / price

        position = {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0),
            "sl_price": self.get_sl_price(price, side)
        }
        return position

    def close_position(self, position: dict, exit_price: float, exit_type: str, timestamp):
        """Close position and record trade"""
        if position["side"] == "LONG":
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["quantity"]

        self.balance += pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1

        # Track drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.trades.append({
            "timestamp": timestamp,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "quantity": position["quantity"],
            "dca_level": position["dca_level"],
            "pnl": pnl,
            "exit_type": exit_type,
            "balance": self.balance
        })

        return pnl

    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest"""
        print("\n" + "="*70)
        print("RUNNING HEDGE + DCA BACKTEST")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"TP ROI: {self.tp_roi*100:.0f}% | SL ROI: {self.sl_roi*100:.0f}%")
        print("="*70)

        # Open initial positions
        first_price = df['close'].iloc[0]
        self.long_position = self.open_position("LONG", first_price)
        self.short_position = self.open_position("SHORT", first_price)

        print(f"\nOpened LONG @ ${first_price:.4f} (TP: ${self.long_position['tp_price']:.4f}, SL: ${self.long_position['sl_price']:.4f})")
        print(f"Opened SHORT @ ${first_price:.4f} (TP: ${self.short_position['tp_price']:.4f}, SL: ${self.short_position['sl_price']:.4f})")

        # Iterate through candles
        for i, (timestamp, row) in enumerate(df.iterrows()):
            high = row['high']
            low = row['low']
            close = row['close']

            # Check LONG position
            if self.long_position:
                # Check TP hit
                if high >= self.long_position["tp_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["tp_price"], "TP", timestamp)
                    print(f"[{timestamp}] LONG TP HIT @ ${self.long_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")
                    # Re-enter immediately
                    self.long_position = self.open_position("LONG", close)
                # Check SL hit
                elif low <= self.long_position["sl_price"]:
                    pnl = self.close_position(self.long_position, self.long_position["sl_price"], "SL", timestamp)
                    print(f"[{timestamp}] LONG SL HIT @ ${self.long_position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.long_position['dca_level']}")
                    # Re-enter immediately
                    self.long_position = self.open_position("LONG", close)
                # Check DCA
                elif self.check_dca_trigger(self.long_position, close):
                    old_entry = self.long_position["entry_price"]
                    self.long_position = self.execute_dca(self.long_position, close)
                    print(f"[{timestamp}] LONG DCA {self.long_position['dca_level']} @ ${close:.4f} | Avg: ${old_entry:.4f} -> ${self.long_position['entry_price']:.4f}")

            # Check SHORT position
            if self.short_position:
                # Check TP hit
                if low <= self.short_position["tp_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["tp_price"], "TP", timestamp)
                    print(f"[{timestamp}] SHORT TP HIT @ ${self.short_position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")
                    # Re-enter immediately
                    self.short_position = self.open_position("SHORT", close)
                # Check SL hit
                elif high >= self.short_position["sl_price"]:
                    pnl = self.close_position(self.short_position, self.short_position["sl_price"], "SL", timestamp)
                    print(f"[{timestamp}] SHORT SL HIT @ ${self.short_position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.short_position['dca_level']}")
                    # Re-enter immediately
                    self.short_position = self.open_position("SHORT", close)
                # Check DCA
                elif self.check_dca_trigger(self.short_position, close):
                    old_entry = self.short_position["entry_price"]
                    self.short_position = self.execute_dca(self.short_position, close)
                    print(f"[{timestamp}] SHORT DCA {self.short_position['dca_level']} @ ${close:.4f} | Avg: ${old_entry:.4f} -> ${self.short_position['entry_price']:.4f}")

        # Calculate final unrealized PNL
        final_price = df['close'].iloc[-1]
        unrealized_long = 0
        unrealized_short = 0

        if self.long_position:
            unrealized_long = (final_price - self.long_position["entry_price"]) * self.long_position["quantity"]
        if self.short_position:
            unrealized_short = (self.short_position["entry_price"] - final_price) * self.short_position["quantity"]

        total_unrealized = unrealized_long + unrealized_short

        # Print results
        self.print_results(df, total_unrealized)

    def print_results(self, df: pd.DataFrame, unrealized_pnl: float):
        """Print backtest results"""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)

        price_start = df['close'].iloc[0]
        price_end = df['close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100

        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\nMarket Performance:")
        print(f"  Price: ${price_start:.4f} -> ${price_end:.4f} ({price_change:+.2f}%)")
        print(f"  Period: {df.index[0]} to {df.index[-1]}")

        print(f"\nStrategy Performance:")
        print(f"  Starting Balance: ${self.start_balance:.2f}")
        print(f"  Ending Balance:   ${self.balance:.2f}")
        print(f"  Realized P&L:     ${self.total_pnl:+.2f}")
        print(f"  Unrealized P&L:   ${unrealized_pnl:+.2f}")
        print(f"  Total Return:     {((self.balance - self.start_balance) / self.start_balance * 100):+.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Wins:             {self.total_wins}")
        print(f"  Losses:           {self.total_losses}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Max Drawdown:     {self.max_drawdown:.2f}%")

        # DCA Analysis
        dca_trades = [t for t in self.trades if t["dca_level"] > 0]
        if dca_trades:
            avg_dca_pnl = sum(t["pnl"] for t in dca_trades) / len(dca_trades)
            print(f"\nDCA Analysis:")
            print(f"  Trades with DCA:  {len(dca_trades)}")
            print(f"  Avg P&L (DCA):    ${avg_dca_pnl:+.2f}")

        print("\n" + "="*70)


def main():
    print("="*70)
    print("HEDGE + DCA STRATEGY BACKTEST - CRASH PERIOD TEST")
    print("="*70)

    # Test on DOTUSDT (volatile crypto)
    symbol = "DOTUSDT"

    backtester = CrashBacktester(symbol, start_balance=100.0)

    # Get 30 days of hourly data
    df = backtester.get_historical_data(days=30, interval="1h")

    if df is not None and len(df) > 0:
        backtester.run_backtest(df)
    else:
        print("Failed to get historical data")


if __name__ == "__main__":
    main()
