#!/usr/bin/env python3
"""
Backtest ONE DIRECTION Strategy with Trend Detection
=====================================================
Instead of hedging (LONG + SHORT), we only enter in the trend direction:
- ADX > 25 + (+DI > -DI) = LONG only
- ADX > 25 + (-DI > +DI) = SHORT only
- ADX < 25 = No trade (wait for clear trend)

Compare this with hedging strategy to see which performs better.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.binance_client import BinanceClient
from config.trading_config import DCA_CONFIG, STRATEGY_CONFIG, SYMBOL_SETTINGS


class TrendDirectionBacktester:
    def __init__(self, symbol: str, start_balance: float = 100.0):
        self.symbol = symbol
        self.start_balance = start_balance
        self.balance = start_balance
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x

        # Get symbol-specific settings
        symbol_config = SYMBOL_SETTINGS.get(symbol, {})
        self.tp_roi = symbol_config.get("tp_roi", DCA_CONFIG["take_profit_roi"])
        self.sl_roi = DCA_CONFIG["stop_loss_roi"]  # 90%
        self.dca_levels = symbol_config.get("dca_levels", DCA_CONFIG["levels"])

        # DCA budget allocation percentages (use FULL budget, not split)
        self.dca_pcts = [
            DCA_CONFIG.get("initial_entry_pct", 0.10),  # 10%
            DCA_CONFIG.get("dca1_pct", 0.15),           # 15%
            DCA_CONFIG.get("dca2_pct", 0.20),           # 20%
            DCA_CONFIG.get("dca3_pct", 0.25),           # 25%
            DCA_CONFIG.get("dca4_pct", 0.30),           # 30%
        ]

        # Trend detection parameters
        self.adx_period = 14
        self.adx_entry_threshold = 25   # Need ADX > 25 to enter
        self.adx_strong_threshold = 40  # ADX > 40 = strong trend (block DCA on wrong side)

        # Position tracking
        self.position = None  # Only ONE position at a time

        # Stats
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_balance = start_balance

        # Trend stats
        self.long_entries = 0
        self.short_entries = 0
        self.no_trade_periods = 0
        self.trend_flips = 0

        # Stop for day
        self.stopped_for_day = False
        self.sl_hit_date = None
        self.days_stopped = 0

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> tuple:
        """Calculate ADX, +DI, -DI"""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0

        # Smoothed values
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period).mean()

        # +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()

        return adx, plus_di, minus_di

    def get_trend_direction(self, adx: float, plus_di: float, minus_di: float) -> str:
        """
        Determine trend direction based on ADX and DI.
        Returns: "LONG", "SHORT", or "NONE"
        """
        if adx < self.adx_entry_threshold:
            return "NONE"  # No clear trend

        if plus_di > minus_di:
            return "LONG"  # Uptrend
        else:
            return "SHORT"  # Downtrend

    def calculate_roi(self, entry_price: float, current_price: float, side: str) -> float:
        """Calculate ROI based on position side"""
        if side == "LONG":
            price_pct = (current_price - entry_price) / entry_price
        else:
            price_pct = (entry_price - current_price) / entry_price
        return price_pct * self.leverage

    def get_tp_price(self, entry_price: float, side: str, dca_level: int) -> float:
        """Calculate TP price"""
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
        """Calculate SL price"""
        sl_pct = self.sl_roi / self.leverage
        if side == "LONG":
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)

    def check_dca_trigger(self, current_price: float) -> bool:
        """Check if DCA should trigger"""
        if self.position is None:
            return False

        dca_level = self.position.get("dca_level", 0)
        if dca_level >= len(self.dca_levels):
            return False

        trigger_roi = self.dca_levels[dca_level]["trigger_roi"]
        current_roi = self.calculate_roi(self.position["entry_price"], current_price, self.position["side"])

        return current_roi <= trigger_roi

    def execute_dca(self, current_price: float):
        """Execute DCA"""
        dca_level = self.position.get("dca_level", 0)
        add_pct = self.dca_pcts[dca_level + 1] if dca_level + 1 < len(self.dca_pcts) else 0.15

        old_qty = self.position["quantity"]
        old_margin = self.position["margin"]
        add_margin = self.start_balance * add_pct  # Full budget (not split like hedge)
        add_qty = (add_margin * self.leverage) / current_price

        new_qty = old_qty + add_qty
        new_margin = old_margin + add_margin

        old_value = old_qty * self.position["entry_price"]
        add_value = add_qty * current_price
        new_entry = (old_value + add_value) / new_qty

        self.position["quantity"] = new_qty
        self.position["margin"] = new_margin
        self.position["entry_price"] = new_entry
        self.position["dca_level"] = dca_level + 1
        self.position["tp_price"] = self.get_tp_price(new_entry, self.position["side"], dca_level + 1)
        self.position["sl_price"] = self.get_sl_price(new_entry, self.position["side"])

    def open_position(self, side: str, price: float) -> dict:
        """Open new position"""
        initial_pct = self.dca_pcts[0]
        initial_margin = self.start_balance * initial_pct  # Full budget
        quantity = (initial_margin * self.leverage) / price

        return {
            "side": side,
            "entry_price": price,
            "quantity": quantity,
            "margin": initial_margin,
            "dca_level": 0,
            "tp_price": self.get_tp_price(price, side, 0),
            "sl_price": self.get_sl_price(price, side),
        }

    def close_position(self, exit_price: float, exit_type: str, timestamp):
        """Close position and record trade"""
        if self.position is None:
            return 0

        side = self.position["side"]
        if side == "LONG":
            price_change_pct = (exit_price - self.position["entry_price"]) / self.position["entry_price"]
        else:
            price_change_pct = (self.position["entry_price"] - exit_price) / self.position["entry_price"]

        roi = price_change_pct * self.leverage
        pnl = self.position["margin"] * roi

        # Cap loss at margin
        if pnl < -self.position["margin"]:
            pnl = -self.position["margin"]

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
            "side": side,
            "entry_price": self.position["entry_price"],
            "exit_price": exit_price,
            "quantity": self.position["quantity"],
            "margin": self.position["margin"],
            "dca_level": self.position["dca_level"],
            "pnl": pnl,
            "exit_type": exit_type,
            "balance": self.balance,
        })

        return pnl

    def check_new_day(self, timestamp) -> bool:
        """Check if new day to restart trading"""
        if not self.stopped_for_day:
            return False

        current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        if current_date > self.sl_hit_date:
            self.stopped_for_day = False
            self.sl_hit_date = None
            return True
        return False

    def get_historical_data(self, days: int = 30, interval: str = "1h"):
        """Fetch historical data"""
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")
        client = BinanceClient(testnet=False, use_demo=False)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        all_data = []
        current_start = start_time

        while current_start < end_time:
            df_chunk = client.get_klines(
                self.symbol,
                interval=interval,
                start_time=int(current_start.timestamp() * 1000),
                limit=1000
            )
            if df_chunk is None or len(df_chunk) == 0:
                break
            all_data.append(df_chunk)
            if len(df_chunk) < 1000:
                break
            last_time = df_chunk.index[-1]
            current_start = last_time + timedelta(hours=1)

        if not all_data:
            return None

        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df

    def run_backtest(self, df: pd.DataFrame):
        """Run the backtest"""
        print("\n" + "=" * 70)
        print("RUNNING ONE-DIRECTION TREND STRATEGY BACKTEST")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Starting Balance: ${self.start_balance:.2f}")
        print(f"Leverage: {self.leverage}x")
        print(f"TP ROI: {self.tp_roi*100:.0f}% | SL ROI: {self.sl_roi*100:.0f}%")
        print(f"Entry Threshold: ADX > {self.adx_entry_threshold}")
        print(f"Strategy: Trade ONLY in trend direction (no hedging)")
        print("=" * 70)

        # Calculate ADX
        adx_series, plus_di_series, minus_di_series = self.calculate_adx(df, self.adx_period)

        last_trend = "NONE"

        for i, (timestamp, row) in enumerate(df.iterrows()):
            if self.balance <= 0:
                print(f"[{timestamp}] LIQUIDATED!")
                break

            if i < self.adx_period * 2:
                continue

            high = row['high']
            low = row['low']
            close = row['close']

            adx = adx_series.iloc[i]
            plus_di = plus_di_series.iloc[i]
            minus_di = minus_di_series.iloc[i]

            if pd.isna(adx):
                continue

            # Check new day restart
            if self.check_new_day(timestamp):
                print(f"[{timestamp}] >>> NEW DAY - Restarting trading")
                last_trend = "NONE"
                continue

            if self.stopped_for_day:
                continue

            # Get current trend direction
            current_trend = self.get_trend_direction(adx, plus_di, minus_di)

            # Manage existing position
            if self.position:
                side = self.position["side"]

                # Check TP
                if side == "LONG" and high >= self.position["tp_price"]:
                    pnl = self.close_position(self.position["tp_price"], "TP", timestamp)
                    print(f"[{timestamp}] {side} TP @ ${self.position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.position['dca_level']}")
                    self.position = None
                    continue

                elif side == "SHORT" and low <= self.position["tp_price"]:
                    pnl = self.close_position(self.position["tp_price"], "TP", timestamp)
                    print(f"[{timestamp}] {side} TP @ ${self.position['tp_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.position['dca_level']}")
                    self.position = None
                    continue

                # Check SL
                if side == "LONG" and low <= self.position["sl_price"]:
                    pnl = self.close_position(self.position["sl_price"], "SL", timestamp)
                    print(f"[{timestamp}] {side} SL @ ${self.position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.position['dca_level']}")
                    self.position = None
                    self.stopped_for_day = True
                    self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                    self.days_stopped += 1
                    print(f"[{timestamp}] >>> STOPPED FOR THE DAY")
                    continue

                elif side == "SHORT" and high >= self.position["sl_price"]:
                    pnl = self.close_position(self.position["sl_price"], "SL", timestamp)
                    print(f"[{timestamp}] {side} SL @ ${self.position['sl_price']:.4f} | P&L: ${pnl:+.2f} | DCA: {self.position['dca_level']}")
                    self.position = None
                    self.stopped_for_day = True
                    self.sl_hit_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                    self.days_stopped += 1
                    print(f"[{timestamp}] >>> STOPPED FOR THE DAY")
                    continue

                # Check trend flip - close and re-enter in new direction
                if current_trend != "NONE" and current_trend != side:
                    # Trend flipped - close current position
                    pnl = self.close_position(close, "TREND_FLIP", timestamp)
                    print(f"[{timestamp}] TREND FLIP: Closed {side} @ ${close:.4f} | P&L: ${pnl:+.2f}")
                    self.position = None
                    self.trend_flips += 1
                    # Will enter new direction below

                # Check DCA (only if trend still supports our position)
                elif self.check_dca_trigger(close):
                    dca_level = self.position["dca_level"]

                    # Block DCA if trend is against us
                    if current_trend != "NONE" and current_trend != side:
                        print(f"[{timestamp}] DCA {dca_level+1} BLOCKED - Trend now {current_trend}, position is {side}")
                    else:
                        old_entry = self.position["entry_price"]
                        self.execute_dca(close)
                        print(f"[{timestamp}] {side} DCA {self.position['dca_level']} @ ${close:.4f} | Avg: ${old_entry:.4f} -> ${self.position['entry_price']:.4f}")

            # Open new position if no position and clear trend
            if self.position is None and current_trend != "NONE":
                self.position = self.open_position(current_trend, close)

                if current_trend == "LONG":
                    self.long_entries += 1
                else:
                    self.short_entries += 1

                print(f"[{timestamp}] ENTER {current_trend} @ ${close:.4f} | ADX: {adx:.1f} | +DI: {plus_di:.1f} | -DI: {minus_di:.1f}")

            elif self.position is None and current_trend == "NONE":
                self.no_trade_periods += 1

            last_trend = current_trend

        # Final results
        self.print_results(df)

        # Return results for comparison
        return {
            "symbol": self.symbol,
            "strategy": "TREND_DIRECTION",
            "balance": self.balance,
            "return_pct": (self.balance - self.start_balance) / self.start_balance * 100,
            "total_trades": len(self.trades),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": (self.total_wins / len(self.trades) * 100) if self.trades else 0,
            "max_drawdown": self.max_drawdown,
            "long_entries": self.long_entries,
            "short_entries": self.short_entries,
            "trend_flips": self.trend_flips,
            "days_stopped": self.days_stopped,
        }

    def print_results(self, df: pd.DataFrame):
        """Print backtest results"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS - ONE DIRECTION TREND STRATEGY")
        print("=" * 70)

        price_start = df['close'].iloc[0]
        price_end = df['close'].iloc[-1]
        price_change = (price_end - price_start) / price_start * 100

        total_trades = len(self.trades)
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\nMarket Performance:")
        print(f"  Price: ${price_start:.4f} -> ${price_end:.4f} ({price_change:+.2f}%)")

        print(f"\nStrategy Performance:")
        print(f"  Starting Balance: ${self.start_balance:.2f}")
        print(f"  Ending Balance:   ${self.balance:.2f}")
        print(f"  Total P&L:        ${self.total_pnl:+.2f}")
        print(f"  Total Return:     {((self.balance - self.start_balance) / self.start_balance * 100):+.2f}%")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Wins:             {self.total_wins}")
        print(f"  Losses:           {self.total_losses}")
        print(f"  Win Rate:         {win_rate:.1f}%")
        print(f"  Max Drawdown:     {self.max_drawdown:.2f}%")

        print(f"\nTrend Statistics:")
        print(f"  LONG Entries:     {self.long_entries}")
        print(f"  SHORT Entries:    {self.short_entries}")
        print(f"  Trend Flips:      {self.trend_flips}")
        print(f"  Days Stopped:     {self.days_stopped}")

        print("=" * 70)


def run_comparison_test():
    """Run both strategies and compare"""
    from backtest_enhanced_boost import EnhancedBoostBacktester

    BACKTEST_DAYS = 90
    symbols = ["DOTUSDT", "BNBUSDT", "AVAXUSDT", "SOLUSDT", "BTCUSDT"]

    print("=" * 100)
    print(f"STRATEGY COMPARISON: TREND DIRECTION vs HEDGING")
    print(f"Period: {BACKTEST_DAYS} days | Symbols: {', '.join(symbols)}")
    print("=" * 100)

    trend_results = []
    hedge_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING: {symbol}")
        print(f"{'='*80}")

        # Run TREND DIRECTION strategy
        print("\n--- TREND DIRECTION STRATEGY ---")
        trend_bt = TrendDirectionBacktester(symbol, start_balance=100.0)
        df = trend_bt.get_historical_data(days=BACKTEST_DAYS, interval="1h")

        if df is not None and len(df) > 0:
            trend_result = trend_bt.run_backtest(df)
            trend_results.append(trend_result)

            # Run HEDGING strategy (Enhanced Boost)
            print("\n--- HEDGING STRATEGY ---")
            hedge_bt = EnhancedBoostBacktester(symbol, start_balance=100.0)
            hedge_result = hedge_bt.run_backtest(df)
            hedge_result["symbol"] = symbol
            hedge_result["strategy"] = "HEDGING"
            hedge_results.append(hedge_result)

    # Print comparison
    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 120)

    print(f"\n{'Symbol':<12} {'Strategy':<18} {'Return':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'WinRate':<10} {'MaxDD':<10}")
    print("-" * 120)

    for i, symbol in enumerate(symbols):
        if i < len(trend_results) and i < len(hedge_results):
            tr = trend_results[i]
            hr = hedge_results[i]

            print(f"{symbol:<12} {'TREND_DIRECTION':<18} {tr['return_pct']:+.1f}%{'':<4} {tr['total_trades']:<8} {tr['wins']:<6} {tr['losses']:<8} {tr['win_rate']:.1f}%{'':<5} {tr['max_drawdown']:.1f}%")
            print(f"{'':<12} {'HEDGING':<18} {hr['return_pct']:+.1f}%{'':<4} {hr['total_trades']:<8} {hr['wins']:<6} {hr['losses']:<8} {hr['win_rate']:.1f}%{'':<5} {hr['max_drawdown']:.1f}%")
            print()

    # Totals
    if trend_results and hedge_results:
        trend_total_return = sum(r['return_pct'] for r in trend_results)
        hedge_total_return = sum(r['return_pct'] for r in hedge_results)
        trend_avg_dd = sum(r['max_drawdown'] for r in trend_results) / len(trend_results)
        hedge_avg_dd = sum(r['max_drawdown'] for r in hedge_results) / len(hedge_results)

        print("=" * 120)
        print(f"{'TOTAL':<12} {'TREND_DIRECTION':<18} {trend_total_return:+.1f}%{'':<4} {'':<8} {'':<6} {'':<8} {'':<10} {trend_avg_dd:.1f}% avg")
        print(f"{'':<12} {'HEDGING':<18} {hedge_total_return:+.1f}%{'':<4} {'':<8} {'':<6} {'':<8} {'':<10} {hedge_avg_dd:.1f}% avg")
        print("=" * 120)

        winner = "TREND DIRECTION" if trend_total_return > hedge_total_return else "HEDGING"
        print(f"\n>>> WINNER: {winner} <<<")


if __name__ == "__main__":
    run_comparison_test()
