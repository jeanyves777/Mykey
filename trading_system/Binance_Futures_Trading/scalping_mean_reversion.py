#!/usr/bin/env python3
"""
Mean Reversion Scalping Strategy for DOTUSDT
=============================================
Uses Bollinger Bands + RSI for entry, with DCA support.

Strategy Logic:
- BUY when price touches lower Bollinger Band AND RSI < 30 (oversold)
- SELL when price touches upper Bollinger Band AND RSI > 70 (overbought)
- Exit at mean (middle band) or TP/SL

This is a cleaner signal than momentum for scalping.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.binance_client import BinanceClient
from config.trading_config import STRATEGY_CONFIG, BACKTEST_CONFIG


@dataclass
class ScalpPosition:
    """Position for scalping"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    margin_used: float
    middle_band: float  # For mean-reversion exit


class MeanReversionScalper:
    """
    Mean Reversion Scalping Strategy
    Uses Bollinger Bands + RSI for entries
    """

    def __init__(self, symbol: str = "DOTUSDT"):
        self.symbol = symbol
        self.client = BinanceClient(testnet=True)

        # Config
        self.initial_balance = BACKTEST_CONFIG["initial_balance"]
        self.leverage = STRATEGY_CONFIG["leverage"]  # 20x
        self.commission = BACKTEST_CONFIG["commission_per_trade"]
        self.slippage = BACKTEST_CONFIG["slippage_pct"]

        # Strategy parameters
        self.bb_period = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands and RSI"""
        df = df.copy()

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        df["bb_std"] = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (self.bb_std * df["bb_std"])
        df["bb_lower"] = df["bb_middle"] - (self.bb_std * df["bb_std"])

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Band %B (position within bands: 0 = lower, 1 = upper)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def fetch_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical data"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        print(f"Fetching {self.symbol} data for {days} days...", flush=True)
        df = self.client.get_historical_klines(self.symbol, "1m", start_date, end_date)
        print(f"Got {len(df)} candles", flush=True)

        # Calculate indicators
        df = self.calculate_indicators(df)
        return df

    def run_backtest(self, df: pd.DataFrame, tp_roi: float, sl_roi: float,
                     exit_at_mean: bool = True) -> Dict:
        """
        Run mean-reversion backtest

        Args:
            df: Historical data with indicators
            tp_roi: Take profit ROI (e.g., 0.03 = 3%)
            sl_roi: Stop loss ROI (e.g., 0.10 = 10%)
            exit_at_mean: Exit when price returns to middle band
        """
        balance = self.initial_balance
        position: Optional[ScalpPosition] = None
        trades = []
        peak_balance = balance
        max_drawdown = 0.0

        # Convert ROI to price movement
        tp_price_pct = tp_roi / self.leverage
        sl_price_pct = sl_roi / self.leverage

        # Entry margin (20% of balance)
        entry_margin_pct = 0.20

        # Cooldown (bars after a trade)
        cooldown = 0
        cooldown_bars = 10

        for i in range(50, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            if pd.isna(bar["rsi"]) or pd.isna(bar["bb_lower"]):
                continue

            # Decrease cooldown
            if cooldown > 0:
                cooldown -= 1

            current_price = bar["close"]
            high = bar["high"]
            low = bar["low"]

            # If in position, check exit
            if position is not None:
                exit_price = None
                exit_reason = None

                if position.side == "LONG":
                    # Check SL
                    if low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    # Check TP
                    elif high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"
                    # Check mean reversion exit
                    elif exit_at_mean and high >= position.middle_band:
                        exit_price = position.middle_band
                        exit_reason = "MEAN"
                else:  # SHORT
                    if high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    elif low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"
                    elif exit_at_mean and low <= position.middle_band:
                        exit_price = position.middle_band
                        exit_reason = "MEAN"

                if exit_price:
                    # Apply slippage
                    if position.side == "LONG":
                        exit_price *= (1 - self.slippage)
                        pnl = (exit_price - position.entry_price) * position.quantity
                    else:
                        exit_price *= (1 + self.slippage)
                        pnl = (position.entry_price - exit_price) * position.quantity

                    # Commission
                    pnl -= exit_price * position.quantity * self.commission

                    # Update balance
                    balance += position.margin_used + pnl

                    hold_time = (bar_time - position.entry_time).total_seconds() / 60
                    trades.append({
                        "entry_time": str(position.entry_time),
                        "exit_time": str(bar_time),
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl / position.margin_used,
                        "exit_reason": exit_reason,
                        "hold_time_min": hold_time
                    })

                    position = None
                    cooldown = cooldown_bars

                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

            # If no position and not in cooldown, check entry
            elif position is None and cooldown == 0:
                side = None

                # LONG signal: Price at lower band AND RSI oversold
                if bar["bb_pct"] < 0.05 and bar["rsi"] < self.rsi_oversold:
                    side = "LONG"

                # SHORT signal: Price at upper band AND RSI overbought
                elif bar["bb_pct"] > 0.95 and bar["rsi"] > self.rsi_overbought:
                    side = "SHORT"

                if side:
                    margin = balance * entry_margin_pct
                    if margin <= 0:
                        continue

                    if side == "LONG":
                        entry_price = current_price * (1 + self.slippage)
                        sl = entry_price * (1 - sl_price_pct)
                        tp = entry_price * (1 + tp_price_pct)
                    else:
                        entry_price = current_price * (1 - self.slippage)
                        sl = entry_price * (1 + sl_price_pct)
                        tp = entry_price * (1 - tp_price_pct)

                    position_value = margin * self.leverage
                    quantity = position_value / entry_price

                    # Deduct margin and entry commission
                    balance -= margin
                    balance -= entry_price * quantity * self.commission

                    position = ScalpPosition(
                        symbol=self.symbol,
                        side=side,
                        entry_price=entry_price,
                        quantity=quantity,
                        entry_time=bar_time,
                        stop_loss=sl,
                        take_profit=tp,
                        margin_used=margin,
                        middle_band=bar["bb_middle"]
                    )

        # Calculate results
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "avg_trade_pnl": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "avg_hold_time_min": 0,
                "trades": []
            }

        winners = [t for t in trades if t["pnl"] > 0]
        losers = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        gross_profit = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "win_rate": len(winners) / len(trades),
            "total_pnl": total_pnl,
            "total_pnl_pct": (balance - self.initial_balance) / self.initial_balance,
            "avg_trade_pnl": total_pnl / len(trades),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "max_drawdown": max_drawdown,
            "avg_hold_time_min": sum(t["hold_time_min"] for t in trades) / len(trades),
            "trades": trades,
            "final_balance": balance
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mean Reversion Scalping for DOTUSDT")
    parser.add_argument("--days", type=int, default=7, help="Days of data (default: 7)")

    args = parser.parse_args()

    scalper = MeanReversionScalper("DOTUSDT")
    df = scalper.fetch_data(days=args.days)

    print("", flush=True)
    print("=" * 70, flush=True)
    print("MEAN REVERSION SCALPING OPTIMIZATION - DOTUSDT", flush=True)
    print("=" * 70, flush=True)
    print("Strategy: Buy at lower BB + RSI<30, Sell at upper BB + RSI>70", flush=True)
    print("", flush=True)

    results = []

    # Test different TP/SL combinations
    print("Testing TP/SL combinations:", flush=True)
    print("-" * 60, flush=True)

    tp_levels = [0.02, 0.03, 0.05, 0.08, 0.10]  # 2% to 10% ROI
    sl_levels = [0.05, 0.10, 0.15, 0.20]  # 5% to 20% ROI

    for tp in tp_levels:
        for sl in sl_levels:
            # Test with mean exit
            r = scalper.run_backtest(df, tp, sl, exit_at_mean=True)
            r["tp_roi"] = tp
            r["sl_roi"] = sl
            r["exit_mode"] = "mean"
            results.append(r)
            print(f"TP={tp*100:4.0f}% SL={sl*100:4.0f}% MEAN | Trades:{r['total_trades']:4d} "
                  f"Win:{r['win_rate']*100:5.1f}% PnL: ${r['total_pnl']:+10.2f}", flush=True)

            # Test without mean exit (TP/SL only)
            r2 = scalper.run_backtest(df, tp, sl, exit_at_mean=False)
            r2["tp_roi"] = tp
            r2["sl_roi"] = sl
            r2["exit_mode"] = "tp_sl"
            results.append(r2)
            print(f"TP={tp*100:4.0f}% SL={sl*100:4.0f}% TP/SL| Trades:{r2['total_trades']:4d} "
                  f"Win:{r2['win_rate']*100:5.1f}% PnL: ${r2['total_pnl']:+10.2f}", flush=True)

    # Summary
    print("", flush=True)
    print("=" * 70, flush=True)
    print("TOP 10 STRATEGIES BY P&L:", flush=True)
    print("=" * 70, flush=True)

    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    for i, r in enumerate(results[:10], 1):
        print(f"{i:2d}. TP={r['tp_roi']*100:4.0f}% SL={r['sl_roi']*100:4.0f}% {r['exit_mode']:5s} | "
              f"Trades:{r['total_trades']:4d} Win:{r['win_rate']*100:5.1f}% "
              f"PnL: ${r['total_pnl']:+10.2f} DD:{r['max_drawdown']*100:5.1f}%", flush=True)

    # Best profitable strategies
    profitable = [r for r in results if r["total_pnl"] > 0]
    if profitable:
        print("", flush=True)
        print("=" * 70, flush=True)
        print("PROFITABLE STRATEGIES:", flush=True)
        print("=" * 70, flush=True)
        for i, r in enumerate(profitable, 1):
            print(f"{i:2d}. TP={r['tp_roi']*100:4.0f}% SL={r['sl_roi']*100:4.0f}% {r['exit_mode']:5s} | "
                  f"Trades:{r['total_trades']:4d} Win:{r['win_rate']*100:5.1f}% "
                  f"PnL: ${r['total_pnl']:+10.2f} Avg: ${r['avg_trade_pnl']:.2f}", flush=True)
    else:
        print("", flush=True)
        print("No profitable strategies found in this period.", flush=True)
        print("Market conditions may not be favorable for mean reversion.", flush=True)

    print("", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
