#!/usr/bin/env python3
"""
Volatility Breakout Scalping Strategy for DOTUSDT
==================================================
Buy on breakouts from consolidation with momentum.
Designed for trending/volatile markets.

Strategy Logic:
- Detect consolidation (low ATR relative to recent)
- BUY when price breaks above consolidation high with volume
- SELL when price breaks below consolidation low with volume
- Quick TP at first sign of momentum exhaustion
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    trailing_stop: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")


class VolatilityBreakoutScalper:
    """
    Volatility Breakout Scalping Strategy
    Trades breakouts from consolidation zones
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
        self.atr_period = 14
        self.consolidation_lookback = 20
        self.volume_multiplier = 1.5  # Volume must be 1.5x average

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR, consolidation zones, and volume indicators"""
        df = df.copy()

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=self.atr_period).mean()

        # Relative ATR (current vs recent average)
        df["atr_avg"] = df["atr"].rolling(window=50).mean()
        df["atr_ratio"] = df["atr"] / df["atr_avg"]

        # Consolidation detection: Recent high/low range
        df["range_high"] = df["high"].rolling(window=self.consolidation_lookback).max()
        df["range_low"] = df["low"].rolling(window=self.consolidation_lookback).min()
        df["range_pct"] = (df["range_high"] - df["range_low"]) / df["close"]

        # Volume analysis
        df["volume_avg"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_avg"]

        # EMA for trend
        df["ema_fast"] = df["close"].ewm(span=8).mean()
        df["ema_slow"] = df["close"].ewm(span=21).mean()
        df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        # Breakout signals
        df["prev_high"] = df["high"].shift(1).rolling(window=10).max()
        df["prev_low"] = df["low"].shift(1).rolling(window=10).min()

        return df

    def fetch_data(self, days: int = 7) -> pd.DataFrame:
        """Fetch historical data"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        print(f"Fetching {self.symbol} data for {days} days...", flush=True)
        df = self.client.get_historical_klines(self.symbol, "1m", start_date, end_date)
        print(f"Got {len(df)} candles", flush=True)

        df = self.calculate_indicators(df)
        return df

    def run_backtest(self, df: pd.DataFrame, tp_roi: float, sl_roi: float,
                     use_trailing: bool = False) -> Dict:
        """
        Run volatility breakout backtest

        Args:
            df: Historical data with indicators
            tp_roi: Take profit ROI
            sl_roi: Stop loss ROI
            use_trailing: Use trailing stop
        """
        balance = self.initial_balance
        position: Optional[ScalpPosition] = None
        trades = []
        peak_balance = balance
        max_drawdown = 0.0

        tp_price_pct = tp_roi / self.leverage
        sl_price_pct = sl_roi / self.leverage
        trailing_pct = sl_roi / self.leverage / 2  # Trail at half SL distance

        entry_margin_pct = 0.20
        cooldown = 0
        cooldown_bars = 15

        for i in range(100, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            if pd.isna(bar["atr"]) or pd.isna(bar["prev_high"]):
                continue

            if cooldown > 0:
                cooldown -= 1

            current_price = bar["close"]
            high = bar["high"]
            low = bar["low"]

            # If in position, check exit
            if position is not None:
                exit_price = None
                exit_reason = None

                # Update trailing stop if enabled
                if use_trailing:
                    if position.side == "LONG":
                        position.highest_since_entry = max(position.highest_since_entry, high)
                        new_trail = position.highest_since_entry * (1 - trailing_pct)
                        if new_trail > position.trailing_stop:
                            position.trailing_stop = new_trail
                    else:
                        position.lowest_since_entry = min(position.lowest_since_entry, low)
                        new_trail = position.lowest_since_entry * (1 + trailing_pct)
                        if position.trailing_stop == 0 or new_trail < position.trailing_stop:
                            position.trailing_stop = new_trail

                if position.side == "LONG":
                    # Trailing stop
                    if use_trailing and position.trailing_stop > 0 and low <= position.trailing_stop:
                        exit_price = position.trailing_stop
                        exit_reason = "TRAIL"
                    # Fixed SL
                    elif low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    # TP
                    elif high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"
                else:  # SHORT
                    if use_trailing and position.trailing_stop > 0 and high >= position.trailing_stop:
                        exit_price = position.trailing_stop
                        exit_reason = "TRAIL"
                    elif high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "SL"
                    elif low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "TP"

                if exit_price:
                    if position.side == "LONG":
                        exit_price *= (1 - self.slippage)
                        pnl = (exit_price - position.entry_price) * position.quantity
                    else:
                        exit_price *= (1 + self.slippage)
                        pnl = (position.entry_price - exit_price) * position.quantity

                    pnl -= exit_price * position.quantity * self.commission
                    balance += position.margin_used + pnl

                    hold_time = (bar_time - position.entry_time).total_seconds() / 60
                    trades.append({
                        "pnl": pnl,
                        "pnl_pct": pnl / position.margin_used,
                        "exit_reason": exit_reason,
                        "hold_time_min": hold_time,
                        "side": position.side
                    })

                    position = None
                    cooldown = cooldown_bars

                    if balance > peak_balance:
                        peak_balance = balance
                    dd = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, dd)

            # Check for breakout entry
            elif position is None and cooldown == 0:
                side = None

                # Volume confirmation
                has_volume = bar["volume_ratio"] > self.volume_multiplier

                # LONG breakout: Price breaks above recent high with volume
                if high > bar["prev_high"] and has_volume and bar["trend"] == 1:
                    side = "LONG"

                # SHORT breakout: Price breaks below recent low with volume
                elif low < bar["prev_low"] and has_volume and bar["trend"] == -1:
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
                        highest_since_entry=high,
                        lowest_since_entry=low
                    )

        # Results
        if not trades:
            return {"total_trades": 0, "winning_trades": 0, "win_rate": 0,
                    "total_pnl": 0, "total_pnl_pct": 0, "max_drawdown": 0,
                    "avg_hold_time_min": 0, "trades": []}

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
            "final_balance": balance
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Volatility Breakout Scalping for DOTUSDT")
    parser.add_argument("--days", type=int, default=7, help="Days of data")

    args = parser.parse_args()

    scalper = VolatilityBreakoutScalper("DOTUSDT")
    df = scalper.fetch_data(days=args.days)

    print("", flush=True)
    print("=" * 70, flush=True)
    print("VOLATILITY BREAKOUT SCALPING - DOTUSDT", flush=True)
    print("=" * 70, flush=True)
    print("Strategy: Trade breakouts from consolidation with volume", flush=True)
    print("", flush=True)

    results = []

    print("Testing combinations:", flush=True)
    print("-" * 60, flush=True)

    tp_levels = [0.02, 0.03, 0.05, 0.08, 0.10]
    sl_levels = [0.05, 0.08, 0.10, 0.15]

    for tp in tp_levels:
        for sl in sl_levels:
            # Without trailing
            r = scalper.run_backtest(df, tp, sl, use_trailing=False)
            r["tp_roi"] = tp
            r["sl_roi"] = sl
            r["trailing"] = False
            results.append(r)

            # With trailing
            r2 = scalper.run_backtest(df, tp, sl, use_trailing=True)
            r2["tp_roi"] = tp
            r2["sl_roi"] = sl
            r2["trailing"] = True
            results.append(r2)

            print(f"TP={tp*100:4.0f}% SL={sl*100:4.0f}% | Fixed: {r['total_trades']:3d}tr "
                  f"Win:{r['win_rate']*100:5.1f}% ${r['total_pnl']:+8.0f} | "
                  f"Trail: {r2['total_trades']:3d}tr Win:{r2['win_rate']*100:5.1f}% ${r2['total_pnl']:+8.0f}",
                  flush=True)

    # Summary
    print("", flush=True)
    print("=" * 70, flush=True)
    print("TOP 10 BY P&L:", flush=True)
    print("=" * 70, flush=True)

    results.sort(key=lambda x: x["total_pnl"], reverse=True)

    for i, r in enumerate(results[:10], 1):
        trail = "TRAIL" if r["trailing"] else "FIXED"
        print(f"{i:2d}. TP={r['tp_roi']*100:4.0f}% SL={r['sl_roi']*100:4.0f}% {trail:5s} | "
              f"Trades:{r['total_trades']:4d} Win:{r['win_rate']*100:5.1f}% "
              f"PnL: ${r['total_pnl']:+10.2f}", flush=True)

    profitable = [r for r in results if r["total_pnl"] > 0]
    if profitable:
        print("", flush=True)
        print(f"PROFITABLE STRATEGIES: {len(profitable)}", flush=True)
        print("=" * 70, flush=True)
        for r in profitable[:5]:
            trail = "TRAIL" if r["trailing"] else "FIXED"
            print(f"TP={r['tp_roi']*100:.0f}% SL={r['sl_roi']*100:.0f}% {trail} | "
                  f"Win:{r['win_rate']*100:.1f}% PnL: ${r['total_pnl']:+.2f} "
                  f"Avg: ${r['avg_trade_pnl']:.2f}", flush=True)
    else:
        print("", flush=True)
        print("No profitable strategies in this period.", flush=True)

    print("", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
