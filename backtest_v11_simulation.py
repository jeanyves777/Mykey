#!/usr/bin/env python3
"""
V11 Backtest Simulation - Replay Historical Trades with New Parameters

This script:
1. Loads completed trades from crypto_trade_logs
2. Fetches 1-minute bar data for the trade period from Alpaca
3. Replays each trade with V10 (old) and V11 (new) parameters
4. Compares the results

Requires:
- Alpaca API credentials configured
- Historical trades in ~/.thevolumeai/crypto_trade_logs/
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sys

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

from trading_system.config import PaperTradingConfig


@dataclass
class SimulationResult:
    """Result of simulating a trade with given parameters"""
    trade_id: str
    symbol: str
    entry_time: str
    entry_price: float

    # V10 (old) results
    v10_exit_price: float
    v10_exit_reason: str
    v10_pnl_pct: float
    v10_hold_minutes: float

    # V11 (new) results
    v11_exit_price: float
    v11_exit_reason: str
    v11_pnl_pct: float
    v11_hold_minutes: float


# V10 Parameters (old)
V10_CONFIG = {
    "target_profit_pct": 1.5,
    "stop_loss_pct": 1.0,
    "trailing_stop_pct": 0.5,
    "trailing_activation_pct": 0.8,
    "max_hold_minutes": 60,
    "min_entry_score": 7,
}

# V13 Parameters - NO TIME EXIT, only TP/SL
V11_CONFIG = {
    "target_profit_pct": 0.5,  # Tighter TP - get out quickly
    "stop_loss_pct": 1.5,      # Reasonable SL
    "trailing_stop_pct": 0.0,  # Disable trailing - just use fixed TP
    "trailing_activation_pct": 999,  # Never activate
    "max_hold_minutes": 999,   # No time exit - wait for TP or SL
    "min_entry_score": 6,
}

# Per-symbol V17 parameters - ALL SYMBOLS ENABLED for data collection
# TP = 1.2% (net +0.7% after fees)
# SL = 1.0%
V11_SYMBOL_PARAMS = {
    "BTC/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
    "ETH/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
    "SOL/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
    "DOGE/USD": {"target_profit_pct": 1.3, "stop_loss_pct": 1.1, "min_entry_score": 5},
    "AVAX/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
    "LINK/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
    "DOT/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
    "SHIB/USD": {"target_profit_pct": 1.3, "stop_loss_pct": 1.1, "min_entry_score": 5},
    "LTC/USD": {"target_profit_pct": 1.2, "stop_loss_pct": 1.0, "min_entry_score": 5},
}


def load_trade_logs() -> List[Dict]:
    """Load all completed crypto trades from logs"""
    log_dir = Path.home() / ".thevolumeai" / "crypto_trade_logs"
    all_trades = []

    for json_file in log_dir.glob("crypto_trades_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                completed = data.get('completed_trades', [])
                all_trades.extend(completed)
                print(f"Loaded {len(completed)} completed trades from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return all_trades


def get_alpaca_client():
    """Get Alpaca client for historical data"""
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        config = PaperTradingConfig.load()
        if not config.is_configured():
            print("ERROR: Alpaca not configured")
            return None

        client = CryptoHistoricalDataClient(config.api_key, config.api_secret)
        return client
    except ImportError:
        print("ERROR: alpaca-py not installed")
        return None


def fetch_bars_for_trade(client, symbol: str, entry_time: str, max_minutes: int = 120) -> List[Dict]:
    """Fetch 1-minute bars for the trade period"""
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import timezone

    # Parse entry time
    entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

    # Fetch bars from entry to entry + max_minutes
    end_dt = entry_dt + timedelta(minutes=max_minutes)

    # Alpaca crypto API wants symbols WITH slash (BTC/USD)
    # The symbol should already be in this format
    alpaca_symbol = symbol  # Keep as BTC/USD

    try:
        request = CryptoBarsRequest(
            symbol_or_symbols=alpaca_symbol,
            timeframe=TimeFrame.Minute,
            start=entry_dt,
            end=end_dt
        )

        barset = client.get_crypto_bars(request)

        # Access via .data attribute (BarSet object)
        if hasattr(barset, 'data') and alpaca_symbol in barset.data:
            bar_list = []
            for bar in barset.data[alpaca_symbol]:
                bar_list.append({
                    'timestamp': bar.timestamp.isoformat(),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),
                })
            return bar_list
        return []

    except Exception as e:
        print(f"  Error fetching bars for {symbol}: {e}")
        return []


def simulate_trade(
    entry_price: float,
    bars: List[Dict],
    tp_pct: float,
    sl_pct: float,
    trailing_pct: float,
    trailing_activation_pct: float,
    max_hold_minutes: int,
    fee_pct: float = 0.25
) -> Tuple[float, str, float, float]:
    """
    Simulate a trade through bar data with given TP/SL parameters.

    Returns: (exit_price, exit_reason, pnl_pct, hold_minutes)
    """
    if not bars:
        return entry_price, "NO_DATA", 0.0, 0.0

    # Calculate target levels
    tp_price = entry_price * (1 + tp_pct / 100)
    sl_price = entry_price * (1 - sl_pct / 100)
    trailing_activation_price = entry_price * (1 + trailing_activation_pct / 100)

    trailing_active = False
    trailing_stop_price = 0.0
    highest_price = entry_price

    for i, bar in enumerate(bars):
        minutes_held = i + 1  # Each bar is 1 minute

        high = bar['high']
        low = bar['low']
        close = bar['close']

        # Update highest price
        if high > highest_price:
            highest_price = high

        # Check stop loss first (using low of bar)
        if low <= sl_price:
            pnl_pct = ((sl_price / entry_price) - 1) * 100 - (fee_pct * 2)
            return sl_price, "STOP_LOSS", pnl_pct, minutes_held

        # Check take profit (using high of bar)
        if high >= tp_price:
            pnl_pct = ((tp_price / entry_price) - 1) * 100 - (fee_pct * 2)
            return tp_price, "TAKE_PROFIT", pnl_pct, minutes_held

        # Check trailing stop activation
        if not trailing_active and high >= trailing_activation_price:
            trailing_active = True
            trailing_stop_price = highest_price * (1 - trailing_pct / 100)

        # Update and check trailing stop
        if trailing_active:
            # Update trailing stop if price made new high
            new_trailing = highest_price * (1 - trailing_pct / 100)
            if new_trailing > trailing_stop_price:
                trailing_stop_price = new_trailing

            # Check if trailing stop hit
            if low <= trailing_stop_price:
                pnl_pct = ((trailing_stop_price / entry_price) - 1) * 100 - (fee_pct * 2)
                return trailing_stop_price, "TRAILING_STOP", pnl_pct, minutes_held

        # Check max hold time
        if minutes_held >= max_hold_minutes:
            pnl_pct = ((close / entry_price) - 1) * 100 - (fee_pct * 2)
            return close, "TIME_EXIT", pnl_pct, minutes_held

    # If we ran out of bars, exit at last close
    last_close = bars[-1]['close']
    pnl_pct = ((last_close / entry_price) - 1) * 100 - (fee_pct * 2)
    return last_close, "END_OF_DATA", pnl_pct, len(bars)


def run_backtest():
    """Main backtest function"""
    print("\n" + "=" * 70)
    print("V11 BACKTEST SIMULATION - Replay Trades with New Parameters")
    print("=" * 70)

    # Load trades
    trades = load_trade_logs()
    if not trades:
        print("No trades found!")
        return

    print(f"\nLoaded {len(trades)} total trades")

    # Filter out trades without valid entry data
    valid_trades = [t for t in trades if t.get('entry_price', 0) > 0 and t.get('entry_time')]
    print(f"Valid trades with entry data: {len(valid_trades)}")

    # Get Alpaca client
    client = get_alpaca_client()
    if not client:
        print("Cannot proceed without Alpaca client")
        return

    # Results storage
    results: List[SimulationResult] = []

    v10_total_pnl = 0.0
    v11_total_pnl = 0.0
    v10_wins = 0
    v11_wins = 0
    trades_skipped = 0

    print(f"\nSimulating {len(valid_trades)} trades...")
    print("-" * 70)

    for i, trade in enumerate(valid_trades):
        symbol = trade['symbol']
        entry_price = trade['entry_price']
        entry_time = trade['entry_time']
        signal_score = trade.get('signal_score', 0)

        print(f"\n[{i+1}/{len(valid_trades)}] {symbol} @ ${entry_price:.4f}")
        print(f"  Entry: {entry_time[:19]}, Score: {signal_score}")

        # V11 entry filter - would we even take this trade?
        v11_params = V11_SYMBOL_PARAMS.get(symbol, V11_CONFIG)
        v11_min_score = v11_params.get('min_entry_score', V11_CONFIG['min_entry_score'])

        v11_would_enter = signal_score >= v11_min_score

        # Fetch historical bars - extended to 24 hours to ensure TP/SL hit
        bars = fetch_bars_for_trade(client, symbol, entry_time, max_minutes=1440)

        if not bars:
            print(f"  No bar data available - skipping")
            trades_skipped += 1
            continue

        print(f"  Fetched {len(bars)} 1-minute bars")

        # Simulate with V10 parameters
        v10_exit, v10_reason, v10_pnl, v10_hold = simulate_trade(
            entry_price=entry_price,
            bars=bars,
            tp_pct=V10_CONFIG['target_profit_pct'],
            sl_pct=V10_CONFIG['stop_loss_pct'],
            trailing_pct=V10_CONFIG['trailing_stop_pct'],
            trailing_activation_pct=V10_CONFIG['trailing_activation_pct'],
            max_hold_minutes=V10_CONFIG['max_hold_minutes'],
        )

        # Simulate with V11 parameters (symbol-specific if available)
        v11_tp = v11_params.get('target_profit_pct', V11_CONFIG['target_profit_pct'])
        v11_sl = v11_params.get('stop_loss_pct', V11_CONFIG['stop_loss_pct'])

        if v11_would_enter:
            v11_exit, v11_reason, v11_pnl, v11_hold = simulate_trade(
                entry_price=entry_price,
                bars=bars,
                tp_pct=v11_tp,
                sl_pct=v11_sl,
                trailing_pct=V11_CONFIG['trailing_stop_pct'],
                trailing_activation_pct=V11_CONFIG['trailing_activation_pct'],
                max_hold_minutes=V11_CONFIG['max_hold_minutes'],
            )
        else:
            # V11 would have skipped this trade
            v11_exit, v11_reason, v11_pnl, v11_hold = 0.0, "SKIPPED (score too low)", 0.0, 0.0

        # Print results
        print(f"  V10: {v10_reason} @ ${v10_exit:.4f} | P&L: {v10_pnl:+.2f}% | Hold: {v10_hold:.0f}min")
        print(f"  V11: {v11_reason} @ ${v11_exit:.4f} | P&L: {v11_pnl:+.2f}% | Hold: {v11_hold:.0f}min")

        # Track results
        # Estimate dollar P&L (assuming $500 position)
        position_value = 500.0
        v10_dollar_pnl = position_value * v10_pnl / 100
        v11_dollar_pnl = position_value * v11_pnl / 100

        v10_total_pnl += v10_dollar_pnl
        v11_total_pnl += v11_dollar_pnl

        if v10_pnl > 0:
            v10_wins += 1
        if v11_pnl > 0:
            v11_wins += 1

        result = SimulationResult(
            trade_id=trade.get('trade_id', ''),
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            v10_exit_price=v10_exit,
            v10_exit_reason=v10_reason,
            v10_pnl_pct=v10_pnl,
            v10_hold_minutes=v10_hold,
            v11_exit_price=v11_exit,
            v11_exit_reason=v11_reason,
            v11_pnl_pct=v11_pnl,
            v11_hold_minutes=v11_hold,
        )
        results.append(result)

    # Final summary
    trades_simulated = len(results)

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    print(f"\nTrades simulated: {trades_simulated}")
    print(f"Trades skipped (no data): {trades_skipped}")

    print(f"\n--- V10 (Old Parameters) ---")
    print(f"  TP: 1.5% | SL: 1.0% | Trailing: 0.5% @ 0.8%")
    print(f"  Total P&L: ${v10_total_pnl:.2f}")
    print(f"  Win Rate: {v10_wins}/{trades_simulated} ({v10_wins/trades_simulated*100:.1f}%)" if trades_simulated > 0 else "  No trades")

    print(f"\n--- V11 (New Parameters) ---")
    print(f"  TP: 0.6-1.0% | SL: 1.5-2.5% | Trailing: 0.3% @ 0.5%")
    print(f"  Min Score: 6-8 (symbol-specific)")
    print(f"  Total P&L: ${v11_total_pnl:.2f}")
    v11_taken = len([r for r in results if r.v11_exit_reason != "SKIPPED (score too low)"])
    v11_skipped = trades_simulated - v11_taken
    print(f"  Trades taken: {v11_taken} ({v11_skipped} filtered by score)")
    print(f"  Win Rate: {v11_wins}/{trades_simulated} ({v11_wins/trades_simulated*100:.1f}%)" if trades_simulated > 0 else "  No trades")

    print(f"\n--- Comparison ---")
    improvement = v11_total_pnl - v10_total_pnl
    print(f"  P&L Improvement: ${improvement:+.2f}")

    # Exit reason breakdown
    print(f"\n--- V10 Exit Reasons ---")
    v10_reasons = {}
    for r in results:
        reason = r.v10_exit_reason
        if reason not in v10_reasons:
            v10_reasons[reason] = 0
        v10_reasons[reason] += 1
    for reason, count in sorted(v10_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    print(f"\n--- V11 Exit Reasons ---")
    v11_reasons = {}
    for r in results:
        reason = r.v11_exit_reason
        if reason not in v11_reasons:
            v11_reasons[reason] = 0
        v11_reasons[reason] += 1
    for reason, count in sorted(v11_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Detailed analysis of winners vs losers
    print(f"\n--- DETAILED ANALYSIS ---")

    # By symbol performance
    symbol_results = {}
    for r in results:
        sym = r.symbol
        if sym not in symbol_results:
            symbol_results[sym] = {"wins": 0, "losses": 0, "pnl": 0}
        if r.v11_pnl_pct > 0:
            symbol_results[sym]["wins"] += 1
        else:
            symbol_results[sym]["losses"] += 1
        symbol_results[sym]["pnl"] += r.v11_pnl_pct

    print("\nV11 Performance by Symbol:")
    for sym, data in sorted(symbol_results.items(), key=lambda x: -x[1]["pnl"]):
        total = data["wins"] + data["losses"]
        win_rate = data["wins"] / total * 100 if total > 0 else 0
        print(f"  {sym}: {data['wins']}W/{data['losses']}L ({win_rate:.0f}%) | P&L: {data['pnl']:+.1f}%")

    # Losers analysis
    print("\n--- V11 STOP_LOSS TRADES (analyze what went wrong) ---")
    losers = [r for r in results if r.v11_exit_reason == "STOP_LOSS"]
    for r in losers:
        print(f"  {r.symbol} @ {r.entry_time[:16]} | Score: ? | P&L: {r.v11_pnl_pct:+.2f}%")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_backtest()
