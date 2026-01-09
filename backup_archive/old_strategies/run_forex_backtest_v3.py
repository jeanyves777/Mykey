"""
Forex Strategy Backtest - TREND-FOLLOWING V3 (Breakout + Pullback)

Tests the current strategy with:
- $1 per pip position sizing
- Delayed trailing stop activation (only at trigger level)
- Pair-specific TP/SL settings
- High volatility pair validation

Uses OANDA historical data for accurate simulation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Import strategy and config
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params, PAIR_VOLATILITY
from trading_system.Forex_Trading.engine.oanda_client import OandaClient


class ForexBacktester:
    """Backtests the Trend-Following V3 strategy with realistic execution"""

    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Initialize OANDA client for historical data
        self.client = OandaClient()

        # Initialize strategy
        self.strategy = ForexScalpingStrategy(
            instruments=list(PAIR_VOLATILITY.keys()),
            max_trades_per_day=10,
            session_cooldown=True
        )

        # Trade tracking
        self.trades = []
        self.open_trade = None

        # Performance metrics
        self.equity_curve = []

    def calculate_pip_value(self, instrument: str, current_price: float) -> float:
        """Calculate pip value for $1 per pip sizing"""
        if instrument in ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"]:
            return 10000  # XXX/USD pairs
        elif instrument == "USD_JPY":
            return int(current_price / 0.01)  # ~15,500 at 155.00
        elif instrument == "USD_CHF":
            return int(current_price / 0.0001)  # ~8,000 at 0.80
        elif instrument == "USD_CAD":
            return int(current_price / 0.0001)  # ~13,800 at 1.38
        else:
            return 10000  # Default

    def calculate_pips(self, instrument: str, entry: float, exit: float, direction: str) -> float:
        """Calculate pips gained/lost"""
        pip_mult = 100 if "JPY" in instrument else 10000
        if direction == "BUY":
            return (exit - entry) * pip_mult
        else:
            return (entry - exit) * pip_mult

    def simulate_trade(self, instrument: str, direction: str, entry_price: float,
                       entry_time: datetime, df_5min: pd.DataFrame, params: dict) -> dict:
        """
        Simulate a single trade with:
        - Fixed SL/TP
        - Delayed trailing stop activation (only at trigger level)

        Returns trade result dict
        """
        tp_pips = params['tp_pips']
        sl_pips = params['sl_pips']
        trail_trigger_pips = params['trail_trigger_pips']
        trail_distance_pips = params['trail_distance_pips']

        pip_mult = 100 if "JPY" in instrument else 10000
        pip_value = 1 / pip_mult  # Price move per pip

        # Calculate SL/TP prices
        if direction == "BUY":
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
            trail_trigger_price = entry_price + (trail_trigger_pips * pip_value)
        else:
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)
            trail_trigger_price = entry_price - (trail_trigger_pips * pip_value)

        # Position size for $1/pip
        units = self.calculate_pip_value(instrument, entry_price)

        # Simulate price movement
        trailing_active = False
        trailing_stop = None
        exit_price = None
        exit_reason = None
        exit_time = None
        max_profit_pips = 0

        # Get candles AFTER entry
        entry_idx = None
        for i, row in df_5min.iterrows():
            if row['time'] >= entry_time:
                entry_idx = i
                break

        if entry_idx is None:
            return None

        # Simulate bar-by-bar
        for idx in range(df_5min.index.get_loc(entry_idx), len(df_5min)):
            candle = df_5min.iloc[idx]
            high = candle['high']
            low = candle['low']
            close = candle['close']
            candle_time = candle['time']

            # Calculate current profit in pips
            if direction == "BUY":
                current_profit_pips = (high - entry_price) * pip_mult
                worst_price = low
            else:
                current_profit_pips = (entry_price - low) * pip_mult
                worst_price = high

            max_profit_pips = max(max_profit_pips, current_profit_pips)

            # Check STOP LOSS first
            if direction == "BUY":
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                    exit_time = candle_time
                    break
            else:
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                    exit_time = candle_time
                    break

            # Check TAKE PROFIT
            if direction == "BUY":
                if high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    exit_time = candle_time
                    break
            else:
                if low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    exit_time = candle_time
                    break

            # DELAYED TRAILING STOP ACTIVATION
            # Only activate when profit reaches trigger level
            if not trailing_active:
                if direction == "BUY":
                    if high >= trail_trigger_price:
                        trailing_active = True
                        trailing_stop = high - (trail_distance_pips * pip_value)
                else:
                    if low <= trail_trigger_price:
                        trailing_active = True
                        trailing_stop = low + (trail_distance_pips * pip_value)
            else:
                # Update trailing stop
                if direction == "BUY":
                    new_trail = high - (trail_distance_pips * pip_value)
                    if new_trail > trailing_stop:
                        trailing_stop = new_trail
                    # Check if trailing stop hit
                    if low <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAIL"
                        exit_time = candle_time
                        break
                else:
                    new_trail = low + (trail_distance_pips * pip_value)
                    if new_trail < trailing_stop:
                        trailing_stop = new_trail
                    # Check if trailing stop hit
                    if high >= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAIL"
                        exit_time = candle_time
                        break

        # If no exit after all candles (trade still open at end of data)
        if exit_price is None:
            exit_price = df_5min.iloc[-1]['close']
            exit_reason = "EOD"  # End of data
            exit_time = df_5min.iloc[-1]['time']

        # Calculate P&L
        pips = self.calculate_pips(instrument, entry_price, exit_price, direction)
        pnl = pips * 1.0  # $1 per pip

        return {
            'instrument': instrument,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pips': pips,
            'pnl': pnl,
            'max_profit_pips': max_profit_pips,
            'trailing_activated': trailing_active,
            'units': units,
            'tp_pips': tp_pips,
            'sl_pips': sl_pips
        }

    def run_backtest(self, instrument: str, days: int = 30):
        """Run backtest for a single instrument over specified days"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING {instrument} - Last {days} days")
        print(f"{'='*60}")

        # Get pair-specific settings
        params = get_scalping_params(instrument)
        print(f"Settings: TP={params['tp_pips']}p, SL={params['sl_pips']}p, "
              f"Trail trigger={params['trail_trigger_pips']}p, Trail dist={params['trail_distance_pips']}p")

        try:
            # Fetch historical data
            print(f"Fetching historical data...")

            # Get 1-min data and convert to DataFrame
            candles_1min = self.client.get_candles(instrument, "M1", count=5000)
            if candles_1min is None or len(candles_1min) < 100:
                print(f"  [ERROR] Insufficient 1-min data for {instrument}")
                return []
            df_1min = pd.DataFrame(candles_1min)

            # Get 5-min data and convert to DataFrame
            candles_5min = self.client.get_candles(instrument, "M5", count=2000)
            if candles_5min is None or len(candles_5min) < 100:
                print(f"  [ERROR] Insufficient 5-min data for {instrument}")
                return []
            df_5min = pd.DataFrame(candles_5min)

            # Get 15-min data and convert to DataFrame
            candles_15min = self.client.get_candles(instrument, "M15", count=800)
            if candles_15min is None or len(candles_15min) < 50:
                print(f"  [ERROR] Insufficient 15-min data for {instrument}")
                return []
            df_15min = pd.DataFrame(candles_15min)

            # Get 30-min data and convert to DataFrame
            candles_30min = self.client.get_candles(instrument, "M30", count=400)
            if candles_30min is None or len(candles_30min) < 50:
                print(f"  [ERROR] Insufficient 30-min data for {instrument}")
                return []
            df_30min = pd.DataFrame(candles_30min)

            print(f"  Data loaded: 1m={len(df_1min)}, 5m={len(df_5min)}, "
                  f"15m={len(df_15min)}, 30m={len(df_30min)}")

        except Exception as e:
            print(f"  [ERROR] Failed to fetch data: {e}")
            return []

        # Run through 5-min candles looking for entries
        trades = []
        cooldown_until = None
        trades_today = 0
        current_date = None

        # Skip first 50 candles for indicator warmup
        for i in range(50, len(df_5min) - 20):
            candle_time = df_5min.iloc[i]['time']

            # Reset daily counter
            candle_date = candle_time.date() if hasattr(candle_time, 'date') else candle_time
            if current_date != candle_date:
                current_date = candle_date
                trades_today = 0
                self.strategy.reset_daily_cooldowns()

            # Check cooldown
            if cooldown_until and candle_time < cooldown_until:
                continue

            # Get data slices up to current candle
            df_1min_slice = df_1min[df_1min['time'] <= candle_time].tail(100)
            df_5min_slice = df_5min.iloc[:i+1].tail(50)
            df_15min_slice = df_15min[df_15min['time'] <= candle_time].tail(50)
            df_30min_slice = df_30min[df_30min['time'] <= candle_time].tail(50)

            if len(df_1min_slice) < 20 or len(df_15min_slice) < 30 or len(df_30min_slice) < 30:
                continue

            # Check for entry signal
            current_price = df_5min.iloc[i]['close']

            # Use strategy's should_enter_trade method
            try:
                signal = self.strategy.should_enter_trade(
                    instrument=instrument,
                    df_1min=df_1min_slice,
                    df_5min=df_5min_slice,
                    df_15min=df_15min_slice,
                    df_30min=df_30min_slice,
                    current_positions=0,
                    trades_today=trades_today,
                    daily_pl_pct=0,
                    now=candle_time if isinstance(candle_time, datetime) else datetime.now(pytz.UTC)
                )
            except Exception as e:
                continue

            if signal['action'] in ['BUY', 'SELL']:
                direction = signal['action']
                entry_price = current_price
                entry_time = candle_time

                # Simulate trade
                df_5min_future = df_5min.iloc[i:]
                result = self.simulate_trade(
                    instrument=instrument,
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=entry_time,
                    df_5min=df_5min_future,
                    params=params
                )

                if result:
                    trades.append(result)
                    trades_today += 1
                    self.strategy.record_trade_session(instrument, candle_time if isinstance(candle_time, datetime) else datetime.now(pytz.UTC))

                    # Set cooldown (skip to after trade exit)
                    if result['exit_time']:
                        cooldown_until = result['exit_time']

                    # Print trade summary
                    won = "WIN" if result['pnl'] > 0 else "LOSS"
                    trail_info = " [TRAIL ACTIVATED]" if result['trailing_activated'] else ""
                    print(f"  {won} {direction} @ {entry_price:.5f} -> {result['exit_reason']} @ {result['exit_price']:.5f} "
                          f"| {result['pips']:+.1f} pips (${result['pnl']:+.2f}){trail_info}")

        return trades

    def run_full_backtest(self, days: int = 30):
        """Run backtest on all instruments"""
        print("\n" + "="*80)
        print("FOREX STRATEGY BACKTEST - TREND-FOLLOWING V3")
        print("="*80)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Position Sizing: $1 per pip (consistent across all pairs)")
        print(f"Trailing Stop: DELAYED ACTIVATION (only at trigger level)")
        print(f"Period: Last {days} days")
        print("="*80)

        all_trades = []

        instruments = list(PAIR_VOLATILITY.keys())

        for instrument in instruments:
            trades = self.run_backtest(instrument, days)
            all_trades.extend(trades)

        # Generate summary
        self.generate_summary(all_trades)

        return all_trades

    def generate_summary(self, trades: list):
        """Generate performance summary"""
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)

        if not trades:
            print("No trades executed during backtest period.")
            return

        # Overall stats
        total_trades = len(trades)
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] <= 0]

        total_pnl = sum(t['pnl'] for t in trades)
        total_pips = sum(t['pips'] for t in trades)

        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        avg_win = sum(t['pnl'] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t['pnl'] for t in losers) / len(losers) if losers else 0

        avg_win_pips = sum(t['pips'] for t in winners) / len(winners) if winners else 0
        avg_loss_pips = sum(t['pips'] for t in losers) / len(losers) if losers else 0

        profit_factor = abs(sum(t['pnl'] for t in winners) / sum(t['pnl'] for t in losers)) if losers and sum(t['pnl'] for t in losers) != 0 else float('inf')

        # Count trailing stop activations
        trail_activated = len([t for t in trades if t['trailing_activated']])

        # Exit reason breakdown
        exit_tp = len([t for t in trades if t['exit_reason'] == 'TP'])
        exit_sl = len([t for t in trades if t['exit_reason'] == 'SL'])
        exit_trail = len([t for t in trades if t['exit_reason'] == 'TRAIL'])
        exit_eod = len([t for t in trades if t['exit_reason'] == 'EOD'])

        print(f"\n[OVERALL PERFORMANCE]")
        print(f"-" * 40)
        print(f"Total Trades:     {total_trades}")
        print(f"Winners:          {len(winners)} ({win_rate:.1f}%)")
        print(f"Losers:           {len(losers)} ({100-win_rate:.1f}%)")
        print(f"")
        print(f"Total P&L:        ${total_pnl:+,.2f}")
        print(f"Total Pips:       {total_pips:+.1f}")
        print(f"")
        print(f"Avg Win:          ${avg_win:+.2f} ({avg_win_pips:+.1f} pips)")
        print(f"Avg Loss:         ${avg_loss:+.2f} ({avg_loss_pips:+.1f} pips)")
        print(f"")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"")
        print(f"Exit Reasons:")
        print(f"  Take Profit:    {exit_tp} ({exit_tp/total_trades*100:.1f}%)")
        print(f"  Stop Loss:      {exit_sl} ({exit_sl/total_trades*100:.1f}%)")
        print(f"  Trailing Stop:  {exit_trail} ({exit_trail/total_trades*100:.1f}%)")
        print(f"  End of Data:    {exit_eod} ({exit_eod/total_trades*100:.1f}%)")
        print(f"")
        print(f"Trailing Stop Activated: {trail_activated}/{total_trades} trades ({trail_activated/total_trades*100:.1f}%)")

        # Per-instrument breakdown
        print(f"\n[PERFORMANCE BY INSTRUMENT]")
        print(f"-" * 60)
        print(f"{'Instrument':<12} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Pips':<10} {'PF':<8}")
        print(f"-" * 60)

        instruments = set(t['instrument'] for t in trades)
        for inst in sorted(instruments):
            inst_trades = [t for t in trades if t['instrument'] == inst]
            inst_winners = [t for t in inst_trades if t['pnl'] > 0]
            inst_losers = [t for t in inst_trades if t['pnl'] <= 0]

            inst_pnl = sum(t['pnl'] for t in inst_trades)
            inst_pips = sum(t['pips'] for t in inst_trades)
            inst_wr = len(inst_winners) / len(inst_trades) * 100 if inst_trades else 0

            inst_pf = abs(sum(t['pnl'] for t in inst_winners) / sum(t['pnl'] for t in inst_losers)) if inst_losers and sum(t['pnl'] for t in inst_losers) != 0 else float('inf')

            pf_str = f"{inst_pf:.2f}" if inst_pf != float('inf') else "INF"

            print(f"{inst:<12} {len(inst_trades):<8} {inst_wr:<7.1f}% ${inst_pnl:<+10.2f} {inst_pips:<+9.1f} {pf_str:<8}")

        print(f"-" * 60)

        # Direction breakdown
        print(f"\n[PERFORMANCE BY DIRECTION]")
        print(f"-" * 40)

        buys = [t for t in trades if t['direction'] == 'BUY']
        sells = [t for t in trades if t['direction'] == 'SELL']

        buy_winners = len([t for t in buys if t['pnl'] > 0])
        sell_winners = len([t for t in sells if t['pnl'] > 0])

        buy_pnl = sum(t['pnl'] for t in buys)
        sell_pnl = sum(t['pnl'] for t in sells)

        print(f"BUY trades:   {len(buys):>3} | Win: {buy_winners/len(buys)*100 if buys else 0:.1f}% | P&L: ${buy_pnl:+.2f}")
        print(f"SELL trades:  {len(sells):>3} | Win: {sell_winners/len(sells)*100 if sells else 0:.1f}% | P&L: ${sell_pnl:+.2f}")

        # Final balance
        final_balance = self.initial_balance + total_pnl
        roi = (total_pnl / self.initial_balance) * 100

        print(f"\n[ACCOUNT SUMMARY]")
        print(f"-" * 40)
        print(f"Initial Balance:  ${self.initial_balance:,.2f}")
        print(f"Final Balance:    ${final_balance:,.2f}")
        print(f"Net P&L:          ${total_pnl:+,.2f}")
        print(f"ROI:              {roi:+.2f}%")
        print("="*80)


def main():
    """Run the backtest"""
    print("\n" + "[*] Starting Forex Strategy Backtest...")

    backtester = ForexBacktester(initial_balance=10000)

    # Run backtest for last 30 days
    trades = backtester.run_full_backtest(days=30)

    print("\n[OK] Backtest complete!")
    print(f"Total trades analyzed: {len(trades)}")


if __name__ == "__main__":
    main()
