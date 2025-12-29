"""
Forex Strategy Backtest - Using LOCAL Historical Data (Sept-Dec 2025)

Tests the current strategy with:
- $1 per pip position sizing
- Delayed trailing stop activation (only at trigger level)
- Pair-specific TP/SL settings
- High volatility pair validation

Uses LOCAL CSV data for comprehensive 3+ month backtest.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# Import strategy and config
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.pair_specific_settings import get_scalping_params, PAIR_VOLATILITY


class LocalDataBacktester:
    """Backtests the Trend-Following V3 strategy using local CSV data"""

    # Map OANDA instrument names to CSV file prefixes
    INSTRUMENT_MAP = {
        "EUR_USD": "EURUSD",
        "GBP_USD": "GBPUSD",
        "USD_JPY": "USDJPY",
        "USD_CHF": "USDCHF",
        "USD_CAD": "USDCAD"
    }

    def __init__(self, data_dir: str, initial_balance: float = 10000):
        self.data_dir = Path(data_dir)
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Initialize strategy
        self.strategy = ForexScalpingStrategy(
            instruments=list(self.INSTRUMENT_MAP.keys()),
            max_trades_per_day=10,
            session_cooldown=True
        )

        # Trade tracking
        self.trades = []

    def load_csv_data(self, instrument: str, timeframe: str) -> pd.DataFrame:
        """
        Load CSV data for an instrument and timeframe.

        Args:
            instrument: OANDA format (e.g., "EUR_USD")
            timeframe: "1", "5", "15", "30", "60"
        """
        prefix = self.INSTRUMENT_MAP.get(instrument)
        if not prefix:
            print(f"  [ERROR] Unknown instrument: {instrument}")
            return None

        filename = f"{prefix}{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            print(f"  [ERROR] File not found: {filepath}")
            return None

        # Read CSV - tab separated, no header
        df = pd.read_csv(
            filepath,
            sep='\t',
            names=['time', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['time']
        )

        # Convert to timezone-aware datetime
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')

        return df

    def calculate_pip_value(self, instrument: str, current_price: float) -> float:
        """Calculate units for $1 per pip sizing"""
        if instrument in ["EUR_USD", "GBP_USD"]:
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
                       entry_idx: int, df_5min: pd.DataFrame, params: dict) -> dict:
        """
        Simulate a single trade with:
        - Fixed SL/TP
        - Delayed trailing stop activation (only at trigger level)
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
        entry_time = df_5min.iloc[entry_idx]['time']

        # Simulate bar-by-bar from entry
        for idx in range(entry_idx + 1, len(df_5min)):
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
                    if low <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAIL"
                        exit_time = candle_time
                        break
                else:
                    new_trail = low + (trail_distance_pips * pip_value)
                    if new_trail < trailing_stop:
                        trailing_stop = new_trail
                    if high >= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "TRAIL"
                        exit_time = candle_time
                        break

        # If no exit (trade still open at end of data)
        if exit_price is None:
            exit_price = df_5min.iloc[-1]['close']
            exit_reason = "EOD"
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

    def run_backtest(self, instrument: str):
        """Run backtest for a single instrument using local data"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING {instrument}")
        print(f"{'='*60}")

        # Get pair-specific settings
        params = get_scalping_params(instrument)
        print(f"Settings: TP={params['tp_pips']}p, SL={params['sl_pips']}p, "
              f"Trail trigger={params['trail_trigger_pips']}p, Trail dist={params['trail_distance_pips']}p")

        # Load all timeframe data
        print(f"Loading local CSV data...")

        df_1min = self.load_csv_data(instrument, "1")
        df_5min = self.load_csv_data(instrument, "5")
        df_15min = self.load_csv_data(instrument, "15")
        df_30min = self.load_csv_data(instrument, "30")

        if df_1min is None or df_5min is None or df_15min is None or df_30min is None:
            print(f"  [ERROR] Could not load all required data for {instrument}")
            return []

        # Get date range
        start_date = df_5min['time'].min()
        end_date = df_5min['time'].max()

        print(f"  Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Candles: 1m={len(df_1min)}, 5m={len(df_5min)}, 15m={len(df_15min)}, 30m={len(df_30min)}")

        # Run through 5-min candles looking for entries
        trades = []
        cooldown_until = None
        trades_today = 0
        current_date = None
        signals_checked = 0

        # Skip first 100 candles for indicator warmup
        print(f"  Scanning for signals...")

        for i in range(100, len(df_5min) - 20):
            candle_time = df_5min.iloc[i]['time']

            # Reset daily counter
            candle_date = candle_time.date()
            if current_date != candle_date:
                current_date = candle_date
                trades_today = 0
                self.strategy.reset_daily_cooldowns()

            # Check cooldown
            if cooldown_until is not None and candle_time < cooldown_until:
                continue

            # Get data slices up to current candle
            df_1min_slice = df_1min[df_1min['time'] <= candle_time].tail(100)
            df_5min_slice = df_5min.iloc[:i+1].tail(50)
            df_15min_slice = df_15min[df_15min['time'] <= candle_time].tail(50)
            df_30min_slice = df_30min[df_30min['time'] <= candle_time].tail(50)

            if len(df_1min_slice) < 20 or len(df_15min_slice) < 30 or len(df_30min_slice) < 30:
                continue

            signals_checked += 1
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
                    now=candle_time
                )
            except Exception as e:
                continue

            if signal['action'] in ['BUY', 'SELL']:
                direction = signal['action']
                entry_price = current_price

                # Simulate trade
                result = self.simulate_trade(
                    instrument=instrument,
                    direction=direction,
                    entry_price=entry_price,
                    entry_idx=i,
                    df_5min=df_5min,
                    params=params
                )

                if result:
                    trades.append(result)
                    trades_today += 1
                    self.strategy.record_trade_session(instrument, candle_time)

                    # Set cooldown (skip to after trade exit)
                    if result['exit_time']:
                        cooldown_until = result['exit_time']

        print(f"  Signals checked: {signals_checked}")
        print(f"  Trades executed: {len(trades)}")

        # Show trade summary
        if trades:
            winners = [t for t in trades if t['pnl'] > 0]
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = len(winners) / len(trades) * 100
            print(f"  Win Rate: {win_rate:.1f}% | P&L: ${total_pnl:+.2f}")

        return trades

    def run_full_backtest(self):
        """Run backtest on all instruments"""
        print("\n" + "="*80)
        print("FOREX STRATEGY BACKTEST - LOCAL DATA (Sept-Dec 2025)")
        print("="*80)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Position Sizing: $1 per pip (consistent across all pairs)")
        print(f"Trailing Stop: DELAYED ACTIVATION (only at trigger level)")
        print(f"Data Source: {self.data_dir}")
        print("="*80)

        all_trades = []

        for instrument in self.INSTRUMENT_MAP.keys():
            trades = self.run_backtest(instrument)
            all_trades.extend(trades)

        # Generate summary
        self.generate_summary(all_trades)

        return all_trades

    def generate_summary(self, trades: list):
        """Generate performance summary"""
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY (Sept - Dec 2025)")
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

        gross_profit = sum(t['pnl'] for t in winners)
        gross_loss = abs(sum(t['pnl'] for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

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
        print(f"Gross Profit:     ${gross_profit:+,.2f}")
        print(f"Gross Loss:       ${gross_loss:,.2f}")
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
        print(f"-" * 70)
        print(f"{'Instrument':<12} {'Trades':<8} {'Win%':<8} {'P&L':<14} {'Pips':<12} {'PF':<8}")
        print(f"-" * 70)

        instruments = set(t['instrument'] for t in trades)
        for inst in sorted(instruments):
            inst_trades = [t for t in trades if t['instrument'] == inst]
            inst_winners = [t for t in inst_trades if t['pnl'] > 0]
            inst_losers = [t for t in inst_trades if t['pnl'] <= 0]

            inst_pnl = sum(t['pnl'] for t in inst_trades)
            inst_pips = sum(t['pips'] for t in inst_trades)
            inst_wr = len(inst_winners) / len(inst_trades) * 100 if inst_trades else 0

            inst_gross_profit = sum(t['pnl'] for t in inst_winners)
            inst_gross_loss = abs(sum(t['pnl'] for t in inst_losers))
            inst_pf = inst_gross_profit / inst_gross_loss if inst_gross_loss > 0 else float('inf')

            pf_str = f"{inst_pf:.2f}" if inst_pf != float('inf') else "INF"

            print(f"{inst:<12} {len(inst_trades):<8} {inst_wr:<7.1f}% ${inst_pnl:<+12.2f} {inst_pips:<+11.1f} {pf_str:<8}")

        print(f"-" * 70)

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

        # Monthly breakdown
        print(f"\n[PERFORMANCE BY MONTH]")
        print(f"-" * 50)

        trades_by_month = {}
        for t in trades:
            month_key = t['entry_time'].strftime('%Y-%m')
            if month_key not in trades_by_month:
                trades_by_month[month_key] = []
            trades_by_month[month_key].append(t)

        for month in sorted(trades_by_month.keys()):
            month_trades = trades_by_month[month]
            month_winners = [t for t in month_trades if t['pnl'] > 0]
            month_pnl = sum(t['pnl'] for t in month_trades)
            month_wr = len(month_winners) / len(month_trades) * 100 if month_trades else 0
            print(f"{month}: {len(month_trades):>3} trades | Win: {month_wr:.1f}% | P&L: ${month_pnl:+.2f}")

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
    print("\n[*] Starting Forex Strategy Backtest with LOCAL DATA...")

    data_dir = r"C:\Users\Jean-Yves\thevolumeainative\trading_system\Forex_Trading\Backtesting_data"

    backtester = LocalDataBacktester(data_dir=data_dir, initial_balance=10000)

    # Run full backtest
    trades = backtester.run_full_backtest()

    print("\n[OK] Backtest complete!")
    print(f"Total trades analyzed: {len(trades)}")


if __name__ == "__main__":
    main()
