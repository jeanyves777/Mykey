"""
FundedNext Forex Futures Strategy - Python Version

EXACT TRANSLATION from NinjaScript C#
Simulates trading Micro Forex Futures with FundedNext $25K challenge rules

SYMBOLS:
- M6E (EUR/USD Micro)
- M6B (GBP/USD Micro)
- MJY (USD/JPY Micro)
- MCD (USD/CAD Micro)
- MSF (USD/CHF Micro)

ENTRY LOGIC:
- Same Combined V2: ANY 2 of 3 signals
- RSI Oversold/Overbought
- Bollinger Band touches
- Range Scalping

FUNDEDNEXT RULES:
- Initial Balance: $25,000
- Max Loss: $1,000 (EOD Balance Trailing)
- Profit Target: $1,250
- Consistency Rule: 40% max per day (automated)
- Daily Loss Limit: -$500 (self-imposed)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from typing import Dict, List, Optional, Tuple

class FundedNextStrategy:
    """
    Forex Futures Strategy for FundedNext Challenge
    """

    # ==================== FUNDEDNEXT SETTINGS ====================
    INITIAL_BALANCE = 25000
    MAX_LOSS_LIMIT = 1000
    PROFIT_TARGET = 1250
    CONSISTENCY_LIMIT = 0.40

    # ==================== PAIR SETTINGS ====================
    PAIR_SETTINGS = {
        "M6E": {  # EUR/USD
            "tp_pips": 20,
            "sl_pips": 16,
            "tp_ticks": 40,
            "sl_ticks": 32,
            "pip_multiplier": 2.0,
            "tick_value": 6.25,
            "contract_size": 12500
        },
        "M6B": {  # GBP/USD
            "tp_pips": 30,
            "sl_pips": 25,
            "tp_ticks": 30,
            "sl_ticks": 25,
            "pip_multiplier": 1.0,
            "tick_value": 6.25,
            "contract_size": 6250
        },
        "MJY": {  # USD/JPY
            "tp_pips": 18,
            "sl_pips": 15,
            "tp_ticks": 180,
            "sl_ticks": 150,
            "pip_multiplier": 10.0,
            "tick_value": 1.25,
            "contract_size": 12500
        },
        "MCD": {  # USD/CAD
            "tp_pips": 20,
            "sl_pips": 16,
            "tp_ticks": 40,
            "sl_ticks": 32,
            "pip_multiplier": 2.0,
            "tick_value": 5.00,
            "contract_size": 10000
        },
        "MSF": {  # USD/CHF
            "tp_pips": 15,
            "sl_pips": 12,
            "tp_ticks": 30,
            "sl_ticks": 24,
            "pip_multiplier": 2.0,
            "tick_value": 6.25,
            "contract_size": 12500
        }
    }

    def __init__(
        self,
        symbols: List[str],
        contracts_per_trade: int = 1,
        daily_loss_limit: float = -500,
        max_concurrent: int = 5,
        max_trades_per_day: int = 50,
        max_trades_per_symbol: int = 10,
        is_challenge_mode: bool = True
    ):
        self.symbols = symbols
        self.contracts_per_trade = contracts_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.max_concurrent = max_concurrent
        self.max_trades_per_day = max_trades_per_day
        self.max_trades_per_symbol = max_trades_per_symbol
        self.is_challenge_mode = is_challenge_mode

        # Tracking
        self.highest_eod_balance = self.INITIAL_BALANCE
        self.current_threshold = self.INITIAL_BALANCE - self.MAX_LOSS_LIMIT
        self.current_balance = self.INITIAL_BALANCE
        self.starting_balance_today = self.INITIAL_BALANCE
        self.today_profit = 0
        self.total_profit = 0

        self.trades_today = 0
        self.trades_per_symbol = {symbol: 0 for symbol in symbols}
        self.open_positions = {}
        self.daily_profits = {}
        self.current_date = None

        self.trades_log = []

        print("=" * 80)
        print("FUNDEDNEXT FOREX FUTURES STRATEGY")
        print("=" * 80)
        print(f"Initial Balance: ${self.INITIAL_BALANCE:,}")
        print(f"Profit Target: ${self.PROFIT_TARGET:,}")
        print(f"Max Loss Limit: ${self.MAX_LOSS_LIMIT:,} (EOD Trailing)")
        print(f"Daily Loss Limit: ${abs(self.daily_loss_limit):,}")
        print(f"Consistency Rule: {self.CONSISTENCY_LIMIT*100}% max per day")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Position Size: {contracts_per_trade} contract per trade")
        print("=" * 80)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators on 15min data"""
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_sma'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_sma'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_sma'] - (2 * df['bb_std'])

        # Support/Resistance
        df['support'] = df['low'].rolling(50).min()
        df['resistance'] = df['high'].rolling(50).max()

        # Stochastic
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        return df

    def check_entry_signals(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """
        Check for entry signals - EXACT copy from NinjaScript
        Returns: 'BUY', 'SELL', or None
        """
        if len(df) < 50:
            return None

        # Get current values
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        current_rsi = df['rsi'].iloc[-1]
        current_stoch = df['stoch'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        prev_bb_upper = df['bb_upper'].iloc[-2]
        prev_bb_lower = df['bb_lower'].iloc[-2]
        support = df['support'].iloc[-1]
        resistance = df['resistance'].iloc[-1]

        # Track signals
        bullish_signals = []
        bearish_signals = []

        # ========== BULLISH SIGNALS ==========

        # Signal 1: RSI Oversold
        if current_rsi < 30:
            bullish_signals.append("RSI_OVERSOLD")

        # Signal 2: Bollinger Lower Band Touch
        if current_price <= bb_lower and prev_price > prev_bb_lower:
            bullish_signals.append("BB_LOWER_TOUCH")

        # Signal 3: Range Support + Confirmation
        dist_to_support = (current_price - support) / support
        if dist_to_support < 0.002 and current_rsi < 35 and current_stoch < 35:
            bullish_signals.append("RANGE_SUPPORT")

        # ========== BEARISH SIGNALS ==========

        # Signal 1: RSI Overbought
        if current_rsi > 70:
            bearish_signals.append("RSI_OVERBOUGHT")

        # Signal 2: Bollinger Upper Band Touch
        if current_price >= bb_upper and prev_price < prev_bb_upper:
            bearish_signals.append("BB_UPPER_TOUCH")

        # Signal 3: Range Resistance + Confirmation
        dist_to_resistance = (resistance - current_price) / current_price
        if dist_to_resistance < 0.002 and current_rsi > 65 and current_stoch > 65:
            bearish_signals.append("RANGE_RESISTANCE")

        # ========== ENTRY DECISION ==========

        if len(bullish_signals) >= 2:
            return 'BUY'
        elif len(bearish_signals) >= 2:
            return 'SELL'
        else:
            return None

    def should_enter_trade(
        self,
        symbol: str,
        signal: str,
        current_price: float,
        current_time: datetime
    ) -> bool:
        """Check if we should enter based on FundedNext rules"""

        # New day check
        if current_time.date() != self.current_date:
            self._handle_new_day(current_time)

        # SAFETY GUARD 1: Check threshold
        buffer = self.current_balance - self.current_threshold
        if self.current_balance <= self.current_threshold:
            print(f"[ACCOUNT FAILED] Balance ${self.current_balance:,.2f} <= Threshold ${self.current_threshold:,.2f}")
            return False

        if buffer < 200:
            print(f"[BUFFER WARNING] Only ${buffer:.2f} from threshold - STOP TRADING")
            return False

        # SAFETY GUARD 2: Daily loss limit
        if self.today_profit <= self.daily_loss_limit:
            return False

        # SAFETY GUARD 3: Profit target
        if self.total_profit >= self.PROFIT_TARGET:
            print(f"[PROFIT TARGET HIT] Total: ${self.total_profit:,.2f} - CHALLENGE PASSED!")
            return False

        # SAFETY GUARD 4: Consistency rule (challenge only)
        if self.is_challenge_mode and self.today_profit > 0:
            if self.total_profit > 0:
                max_today = self.total_profit * self.CONSISTENCY_LIMIT
                if self.today_profit >= max_today:
                    return False

            # Hard cap at +$400/day
            if self.today_profit >= 400:
                return False

        # SAFETY GUARD 5: Max trades
        if self.trades_today >= self.max_trades_per_day:
            return False

        if self.trades_per_symbol[symbol] >= self.max_trades_per_symbol:
            return False

        # SAFETY GUARD 6: Max concurrent
        if len(self.open_positions) >= self.max_concurrent:
            return False

        # SAFETY GUARD 7: Already have position on this symbol
        if symbol in self.open_positions:
            return False

        return True

    def enter_trade(
        self,
        symbol: str,
        signal: str,
        entry_price: float,
        entry_time: datetime
    ):
        """Enter a trade"""
        settings = self.PAIR_SETTINGS[symbol]

        # Calculate TP/SL prices
        if signal == 'BUY':
            # Convert ticks to price movement
            tick_size = 0.00005 if symbol in ['M6E', 'MCD', 'MSF'] else (0.0001 if symbol == 'M6B' else 0.000001)

            tp_price = entry_price + (settings['tp_ticks'] * tick_size)
            sl_price = entry_price - (settings['sl_ticks'] * tick_size)

        else:  # SELL
            tick_size = 0.00005 if symbol in ['M6E', 'MCD', 'MSF'] else (0.0001 if symbol == 'M6B' else 0.000001)

            tp_price = entry_price - (settings['tp_ticks'] * tick_size)
            sl_price = entry_price + (settings['sl_ticks'] * tick_size)

        # Store position
        self.open_positions[symbol] = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'contracts': self.contracts_per_trade
        }

        # Update counters
        self.trades_today += 1
        self.trades_per_symbol[symbol] += 1

    def check_exit(
        self,
        symbol: str,
        current_price: float,
        current_time: datetime
    ) -> Optional[Dict]:
        """Check if position should be closed"""
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]

        # Check TP/SL
        hit_tp = False
        hit_sl = False

        if pos['signal'] == 'BUY':
            if current_price >= pos['tp_price']:
                hit_tp = True
            elif current_price <= pos['sl_price']:
                hit_sl = True
        else:  # SELL
            if current_price <= pos['tp_price']:
                hit_tp = True
            elif current_price >= pos['sl_price']:
                hit_sl = True

        if hit_tp or hit_sl:
            return self.exit_trade(symbol, current_price, current_time, 'TP' if hit_tp else 'SL')

        return None

    def exit_trade(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> Dict:
        """Close a trade"""
        pos = self.open_positions[symbol]
        settings = self.PAIR_SETTINGS[symbol]

        # Calculate P&L
        if pos['signal'] == 'BUY':
            price_diff = exit_price - pos['entry_price']
        else:
            price_diff = pos['entry_price'] - exit_price

        # Convert to ticks
        tick_size = 0.00005 if symbol in ['M6E', 'MCD', 'MSF'] else (0.0001 if symbol == 'M6B' else 0.000001)
        ticks = price_diff / tick_size

        # Calculate P&L in dollars
        gross_pnl = ticks * settings['tick_value'] * pos['contracts']
        commission = 0.85 * pos['contracts'] * 2  # Round trip
        net_pnl = gross_pnl - commission

        # Update balance
        self.current_balance += net_pnl
        self.today_profit += net_pnl
        self.total_profit = self.current_balance - self.INITIAL_BALANCE

        # Log trade
        trade_log = {
            'symbol': symbol,
            'signal': pos['signal'],
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'ticks': ticks,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'balance': self.current_balance,
            'total_profit': self.total_profit
        }

        self.trades_log.append(trade_log)

        # Remove position
        del self.open_positions[symbol]

        return trade_log

    def _handle_new_day(self, current_time: datetime):
        """Handle new trading day"""
        # Update EOD balance from previous day
        if self.current_date is not None:
            if self.current_balance > self.highest_eod_balance:
                self.highest_eod_balance = self.current_balance
                self.current_threshold = self.highest_eod_balance - self.MAX_LOSS_LIMIT

                print(f"[EOD UPDATE] New Highest: ${self.highest_eod_balance:,.2f}, Threshold: ${self.current_threshold:,.2f}")

            # Store daily profit
            date_key = self.current_date.strftime('%Y-%m-%d')
            self.daily_profits[date_key] = self.today_profit

        # Reset for new day
        self.current_date = current_time.date()
        self.starting_balance_today = self.current_balance
        self.today_profit = 0
        self.trades_today = 0
        self.trades_per_symbol = {symbol: 0 for symbol in self.symbols}

        print(f"\n[NEW DAY] {self.current_date} - Balance: ${self.current_balance:,.2f}, Threshold: ${self.current_threshold:,.2f}")

    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        if not self.trades_log:
            return {}

        df_trades = pd.DataFrame(self.trades_log)

        winners = df_trades[df_trades['net_pnl'] > 0]
        losers = df_trades[df_trades['net_pnl'] < 0]

        total_trades = len(df_trades)
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        total_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0
        avg_trade = df_trades['net_pnl'].mean()

        # Check consistency rule
        max_day_profit = 0
        max_day_pct = 0
        if self.daily_profits and self.total_profit > 0:
            max_day_profit = max(self.daily_profits.values())
            max_day_pct = (max_day_profit / self.total_profit * 100) if self.total_profit > 0 else 0

        return {
            'total_trades': total_trades,
            'winners': win_count,
            'losers': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': self.total_profit,
            'total_return_pct': (self.total_profit / self.INITIAL_BALANCE * 100),
            'final_balance': self.current_balance,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'max_day_profit': max_day_profit,
            'max_day_pct': max_day_pct,
            'consistency_compliant': max_day_pct <= 40,
            'challenge_passed': self.total_profit >= self.PROFIT_TARGET,
            'account_failed': self.current_balance <= self.current_threshold
        }
