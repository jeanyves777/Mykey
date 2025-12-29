"""
OANDA Strategy → NinjaTrader Bridge

Runs your proven OANDA strategy for signal generation
Sends trading signals to NinjaTrader for execution on futures

ALL FUNDEDNEXT RULES AUTOMATED:
✓ EOD Balance Trailing ($1,000 max loss)
✓ Buffer Protection ($200 minimum)
✓ Daily Loss Limit (-$500)
✓ Profit Target (+$1,250)
✓ Consistency Rule (40% max/day, $400 cap)
✓ Position Limits (5 max concurrent, 10/day per symbol)
✓ Exact TP/SL per symbol (in ticks)
"""

import socket
import json
import time
from datetime import datetime, date
from typing import Dict, List, Optional
from collections import defaultdict

# Import your OANDA client
import sys
sys.path.insert(0, '.')
from trading_system.Forex_Trading.engine.oanda_client import OandaClient


class OandaToNinjaTraderBridge:
    """
    Bridge between OANDA (signal generation) and NinjaTrader (execution)

    Uses OANDA for:
    - Real-time market data
    - Signal generation (your proven strategy)

    Uses NinjaTrader for:
    - Order execution on futures
    - Actual fills and positions
    """

    # ==================== FUNDEDNEXT CHALLENGE SETTINGS ====================

    INITIAL_BALANCE = 25000
    MAX_LOSS_LIMIT = 1000
    PROFIT_TARGET = 1250
    DAILY_LOSS_LIMIT = -500
    MAX_CONCURRENT = 5
    MAX_TRADES_PER_DAY = 50
    MAX_TRADES_PER_SYMBOL = 10
    CONSISTENCY_LIMIT = 0.40  # 40% max per day
    DAILY_PROFIT_CAP = 400    # Hard cap during challenge

    # ==================== SYMBOL MAPPING ====================

    # OANDA → NinjaTrader futures
    SYMBOL_MAP = {
        'EUR_USD': 'M6E',
        'GBP_USD': 'M6B',
        'USD_JPY': 'MJY',
        'USD_CAD': 'MCD',
        'USD_CHF': 'MSF'
    }

    # ==================== TP/SL SETTINGS (EXACT FROM YOUR SPEC) ====================

    PAIR_SETTINGS = {
        'M6E': {  # EUR/USD
            'tp_pips': 20, 'sl_pips': 16,
            'tp_ticks': 40, 'sl_ticks': 32,
            'tick_size': 0.00005, 'tick_value': 6.25
        },
        'M6B': {  # GBP/USD
            'tp_pips': 30, 'sl_pips': 25,
            'tp_ticks': 30, 'sl_ticks': 25,
            'tick_size': 0.0001, 'tick_value': 6.25
        },
        'MJY': {  # USD/JPY
            'tp_pips': 18, 'sl_pips': 15,
            'tp_ticks': 180, 'sl_ticks': 150,
            'tick_size': 0.000001, 'tick_value': 1.25
        },
        'MCD': {  # USD/CAD
            'tp_pips': 20, 'sl_pips': 16,
            'tp_ticks': 40, 'sl_ticks': 32,
            'tick_size': 0.00005, 'tick_value': 5.00
        },
        'MSF': {  # USD/CHF
            'tp_pips': 15, 'sl_pips': 12,
            'tp_ticks': 30, 'sl_ticks': 24,
            'tick_size': 0.00005, 'tick_value': 6.25
        }
    }

    def __init__(self, is_challenge_mode: bool = True):
        """Initialize bridge"""
        self.is_challenge_mode = is_challenge_mode

        # OANDA client for signals
        self.oanda_client = OandaClient()

        # NinjaTrader bridge connection
        self.nt_host = 'localhost'
        self.nt_port = 8888

        # FundedNext tracking
        self.current_balance = self.INITIAL_BALANCE
        self.highest_eod_balance = self.INITIAL_BALANCE
        self.current_threshold = self.INITIAL_BALANCE - self.MAX_LOSS_LIMIT
        self.total_profit = 0

        # Daily tracking
        self.current_date = date.today()
        self.daily_profits = {}
        self.trades_today = 0
        self.trades_per_symbol = defaultdict(int)
        self.starting_balance_today = self.INITIAL_BALANCE

        # Position tracking
        self.open_positions = {}  # {symbol: {side, entry_price, ...}}
        self.trades_log = []

        print("[BRIDGE] ========================================")
        print("[BRIDGE] OANDA → NINJATRADER SIGNAL BRIDGE")
        print("[BRIDGE] ========================================")
        print(f"[BRIDGE] Mode: {'CHALLENGE' if is_challenge_mode else 'FUNDED'}")
        print(f"[BRIDGE] Initial Balance: ${self.INITIAL_BALANCE:,}")
        print(f"[BRIDGE] Profit Target: ${self.PROFIT_TARGET:,}")
        print(f"[BRIDGE] Max Loss: ${self.MAX_LOSS_LIMIT:,}")
        print("[BRIDGE] ========================================")

    def send_signal_to_ninjatrader(self, signal: Dict) -> bool:
        """Send trading signal to NinjaTrader bridge"""
        try:
            # Convert to JSON
            message = json.dumps(signal)

            # Connect to NinjaTrader bridge
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.nt_host, self.nt_port))

            # Send signal
            sock.sendall(message.encode('utf-8'))

            # Wait for acknowledgment
            response = sock.recv(1024).decode('utf-8')
            sock.close()

            if response == "OK":
                print(f"  ✓ Signal sent to NinjaTrader: {signal['Action']} {signal['Symbol']} {signal['Side']}")
                return True
            else:
                print(f"  ⚠ Unexpected response from NinjaTrader: {response}")
                return False

        except Exception as e:
            print(f"  ❌ Error sending signal to NinjaTrader: {e}")
            return False

    def calculate_sl_tp(self, symbol: str, side: str, entry_price: float) -> tuple:
        """Calculate stop loss and take profit prices"""
        settings = self.PAIR_SETTINGS[symbol]
        tick_size = settings['tick_size']
        sl_ticks = settings['sl_ticks']
        tp_ticks = settings['tp_ticks']

        if side == 'BUY':
            sl_price = entry_price - (sl_ticks * tick_size)
            tp_price = entry_price + (tp_ticks * tick_size)
        else:  # SELL
            sl_price = entry_price + (sl_ticks * tick_size)
            tp_price = entry_price - (tp_ticks * tick_size)

        return sl_price, tp_price

    def check_fundednext_rules(self) -> Dict[str, bool]:
        """
        Check all FundedNext rules

        Returns dict with rule status:
            {
                'can_trade': True/False,
                'reason': str (if can't trade),
                'challenge_passed': True/False,
                'account_failed': True/False
            }
        """
        result = {'can_trade': True, 'reason': '', 'challenge_passed': False, 'account_failed': False}

        # RULE 1: Check if account failed (below threshold)
        if self.current_balance <= self.current_threshold:
            result['can_trade'] = False
            result['account_failed'] = True
            result['reason'] = f"Account failed! Balance ${self.current_balance:.2f} <= Threshold ${self.current_threshold:.2f}"
            return result

        # RULE 2: Check buffer protection
        buffer = self.current_balance - self.current_threshold
        if buffer < 200:
            result['can_trade'] = False
            result['reason'] = f"Buffer too low (${buffer:.2f} < $200)"
            return result

        # RULE 3: Check if profit target reached
        if self.total_profit >= self.PROFIT_TARGET:
            result['can_trade'] = False
            result['challenge_passed'] = True
            result['reason'] = f"Challenge passed! Profit ${self.total_profit:.2f} >= ${self.PROFIT_TARGET}"
            return result

        # RULE 4: Check daily loss limit
        today_profit = self.current_balance - self.starting_balance_today
        if today_profit <= self.DAILY_LOSS_LIMIT:
            result['can_trade'] = False
            result['reason'] = f"Daily loss limit hit (${today_profit:.2f} <= ${self.DAILY_LOSS_LIMIT})"
            return result

        # RULE 5: Check consistency rule (challenge only)
        if self.is_challenge_mode and today_profit > 0:
            total_profit_to_date = self.current_balance - self.INITIAL_BALANCE
            if total_profit_to_date > 0:
                max_today_profit = total_profit_to_date * self.CONSISTENCY_LIMIT
                if today_profit >= max_today_profit:
                    result['can_trade'] = False
                    result['reason'] = f"Consistency rule: Today ${today_profit:.2f} >= 40% of total ${total_profit_to_date:.2f}"
                    return result

            # Hard cap at $400/day
            if today_profit >= self.DAILY_PROFIT_CAP:
                result['can_trade'] = False
                result['reason'] = f"Daily profit cap reached (${today_profit:.2f} >= ${self.DAILY_PROFIT_CAP})"
                return result

        # RULE 6: Check max trades
        if self.trades_today >= self.MAX_TRADES_PER_DAY:
            result['can_trade'] = False
            result['reason'] = f"Max trades/day reached ({self.trades_today} >= {self.MAX_TRADES_PER_DAY})"
            return result

        # RULE 7: Check max concurrent positions
        if len(self.open_positions) >= self.MAX_CONCURRENT:
            result['can_trade'] = False
            result['reason'] = f"Max concurrent positions ({len(self.open_positions)} >= {self.MAX_CONCURRENT})"
            return result

        return result

    def enter_trade(self, oanda_symbol: str, side: str, entry_price: float):
        """
        Generate entry signal and send to NinjaTrader
        """
        # Convert to futures symbol
        nt_symbol = self.SYMBOL_MAP[oanda_symbol]

        # Check if can trade this symbol
        if self.trades_per_symbol[nt_symbol] >= self.MAX_TRADES_PER_SYMBOL:
            print(f"  ⚠ Max trades/day for {nt_symbol} reached ({self.MAX_TRADES_PER_SYMBOL})")
            return

        # Calculate SL and TP
        sl_price, tp_price = self.calculate_sl_tp(nt_symbol, side, entry_price)

        # Create signal
        signal = {
            'Action': 'ENTRY',
            'Symbol': nt_symbol,
            'Side': side,
            'Quantity': 1,
            'EntryPrice': entry_price,
            'StopLoss': sl_price,
            'TakeProfit': tp_price,
            'Timestamp': datetime.now().isoformat()
        }

        # Send to NinjaTrader
        if self.send_signal_to_ninjatrader(signal):
            # Track position
            self.open_positions[nt_symbol] = {
                'side': side,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'entry_time': datetime.now()
            }

            self.trades_today += 1
            self.trades_per_symbol[nt_symbol] += 1

            print(f"[ENTRY] {nt_symbol} {side} @ {entry_price:.5f}")
            print(f"  TP: {tp_price:.5f}, SL: {sl_price:.5f}")
            print(f"  Trades today: {self.trades_today}/{self.MAX_TRADES_PER_DAY}")

    def exit_trade(self, oanda_symbol: str, exit_price: float, reason: str = "Signal"):
        """
        Generate exit signal and send to NinjaTrader
        """
        nt_symbol = self.SYMBOL_MAP[oanda_symbol]

        if nt_symbol not in self.open_positions:
            return

        pos = self.open_positions[nt_symbol]

        # Create signal
        signal = {
            'Action': 'EXIT',
            'Symbol': nt_symbol,
            'Side': 'SELL' if pos['side'] == 'BUY' else 'BUY',
            'Quantity': 1,
            'EntryPrice': exit_price,
            'StopLoss': 0,
            'TakeProfit': 0,
            'Timestamp': datetime.now().isoformat()
        }

        # Send to NinjaTrader
        if self.send_signal_to_ninjatrader(signal):
            # Calculate P&L
            settings = self.PAIR_SETTINGS[nt_symbol]
            if pos['side'] == 'BUY':
                pnl_pips = (exit_price - pos['entry_price']) / settings['tick_size']
            else:
                pnl_pips = (pos['entry_price'] - exit_price) / settings['tick_size']

            pnl_usd = pnl_pips * settings['tick_value']

            # Update balance
            self.current_balance += pnl_usd
            self.total_profit = self.current_balance - self.INITIAL_BALANCE

            # Log trade
            self.trades_log.append({
                'symbol': nt_symbol,
                'side': pos['side'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl_usd,
                'reason': reason,
                'time': datetime.now()
            })

            # Remove from tracking
            del self.open_positions[nt_symbol]

            print(f"[EXIT] {nt_symbol} {reason}: ${pnl_usd:+.2f}")
            print(f"  Balance: ${self.current_balance:,.2f} | Total P&L: ${self.total_profit:+,.2f}")

    def update_eod_balance(self):
        """Update end-of-day balance and threshold"""
        eod_balance = self.current_balance

        if eod_balance > self.highest_eod_balance:
            self.highest_eod_balance = eod_balance
            self.current_threshold = self.highest_eod_balance - self.MAX_LOSS_LIMIT
            print(f"\n[EOD] New highest balance: ${self.highest_eod_balance:,.2f}")
            print(f"[EOD] New threshold: ${self.current_threshold:,.2f}")

    def run(self):
        """Main trading loop"""
        print("\n[BRIDGE] Starting trading loop...")
        print("[BRIDGE] Waiting for OANDA signals...\n")

        # TODO: Integrate with your actual OANDA strategy
        # For now, this is a placeholder

        while True:
            try:
                # Check if new day
                if date.today() != self.current_date:
                    self.update_eod_balance()
                    self.current_date = date.today()
                    self.starting_balance_today = self.current_balance
                    self.trades_today = 0
                    self.trades_per_symbol.clear()

                # Check FundedNext rules
                rules = self.check_fundednext_rules()

                if not rules['can_trade']:
                    print(f"\n[STOP] {rules['reason']}")

                    if rules['challenge_passed']:
                        print("\n🎉 CHALLENGE PASSED! 🎉")
                        break
                    elif rules['account_failed']:
                        print("\n❌ ACCOUNT FAILED")
                        break
                    else:
                        # Just stop for the day
                        time.sleep(60)
                        continue

                # TODO: Get signals from your OANDA strategy here
                # Example:
                # signals = your_oanda_strategy.check_signals()
                # for signal in signals:
                #     if signal['action'] == 'ENTRY':
                #         self.enter_trade(signal['symbol'], signal['side'], signal['price'])
                #     elif signal['action'] == 'EXIT':
                #         self.exit_trade(signal['symbol'], signal['price'], signal['reason'])

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n[BRIDGE] Stopping...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(5)


if __name__ == "__main__":
    # Create bridge (challenge mode)
    bridge = OandaToNinjaTraderBridge(is_challenge_mode=True)

    # Run
    bridge.run()
