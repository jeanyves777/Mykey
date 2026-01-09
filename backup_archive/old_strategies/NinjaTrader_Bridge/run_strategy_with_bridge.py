"""
Run OANDA Strategy with NinjaTrader Bridge

This script:
1. Monitors OANDA market data for signals
2. Generates trading signals using your proven strategy
3. Sends signals to NinjaTrader bridge for execution
4. Bridge executes on whatever account NinjaTrader is connected to

BEFORE RUNNING:
1. Start NinjaTrader 8
2. Connect to Sim101 (or your FundedNext account)
3. Run NinjaTraderBridge.exe
4. Then run this script
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

import socket
import json
import time
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict

from trading_system.Forex_Trading.engine.oanda_client import OandaClient


class StrategyToBridgeRunner:
    """
    Runs OANDA strategy and sends signals to NinjaTrader bridge
    """

    # FundedNext Challenge Settings
    INITIAL_BALANCE = 25000
    MAX_LOSS_LIMIT = 1000
    PROFIT_TARGET = 1250
    DAILY_LOSS_LIMIT = -500
    MAX_CONCURRENT = 5
    MAX_TRADES_PER_DAY = 50
    MAX_TRADES_PER_SYMBOL = 10
    CONSISTENCY_LIMIT = 0.40
    DAILY_PROFIT_CAP = 400

    # Symbol mapping: OANDA → NinjaTrader
    SYMBOL_MAP = {
        'EUR_USD': 'M6E',
        'GBP_USD': 'M6B',
        'USD_JPY': 'MJY',
        'USD_CAD': 'MCD',
        'USD_CHF': 'MSF'
    }

    # TP/SL Settings (exact from your spec)
    PAIR_SETTINGS = {
        'M6E': {
            'tp_pips': 20, 'sl_pips': 16,
            'tp_ticks': 40, 'sl_ticks': 32,
            'tick_size': 0.00005, 'tick_value': 6.25
        },
        'M6B': {
            'tp_pips': 30, 'sl_pips': 25,
            'tp_ticks': 30, 'sl_ticks': 25,
            'tick_size': 0.0001, 'tick_value': 6.25
        },
        'MJY': {
            'tp_pips': 18, 'sl_pips': 15,
            'tp_ticks': 180, 'sl_ticks': 150,
            'tick_size': 0.000001, 'tick_value': 1.25
        },
        'MCD': {
            'tp_pips': 20, 'sl_pips': 16,
            'tp_ticks': 40, 'sl_ticks': 32,
            'tick_size': 0.00005, 'tick_value': 5.00
        },
        'MSF': {
            'tp_pips': 15, 'sl_pips': 12,
            'tp_ticks': 30, 'sl_ticks': 24,
            'tick_size': 0.00005, 'tick_value': 6.25
        }
    }

    def __init__(self, is_challenge_mode: bool = True):
        self.is_challenge_mode = is_challenge_mode

        # OANDA client for market data
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
        from datetime import date
        self.current_date = date.today()
        self.daily_profits = {}
        self.trades_today = 0
        self.trades_per_symbol = defaultdict(int)
        self.starting_balance_today = self.INITIAL_BALANCE

        # Position tracking
        self.open_positions = {}
        self.trades_log = []

        print("=" * 80)
        print("OANDA STRATEGY → NINJATRADER BRIDGE")
        print("=" * 80)
        print(f"Mode: {'CHALLENGE' if is_challenge_mode else 'FUNDED'}")
        print(f"Initial Balance: ${self.INITIAL_BALANCE:,}")
        print(f"Profit Target: ${self.PROFIT_TARGET:,}")
        print(f"Max Loss: ${self.MAX_LOSS_LIMIT:,}")
        print("=" * 80)
        print()

    def send_signal_to_bridge(self, signal: Dict) -> bool:
        """Send trading signal to NinjaTrader bridge"""
        try:
            message = json.dumps(signal)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.nt_host, self.nt_port))
            sock.sendall(message.encode('utf-8'))
            response = sock.recv(1024).decode('utf-8')
            sock.close()

            if response == "OK":
                print(f"  ✓ Signal sent to NinjaTrader: {signal['Action']} {signal['Symbol']} {signal['Side']}")
                return True
            else:
                print(f"  ⚠ Unexpected response: {response}")
                return False

        except ConnectionRefusedError:
            print(f"  ❌ Cannot connect to bridge - is NinjaTraderBridge.exe running?")
            return False
        except Exception as e:
            print(f"  ❌ Error sending signal: {e}")
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
        """Check all FundedNext rules"""
        result = {'can_trade': True, 'reason': '', 'challenge_passed': False, 'account_failed': False}

        # Rule 1: Account failed check
        if self.current_balance <= self.current_threshold:
            result['can_trade'] = False
            result['account_failed'] = True
            result['reason'] = f"Account failed! Balance ${self.current_balance:.2f} <= Threshold ${self.current_threshold:.2f}"
            return result

        # Rule 2: Buffer protection
        buffer = self.current_balance - self.current_threshold
        if buffer < 200:
            result['can_trade'] = False
            result['reason'] = f"Buffer too low (${buffer:.2f} < $200)"
            return result

        # Rule 3: Profit target
        if self.total_profit >= self.PROFIT_TARGET:
            result['can_trade'] = False
            result['challenge_passed'] = True
            result['reason'] = f"Challenge passed! Profit ${self.total_profit:.2f}"
            return result

        # Rule 4: Daily loss limit
        today_profit = self.current_balance - self.starting_balance_today
        if today_profit <= self.DAILY_LOSS_LIMIT:
            result['can_trade'] = False
            result['reason'] = f"Daily loss limit hit (${today_profit:.2f})"
            return result

        # Rule 5: Consistency rule
        if self.is_challenge_mode and today_profit > 0:
            total_profit_to_date = self.current_balance - self.INITIAL_BALANCE
            if total_profit_to_date > 0:
                max_today_profit = total_profit_to_date * self.CONSISTENCY_LIMIT
                if today_profit >= max_today_profit:
                    result['can_trade'] = False
                    result['reason'] = f"Consistency rule: Today ${today_profit:.2f} >= 40% of total"
                    return result
            if today_profit >= self.DAILY_PROFIT_CAP:
                result['can_trade'] = False
                result['reason'] = f"Daily profit cap (${today_profit:.2f} >= ${self.DAILY_PROFIT_CAP})"
                return result

        # Rule 6: Max trades
        if self.trades_today >= self.MAX_TRADES_PER_DAY:
            result['can_trade'] = False
            result['reason'] = f"Max trades/day reached ({self.trades_today})"
            return result

        # Rule 7: Max concurrent
        if len(self.open_positions) >= self.MAX_CONCURRENT:
            result['can_trade'] = False
            result['reason'] = f"Max concurrent positions ({len(self.open_positions)})"
            return result

        return result

    def test_connection(self):
        """Test connection to NinjaTrader bridge"""
        print("[TEST] Testing connection to NinjaTrader bridge...")

        test_signal = {
            'Action': 'ENTRY',
            'Symbol': 'M6E',
            'Side': 'BUY',
            'Quantity': 1,
            'EntryPrice': 1.05000,
            'StopLoss': 1.04800,
            'TakeProfit': 1.05200,
            'Timestamp': datetime.now().isoformat()
        }

        if self.send_signal_to_bridge(test_signal):
            print("[TEST] ✓ Connection successful!")
            print("[TEST] Check NinjaTrader for the test order")
            return True
        else:
            print("[TEST] ❌ Connection failed")
            print()
            print("Make sure:")
            print("1. NinjaTrader 8 is running")
            print("2. NinjaTraderBridge.exe is running")
            print("3. Bridge shows 'Status: ACTIVE'")
            return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STRATEGY → NINJATRADER BRIDGE TEST")
    print("=" * 80)
    print()
    print("This will send a TEST signal to NinjaTrader")
    print()
    print("Prerequisites:")
    print("1. NinjaTrader 8 is running")
    print("2. Connected to Sim101 (or your account)")
    print("3. NinjaTraderBridge.exe is running")
    print()
    print("=" * 80)
    print()

    input("Press ENTER to test connection...")

    runner = StrategyToBridgeRunner(is_challenge_mode=True)

    if runner.test_connection():
        print()
        print("=" * 80)
        print("✓ SUCCESS - Bridge is working!")
        print("=" * 80)
        print()
        print("Next step: Tell me which OANDA live script to integrate")
        print("I'll add the signal-sending code to connect your strategy!")
    else:
        print()
        print("=" * 80)
        print("❌ Connection failed - check the prerequisites above")
        print("=" * 80)
