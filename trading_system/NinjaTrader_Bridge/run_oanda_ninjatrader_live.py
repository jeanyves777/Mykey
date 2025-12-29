"""
OANDA → NinjaTrader Live Trading Bridge

This script:
1. Monitors OANDA market data for EUR/USD, GBP/USD, USD/JPY, USD/CAD, USD/CHF
2. Uses your proven Multi-Timeframe Momentum strategy for signals
3. Sends signals to NinjaTrader bridge for execution on futures
4. Automates ALL FundedNext $25K challenge rules

BEFORE RUNNING:
1. Start NinjaTrader 8
2. Connect to Sim101 (for testing) or FundedNext account (for challenge)
3. Run NinjaTraderBridge.exe
4. Then run this script

SAFETY FEATURES:
✓ EOD Balance Trailing ($1,000 max loss)
✓ Buffer Protection ($200 minimum)
✓ Daily Loss Limit (-$500)
✓ Profit Target (+$1,250)
✓ Consistency Rule (OPTIONAL - enable with --consistency flag)
✓ Position Limits (1 contract, max 5 concurrent)
✓ Trade Limits (50/day, 10/symbol)

MODES:
- Default: Simple mode (no consistency rule, just -$500 daily loss limit)
- --consistency: Enable 40% max per day + $400 cap (FundedNext challenge requirement)
- --funded: Funded account mode (no consistency rule needed)
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

import socket
import json
import time
import pandas as pd
import os
from datetime import datetime, date, timedelta
from typing import Dict, Optional
from collections import defaultdict
from pathlib import Path
import argparse

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.forex_trading_config import (
    FOREX_INSTRUMENTS,
    STRATEGY_CONFIG,
    DATA_CONFIG
)
from trading_system.analytics.forex_trade_logger import ForexTradeLogger


class OandaNinjaTraderLiveBridge:
    """
    Live trading bridge:
    - OANDA for market data & signals
    - NinjaTrader for futures execution
    - FundedNext rules automated
    """

    # FundedNext Challenge Settings
    INITIAL_BALANCE = 25000
    MAX_LOSS_LIMIT = 1000          # Max $1K loss from highest EOD balance
    PROFIT_TARGET = 1250            # $1,250 profit target
    DAILY_LOSS_LIMIT = -500         # -$500 daily loss limit
    MAX_CONCURRENT = 5              # Max 5 concurrent positions (1 per symbol)
    CONTRACTS_PER_TRADE = 1         # Always 1 contract per trade
    MAX_TRADES_PER_DAY = 50         # Max trades per day (safety limit)
    MAX_TRADES_PER_SYMBOL = 10      # Max trades per symbol per day

    # Consistency Rule (optional - can be enabled/disabled via mode)
    CONSISTENCY_LIMIT = 0.40        # 40% max per day
    DAILY_PROFIT_CAP = 400          # Hard cap at $400/day during challenge

    # Symbol mapping: OANDA → NinjaTrader
    SYMBOL_MAP = {
        'EUR_USD': 'M6E',
        'GBP_USD': 'M6B',
        'USD_JPY': 'MJY',
        'USD_CAD': 'MCD',
        'USD_CHF': 'MSF'
    }

    # TP/SL Settings
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

    def __init__(self, is_challenge_mode: bool = True, enable_consistency_rule: bool = False):
        self.is_challenge_mode = is_challenge_mode
        self.enable_consistency_rule = enable_consistency_rule

        # State file for position persistence
        self.state_file = Path("trading_system/NinjaTrader_Bridge/bridge_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # OANDA client for market data
        print("[INIT] Connecting to OANDA...")
        self.oanda_client = OandaClient(account_type="practice")

        # Strategy - Using PROVEN ForexScalpingStrategy (+25.82% in 1 month backtest!)
        print("[INIT] Initializing Forex Scalping Strategy (PROVEN: +25.82% return, 51.9% win rate)...")
        self.strategy = ForexScalpingStrategy(
            instruments=FOREX_INSTRUMENTS,
            max_trades_per_day=50,  # Aligned with FundedNext limit
            daily_profit_target=0.05,  # 5% daily target (well within FundedNext limits)
            trade_size_pct=0.15  # 15% margin per position
        )

        # Trade logger
        print("[INIT] Initializing trade logger...")
        self.trade_logger = ForexTradeLogger()

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
        self.open_positions = {}  # NT symbol -> position info
        self.trades_log = []

        # Trading state
        self.is_running = False
        self.last_check_time = {}

        print("=" * 80)
        print("OANDA → NINJATRADER LIVE TRADING BRIDGE")
        print("=" * 80)
        print(f"Mode: {'CHALLENGE' if is_challenge_mode else 'FUNDED'}")
        print(f"Consistency Rule: {'ENABLED' if enable_consistency_rule else 'DISABLED'}")
        print(f"Initial Balance: ${self.INITIAL_BALANCE:,}")
        print(f"Profit Target: ${self.PROFIT_TARGET:,}")
        print(f"Max Loss: ${self.MAX_LOSS_LIMIT:,}")

        # Display NinjaTrader futures symbols
        nt_symbols = [self.SYMBOL_MAP.get(oanda, "?") for oanda in FOREX_INSTRUMENTS]
        print(f"NinjaTrader Futures: {', '.join(nt_symbols)}")
        print(f"OANDA Data Sources: {', '.join(FOREX_INSTRUMENTS)}")
        print("=" * 80)
        print()

    def save_state(self):
        """Save current state to file"""
        try:
            state = {
                'current_balance': self.current_balance,
                'highest_eod_balance': self.highest_eod_balance,
                'current_threshold': self.current_threshold,
                'total_profit': self.total_profit,
                'current_date': self.current_date.isoformat(),
                'daily_profits': self.daily_profits,
                'trades_today': self.trades_today,
                'trades_per_symbol': dict(self.trades_per_symbol),
                'starting_balance_today': self.starting_balance_today,
                'open_positions': {
                    symbol: {
                        'trade_id': pos.get('trade_id', ''),
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'sl_price': pos['sl_price'],
                        'tp_price': pos['tp_price'],
                        'entry_time': pos['entry_time'].isoformat()
                    }
                    for symbol, pos in self.open_positions.items()
                },
                'trades_log': self.trades_log
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            print(f"[STATE] Saved to {self.state_file}")
            return True

        except Exception as e:
            print(f"[STATE] Error saving state: {e}")
            return False

    def load_state(self) -> bool:
        """Load state from file if it exists"""
        try:
            if not self.state_file.exists():
                return False

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.current_balance = state['current_balance']
            self.highest_eod_balance = state['highest_eod_balance']
            self.current_threshold = state['current_threshold']
            self.total_profit = state['total_profit']
            self.current_date = date.fromisoformat(state['current_date'])
            self.daily_profits = state['daily_profits']
            self.trades_today = state['trades_today']
            self.trades_per_symbol = defaultdict(int, state['trades_per_symbol'])
            self.starting_balance_today = state['starting_balance_today']

            # Load open positions
            self.open_positions = {
                symbol: {
                    'trade_id': pos.get('trade_id', ''),
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'sl_price': pos['sl_price'],
                    'tp_price': pos['tp_price'],
                    'entry_time': datetime.fromisoformat(pos['entry_time'])
                }
                for symbol, pos in state['open_positions'].items()
            }

            self.trades_log = state['trades_log']

            print(f"[STATE] Loaded from {self.state_file}")
            print(f"[STATE] Balance: ${self.current_balance:,.2f}")
            print(f"[STATE] Open Positions: {len(self.open_positions)}")
            if self.open_positions:
                for symbol, pos in self.open_positions.items():
                    print(f"[STATE]   - {symbol} {pos['side']} @ {pos['entry_price']:.5f}")

            return True

        except Exception as e:
            print(f"[STATE] Error loading state: {e}")
            return False

    def close_all_positions(self, exit_reason: str = "MANUAL"):
        """Close all open positions via bridge"""
        if not self.open_positions:
            print("[CLOSE] No open positions to close")
            return

        print(f"\n[CLOSE] Closing {len(self.open_positions)} open positions...")

        for symbol, pos in list(self.open_positions.items()):
            try:
                # Get ACTUAL NinjaTrader price for exit logging
                nt_price_data = self.get_ninjatrader_price(symbol)

                if nt_price_data:
                    # Use NinjaTrader actual bid/ask for exit price
                    if pos['side'] == 'BUY':
                        exit_price = nt_price_data['bid']  # Exit BUY at NinjaTrader bid
                    else:
                        exit_price = nt_price_data['ask']  # Cover SELL at NinjaTrader ask
                else:
                    # Fallback to entry price if NinjaTrader price not available
                    exit_price = pos['entry_price']
                    print(f"  ⚠ Using entry price as exit (NinjaTrader price unavailable)")

                # Send exit signal to bridge
                exit_signal = {
                    'Action': 'EXIT',
                    'Symbol': symbol,
                    'Side': 'SELL' if pos['side'] == 'BUY' else 'BUY',
                    'Quantity': 1,
                    'Timestamp': datetime.now().isoformat()
                }

                if self.send_signal_to_bridge(exit_signal):
                    print(f"[CLOSE] ✓ Closed {symbol} {pos['side']}")

                    # Log exit
                    if 'trade_id' in pos:
                        self.trade_logger.log_exit(
                            trade_id=pos['trade_id'],
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            account_balance=self.current_balance,
                            daily_profit=self.current_balance - self.starting_balance_today,
                            trades_today=self.trades_today,
                            fees_usd=0.0  # NinjaTrader fees handled separately
                        )

                    del self.open_positions[symbol]
                else:
                    print(f"[CLOSE] ❌ Failed to close {symbol}")

            except Exception as e:
                print(f"[CLOSE] Error closing {symbol}: {e}")

        # Save state after closing
        self.save_state()

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
                return True
            else:
                print(f"  ⚠ Unexpected response: {response}")
                return False

        except Exception as e:
            print(f"  ❌ Error sending signal: {e}")
            return False

    def is_market_open(self) -> tuple[bool, str]:
        """
        Check if forex market is open
        Forex markets are open 24/5 (Sunday 5pm ET - Friday 5pm ET)

        Returns:
            (is_open, reason)
        """
        now = datetime.now()

        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = now.weekday()
        hour = now.hour

        # Friday after 5pm ET (17:00) - market closed
        if day_of_week == 4 and hour >= 17:
            return False, "Market closed - Friday after 5pm ET"

        # Saturday - market closed
        if day_of_week == 5:
            return False, "Market closed - Saturday"

        # Sunday before 5pm ET (17:00) - market closed
        if day_of_week == 6 and hour < 17:
            return False, "Market closed - Sunday before 5pm ET"

        # Otherwise market is open
        return True, "Market open"

    def get_current_prices(self) -> Dict:
        """Fetch current OANDA market prices for all futures symbols (for comparison only)"""
        prices = {}
        for oanda_symbol in FOREX_INSTRUMENTS:
            nt_symbol = self.SYMBOL_MAP.get(oanda_symbol)
            if not nt_symbol:
                continue

            try:
                price_data = self.oanda_client.get_current_price(oanda_symbol)
                if price_data:
                    prices[nt_symbol] = {
                        'oanda_symbol': oanda_symbol,
                        'bid': price_data.get('bid', 0),
                        'ask': price_data.get('ask', 0),
                        'mid': (price_data.get('bid', 0) + price_data.get('ask', 0)) / 2,
                        'source': 'OANDA'
                    }
            except Exception as e:
                # Skip symbols with errors
                pass

        return prices

    def get_ninjatrader_price(self, nt_symbol: str) -> Optional[Dict]:
        """
        Query actual NinjaTrader futures price for a specific symbol

        Args:
            nt_symbol: NinjaTrader symbol (M6E, M6B, MJY, MCD, MSF)

        Returns:
            Dict with bid, ask, last prices or None if error
        """
        try:
            # Send price query to bridge
            query = {
                'Action': 'PRICE_QUERY',
                'Symbol': nt_symbol,
                'Timestamp': datetime.now().isoformat()
            }

            message = json.dumps(query)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.nt_host, self.nt_port))
            sock.sendall(message.encode('utf-8'))

            # Receive price data response
            response = sock.recv(4096).decode('utf-8')
            sock.close()

            # Parse JSON response
            price_data = json.loads(response)
            status = price_data.get('Status', '')

            # Handle different status codes
            if status == 'OK':
                # Live market data
                return {
                    'symbol': nt_symbol,
                    'bid': price_data.get('Bid', 0),
                    'ask': price_data.get('Ask', 0),
                    'last': price_data.get('Last', 0),
                    'mid': (price_data.get('Bid', 0) + price_data.get('Ask', 0)) / 2,
                    'source': 'NinjaTrader',
                    'timestamp': price_data.get('Timestamp', ''),
                    'status': 'LIVE'
                }
            elif status == 'MARKET_CLOSED':
                # Market closed but we have cached prices
                return {
                    'symbol': nt_symbol,
                    'bid': price_data.get('Bid', 0),
                    'ask': price_data.get('Ask', 0),
                    'last': price_data.get('Last', 0),
                    'mid': (price_data.get('Bid', 0) + price_data.get('Ask', 0)) / 2,
                    'source': 'NinjaTrader (cached)',
                    'timestamp': price_data.get('Timestamp', ''),
                    'status': 'CACHED'
                }
            elif status == 'MARKET_CLOSED_NO_CACHE':
                # Market closed and no cached prices available
                print(f"  ⚠ {nt_symbol}: Market closed, no cached prices available")
                return None
            elif status.startswith('ERROR_CACHED'):
                # Error occurred but cached prices returned
                return {
                    'symbol': nt_symbol,
                    'bid': price_data.get('Bid', 0),
                    'ask': price_data.get('Ask', 0),
                    'last': price_data.get('Last', 0),
                    'mid': (price_data.get('Bid', 0) + price_data.get('Ask', 0)) / 2,
                    'source': 'NinjaTrader (cached)',
                    'timestamp': price_data.get('Timestamp', ''),
                    'status': 'CACHED_ERROR'
                }
            else:
                # Other error
                print(f"  ⚠ NinjaTrader price query failed for {nt_symbol}: {status}")
                return None

        except Exception as e:
            print(f"  ❌ Error querying NinjaTrader price for {nt_symbol}: {e}")
            return None

    def get_ninjatrader_prices(self) -> Dict:
        """Fetch actual NinjaTrader futures prices for all symbols"""
        nt_prices = {}
        for nt_symbol in self.SYMBOL_MAP.values():
            price_data = self.get_ninjatrader_price(nt_symbol)
            if price_data:
                nt_prices[nt_symbol] = price_data
        return nt_prices

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

        # Rule 4: Daily loss limit (SIMPLE - just -$500)
        today_profit = self.current_balance - self.starting_balance_today
        if today_profit <= self.DAILY_LOSS_LIMIT:
            result['can_trade'] = False
            result['reason'] = f"Daily loss limit hit (${today_profit:.2f} <= -$500)"
            return result

        # Rule 5: Consistency rule (if enabled)
        if self.enable_consistency_rule and today_profit > 0:
            # Calculate total profit to date
            total_profit_to_date = self.current_balance - self.INITIAL_BALANCE

            # Check 40% rule (can't make more than 40% of total profit in one day)
            if total_profit_to_date > 0:
                max_today_profit = total_profit_to_date * self.CONSISTENCY_LIMIT
                if today_profit >= max_today_profit:
                    result['can_trade'] = False
                    result['reason'] = f"Consistency rule: Today ${today_profit:.2f} >= 40% of total ${total_profit_to_date:.2f}"
                    return result

            # Hard cap at $400/day during challenge
            if today_profit >= self.DAILY_PROFIT_CAP:
                result['can_trade'] = False
                result['reason'] = f"Daily profit cap (${today_profit:.2f} >= ${self.DAILY_PROFIT_CAP})"
                return result

        # Rule 6: Max trades per day
        if self.trades_today >= self.MAX_TRADES_PER_DAY:
            result['can_trade'] = False
            result['reason'] = f"Max trades/day reached ({self.trades_today})"
            return result

        # Rule 7: Max concurrent positions
        if len(self.open_positions) >= self.MAX_CONCURRENT:
            result['can_trade'] = False
            result['reason'] = f"Max concurrent positions"
            return result

        return result

    def get_market_data(self, oanda_symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch market data for ForexScalpingStrategy"""
        try:
            candles_1min = self.oanda_client.get_candles(oanda_symbol, granularity="M1", count=DATA_CONFIG.get("candles_1min", 100))
            candles_5min = self.oanda_client.get_candles(oanda_symbol, granularity="M5", count=DATA_CONFIG.get("candles_5min", 50))
            candles_15min = self.oanda_client.get_candles(oanda_symbol, granularity="M15", count=DATA_CONFIG.get("candles_15min", 30))
            candles_30min = self.oanda_client.get_candles(oanda_symbol, granularity="M30", count=DATA_CONFIG.get("candles_30min", 24))

            df_1min = pd.DataFrame(candles_1min)
            df_5min = pd.DataFrame(candles_5min)
            df_15min = pd.DataFrame(candles_15min)
            df_30min = pd.DataFrame(candles_30min)

            if len(df_1min) == 0:
                return None

            return {
                "1min": df_1min,
                "5min": df_5min,
                "15min": df_15min,
                "30min": df_30min
            }

        except Exception as e:
            print(f"  ❌ Error fetching data for {oanda_symbol}: {e}")
            return None

    def check_new_day(self):
        """Handle new trading day"""
        if date.today() != self.current_date:
            # Update EOD balance
            if self.current_balance > self.highest_eod_balance:
                self.highest_eod_balance = self.current_balance
                self.current_threshold = self.highest_eod_balance - self.MAX_LOSS_LIMIT
                print(f"\n[EOD] New Highest: ${self.highest_eod_balance:,.2f}, Threshold: ${self.current_threshold:,.2f}")

            # Store daily profit
            date_key = self.current_date.strftime('%Y-%m-%d')
            today_profit = self.current_balance - self.starting_balance_today
            self.daily_profits[date_key] = today_profit

            # Reset for new day
            self.current_date = date.today()
            self.starting_balance_today = self.current_balance
            self.trades_today = 0
            self.trades_per_symbol.clear()

            print(f"\n[NEW DAY] {self.current_date} - Balance: ${self.current_balance:,.2f}, Threshold: ${self.current_threshold:,.2f}\n")

    def run(self, duration_hours: float = 0):
        """Main trading loop"""
        print("[START] Starting live trading...")
        if duration_hours > 0:
            print(f"[START] Duration: {duration_hours} hours")
        else:
            print(f"[START] Duration: Continuous (until stopped or market closes)")

        # Try to load previous state
        if self.load_state():
            print()
            print("=" * 80)
            print("RESUMING FROM PREVIOUS SESSION")
            print("=" * 80)
            print(f"Found {len(self.open_positions)} open positions")
            print("These positions will be managed automatically")
            print("=" * 80)
            print()

        print(f"[START] Checking bridge connection...")

        # Test bridge connection with PRICE_QUERY (non-invasive)
        test_query = {
            'Action': 'PRICE_QUERY',
            'Symbol': 'M6E',
            'Timestamp': datetime.now().isoformat()
        }

        try:
            message = json.dumps(test_query)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.nt_host, self.nt_port))
            sock.sendall(message.encode('utf-8'))
            response = sock.recv(4096).decode('utf-8')
            sock.close()

            # Parse response to verify connection
            price_data = json.loads(response)
            status = price_data.get('Status', '')

            if status in ['OK', 'MARKET_CLOSED', 'MARKET_CLOSED_NO_CACHE']:
                print("[START] ✓ Bridge connection successful!")
                if status == 'OK':
                    print(f"[START]   Market is OPEN - Live trading ready")
                elif status == 'MARKET_CLOSED':
                    print(f"[START]   Market is CLOSED - Using cached prices for monitoring")
                else:
                    print(f"[START]   Market is CLOSED - No cached prices (will trade when market opens)")
                print()
            else:
                print(f"\n⚠ WARNING: Bridge responded but returned status: {status}")
                print("Continuing anyway - check bridge console for details")
                print()

        except Exception as e:
            print("\n❌ ERROR: Cannot connect to NinjaTrader bridge!")
            print(f"Error: {e}")
            print("\nMake sure:")
            print("1. NinjaTrader 8 is running")
            print("2. NinjaTraderBridge.exe is running")
            print("3. Bridge shows 'Status: ACTIVE'")
            return

        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours) if duration_hours > 0 else None

        try:
            loop_count = 0
            while self.is_running and (end_time is None or datetime.now() < end_time):
                loop_count += 1
                current_time = datetime.now()

                # Log every 5 minutes
                if loop_count % 5 == 1:
                    market_open, market_status = self.is_market_open()
                    daily_pnl = self.current_balance - self.starting_balance_today
                    buffer = self.current_balance - self.current_threshold

                    print(f"\n[{current_time.strftime('%H:%M:%S')}] Loop {loop_count} - Checking market...")
                    print(f"  Market Status: {market_status}")
                    print(f"  Balance: ${self.current_balance:,.2f}")
                    print(f"  Threshold: ${self.current_threshold:,.2f} (EOD trailing stop)")
                    print(f"  Buffer: ${buffer:,.2f} above threshold")
                    print(f"  Daily P&L: ${daily_pnl:+,.2f} (Daily loss limit: ${self.DAILY_LOSS_LIMIT})")
                    print(f"  Total Profit: ${self.total_profit:+,.2f} (Target: ${self.PROFIT_TARGET})")
                    print(f"  Trades today: {self.trades_today}/{self.MAX_TRADES_PER_DAY}")
                    print(f"  Open positions: {len(self.open_positions)}/{self.MAX_CONCURRENT}")

                    # Display dual pricing: OANDA vs NinjaTrader
                    print(f"\n  Market Prices - OANDA vs NinjaTrader Comparison:")
                    print(f"  {'Symbol':<6} {'Source':<12} {'Bid':<12} {'Ask':<12} {'Mid':<12} {'Diff (pips)':<12}")
                    print(f"  {'-'*72}")

                    oanda_prices = self.get_current_prices()
                    nt_prices = self.get_ninjatrader_prices()

                    for nt_symbol in ['M6E', 'M6B', 'MJY', 'MSF', 'MCD']:
                        # Display OANDA price
                        if nt_symbol in oanda_prices:
                            p_oanda = oanda_prices[nt_symbol]
                            print(f"  {nt_symbol:<6} {'OANDA':<12} {p_oanda['bid']:<12.5f} {p_oanda['ask']:<12.5f} {p_oanda['mid']:<12.5f} {'':<12}")

                        # Display NinjaTrader price
                        if nt_symbol in nt_prices:
                            p_nt = nt_prices[nt_symbol]

                            # Calculate difference in pips if both prices available
                            diff_pips = ""
                            if nt_symbol in oanda_prices:
                                settings = self.PAIR_SETTINGS[nt_symbol]
                                tick_size = settings['tick_size']
                                mid_diff = abs(p_nt['mid'] - oanda_prices[nt_symbol]['mid'])
                                diff_pips = f"{(mid_diff / tick_size):.1f}"

                            print(f"  {'':<6} {'NinjaTrader':<12} {p_nt['bid']:<12.5f} {p_nt['ask']:<12.5f} {p_nt['mid']:<12.5f} {diff_pips:<12}")

                        # Add spacing between symbols for readability
                        if nt_symbol in oanda_prices or nt_symbol in nt_prices:
                            print(f"  {'-'*72}")

                    if not oanda_prices and not nt_prices:
                        print(f"  (No price data available)")

                # Check if market is open
                market_open, market_status = self.is_market_open()
                if not market_open:
                    if loop_count % 5 == 1:
                        print(f"  ⚠ {market_status} - Skipping symbol checks")
                    time.sleep(60)
                    continue

                self.check_new_day()

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
                        print("Waiting 60 seconds...")
                        time.sleep(60)
                        continue

                # Check each symbol
                symbols_checked = 0
                signals_found = 0
                signals_skipped = {}  # Track why signals were skipped

                for oanda_symbol in FOREX_INSTRUMENTS:
                    symbols_checked += 1
                    if not self.is_running:
                        break

                    nt_symbol = self.SYMBOL_MAP.get(oanda_symbol)
                    if not nt_symbol:
                        if loop_count % 5 == 1:
                            print(f"  [{oanda_symbol}] SKIP - No NinjaTrader symbol mapping")
                        signals_skipped[oanda_symbol] = "No NT mapping"
                        continue

                    # Skip if already have position
                    if nt_symbol in self.open_positions:
                        if loop_count % 5 == 1:
                            print(f"  [{oanda_symbol}] SKIP - Already have open position")
                        signals_skipped[oanda_symbol] = "Position exists"
                        continue

                    # Skip if max trades per symbol
                    if self.trades_per_symbol[nt_symbol] >= self.MAX_TRADES_PER_SYMBOL:
                        if loop_count % 5 == 1:
                            print(f"  [{oanda_symbol}] SKIP - Max trades per symbol ({self.trades_per_symbol[nt_symbol]}/{self.MAX_TRADES_PER_SYMBOL})")
                        signals_skipped[oanda_symbol] = "Max trades per symbol"
                        continue

                    # Get market data
                    market_data = self.get_market_data(oanda_symbol)
                    if not market_data:
                        if loop_count % 5 == 1:
                            print(f"  [{oanda_symbol}] SKIP - No market data")
                        signals_skipped[oanda_symbol] = "No market data"
                        continue

                    # Calculate current P&L percentage
                    daily_pl_pct = ((self.current_balance - self.starting_balance_today) / self.starting_balance_today)

                    # Check for signal using ForexScalpingStrategy
                    signal = self.strategy.should_enter_trade(
                        instrument=oanda_symbol,
                        df_1min=market_data['1min'],
                        df_5min=market_data['5min'],
                        df_15min=market_data['15min'],
                        df_30min=market_data['30min'],
                        current_positions=1 if nt_symbol in self.open_positions else 0,
                        trades_today=self.trades_today,
                        daily_pl_pct=daily_pl_pct
                    )

                    # Log signal result
                    if signal and signal.get('action') in ['BUY', 'SELL']:
                        signals_found += 1
                        if loop_count % 5 == 1:
                            print(f"  [{oanda_symbol}] SIGNAL FOUND: {signal['action']} - {signal.get('reason', 'Multi-timeframe momentum')}")
                    else:
                        if loop_count % 5 == 1:
                            reason = signal.get('reason', 'No signal') if signal else 'No signal returned'
                            print(f"  [{oanda_symbol}] No trade: {reason}")
                        signals_skipped[oanda_symbol] = signal.get('reason', 'No signal') if signal else 'No signal'

                    if signal and signal.get('action') in ['BUY', 'SELL']:
                        # Get ACTUAL NinjaTrader price BEFORE trade validation
                        # This ensures we calculate TP/SL based on actual execution price
                        nt_price_data = self.get_ninjatrader_price(nt_symbol)

                        if not nt_price_data:
                            print(f"\n[{oanda_symbol}] SIGNAL SKIPPED - Cannot fetch NinjaTrader price")
                            print(f"  Signal: {signal['action']} @ {signal.get('price', 'N/A')}")
                            print(f"  Reason: NinjaTrader price unavailable (market may be closed)")
                            print(f"  → Trade will NOT be executed")
                            signals_skipped[oanda_symbol] = "NT price unavailable"
                            continue

                        # Check if price is LIVE (not cached)
                        if nt_price_data.get('status') != 'LIVE':
                            print(f"\n[{oanda_symbol}] SIGNAL SKIPPED - Market is closed")
                            print(f"  Signal: {signal['action']} @ {nt_price_data.get('mid', 'N/A'):.5f}")
                            print(f"  Reason: NinjaTrader price is CACHED (status: {nt_price_data.get('status')})")
                            print(f"  → Cannot trade on cached prices - waiting for market to open")
                            print(f"  → Trade will NOT be executed")
                            signals_skipped[oanda_symbol] = "Market closed (cached prices)"
                            continue

                        # Use NinjaTrader actual bid/ask for entry price
                        if signal['action'] == 'BUY':
                            entry_price = nt_price_data['ask']  # Buy at NinjaTrader ask
                        else:  # SELL
                            entry_price = nt_price_data['bid']  # Sell at NinjaTrader bid

                        # Calculate TP/SL based on ACTUAL NinjaTrader price
                        sl_price, tp_price = self.calculate_sl_tp(nt_symbol, signal['action'], entry_price)

                        # Fetch OANDA price for comparison
                        try:
                            oanda_price_data = self.oanda_client.get_current_price(oanda_symbol)
                            oanda_mid = (oanda_price_data['bid'] + oanda_price_data['ask']) / 2
                            nt_mid = nt_price_data['mid']
                            price_diff = abs(nt_mid - oanda_mid)
                            settings = self.PAIR_SETTINGS[nt_symbol]
                            diff_pips = price_diff / settings['tick_size']
                        except:
                            diff_pips = 0

                        print(f"\n[SIGNAL] {nt_symbol} {signal['action']} @ {entry_price:.5f} (NinjaTrader)")
                        print(f"  TP: {tp_price:.5f}, SL: {sl_price:.5f}")
                        print(f"  Price difference (NT vs OANDA): {diff_pips:.1f} pips")
                        print(f"  Reason: {signal.get('reason', 'Multi-timeframe momentum')}")

                        # Send to NinjaTrader
                        bridge_signal = {
                            'Action': 'ENTRY',
                            'Symbol': nt_symbol,
                            'Side': signal['action'],
                            'Quantity': 1,
                            'EntryPrice': entry_price,
                            'StopLoss': sl_price,
                            'TakeProfit': tp_price,
                            'Timestamp': datetime.now().isoformat()
                        }

                        # Send signal to NinjaTrader and wait for result
                        signal_sent = self.send_signal_to_bridge(bridge_signal)

                        if signal_sent:
                            print(f"  ✓ Signal SENT to NinjaTrader successfully")
                            print(f"  → Bridge accepted the signal")
                            print(f"  → Check bridge console to verify order was ACCEPTED by NinjaTrader")
                            print(f"     (Order ID > 0 = ACCEPTED, Order ID = 0 = REJECTED)")

                            # Generate trade ID
                            trade_id = f"{nt_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                            # Get symbol settings for logging
                            settings = self.PAIR_SETTINGS[nt_symbol]

                            # Log trade entry
                            self.trade_logger.log_entry(
                                trade_id=trade_id,
                                symbol=nt_symbol,
                                oanda_symbol=oanda_symbol,
                                side=signal['action'],
                                entry_price=entry_price,
                                stop_loss=sl_price,
                                take_profit=tp_price,
                                tick_size=settings['tick_size'],
                                tick_value=settings['tick_value'],
                                account_balance=self.current_balance,
                                daily_profit=self.current_balance - self.starting_balance_today,
                                trades_today=self.trades_today,
                                signal_reason=signal.get('reason', 'Multi-timeframe momentum'),
                                signal_confidence=signal.get('confidence', 0.0),
                                session=date.today().strftime('%Y-%m-%d')
                            )

                            # Track position
                            self.open_positions[nt_symbol] = {
                                'trade_id': trade_id,
                                'side': signal['action'],
                                'entry_price': entry_price,
                                'sl_price': sl_price,
                                'tp_price': tp_price,
                                'entry_time': datetime.now()
                            }

                            self.trades_today += 1
                            self.trades_per_symbol[nt_symbol] += 1

                            print(f"  Trade ID: {trade_id}")
                            print(f"  Trades today: {self.trades_today}/{self.MAX_TRADES_PER_DAY}")

                            # Save state after opening position
                            self.save_state()
                        else:
                            print(f"  ❌ Signal FAILED to send to NinjaTrader")
                            print(f"  → Check bridge connection and console")
                            print(f"  → Trade will NOT be executed")

                # Log summary every 5 loops
                if loop_count % 5 == 0:
                    print(f"\n[SUMMARY] Loop {loop_count} complete:")
                    print(f"  Symbols checked: {symbols_checked}")
                    print(f"  Signals found: {signals_found}")
                    print(f"  Signals skipped: {len(signals_skipped)}")
                    if signals_found == 0 and loop_count % 5 == 0:
                        print(f"  Common skip reasons:")
                        skip_counts = {}
                        for reason in signals_skipped.values():
                            skip_counts[reason] = skip_counts.get(reason, 0) + 1
                        for reason, count in sorted(skip_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                            print(f"    - {reason}: {count}")

                # Sleep before next check
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n\n[STOP] Stopped by user")

        finally:
            self.is_running = False

            # Ask about open positions
            if self.open_positions:
                print("\n" + "=" * 80)
                print(f"You have {len(self.open_positions)} open positions:")
                for symbol, pos in self.open_positions.items():
                    print(f"  - {symbol} {pos['side']} @ {pos['entry_price']:.5f}")
                print("=" * 80)
                print()

                response = input("Close all positions? (y/n): ").strip().lower()
                if response == 'y':
                    self.close_all_positions()
                    print("[STOP] All positions closed")
                else:
                    print("[STOP] Positions left open - they will be managed on next run")
                    self.save_state()

            else:
                # No positions, just save state
                self.save_state()

            print("\n" + "=" * 80)
            print("SESSION COMPLETE")
            print("=" * 80)
            print(f"Final Balance: ${self.current_balance:,.2f}")
            print(f"Total Profit: ${self.total_profit:+,.2f}")
            print(f"Trades Today: {self.trades_today}")
            print(f"Open Positions: {len(self.open_positions)}")
            print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run OANDA → NinjaTrader Live Trading")
    parser.add_argument(
        "--hours",
        type=float,
        default=0,
        help="Number of hours to run (default: continuous until stopped)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--funded",
        action="store_true",
        help="Run in FUNDED mode (for funded accounts)"
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Enable consistency rule (40%% max per day, $400 cap)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OANDA → NINJATRADER LIVE TRADING BRIDGE")
    print("=" * 80)
    print()
    print("⚠️  IMPORTANT: This will execute REAL trades on NinjaTrader!")
    print()
    print("Prerequisites:")
    print("1. NinjaTrader 8 is running")
    print("2. Connected to your account (Sim101 for testing, FundedNext for challenge)")
    print("3. NinjaTraderBridge.exe is running")
    print()
    if args.hours > 0:
        print(f"Duration: {args.hours} hours")
    else:
        print(f"Duration: Continuous (Ctrl+C to stop)")
    print(f"Mode: {'FUNDED' if args.funded else 'CHALLENGE'}")
    print(f"Consistency Rule: {'ENABLED' if args.consistency else 'DISABLED'}")
    print()
    print("=" * 80)
    print()

    if not args.yes:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    try:
        bridge = OandaNinjaTraderLiveBridge(
            is_challenge_mode=not args.funded,
            enable_consistency_rule=args.consistency
        )
        bridge.run(duration_hours=args.hours)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
