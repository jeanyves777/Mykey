"""
SCALPING STRATEGY - PAPER TRADING RUNNER
=========================================

Multi-indicator validation scalping with 5-15 pip targets
STRICT entry requirements - NO OVERTRADING

Requirements (ALL must pass):
1. Trend alignment (EMA9>EMA21>EMA50)
2. Momentum confirmation (RSI 40-60)
3. MACD trigger (crossover + histogram)
4. Candle confirmation (strong body >40%)
5. Volatility filter (ATR in range)

Target: 75%+ win rate with quick in/out scalps
"""

import sys
import os
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config and strategy
from trading_system.Forex_Trading.config import scalping_config as config
from trading_system.Forex_Trading.strategies.scalping_strategy import get_signal, calculate_indicators

# Import OANDA client
sys.path.insert(0, str(Path(__file__).parent / 'trading_system' / 'Forex_Trading' / 'engine'))
from oanda_client import OandaClient


class ScalpingTrader:
    """Scalping trader with strict multi-indicator validation."""
    
    def __init__(self, account_type='practice'):
        self.account_type = account_type
        self.client = OandaClient(account_type=account_type)
        
        # Track positions and performance
        self.positions = {}  # pair -> position info
        self.last_trade_time = {}  # pair -> last trade timestamp
        self.daily_trades = 0
        self.daily_pnl_pips = 0
        self.consecutive_losses = 0
        
        # Reset daily stats at start
        self.last_reset_date = datetime.utcnow().date()
        
        print("=" * 70)
        print("SCALPING STRATEGY INITIALIZED")
        print("=" * 70)
        config.print_config_info()
    
    def reset_daily_stats(self):
        """Reset daily counters."""
        self.daily_trades = 0
        self.daily_pnl_pips = 0
        self.last_reset_date = datetime.utcnow().date()
        print(f"\n[{datetime.utcnow()}] Daily stats reset")
    
    def check_daily_limits(self) -> bool:
        """Check if we've hit daily limits."""
        # Check if new day
        if datetime.utcnow().date() > self.last_reset_date:
            self.reset_daily_stats()
        
        # Check max daily loss
        if self.daily_pnl_pips <= -config.MAX_DAILY_LOSS_PIPS:
            print(f"[LIMIT] Daily loss limit hit: {self.daily_pnl_pips} pips")
            return False
        
        # Check max daily trades
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            print(f"[LIMIT] Daily trade limit hit: {self.daily_trades} trades")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            print(f"[LIMIT] Consecutive loss limit hit: {self.consecutive_losses} losses")
            return False
        
        return True
    
    def check_cooldown(self, pair: str) -> bool:
        """Check if cooldown period has passed for this pair."""
        if pair not in self.last_trade_time:
            return True
        
        settings = config.get_pair_settings(pair)
        cooldown_minutes = settings.get('cooldown_minutes', config.COOLDOWN_MINUTES)
        
        if cooldown_minutes == 0:
            return True
        
        time_since = (datetime.utcnow() - self.last_trade_time[pair]).total_seconds() / 60
        return time_since >= cooldown_minutes
    
    def check_session_filter(self, pair: str) -> bool:
        """Check if we're in allowed trading session."""
        hour_utc = datetime.utcnow().hour
        return config.is_allowed_hour(pair, hour_utc)
    
    def get_candles(self, pair: str, count: int = None) -> list:
        """Fetch recent candles from OANDA."""
        if count is None:
            count = config.CANDLE_COUNT
        
        return self.client.get_candles(pair, config.TIMEFRAME, count=count)
    
    def calculate_position_size(self, pair: str) -> int:
        """Calculate position size in units."""
        return config.POSITION_SIZE_UNITS
    
    def check_entry_signal(self, pair: str) -> tuple:
        """Check if there's a valid entry signal for this pair."""
        # Check if already in position
        if pair in self.positions:
            return None, "Already in position"
        
        # Check cooldown
        if not self.check_cooldown(pair):
            return None, "In cooldown period"
        
        # Check session filter
        if not self.check_session_filter(pair):
            return None, "Outside trading session"
        
        # Fetch candles
        candles = self.get_candles(pair)
        if not candles or len(candles) < 50:
            return None, "Insufficient data"
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Get signal from multi-indicator strategy
        signal, reason = get_signal(pair, df, config)
        
        return signal, reason
    
    def place_order(self, pair: str, direction: str):
        """Place scalping order with TP/SL."""
        settings = config.get_pair_settings(pair)
        tp_pips = settings['tp_pips']
        sl_pips = settings['sl_pips']
        
        units = self.calculate_position_size(pair)
        if direction == 'SELL':
            units = -units
        
        try:
            # Place market order with TP/SL
            response = self.client.place_order(
                instrument=pair,
                units=units,
                tp_pips=tp_pips,
                sl_pips=sl_pips
            )
            
            if response:
                # Track position
                self.positions[pair] = {
                    'direction': direction,
                    'entry_time': datetime.utcnow(),
                    'tp_pips': tp_pips,
                    'sl_pips': sl_pips,
                    'order_id': response.get('orderFillTransaction', {}).get('id')
                }
                
                # Update tracking
                self.last_trade_time[pair] = datetime.utcnow()
                self.daily_trades += 1
                
                print(f"\n[ENTRY] {pair} {direction}")
                print(f"  TP: {tp_pips}p | SL: {sl_pips}p")
                print(f"  Daily trades: {self.daily_trades}/{config.MAX_DAILY_TRADES}")
                print(f"  Daily P&L: {self.daily_pnl_pips:+.0f} pips")
                
                return True
        
        except Exception as e:
            print(f"[ERROR] Failed to place order for {pair}: {e}")
            return False
    
    def check_exits(self):
        """Check if any positions have been closed."""
        open_positions = self.client.get_open_positions()
        open_pairs = {pos['instrument'] for pos in open_positions}
        
        # Check for closed positions
        closed_pairs = []
        for pair in list(self.positions.keys()):
            if pair not in open_pairs:
                closed_pairs.append(pair)
        
        # Process closed positions
        for pair in closed_pairs:
            pos_info = self.positions.pop(pair)
            
            # Try to get trade result (simplified - assume SL or TP)
            # In real implementation, check trade history for actual P&L
            print(f"\n[EXIT] {pair} {pos_info['direction']} closed")
            print(f"  Entry time: {pos_info['entry_time']}")
            print(f"  Hold time: {datetime.utcnow() - pos_info['entry_time']}")
    
    def run_trading_loop(self):
        """Main trading loop."""
        print("\n" + "=" * 70)
        print("STARTING SCALPING STRATEGY")
        print("=" * 70)
        print(f"Account: {self.account_type.upper()}")
        print(f"Pairs: {', '.join(config.SCALPING_PAIRS)}")
        print(f"Press Ctrl+C to stop")
        print("=" * 70)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                current_time = datetime.utcnow()
                
                # Check daily limits
                if not self.check_daily_limits():
                    print(f"\n[{current_time}] Daily limits reached. Waiting for next day...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Check exits first
                self.check_exits()
                
                # Status update every 10 iterations
                if iteration % 10 == 0:
                    print(f"\n[{current_time}] Status:")
                    print(f"  Positions: {len(self.positions)}/{config.MAX_CONCURRENT_POSITIONS}")
                    print(f"  Daily trades: {self.daily_trades}/{config.MAX_DAILY_TRADES}")
                    print(f"  Daily P&L: {self.daily_pnl_pips:+.0f} pips")
                
                # Check for new entries (if not at max positions)
                if len(self.positions) < config.MAX_CONCURRENT_POSITIONS:
                    for pair in config.SCALPING_PAIRS:
                        signal, reason = self.check_entry_signal(pair)
                        
                        if signal:
                            print(f"\n[SIGNAL] {pair} {signal}: {reason}")
                            self.place_order(pair, signal)
                            break  # Only one entry per iteration
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
            
            except KeyboardInterrupt:
                print("\n\nStopping trading...")
                break
            
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # Wait 1 minute on error


def main():
    parser = argparse.ArgumentParser(description="Run Scalping Strategy")
    parser.add_argument(
        "--account",
        type=str,
        default="practice",
        choices=["practice", "live"],
        help="Account type: practice (demo) or live"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    if args.account == "live":
        print("\n" + "=" * 70)
        print("WARNING: LIVE TRADING WITH REAL MONEY!")
        print("=" * 70)
        print("\nThis is a scalping strategy that requires:")
        print("  - Fast execution")
        print("  - Low spreads")
        print("  - Proper risk management")
        print("\nMake sure you:")
        print("  - Understand the strategy completely")
        print("  - Have tested it thoroughly on paper account")
        print("  - Are comfortable with the risk")
        print("=" * 70)
    
    if not args.yes:
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    try:
        trader = ScalpingTrader(account_type=args.account)
        trader.run_trading_loop()
    
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
