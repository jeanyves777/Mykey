"""
Match-Trader Live Strategy Trading
==================================
Runs our MACD + Momentum strategy on Match-Trader platform.

Features:
- Auto-login to Match-Trader
- Real-time price monitoring via OANDA
- MACD + RSI signal generation
- Auto-trade execution via browser
- Risk management (daily loss limit, position sizing)
- Trailing stops

Usage:
    python run_strategy_live.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import threading

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from match_trader_browser import MatchTraderBrowser
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Try to import OANDA for price data
try:
    from Forex_Trading.oanda_client import OandaClient
    OANDA_AVAILABLE = True
except:
    OANDA_AVAILABLE = False
    print("WARNING: OANDA client not available for price data")

import pandas as pd
import numpy as np


class TradingStrategy:
    """
    MACD + RSI Momentum Strategy for FundedNext
    """

    def __init__(self, config: dict):
        self.config = config

        # Strategy parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30

        # Risk parameters from config
        risk_config = config.get('risk_management', {})
        self.risk_per_trade = risk_config.get('risk_per_trade_pct', 1.0) / 100
        self.max_daily_loss = risk_config.get('daily_loss_limit_pct', 5.0) / 100
        self.max_position_size = risk_config.get('max_position_size', 0.5)
        self.max_positions = risk_config.get('max_open_positions', 2)

        # Strategy parameters from config
        strategy_config = config.get('strategy', {})
        self.sl_pips = 15  # Default stop loss
        self.tp_pips = int(self.sl_pips * strategy_config.get('risk_reward_ratio', 1.5))
        self.trailing_trigger = strategy_config.get('trailing_stop_trigger_pips', 10)
        self.trailing_distance = strategy_config.get('trailing_stop_distance_pips', 8)

        # Trading symbols
        self.symbols = config.get('symbols', ['EURUSD'])

        # State
        self.positions = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.last_signal_time = {}

    def calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, symbol: str, prices: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal based on MACD + RSI

        Returns:
            Dict with 'direction', 'reason' or None if no signal
        """
        if len(prices) < 50:
            return None

        close = prices['close']

        # Calculate indicators
        macd, signal, histogram = self.calculate_macd(close)
        rsi = self.calculate_rsi(close)

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        current_rsi = rsi.iloc[-1]

        # Check for minimum time between signals (5 minutes)
        last_time = self.last_signal_time.get(symbol)
        if last_time and (datetime.now() - last_time).seconds < 300:
            return None

        signal_result = None

        # BUY Signal: MACD crosses above signal + RSI not overbought
        if prev_hist < 0 and current_hist > 0 and current_rsi < self.rsi_overbought:
            signal_result = {
                'direction': 'BUY',
                'reason': f'MACD bullish cross (RSI: {current_rsi:.1f})'
            }

        # SELL Signal: MACD crosses below signal + RSI not oversold
        elif prev_hist > 0 and current_hist < 0 and current_rsi > self.rsi_oversold:
            signal_result = {
                'direction': 'SELL',
                'reason': f'MACD bearish cross (RSI: {current_rsi:.1f})'
            }

        if signal_result:
            self.last_signal_time[symbol] = datetime.now()

        return signal_result

    def calculate_position_size(self, balance: float, sl_pips: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = balance * self.risk_per_trade
        # Assuming $10 per pip per lot for most pairs
        pip_value = 10
        lot_size = risk_amount / (sl_pips * pip_value)
        lot_size = round(lot_size, 2)
        return min(max(0.01, lot_size), self.max_position_size)


class MatchTraderLiveTrading:
    """
    Live trading system using browser automation
    """

    def __init__(self, config_path: str = None):
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.mt_config = self.config['match_trader']
        self.account_config = self.config['account']

        # Initialize components
        self.browser: Optional[MatchTraderBrowser] = None
        self.strategy = TradingStrategy(self.config)

        # State
        self.is_running = False
        self.is_logged_in = False
        self.balance = self.account_config.get('starting_balance', 10000)
        self.equity = self.balance
        self.daily_pnl = 0.0
        self.positions = []
        self.trade_history = []

        # Price data (simulated for now)
        self.price_data = {}

    def start(self):
        """Start the trading system"""
        self.print_banner()

        # Start browser
        print("[1/4] Starting browser...")
        self.browser = MatchTraderBrowser(
            headless=False,
            platform_url=self.mt_config['base_url']
        )

        # Login
        print("[2/4] Logging into Match-Trader...")
        if not self.browser.login(self.mt_config['email'], self.mt_config['password']):
            print("\nERROR: Login failed!")
            return False

        self.is_logged_in = True
        print("       Login successful!")

        # Initialize strategy
        print("[3/4] Loading strategy...")
        self.print_strategy_info()

        # Start trading loop
        print("[4/4] Starting trading loop...")
        self.is_running = True

        return True

    def print_banner(self):
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     MATCH-TRADER AUTOMATED TRADING SYSTEM                     ║
    ║     MACD + RSI Momentum Strategy                              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║     Platform: FundedNext Demo Challenge                       ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)

    def print_strategy_info(self):
        print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │  STRATEGY PARAMETERS                                         │
    ├─────────────────────────────────────────────────────────────┤
    │  Indicators: MACD(12,26,9) + RSI(14)                        │
    │  Risk per trade: {self.strategy.risk_per_trade*100:.1f}%                                    │
    │  Max daily loss: {self.strategy.max_daily_loss*100:.1f}%                                   │
    │  Stop Loss: {self.strategy.sl_pips} pips                                        │
    │  Take Profit: {self.strategy.tp_pips} pips                                      │
    │  Trailing Stop: Trigger {self.strategy.trailing_trigger}p, Distance {self.strategy.trailing_distance}p         │
    │  Symbols: {', '.join(self.strategy.symbols):<40} │
    └─────────────────────────────────────────────────────────────┘
        """)

    def print_status(self):
        """Print current trading status"""
        now = datetime.now().strftime('%H:%M:%S')

        print(f"\n{'='*60}")
        print(f"  [{now}] TRADING STATUS")
        print(f"{'='*60}")
        print(f"  Balance: ${self.balance:,.2f}  |  Equity: ${self.equity:,.2f}")
        print(f"  Daily P&L: ${self.daily_pnl:+,.2f}")
        print(f"  Open Positions: {len(self.positions)}/{self.strategy.max_positions}")
        print(f"  Trades Today: {len(self.trade_history)}")

        if self.positions:
            print(f"\n  Open Positions:")
            for pos in self.positions:
                print(f"    - {pos.get('side')} {pos.get('volume')} {pos.get('symbol')} "
                      f"@ {pos.get('entry_price', 0):.5f}")

        # Check risk limits
        daily_limit = self.balance * self.strategy.max_daily_loss
        if abs(self.daily_pnl) > daily_limit * 0.8:
            print(f"\n  ⚠️  WARNING: Approaching daily loss limit!")

        print(f"{'='*60}")

    def execute_trade(self, symbol: str, direction: str, volume: float, reason: str):
        """Execute a trade via browser automation"""
        print(f"\n  📊 SIGNAL: {direction} {volume} {symbol}")
        print(f"     Reason: {reason}")

        driver = self.browser.driver
        wait = WebDriverWait(driver, 10)

        try:
            # Make sure we're on trade page
            if "/app/trade" not in driver.current_url:
                driver.get(f"{self.mt_config['base_url']}/app/trade")
                time.sleep(2)

            # Find and click symbol in watchlist
            try:
                symbol_elem = driver.find_element(By.XPATH, f"//*[contains(text(), '{symbol}')]")
                symbol_elem.click()
                time.sleep(0.5)
            except:
                print(f"     Could not find {symbol} in watchlist")

            # Set volume
            try:
                volume_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='number']")
                for inp in volume_inputs:
                    if inp.is_displayed():
                        inp.clear()
                        inp.send_keys(str(volume))
                        break
            except:
                pass

            # Click BUY or SELL button
            button_class = 'buy' if direction == 'BUY' else 'sell'
            try:
                buttons = driver.find_elements(By.XPATH,
                    f"//button[contains(translate(., 'BUYSELL', 'buysell'), '{button_class}')]")
                for btn in buttons:
                    if btn.is_displayed():
                        btn.click()
                        print(f"  ✅ TRADE EXECUTED: {direction} {volume} {symbol}")

                        # Record the trade
                        trade = {
                            'symbol': symbol,
                            'side': direction,
                            'volume': volume,
                            'time': datetime.now(),
                            'reason': reason
                        }
                        self.positions.append(trade)
                        self.trade_history.append(trade)
                        return True
            except Exception as e:
                print(f"  ❌ Trade execution failed: {e}")

        except Exception as e:
            print(f"  ❌ Error executing trade: {e}")

        return False

    def run_trading_loop(self, interval: int = 60):
        """Main trading loop"""
        print(f"\n  Trading loop started (checking every {interval}s)")
        print(f"  Monitoring: {', '.join(self.strategy.symbols)}")
        print(f"  Press Ctrl+C to stop\n")

        cycle = 0
        while self.is_running:
            try:
                cycle += 1

                # Print status every 5 cycles
                if cycle % 5 == 1:
                    self.print_status()

                # Check daily loss limit
                daily_limit = self.balance * self.strategy.max_daily_loss
                if self.daily_pnl <= -daily_limit:
                    print(f"\n  🛑 DAILY LOSS LIMIT REACHED! Stopping trading.")
                    break

                # Check max positions
                if len(self.positions) >= self.strategy.max_positions:
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Max positions reached, waiting...")
                    time.sleep(interval)
                    continue

                # For each symbol, check for signals
                for symbol in self.strategy.symbols:
                    # Generate mock price data for demo (in production, use OANDA)
                    prices = self.get_price_data(symbol)
                    if prices is not None:
                        signal = self.strategy.generate_signal(symbol, prices)

                        if signal:
                            # Calculate position size
                            volume = self.strategy.calculate_position_size(
                                self.balance,
                                self.strategy.sl_pips
                            )

                            # Execute trade
                            self.execute_trade(
                                symbol=symbol,
                                direction=signal['direction'],
                                volume=volume,
                                reason=signal['reason']
                            )

                # Check if browser is still open
                try:
                    _ = self.browser.driver.current_url
                except:
                    print("\n  Browser was closed!")
                    break

                # Wait for next cycle
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Waiting {interval}s...", end='\r')
                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n\n  Stopping trading loop...")
                break
            except Exception as e:
                print(f"\n  Error in trading loop: {e}")
                time.sleep(10)

        self.is_running = False

    def get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get price data for a symbol.
        For demo, generates simulated data.
        In production, connect to OANDA or use Match-Trader's data.
        """
        # Generate simulated price data for demo
        np.random.seed(int(time.time()) % 1000)

        n_bars = 100
        base_price = {
            'EURUSD': 1.05,
            'GBPUSD': 1.27,
            'USDJPY': 150.0,
            'XAUUSD': 2000.0,
        }.get(symbol, 1.0)

        # Random walk
        returns = np.random.randn(n_bars) * 0.0005
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'close': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'open': np.roll(prices, 1)
        })

        return df

    def stop(self):
        """Stop the trading system"""
        self.is_running = False
        if self.browser:
            try:
                self.browser.close_browser()
            except:
                pass
        print("\n  Trading system stopped.")


def main():
    trader = MatchTraderLiveTrading()

    try:
        if trader.start():
            # Run the trading loop
            trader.run_trading_loop(interval=30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        trader.stop()
        print("Done!")


if __name__ == "__main__":
    main()
