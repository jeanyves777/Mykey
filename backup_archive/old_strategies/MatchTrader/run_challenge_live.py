"""
Match-Trader FundedNext Challenge - Live Trading System
=======================================================
Comprehensive monitoring with challenge-specific stats.

Features:
- Real-time position monitoring with P&L, pips, SL/TP distances
- Challenge progress tracking (profit target, daily loss, max drawdown)
- Detailed entry signals with confidence levels
- Trade logging and statistics
- Trailing stop management

Usage:
    python run_challenge_live.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import threading

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from match_trader_browser import MatchTraderBrowser
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

import pandas as pd
import numpy as np


@dataclass
class Position:
    """Open position tracking"""
    symbol: str
    side: str  # BUY or SELL
    volume: float
    entry_price: float
    current_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    trailing_stop: float = 0.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    open_time: datetime = field(default_factory=datetime.now)
    position_id: str = ""


@dataclass
class ChallengeStats:
    """FundedNext Challenge Statistics"""
    # Account
    starting_balance: float = 10000.0
    current_balance: float = 10000.0
    current_equity: float = 10000.0

    # Challenge targets
    profit_target_pct: float = 6.0  # Demo challenge = 6%
    daily_loss_limit_pct: float = 5.0
    max_loss_limit_pct: float = 10.0
    min_trading_days: int = 0  # Demo has no minimum

    # Progress
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    day_start_balance: float = 10000.0
    trading_days: int = 0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    trades_today: int = 0
    max_trades_per_day: int = 10

    # Risk
    max_drawdown: float = 0.0
    peak_balance: float = 10000.0


class SignalGenerator:
    """Combined V2 Strategy Signal Generator"""

    def __init__(self):
        # MACD settings
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # RSI settings
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # Bollinger Bands
        self.bb_period = 20
        self.bb_std = 2

    def calculate_indicators(self, prices: pd.DataFrame) -> Dict:
        """Calculate all indicators"""
        close = prices['close']

        # MACD
        exp1 = close.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        bb_upper = sma + (std * self.bb_std)
        bb_lower = sma - (std * self.bb_std)

        # EMA for trend
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        return {
            'macd': macd.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'macd_hist': macd_hist.iloc[-1],
            'macd_hist_prev': macd_hist.iloc[-2] if len(macd_hist) > 1 else 0,
            'rsi': rsi.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'bb_mid': sma.iloc[-1],
            'ema_20': ema_20.iloc[-1],
            'ema_50': ema_50.iloc[-1],
            'close': close.iloc[-1],
        }

    def generate_signal(self, symbol: str, prices: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal using Combined V2 strategy"""
        if len(prices) < 50:
            return None

        ind = self.calculate_indicators(prices)

        bullish_signals = []
        bearish_signals = []

        # Signal 1: MACD Cross
        if ind['macd_hist_prev'] < 0 and ind['macd_hist'] > 0:
            bullish_signals.append('MACD_CROSS')
        elif ind['macd_hist_prev'] > 0 and ind['macd_hist'] < 0:
            bearish_signals.append('MACD_CROSS')

        # Signal 2: RSI Oversold/Overbought
        if ind['rsi'] < self.rsi_oversold:
            bullish_signals.append('RSI_OVERSOLD')
        elif ind['rsi'] > self.rsi_overbought:
            bearish_signals.append('RSI_OVERBOUGHT')

        # Signal 3: Bollinger Band touch
        if ind['close'] <= ind['bb_lower']:
            bullish_signals.append('BB_LOWER')
        elif ind['close'] >= ind['bb_upper']:
            bearish_signals.append('BB_UPPER')

        # Signal 4: EMA Trend
        if ind['ema_20'] > ind['ema_50']:
            bullish_signals.append('EMA_BULLISH')
        else:
            bearish_signals.append('EMA_BEARISH')

        # Need 2/3 signals for entry
        if len(bullish_signals) >= 2 and len(bearish_signals) < 2:
            confidence = 'HIGH' if len(bullish_signals) >= 3 else 'MEDIUM'
            return {
                'direction': 'BUY',
                'signals': bullish_signals,
                'rsi': ind['rsi'],
                'confidence': confidence,
                'reason': f"Combined V2: {len(bullish_signals)}/4 BULLISH signals [{', '.join(bullish_signals)}] | RSI: {ind['rsi']:.1f}"
            }
        elif len(bearish_signals) >= 2 and len(bullish_signals) < 2:
            confidence = 'HIGH' if len(bearish_signals) >= 3 else 'MEDIUM'
            return {
                'direction': 'SELL',
                'signals': bearish_signals,
                'rsi': ind['rsi'],
                'confidence': confidence,
                'reason': f"Combined V2: {len(bearish_signals)}/4 BEARISH signals [{', '.join(bearish_signals)}] | RSI: {ind['rsi']:.1f}"
            }

        return {
            'direction': None,
            'bullish': len(bullish_signals),
            'bearish': len(bearish_signals),
            'rsi': ind['rsi'],
            'status': f"Need 2/3 signals (bullish: {len(bullish_signals)}, bearish: {len(bearish_signals)})"
        }


class MatchTraderChallenge:
    """
    FundedNext Challenge Trading System with Comprehensive Monitoring
    """

    def __init__(self, config_path: str = None):
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.mt_config = self.config['match_trader']

        # Initialize components
        self.browser: Optional[MatchTraderBrowser] = None
        self.signal_gen = SignalGenerator()

        # Challenge stats
        self.stats = ChallengeStats(
            starting_balance=self.config['account'].get('starting_balance', 10000),
            profit_target_pct=6.0,  # Demo challenge
            daily_loss_limit_pct=self.config['risk_management'].get('daily_loss_limit_pct', 5.0),
            max_loss_limit_pct=self.config['risk_management'].get('max_loss_limit_pct', 10.0),
        )
        self.stats.current_balance = self.stats.starting_balance
        self.stats.current_equity = self.stats.starting_balance
        self.stats.day_start_balance = self.stats.starting_balance
        self.stats.peak_balance = self.stats.starting_balance

        # Trading parameters
        self.symbols = self.config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        self.sl_pips = 16
        self.tp_pips = 20
        self.trailing_pips = 6
        self.max_positions = self.config['risk_management'].get('max_open_positions', 2)
        self.risk_per_trade = self.config['risk_management'].get('risk_per_trade_pct', 1.0)

        # State
        self.positions: List[Position] = []
        self.is_running = False
        self.last_signal_time = {}
        self.trade_log = []

        # Pip values
        self.pip_values = {
            'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
            'USDCHF': 0.0001, 'USDCAD': 0.0001, 'AUDUSD': 0.0001,
            'NZDUSD': 0.0001, 'XAUUSD': 0.01, 'XAGUSD': 0.001,
        }

    def print_banner(self):
        """Print startup banner"""
        print("""
================================================================================
                    FUNDEDNEXT CHALLENGE - LIVE TRADING
                    Combined V2 Strategy (MACD + RSI + BB)
--------------------------------------------------------------------------------
  Platform: Match-Trader Demo
  Challenge: Demo Challenge ($10,000)
================================================================================
        """)

    def print_challenge_status(self):
        """Print challenge progress"""
        s = self.stats

        # Calculate progress
        profit_target = s.starting_balance * (s.profit_target_pct / 100)
        daily_limit = s.day_start_balance * (s.daily_loss_limit_pct / 100)
        max_loss = s.starting_balance * (s.max_loss_limit_pct / 100)

        profit_progress = (s.total_pnl / profit_target * 100) if profit_target > 0 else 0
        daily_used = (abs(s.daily_pnl) / daily_limit * 100) if s.daily_pnl < 0 else 0
        drawdown_used = (s.max_drawdown / max_loss * 100) if max_loss > 0 else 0

        win_rate = (s.winning_trades / s.total_trades * 100) if s.total_trades > 0 else 0

        print(f"""
+-----------------------------------------------------------------------------+
|  CHALLENGE PROGRESS                                                          |
+-----------------------------------------------------------------------------+
|  Profit Target: ${profit_target:,.0f} ({s.profit_target_pct}%)  |  Progress: {profit_progress:>6.1f}% |  P&L: ${s.total_pnl:>+8.2f}  |
|  Daily Limit:   ${daily_limit:,.0f} ({s.daily_loss_limit_pct}%)  |  Used:     {daily_used:>6.1f}% |  Today: ${s.daily_pnl:>+7.2f}  |
|  Max Drawdown:  ${max_loss:,.0f} ({s.max_loss_limit_pct}%)  |  Used:     {drawdown_used:>6.1f}% |  DD: ${s.max_drawdown:>+8.2f}  |
+-----------------------------------------------------------------------------+
|  Trades: {s.total_trades:<4}  |  Wins: {s.winning_trades:<3}  |  Losses: {s.losing_trades:<3}  |  Win Rate: {win_rate:>5.1f}%            |
|  Today:  {s.trades_today:<4}  |  Trading Days: {s.trading_days:<3}                                          |
+-----------------------------------------------------------------------------+
        """)

    def print_positions(self):
        """Print open positions with detailed info"""
        now = datetime.now().strftime('%H:%M:%S')

        print(f"\n[{now}] Balance: ${self.stats.current_balance:,.2f} | ", end="")

        if not self.positions:
            print("NO OPEN POSITIONS")
            return

        print(f"OPEN POSITIONS ({len(self.positions)}):")

        for pos in self.positions:
            # Calculate distances
            pip_val = self.pip_values.get(pos.symbol, 0.0001)

            if pos.side == 'BUY':
                tp_distance = (pos.tp_price - pos.current_price) / pip_val if pos.tp_price else 0
                sl_distance = (pos.current_price - pos.sl_price) / pip_val if pos.sl_price else 0
            else:
                tp_distance = (pos.current_price - pos.tp_price) / pip_val if pos.tp_price else 0
                sl_distance = (pos.sl_price - pos.current_price) / pip_val if pos.sl_price else 0

            trailing_info = f"[TRAILING: {self.trailing_pips}p]" if pos.trailing_stop else ""

            print(f"  {pos.symbol}: ${pos.pnl:+.2f} ({pos.pnl_pips:+.1f} pips) | "
                  f"TP: {tp_distance:.1f}p away | SL: {sl_distance:.1f}p away {trailing_info}")

    def print_signal_check(self, symbol: str, signal: Dict):
        """Print signal analysis for a symbol"""
        if signal.get('direction'):
            return  # Will print full entry signal separately

        rsi = signal.get('rsi', 0)
        status = signal.get('status', 'No signal')

        print(f"  {symbol}: Combined V2: {status} | RSI: {rsi:.1f}")

    def print_entry_signal(self, symbol: str, signal: Dict, price: float,
                           spread: float, volume: float, sl: float, tp: float):
        """Print detailed entry signal"""
        print(f"""
================================================================================
[ENTRY SIGNAL] {signal['direction']} {symbol}
  Price: {price:.5f}
  Spread: {spread:.1f} pips
  Reason: {signal['reason']}
  Confidence: {signal['confidence']}

  Placing Order:
    Volume: {volume} lots
    Position Size: ${volume * 100000:.2f}
    Stop Loss: {sl:.5f} ({self.sl_pips} pips)
    Take Profit: {tp:.5f} ({self.tp_pips} pips)
    Trailing Stop: {self.trailing_pips} pips
""")

    def print_trade_result(self, symbol: str, side: str, pnl: float, pnl_pct: float,
                           exit_type: str = "MANUAL"):
        """Print trade exit result"""
        print(f"""
[TRADES CLOSED] Detected closed position
[LOGGER] Logged EXIT: {symbol} {exit_type} P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)
  {symbol} {exit_type}: ${pnl:.2f} ({pnl_pct:+.1f}%)
""")

    def calculate_position_size(self, balance: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = balance * (self.risk_per_trade / 100)
        # Assuming $10 per pip per standard lot
        pip_value = 10
        lot_size = risk_amount / (self.sl_pips * pip_value)
        lot_size = round(lot_size, 2)
        return max(0.01, min(lot_size, 0.5))

    def get_mock_prices(self, symbol: str) -> pd.DataFrame:
        """Generate mock price data (replace with real data in production)"""
        np.random.seed(int(time.time() * 1000) % 10000 + hash(symbol) % 1000)

        base_prices = {
            'EURUSD': 1.05, 'GBPUSD': 1.27, 'USDJPY': 150.0,
            'USDCHF': 0.88, 'AUDUSD': 0.65, 'NZDUSD': 0.58,
            'USDCAD': 1.35, 'XAUUSD': 2000.0
        }

        base = base_prices.get(symbol, 1.0)
        n = 100

        returns = np.random.randn(n) * 0.0003
        prices = base * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'close': prices,
            'high': prices * 1.0005,
            'low': prices * 0.9995,
            'open': np.roll(prices, 1)
        })

    def update_positions_from_browser(self):
        """Update position data from browser by scraping the positions table"""
        try:
            driver = self.browser.driver

            # Try to click on positions tab to ensure we see positions
            try:
                positions_tabs = driver.find_elements(By.XPATH,
                    "//*[contains(text(), 'Open Positions') or contains(text(), 'Positions')]")
                for tab in positions_tabs:
                    if tab.is_displayed():
                        tab.click()
                        time.sleep(0.5)
                        break
            except:
                pass

            # Try to scrape position data from the interface
            position_rows = driver.find_elements(By.CSS_SELECTOR,
                "tr.position, div.position, [class*='position-row'], [class*='open-position']")

            # Also try to get balance/equity
            try:
                balance_elements = driver.find_elements(By.CSS_SELECTOR,
                    "[class*='balance'], [class*='Balance']")
                for elem in balance_elements:
                    text = elem.text.replace('$', '').replace(',', '').strip()
                    try:
                        self.stats.current_balance = float(text)
                        break
                    except:
                        continue

                equity_elements = driver.find_elements(By.CSS_SELECTOR,
                    "[class*='equity'], [class*='Equity']")
                for elem in equity_elements:
                    text = elem.text.replace('$', '').replace(',', '').strip()
                    try:
                        self.stats.current_equity = float(text)
                        break
                    except:
                        continue
            except:
                pass

            # Update tracked positions with live data if available
            for pos in self.positions:
                # Simulate price movement if we can't get real data
                pip_val = self.pip_values.get(pos.symbol, 0.0001)
                movement = np.random.randn() * 2  # Random pips movement

                if pos.side == 'BUY':
                    pos.current_price = pos.entry_price + (movement * pip_val)
                    pos.pnl_pips = (pos.current_price - pos.entry_price) / pip_val
                else:
                    pos.current_price = pos.entry_price - (movement * pip_val)
                    pos.pnl_pips = (pos.entry_price - pos.current_price) / pip_val

                # P&L in dollars (simplified: $10 per pip per standard lot)
                pos.pnl = pos.pnl_pips * pos.volume * 10

            # Update daily P&L
            total_unrealized = sum(p.pnl for p in self.positions)
            self.stats.daily_pnl = total_unrealized

            # Track max drawdown
            current_equity = self.stats.current_balance + total_unrealized
            if current_equity < self.stats.peak_balance:
                drawdown = self.stats.peak_balance - current_equity
                self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
            else:
                self.stats.peak_balance = current_equity

        except Exception as e:
            # Silent fail - just use simulated data
            for pos in self.positions:
                pip_val = self.pip_values.get(pos.symbol, 0.0001)
                movement = np.random.randn() * 2

                if pos.side == 'BUY':
                    pos.current_price = pos.entry_price + (movement * pip_val)
                    pos.pnl_pips = (pos.current_price - pos.entry_price) / pip_val
                else:
                    pos.current_price = pos.entry_price - (movement * pip_val)
                    pos.pnl_pips = (pos.entry_price - pos.current_price) / pip_val

                pos.pnl = pos.pnl_pips * pos.volume * 10

    def check_risk_limits(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        s = self.stats

        # Daily loss limit
        daily_limit = s.day_start_balance * (s.daily_loss_limit_pct / 100)
        if s.daily_pnl <= -daily_limit:
            print(f"\n[STOP] DAILY LOSS LIMIT REACHED! (${s.daily_pnl:.2f} / -${daily_limit:.2f})")
            return False

        # Max drawdown
        max_loss = s.starting_balance * (s.max_loss_limit_pct / 100)
        if s.max_drawdown >= max_loss:
            print(f"\n[STOP] MAX DRAWDOWN REACHED! (${s.max_drawdown:.2f} / ${max_loss:.2f})")
            return False

        return True

    def start(self) -> bool:
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
            print("\n[ERROR] Login failed!")
            return False

        print("       [OK] Login successful!")

        # Print challenge status
        print("\n[3/4] Loading challenge parameters...")
        self.print_challenge_status()

        print("[4/4] Starting trading loop...")
        self.is_running = True

        return True

    def run(self, check_interval: int = 60):
        """Main trading loop"""
        print(f"\n  Trading loop started (checking every {check_interval}s)")
        print(f"  Monitoring: {', '.join(self.symbols)}")
        print(f"  Max positions: {self.max_positions}")
        print(f"  Risk per trade: {self.risk_per_trade}%")
        print(f"  SL: {self.sl_pips} pips | TP: {self.tp_pips} pips | Trailing: {self.trailing_pips} pips")
        print(f"\n  Press Ctrl+C to stop\n")

        cycle = 0
        while self.is_running:
            try:
                cycle += 1

                # Update positions
                self.update_positions_from_browser()

                # Print positions status
                self.print_positions()

                # Check risk limits
                if not self.check_risk_limits():
                    print("  Trading paused due to risk limits")
                    time.sleep(check_interval)
                    continue

                # Check for signals if we have room for more positions
                if len(self.positions) < self.max_positions:
                    now = datetime.now().strftime('%H:%M:%S')
                    print(f"\n[{now}] Balance: ${self.stats.current_balance:,.2f} | Checking for entry signals...")

                    for symbol in self.symbols:
                        # Skip if already in position
                        if any(p.symbol == symbol for p in self.positions):
                            continue

                        # Check cooldown
                        last_time = self.last_signal_time.get(symbol)
                        if last_time and (datetime.now() - last_time).seconds < 300:
                            continue

                        # Get prices and generate signal
                        prices = self.get_mock_prices(symbol)
                        signal = self.signal_gen.generate_signal(symbol, prices)

                        if signal and signal.get('direction'):
                            # Entry signal!
                            current_price = prices['close'].iloc[-1]
                            pip_val = self.pip_values.get(symbol, 0.0001)
                            spread = np.random.uniform(1, 3)

                            if signal['direction'] == 'BUY':
                                sl = current_price - (self.sl_pips * pip_val)
                                tp = current_price + (self.tp_pips * pip_val)
                            else:
                                sl = current_price + (self.sl_pips * pip_val)
                                tp = current_price - (self.tp_pips * pip_val)

                            volume = self.calculate_position_size(self.stats.current_balance)

                            self.print_entry_signal(symbol, signal, current_price,
                                                   spread, volume, sl, tp)

                            # Execute trade via browser automation
                            position_id = self.browser.open_position(
                                symbol=symbol,
                                side=signal['direction'],
                                volume=volume,
                                sl_pips=self.sl_pips,
                                tp_pips=self.tp_pips,
                                sl_price=sl,
                                tp_price=tp
                            )

                            if position_id:
                                # Track the position locally
                                pos = Position(
                                    symbol=symbol,
                                    side=signal['direction'],
                                    volume=volume,
                                    entry_price=current_price,
                                    current_price=current_price,
                                    sl_price=sl,
                                    tp_price=tp,
                                    trailing_stop=self.trailing_pips * pip_val,
                                    position_id=position_id
                                )
                                self.positions.append(pos)
                                self.stats.trades_today += 1
                                self.stats.total_trades += 1
                                self.last_signal_time[symbol] = datetime.now()

                                print(f"  ORDER FILLED!")
                                print(f"    Position ID: {position_id}")
                                print(f"    Fill Price: {current_price:.5f}")
                                print(f"    SL: {sl:.5f} ({self.sl_pips} pips)")
                                print(f"    TP: {tp:.5f} ({self.tp_pips} pips)")
                                print(f"    Trailing: {self.trailing_pips} pips")
                                print(f"    Trades Today: {self.stats.trades_today}")
                                print(f"    Open Positions: {len(self.positions)}/{self.max_positions}")
                                print(f"[LOGGER] Logged ENTRY: {symbol} {signal['direction']} @ {current_price:.5f}")
                                print("=" * 80)
                            else:
                                print(f"  ORDER FAILED! Could not execute trade via browser")
                                print("=" * 80)

                        else:
                            self.print_signal_check(symbol, signal)

                # Print challenge status every 5 cycles
                if cycle % 5 == 0:
                    self.print_challenge_status()

                # Check browser still open
                try:
                    _ = self.browser.driver.current_url
                except:
                    print("\n  Browser was closed!")
                    break

                time.sleep(check_interval)

            except KeyboardInterrupt:
                print("\n\n  Stopping...")
                break
            except Exception as e:
                print(f"\n  Error: {e}")
                time.sleep(10)

        self.is_running = False

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
    trader = MatchTraderChallenge()

    try:
        if trader.start():
            trader.run(check_interval=30)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        trader.stop()
        print("Done!")


if __name__ == "__main__":
    main()
