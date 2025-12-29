"""
Crypto Paper Trading Engine
===========================
Simulated trading engine for crypto margin trading.
Uses Binance for real-time price data, simulates trades locally.

Features:
- Real-time price monitoring from Binance
- Simulated order execution
- Automatic TP/SL management
- Session-based loss cooldown
- Full trade logging
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.binance_data_client import BinanceDataClient
from strategies.crypto_margin_strategy import calculate_indicators, get_signal
from utils.crypto_trade_logger import CryptoTradeLogger


class CryptoPaperTradingEngine:
    """Paper trading engine for crypto margin trading"""

    def __init__(self, config):
        """
        Initialize paper trading engine.

        Args:
            config: Configuration module (crypto_paper_config or crypto_live_config)
        """
        self.config = config

        # Initialize components
        self.binance = BinanceDataClient()
        self.logger = CryptoTradeLogger(
            log_dir=config.LOG_DIR,
            account_type=config.ACCOUNT_TYPE
        )

        # Paper account state
        self.balance = config.INITIAL_BALANCE
        self.starting_balance = config.INITIAL_BALANCE
        self.equity = self.balance
        self.open_positions: Dict[str, Dict] = {}  # pair -> position
        self.trade_counter = 0

        # Cooldown tracking (pair -> unlock_time)
        self.pair_cooldown: Dict[str, datetime] = {}
        self.last_trade_time: Dict[str, datetime] = {}

        # Session stats
        self.session_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
        }

        print(f"\n{'='*70}")
        print("CRYPTO PAPER TRADING ENGINE INITIALIZED")
        print(f"{'='*70}")
        print(f"Starting Balance: ${self.balance:,.2f}")
        print(f"Max Positions: {config.MAX_CONCURRENT_POSITIONS}")
        print(f"Trading Pairs: {len(config.TRADING_PAIRS)}")
        print(f"{'='*70}\n")

    def get_current_prices(self, pairs: List[str]) -> Dict[str, float]:
        """Get current prices for all trading pairs."""
        prices = {}
        for pair in pairs:
            price_data = self.binance.get_current_price(pair)
            if price_data:
                prices[pair] = price_data['price']
        return prices

    def get_candle_data(self, pair: str, count: int = 100) -> Optional[pd.DataFrame]:
        """
        Get recent candle data for analysis.

        Args:
            pair: Trading pair
            count: Number of candles

        Returns:
            DataFrame with OHLCV data and indicators
        """
        candles = self.binance.get_klines(pair, interval='1m', limit=count)

        if not candles:
            return None

        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

        # Calculate indicators
        df = calculate_indicators(df)

        return df

    def check_open_positions(self, prices: Dict[str, float]):
        """
        Check and update open positions for TP/SL hits.

        Args:
            prices: Current prices dict
        """
        positions_to_close = []

        for pair, position in self.open_positions.items():
            if pair not in prices:
                continue

            current_price = prices[pair]
            entry_price = position['entry_price']
            direction = position['direction']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']

            # Calculate current P&L
            volume = position['volume']
            leverage = position['leverage']

            if direction == 'BUY':
                pnl = (current_price - entry_price) * volume * leverage
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage

                # Check TP/SL
                if current_price >= take_profit:
                    positions_to_close.append((pair, take_profit, 'TP'))
                elif current_price <= stop_loss:
                    positions_to_close.append((pair, stop_loss, 'SL'))
            else:  # SELL
                pnl = (entry_price - current_price) * volume * leverage
                pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage

                # Check TP/SL
                if current_price <= take_profit:
                    positions_to_close.append((pair, take_profit, 'TP'))
                elif current_price >= stop_loss:
                    positions_to_close.append((pair, stop_loss, 'SL'))

            # Update position's current P&L
            position['current_price'] = current_price
            position['current_pnl'] = pnl
            position['current_pnl_pct'] = pnl_pct

        # Close positions that hit TP/SL
        for pair, exit_price, reason in positions_to_close:
            self.close_position(pair, exit_price, reason)

    def open_position(
        self,
        pair: str,
        direction: str,
        price: float,
        signal_reason: str
    ) -> bool:
        """
        Open a new position.

        Args:
            pair: Trading pair
            direction: 'BUY' or 'SELL'
            price: Entry price
            signal_reason: Reason for entry

        Returns:
            True if position opened successfully
        """
        # Check if already have position in this pair
        if pair in self.open_positions:
            return False

        # Check max positions
        if len(self.open_positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return False

        # Get pair settings
        settings = self.config.get_pair_settings(pair)
        leverage = settings['leverage']
        tp_pct = settings['tp_pct']
        sl_pct = settings['sl_pct']

        # Calculate position size
        volume = self.config.calculate_position_size(
            account_balance=self.balance,
            entry_price=price,
            sl_pct=sl_pct,
            leverage=leverage
        )

        # Calculate TP/SL prices
        take_profit, stop_loss = self.config.calculate_tp_sl(pair, price, direction)

        # Generate trade ID
        self.trade_counter += 1
        trade_id = f"PAPER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.trade_counter:04d}"

        # Create position
        position = {
            'trade_id': trade_id,
            'pair': pair,
            'direction': direction,
            'entry_price': price,
            'volume': volume,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': settings['strategy'],
            'entry_time': datetime.utcnow(),
            'current_price': price,
            'current_pnl': 0.0,
            'current_pnl_pct': 0.0,
        }

        self.open_positions[pair] = position

        # Log entry
        self.logger.log_trade_entry(
            trade_id=trade_id,
            pair=pair,
            direction=direction,
            entry_price=price,
            volume=volume,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=settings['strategy'],
            signal_reason=signal_reason,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )

        # Update last trade time
        self.last_trade_time[pair] = datetime.utcnow()

        return True

    def close_position(self, pair: str, exit_price: float, reason: str) -> Optional[float]:
        """
        Close an open position.

        Args:
            pair: Trading pair
            exit_price: Exit price
            reason: Exit reason ('TP', 'SL', 'MANUAL')

        Returns:
            Realized P&L or None if no position
        """
        if pair not in self.open_positions:
            return None

        position = self.open_positions.pop(pair)
        trade_id = position['trade_id']
        entry_price = position['entry_price']
        volume = position['volume']
        leverage = position['leverage']
        direction = position['direction']

        # Calculate P&L
        if direction == 'BUY':
            pnl = (exit_price - entry_price) * volume * leverage
        else:
            pnl = (entry_price - exit_price) * volume * leverage

        # Update balance
        self.balance += pnl

        # Update session stats
        self.session_stats['total_trades'] += 1
        self.session_stats['total_pnl'] += pnl
        if pnl > 0:
            self.session_stats['wins'] += 1
        elif pnl < 0:
            self.session_stats['losses'] += 1

            # Apply cooldown on loss
            cooldown_mins = self.config.get_cooldown_minutes(pair)
            if cooldown_mins > 0:
                self.pair_cooldown[pair] = datetime.utcnow() + timedelta(minutes=cooldown_mins)
                print(f"  [Cooldown] {pair} blocked for {cooldown_mins} minutes after loss")

        # Log exit
        self.logger.log_trade_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=reason,
            pnl=pnl,
        )

        return pnl

    def is_pair_in_cooldown(self, pair: str) -> Tuple[bool, str]:
        """
        Check if pair is in cooldown.

        Returns:
            Tuple of (is_blocked, reason_string)
        """
        # Check loss cooldown
        if pair in self.pair_cooldown:
            unlock_time = self.pair_cooldown[pair]
            if datetime.utcnow() < unlock_time:
                remaining = (unlock_time - datetime.utcnow()).total_seconds() / 60
                return True, f"Loss cooldown ({remaining:.1f}m remaining)"
            else:
                del self.pair_cooldown[pair]

        # Check trade cooldown
        cooldown_mins = self.config.get_cooldown_minutes(pair)
        if cooldown_mins > 0 and pair in self.last_trade_time:
            min_next_trade = self.last_trade_time[pair] + timedelta(minutes=cooldown_mins)
            if datetime.utcnow() < min_next_trade:
                remaining = (min_next_trade - datetime.utcnow()).total_seconds() / 60
                return True, f"Trade cooldown ({remaining:.1f}m remaining)"

        return False, ""

    def check_for_signals(self, pairs: List[str]) -> List[Tuple[str, str, str, float]]:
        """
        Check all pairs for entry signals.

        Args:
            pairs: List of trading pairs

        Returns:
            List of (pair, direction, reason, price) tuples
        """
        signals = []

        for pair in pairs:
            # Skip if already have position
            if pair in self.open_positions:
                continue

            # Skip if in cooldown
            in_cooldown, cooldown_reason = self.is_pair_in_cooldown(pair)
            if in_cooldown:
                continue

            # Check hour filter
            hour_utc = datetime.utcnow().hour
            if not self.config.is_allowed_hour(pair, hour_utc):
                continue

            # Get candle data
            df = self.get_candle_data(pair, count=100)
            if df is None or len(df) < 50:
                continue

            # Get signal
            signal, reason = get_signal(pair, df, self.config)

            if signal:
                current_price = float(df.iloc[-1]['close'])
                signals.append((pair, signal, reason, current_price))

        return signals

    def display_status(self, prices: Dict[str, float]):
        """Display current trading status."""
        now = datetime.utcnow()

        print(f"\n{'='*70}")
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')} UTC] PAPER TRADING STATUS")
        print(f"{'='*70}")
        print(f"Balance: ${self.balance:,.2f} | Starting: ${self.starting_balance:,.2f} | P&L: ${self.balance - self.starting_balance:+,.2f}")

        # Session stats
        wins = self.session_stats['wins']
        losses = self.session_stats['losses']
        total = self.session_stats['total_trades']
        win_rate = (wins / total * 100) if total > 0 else 0
        print(f"Session: {wins}W/{losses}L ({win_rate:.1f}%) | Total P&L: ${self.session_stats['total_pnl']:+,.2f}")

        # Open positions
        if self.open_positions:
            print(f"\n{'OPEN POSITIONS':^70}")
            print(f"{'-'*70}")
            print(f"{'PAIR':<12} {'DIR':<6} {'ENTRY':<12} {'CURRENT':<12} {'P&L':>10} {'P&L%':>8}")
            print(f"{'-'*70}")

            total_unrealized = 0
            for pair, pos in self.open_positions.items():
                current = pos.get('current_price', pos['entry_price'])
                pnl = pos.get('current_pnl', 0)
                pnl_pct = pos.get('current_pnl_pct', 0)
                total_unrealized += pnl

                print(f"{pair:<12} {pos['direction']:<6} ${pos['entry_price']:<11,.2f} ${current:<11,.2f} ${pnl:>+9,.2f} {pnl_pct:>+7.2f}%")

            print(f"{'-'*70}")
            print(f"{'TOTAL UNREALIZED:':<44} ${total_unrealized:>+9,.2f}")
        else:
            print("\nNo open positions")

        # Cooldowns
        active_cooldowns = []
        for pair in self.config.TRADING_PAIRS:
            in_cooldown, reason = self.is_pair_in_cooldown(pair)
            if in_cooldown:
                active_cooldowns.append(f"{pair}: {reason}")

        if active_cooldowns:
            print(f"\nCooldowns: {', '.join(active_cooldowns)}")

        print(f"{'='*70}")

    def run(self, display_interval: int = 60, signal_check_interval: int = 60):
        """
        Main trading loop.

        Args:
            display_interval: Seconds between status displays
            signal_check_interval: Seconds between signal checks
        """
        print("\n" + "="*70)
        print("STARTING PAPER TRADING")
        print("="*70)
        print(f"Trading Pairs: {', '.join(self.config.TRADING_PAIRS)}")
        print(f"Press Ctrl+C to stop")
        print("="*70 + "\n")

        iteration = 0
        last_display = 0
        last_signal_check = 0

        try:
            while True:
                iteration += 1
                current_time = time.time()

                # Get current prices
                prices = self.get_current_prices(self.config.TRADING_PAIRS)

                if not prices:
                    print("[Warning] Could not get prices, retrying...")
                    time.sleep(5)
                    continue

                # Check open positions for TP/SL
                self.check_open_positions(prices)

                # Check for new signals periodically
                if current_time - last_signal_check >= signal_check_interval:
                    last_signal_check = current_time

                    signals = self.check_for_signals(self.config.TRADING_PAIRS)

                    for pair, direction, reason, price in signals:
                        print(f"\n[SIGNAL] {pair} {direction}: {reason}")
                        success = self.open_position(pair, direction, price, reason)
                        if success:
                            print(f"  -> Position opened @ ${price:,.2f}")

                # Display status periodically
                if current_time - last_display >= display_interval:
                    last_display = current_time
                    self.display_status(prices)

                # Sleep between iterations
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown procedure."""
        print("\n" + "="*70)
        print("PAPER TRADING SESSION ENDED")
        print("="*70)

        # Print final stats
        self.logger.print_session_stats()

        # Generate summary
        self.logger.generate_session_summary()
        self.logger.export_to_csv()

        # Close any open positions at market
        if self.open_positions:
            print("\nClosing open positions...")
            prices = self.get_current_prices(list(self.open_positions.keys()))
            for pair in list(self.open_positions.keys()):
                if pair in prices:
                    self.close_position(pair, prices[pair], 'SHUTDOWN')

        # Final balance
        print(f"\nFinal Balance: ${self.balance:,.2f}")
        print(f"Total P&L: ${self.balance - self.starting_balance:+,.2f}")
        print(f"Return: {((self.balance / self.starting_balance) - 1) * 100:+.2f}%")
        print("="*70)


if __name__ == "__main__":
    # Test with paper config
    from config import crypto_paper_config as config

    config.print_config_info()

    engine = CryptoPaperTradingEngine(config)
    engine.run()
