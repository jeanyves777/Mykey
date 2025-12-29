#!/usr/bin/env python3
"""
BTCC Futures Paper Trading
==========================
Run paper trading simulation for BTCC Futures.

Usage:
    python run_btcc_paper_trading.py
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BTCC_Futures.btcc_trading_engine import BTCCTradingEngine
from BTCC_Futures.btcc_config import BTCCConfig
from BTCC_Futures.btcc_strategies import get_strategy, OHLCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BTCCPaperTrader:
    """Paper trading runner for BTCC Futures"""

    def __init__(self):
        # Configure for paper trading
        self.config = BTCCConfig(
            TRADING_MODE="paper",
            PAPER_INITIAL_BALANCE=10000.0,
            PAPER_COMMISSION_PCT=0.06,
            MAX_POSITIONS=5,
            POSITION_SIZE_PCT=2.0,
            DEFAULT_LEVERAGE=20,
        )

        # Enable symbols
        self.config.SYMBOLS = {
            'BTCUSDT': {
                'enabled': True,
                'leverage': 20,
                'min_volume': 0.001,
                'max_volume': 10.0,
                'volume_step': 0.001,
                'digits': 2,
                'strategy': 'SCALPING_MOMENTUM',
                'tp_pct': 3.0,
                'sl_pct': 1.5,
            },
            'ETHUSDT': {
                'enabled': True,
                'leverage': 15,
                'min_volume': 0.01,
                'max_volume': 50.0,
                'volume_step': 0.01,
                'digits': 2,
                'strategy': 'RSI_REVERSAL',
                'tp_pct': 3.5,
                'sl_pct': 2.0,
            },
            'SOLUSDT': {
                'enabled': True,
                'leverage': 10,
                'min_volume': 0.1,
                'max_volume': 100.0,
                'volume_step': 0.1,
                'digits': 3,
                'strategy': 'EMA_CROSSOVER',
                'tp_pct': 4.0,
                'sl_pct': 2.0,
            },
        }

        self.engine = BTCCTradingEngine(self.config)
        self.running = False

        # Sample price data (in real use, fetch from exchange)
        self.mock_candles = {}

    def _generate_mock_candles(self, symbol: str, current_price: float) -> list:
        """Generate mock OHLCV data for testing."""
        candles = []
        price = current_price * 0.95  # Start 5% lower

        for i in range(100):
            # Random walk
            change = price * 0.002 * (1 if i % 3 == 0 else -0.5)
            price += change

            candles.append(OHLCV(
                timestamp=datetime.now(),
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=1000000 + i * 10000,
            ))

        # Set last candle to current price
        candles[-1].close = current_price
        candles[-1].high = max(candles[-1].high, current_price)
        candles[-1].low = min(candles[-1].low, current_price)

        return candles

    def _strategy_evaluator(self, symbol: str, config: BTCCConfig):
        """Evaluate strategy for a symbol."""
        from BTCC_Futures.btcc_trading_engine import TradeSignal

        symbol_config = config.get_symbol_config(symbol)
        if not symbol_config:
            return None

        # Get current price (mock)
        current_price = self._get_mock_price(symbol)

        # Generate or update candles
        if symbol not in self.mock_candles:
            self.mock_candles[symbol] = self._generate_mock_candles(symbol, current_price)

        candles = self.mock_candles[symbol]

        # Get strategy
        strategy_name = symbol_config.get('strategy', 'SCALPING_MOMENTUM')
        strategy = get_strategy(strategy_name, config.get_strategy_params(strategy_name))

        if not strategy:
            return None

        # Evaluate
        signal = strategy.evaluate(candles, current_price)

        if signal:
            # Calculate TP/SL
            tp_pct = symbol_config.get('tp_pct', 3.0) / 100
            sl_pct = symbol_config.get('sl_pct', 1.5) / 100

            if signal['direction'] == 1:  # Long
                take_profit = current_price * (1 + tp_pct)
                stop_loss = current_price * (1 - sl_pct)
            else:  # Short
                take_profit = current_price * (1 - tp_pct)
                stop_loss = current_price * (1 + sl_pct)

            return TradeSignal(
                symbol=symbol,
                direction=signal['direction'],
                strength=signal['strength'],
                strategy=strategy_name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        return None

    def _get_mock_price(self, symbol: str) -> float:
        """Get mock price for symbol."""
        base_prices = {
            'BTCUSDT': 100000.0,
            'ETHUSDT': 3500.0,
            'SOLUSDT': 200.0,
            'XRPUSDT': 2.0,
            'DOGEUSDT': 0.35,
        }

        import random
        base = base_prices.get(symbol, 100.0)
        return base * (1 + random.uniform(-0.02, 0.02))

    def run(self):
        """Run the paper trading session."""
        print("\n" + "=" * 70)
        print("BTCC FUTURES PAPER TRADING")
        print("=" * 70)
        print(f"\nInitial Balance: ${self.config.PAPER_INITIAL_BALANCE:,.2f}")
        print(f"Max Positions: {self.config.MAX_POSITIONS}")
        print(f"Position Size: {self.config.POSITION_SIZE_PCT}%")
        print(f"Default Leverage: {self.config.DEFAULT_LEVERAGE}x")
        print(f"\nEnabled Symbols: {', '.join(self.config.get_enabled_symbols())}")
        print("=" * 70)
        print("\nStarting paper trading... (Press Ctrl+C to stop)")
        print("-" * 70)

        # Set strategy evaluator
        self.engine.strategy_evaluator = self._strategy_evaluator

        # Start engine
        self.engine.start()
        self.running = True

        try:
            while self.running:
                # Update mock prices
                for symbol in self.config.get_enabled_symbols():
                    price = self._get_mock_price(symbol)
                    self.engine.update_price(symbol, price)

                # Print status
                status = self.engine.get_status()
                positions = self.engine.get_positions()

                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Balance: ${status['balance']:,.2f} | "
                      f"Positions: {status['open_positions']} | "
                      f"Daily P/L: ${status['daily_pnl']:+,.2f} | "
                      f"Trades: {status['trades_today']} | "
                      f"Win Rate: {status['win_rate']:.1f}%",
                      end='', flush=True)

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nStopping paper trading...")

        finally:
            self.engine.stop()
            self._print_summary()

    def _print_summary(self):
        """Print trading session summary."""
        status = self.engine.get_status()

        print("\n" + "=" * 70)
        print("PAPER TRADING SESSION SUMMARY")
        print("=" * 70)
        print(f"Final Balance:     ${status['balance']:,.2f}")
        print(f"Total P/L:         ${status['total_pnl']:+,.2f}")
        print(f"Daily P/L:         ${status['daily_pnl']:+,.2f}")
        print(f"Total Trades:      {status['winning_trades'] + status['losing_trades']}")
        print(f"Winning Trades:    {status['winning_trades']}")
        print(f"Losing Trades:     {status['losing_trades']}")
        print(f"Win Rate:          {status['win_rate']:.1f}%")
        print("=" * 70)


def main():
    """Main entry point."""
    trader = BTCCPaperTrader()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        trader.running = False

    signal.signal(signal.SIGINT, signal_handler)

    trader.run()


if __name__ == "__main__":
    main()
