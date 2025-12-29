"""
DCA (Dollar Cost Averaging) Trading Engine
==========================================

Production-ready trading engine implementing:
- Momentum-based entry signals
- Progressive DCA position sizing
- Proper risk management
- Trade logging and analytics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import logging

from .dca_config import DCAConfig, load_dca_config


@dataclass
class Position:
    """Represents an open trading position with DCA capability."""
    symbol: str
    entry_time: datetime
    entry_bar: int
    entry_price: float
    quantity: float
    avg_entry_price: float
    stop_loss: float
    take_profit: float
    dca_stages_used: int = 0
    dca_prices: List[float] = field(default_factory=list)
    dca_quantities: List[float] = field(default_factory=list)
    total_cost: float = 0.0

    def add_dca(self, price: float, quantity: float):
        """Add a DCA entry to the position."""
        self.dca_stages_used += 1
        self.dca_prices.append(price)
        self.dca_quantities.append(quantity)

        # Update average entry price
        old_cost = self.avg_entry_price * self.quantity
        new_cost = price * quantity
        self.total_cost = old_cost + new_cost
        self.quantity += quantity
        self.avg_entry_price = self.total_cost / self.quantity

        # Update take profit based on new average entry
        # (Stop loss stays at original level)

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return self.quantity * (current_price - self.avg_entry_price)

    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        return (current_price / self.avg_entry_price - 1) * 100


@dataclass
class TradeResult:
    """Result of a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    avg_entry_price: float
    exit_price: float
    quantity: float
    dca_stages_used: int
    pnl: float
    pnl_pct: float
    outcome: str  # 'TP', 'SL', 'TIMEOUT'
    hold_bars: int
    commission: float


class DCAEngine:
    """
    DCA Trading Engine with momentum-based signals.

    Features:
    - Calculates momentum and RSI signals
    - Manages positions with progressive DCA
    - Tracks P&L and generates analytics
    """

    def __init__(self, config: Optional[DCAConfig] = None):
        """Initialize the DCA engine."""
        self.config = config or load_dca_config()
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeResult] = []
        self.last_trade_bar: Dict[str, int] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup trade logging."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('DCAEngine')
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_dir / f'dca_trades_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on momentum and RSI.

        All indicators are SHIFTED by 1 bar to prevent look-ahead bias.
        """
        df = df.copy()

        # Momentum (5-bar return, then shifted to avoid look-ahead)
        df['momentum'] = df['close'].pct_change(self.config.momentum_period).shift(1) * 100

        # RSI (compute fully, THEN shift once - matches profitable test)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).shift(1)  # Shift AFTER computing

        # Trend filter (SMA, shifted)
        if self.config.use_trend_filter:
            df['sma'] = df['close'].rolling(self.config.trend_sma_period).mean().shift(1)
            df['above_sma'] = df['close'].shift(1) > df['sma']
        else:
            df['above_sma'] = True

        # Signal conditions - MATCHES PROFITABLE TEST
        # Profitable signal: momentum > 0.2% AND RSI < 65
        momentum_ok = df['momentum'] > self.config.momentum_threshold
        rsi_ok = df['rsi'] < self.config.rsi_max_for_entry  # ONLY check max, no min filter

        df['signal'] = (momentum_ok & rsi_ok).astype(int)

        return df

    def should_enter(self, df: pd.DataFrame, bar_idx: int, symbol: str) -> bool:
        """Check if we should enter a new position."""
        # Already have position in this symbol
        if symbol in self.positions:
            return False

        # Max positions reached
        if len(self.positions) >= self.config.max_open_trades:
            return False

        # Cooldown check
        last_bar = self.last_trade_bar.get(symbol, -self.config.cooldown_bars - 1)
        if (bar_idx - last_bar) < self.config.cooldown_bars:
            return False

        # Check signal
        if df['signal'].iloc[bar_idx] == 1:
            return True

        return False

    def enter_position(self, df: pd.DataFrame, bar_idx: int, symbol: str) -> Optional[Position]:
        """Enter a new position with initial sizing."""
        entry_price = df['close'].iloc[bar_idx]
        entry_time = df['timestamp'].iloc[bar_idx] if 'timestamp' in df.columns else datetime.now()

        # Calculate position size (initial 25% of risk)
        quantity = self.config.get_entry_size(self.capital, entry_price)

        # Calculate TP/SL
        stop_loss = entry_price * (1 - self.config.stop_loss_pct)
        take_profit = entry_price * (1 + self.config.take_profit_pct)

        # Calculate DCA levels
        dca_levels = self.config.get_dca_levels(entry_price)

        position = Position(
            symbol=symbol,
            entry_time=entry_time,
            entry_bar=bar_idx,
            entry_price=entry_price,
            quantity=quantity,
            avg_entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            dca_prices=[],
            dca_quantities=[],
            total_cost=entry_price * quantity
        )

        self.positions[symbol] = position
        self.last_trade_bar[symbol] = bar_idx

        if self.config.verbose:
            self.logger.info(f"ENTER {symbol}: Price=${entry_price:.2f}, Qty={quantity:.6f}, "
                            f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}")

        return position

    def check_dca(self, position: Position, current_low: float) -> Tuple[bool, float]:
        """
        Check if DCA should be triggered.

        Returns (should_dca, dca_price)
        """
        if not self.config.use_dca:
            return False, 0

        if position.dca_stages_used >= self.config.max_dca_stages:
            return False, 0

        # Calculate next DCA level
        next_dca_idx = position.dca_stages_used
        dca_price = position.entry_price * (1 - self.config.dca_spacing_pct * (next_dca_idx + 1))

        # Check if price reached DCA level
        if current_low <= dca_price:
            return True, dca_price

        return False, 0

    def execute_dca(self, position: Position, dca_price: float) -> float:
        """Execute a DCA entry."""
        # Get the multiplier for this DCA stage
        stage_idx = position.dca_stages_used + 1  # +1 because entry is index 0
        if stage_idx >= len(self.config.dca_multipliers):
            return 0

        multiplier = self.config.dca_multipliers[stage_idx]
        base_risk = self.capital * self.config.risk_per_trade_pct
        dca_risk = base_risk * multiplier
        position_value = dca_risk / self.config.stop_loss_pct
        dca_quantity = position_value / dca_price

        position.add_dca(dca_price, dca_quantity)

        # Update take profit based on new average entry
        position.take_profit = position.avg_entry_price * (1 + self.config.take_profit_pct)

        if self.config.verbose:
            self.logger.info(f"DCA {position.symbol} Stage {position.dca_stages_used}: "
                            f"Price=${dca_price:.2f}, Qty={dca_quantity:.6f}, "
                            f"New Avg=${position.avg_entry_price:.2f}")

        return dca_quantity

    def check_exit(self, position: Position, bar_high: float, bar_low: float,
                   bar_close: float, bar_idx: int) -> Tuple[bool, str, float]:
        """
        Check if position should be exited.

        Returns (should_exit, outcome, exit_price)
        """
        # Check TP (based on current average entry)
        effective_tp = position.avg_entry_price * (1 + self.config.take_profit_pct)
        if bar_high >= effective_tp:
            return True, 'TP', effective_tp

        # Check SL (stays at original level)
        if bar_low <= position.stop_loss:
            return True, 'SL', position.stop_loss

        # Check timeout
        hold_bars = bar_idx - position.entry_bar
        if hold_bars >= self.config.max_hold_bars:
            return True, 'TIMEOUT', bar_close

        return False, '', 0

    def exit_position(self, symbol: str, exit_price: float, outcome: str,
                      bar_idx: int, exit_time: datetime) -> TradeResult:
        """Exit a position and record the trade."""
        position = self.positions.pop(symbol)

        # Calculate P&L
        pnl_gross = position.quantity * (exit_price - position.avg_entry_price)
        commission = position.total_cost * self.config.commission_pct * 2  # Entry + exit
        pnl_net = pnl_gross - commission
        pnl_pct = (pnl_net / self.capital) * 100

        # Update capital
        self.capital += pnl_net

        result = TradeResult(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            avg_entry_price=position.avg_entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            dca_stages_used=position.dca_stages_used,
            pnl=pnl_net,
            pnl_pct=pnl_pct,
            outcome=outcome,
            hold_bars=bar_idx - position.entry_bar,
            commission=commission
        )

        self.trade_history.append(result)
        self.last_trade_bar[symbol] = bar_idx

        if self.config.verbose:
            self.logger.info(f"EXIT {symbol}: {outcome} @ ${exit_price:.2f}, "
                            f"P&L=${pnl_net:.2f} ({pnl_pct:.2f}%), "
                            f"DCA stages={position.dca_stages_used}")

        return result

    def run_backtest(self, df: pd.DataFrame, symbol: str = 'BTCUSD') -> Dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLC data
            symbol: Trading symbol

        Returns:
            Dictionary with backtest results
        """
        # Calculate signals
        df = self.calculate_signals(df)

        n = len(df)
        start_capital = self.capital

        for i in range(100, n - 50):
            bar_high = df['high'].iloc[i]
            bar_low = df['low'].iloc[i]
            bar_close = df['close'].iloc[i]
            bar_time = df['timestamp'].iloc[i] if 'timestamp' in df.columns else datetime.now()

            # Check existing positions
            if symbol in self.positions:
                position = self.positions[symbol]

                # Check for DCA trigger
                should_dca, dca_price = self.check_dca(position, bar_low)
                if should_dca:
                    self.execute_dca(position, dca_price)

                # Check for exit
                should_exit, outcome, exit_price = self.check_exit(
                    position, bar_high, bar_low, bar_close, i
                )
                if should_exit:
                    self.exit_position(symbol, exit_price, outcome, i, bar_time)

            # Check for new entry
            elif self.should_enter(df, i, symbol):
                self.enter_position(df, i, symbol)

        # Close any remaining positions
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_price = df['close'].iloc[-1]
            exit_time = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
            self.exit_position(symbol, exit_price, 'END', n - 1, exit_time)

        # Calculate results
        return self.get_results(start_capital)

    def get_results(self, start_capital: float = None) -> Dict:
        """Get trading results summary."""
        if start_capital is None:
            start_capital = self.config.initial_capital

        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'return_pct': 0,
                'final_capital': self.capital
            }

        trades_df = pd.DataFrame([{
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'outcome': t.outcome,
            'dca_stages': t.dca_stages_used,
            'hold_bars': t.hold_bars
        } for t in self.trade_history])

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        total_win = wins['pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.0001

        results = {
            'total_trades': len(trades_df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades_df),
            'total_pnl': trades_df['pnl'].sum(),
            'return_pct': (self.capital / start_capital - 1) * 100,
            'final_capital': self.capital,
            'profit_factor': total_win / total_loss,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'avg_hold_bars': trades_df['hold_bars'].mean(),
            'avg_dca_stages': trades_df['dca_stages'].mean(),
            'outcomes': trades_df['outcome'].value_counts().to_dict()
        }

        return results

    def print_results(self):
        """Print formatted results."""
        results = self.get_results()

        print("\n" + "=" * 60)
        print("DCA TRADING RESULTS")
        print("=" * 60)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Wins: {results['wins']} | Losses: {results['losses']}")
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"")
        print(f"Starting Capital: ${self.config.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print(f"Return: {results['return_pct']:+.1f}%")
        print(f"")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Avg Win: ${results['avg_win']:.2f}")
        print(f"Avg Loss: ${results['avg_loss']:.2f}")
        print(f"")
        print(f"Avg Hold: {results['avg_hold_bars']:.0f} bars")
        print(f"Avg DCA Stages: {results['avg_dca_stages']:.1f}")
        print(f"")
        print("Outcomes:")
        for outcome, count in results['outcomes'].items():
            pct = count / results['total_trades'] * 100
            print(f"  {outcome}: {count} ({pct:.1f}%)")

    def save_trades(self, filepath: str = None):
        """Save trade history to CSV."""
        if not self.trade_history:
            return

        if filepath is None:
            filepath = Path(self.config.log_dir) / f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'avg_entry_price': t.avg_entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'dca_stages': t.dca_stages_used,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'outcome': t.outcome,
            'hold_bars': t.hold_bars,
            'commission': t.commission
        } for t in self.trade_history])

        trades_df.to_csv(filepath, index=False)
        print(f"Trades saved to {filepath}")


if __name__ == "__main__":
    # Test the engine
    print("Testing DCA Engine...")

    config = load_dca_config()
    print(config)

    engine = DCAEngine(config)

    # Load test data
    data_file = Path(config.data_dir) / "BTCUSD_1m.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        df = df.tail(50000).reset_index(drop=True)

        print(f"\nRunning backtest on {len(df):,} bars...")
        results = engine.run_backtest(df, 'BTCUSD')
        engine.print_results()
        engine.save_trades()
    else:
        print(f"Data file not found: {data_file}")
