"""
Backtesting Engine for High-Frequency Crypto Trading
======================================================

Realistic backtesting with:
- CENTRALIZED RISK MANAGEMENT (same as live trading)
- Slippage and commission modeling
- Order execution simulation
- Position and risk management
- Detailed trade logging and analytics
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import centralized risk management
try:
    from ..risk_management import RiskManager, RiskConfig, PositionManager, TradeExecutor, ExecutionMode
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    # SHORT removed - Spot crypto trading is BUY-only (no short selling)


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: int
    symbol: str
    side: OrderSide
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    signal_confidence: float = 0.0
    holding_bars: int = 0


@dataclass
class Position:
    """Current position state for BUY-only spot trading with DCA support."""
    symbol: str
    side: PositionStatus = PositionStatus.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0  # Average entry price (updated on DCA)
    entry_time: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    # DCA (Dollar Cost Averaging) tracking
    dca_count: int = 0           # Number of DCA entries
    total_cost: float = 0.0      # Total cost basis for avg price calculation
    dca_levels: List[float] = field(default_factory=list)  # Price levels where we averaged


@dataclass
class BacktestConfig:
    """Configuration for HF SCALPING backtest - tight TP/SL, quick exits."""
    initial_capital: float = 100000.0
    position_size_pct: float = 0.02
    max_position_size: float = 10000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage

    # HF SCALPING - Tight TP/SL for quick exits
    take_profit_pct: float = 0.004    # 0.4% TP
    stop_loss_pct: float = 0.003      # 0.3% SL
    trailing_stop_pct: float = 0.002  # 0.2% trailing

    # DCA DISABLED for scalping
    dca_enabled: bool = False
    max_dca_entries: int = 0
    dca_level_1_trigger_pct: float = 0.01
    dca_level_1_multiplier: float = 1.0
    dca_level_2_trigger_pct: float = 0.015
    dca_level_2_multiplier: float = 1.0
    dca_level_3_trigger_pct: float = 0.02
    dca_level_3_multiplier: float = 1.0
    dca_level_4_trigger_pct: float = 0.025
    dca_level_4_multiplier: float = 1.0
    dca_profit_target_pct: float = 0.01
    sl_after_last_dca_pct: float = 0.01

    # HF settings
    min_confidence: float = 0.55
    cooldown_bars: int = 1            # Minimal cooldown
    max_daily_trades: int = 500       # Allow many trades
    max_daily_loss_pct: float = 0.03  # 3% max daily loss


class BacktestEngine:
    """
    High-frequency crypto backtesting engine for SPOT TRADING (BUY-only).

    Features:
    - BUY-ONLY signals (no short selling for spot crypto)
    - DCA (Dollar Cost Averaging) for trades that go against us
    - Wide stop loss to allow averaging down
    - Exit in profit after averaging
    - CENTRALIZED RISK MANAGEMENT (same rules as live trading)
    - Realistic order execution with slippage
    - Commission modeling
    - Comprehensive performance metrics

    Strategy:
    - Enter on BUY signal with initial position
    - If price drops, average down with DCA at defined intervals
    - Exit when averaged position is in profit
    - Final stop loss only triggered if all DCA levels exhausted
    """

    def __init__(self, config: Optional[BacktestConfig] = None,
                 risk_config: 'RiskConfig' = None,
                 use_centralized_risk: bool = True):
        """
        Initialize the backtest engine.

        Args:
            config: BacktestConfig object (uses defaults if None)
            risk_config: RiskConfig for centralized risk management
            use_centralized_risk: Whether to use centralized RiskManager
        """
        self.config = config or BacktestConfig()
        self.use_centralized_risk = use_centralized_risk and HAS_RISK_MANAGEMENT

        # Initialize centralized risk management if available
        if self.use_centralized_risk:
            self.risk_config = risk_config or RiskConfig(
                base_position_size_pct=self.config.position_size_pct,
                max_position_value=self.config.max_position_size,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                trailing_stop_pct=self.config.trailing_stop_pct,
                max_daily_loss_pct=self.config.max_daily_loss_pct,
                max_daily_trades=self.config.max_daily_trades,
                min_confidence=self.config.min_confidence,
                cooldown_seconds=self.config.cooldown_bars * 60,
                # Dynamic DCA settings
                dca_enabled=self.config.dca_enabled,
                max_dca_entries=self.config.max_dca_entries,
                dca_profit_target_pct=self.config.dca_profit_target_pct
            )
            self.risk_manager = RiskManager(self.risk_config)
            self.position_manager = PositionManager(self.risk_manager)
        else:
            self.risk_manager = None
            self.position_manager = None

        self.reset()

    def reset(self):
        """Reset the backtest state."""
        self.capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.position = Position(symbol="", dca_levels=[])
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_pnl: List[float] = []
        self.trade_id_counter = 0
        self.current_bar = 0
        self.last_trade_bar = -self.config.cooldown_bars
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        self.current_date = None

        # DCA tracking
        self.dca_trades: List[dict] = []  # Track DCA entries for the current position

        # Reset centralized risk management
        if self.use_centralized_risk and self.risk_manager:
            self.risk_manager.reset()
            self.risk_manager.initialize(self.config.initial_capital)
            self.position_manager.reset()

    def run(self,
            data: pd.DataFrame,
            signals: np.ndarray,
            confidences: np.ndarray,
            symbol: str = "CRYPTO",
            verbose: bool = True) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with datetime index or column
            signals: Array of signals (1=buy, -1=sell, 0=hold)
            confidences: Array of confidence scores (0-1)
            symbol: Trading symbol
            verbose: Print progress

        Returns:
            Dictionary with backtest results
        """
        self.reset()
        self.position.symbol = symbol

        n_bars = len(data)

        if verbose:
            print("=" * 60)
            print(f"Running Backtest: {symbol}")
            print(f"Bars: {n_bars}")
            print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
            print("=" * 60)

        # Extract price data
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # Get datetime if available (convert to numpy array for consistent access)
        if 'datetime' in data.columns:
            datetimes = pd.to_datetime(data['datetime']).values
        elif isinstance(data.index, pd.DatetimeIndex):
            datetimes = data.index.values
        else:
            datetimes = pd.date_range(start='2024-01-01', periods=n_bars, freq='1min').values

        # Main backtest loop
        for i in range(n_bars):
            self.current_bar = i
            current_time = datetimes[i] if i < len(datetimes) else datetime.now()
            current_price = closes[i]
            current_high = highs[i]
            current_low = lows[i]

            # Check for new day
            current_day = current_time.date() if hasattr(current_time, 'date') else None
            if current_day != self.current_date:
                self.current_date = current_day
                self.daily_trade_count = 0
                self.daily_loss = 0.0

            # Update position P&L and check stops
            if self.position.side != PositionStatus.FLAT:
                self._update_position(current_price, current_high, current_low, current_time)

            # Check for DCA opportunity (if in position and price dropped)
            if self.position.side == PositionStatus.LONG and self.config.dca_enabled:
                self._check_dca_opportunity(current_price, current_time)

            # Check for exit conditions
            if self.position.side != PositionStatus.FLAT:
                exit_signal = self._check_exit_conditions(
                    current_price, current_high, current_low, current_time
                )
                if exit_signal:
                    self._close_position(current_price, current_time)

            # Check for entry signal - BUY ONLY for spot trading
            if self.position.side == PositionStatus.FLAT:
                signal = signals[i] if i < len(signals) else 0
                confidence = confidences[i] if i < len(confidences) else 0

                # SPOT TRADING: Only BUY signals allowed (ignore SELL signals)
                if signal == 1 and self._can_trade(signal, confidence):
                    self._open_position(
                        OrderSide.BUY, current_price, current_time, confidence
                    )
                # Note: signal == -1 (SELL) is ignored for spot trading

            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)

        # Close any remaining position
        if self.position.side != PositionStatus.FLAT:
            last_time = datetimes[-1] if len(datetimes) > 0 else datetime.now()
            self._close_position(closes[-1], last_time)

        # Calculate results
        results = self._calculate_results()

        if verbose:
            self._print_results(results)

        return results

    def _can_trade(self, signal: int, confidence: float) -> bool:
        """Check if we can take a trade."""
        if signal == 0:
            return False

        if confidence < self.config.min_confidence:
            return False

        if self.current_bar - self.last_trade_bar < self.config.cooldown_bars:
            return False

        if self.daily_trade_count >= self.config.max_daily_trades:
            return False

        if self.daily_loss >= self.config.max_daily_loss_pct * self.initial_capital:
            return False

        return True

    def _open_position(self,
                       side: OrderSide,
                       price: float,
                       time: datetime,
                       confidence: float):
        """Open a new position (BUY-only for spot trading)."""
        # Calculate position size
        position_value = min(
            self.capital * self.config.position_size_pct,
            self.config.max_position_size
        )

        # Apply slippage (always buying in spot)
        slippage = price * self.config.slippage_pct
        entry_price = price + slippage  # BUY only

        quantity = position_value / entry_price

        # Apply commission
        commission = position_value * self.config.commission_rate

        # Update position (BUY-only = LONG)
        self.position.side = PositionStatus.LONG
        self.position.quantity = quantity
        self.position.entry_price = entry_price
        self.position.entry_time = time
        self.position.highest_price = entry_price
        self.position.lowest_price = entry_price

        # DCA tracking - initialize for new position
        self.position.dca_count = 0
        self.position.total_cost = position_value
        self.position.dca_levels = [entry_price]
        self.dca_trades = []  # Reset DCA trades list

        # Create trade record
        self.trade_id_counter += 1
        trade = Trade(
            trade_id=self.trade_id_counter,
            symbol=self.position.symbol,
            side=side,
            entry_time=time,
            entry_price=entry_price,
            quantity=quantity,
            commission=commission,
            slippage=slippage * quantity,
            signal_confidence=confidence
        )
        self.trades.append(trade)

        # Update state
        self.capital -= commission
        self.last_trade_bar = self.current_bar
        self.daily_trade_count += 1

    def _update_position(self,
                         current_price: float,
                         high: float,
                         low: float,
                         time: datetime):
        """Update position state (LONG only for spot trading)."""
        if self.position.side == PositionStatus.LONG:
            # Calculate unrealized P&L based on average entry price
            self.position.unrealized_pnl = (current_price - self.position.entry_price) * self.position.quantity
            self.position.highest_price = max(self.position.highest_price, high)
            self.position.lowest_price = min(self.position.lowest_price, low)

    def _check_exit_conditions(self,
                                current_price: float,
                                high: float,
                                low: float,
                                time: datetime) -> bool:
        """
        HF SCALPING EXIT - Simple and fast.
        1. Take Profit hit
        2. Stop Loss hit
        3. Trailing stop (if in profit)
        """
        entry_price = self.position.entry_price

        if self.position.side != PositionStatus.LONG:
            return False

        # === TAKE PROFIT ===
        if high >= entry_price * (1 + self.config.take_profit_pct):
            return True

        # === STOP LOSS ===
        if low <= entry_price * (1 - self.config.stop_loss_pct):
            return True

        # === TRAILING STOP ===
        # Activate when in profit
        if self.position.highest_price > entry_price * 1.001:  # 0.1% profit threshold
            trail_price = self.position.highest_price * (1 - self.config.trailing_stop_pct)
            if low <= trail_price:
                return True

        return False

    def _close_position(self, price: float, time: datetime):
        """Close the current position (LONG only for spot trading)."""
        if self.position.side == PositionStatus.FLAT:
            return

        # Apply slippage (selling)
        slippage = price * self.config.slippage_pct
        exit_price = price - slippage  # LONG sell = price - slippage

        # Calculate P&L (LONG position)
        pnl = (exit_price - self.position.entry_price) * self.position.quantity

        # Apply exit commission
        exit_commission = exit_price * self.position.quantity * self.config.commission_rate
        pnl -= exit_commission

        # Update trade record (update the original entry trade)
        current_trade = self.trades[-1]
        current_trade.exit_time = time
        current_trade.exit_price = exit_price
        current_trade.pnl = pnl
        current_trade.pnl_pct = pnl / self.position.total_cost if self.position.total_cost > 0 else 0
        current_trade.commission += exit_commission
        current_trade.slippage += slippage * self.position.quantity
        current_trade.holding_bars = self.current_bar - self.last_trade_bar

        # Update capital (return total position value + P&L)
        self.capital += self.position.total_cost + pnl

        # Track daily loss
        if pnl < 0:
            self.daily_loss += abs(pnl)

        # Reset position
        self.position.side = PositionStatus.FLAT
        self.position.quantity = 0
        self.position.entry_price = 0
        self.position.unrealized_pnl = 0
        self.position.dca_count = 0
        self.position.total_cost = 0
        self.position.dca_levels = []
        self.dca_trades = []

    def _get_dca_trigger_and_multiplier(self, dca_level: int) -> tuple:
        """
        Get trigger percentage and multiplier for each DCA level.
        DYNAMIC DCA: Aggressive → Aggressive → Moderate → Conservative
        """
        if dca_level == 0:
            return self.config.dca_level_1_trigger_pct, self.config.dca_level_1_multiplier
        elif dca_level == 1:
            return self.config.dca_level_2_trigger_pct, self.config.dca_level_2_multiplier
        elif dca_level == 2:
            return self.config.dca_level_3_trigger_pct, self.config.dca_level_3_multiplier
        elif dca_level == 3:
            return self.config.dca_level_4_trigger_pct, self.config.dca_level_4_multiplier
        else:
            # Fallback for any additional levels
            return 0.02, 1.0

    def _check_dca_opportunity(self, current_price: float, time: datetime):
        """
        Check if we should add to position (DYNAMIC DCA).

        DYNAMIC DCA Strategy:
        - Level 1 & 2: AGGRESSIVE (2x multiplier, tighter triggers)
        - Level 3: MODERATE (1.5x multiplier)
        - Level 4: CONSERVATIVE (1x multiplier, widest trigger)
        - SL only triggers AFTER all DCA exhausted
        """
        if not self.config.dca_enabled:
            return

        if self.position.side != PositionStatus.LONG:
            return

        if self.position.dca_count >= self.config.max_dca_entries:
            return  # Already at max DCA

        # Get trigger for current DCA level
        trigger_pct, _ = self._get_dca_trigger_and_multiplier(self.position.dca_count)

        # Calculate the trigger price based on last entry level
        last_entry_price = self.position.dca_levels[-1]
        dca_trigger_price = last_entry_price * (1 - trigger_pct)

        # Check if current price has dropped to trigger level
        if current_price <= dca_trigger_price:
            self._execute_dca(current_price, time)

    def _execute_dca(self, price: float, time: datetime):
        """Execute a DYNAMIC DCA buy (Aggressive → Moderate → Conservative)."""
        # Get multiplier for current DCA level
        _, multiplier = self._get_dca_trigger_and_multiplier(self.position.dca_count)

        # Calculate DCA position size based on level multiplier
        base_size = self.config.position_size_pct * self.capital
        dca_size = base_size * multiplier
        dca_size = min(dca_size, self.config.max_position_size)

        # Apply slippage
        slippage = price * self.config.slippage_pct
        dca_price = price + slippage  # BUY

        dca_quantity = dca_size / dca_price

        # Apply commission
        dca_commission = dca_size * self.config.commission_rate

        # Check if we have enough capital
        if dca_size + dca_commission > self.capital:
            return  # Not enough capital for DCA

        # Update position with averaged values
        new_total_quantity = self.position.quantity + dca_quantity
        new_total_cost = self.position.total_cost + dca_size

        # Calculate new average entry price
        new_avg_price = new_total_cost / new_total_quantity

        # Update position
        self.position.quantity = new_total_quantity
        self.position.entry_price = new_avg_price  # Average entry price
        self.position.total_cost = new_total_cost
        self.position.dca_count += 1
        self.position.dca_levels.append(dca_price)

        # Track DCA entry
        self.dca_trades.append({
            'dca_number': self.position.dca_count,
            'price': dca_price,
            'quantity': dca_quantity,
            'value': dca_size,
            'time': time,
            'new_avg_price': new_avg_price
        })

        # Deduct commission from capital
        self.capital -= dca_commission

        # Update the trade record to reflect DCA
        if self.trades:
            current_trade = self.trades[-1]
            current_trade.quantity = new_total_quantity
            current_trade.commission += dca_commission

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity."""
        if self.position.side == PositionStatus.FLAT:
            return self.capital
        else:
            return self.capital + self.position.unrealized_pnl

    def _calculate_results(self) -> Dict:
        """Calculate comprehensive backtest results."""
        if not self.trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.capital,
                'total_return': 0,
                'total_return_pct': 0,
                'total_pnl': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'win_rate_pct': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'calmar_ratio': 0,
                'total_commission': 0,
                'total_slippage': 0,
                'equity_curve': [self.initial_capital],
                'trades': []
            }

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trades)
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0

        # Risk metrics from equity curve
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 * 24 * 60 / len(equity_curve))  # Annualize for 1-min bars

        # Sharpe ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)
        else:
            sharpe = 0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino = (returns.mean() / negative_returns.std()) * np.sqrt(252 * 24 * 60)
        else:
            sortino = 0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max()

        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Trade duration stats
        holding_bars = [t.holding_bars for t in self.trades]
        avg_holding = np.mean(holding_bars) if holding_bars else 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annual_return': annual_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar,
            'avg_holding_bars': avg_holding,
            'equity_curve': equity_curve.tolist(),
            'trades': [self._trade_to_dict(t) for t in self.trades]
        }

    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert trade to dictionary."""
        # Handle numpy datetime64 conversion
        def dt_to_str(dt):
            if dt is None:
                return None
            if hasattr(dt, 'isoformat'):
                return dt.isoformat()
            # Handle numpy.datetime64
            return str(pd.Timestamp(dt))

        return {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'side': trade.side.value,
            'entry_time': dt_to_str(trade.entry_time),
            'entry_price': trade.entry_price,
            'exit_time': dt_to_str(trade.exit_time),
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'commission': trade.commission,
            'holding_bars': trade.holding_bars
        }

    def _print_results(self, results: Dict):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"\nCapital: ${results['initial_capital']:,.2f} -> ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print(f"\nTrades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate_pct']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Avg Win: ${results['avg_win']:,.2f}")
        print(f"Avg Loss: ${results['avg_loss']:,.2f}")
        print(f"\nSharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
        print(f"\nTotal Commission: ${results['total_commission']:,.2f}")
        print(f"Total Slippage: ${results['total_slippage']:,.2f}")
        print("=" * 60)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([self._trade_to_dict(t) for t in self.trades])

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Equity curve
            ax1 = axes[0, 0]
            equity = results['equity_curve']
            ax1.plot(equity, 'b-', linewidth=1)
            ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Bar')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)

            # Drawdown
            ax2 = axes[0, 1]
            equity_arr = np.array(equity)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (peak - equity_arr) / peak * 100
            ax2.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.3)
            ax2.plot(drawdown, 'r-', linewidth=1)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Bar')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

            # Trade P&L distribution
            ax3 = axes[1, 0]
            pnls = [t['pnl'] for t in results['trades']]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_title('Trade P&L')
            ax3.set_xlabel('Trade #')
            ax3.set_ylabel('P&L ($)')
            ax3.grid(True, alpha=0.3)

            # Cumulative P&L
            ax4 = axes[1, 1]
            cum_pnl = np.cumsum(pnls)
            ax4.plot(cum_pnl, 'b-', linewidth=2)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.3)
            ax4.set_title('Cumulative P&L')
            ax4.set_xlabel('Trade #')
            ax4.set_ylabel('Cumulative P&L ($)')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")


if __name__ == "__main__":
    # Test backtest engine
    print("Testing Backtest Engine...")

    # Create synthetic data
    np.random.seed(42)
    n_bars = 10000

    # Simulate price movement
    returns = np.random.normal(0.0001, 0.002, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'datetime': pd.date_range(start='2024-01-01', periods=n_bars, freq='1min'),
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.003, n_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.003, n_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })

    # Create random signals
    signals = np.random.choice([-1, 0, 0, 0, 1], n_bars)  # More holds than trades
    confidences = np.random.uniform(0.5, 0.9, n_bars)

    # Run backtest
    engine = BacktestEngine()
    results = engine.run(data, signals, confidences, symbol="TEST")
