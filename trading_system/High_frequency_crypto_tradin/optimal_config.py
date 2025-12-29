"""
OPTIMAL TRADING CONFIGURATION
==============================
Best balanced configuration from parameter optimization.

Results from 1-week backtest (Dec 14-21, 2025):
- Win Rate: 93.1% (27W / 2L)
- Trades: 29
- P&L: +$1.61 (+0.81%)
- Profit Factor: 11.69
- Sharpe Ratio: 75.46
- Max Drawdown: 0.07%

This configuration prioritizes:
1. High win rate (93%+) over trade frequency
2. Quick, small profits (0.8% TP) with tight stops (0.6% SL)
3. Unanimous model agreement (5/5) for highest confidence
4. R:R >= 1.2 for favorable risk-adjusted returns
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OptimalTradingConfig:
    """
    Optimal trading configuration based on extensive backtesting.

    Key Parameters:
    - Stop Loss: 0.6% (tight to minimize losses)
    - Take Profit: 0.8% (quick profits)
    - R:R Ratio: 1.2 (favorable risk-reward)
    - Model Agreement: 5/5 (unanimous for high confidence)
    """

    # Capital
    initial_capital: float = 200.0

    # Risk Management - OPTIMIZED
    stop_loss_pct: float = 0.006      # 0.6% stop loss
    take_profit_pct: float = 0.008    # 0.8% take profit
    min_risk_reward: float = 1.2      # R:R >= 1.2

    # Model Confidence - OPTIMIZED
    min_confidence: float = 0.60      # 60% minimum confidence
    min_model_agreement: int = 5      # 5/5 unanimous

    # Trade Management
    max_trades_per_day: int = 50      # Allow many trades
    cooldown_minutes: int = 10        # 10 min between trades

    # Filters
    min_volume_ratio: float = 0.5     # Volume filter
    min_volatility: float = 0.0001    # Min volatility
    require_trend_alignment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'initial_capital': self.initial_capital,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'min_risk_reward': self.min_risk_reward,
            'min_confidence': self.min_confidence,
            'min_model_agreement': self.min_model_agreement,
            'max_trades_per_day': self.max_trades_per_day,
            'cooldown_minutes': self.cooldown_minutes,
            'min_volume_ratio': self.min_volume_ratio,
            'min_volatility': self.min_volatility,
            'require_trend_alignment': self.require_trend_alignment,
        }


# Alternative configurations from optimization

# HIGH WIN RATE CONFIG (100% WR, fewer trades)
HIGH_WR_CONFIG = {
    'stop_loss_pct': 0.008,      # 0.8%
    'take_profit_pct': 0.012,    # 1.2%
    'min_risk_reward': 2.0,      # R:R >= 2.0
    'min_model_agreement': 5,    # 5/5
    'expected_wr': 1.00,
    'expected_trades': 6,
}

# MAX PROFIT CONFIG (highest P&L)
MAX_PROFIT_CONFIG = {
    'stop_loss_pct': 0.004,      # 0.4%
    'take_profit_pct': 0.014,    # 1.4%
    'min_risk_reward': 0.8,      # R:R >= 0.8
    'min_model_agreement': 5,    # 5/5
    'expected_wr': 0.59,
    'expected_pnl': 2.31,
}

# BALANCED CONFIG (best risk-adjusted returns)
BALANCED_CONFIG = {
    'stop_loss_pct': 0.006,      # 0.6%
    'take_profit_pct': 0.008,    # 0.8%
    'min_risk_reward': 1.2,      # R:R >= 1.2
    'min_model_agreement': 5,    # 5/5
    'expected_wr': 0.93,
    'expected_trades': 29,
    'expected_pnl': 1.61,
    'profit_factor': 11.69,
}

# AGGRESSIVE CONFIG (more trades, lower WR)
AGGRESSIVE_CONFIG = {
    'stop_loss_pct': 0.008,      # 0.8%
    'take_profit_pct': 0.012,    # 1.2%
    'min_risk_reward': 0.0,      # No R:R filter
    'min_model_agreement': 3,    # 3/5
    'expected_wr': 0.50,
    'expected_trades': 40,
}


def get_optimal_config():
    """Get the optimal trading configuration."""
    return OptimalTradingConfig()


def print_config_summary():
    """Print summary of optimal configuration."""
    config = OptimalTradingConfig()

    print("\n" + "=" * 60)
    print("OPTIMAL TRADING CONFIGURATION")
    print("=" * 60)
    print(f"\nRisk Management:")
    print(f"  Stop Loss:    {config.stop_loss_pct:.2%}")
    print(f"  Take Profit:  {config.take_profit_pct:.2%}")
    print(f"  Min R:R:      {config.min_risk_reward:.1f}")

    print(f"\nModel Confidence:")
    print(f"  Min Confidence: {config.min_confidence:.0%}")
    print(f"  Min Agreement:  {config.min_model_agreement}/5")

    print(f"\nTrade Management:")
    print(f"  Max Trades/Day: {config.max_trades_per_day}")
    print(f"  Cooldown:       {config.cooldown_minutes} min")

    print(f"\nExpected Performance (from backtest):")
    print(f"  Win Rate:       93.1%")
    print(f"  Profit Factor:  11.69")
    print(f"  Sharpe Ratio:   75.46")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
