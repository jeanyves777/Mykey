"""
DCA (Dollar Cost Averaging) Trading Configuration
==================================================

ALWAYS HOLD STRATEGY:
- Always maintain a BASE position (never closed)
- TRADE position uses DCA logic and closes at TP
- When TP hit: close TRADE positions, keep HOLD, re-enter new TRADE
- No entry signals required - always in the market

HYBRID DCA FILTER:
- DCA 1-2: Easy triggers (just price drop + weak RSI filter)
- DCA 3-4: Strict triggers (require reversal confirmation)

Risk-Based Position Sizing:
- $333 per asset (BTC, ETH, SOL)
- Max 4 DCA levels per position
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class DCAConfig:
    """Configuration for DCA-based crypto trading."""

    # ================================================================
    # ACCOUNT SETTINGS
    # ================================================================
    initial_capital: float = 1000.0      # Starting capital
    risk_per_trade_pct: float = 0.02     # 2% risk per trade
    max_open_trades: int = 3             # Max concurrent positions (one per symbol)
    allocation_per_symbol: float = 333.0  # $333 per symbol (BTC, ETH, SOL)

    # ================================================================
    # ALWAYS HOLD STRATEGY SETTINGS
    # ================================================================
    use_always_hold: bool = True          # Enable Always Hold strategy
    hold_position_pct: float = 0.0        # NO separate HOLD - use ALL allocation for trading
    trade_position_pct: float = 1.0       # 100% of allocation = $333 for Entry + 4 DCAs
    # Full $333 per symbol distributed across: Entry + DCA1 + DCA2 + DCA3 + DCA4 = $333 total

    # No entry signals - immediately enter on startup
    require_entry_signal: bool = False    # Disabled for Always Hold

    # When TRADE position hits TP, immediately re-enter new TRADE
    auto_reentry_on_tp: bool = True

    # ================================================================
    # SIGNAL SETTINGS (Momentum-based)
    # ================================================================
    momentum_period: int = 5             # Bars for momentum calculation
    momentum_threshold: float = 0.2      # Min momentum % to enter (0.2%)
    rsi_period: int = 14                 # RSI calculation period
    rsi_max_for_entry: float = 65.0      # Don't enter if RSI > this (overbought)
    rsi_min_for_entry: float = 20.0      # Don't enter if RSI < this (too oversold)

    # Optional: Simple trend filter (disabled by default - not in profitable test)
    use_trend_filter: bool = False       # Set False to match profitable test
    trend_sma_period: int = 50           # SMA period for trend
    require_above_sma: bool = True       # Only enter if price > SMA

    # ================================================================
    # HYBRID DCA FILTER (Easy 1-2, Strict 3-4)
    # ================================================================
    # DCA 1-2: Easy triggers - just need price drop + basic RSI check
    # DCA 3-4: Strict triggers - require reversal confirmation

    use_smart_dca_filter: bool = True     # Enable Hybrid DCA filtering
    adx_period: int = 14                  # ADX calculation period

    # EASY DCA (Level 1-2): Minimal requirements
    easy_dca_max_level: int = 2           # DCA 1-2 use easy filters
    easy_dca_rsi_threshold: float = 45.0  # Just need RSI < 45 (not overbought)
    easy_dca_adx_max: float = 40.0        # Allow DCA even in moderate trends (ADX < 40)

    # STRICT DCA (Level 3-4): Require reversal confirmation
    strict_dca_min_level: int = 3         # DCA 3-4 use strict filters
    strict_dca_adx_max: float = 25.0      # Only DCA in weak trends (ADX < 25)
    strict_dca_rsi_threshold: float = 35.0  # Need RSI < 35 (oversold)
    require_reversal_candle: bool = True  # Must have reversal candle pattern
    min_reversal_body_ratio: float = 0.5  # Body must be 50% of candle range

    # EMA for reversal detection (used for strict DCA)
    ema_fast_period: int = 8
    ema_slow_period: int = 21

    # Volume confirmation for strict DCA
    require_volume_spike: bool = True     # DCA 3-4 need volume spike
    volume_spike_multiplier: float = 1.5  # Volume must be 1.5x average

    # Time-based cooldown between DCAs
    dca_cooldown_minutes: int = 30        # Min 30 minutes between DCA entries
    strict_dca_cooldown_minutes: int = 60 # DCA 3-4 need 60 min cooldown

    # ================================================================
    # TAKE PROFIT / STOP LOSS
    # ================================================================
    # IMPORTANT: Alpaca charges 0.15% per trade (0.30% round trip)
    # TP must exceed 0.30% just to break even!
    # Target: 0.8% TP = 0.5% NET profit after fees
    take_profit_pct: float = 0.008       # 0.8% TP (nets ~0.5% after 0.30% fees)
    stop_loss_pct: float = 0.006         # 0.6% SL (slightly wider for crypto volatility)
    max_hold_bars: int = 300             # Max bars to hold (5 hours on 1-min)

    # ================================================================
    # SMART DCA EXIT SETTINGS
    # ================================================================
    # Reduce TP as DCA level increases (exit faster with larger positions)
    # But NEVER go below fee threshold!
    # DCA 0: 0.8%, DCA 1: 0.7%, DCA 2: 0.6%, DCA 3: 0.5%, DCA 4: 0.45%
    tp_reduction_per_dca: float = 0.001  # Reduce TP by 0.1% per DCA level
    min_tp_pct: float = 0.0045           # Minimum TP (0.45%) - MUST exceed 0.30% fees!

    # Breakeven protection for high DCA levels
    breakeven_dca_threshold: int = 3     # Enable BE lock at DCA level 3+
    breakeven_buffer_pct: float = 0.004  # Lock BE when +0.4% profit (covers 0.30% fees + buffer)

    # Emergency exit for max DCA positions
    emergency_exit_dca_level: int = 4    # At DCA 4, use emergency exit
    emergency_exit_profit_pct: float = 0.004   # Exit at +0.4% profit (nets +0.1% after fees)

    # ================================================================
    # DCA (Dollar Cost Averaging) SETTINGS
    # ================================================================
    use_dca: bool = True
    dca_spacing_pct: float = 0.005       # 0.5% between DCA levels (wider for crypto volatility)
    max_dca_stages: int = 4              # Max additional entries

    # DCA multipliers - NORMALIZED to sum to 1.0
    # These represent the FRACTION of FULL $333 allocation for each level
    # Entry: 20%, DCA1: 15%, DCA2: 18%, DCA3: 22%, DCA4: 25% = 100% total
    # With FULL $333 allocation:
    #   Entry: $66.60, DCA1: $49.95, DCA2: $59.94, DCA3: $73.26, DCA4: $83.25 = $333 total
    dca_multipliers: List[float] = field(default_factory=lambda: [0.20, 0.15, 0.18, 0.22, 0.25])

    # Max daily DCA entries per symbol (prevent over-averaging in crashes)
    max_dca_per_day: int = 4              # Max 4 DCA entries per symbol per day

    # ================================================================
    # TRADING FEES (ALPACA CRYPTO)
    # ================================================================
    # Alpaca charges 0.15% per trade (maker/taker)
    # Round trip = 0.30% (entry + exit)
    commission_pct: float = 0.0015       # 0.15% per trade (Alpaca standard crypto fee)

    # ================================================================
    # COOLDOWN
    # ================================================================
    cooldown_bars: int = 50              # Min bars between trades (matches profitable test)

    # ================================================================
    # SYMBOLS (Only profitable ones - BTC, ETH, SOL)
    # ================================================================
    symbols: List[str] = field(default_factory=lambda: ['BTCUSD', 'ETHUSD', 'SOLUSD'])

    # ================================================================
    # DATA SETTINGS
    # ================================================================
    timeframe: str = '1m'
    data_dir: str = field(default_factory=lambda: str(Path(__file__).parent / "Crypto_Data_Fresh"))

    # ================================================================
    # LOGGING
    # ================================================================
    log_dir: str = field(default_factory=lambda: str(Path(__file__).parent / "logs" / "dca_trading"))
    log_trades: bool = True
    verbose: bool = True  # Set True to see data errors

    def get_hold_quantity(self, symbol: str, price: float) -> float:
        """
        Calculate HOLD position quantity (never closed).

        Args:
            symbol: Trading symbol
            price: Current price

        Returns:
            Quantity for HOLD position
        """
        hold_value = self.allocation_per_symbol * self.hold_position_pct
        return hold_value / price

    def get_trade_quantity(self, symbol: str, price: float, dca_level: int = 0) -> float:
        """
        Calculate TRADE position quantity (closes at TP, uses DCA).

        The multipliers sum to 1.0, so total TRADE allocation = trade_position_pct * allocation_per_symbol
        Each DCA level gets its fraction of that total.

        Args:
            symbol: Trading symbol
            price: Current price
            dca_level: DCA level (0 = initial entry, 1-4 = DCA levels)

        Returns:
            Quantity for TRADE position at this DCA level
        """
        # Total TRADE allocation (e.g., $333 * 0.70 = $233)
        trade_allocation = self.allocation_per_symbol * self.trade_position_pct

        # Get multiplier for this level (fraction of trade allocation)
        multiplier = self.dca_multipliers[dca_level] if dca_level < len(self.dca_multipliers) else 0.20

        # Position value for this level
        position_value = trade_allocation * multiplier

        return position_value / price

    def get_dca_levels(self, entry_price: float) -> List[float]:
        """Calculate DCA price levels below entry."""
        levels = []
        for i in range(1, self.max_dca_stages + 1):
            dca_price = entry_price * (1 - self.dca_spacing_pct * i)
            levels.append(dca_price)
        return levels

    def is_easy_dca(self, dca_level: int) -> bool:
        """Check if this DCA level uses easy filters."""
        return dca_level <= self.easy_dca_max_level

    def is_strict_dca(self, dca_level: int) -> bool:
        """Check if this DCA level uses strict filters."""
        return dca_level >= self.strict_dca_min_level

    def get_dca_cooldown(self, dca_level: int) -> int:
        """Get cooldown in minutes for this DCA level."""
        if self.is_strict_dca(dca_level):
            return self.strict_dca_cooldown_minutes
        return self.dca_cooldown_minutes

    def __str__(self):
        total_allocation = self.allocation_per_symbol  # Full $333

        # Calculate DCA breakdown - using FULL allocation
        dca_breakdown = []
        running_total = 0.0
        for i, mult in enumerate(self.dca_multipliers):
            stage = "Entry" if i == 0 else f"DCA{i}"
            val = total_allocation * mult
            running_total += val
            dca_breakdown.append(f"{stage}: ${val:.2f} ({mult*100:.0f}%)")

        return f"""
Always Hold + Hybrid DCA Configuration
======================================
Capital: ${self.initial_capital:,.2f}
Per Symbol: ${self.allocation_per_symbol:,.2f} (FULL allocation used)

Position Breakdown (Entry + 4 DCAs = ${total_allocation:.2f} total):
  {chr(10).join('  ' + b for b in dca_breakdown)}
  ----------------------------------------
  TOTAL: ${running_total:.2f}

HYBRID DCA Filters:
  Easy DCA (1-2): RSI < {self.easy_dca_rsi_threshold}, ADX < {self.easy_dca_adx_max}
  Strict DCA (3-4): RSI < {self.strict_dca_rsi_threshold}, ADX < {self.strict_dca_adx_max}, reversal candle

Targets:
  Take Profit: {self.take_profit_pct*100:.2f}% (net ~{(self.take_profit_pct - 0.003)*100:.2f}% after fees)
  DCA Spacing: {self.dca_spacing_pct*100:.2f}%
  Max DCA stages: {self.max_dca_stages}

Commission: {self.commission_pct*100:.2f}% per trade
Symbols: {self.symbols}
"""


# Default configuration
def load_dca_config() -> DCAConfig:
    """Load the default DCA configuration."""
    return DCAConfig()


def load_always_hold_config() -> DCAConfig:
    """Load Always Hold configuration."""
    return DCAConfig(
        use_always_hold=True,
        require_entry_signal=False,
        auto_reentry_on_tp=True
    )


if __name__ == "__main__":
    # Test configuration
    config = load_always_hold_config()
    print(config)

    print("\n" + "=" * 60)
    print("POSITION SIZING VERIFICATION - FULL $333 ALLOCATION")
    print("=" * 60)

    btc_price = 100000
    eth_price = 3500
    sol_price = 200

    for symbol, price in [('BTCUSD', btc_price), ('ETHUSD', eth_price), ('SOLUSD', sol_price)]:
        print(f"\n{symbol} at ${price:,.2f}:")

        total_value = 0.0

        # Entry position
        entry_qty = config.get_trade_quantity(symbol, price, dca_level=0)
        entry_val = entry_qty * price
        total_value += entry_val
        print(f"  Entry: {entry_qty:.6f} = ${entry_val:.2f} (20%)")

        # DCA levels
        levels = config.get_dca_levels(price)
        for i, level in enumerate(levels, 1):
            dca_qty = config.get_trade_quantity(symbol, level, dca_level=i)
            dca_val = dca_qty * level
            total_value += dca_val
            filter_type = "EASY" if config.is_easy_dca(i) else "STRICT"
            pct = config.dca_multipliers[i] * 100
            print(f"  DCA {i}: {dca_qty:.6f} @ ${level:,.2f} = ${dca_val:.2f} ({pct:.0f}%) - {filter_type}")

        print(f"  ----------------------------------------")
        print(f"  TOTAL: ${total_value:.2f} (should be ~${config.allocation_per_symbol:.2f})")
