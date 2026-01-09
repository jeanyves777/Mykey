"""
Multi-Symbol OTM Day Trading Engine

Trades OTM options on multiple symbols: SPY, QQQ, AMD, IWM, PLTR, BAC
Uses day-specific parameters for DTE, strike distance, targets, and stops.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trading_system.config.multi_symbol_otm_config import MultiSymbolOTMConfig, load_config_from_file
from trading_system.strategies.multi_symbol_otm_strategy import MultiSymbolOTMStrategy, SignalResult

# Use our custom AlpacaClient which handles data fetching properly
try:
    from trading_system.engine.alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARNING] AlpacaClient not available. Running in simulation mode.")


@dataclass
class Position:
    """Represents an open option position."""
    symbol: str  # Underlying symbol (SPY, QQQ, etc.)
    contract_symbol: str  # Option contract symbol
    direction: str  # 'BULLISH' or 'BEARISH'
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: str
    quantity: int
    entry_price: float
    entry_time: datetime
    target_price: float
    stop_price: float
    current_price: float = 0.0
    order_id: str = ""


@dataclass
class SymbolState:
    """State for a single symbol."""
    symbol: str
    bars_1min: List[dict] = field(default_factory=list)
    bars_5min: List[dict] = field(default_factory=list)
    last_signal: Optional[SignalResult] = None
    last_trade_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None


class MultiSymbolOTMEngine:
    """
    Multi-Symbol OTM Day Trading Engine.

    Features:
    - Trades multiple symbols simultaneously
    - Day-specific parameters (DTE, strike, target, stop)
    - 4-layer signal validation with V4 pullback detection
    - Position management with targets and stops
    """

    def __init__(self, config: MultiSymbolOTMConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}  # contract_symbol -> Position
        self.symbol_states: Dict[str, SymbolState] = {}  # symbol -> SymbolState
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.running: bool = False

        # Initialize symbol states
        for symbol in config.symbols:
            self.symbol_states[symbol] = SymbolState(symbol=symbol)

        # Initialize Alpaca client (uses our custom AlpacaClient)
        self.client = None

        if ALPACA_AVAILABLE and config.api_key and config.api_secret:
            try:
                self.client = AlpacaClient(
                    api_key=config.api_key,
                    api_secret=config.api_secret,
                    paper=config.use_paper
                )
                self._log("AlpacaClient initialized successfully", "SUCCESS")
            except Exception as e:
                self._log(f"Failed to initialize Alpaca: {e}", "ERROR")

    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_colors = {
            "INFO": "",
            "SUCCESS": "[+]",
            "WARNING": "[!]",
            "ERROR": "[X]",
            "SIGNAL": "[~]",
            "TRADE": "[$]"
        }
        prefix = level_colors.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")

    def _get_weekday(self) -> int:
        """Get current weekday (0=Monday, 4=Friday)."""
        return datetime.now().weekday()

    def _get_day_params(self) -> Tuple[int, int, int, float, float, float]:
        """Get day-specific parameters."""
        weekday = self._get_weekday()
        config = self.config.get_day_config(weekday)
        return (
            config.dte,
            config.strike_otm,
            config.strike_otm_max,
            config.budget,
            config.target_pct,
            config.stop_loss_pct
        )

    def _is_trading_hours(self) -> bool:
        """Check if within trading hours."""
        now = datetime.now().time()
        return self.config.entry_time_start <= now <= self.config.entry_time_end

    def _is_force_exit_time(self) -> bool:
        """Check if it's time to force exit all positions."""
        now = datetime.now().time()
        return now >= self.config.force_exit_time

    def _can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """Check if we can open a new position for this symbol."""
        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached"

        # Check max positions per symbol
        symbol_positions = sum(1 for p in self.positions.values() if p.symbol == symbol)
        if symbol_positions >= self.config.max_positions_per_symbol:
            return False, f"Max positions for {symbol} reached"

        # Check cooldown
        state = self.symbol_states.get(symbol)
        if state and state.cooldown_until:
            if datetime.now() < state.cooldown_until:
                return False, f"{symbol} in cooldown until {state.cooldown_until.strftime('%H:%M:%S')}"

        # Check daily budget
        dte, otm_min, otm_max, budget, target_pct, stop_pct = self._get_day_params()
        spent_today = sum(p.entry_price * p.quantity * 100 for p in self.positions.values())
        if spent_today >= budget:
            return False, f"Daily budget (${budget}) exhausted"

        return True, "OK"

    def _fetch_bars(self, symbol: str) -> Tuple[List[dict], List[dict]]:
        """Fetch 1-minute and 5-minute bars for a symbol."""
        from datetime import timezone
        import pytz

        bars_1min = []
        bars_5min = []

        if not self.client:
            self._log(f"  [{symbol}] No client available", "WARNING")
            return bars_1min, bars_5min

        try:
            # Use a proper time range for fetching bars
            est = pytz.timezone('US/Eastern')
            now = datetime.now(est)
            start_1min = now - timedelta(hours=3)  # 3 hours for 120+ 1-min bars
            start_5min = now - timedelta(hours=6)  # 6 hours for 50+ 5-min bars

            # Fetch 1-minute bars using our AlpacaClient
            raw_bars_1min = self.client.get_stock_bars(
                symbol,
                timeframe='1Min',
                start=start_1min,
                limit=120
            )

            self._log(f"  [{symbol}] Raw 1-min bars returned: {len(raw_bars_1min)}", "INFO")

            for bar in raw_bars_1min:
                bars_1min.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })

            # Fetch 5-minute bars
            raw_bars_5min = self.client.get_stock_bars(
                symbol,
                timeframe='5Min',
                start=start_5min,
                limit=50
            )

            self._log(f"  [{symbol}] Raw 5-min bars returned: {len(raw_bars_5min)}", "INFO")

            for bar in raw_bars_5min:
                bars_5min.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })

            # Log if no data found
            if not bars_1min and not bars_5min:
                self._log(f"  [{symbol}] No bar data returned from API", "WARNING")

        except Exception as e:
            self._log(f"  [{symbol}] Error fetching bars: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        return bars_1min, bars_5min

    def _get_option_chain(self, symbol: str, dte: int) -> List[dict]:
        """Get option chain for a symbol with specific DTE."""
        options = []

        if not self.client:
            return options

        try:
            # Calculate expiry date
            today = datetime.now().date()
            expiry = today + timedelta(days=dte)

            # For 0 DTE, use today's expiry
            if dte == 0:
                expiry = today

            # Adjust for weekends
            while expiry.weekday() > 4:  # Saturday = 5, Sunday = 6
                expiry += timedelta(days=1)

            expiry_str = expiry.strftime("%Y-%m-%d")

            # Get current price
            state = self.symbol_states.get(symbol)
            if not state or not state.bars_1min:
                return options

            current_price = state.bars_1min[-1]['close']

            # Calculate strike range
            if symbol in ['SPY', 'QQQ', 'IWM']:
                strike_increment = 1.0
                strike_range = 10
            elif symbol in ['AMD', 'PLTR', 'BAC']:
                strike_increment = 0.5
                strike_range = 5
            else:
                strike_increment = 1.0
                strike_range = 5

            # Generate potential strikes
            base_strike = round(current_price / strike_increment) * strike_increment

            for i in range(-strike_range, strike_range + 1):
                strike = base_strike + (i * strike_increment)

                # Add call option
                options.append({
                    'symbol': symbol,
                    'strike': strike,
                    'expiry': expiry_str,
                    'option_type': 'call',
                    'bid': 0.50,  # Placeholder - would fetch from API
                    'ask': 0.55,
                    'contract_symbol': f"{symbol}{expiry.strftime('%y%m%d')}C{int(strike * 1000):08d}"
                })

                # Add put option
                options.append({
                    'symbol': symbol,
                    'strike': strike,
                    'expiry': expiry_str,
                    'option_type': 'put',
                    'bid': 0.50,
                    'ask': 0.55,
                    'contract_symbol': f"{symbol}{expiry.strftime('%y%m%d')}P{int(strike * 1000):08d}"
                })

        except Exception as e:
            self._log(f"Error getting option chain for {symbol}: {e}", "ERROR")

        return options

    def _execute_entry(self, symbol: str, signal: SignalResult, option: dict) -> Optional[Position]:
        """Execute entry order for an option."""
        if not option:
            return None

        dte, otm_min, otm_max, budget, target_pct, stop_pct = self._get_day_params()

        # Calculate position size based on budget
        ask_price = option.get('ask', 0.55)
        if ask_price <= 0:
            return None

        # Calculate quantity (each contract = 100 shares)
        max_contracts = int(budget / (ask_price * 100))
        quantity = max(1, min(max_contracts, 2))  # 1-2 contracts max

        # Check price limits
        if ask_price < self.config.min_option_price or ask_price > self.config.max_option_price:
            self._log(f"Option price ${ask_price:.2f} outside limits [${self.config.min_option_price:.2f}-${self.config.max_option_price:.2f}]", "WARNING")
            return None

        # Calculate target and stop prices
        target_price = ask_price * (1 + target_pct / 100)
        stop_price = ask_price * (1 - stop_pct / 100)

        self._log(f"", "INFO")
        self._log(f"{'='*60}", "INFO")
        self._log(f"    >>> V4 APPROVED: EXECUTING ENTRY <<<", "SUCCESS")
        self._log(f"{'='*60}", "INFO")
        self._log(f"    Symbol: {symbol}", "INFO")
        self._log(f"    Direction: {signal.direction}", "INFO")
        self._log(f"    Option: {option['option_type'].upper()} @ ${option['strike']:.2f}", "INFO")
        self._log(f"    Expiry: {option['expiry']} (DTE: {dte})", "INFO")
        self._log(f"    Entry: ${ask_price:.2f} x {quantity} contracts", "INFO")
        self._log(f"    Target: ${target_price:.2f} (+{target_pct}%)", "INFO")
        self._log(f"    Stop: ${stop_price:.2f} (-{stop_pct}%)", "INFO")
        self._log(f"    Confidence: {signal.confidence:.1%}", "INFO")
        self._log(f"{'='*60}", "INFO")

        # Create position
        position = Position(
            symbol=symbol,
            contract_symbol=option['contract_symbol'],
            direction=signal.direction,
            option_type=option['option_type'],
            strike=option['strike'],
            expiry=option['expiry'],
            quantity=quantity,
            entry_price=ask_price,
            entry_time=datetime.now(),
            target_price=target_price,
            stop_price=stop_price,
            current_price=ask_price
        )

        # Execute order via Alpaca (if available)
        if self.client:
            try:
                order = self.client.submit_option_market_order(
                    symbol=option['contract_symbol'],
                    qty=quantity,
                    side='buy'
                )
                position.order_id = str(order.get('id', ''))
                self._log(f"    Order submitted: {order.get('id', 'N/A')}", "SUCCESS")
            except Exception as e:
                self._log(f"    Order failed: {e}", "ERROR")
                self._log(f"    [PAPER] Simulating entry...", "INFO")

        # Store position
        self.positions[position.contract_symbol] = position
        self.daily_trades += 1

        # Set cooldown for this symbol (5 minutes)
        self.symbol_states[symbol].cooldown_until = datetime.now() + timedelta(minutes=5)
        self.symbol_states[symbol].last_trade_time = datetime.now()

        return position

    def _check_exit_conditions(self, position: Position) -> Tuple[bool, str]:
        """Check if position should be exited."""
        # Force exit time
        if self._is_force_exit_time():
            return True, "FORCE_EXIT_TIME"

        # Target hit
        if position.current_price >= position.target_price:
            return True, "TARGET_HIT"

        # Stop loss hit
        if position.current_price <= position.stop_price:
            return True, "STOP_LOSS"

        return False, ""

    def _execute_exit(self, position: Position, reason: str) -> float:
        """Execute exit order for a position."""
        pnl = (position.current_price - position.entry_price) * position.quantity * 100
        pnl_pct = (position.current_price / position.entry_price - 1) * 100

        self._log(f"", "INFO")
        self._log(f"{'='*60}", "INFO")
        self._log(f"    >>> EXITING POSITION: {reason} <<<", "TRADE")
        self._log(f"{'='*60}", "INFO")
        self._log(f"    Symbol: {position.symbol}", "INFO")
        self._log(f"    Contract: {position.contract_symbol}", "INFO")
        self._log(f"    Entry: ${position.entry_price:.2f}", "INFO")
        self._log(f"    Exit: ${position.current_price:.2f}", "INFO")
        self._log(f"    P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)", "SUCCESS" if pnl > 0 else "ERROR")
        self._log(f"{'='*60}", "INFO")

        # Execute sell order via Alpaca
        if self.client:
            try:
                order = self.client.submit_option_market_order(
                    symbol=position.contract_symbol,
                    qty=position.quantity,
                    side='sell'
                )
                self._log(f"    Exit order submitted: {order.get('id', 'N/A')}", "SUCCESS")
            except Exception as e:
                self._log(f"    Exit order failed: {e}", "ERROR")
                self._log(f"    [PAPER] Simulating exit...", "INFO")

        # Update daily P&L
        self.daily_pnl += pnl

        # Remove position
        if position.contract_symbol in self.positions:
            del self.positions[position.contract_symbol]

        return pnl

    def _update_position_prices(self):
        """Update current prices for all positions."""
        # In a real implementation, this would fetch live option quotes
        # For now, we simulate price movements based on underlying
        for contract_symbol, position in self.positions.items():
            state = self.symbol_states.get(position.symbol)
            if state and state.bars_1min:
                # Simple simulation: option moves ~50% of underlying move
                underlying_change = (state.bars_1min[-1]['close'] / state.bars_1min[-2]['close'] - 1) if len(state.bars_1min) >= 2 else 0

                if position.option_type == 'call':
                    position.current_price = position.entry_price * (1 + underlying_change * 0.5)
                else:
                    position.current_price = position.entry_price * (1 - underlying_change * 0.5)

    def _analyze_symbol(self, symbol: str) -> Optional[SignalResult]:
        """Analyze a symbol and generate signal."""
        state = self.symbol_states.get(symbol)
        if not state:
            return None

        # Fetch latest bars
        bars_1min, bars_5min = self._fetch_bars(symbol)

        if len(bars_1min) < 30 or len(bars_5min) < 20:
            return None

        # Update state
        state.bars_1min = bars_1min
        state.bars_5min = bars_5min

        # Generate signal using 4-layer strategy
        signal = MultiSymbolOTMStrategy.generate_signal(bars_1min, bars_5min)
        state.last_signal = signal

        return signal

    def _print_status(self):
        """Print current status."""
        weekday = self._get_weekday()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dte, otm_min, otm_max, budget, target_pct, stop_pct = self._get_day_params()

        self._log(f"", "INFO")
        self._log(f"{'='*60}", "INFO")
        self._log(f"    MULTI-SYMBOL OTM DAY TRADING ENGINE", "INFO")
        self._log(f"{'='*60}", "INFO")
        self._log(f"    Day: {day_names[weekday]}", "INFO")
        self._log(f"    DTE: {dte} | Strike: {otm_min}-{otm_max} OTM", "INFO")
        self._log(f"    Budget: ${budget:.0f} | Target: +{target_pct}% | Stop: -{stop_pct}%", "INFO")
        self._log(f"    Symbols: {', '.join(self.config.symbols)}", "INFO")
        self._log(f"    Positions: {len(self.positions)}/{self.config.max_positions}", "INFO")
        self._log(f"    Daily P&L: ${self.daily_pnl:.2f} | Trades: {self.daily_trades}", "INFO")
        self._log(f"{'='*60}", "INFO")

    def run(self):
        """Main trading loop."""
        self.running = True
        self._print_status()

        self._log("Starting Multi-Symbol OTM Engine...", "INFO")
        self._log(f"Trading hours: {self.config.entry_time_start} - {self.config.entry_time_end}", "INFO")
        self._log(f"Force exit: {self.config.force_exit_time}", "INFO")

        scan_interval = 30  # Seconds between scans
        last_scan = datetime.min

        while self.running:
            try:
                now = datetime.now()

                # Check if market is closed (weekend)
                if now.weekday() > 4:
                    self._log("Market closed (weekend). Waiting...", "INFO")
                    time.sleep(60)
                    continue

                # Update position prices
                self._update_position_prices()

                # Check exit conditions for all positions
                for contract_symbol in list(self.positions.keys()):
                    position = self.positions.get(contract_symbol)
                    if position:
                        should_exit, reason = self._check_exit_conditions(position)
                        if should_exit:
                            self._execute_exit(position, reason)

                # Force exit all positions if time
                if self._is_force_exit_time() and self.positions:
                    self._log("Force exit time reached. Closing all positions...", "WARNING")
                    for contract_symbol in list(self.positions.keys()):
                        position = self.positions.get(contract_symbol)
                        if position:
                            self._execute_exit(position, "FORCE_EXIT_TIME")

                # Check if within trading hours for new entries
                if not self._is_trading_hours():
                    current_time = now.time()
                    if current_time < self.config.entry_time_start:
                        self._log(f"Pre-market: Waiting for entry window ({self.config.entry_time_start})...", "INFO")
                    elif current_time > self.config.entry_time_end:
                        self._log(f"Post-entry: Entry window closed at {self.config.entry_time_end}. Monitoring positions only.", "INFO")
                    time.sleep(30)
                    continue

                # Scan for opportunities
                if (now - last_scan).total_seconds() >= scan_interval:
                    last_scan = now

                    self._log(f"", "INFO")
                    self._log(f"{'='*60}", "INFO")
                    self._log(f"Scanning {len(self.config.symbols)} symbols...", "INFO")
                    self._log(f"{'='*60}", "INFO")

                    for symbol in self.config.symbols:
                        # Check if we can open a position
                        can_trade, reason = self._can_open_position(symbol)
                        if not can_trade:
                            self._log(f"  [{symbol}] Skip: {reason}", "INFO")
                            continue

                        # Fetch and analyze symbol
                        self._log(f"  [{symbol}] Fetching data...", "INFO")
                        signal = self._analyze_symbol(symbol)

                        # Get state for logging
                        state = self.symbol_states.get(symbol)
                        bars_1min_count = len(state.bars_1min) if state else 0
                        bars_5min_count = len(state.bars_5min) if state else 0

                        if bars_1min_count == 0:
                            self._log(f"  [{symbol}] No 1-min data available", "WARNING")
                            continue

                        current_price = state.bars_1min[-1]['close'] if state and state.bars_1min else 0
                        self._log(f"  [{symbol}] Price: ${current_price:.2f} | Bars: {bars_1min_count} (1m), {bars_5min_count} (5m)", "INFO")

                        if not signal:
                            self._log(f"  [{symbol}] No signal generated (insufficient data)", "INFO")
                            continue

                        if signal.direction == 'NEUTRAL':
                            # Show why it's neutral
                            l1 = signal.layer_scores.get('layer1', 0)
                            l2 = signal.layer_scores.get('layer2', 0)
                            l3 = signal.layer_scores.get('layer3', 0)
                            self._log(f"  [{symbol}] NEUTRAL - L1:{l1:.2f} L2:{l2:.2f} L3:{l3:.2f}", "INFO")
                            if signal.details.get('reason'):
                                self._log(f"           Reason: {signal.details.get('reason')}", "INFO")
                            continue

                        # Log signal details
                        self._log(f"", "SIGNAL")
                        self._log(f">>> {symbol} SIGNAL: {signal.direction} <<<", "SIGNAL")
                        self._log(f"    Confidence: {signal.confidence:.1%}", "INFO")
                        self._log(f"    L1 (Price Action): {signal.layer_scores.get('layer1', 0):.2f}", "INFO")
                        self._log(f"    L2 (Technical): {signal.layer_scores.get('layer2', 0):.2f}", "INFO")
                        self._log(f"    L3 (Momentum): {signal.layer_scores.get('layer3', 0):.2f}", "INFO")
                        self._log(f"    L4 Pullback: {signal.layer_scores.get('layer4_pullback', 0):.2f}", "INFO")
                        self._log(f"    L4 Recovery: {signal.layer_scores.get('layer4_recovery', 0):.2f}", "INFO")

                        # Get day parameters
                        dte, otm_min, otm_max, budget, target_pct, stop_pct = self._get_day_params()

                        # Get option chain
                        option_chain = self._get_option_chain(symbol, dte)

                        if not option_chain:
                            self._log(f"    No options available for {symbol}", "WARNING")
                            continue

                        # Get current price
                        state = self.symbol_states.get(symbol)
                        if not state or not state.bars_1min:
                            continue

                        current_price = state.bars_1min[-1]['close']

                        # Select strike
                        option = MultiSymbolOTMStrategy.select_strike(
                            current_price,
                            option_chain,
                            signal.direction,
                            otm_min,
                            otm_max
                        )

                        if option:
                            self._execute_entry(symbol, signal, option)

                # Print periodic status
                if now.minute % 5 == 0 and now.second < 5:
                    self._print_status()

                time.sleep(5)

            except KeyboardInterrupt:
                self._log("Shutdown requested...", "WARNING")
                self.running = False
            except Exception as e:
                self._log(f"Error in main loop: {e}", "ERROR")
                time.sleep(10)

        # Cleanup
        self._log("Shutting down...", "INFO")
        if self.positions:
            self._log(f"Closing {len(self.positions)} remaining positions...", "WARNING")
            for contract_symbol in list(self.positions.keys()):
                position = self.positions.get(contract_symbol)
                if position:
                    self._execute_exit(position, "SHUTDOWN")

        self._log(f"Final P&L: ${self.daily_pnl:.2f} | Total Trades: {self.daily_trades}", "INFO")

    def stop(self):
        """Stop the trading engine."""
        self.running = False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Symbol OTM Day Trading Engine')
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    parser.add_argument('-y', '--yes', action='store_true', help='Auto-confirm start')
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        # Try default config path
        default_config_path = os.path.join(
            os.path.expanduser('~'),
            '.thevolumeai',
            'multi_symbol_otm_config.json'
        )
        if os.path.exists(default_config_path):
            config = load_config_from_file(default_config_path)
        else:
            config = MultiSymbolOTMConfig()

    # Display configuration
    print("\n" + "="*60)
    print("    MULTI-SYMBOL OTM DAY TRADING ENGINE")
    print("="*60)
    print(f"    Account: ${config.total_account:.0f}")
    print(f"    Symbols: {', '.join(config.symbols)}")
    print(f"    Max Positions: {config.max_positions}")
    print(f"    Paper Trading: {config.use_paper}")
    print("="*60 + "\n")

    # Confirm start
    if not args.yes:
        response = input("Start trading engine? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create and run engine
    engine = MultiSymbolOTMEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
