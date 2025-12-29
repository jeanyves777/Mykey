#!/usr/bin/env python3
"""
THE VOLUME AI - MARA Continuous Trading Engine

Engine for continuous MARA options trading with:
- ATM contract selection at each entry
- Weekly expiry options
- Volume-based entry validation
- Automatic re-entry after position closes
- Full trade logging with Greeks
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import pytz

from trading_system.strategies.mara_continuous_momentum import (
    MARAContinuousMomentumConfig,
    MARAContinuousMomentumStrategy,
    TechnicalIndicators,
    PriceActionSignal,
    VolumeAnalysis,
    ATMContractInfo
)
from trading_system.engine.alpaca_client import AlpacaClient, Quote
from trading_system.analytics.options_trade_logger import OptionsTradeLogger

EST = pytz.timezone('US/Eastern')


@dataclass
class Position:
    """Active position information."""
    symbol: str = ""
    option_symbol: str = ""
    option_type: str = ""  # 'call' or 'put'
    strike: float = 0.0
    expiration: datetime = None
    entry_price: float = 0.0
    entry_time: datetime = None
    qty: int = 0
    entry_order_id: str = ""
    tp_order_id: str = ""
    entry_underlying_price: float = 0.0
    greeks: Dict[str, float] = field(default_factory=dict)
    # Real-time data from Alpaca
    current_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_dollars: float = 0.0
    exit_price: Optional[float] = None


@dataclass
class TradingSession:
    """Session state for continuous trading."""
    is_running: bool = False
    position: Optional[Position] = None
    trades_today: int = 0
    last_exit_time: Optional[datetime] = None
    daily_pnl: float = 0.0
    winners: int = 0
    losers: int = 0


class MARAContinuousTradingEngine:
    """
    Continuous trading engine for MARA options.

    Flow:
    1. Monitor MARA price and volume
    2. When volume spike detected + momentum confirmed -> select ATM contract
    3. Enter position with TP limit order
    4. Monitor for TP/SL/max hold
    5. After exit, wait cooldown then repeat
    """

    def __init__(
        self,
        config: MARAContinuousMomentumConfig,
        api_key: str,
        api_secret: str
    ):
        self.config = config
        self.strategy = MARAContinuousMomentumStrategy(config)
        self.client = AlpacaClient(api_key, api_secret, paper=True)
        self.trade_logger = OptionsTradeLogger()

        self.session = TradingSession()
        self.latest_underlying_quote: Optional[Quote] = None
        self.latest_option_quote: Optional[Quote] = None

        # Technical data storage
        self.bars_1min: List[Dict] = []
        self.bars_5min: List[Dict] = []

    def _log(self, message: str, level: str = "INFO"):
        """Print timestamped log message."""
        timestamp = datetime.now(EST).strftime("%H:%M:%S")
        prefix = {
            "INFO": "",
            "TRADE": "[TRADE]",
            "WARN": "[WARN]",
            "ERROR": "[ERROR]",
            "SIGNAL": "[SIGNAL]"
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EST)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def _fetch_bars(self) -> bool:
        """Fetch 1-min and 5-min bars for technical analysis."""
        try:
            # Get 1-minute bars (last 30 bars)
            self.bars_1min = self.client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe="1Min",
                limit=30
            )

            # Get 5-minute bars (last 10 bars)
            self.bars_5min = self.client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe="5Min",
                limit=10
            )

            return len(self.bars_1min) >= 20 and len(self.bars_5min) >= 5

        except Exception as e:
            self._log(f"Error fetching bars: {e}", "ERROR")
            return False

    def _calculate_indicators(self) -> TechnicalIndicators:
        """Calculate technical indicators from 1-min bars."""
        indicators = TechnicalIndicators()

        if len(self.bars_1min) < 20:
            return indicators

        closes = [b.close for b in self.bars_1min]
        volumes = [b.volume for b in self.bars_1min]
        highs = [b.high for b in self.bars_1min]
        lows = [b.low for b in self.bars_1min]

        # EMA calculations
        indicators.ema_9 = self._ema(closes, 9)
        indicators.ema_20 = self._ema(closes, 20)

        # VWAP (simplified - cumulative)
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        cum_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
        cum_vol = sum(volumes)
        indicators.vwap = cum_tp_vol / cum_vol if cum_vol > 0 else closes[-1]

        # RSI
        indicators.rsi = self._rsi(closes, 14)

        # MACD
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        indicators.macd = ema_12 - ema_26
        # Simplified signal line
        indicators.macd_signal = indicators.macd * 0.9

        # Bollinger Bands
        sma_20 = sum(closes[-20:]) / 20
        std_20 = (sum((c - sma_20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        indicators.bb_mid = sma_20
        indicators.bb_upper = sma_20 + 2 * std_20
        indicators.bb_lower = sma_20 - 2 * std_20

        # Volume
        indicators.volume = volumes[-1]
        indicators.avg_volume = sum(volumes) / len(volumes)

        # ATR
        indicators.atr = self._atr(highs, lows, closes, 14)

        return indicators

    def _ema(self, data: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if data else 0

        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        """Calculate ATR."""
        if len(highs) < period + 1:
            return highs[-1] - lows[-1] if highs else 0

        true_ranges = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        return sum(true_ranges[-period:]) / period

    def _find_atm_contract(self, direction: str) -> Optional[ATMContractInfo]:
        """
        Find ATM contract for the given direction.
        Uses weekly expiry and validates liquidity.
        """
        if not self.latest_underlying_quote:
            return None

        current_price = self.latest_underlying_quote.mid
        option_type = "call" if direction == "BULLISH" else "put"

        # Get weekly expiry
        now = datetime.now(EST)
        expiry_date = self.strategy.get_weekly_expiry(now)

        # Calculate days to expiry
        days_to_expiry = (expiry_date.date() - now.date()).days

        # If weekly is 0DTE and we want min 1 DTE, get next week
        if days_to_expiry < self.config.min_days_to_expiry:
            expiry_date = expiry_date + timedelta(days=7)
            days_to_expiry = (expiry_date.date() - now.date()).days

        self._log(f"Looking for {option_type.upper()} ATM, expiry {expiry_date.strftime('%Y-%m-%d')} ({days_to_expiry} DTE)")

        try:
            # Get option chain from Alpaca
            chain = self.client.get_option_chain(
                self.config.underlying_symbol,
                expiry_date.strftime("%Y-%m-%d")
            )

            if not chain:
                self._log("No option chain available", "WARN")
                return None

            # Filter for the option type we want
            options = [o for o in chain if o.get('option_type', '').lower() == option_type]

            if not options:
                self._log(f"No {option_type} options found", "WARN")
                return None

            # Find ATM (closest strike to current price)
            atm_option = min(options, key=lambda o: abs(o['strike'] - current_price))

            # Build contract info
            contract = ATMContractInfo(
                symbol=atm_option.get('symbol', ''),
                strike=atm_option['strike'],
                expiration=expiry_date,
                days_to_expiry=days_to_expiry,
                option_type=option_type,
                bid=atm_option.get('bid', 0),
                ask=atm_option.get('ask', 0),
                mid=(atm_option.get('bid', 0) + atm_option.get('ask', 0)) / 2,
                volume=atm_option.get('volume', 0),
                open_interest=atm_option.get('open_interest', 0),
                delta=atm_option.get('delta', 0.5 if option_type == 'call' else -0.5)
            )

            # Validate contract meets requirements
            contract = self.strategy.validate_contract(contract, current_price)

            if contract.is_valid:
                self._log(f"Found ATM: {contract.symbol} Strike=${contract.strike} "
                         f"Bid=${contract.bid:.2f} Ask=${contract.ask:.2f} "
                         f"Spread={contract.spread_pct:.1f}%")
            else:
                self._log(f"ATM contract rejected: {contract.rejection_reason}", "WARN")

            return contract

        except Exception as e:
            self._log(f"Error finding ATM contract: {e}", "ERROR")
            return None

    def _find_atm_strike_manual(self, direction: str) -> Optional[ATMContractInfo]:
        """
        Manually construct ATM option symbol when chain API not available.
        """
        if not self.latest_underlying_quote:
            return None

        current_price = self.latest_underlying_quote.mid
        option_type = "call" if direction == "BULLISH" else "put"

        # Round to nearest $0.50 or $1 strike (MARA has $0.50 strikes)
        # For MARA around $12, strikes are typically $0.50 apart
        atm_strike = round(current_price * 2) / 2

        # Get weekly expiry
        now = datetime.now(EST)
        expiry_date = self.strategy.get_weekly_expiry(now)
        days_to_expiry = (expiry_date.date() - now.date()).days

        # If 0DTE, get next week
        if days_to_expiry < self.config.min_days_to_expiry:
            expiry_date = expiry_date + timedelta(days=7)
            days_to_expiry = (expiry_date.date() - now.date()).days

        # Build OCC symbol: MARA251212C00012500 (for $12.50 call expiring 12/12/25)
        expiry_str = expiry_date.strftime("%y%m%d")
        opt_type_char = "C" if option_type == "call" else "P"
        strike_int = int(atm_strike * 1000)
        occ_symbol = f"MARA{expiry_str}{opt_type_char}{strike_int:08d}"

        self._log(f"Constructed ATM symbol: {occ_symbol} (Strike ${atm_strike})")

        # Get quote for this option
        try:
            option_quote = self.client.get_latest_option_quote(occ_symbol)

            if option_quote:
                contract = ATMContractInfo(
                    symbol=occ_symbol,
                    strike=atm_strike,
                    expiration=expiry_date,
                    days_to_expiry=days_to_expiry,
                    option_type=option_type,
                    bid=option_quote.bid,
                    ask=option_quote.ask,
                    mid=option_quote.mid,
                    volume=0,  # Not available from quote
                    open_interest=0,
                    delta=0.5 if option_type == "call" else -0.5
                )

                # Calculate spread
                if contract.mid > 0:
                    contract.spread_pct = (contract.ask - contract.bid) / contract.mid * 100

                # Simple validation
                if contract.mid > 0 and contract.spread_pct < self.config.max_bid_ask_spread_pct:
                    contract.is_valid = True
                    self._log(f"ATM valid: ${contract.strike} Mid=${contract.mid:.2f} "
                             f"Spread={contract.spread_pct:.1f}%")
                else:
                    contract.is_valid = False
                    contract.rejection_reason = f"Spread too wide ({contract.spread_pct:.1f}%)"

                return contract
            else:
                self._log(f"Could not get quote for {occ_symbol}", "WARN")
                return None

        except Exception as e:
            self._log(f"Error getting ATM quote: {e}", "ERROR")
            return None

    def _check_pullback_htf(self, direction: str) -> dict:
        """
        V4 PULLBACK DETECTION LAYER using 5-minute bars (Higher TimeFrame).

        Uses 5-min bars instead of 1-min to reduce noise and give more reliable signals.
        - For BULLISH: Wait for dip/pullback + recovery (green candle)
        - For BEARISH: Wait for bounce/rally + rejection (red candle)

        Returns:
            dict with ready_to_enter, pullback_detected, recovery_detected, reasons
        """
        # We already have bars_5min from _fetch_bars(), use them
        if not self.bars_5min or len(self.bars_5min) < 10:
            self._log("  Not enough 5-min bars for pullback analysis", "WARN")
            # If we don't have enough data, allow entry to not miss opportunities
            return {
                'ready_to_enter': True,
                'pullback_detected': False,
                'recovery_detected': False,
                'pullback_score': 0,
                'recovery_score': 0,
                'reasons': ['Not enough data - allowing entry']
            }

        try:
            # Use the strategy's pullback detection method (5-min HTF version)
            result = MARAContinuousMomentumStrategy.check_pullback_entry_htf(self.bars_5min, direction)

            # Log the analysis
            self._log(f"  Direction: {direction}", "INFO")
            indicators = result.get('indicators', {})
            if indicators:
                self._log(f"  Price: ${indicators.get('price', 0):.2f} | RSI: {indicators.get('rsi', 0):.1f}")
                self._log(f"  VWAP: ${indicators.get('vwap', 0):.2f} | Pullback from high: {indicators.get('pullback_from_high_pct', 0):.2f}%")

            return result

        except Exception as e:
            self._log(f"  Error in pullback check: {e}", "ERROR")
            # On error, allow entry to not miss opportunities
            return {
                'ready_to_enter': True,
                'pullback_detected': False,
                'recovery_detected': False,
                'pullback_score': 0,
                'recovery_score': 0,
                'reasons': [f'Error: {e} - allowing entry']
            }

    def _enter_position(self, contract: ATMContractInfo, direction: str) -> bool:
        """Enter a position with the selected contract."""
        if not contract.is_valid:
            return False

        # Check available buying power FIRST
        try:
            account = self.client.get_account()
            buying_power = float(account.get('buying_power', 0))
            self._log(f"Available buying power: ${buying_power:.2f}")

            if buying_power < 50:
                self._log(f"INSUFFICIENT BUYING POWER (${buying_power:.2f}) - cannot enter position", "WARN")
                self._log("Check Alpaca for existing positions or pending orders", "WARN")
                return False
        except Exception as e:
            self._log(f"Could not check buying power: {e}", "WARN")

        # Calculate position size based on available buying power
        # Use the smaller of configured position value or available buying power
        available_for_trade = min(self.config.fixed_position_value, buying_power * 0.95)  # Keep 5% buffer

        if contract.mid <= 0:
            self._log("Invalid contract price", "WARN")
            return False

        # Calculate how many contracts we can afford
        cost_per_contract = contract.mid * 100  # Options are 100 shares per contract
        max_qty = int(available_for_trade / cost_per_contract)

        if max_qty <= 0:
            self._log(f"Cannot afford any contracts (need ${cost_per_contract:.2f}, have ${available_for_trade:.2f})", "WARN")
            return False

        # Use strategy's position size but cap at what we can afford
        qty = min(self.strategy.calculate_position_size(contract.mid), max_qty)
        if qty <= 0:
            self._log("Position size too small", "WARN")
            return False

        self._log(f"Position sizing: ${available_for_trade:.2f} available -> {qty} contracts @ ${contract.mid:.2f}")

        self._log("=" * 70, "TRADE")
        self._log(f"ENTERING {direction} POSITION", "TRADE")
        self._log(f"  Contract: {contract.symbol}", "TRADE")
        self._log(f"  Strike: ${contract.strike}", "TRADE")
        self._log(f"  Expiry: {contract.expiration.strftime('%Y-%m-%d')} ({contract.days_to_expiry} DTE)", "TRADE")
        self._log(f"  Type: {contract.option_type.upper()}", "TRADE")
        self._log(f"  Mid Price: ${contract.mid:.2f}", "TRADE")
        self._log(f"  Quantity: {qty} contracts", "TRADE")
        self._log(f"  Est. Value: ${contract.mid * qty * 100:.2f}", "TRADE")

        # Submit market buy order
        try:
            order_result = self.client.submit_option_order(
                symbol=contract.symbol,
                qty=qty,
                side="buy",
                order_type="market"
            )

            if not order_result or not order_result.get('success'):
                self._log(f"Order failed: {order_result}", "ERROR")
                return False

            order_id = order_result.get('order_id', '')
            # Get fill price - try multiple possible keys
            actual_fill_price = (
                order_result.get('filled_avg_price') or
                order_result.get('avg_fill_price') or
                order_result.get('filled_price') or
                contract.mid
            )
            # Ensure it's a float
            if actual_fill_price is None:
                actual_fill_price = contract.mid
            actual_fill_price = float(actual_fill_price)

            self._log(f"  Order ID: {order_id}", "TRADE")
            self._log(f"  Fill Price: ${actual_fill_price:.2f}", "TRADE")
            self._log("  >>> POSITION OPENED - Monitoring for TP/SL <<<", "SUCCESS")

            # Calculate and place TP limit order
            tp_price = round(actual_fill_price * (1 + self.config.target_profit_pct / 100), 2)
            self._log(f"  TP Target: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", "TRADE")

            tp_result = self.client.submit_option_order(
                symbol=contract.symbol,
                qty=qty,
                side="sell",
                order_type="limit",
                limit_price=tp_price
            )

            tp_order_id = tp_result.get('order_id', '') if tp_result else ''

            # Create position object
            self.session.position = Position(
                symbol=self.config.underlying_symbol,
                option_symbol=contract.symbol,
                option_type=contract.option_type,
                strike=contract.strike,
                expiration=contract.expiration,
                entry_price=actual_fill_price,
                entry_time=datetime.now(EST),
                qty=qty,
                entry_order_id=order_id,
                tp_order_id=tp_order_id,
                entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0
            )

            self.session.trades_today += 1

            # Fetch and log Greeks
            try:
                greeks = self.client.get_option_greeks(contract.symbol)
                if greeks:
                    self.session.position.greeks = {
                        'delta': greeks.delta,
                        'gamma': greeks.gamma,
                        'theta': greeks.theta,
                        'vega': greeks.vega,
                        'iv': greeks.implied_volatility
                    }
                    self._log(f"  Greeks: Δ={greeks.delta:.3f} Γ={greeks.gamma:.4f} "
                             f"Θ={greeks.theta:.3f} V={greeks.vega:.3f} IV={greeks.implied_volatility:.1%}", "TRADE")
            except Exception as ge:
                self._log(f"  Could not fetch Greeks: {ge}", "WARN")

            # Log to trade logger
            try:
                trade_id = f"CONT_MARA_{contract.symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"
                self.trade_logger.log_entry(
                    trade_id=trade_id,
                    underlying_symbol=self.config.underlying_symbol,
                    option_symbol=contract.symbol,
                    option_type=contract.option_type,
                    strike_price=contract.strike,
                    expiration_date=contract.expiration.strftime("%Y-%m-%d"),
                    entry_time=datetime.now(EST),
                    entry_price=actual_fill_price,
                    entry_qty=qty,
                    entry_order_id=order_id,
                    entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0,
                    greeks=self.session.position.greeks,
                    target_profit_pct=self.config.target_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    notes=f"MARA CONTINUOUS | {direction} | {contract.days_to_expiry} DTE"
                )
                self.session.position.entry_order_id = trade_id
            except Exception as le:
                self._log(f"  Trade log error: {le}", "WARN")

            self._log("=" * 70, "TRADE")
            return True

        except Exception as e:
            self._log(f"Error entering position: {e}", "ERROR")
            return False

    def _check_position(self) -> Optional[str]:
        """
        Check position for exit conditions.
        Returns exit reason or None if still holding.

        FETCHES POSITION DATA DIRECTLY FROM ALPACA (like COIN engine).
        """
        if not self.session.position:
            return None

        pos = self.session.position
        now = datetime.now(EST)

        # Check if TP limit order was filled FIRST
        if pos.tp_order_id:
            try:
                order_status = self.client.get_order_status(pos.tp_order_id)
                if order_status and order_status.get('status') == 'filled':
                    # Store fill price for exit
                    fill_price = order_status.get('filled_avg_price')
                    if fill_price:
                        pos.exit_price = float(fill_price)
                    return "TAKE_PROFIT"
            except:
                pass

        # FETCH POSITION DATA DIRECTLY FROM ALPACA for accurate real-time data
        # (Like COIN engine does - use direct symbol lookup)
        alpaca_pos = self.client.get_position_by_symbol(pos.option_symbol)

        # If position is gone from Alpaca but we have a TP order, check if it filled
        if not alpaca_pos and pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                tp_status = tp_order.get('status', 'unknown') if tp_order else 'unknown'

                if tp_status == 'filled':
                    fill_price = float(tp_order.get('filled_avg_price', pos.entry_price * 1.075))
                    pos.exit_price = fill_price
                    self._log(f"TP LIMIT ORDER FILLED @ ${fill_price:.2f}!", "SUCCESS")
                    return "TAKE_PROFIT"
                elif tp_status in ['canceled', 'expired', 'rejected']:
                    self._log(f"Position gone, TP order {tp_status}", "WARN")
                    self.session.position = None
                    return None
            except Exception as e:
                self._log(f"Error checking TP order: {e}", "WARN")

        # Use Alpaca data if available (like COIN does)
        if alpaca_pos:
            current_price = float(alpaca_pos.get('current_price', 0))
            pnl_pct = float(alpaca_pos.get('unrealized_plpc', 0))
            pnl_dollars = float(alpaca_pos.get('unrealized_pl', 0))
            entry_price = float(alpaca_pos.get('avg_entry_price', pos.entry_price))
            market_value = float(alpaca_pos.get('market_value', 0))
            cost_basis = float(alpaca_pos.get('cost_basis', 0))

            # Update local position with Alpaca's actual entry price
            if abs(entry_price - pos.entry_price) > 0.01:
                self._log(f"Updating entry from Alpaca: ${pos.entry_price:.2f} -> ${entry_price:.2f}", "INFO")
                pos.entry_price = entry_price

            # Store current data for exit
            pos.current_price = current_price
            pos.pnl_pct = pnl_pct
            pos.pnl_dollars = pnl_dollars

            # Log status with Alpaca data
            self._log(f"[ALPACA] {pos.option_symbol} Entry=${entry_price:.2f} Current=${current_price:.2f} "
                     f"P&L=${pnl_dollars:+.2f} ({pnl_pct:+.1f}%) MktVal=${market_value:.2f}")

            # Check stop loss using Alpaca P&L
            if pnl_pct <= -self.config.stop_loss_pct:
                return "STOP_LOSS"

        else:
            # Fallback to quote-based pricing if Alpaca position not found
            try:
                option_quote = self.client.get_latest_option_quote(pos.option_symbol)
                if option_quote:
                    self.latest_option_quote = option_quote
                    current_price = option_quote.mid
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                    pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

                    pos.current_price = current_price
                    pos.pnl_pct = pnl_pct
                    pos.pnl_dollars = pnl_dollars

                    self._log(f"[QUOTE] {pos.option_symbol} Entry=${pos.entry_price:.2f} Current=${current_price:.2f} "
                             f"P&L=${pnl_dollars:+.2f} ({pnl_pct:+.1f}%)")

                    if pnl_pct <= -self.config.stop_loss_pct:
                        return "STOP_LOSS"
            except Exception as e:
                self._log(f"Error getting quote: {e}", "WARN")

        # Check max hold time
        hold_minutes = (now - pos.entry_time).total_seconds() / 60
        if hold_minutes >= self.config.max_hold_minutes:
            return "MAX_HOLD_TIME"

        # Check force exit time
        force_exit = datetime.strptime(self.config.force_exit_time, "%H:%M:%S").time()
        if now.time() >= force_exit:
            return "FORCE_EXIT"

        return None

    def _exit_position(self, reason: str) -> bool:
        """Exit the current position."""
        if not self.session.position:
            return False

        pos = self.session.position
        now = datetime.now(EST)

        self._log("=" * 70, "TRADE")
        self._log(f"EXITING POSITION - {reason}", "TRADE")

        # Cancel TP order if exists and not filled
        if pos.tp_order_id and reason != "TAKE_PROFIT":
            try:
                self.client.cancel_order(pos.tp_order_id)
                self._log(f"  Cancelled TP order: {pos.tp_order_id}", "TRADE")
            except:
                pass

        # Get exit price - use stored exit_price if available (e.g., from TP fill)
        exit_price = pos.exit_price if pos.exit_price else pos.current_price
        exit_order_id = ""

        # Use stored Alpaca P&L data if available (from _check_position)
        pnl_dollars_from_alpaca = pos.pnl_dollars if pos.pnl_dollars else None
        pnl_pct_from_alpaca = pos.pnl_pct if pos.pnl_pct else None

        if reason == "TAKE_PROFIT":
            # TP was filled, get fill price from order
            try:
                order_status = self.client.get_order_status(pos.tp_order_id)
                if order_status:
                    fill_price = order_status.get('filled_avg_price')
                    if fill_price is not None:
                        exit_price = float(fill_price)
                    exit_order_id = pos.tp_order_id
            except:
                # Fallback to expected TP price
                if not exit_price:
                    exit_price = pos.entry_price * (1 + self.config.target_profit_pct / 100)
        else:
            # Market sell
            try:
                sell_result = self.client.submit_option_order(
                    symbol=pos.option_symbol,
                    qty=pos.qty,
                    side="sell",
                    order_type="market"
                )
                if sell_result and sell_result.get('success'):
                    fill_price = sell_result.get('filled_avg_price')
                    if fill_price is not None:
                        exit_price = float(fill_price)
                    exit_order_id = sell_result.get('order_id', '')
                    self._log(f"  Sell order placed: {exit_order_id}", "TRADE")
            except Exception as e:
                self._log(f"  Error selling: {e}", "ERROR")

        # Final fallback - try to get current quote
        if not exit_price or exit_price <= 0:
            try:
                current_quote = self.client.get_latest_option_quote(pos.option_symbol)
                if current_quote:
                    exit_price = current_quote.mid
            except:
                pass

        # Last resort fallback
        if not exit_price or exit_price <= 0:
            exit_price = pos.entry_price
            self._log(f"  Warning: Using entry price as exit price fallback", "WARN")

        # Use Alpaca P&L if available, otherwise calculate
        if pnl_dollars_from_alpaca is not None and pnl_pct_from_alpaca is not None:
            gross_pnl = pnl_dollars_from_alpaca
            pnl_pct = pnl_pct_from_alpaca
            fees = pos.qty * 1.30  # Estimated fees
            net_pnl = gross_pnl - fees
            self._log(f"  [ALPACA P&L] Gross: ${gross_pnl:+.2f} ({pnl_pct:+.1f}%)", "TRADE")
        else:
            # Calculate P&L manually as fallback
            gross_pnl = (exit_price - pos.entry_price) * pos.qty * 100
            fees = pos.qty * 1.30  # Estimated fees
            net_pnl = gross_pnl - fees
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0

        hold_minutes = (now - pos.entry_time).total_seconds() / 60

        self._log(f"  Exit Price: ${exit_price:.2f}", "TRADE")
        self._log(f"  Gross P&L: ${gross_pnl:+.2f}", "TRADE")
        self._log(f"  Net P&L: ${net_pnl:+.2f} ({pnl_pct:+.1f}%)", "TRADE")
        self._log(f"  Hold Time: {hold_minutes:.1f} minutes", "TRADE")

        # Update session stats
        self.session.daily_pnl += net_pnl
        if net_pnl > 0:
            self.session.winners += 1
        else:
            self.session.losers += 1

        # Log trade exit
        try:
            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=now,
                exit_price=exit_price,
                exit_qty=pos.qty,
                exit_order_id=exit_order_id,
                exit_reason=reason,
                exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0,
                fees_paid=fees,
                notes=f"MARA CONTINUOUS | EXIT: {reason} - Hold time: {hold_minutes:.1f}m"
            )
        except Exception as le:
            self._log(f"  Trade log error: {le}", "WARN")

        self._log("=" * 70, "TRADE")

        # Record exit for cooldown
        self.strategy.record_trade_exit(now)
        self.session.last_exit_time = now
        self.session.position = None

        return True

    def _print_status(self):
        """Print current status summary."""
        now = datetime.now(EST)
        in_cooldown, cooldown_mins = self.strategy.is_in_cooldown(now)

        status = "IN POSITION" if self.session.position else ("COOLDOWN" if in_cooldown else "SCANNING")

        print()
        self._log(f"Status: {status} | Trades: {self.session.trades_today} | "
                 f"P&L: ${self.session.daily_pnl:+.2f} | W/L: {self.session.winners}/{self.session.losers}")

        if in_cooldown:
            self._log(f"  Cooldown: {cooldown_mins}m remaining")

        if self.latest_underlying_quote:
            self._log(f"  MARA: ${self.latest_underlying_quote.mid:.2f}")
        print()

    def _resume_existing_positions(self):
        """Check for and resume any existing MARA options positions from Alpaca.

        Uses multiple methods like COIN engine:
        1. Direct positions API (REST + SDK)
        2. Open SELL orders as backup (TP orders indicate position exists)
        3. Cancel orphaned TP orders if position is gone
        """
        self._log("Checking for existing MARA options positions...")
        try:
            # Method 0: Try direct REST API first (most reliable)
            self._log("  Trying direct REST API...")
            all_positions = self.client.get_positions_raw()

            # Method 1: Also try SDK if REST returned nothing
            if not all_positions:
                self._log("  REST returned nothing, trying SDK...")
                all_positions = self.client.get_positions()

            # METHOD 2: If positions API is empty, check open SELL orders as backup
            # (Like COIN engine does - if there's a TP SELL order, position MUST exist)
            if not all_positions:
                self._log("  Positions API empty - checking open orders as backup...", "INFO")
                try:
                    open_orders = self.client.get_open_orders("MARA")
                    self._log(f"  Found {len(open_orders)} open orders for MARA")

                    for order in open_orders:
                        symbol = order.get('symbol', '')
                        side = order.get('side', '')
                        order_id = order.get('id', '')

                        # Look for SELL orders (these are TP orders) - position MUST exist!
                        if side == 'sell' and 'MARA' in symbol and len(symbol) > 10:
                            self._log(f"  FOUND TP ORDER (position must exist): {symbol}", "WARN")
                            self._log(f"    Order ID: {order_id}", "WARN")
                            self._log(f"    Limit Price: ${float(order.get('limit_price', 0)):.2f}", "WARN")
                            self._log(f"    Qty: {order.get('qty')}", "WARN")

                            # Recover position from TP order info (like COIN does)
                            qty = int(float(order.get('qty', 0)))
                            tp_price = float(order.get('limit_price', 0))
                            # Estimate entry price from TP price (TP = entry * 1.075)
                            estimated_entry = tp_price / (1 + self.config.target_profit_pct / 100)

                            # Parse OCC symbol for strike and expiry
                            # Format: MARA251212P00012000
                            try:
                                opt_type = 'put' if 'P' in symbol[10:12] else 'call'
                                strike = int(symbol[-8:]) / 1000
                                expiry_str = symbol[4:10]  # YYMMDD
                                expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
                            except Exception:
                                opt_type = 'put'
                                strike = 12.0
                                expiry = datetime.now(EST)

                            # Create position object from TP order
                            self.session.position = Position(
                                symbol=self.config.underlying_symbol,
                                option_symbol=symbol,
                                option_type=opt_type,
                                strike=strike,
                                expiration=expiry,
                                entry_price=estimated_entry,
                                entry_time=datetime.now(EST),  # Approximation
                                qty=qty,
                                entry_order_id='recovered',
                                tp_order_id=order_id,  # Keep the existing TP order!
                                entry_underlying_price=0
                            )

                            self._log("=" * 70, "SUCCESS")
                            self._log(f">>> POSITION RECOVERED FROM TP ORDER", "SUCCESS")
                            self._log(f"    Symbol: {symbol}", "SUCCESS")
                            self._log(f"    Type: {opt_type.upper()} | Strike: ${strike:.2f}", "SUCCESS")
                            self._log(f"    Qty: {qty} | Est. Entry: ${estimated_entry:.2f}", "SUCCESS")
                            self._log(f"    TP: ${tp_price:.2f} | SL: ${estimated_entry * (1 - self.config.stop_loss_pct / 100):.2f}", "SUCCESS")
                            self._log(f"    TP Order ID: {order_id}", "SUCCESS")
                            self._log("=" * 70, "SUCCESS")
                            self._log(">>> RESUMING POSITION MANAGEMENT <<<", "SUCCESS")

                            self.session.trades_today = 1
                            return True

                except Exception as e:
                    self._log(f"  Error checking open orders: {e}", "WARN")

            self._log(f"  Found {len(all_positions)} total positions (all asset classes)")

            # Log what we see for debugging
            for p in all_positions:
                self._log(f"    Position: {p.get('symbol', '?')} qty={p.get('qty', 0)}")

            # Method 2: Try direct symbol lookup for specific MARA options
            # Generate possible symbols based on current date
            now = datetime.now(EST)
            direct_symbols = []

            # Check this week and next week
            for week_offset in [0, 7]:
                # Find Friday
                days_until_friday = (4 - now.weekday()) % 7
                if days_until_friday == 0 and now.hour >= 16:
                    days_until_friday = 7
                expiry = now + timedelta(days=days_until_friday + week_offset)
                expiry_str = expiry.strftime("%y%m%d")

                # Check common strikes around $12 (MARA's typical price)
                for strike in [11.0, 11.5, 12.0, 12.5, 13.0]:
                    strike_int = int(strike * 1000)
                    for opt_type in ['P', 'C']:
                        symbol = f"MARA{expiry_str}{opt_type}{strike_int:08d}"
                        direct_symbols.append(symbol)

            self._log(f"  Checking {len(direct_symbols)} possible MARA option symbols...")

            mara_positions = []

            # First check all_positions for MARA symbols
            for pos in all_positions:
                symbol = pos.get('symbol', '')
                if symbol.startswith('MARA') and len(symbol) > 10:  # Option symbols are long
                    qty = int(float(pos.get('qty', 0)))  # Convert string to int
                    if qty > 0:
                        pos['qty'] = qty  # Update with converted value
                        mara_positions.append(pos)
                        self._log(f"    Found in all_positions: {symbol}", "SUCCESS")

            # Also try direct lookup for each potential symbol
            for sym in direct_symbols:
                try:
                    direct_pos = self.client.get_position_by_symbol(sym)
                    if direct_pos:
                        qty = int(float(direct_pos.get('qty', 0)))  # Convert string to int
                        if qty > 0:
                            direct_pos['qty'] = qty  # Update with converted value
                            # Check not already in list
                            if not any(p.get('symbol') == sym for p in mara_positions):
                                mara_positions.append(direct_pos)
                                self._log(f"    Found via direct lookup: {sym}", "SUCCESS")
                except Exception as e:
                    pass  # Symbol not found is expected for most

            # Also try to get options positions specifically
            try:
                opt_positions = self.client.get_options_positions()
                self._log(f"  get_options_positions() returned {len(opt_positions)} positions")
                for pos in opt_positions:
                    symbol = pos.get('symbol', '')
                    if symbol.startswith('MARA'):
                        if not any(p.get('symbol') == symbol for p in mara_positions):
                            mara_positions.append(pos)
                            self._log(f"    Found via options API: {symbol}", "SUCCESS")
            except Exception as e:
                self._log(f"  get_options_positions error: {e}", "WARN")

            # Process any MARA positions found
            for pos in mara_positions:
                symbol = pos.get('symbol', '')
                qty = int(float(pos.get('qty', 0)))  # Ensure qty is int

                if qty <= 0:
                    continue  # Skip closed positions

                entry_price = float(pos.get('avg_entry_price', 0))
                current_price = float(pos.get('current_price', entry_price))
                market_value = float(pos.get('market_value', 0))
                cost_basis = float(pos.get('cost_basis', entry_price * qty * 100))

                # P&L from Alpaca (already normalized by get_positions_raw)
                # unrealized_pl is dollar P&L, unrealized_plpc is percentage
                unrealized_pnl = float(pos.get('unrealized_pl', 0))
                pnl_pct = float(pos.get('unrealized_plpc', 0))

                # Parse option type from symbol (C for call, P for put after date)
                # Symbol format: MARA251212P00012000
                opt_type = 'put' if 'P' in symbol[10:12] else 'call'

                # Parse strike from symbol (last 8 chars / 1000)
                try:
                    strike = int(symbol[-8:]) / 1000
                except:
                    strike = 0

                self._log("=" * 70)
                self._log(f">>> FOUND EXISTING POSITION: {symbol}", "SUCCESS")
                self._log(f"    Type: {opt_type.upper()} | Strike: ${strike:.2f}", "SUCCESS")
                self._log(f"    Qty: {qty} | Entry: ${entry_price:.2f} | Current: ${current_price:.2f}", "SUCCESS")
                self._log(f"    Cost Basis: ${cost_basis:.2f} | Market Value: ${market_value:.2f}", "SUCCESS")
                self._log(f"    P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.1f}%)", "SUCCESS")
                self._log("=" * 70)

                # Create position object to track it
                self.session.position = Position(
                    symbol=self.config.underlying_symbol,
                    option_symbol=symbol,
                    option_type=opt_type,
                    strike=strike,
                    expiration=datetime.now(EST),  # Placeholder
                    entry_price=float(entry_price),
                    entry_time=datetime.now(EST),  # We don't know actual entry time
                    qty=int(qty),
                    entry_order_id='resumed',
                    tp_order_id='',
                    entry_underlying_price=0
                )
                self._log(">>> RESUMING POSITION MANAGEMENT <<<", "SUCCESS")
                return True

            # Check buying power to see if funds are tied up (indicates position exists)
            try:
                account = self.client.get_account()
                buying_power = float(account.get('buying_power', 0))
                cash = float(account.get('cash', 0))
                equity = float(account.get('equity', 0))
                self._log(f"  Account: Cash=${cash:.2f} | BP=${buying_power:.2f} | Equity=${equity:.2f}")

                # If buying power is very low but we didn't find a position,
                # something is wrong - maybe check recent orders
                if buying_power < 50:
                    self._log("  >>> LOW BUYING POWER - checking recent orders...", "WARN")
                    try:
                        recent_orders = self.client.get_orders(status='all', limit=10)
                        for order in recent_orders:
                            if 'MARA' in order.get('symbol', ''):
                                self._log(f"    Recent order: {order.get('symbol')} {order.get('side')} "
                                         f"qty={order.get('qty')} status={order.get('status')}", "WARN")
                    except:
                        pass
            except Exception as e:
                self._log(f"  Account check error: {e}", "WARN")

            self._log("No existing MARA positions found.")
            return False
        except Exception as e:
            self._log(f"Error checking positions: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False

    def run(self):
        """Main trading loop."""
        self._log("Starting MARA Continuous Trading Engine")
        self._log(f"  Position Size: ${self.config.fixed_position_value}")
        self._log(f"  TP: {self.config.target_profit_pct}% | SL: {self.config.stop_loss_pct}%")
        self._log(f"  Cooldown: {self.config.cooldown_minutes}m between trades")
        self._log(f"  Max Trades/Day: {self.config.max_trades_per_day}")
        self._log(f"  Expiry: Weekly (min {self.config.min_days_to_expiry} DTE)")

        # Check for existing positions on startup
        self._resume_existing_positions()

        self.session.is_running = True
        poll_count = 0

        while self.session.is_running:
            try:
                now = datetime.now(EST)

                # Check market hours
                if not self._is_market_open():
                    self._log("Market closed. Waiting...")
                    time.sleep(60)
                    continue

                # Get underlying quote
                self.latest_underlying_quote = self.client.get_latest_stock_quote(
                    self.config.underlying_symbol
                )

                if not self.latest_underlying_quote:
                    self._log("Could not get MARA quote", "WARN")
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                # If we have a position, check it
                if self.session.position:
                    exit_reason = self._check_position()
                    if exit_reason:
                        self._exit_position(exit_reason)
                else:
                    # No position - look for entry
                    can_trade, reason = self.strategy.can_trade(now)

                    if can_trade:
                        # Fetch technical data
                        if self._fetch_bars():
                            indicators = self._calculate_indicators()

                            # Analyze price action (verbose=True to show PA breakdown)
                            price_action = self.strategy.analyze_price_action(self.bars_5min, verbose=True)

                            # Analyze volume
                            volume_analysis = self.strategy.analyze_volume(
                                indicators.volume,
                                indicators.avg_volume
                            )

                            # Get entry signal
                            should_enter, direction, opt_type, reasons = self.strategy.get_entry_signal(
                                self.latest_underlying_quote.mid,
                                indicators,
                                price_action,
                                volume_analysis,
                                now
                            )

                            # Always log signal analysis for debugging
                            tech_score, tech_dir, _ = self.strategy.calculate_technical_score(
                                self.latest_underlying_quote.mid, indicators
                            )
                            self._log(f"MARA ${self.latest_underlying_quote.mid:.2f} | "
                                     f"Tech: {tech_dir}({tech_score}) | "
                                     f"PA: {price_action.direction}({price_action.score}) | "
                                     f"Vol: {volume_analysis.trend}({volume_analysis.volume_ratio:.1f}x)")

                            if should_enter:
                                self._log(f"V3 SIGNAL: {direction} -> {'CALLS' if direction == 'BULLISH' else 'PUTS'}", "SIGNAL")
                                self._log(f"  Reasons: {', '.join(reasons)}", "SIGNAL")

                                # ========== V4 PULLBACK DETECTION LAYER (5-MIN HTF) ==========
                                self._log("-" * 50)
                                self._log("V4 PULLBACK DETECTION (5-min HTF)", "INFO")

                                pullback_result = self._check_pullback_htf(direction)

                                if pullback_result['ready_to_enter']:
                                    self._log(f"  ✅ PULLBACK CONDITIONS MET - Entering {direction}", "SUCCESS")

                                    # Find ATM contract
                                    contract = self._find_atm_strike_manual(direction)

                                    if contract and contract.is_valid:
                                        self._enter_position(contract, direction)
                                    else:
                                        self._log("Could not find valid ATM contract", "WARN")
                                else:
                                    self._log(f"  ⏳ WAITING FOR BETTER ENTRY...", "WARN")
                                    self._log(f"     Pullback: {'YES' if pullback_result['pullback_detected'] else 'NO'} (score: {pullback_result['pullback_score']})")
                                    self._log(f"     Recovery: {'YES' if pullback_result['recovery_detected'] else 'NO'} (score: {pullback_result['recovery_score']})")
                                    for reason in pullback_result.get('reasons', [])[-5:]:
                                        self._log(f"       - {reason}")
                            else:
                                if direction in ["CONFLICT", "NEUTRAL", "NONE"]:
                                    self._log(f"  No entry: {direction} - {', '.join(reasons)}")
                        else:
                            self._log("Could not fetch bars", "WARN")
                    else:
                        if poll_count % 6 == 0:  # Every minute
                            self._log(f"Cannot trade: {reason}")

                # Print status periodically
                poll_count += 1
                if poll_count % 12 == 0:  # Every 2 minutes
                    self._print_status()

                time.sleep(self.config.poll_interval_seconds)

            except KeyboardInterrupt:
                self._log("Shutdown requested")
                break
            except Exception as e:
                self._log(f"Error in main loop: {e}", "ERROR")
                time.sleep(self.config.poll_interval_seconds)

        # Cleanup
        if self.session.position:
            self._log("Exiting remaining position on shutdown")
            self._exit_position("SHUTDOWN")

        self._log("Engine stopped")
        self._print_status()

    def stop(self):
        """Stop the trading engine."""
        self.session.is_running = False
