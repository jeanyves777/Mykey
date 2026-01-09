"""
Live M1 Scalping Engine for OANDA
==================================

Executes the optimized scalping strategy on OANDA.
Each pair trades ONLY during its optimal session.

Strategy:
- ADX > 25 (trending market)
- ATR > 1.2x average (volatility expansion)
- DI difference > 5 (strong directional move)
- HTF trend confirmation (EMA50 vs EMA200)
- TP: 15 pips, SL: 10 pips
- Trailing stop: activates at +8 pips, trails 3 pips behind

Usage:
    python -m trading_system.Forex_Trading.Forex_Trading_ML_Version.engine.live_scalping_engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from trading_system.Forex_Trading.Forex_Trading_ML_Version.config.scalping_config import (
    ScalpingConfig, PairConfig, load_scalping_config, print_scalping_config
)


@dataclass
class Position:
    """Active position tracking."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    units: float
    trade_id: str
    entry_time: datetime
    tp_price: float
    sl_price: float
    trailing_active: bool = False
    trailing_sl: Optional[float] = None
    peak_profit_pips: float = 0.0


class OandaAPI:
    """OANDA REST API wrapper."""

    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.base_url = config.base_url
        self.account_id = config.account_id
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        }

    def get_candles(self, symbol: str, count: int = 500, granularity: str = 'M1') -> pd.DataFrame:
        """Fetch candle data from OANDA."""
        url = f"{self.base_url}/v3/instruments/{symbol}/candles"
        params = {
            'count': count,
            'granularity': granularity,
            'price': 'M'  # Mid prices
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            candles = []
            for c in data.get('candles', []):
                if c['complete']:
                    candles.append({
                        'datetime': pd.to_datetime(c['time']),
                        'open': float(c['mid']['o']),
                        'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']),
                        'close': float(c['mid']['c']),
                        'volume': int(c['volume'])
                    })

            df = pd.DataFrame(candles)
            if not df.empty:
                df.set_index('datetime', inplace=True)
            return df

        except Exception as e:
            print(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask prices."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {'instruments': symbol}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            for price in data.get('prices', []):
                if price['instrument'] == symbol:
                    bid = float(price['bids'][0]['price'])
                    ask = float(price['asks'][0]['price'])
                    return bid, ask

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")

        return 0.0, 0.0

    def get_account_balance(self) -> float:
        """Get account balance."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return float(data['account']['balance'])
        except Exception as e:
            print(f"Error getting account balance: {e}")
            return 0.0

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get('positions', [])
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def place_market_order(self, symbol: str, units: int, tp_price: float, sl_price: float) -> Optional[str]:
        """Place a market order with TP and SL."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"

        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': symbol,
                'units': str(units),
                'takeProfitOnFill': {
                    'price': f'{tp_price:.5f}'
                },
                'stopLossOnFill': {
                    'price': f'{sl_price:.5f}'
                }
            }
        }

        try:
            response = requests.post(url, headers=self.headers, json=order_data)
            response.raise_for_status()
            data = response.json()

            if 'orderFillTransaction' in data:
                trade_id = data['orderFillTransaction'].get('tradeOpened', {}).get('tradeID')
                return trade_id

        except Exception as e:
            print(f"Error placing order for {symbol}: {e}")

        return None

    def close_position(self, symbol: str) -> bool:
        """Close all units of a position."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{symbol}/close"

        try:
            response = requests.put(url, headers=self.headers, json={'longUnits': 'ALL', 'shortUnits': 'ALL'})
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
            return False

    def modify_trade_sl(self, trade_id: str, new_sl: float) -> bool:
        """Modify stop loss on an existing trade."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/orders"

        try:
            response = requests.put(url, headers=self.headers, json={
                'stopLoss': {'price': f'{new_sl:.5f}'}
            })
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error modifying SL for trade {trade_id}: {e}")
            return False


class ScalpingSignalGenerator:
    """Generate scalping signals based on ADX + ATR + DI + HTF trend."""

    def __init__(self, config: ScalpingConfig):
        self.config = config

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smoothed averages
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    def get_htf_trend(self, df: pd.DataFrame) -> str:
        """Get higher timeframe trend using EMA50 vs EMA200."""
        if len(df) < 200:
            return 'NONE'

        ema_fast = df['close'].ewm(span=self.config.htf_ema_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.htf_ema_slow, adjust=False).mean()

        current_price = df['close'].iloc[-1]
        ema_f = ema_fast.iloc[-1]
        ema_s = ema_slow.iloc[-1]

        if ema_f > ema_s and current_price > ema_f:
            return 'BUY'
        elif ema_f < ema_s and current_price < ema_f:
            return 'SELL'
        return 'NONE'

    def generate_signal(self, df: pd.DataFrame, pair_config: PairConfig) -> Tuple[Optional[str], float, str]:
        """
        Generate trading signal for a pair.

        Returns: (signal, confidence, reason)
            signal: 'BUY', 'SELL', or None
            confidence: 0.0 to 1.0
            reason: explanation string
        """
        if len(df) < 200:
            return None, 0.0, "Insufficient data"

        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(df, 14)
        atr = self.calculate_atr(df, 14)
        atr_avg = atr.rolling(window=20).mean()

        # Get latest values
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_atr = atr.iloc[-1]
        current_atr_avg = atr_avg.iloc[-1]

        if pd.isna(current_adx) or pd.isna(current_atr) or pd.isna(current_atr_avg):
            return None, 0.0, "Indicator NaN"

        # Check signal conditions
        di_diff = abs(current_plus_di - current_minus_di)

        # 1. ADX must be above threshold
        if current_adx < self.config.adx_threshold:
            return None, 0.0, f"ADX {current_adx:.1f} < {self.config.adx_threshold}"

        # 2. ATR must show volatility expansion
        if current_atr < current_atr_avg * self.config.atr_expansion_mult:
            return None, 0.0, f"ATR {current_atr:.5f} < {current_atr_avg * self.config.atr_expansion_mult:.5f}"

        # 3. DI difference must be significant
        if di_diff < self.config.di_difference_min:
            return None, 0.0, f"DI diff {di_diff:.1f} < {self.config.di_difference_min}"

        # 4. Higher timeframe trend confirmation
        htf_trend = self.get_htf_trend(df)

        if htf_trend == 'BUY' and current_plus_di > current_minus_di:
            confidence = min(0.9, 0.5 + (current_adx / 100) + (di_diff / 50))
            return 'BUY', confidence, f"ADX={current_adx:.1f}, DI+={current_plus_di:.1f}, DI-={current_minus_di:.1f}"

        elif htf_trend == 'SELL' and current_minus_di > current_plus_di:
            confidence = min(0.9, 0.5 + (current_adx / 100) + (di_diff / 50))
            return 'SELL', confidence, f"ADX={current_adx:.1f}, DI+={current_plus_di:.1f}, DI-={current_minus_di:.1f}"

        return None, 0.0, f"HTF trend {htf_trend} doesn't match DI"


class LiveScalpingEngine:
    """Live scalping engine for OANDA."""

    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.api = OandaAPI(config)
        self.signal_gen = ScalpingSignalGenerator(config)
        self.positions: Dict[str, Position] = {}
        self.daily_losses = 0
        self.daily_wins = 0
        self.last_trade_date = None
        self.running = False

    def get_current_hour_utc(self) -> int:
        """Get current hour in UTC."""
        return datetime.now(timezone.utc).hour

    def reset_daily_counters(self):
        """Reset daily win/loss counters at midnight."""
        today = datetime.now(timezone.utc).date()
        if self.last_trade_date != today:
            self.daily_losses = 0
            self.daily_wins = 0
            self.last_trade_date = today
            print(f"\n[{datetime.now(timezone.utc)}] New trading day - counters reset")

    def calculate_position_size(self, pair_config: PairConfig) -> int:
        """Calculate position size in units based on dollars per pip."""
        # For standard lot: 1 pip = $10 for 100,000 units
        # So for $5/pip, we need 50,000 units
        units_per_dollar = 10000  # 10,000 units = $1/pip for most pairs

        if pair_config.pip_value == 0.01:  # JPY pairs
            units_per_dollar = 100  # Simplified for JPY

        return int(self.config.dollars_per_pip * units_per_dollar)

    def calculate_tp_sl_prices(self, pair_config: PairConfig, direction: str,
                                entry_price: float) -> Tuple[float, float]:
        """Calculate TP and SL prices."""
        pip_value = pair_config.pip_value

        if direction == 'BUY':
            tp_price = entry_price + (pair_config.tp_pips * pip_value)
            sl_price = entry_price - (pair_config.sl_pips * pip_value)
        else:
            tp_price = entry_price - (pair_config.tp_pips * pip_value)
            sl_price = entry_price + (pair_config.sl_pips * pip_value)

        return tp_price, sl_price

    def open_position(self, pair_config: PairConfig, signal: str, confidence: float, reason: str):
        """Open a new position."""
        symbol = pair_config.symbol

        # Check if already have position
        if symbol in self.positions:
            return

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            print(f"  Max positions reached ({self.config.max_positions})")
            return

        # Check daily loss limit
        if self.daily_losses >= self.config.max_daily_losses:
            print(f"  Daily loss limit reached ({self.daily_losses})")
            return

        # Get current price
        bid, ask = self.api.get_current_price(symbol)
        if bid == 0 or ask == 0:
            return

        # Calculate entry price based on direction
        entry_price = ask if signal == 'BUY' else bid

        # Calculate position size
        units = self.calculate_position_size(pair_config)
        if signal == 'SELL':
            units = -units

        # Calculate TP/SL
        tp_price, sl_price = self.calculate_tp_sl_prices(pair_config, signal, entry_price)

        # Place order
        trade_id = self.api.place_market_order(symbol, units, tp_price, sl_price)

        if trade_id:
            self.positions[symbol] = Position(
                symbol=symbol,
                direction=signal,
                entry_price=entry_price,
                units=abs(units),
                trade_id=trade_id,
                entry_time=datetime.now(timezone.utc),
                tp_price=tp_price,
                sl_price=sl_price
            )
            print(f"  OPENED {signal} {symbol} @ {entry_price:.5f}")
            print(f"    TP: {tp_price:.5f}, SL: {sl_price:.5f}")
            print(f"    Reason: {reason}")

    def update_trailing_stops(self):
        """Update trailing stops for all positions."""
        for symbol, pos in list(self.positions.items()):
            pair_config = self.config.pairs.get(symbol)
            if not pair_config:
                continue

            bid, ask = self.api.get_current_price(symbol)
            if bid == 0:
                continue

            current_price = bid if pos.direction == 'BUY' else ask
            pip_value = pair_config.pip_value

            # Calculate current profit in pips
            if pos.direction == 'BUY':
                profit_pips = (current_price - pos.entry_price) / pip_value
            else:
                profit_pips = (pos.entry_price - current_price) / pip_value

            # Check if trailing should activate
            if not pos.trailing_active and profit_pips >= pair_config.trailing_activation:
                pos.trailing_active = True
                pos.peak_profit_pips = profit_pips

                # Set initial trailing SL
                if pos.direction == 'BUY':
                    pos.trailing_sl = pos.entry_price + ((profit_pips - pair_config.trailing_distance) * pip_value)
                else:
                    pos.trailing_sl = pos.entry_price - ((profit_pips - pair_config.trailing_distance) * pip_value)

                # Update on OANDA
                self.api.modify_trade_sl(pos.trade_id, pos.trailing_sl)
                print(f"  {symbol}: Trailing activated @ {profit_pips:.1f} pips profit, SL -> {pos.trailing_sl:.5f}")

            # Update trailing SL if price moved further
            elif pos.trailing_active and profit_pips > pos.peak_profit_pips:
                pos.peak_profit_pips = profit_pips

                if pos.direction == 'BUY':
                    new_sl = pos.entry_price + ((profit_pips - pair_config.trailing_distance) * pip_value)
                    if new_sl > pos.trailing_sl:
                        pos.trailing_sl = new_sl
                        self.api.modify_trade_sl(pos.trade_id, pos.trailing_sl)
                        print(f"  {symbol}: Trail updated, SL -> {pos.trailing_sl:.5f} (+{profit_pips:.1f}p)")
                else:
                    new_sl = pos.entry_price - ((profit_pips - pair_config.trailing_distance) * pip_value)
                    if new_sl < pos.trailing_sl:
                        pos.trailing_sl = new_sl
                        self.api.modify_trade_sl(pos.trade_id, pos.trailing_sl)
                        print(f"  {symbol}: Trail updated, SL -> {pos.trailing_sl:.5f} (+{profit_pips:.1f}p)")

    def check_closed_positions(self):
        """Check for positions that were closed by TP/SL."""
        open_positions = self.api.get_open_positions()
        open_symbols = {p['instrument'] for p in open_positions}

        for symbol in list(self.positions.keys()):
            if symbol not in open_symbols:
                pos = self.positions.pop(symbol)

                # Determine if win or loss (approximate)
                bid, ask = self.api.get_current_price(symbol)
                if bid > 0:
                    current_price = bid if pos.direction == 'BUY' else ask
                    pip_value = self.config.pairs[symbol].pip_value

                    if pos.direction == 'BUY':
                        pnl_pips = (current_price - pos.entry_price) / pip_value
                    else:
                        pnl_pips = (pos.entry_price - current_price) / pip_value

                    pnl_dollars = pnl_pips * self.config.dollars_per_pip

                    if pnl_pips > 0:
                        self.daily_wins += 1
                        print(f"  CLOSED {symbol}: WIN +{pnl_pips:.1f} pips (${pnl_dollars:.2f})")
                    else:
                        self.daily_losses += 1
                        print(f"  CLOSED {symbol}: LOSS {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
                else:
                    print(f"  CLOSED {symbol}: Position exited")

    def run_once(self):
        """Run one iteration of the trading loop."""
        self.reset_daily_counters()
        current_hour = self.get_current_hour_utc()

        # Check for closed positions
        self.check_closed_positions()

        # Update trailing stops
        self.update_trailing_stops()

        # Get pairs that should be trading now
        active_pairs = self.config.get_pairs_for_hour(current_hour)

        if not active_pairs:
            return

        # Check each active pair for signals
        for pair_config in active_pairs:
            symbol = pair_config.symbol

            # Skip if already have position
            if symbol in self.positions:
                continue

            # Skip if at daily loss limit
            if self.daily_losses >= self.config.max_daily_losses:
                continue

            # Get candle data
            df = self.api.get_candles(symbol, count=300, granularity='M1')
            if df.empty or len(df) < 200:
                continue

            # Generate signal
            signal, confidence, reason = self.signal_gen.generate_signal(df, pair_config)

            if signal:
                print(f"\n[{datetime.now(timezone.utc)}] SIGNAL: {signal} {symbol}")
                self.open_position(pair_config, signal, confidence, reason)

    def run(self, interval_seconds: int = 60):
        """Run the trading loop."""
        print_scalping_config(self.config)

        print(f"\n{'='*80}")
        print("STARTING LIVE SCALPING ENGINE")
        print(f"{'='*80}")
        print(f"Checking every {interval_seconds} seconds...")
        print("Press Ctrl+C to stop\n")

        self.running = True

        try:
            while self.running:
                try:
                    current_hour = self.get_current_hour_utc()
                    active_pairs = self.config.get_pairs_for_hour(current_hour)

                    if active_pairs:
                        print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
                              f"Active: {[p.symbol for p in active_pairs]} | "
                              f"Positions: {len(self.positions)} | "
                              f"W/L today: {self.daily_wins}/{self.daily_losses}")

                    self.run_once()

                except Exception as e:
                    print(f"Error in trading loop: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nStopping trading engine...")
            self.running = False

        print(f"\nFinal stats - Wins: {self.daily_wins}, Losses: {self.daily_losses}")


def main():
    """Main entry point."""
    config = load_scalping_config()

    # Validate API credentials
    if not config.api_key or not config.account_id:
        print("ERROR: OANDA API credentials not found in environment variables")
        print("Please set OANDA_API_KEY and OANDA_ACCOUNT_ID")
        return

    engine = LiveScalpingEngine(config)

    # Get account balance
    balance = engine.api.get_account_balance()
    if balance > 0:
        print(f"\nAccount Balance: ${balance:,.2f}")
    else:
        print("WARNING: Could not fetch account balance")

    # Run the engine
    engine.run(interval_seconds=60)  # Check every minute


if __name__ == "__main__":
    main()
