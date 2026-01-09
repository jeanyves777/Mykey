#!/usr/bin/env python3
"""Add per-symbol stats updates when trades close"""

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

with open(LIVE_ENGINE_PATH, "r") as f:
    content = f.read()

# Find and update the trade stats section
old_update = '''                    # Update daily stats
                    self.daily_pnl += realized_pnl
                    self.daily_trades += 1

                    hedge_label = f" [{closed_side}]" if self.hedge_mode else ""'''

new_update = '''                    # Update daily stats
                    self.daily_pnl += realized_pnl
                    self.daily_trades += 1

                    # Update per-symbol stats
                    if hasattr(self, "symbol_stats") and actual_symbol in self.symbol_stats:
                        self.symbol_stats[actual_symbol]["pnl"] += realized_pnl
                        if exit_type == "TP":
                            self.symbol_stats[actual_symbol]["tp_count"] += 1
                            self.symbol_stats[actual_symbol]["wins"] += 1
                        elif exit_type == "SL":
                            self.symbol_stats[actual_symbol]["sl_count"] += 1
                            self.symbol_stats[actual_symbol]["losses"] += 1
                        elif realized_pnl >= 0:
                            self.symbol_stats[actual_symbol]["wins"] += 1
                        else:
                            self.symbol_stats[actual_symbol]["losses"] += 1

                    hedge_label = f" [{closed_side}]" if self.hedge_mode else ""'''

if old_update in content:
    content = content.replace(old_update, new_update)
    print("1. Added per-symbol stats update in trade close section")
else:
    print("1. ERROR: Could not find trade stats update section")

# Also reset symbol stats on daily reset
old_reset = '''            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0'''

new_reset = '''            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0

            # Reset per-symbol stats
            if hasattr(self, "symbol_stats"):
                for symbol in self.symbols:
                    self.symbol_stats[symbol] = {
                        "wins": 0,
                        "losses": 0,
                        "tp_count": 0,
                        "sl_count": 0,
                        "pnl": 0.0
                    }'''

if old_reset in content:
    content = content.replace(old_reset, new_reset)
    print("2. Added per-symbol stats reset on daily reset")
else:
    print("2. ERROR: Could not find daily reset section")

with open(LIVE_ENGINE_PATH, "w") as f:
    f.write(content)

print("Done!")
