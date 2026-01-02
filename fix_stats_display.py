#!/usr/bin/env python3
"""Add per-symbol stats display"""

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

with open(LIVE_ENGINE_PATH, "r") as f:
    content = f.read()

# Find and replace the session stats section
old_section = '''        print(f"\\n--- SESSION STATS (Runtime: {runtime}) ---")
        print(f"  Trades Today: {self.daily_trades} (W:{self.daily_wins} / L:{self.daily_losses})")
        win_rate = (self.daily_wins / self.daily_trades * 100) if self.daily_trades > 0 else 0
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Daily P&L: ${self.daily_pnl:+.2f}")

        print(f"\\nNext scan in 60 seconds... (Ctrl+C to stop)")'''

new_section = '''        print(f"\\n--- SESSION STATS (Runtime: {runtime}) ---")
        print(f"  Trades Today: {self.daily_trades} (W:{self.daily_wins} / L:{self.daily_losses})")
        win_rate = (self.daily_wins / self.daily_trades * 100) if self.daily_trades > 0 else 0
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Daily P&L: ${self.daily_pnl:+.2f}")

        # Per-symbol statistics
        if hasattr(self, "symbol_stats") and self.symbol_stats:
            print(f"\\n--- PER-SYMBOL STATS ---")
            for symbol in self.symbols:
                stats = self.symbol_stats.get(symbol, {"wins": 0, "losses": 0, "tp_count": 0, "sl_count": 0, "pnl": 0.0})
                total_trades = stats["wins"] + stats["losses"]
                if total_trades > 0:
                    sym_win_rate = stats["wins"] / total_trades * 100
                    pnl_sign = "+" if stats["pnl"] >= 0 else ""
                    print(f"  {symbol}: W:{stats[\\"wins\\"]}/L:{stats[\\"losses\\"]} ({sym_win_rate:.0f}%) | TP:{stats[\\"tp_count\\"]} SL:{stats[\\"sl_count\\"]} | P&L: ${pnl_sign}{stats[\\"pnl\\"]:.2f}")
                else:
                    print(f"  {symbol}: No trades yet")

        print(f"\\nNext scan in 60 seconds... (Ctrl+C to stop)")'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open(LIVE_ENGINE_PATH, "w") as f:
        f.write(content)
    print("SUCCESS: Added per-symbol stats display")
else:
    print("ERROR: Could not find section to replace")
