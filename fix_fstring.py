#!/usr/bin/env python3
"""Fix f-string quote issues in per-symbol stats"""

LIVE_ENGINE_PATH = "/root/thevolumeainative/trading_system/Binance_Futures_Trading/engine/live_trading_engine.py"

with open(LIVE_ENGINE_PATH, "r") as f:
    content = f.read()

# Fix the problematic print statement - extract variables before using in f-string
old_code = '''        # Per-symbol statistics
        if hasattr(self, "symbol_stats") and self.symbol_stats:
            print(f"\\n--- PER-SYMBOL STATS ---")
            for symbol in self.symbols:
                stats = self.symbol_stats.get(symbol, {"wins": 0, "losses": 0, "tp_count": 0, "sl_count": 0, "pnl": 0.0})
                total_trades = stats["wins"] + stats["losses"]
                if total_trades > 0:
                    sym_win_rate = stats["wins"] / total_trades * 100
                    pnl_sign = "+" if stats["pnl"] >= 0 else ""
                    print(f"  {symbol}: W:{stats["wins"]}/L:{stats["losses"]} ({sym_win_rate:.0f}%) | TP:{stats["tp_count"]} SL:{stats["sl_count"]} | P&L: ${pnl_sign}{stats["pnl"]:.2f}")
                else:
                    print(f"  {symbol}: No trades yet")'''

new_code = '''        # Per-symbol statistics
        if hasattr(self, "symbol_stats") and self.symbol_stats:
            print(f"\\n--- PER-SYMBOL STATS ---")
            for symbol in self.symbols:
                stats = self.symbol_stats.get(symbol, {"wins": 0, "losses": 0, "tp_count": 0, "sl_count": 0, "pnl": 0.0})
                wins = stats["wins"]
                losses = stats["losses"]
                tp_count = stats["tp_count"]
                sl_count = stats["sl_count"]
                pnl = stats["pnl"]
                total_trades = wins + losses
                if total_trades > 0:
                    sym_win_rate = wins / total_trades * 100
                    pnl_sign = "+" if pnl >= 0 else ""
                    print(f"  {symbol}: W:{wins}/L:{losses} ({sym_win_rate:.0f}%) | TP:{tp_count} SL:{sl_count} | P&L: ${pnl_sign}{pnl:.2f}")
                else:
                    print(f"  {symbol}: No trades yet")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(LIVE_ENGINE_PATH, "w") as f:
        f.write(content)
    print("SUCCESS: Fixed f-string quote issues")
else:
    print("ERROR: Could not find code to replace")
    # Print what we're looking for
    print("Looking for:")
    print(repr(old_code[:100]))
