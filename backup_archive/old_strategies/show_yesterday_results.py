#!/usr/bin/env python3
"""
Show yesterday's trading performance with recovered ETHUSDT trades.
"""

print("\n" + "="*70)
print("YESTERDAY'S TRADING RESULTS (Jan 8, 2026) - RECOVERED")
print("="*70)

trades = {
    "ETHUSDT": {"wins": 2, "losses": 0, "pnl": 0.85, "trades": [
        "19:57:19 → PnL: $+0.44",
        "22:13:48 → PnL: $+0.41"
    ]},
    "DOTUSDT": {"wins": 3, "losses": 1, "pnl": -0.38, "trades": [
        "07:23:50 → PnL: $+1.03",
        "15:24:13 → PnL: $-2.07",
        "20:01:19 → PnL: $+0.33",
        "22:13:29 → PnL: $+0.33"
    ]},
    "BNBUSDT": {"wins": 1, "losses": 1, "pnl": 2.10, "trades": [
        "06:17:14 → PnL: $+2.99",
        "15:21:14 → PnL: $-0.89"
    ]},
    "XRPUSDT": {"wins": 0, "losses": 1, "pnl": -1.18, "trades": [
        "02:18:39 → PnL: $-1.18"
    ]},
    "ADAUSDT": {"wins": 1, "losses": 0, "pnl": 0.61, "trades": [
        "04:26:02 → PnL: $+0.61"
    ]}
}

total_trades = 0
total_wins = 0
total_losses = 0
total_pnl = 0

print("\n📊 BY SYMBOL:")
for symbol, data in trades.items():
    wins = data["wins"]
    losses = data["losses"]
    pnl = data["pnl"]
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    total_trades += wins + losses
    total_wins += wins
    total_losses += losses
    total_pnl += pnl
    
    print(f"\n{symbol}:")
    print(f"  {wins}W/{losses}L | WR: {wr:.0f}% | PnL: ${pnl:+.2f}")
    for trade in data["trades"]:
        print(f"    {trade}")

print("\n" + "="*70)
print("📈 YESTERDAY'S TOTAL:")
print(f"  Total Trades: {total_trades}")
print(f"  Wins/Losses: {total_wins}W/{total_losses}L")
print(f"  Win Rate: {total_wins / total_trades * 100:.1f}%")
print(f"  Total PnL: ${total_pnl:+.2f}")
print("="*70)

print("\n✅ RECOVERED TRADES:")
print("  ETHUSDT had 2 winning trades ($+0.85) that were missing!")
print("  All 10 trades from yesterday are now accounted for.")
print("\n🚀 TODAY (Jan 9): Fresh start with 7 symbols:")
print("  BTCUSDT, ETHUSDT, SOLUSDT, DOTUSDT, BNBUSDT, XRPUSDT, ADAUSDT")
print("="*70 + "\n")
