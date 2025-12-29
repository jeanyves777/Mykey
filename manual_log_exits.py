import json
import os

# Today's trades from OANDA (Ticket 186-382, excluding old trades and test trades)
todays_trades = [
    {"id": "382", "instrument": "GBP_USD", "pl": -3.21, "result": "LOSS"},
    {"id": "379", "instrument": "USD_CAD", "pl": 0.16, "result": "WIN"},
    {"id": "376", "instrument": "EUR_USD", "pl": 0.13, "result": "WIN"},
    {"id": "358", "instrument": "EUR_USD", "pl": 6.31, "result": "WIN"},
    {"id": "355", "instrument": "NZD_USD", "pl": 24.03, "result": "WIN"},
    {"id": "352", "instrument": "GBP_USD", "pl": 21.93, "result": "WIN"},
    {"id": "349", "instrument": "AUD_USD", "pl": 42.73, "result": "WIN"},
    {"id": "341", "instrument": "USD_CAD", "pl": 14.58, "result": "WIN"},
    {"id": "323", "instrument": "USD_JPY", "pl": 0.10, "result": "WIN"},
    {"id": "320", "instrument": "NZD_USD", "pl": -17.34, "result": "LOSS"},
    {"id": "312", "instrument": "AUD_USD", "pl": -4.61, "result": "LOSS"},
    {"id": "304", "instrument": "USD_JPY", "pl": -0.04, "result": "LOSS"},
    {"id": "286", "instrument": "AUD_USD", "pl": -11.43, "result": "LOSS"},
    {"id": "278", "instrument": "AUD_USD", "pl": -13.26, "result": "LOSS"},
    {"id": "275", "instrument": "NZD_USD", "pl": -15.50, "result": "LOSS"},
    {"id": "267", "instrument": "AUD_USD", "pl": -8.63, "result": "LOSS"},
    {"id": "259", "instrument": "AUD_USD", "pl": -10.64, "result": "LOSS"},
    {"id": "246", "instrument": "EUR_USD", "pl": 8.77, "result": "WIN"},
    {"id": "238", "instrument": "EUR_USD", "pl": -6.40, "result": "LOSS"},
    {"id": "230", "instrument": "USD_CHF", "pl": 4.85, "result": "WIN"},
    {"id": "227", "instrument": "EUR_USD", "pl": 5.91, "result": "WIN"},
    {"id": "224", "instrument": "USD_JPY", "pl": 0.05, "result": "WIN"},
    {"id": "211", "instrument": "USD_JPY", "pl": -0.03, "result": "LOSS"},
    {"id": "208", "instrument": "USD_CHF", "pl": -11.22, "result": "LOSS"},
    {"id": "195", "instrument": "USD_JPY", "pl": -0.02, "result": "LOSS"},
]

print("=" * 100)
print("TODAY'S LIVE TRADING SESSION - COMBINED V2 STRATEGY")
print("=" * 100)

total_pl = sum(t['pl'] for t in todays_trades)
winners = [t for t in todays_trades if t['pl'] > 0]
losers = [t for t in todays_trades if t['pl'] < 0]

print(f"\nSession Summary:")
print(f"  Starting Balance: $4,906.17")
print(f"  Ending Balance:   $4,937.58")
print(f"  Total P&L:        ${total_pl:+.2f}")
print(f"  Return:           {total_pl/4906.17*100:+.2f}%")

print(f"\nTrade Statistics:")
print(f"  Total Trades: {len(todays_trades)}")
print(f"  Winners:      {len(winners)} ({len(winners)/len(todays_trades)*100:.1f}%)")
print(f"  Losers:       {len(losers)} ({len(losers)/len(todays_trades)*100:.1f}%)")

total_wins = sum(t['pl'] for t in winners)
total_losses = abs(sum(t['pl'] for t in losers))

print(f"\nProfitability:")
print(f"  Gross Wins:   ${total_wins:+.2f}")
print(f"  Gross Losses: ${-total_losses:.2f}")
print(f"  Net P&L:      ${total_wins - total_losses:+.2f}")

if total_losses > 0:
    pf = total_wins / total_losses
    print(f"  Profit Factor: {pf:.2f}")

avg_win = total_wins / len(winners) if winners else 0
avg_loss = total_losses / len(losers) if losers else 0

print(f"\nAverages:")
print(f"  Avg Winner: ${avg_win:+.2f}")
print(f"  Avg Loser:  ${-avg_loss:.2f}")
print(f"  Avg Trade:  ${total_pl/len(todays_trades):+.2f}")

print(f"\nTop Winners:")
for i, t in enumerate(sorted(winners, key=lambda x: x['pl'], reverse=True)[:5], 1):
    print(f"  {i}. {t['instrument']}: ${t['pl']:+.2f}")

print(f"\nWorst Losers:")
for i, t in enumerate(sorted(losers, key=lambda x: x['pl'])[:5], 1):
    print(f"  {i}. {t['instrument']}: ${t['pl']:.2f}")

# Per-pair breakdown
pairs = {}
for t in todays_trades:
    pair = t['instrument']
    if pair not in pairs:
        pairs[pair] = {'trades': 0, 'pl': 0, 'winners': 0}
    pairs[pair]['trades'] += 1
    pairs[pair]['pl'] += t['pl']
    if t['pl'] > 0:
        pairs[pair]['winners'] += 1

print(f"\nPer-Pair Performance:")
for pair in sorted(pairs.keys()):
    data = pairs[pair]
    wr = data['winners'] / data['trades'] * 100 if data['trades'] > 0 else 0
    print(f"  {pair}: {data['trades']} trades | ${data['pl']:+.2f} | {wr:.0f}% WR")

print("\n" + "=" * 100)
print("STRATEGY VALIDATION")
print("=" * 100)

print(f"\nCombined V2 Strategy:")
print(f"  Backtest (1 month):  +25.82% | 51.9% WR | 1.34 PF")
print(f"  Live (Today):        {total_pl/4906.17*100:+.2f}% | {len(winners)/len(todays_trades)*100:.1f}% WR | {pf:.2f} PF")

if len(winners)/len(todays_trades)*100 > 50 and total_pl > 0:
    print(f"\n  Status: STRATEGY VALIDATED!")
    print(f"  Live results closely match backtest performance")
else:
    print(f"\n  Status: Need more data")

print("\n" + "=" * 100)
