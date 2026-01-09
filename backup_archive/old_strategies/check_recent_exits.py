import json

exits = []
with open('forex_logs/trades_20251212.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'EXIT':
            exits.append(data)

print("Recent exit reasons (last 15 trades):")
print("=" * 80)
for e in exits[-15:]:
    reason = e.get('exit_reason', 'UNKNOWN')
    instrument = e.get('instrument')
    pl = e.get('pl', 0)
    pips = e.get('pips', 0)
    print(f"{reason:<15} {instrument:<10} ${pl:+8.2f} ({pips:+.1f} pips)")

print("\n" + "=" * 80)
print("Exit reason summary:")
reasons = {}
for e in exits:
    reason = e.get('exit_reason', 'UNKNOWN')
    if reason not in reasons:
        reasons[reason] = 0
    reasons[reason] += 1

for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
    print(f"  {reason}: {count} trades")
