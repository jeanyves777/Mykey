#!/usr/bin/env python3
"""
Test Smart Compounding Logic
=============================
Simulates trades to verify the 50/50 split works correctly.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.trading_config import SMART_COMPOUNDING_CONFIG

def test_smart_compounding():
    """Test the smart compounding logic with simulated trades"""

    print("=" * 60)
    print("SMART COMPOUNDING TEST")
    print("=" * 60)

    # Configuration
    initial_capital = SMART_COMPOUNDING_CONFIG.get("initial_capital", 125.0)
    compound_pct = SMART_COMPOUNDING_CONFIG.get("compound_pct", 0.50)
    reserve_pct = SMART_COMPOUNDING_CONFIG.get("reserve_pct", 0.50)

    print(f"\nConfiguration:")
    print(f"  Initial Capital: ${initial_capital:.2f}")
    print(f"  Compound Rate: {compound_pct*100:.0f}%")
    print(f"  Reserve Rate: {reserve_pct*100:.0f}%")

    # Simulate state
    trading_capital = initial_capital
    reserve_fund = 0.0
    total_profit = 0.0

    # Simulated trades
    trades = [
        {"pnl": 5.00, "desc": "DOT LONG TP"},
        {"pnl": 3.50, "desc": "AVAX SHORT TP"},
        {"pnl": -2.00, "desc": "DOT SHORT SL"},  # Loss
        {"pnl": 4.20, "desc": "DOT LONG TP"},
        {"pnl": 6.00, "desc": "AVAX LONG TP"},
        {"pnl": 2.80, "desc": "DOT SHORT TP"},
    ]

    print(f"\n{'='*60}")
    print(f"{'Trade':<20} {'P&L':>8} {'Compound':>10} {'Reserve':>10} {'Trading Cap':>12} {'Reserve Fund':>12}")
    print(f"{'='*60}")
    print(f"{'START':<20} {'':<8} {'':<10} {'':<10} ${trading_capital:>10.2f} ${reserve_fund:>10.2f}")

    for trade in trades:
        pnl = trade["pnl"]
        desc = trade["desc"]
        total_profit += pnl

        if pnl > 0:
            # PROFIT: Split 50/50
            compound_amount = pnl * compound_pct
            reserve_amount = pnl * reserve_pct
            trading_capital += compound_amount
            reserve_fund += reserve_amount
        else:
            # LOSS: Only affects total profit tracking
            compound_amount = 0
            reserve_amount = 0
            # Note: Reserve is protected, trading capital unchanged here
            # In reality, the balance on Binance decreases from losses

        print(f"{desc:<20} ${pnl:>+7.2f} ${compound_amount:>+9.2f} ${reserve_amount:>+9.2f} ${trading_capital:>10.2f} ${reserve_fund:>10.2f}")

    print(f"{'='*60}")

    # Summary
    print(f"\nFINAL STATE:")
    print(f"  Trading Capital: ${trading_capital:.2f}")
    print(f"  Reserve Fund: ${reserve_fund:.2f} (PROTECTED)")
    print(f"  Total Profit: ${total_profit:.2f}")

    # Verification
    expected_trading = initial_capital + sum(max(t["pnl"], 0) * compound_pct for t in trades)
    expected_reserve = sum(max(t["pnl"], 0) * reserve_pct for t in trades)

    print(f"\nVERIFICATION:")
    print(f"  Expected Trading Capital: ${expected_trading:.2f}")
    print(f"  Expected Reserve Fund: ${expected_reserve:.2f}")
    match = abs(trading_capital - expected_trading) < 0.01 and abs(reserve_fund - expected_reserve) < 0.01
    print(f"  Match: {'PASS' if match else 'FAIL'}")

    print(f"\n{'='*60}")
    print("KEY INSIGHT:")
    print(f"  - Started with ${initial_capital:.2f}")
    print(f"  - Made ${sum(max(t['pnl'], 0) for t in trades):.2f} in profits")
    print(f"  - Trading now with ${trading_capital:.2f} (+${trading_capital - initial_capital:.2f} compounded)")
    print(f"  - Protected ${reserve_fund:.2f} in reserve (withdrawable anytime)")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_smart_compounding()
