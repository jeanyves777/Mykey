#!/usr/bin/env python3
"""
Test the $5 minimum margin requirement for position sizing.
Shows when symbols will wait due to insufficient balance.
"""

def test_position_sizing():
    print("\n" + "="*70)
    print("POSITION SIZING - $5 MINIMUM MARGIN TEST")
    print("="*70)
    
    leverage = 20
    min_margin = 5.0
    buffer = 0.95
    
    test_cases = [
        ("Full Balance", 23.37, 7, 0),  # Current state: $23.37, 7 symbols, 0 positions
        ("Low Balance", 15.00, 7, 0),   # Low balance scenario
        ("Critical Low", 10.00, 7, 3),  # Very low with 3 positions already open
        ("Depleted", 8.00, 7, 5),       # Almost depleted, 5 positions open
    ]
    
    print("\nLEVERAGE: 20x")
    print("MINIMUM MARGIN: $5.00 per trade")
    print("BUFFER: 95% of calculated margin")
    print()
    
    for scenario, balance, total_symbols, positions_open in test_cases:
        print(f"\n{'─'*70}")
        print(f"SCENARIO: {scenario}")
        print(f"  Available Balance: ${balance:.2f}")
        print(f"  Total Symbols: {total_symbols}")
        print(f"  Open Positions: {positions_open}")
        
        symbols_needing_margin = total_symbols - positions_open
        
        if symbols_needing_margin == 0:
            print(f"  ✅ All symbols have positions - no new trades needed")
            continue
        
        margin_per_symbol = (balance / symbols_needing_margin) * buffer
        
        print(f"  Symbols Waiting: {symbols_needing_margin}")
        print(f"  Margin Per Symbol: ${margin_per_symbol:.2f}")
        
        if margin_per_symbol >= min_margin:
            position_value = margin_per_symbol * leverage
            print(f"  ✅ SUFFICIENT: Each symbol can trade")
            print(f"     Position Value: ${position_value:.2f} (${margin_per_symbol:.2f} × {leverage}x)")
        else:
            print(f"  ❌ INSUFFICIENT: Symbols must wait!")
            print(f"     Need: ${min_margin:.2f} | Available: ${margin_per_symbol:.2f}")
            print(f"     Shortfall: ${min_margin - margin_per_symbol:.2f} per symbol")
            # How much total balance needed?
            min_balance_needed = (min_margin / buffer) * symbols_needing_margin
            print(f"     Minimum Balance Needed: ${min_balance_needed:.2f}")
    
    print("\n" + "="*70)
    print("EXAMPLE LOG OUTPUTS:")
    print("="*70)
    
    print("\n✅ WHEN BALANCE IS SUFFICIENT:")
    print("   [BTCUSDT] Dynamic margin: $3.16 ($23.37 available / 7 symbols * 0.95)")
    print("   [BTCUSDT] Position size: 0.007 (margin $3.16 x 20x leverage = $63.20)")
    print("   [BTCUSDT] Opening LONG position...")
    
    print("\n❌ WHEN BALANCE IS INSUFFICIENT:")
    print("   [ETHUSDT] Insufficient balance for minimum trade size.")
    print("   [ETHUSDT] Need: $5.00 | Available: $2.71 ($8.00 total / 3 symbols).")
    print("   [ETHUSDT] Symbol will wait for more balance.")
    print("   [ETHUSDT] Invalid position size: 0")
    
    print("\n" + "="*70)
    print("KEY POINTS:")
    print("="*70)
    print("  • Each symbol MUST have $5 minimum margin to trade")
    print("  • Available balance is split equally among symbols without positions")
    print("  • If margin < $5, symbol waits until more balance is available")
    print("  • As positions close, freed margin is distributed to waiting symbols")
    print("  • With 20x leverage, $5 margin = $100 position value")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_position_sizing()
