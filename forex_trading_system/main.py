"""
Forex Trading System - Main Entry Point
=======================================
HTF Confluence Strategy for Forex markets using OANDA API v20.

Usage:
    python main.py [--live] [--symbols EUR_USD,GBP_USD]

Options:
    --live              Use live account (default: practice account)
    --symbols           Comma-separated list of forex pairs (default: from config)
    --capital           Total capital to use (default: account balance)
    --risk              Risk per trade as percentage (default: 2%)
"""

import sys
import argparse
from engine.htf_confluence_live_engine import HTFConfluenceForexEngine


def main():
    """Main entry point for the trading system."""
    
    parser = argparse.ArgumentParser(
        description="HTF Confluence Forex Trading System"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live account (default: practice account)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated forex pairs (e.g., EUR_USD,GBP_USD)"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Total capital to use (default: account balance)"
    )
    
    parser.add_argument(
        "--risk",
        type=float,
        default=2.0,
        help="Risk per trade as percentage (default: 2%%)"
    )
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Convert risk to fraction
    risk_per_trade = args.risk / 100.0
    
    # Initialize trading engine
    print("\n" + "=" * 60)
    print("HTF CONFLUENCE FOREX TRADING SYSTEM")
    print("=" * 60)
    print(f"Mode: {'LIVE' if not args.live else 'PRACTICE'}")
    if symbols:
        print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk per trade: {args.risk}%")
    if args.capital:
        print(f"Capital: ${args.capital:.2f}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the engine.\n")
    
    engine = HTFConfluenceForexEngine(
        symbols=symbols,
        testnet=not args.live,
        total_capital=args.capital,
        risk_per_trade=risk_per_trade
    )
    
    # Run the engine
    engine.run()


if __name__ == "__main__":
    main()
