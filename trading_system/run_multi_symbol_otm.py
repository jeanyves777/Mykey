#!/usr/bin/env python3
"""
Multi-Symbol OTM Day Trading - Runner Script

Weekly OTM Day Trading Plan — $200 Account
Symbols: SPY, QQQ, AMD, IWM, PLTR, BAC

Day-specific parameters:
| Day       | DTE | Strike   | Budget | Target | Stop   |
|-----------|-----|----------|--------|--------|--------|
| Monday    | 4   | 1 OTM    | $50    | +30%   | -30%   |
| Tuesday   | 3   | 1 OTM    | $50    | +35%   | -35%   |
| Wednesday | 2   | 1-2 OTM  | $50    | +40%   | -40%   |
| Thursday  | 1   | 1-2 OTM  | $50    | +50%   | -45%   |
| Friday    | 0   | 1-2 OTM  | $60    | +75%   | -50%   |

Usage:
    python -m trading_system.run_multi_symbol_otm -y
    python -m trading_system.run_multi_symbol_otm --config path/to/config.json
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system.config.multi_symbol_otm_config import (
    MultiSymbolOTMConfig,
    DayConfig,
    load_config_from_file
)
from trading_system.engine.multi_symbol_otm_engine import MultiSymbolOTMEngine


def print_banner():
    """Print startup banner."""
    print()
    print("=" * 70)
    print("    MULTI-SYMBOL OTM DAY TRADING ENGINE")
    print("    Weekly OTM Day Trading Plan - $200 Account")
    print("=" * 70)
    print()
    print("    Symbols: SPY, QQQ, AMD, IWM, PLTR, BAC")
    print()
    print("    Day-specific parameters:")
    print("    +-----------+-----+----------+--------+--------+--------+")
    print("    | Day       | DTE | Strike   | Budget | Target | Stop   |")
    print("    +-----------+-----+----------+--------+--------+--------+")
    print("    | Monday    |  4  | 1 OTM    |  $50   |  +30%  |  -30%  |")
    print("    | Tuesday   |  3  | 1 OTM    |  $50   |  +35%  |  -35%  |")
    print("    | Wednesday |  2  | 1-2 OTM  |  $50   |  +40%  |  -40%  |")
    print("    | Thursday  |  1  | 1-2 OTM  |  $50   |  +50%  |  -45%  |")
    print("    | Friday    |  0  | 1-2 OTM  |  $60   |  +75%  |  -50%  |")
    print("    +-----------+-----+----------+--------+--------+--------+")
    print()


def print_current_day_params(config: MultiSymbolOTMConfig):
    """Print parameters for today."""
    weekday = datetime.now().weekday()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    if weekday > 4:
        print(f"    Today is {day_names[weekday]} - Market Closed")
        return

    day_config = config.get_day_config(weekday)
    otm_min, otm_max = config.get_strike_otm_range(weekday)

    print(f"    TODAY ({day_names[weekday]}):")
    print(f"    +-- DTE: {day_config.dte}")
    print(f"    +-- Strike: {otm_min}-{otm_max} OTM")
    print(f"    +-- Budget: ${day_config.budget:.0f}")
    print(f"    +-- Target: +{day_config.target_pct:.0f}%")
    print(f"    +-- Stop Loss: -{day_config.stop_loss_pct:.0f}%")
    print()


def load_api_credentials(config: MultiSymbolOTMConfig) -> MultiSymbolOTMConfig:
    """Load API credentials from config file."""
    # Try to load from paper_trading_config.json first
    config_paths = [
        os.path.join(os.path.expanduser('~'), '.thevolumeai', 'multi_symbol_otm_config.json'),
        os.path.join(os.path.expanduser('~'), '.thevolumeai', 'paper_trading_config.json'),
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                config.api_key = data.get('api_key', '')
                config.api_secret = data.get('api_secret', '')
                config.use_paper = data.get('use_paper', True)
                print(f"    Loaded credentials from: {config_path}")
                break
            except Exception as e:
                print(f"    Warning: Could not load {config_path}: {e}")

    return config


def create_default_config_file():
    """Create default config file if it doesn't exist."""
    config_dir = os.path.join(os.path.expanduser('~'), '.thevolumeai')
    config_path = os.path.join(config_dir, 'multi_symbol_otm_config.json')

    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)

        default_config = {
            "api_key": "",
            "api_secret": "",
            "use_paper": True,
            "total_account": 200.0,
            "symbols": ["SPY", "QQQ", "AMD", "IWM", "PLTR", "BAC"],
            "max_positions": 2,
            "max_positions_per_symbol": 1,
            "min_option_price": 0.10,
            "max_option_price": 1.00,
            "entry_time_start": "09:35:00",
            "entry_time_end": "15:30:00",
            "force_exit_time": "15:50:00",
            "day_configs": {
                "0": {"dte": 4, "strike_otm": 1, "strike_otm_max": 1, "budget": 50.0, "target_pct": 30.0, "stop_loss_pct": 30.0},
                "1": {"dte": 3, "strike_otm": 1, "strike_otm_max": 1, "budget": 50.0, "target_pct": 35.0, "stop_loss_pct": 35.0},
                "2": {"dte": 2, "strike_otm": 1, "strike_otm_max": 2, "budget": 50.0, "target_pct": 40.0, "stop_loss_pct": 40.0},
                "3": {"dte": 1, "strike_otm": 1, "strike_otm_max": 2, "budget": 50.0, "target_pct": 50.0, "stop_loss_pct": 45.0},
                "4": {"dte": 0, "strike_otm": 1, "strike_otm_max": 2, "budget": 60.0, "target_pct": 75.0, "stop_loss_pct": 50.0}
            }
        }

        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"    Created default config: {config_path}")
            print(f"    Please add your API credentials to this file.")
        except Exception as e:
            print(f"    Warning: Could not create config file: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Symbol OTM Day Trading Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_multi_symbol_otm -y
    python -m trading_system.run_multi_symbol_otm --config ~/config.json
    python -m trading_system.run_multi_symbol_otm --symbols SPY QQQ AMD
        """
    )
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    parser.add_argument('-y', '--yes', action='store_true', help='Auto-confirm start')
    parser.add_argument('--symbols', nargs='+', help='Override symbols to trade')
    parser.add_argument('--budget', type=float, help='Override daily budget')
    parser.add_argument('--paper', action='store_true', default=True, help='Use paper trading (default)')
    parser.add_argument('--live', action='store_true', help='Use live trading (CAUTION!)')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Create default config if needed
    create_default_config_file()

    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = MultiSymbolOTMConfig()

    # Load API credentials
    config = load_api_credentials(config)

    # Apply command line overrides
    if args.symbols:
        config.symbols = args.symbols
        print(f"    Override symbols: {', '.join(args.symbols)}")

    if args.budget:
        for day in config.day_configs.values():
            day.budget = args.budget
        print(f"    Override budget: ${args.budget:.0f}/day")

    if args.live:
        config.use_paper = False
        print("    *** LIVE TRADING MODE ***")
    else:
        config.use_paper = True
        print("    Paper trading mode")

    # Print current day parameters
    print()
    print_current_day_params(config)

    # Check API credentials
    if not config.api_key or not config.api_secret:
        print("    ERROR: API credentials not found!")
        print("    Please add your Alpaca API key and secret to:")
        print(f"    {os.path.join(os.path.expanduser('~'), '.thevolumeai', 'multi_symbol_otm_config.json')}")
        print()
        return

    # Confirm start
    print("=" * 70)
    if not args.yes:
        response = input("\n    Start Multi-Symbol OTM trading engine? (y/n): ")
        if response.lower() != 'y':
            print("    Aborted.")
            return

    print()
    print("    Starting engine...")
    print()

    # Create and run engine
    try:
        engine = MultiSymbolOTMEngine(config)
        engine.run()
    except KeyboardInterrupt:
        print("\n    Shutdown requested by user.")
    except Exception as e:
        print(f"\n    ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
