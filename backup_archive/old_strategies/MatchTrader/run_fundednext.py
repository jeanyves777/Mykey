"""
Run Match-Trader Trading System for FundedNext
==============================================

Usage:
    1. Edit config.json with your FundedNext credentials
    2. Run: python run_fundednext.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from match_trader_client import MatchTraderClient
from match_trader_engine import MatchTraderEngine


def load_config() -> dict:
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print("ERROR: config.json not found!")
        print("Please create config.json with your FundedNext credentials.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        return json.load(f)


def print_banner():
    """Print startup banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   ███████╗██╗   ██╗███╗   ██╗██████╗ ███████╗██████╗              ║
    ║   ██╔════╝██║   ██║████╗  ██║██╔══██╗██╔════╝██╔══██╗             ║
    ║   █████╗  ██║   ██║██╔██╗ ██║██║  ██║█████╗  ██║  ██║             ║
    ║   ██╔══╝  ██║   ██║██║╚██╗██║██║  ██║██╔══╝  ██║  ██║             ║
    ║   ██║     ╚██████╔╝██║ ╚████║██████╔╝███████╗██████╔╝             ║
    ║   ╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═════╝              ║
    ║                                                                   ║
    ║   ███╗   ██╗███████╗██╗  ██╗████████╗                             ║
    ║   ████╗  ██║██╔════╝╚██╗██╔╝╚══██╔══╝                             ║
    ║   ██╔██╗ ██║█████╗   ╚███╔╝    ██║                                ║
    ║   ██║╚██╗██║██╔══╝   ██╔██╗    ██║                                ║
    ║   ██║ ╚████║███████╗██╔╝ ██╗   ██║                                ║
    ║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝   ╚═╝                                ║
    ║                                                                   ║
    ║          MATCH-TRADER AUTOMATED TRADING SYSTEM                    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


def print_rules(config: dict):
    """Print FundedNext account rules"""
    account = config.get('account', {})
    risk = config.get('risk_management', {})

    balance = account.get('starting_balance', 6000)
    daily_limit = balance * risk.get('daily_loss_limit_pct', 5) / 100
    max_loss = balance * risk.get('max_loss_limit_pct', 10) / 100

    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  FUNDEDNEXT STELLAR 2-STEP ACCOUNT RULES                        │
    ├─────────────────────────────────────────────────────────────────┤
    │  Account Size:      ${balance:,.0f}                                  │
    │  Phase:             {account.get('phase', 1)}                                         │
    ├─────────────────────────────────────────────────────────────────┤
    │  Phase 1 Target:    8% (${balance * 0.08:,.0f})                             │
    │  Phase 2 Target:    5% (${balance * 0.05:,.0f})                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  Daily Loss Limit:  {risk.get('daily_loss_limit_pct', 5)}% (${daily_limit:,.0f})                             │
    │  Max Drawdown:      {risk.get('max_loss_limit_pct', 10)}% (${max_loss:,.0f})                            │
    │  Min Trading Days:  5 days                                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  Risk Per Trade:    {risk.get('risk_per_trade_pct', 1)}%                                       │
    │  Max Position Size: {risk.get('max_position_size', 0.3)} lots                                   │
    │  Max Open Positions: {risk.get('max_open_positions', 2)}                                        │
    └─────────────────────────────────────────────────────────────────┘
    """)


def main():
    """Main entry point"""
    print_banner()

    # Load configuration
    print("[1/4] Loading configuration...")
    config = load_config()

    # Validate config
    mt_config = config.get('match_trader', {})
    if mt_config.get('base_url', '').startswith('YOUR_'):
        print("\n⚠️  ERROR: Please update config.json with your FundedNext credentials!")
        print("   You need to fill in:")
        print("   - base_url: Your Match-Trader server URL from FundedNext")
        print("   - broker_id: Your broker ID")
        print("   - email: Your login email")
        print("   - password: Your password")
        sys.exit(1)

    print_rules(config)

    # Initialize client
    print("[2/4] Connecting to Match-Trader...")
    client = MatchTraderClient(
        base_url=mt_config['base_url'],
        broker_id=mt_config['broker_id'],
        system_uuid=mt_config.get('system_uuid')
    )

    # Login
    print("[3/4] Authenticating...")
    if not client.login(mt_config['email'], mt_config['password']):
        print("ERROR: Login failed! Check your credentials.")
        sys.exit(1)

    print("✓ Connected and authenticated!")

    # Get initial balance
    balance = client.get_balance()
    if balance:
        print(f"\n  Account Balance: ${balance.balance:,.2f}")
        print(f"  Account Equity:  ${balance.equity:,.2f}")
        print(f"  Free Margin:     ${balance.margin_free:,.2f}")

    # Create engine
    print("\n[4/4] Initializing trading engine...")
    engine_config = {
        'starting_balance': config['account']['starting_balance'],
        'daily_loss_limit_pct': config['risk_management']['daily_loss_limit_pct'],
        'max_loss_limit_pct': config['risk_management']['max_loss_limit_pct'],
        'max_position_size': config['risk_management']['max_position_size'],
        'max_open_positions': config['risk_management']['max_open_positions'],
        'risk_per_trade_pct': config['risk_management']['risk_per_trade_pct'],
    }

    engine = MatchTraderEngine(client=client, config=engine_config)

    print("\n✓ Engine initialized!")
    print("\n" + "=" * 60)
    print("  TRADING SYSTEM READY")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Start trading loop
    try:
        engine.run(check_interval=5.0)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        engine.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()
