"""
Match-Trader Live Trading via Browser Automation
================================================
This script logs into Match-Trader and keeps the browser open
for you to monitor or for automated trading.

Usage:
    python run_match_trader_live.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from match_trader_browser import MatchTraderBrowser


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         MATCH-TRADER BROWSER AUTOMATION                       ║
    ║         For FundedNext Demo Challenge                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    mt_config = config['match_trader']

    print(f"Platform: {mt_config['base_url']}")
    print(f"Email: {mt_config['email']}")
    print()

    # Create browser client (visible mode)
    client = MatchTraderBrowser(
        headless=False,  # Show browser
        platform_url=mt_config['base_url']
    )

    try:
        # Login
        print("[1/2] Logging into Match-Trader...")
        if not client.login(mt_config['email'], mt_config['password']):
            print("\nERROR: Login failed!")
            print("Check your email and password in config.json")
            input("Press Enter to exit...")
            return

        print("       Login successful!")

        # Keep browser running
        print("\n[2/2] Browser is now RUNNING")
        print()
        print("=" * 60)
        print("  MATCH-TRADER IS OPEN AND READY")
        print("=" * 60)
        print()
        print("  The browser will stay open.")
        print("  You can trade manually or let the bot trade.")
        print()
        print("  Press Ctrl+C in this window to close browser.")
        print()
        print("=" * 60)

        # Keep running forever until user stops
        while True:
            try:
                # Check if browser is still open
                _ = client.driver.current_url
                time.sleep(5)
            except:
                print("\n  Browser was closed manually.")
                break

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        try:
            client.close_browser()
        except:
            pass
        print("Done!")


if __name__ == "__main__":
    main()
