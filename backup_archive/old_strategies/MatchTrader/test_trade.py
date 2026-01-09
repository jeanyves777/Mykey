"""
Test Trade on Match-Trader
==========================
Places a small test trade to verify the browser automation works.

Usage:
    python test_trade.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from match_trader_browser import MatchTraderBrowser
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


def place_test_trade(client: MatchTraderBrowser, symbol: str = "EURUSD",
                     side: str = "BUY", volume: float = 0.01):
    """
    Place a test trade using browser automation.

    Args:
        client: Authenticated MatchTraderBrowser
        symbol: Trading symbol
        side: BUY or SELL
        volume: Lot size (0.01 = micro lot)
    """
    print(f"\n{'='*60}")
    print(f"  PLACING TEST TRADE: {side} {volume} {symbol}")
    print(f"{'='*60}")

    driver = client.driver
    wait = WebDriverWait(driver, 15)

    try:
        # Step 1: Make sure we're on the trade page
        current_url = driver.current_url
        if "/app/trade" not in current_url:
            print(f"  Navigating to trade page...")
            driver.get(f"{client.platform_url}/app/trade")
            time.sleep(3)

        # Take screenshot of current state
        screenshot_path = Path(__file__).parent / "screenshots"
        screenshot_path.mkdir(exist_ok=True)
        driver.save_screenshot(str(screenshot_path / f"before_trade_{datetime.now().strftime('%H%M%S')}.png"))
        print(f"  Screenshot saved")

        # Step 2: Find and click on the symbol in the watchlist
        print(f"  Looking for {symbol} in watchlist...")

        # Try to find EURUSD in the left panel
        symbol_found = False
        symbol_selectors = [
            f"//div[contains(text(), '{symbol}')]",
            f"//span[contains(text(), '{symbol}')]",
            f"//*[contains(@class, 'symbol') and contains(text(), '{symbol}')]",
            f"//*[contains(text(), '{symbol[:3]}') and contains(text(), '{symbol[3:]}')]",
        ]

        for selector in symbol_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                for elem in elements:
                    if elem.is_displayed():
                        elem.click()
                        print(f"  Clicked on {symbol}")
                        symbol_found = True
                        time.sleep(1)
                        break
                if symbol_found:
                    break
            except Exception as e:
                continue

        if not symbol_found:
            print(f"  WARNING: Could not find {symbol} in watchlist")
            print(f"  Will try to trade anyway...")

        # Step 3: Find the volume/lot size input
        print(f"  Setting volume to {volume}...")

        volume_selectors = [
            "input[type='number']",
            "input[name='volume']",
            "input[name='lots']",
            "input[name='amount']",
            "[class*='volume'] input",
            "[class*='lot'] input",
            "[class*='amount'] input",
        ]

        volume_input = None
        for selector in volume_selectors:
            try:
                inputs = driver.find_elements(By.CSS_SELECTOR, selector)
                for inp in inputs:
                    if inp.is_displayed() and inp.is_enabled():
                        # Check if it looks like a volume input
                        placeholder = inp.get_attribute('placeholder') or ''
                        name = inp.get_attribute('name') or ''
                        if any(x in placeholder.lower() + name.lower() for x in ['vol', 'lot', 'amount', 'size']):
                            volume_input = inp
                            break
                        # Or just use the first number input
                        if inp.get_attribute('type') == 'number' and not volume_input:
                            volume_input = inp
                if volume_input:
                    break
            except:
                continue

        if volume_input:
            volume_input.click()
            volume_input.send_keys(Keys.CONTROL + "a")
            volume_input.send_keys(str(volume))
            print(f"  Volume set to {volume}")
            time.sleep(0.5)
        else:
            print(f"  WARNING: Could not find volume input")

        # Step 4: Find and click the BUY or SELL button
        print(f"  Looking for {side} button...")

        if side.upper() == "BUY":
            button_patterns = [
                "//button[contains(translate(text(), 'BUY', 'buy'), 'buy')]",
                "//button[contains(@class, 'buy')]",
                "//div[contains(@class, 'buy') and contains(@class, 'button')]",
                "//*[contains(@class, 'buy-btn')]",
                "//button[contains(text(), 'Buy')]",
                "//button[contains(text(), 'BUY')]",
            ]
        else:
            button_patterns = [
                "//button[contains(translate(text(), 'SELL', 'sell'), 'sell')]",
                "//button[contains(@class, 'sell')]",
                "//div[contains(@class, 'sell') and contains(@class, 'button')]",
                "//*[contains(@class, 'sell-btn')]",
                "//button[contains(text(), 'Sell')]",
                "//button[contains(text(), 'SELL')]",
            ]

        trade_button = None
        for pattern in button_patterns:
            try:
                buttons = driver.find_elements(By.XPATH, pattern)
                for btn in buttons:
                    if btn.is_displayed() and btn.is_enabled():
                        trade_button = btn
                        break
                if trade_button:
                    break
            except:
                continue

        if trade_button:
            # Take screenshot before clicking
            driver.save_screenshot(str(screenshot_path / f"before_click_{datetime.now().strftime('%H%M%S')}.png"))

            print(f"  Clicking {side} button...")
            trade_button.click()
            time.sleep(2)

            # Take screenshot after clicking
            driver.save_screenshot(str(screenshot_path / f"after_click_{datetime.now().strftime('%H%M%S')}.png"))

            print(f"\n  {'='*50}")
            print(f"  TRADE EXECUTED: {side} {volume} {symbol}")
            print(f"  {'='*50}")
            print(f"  Check the platform to confirm the trade.")

            return True
        else:
            print(f"  ERROR: Could not find {side} button!")
            driver.save_screenshot(str(screenshot_path / f"no_button_{datetime.now().strftime('%H%M%S')}.png"))
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        driver.save_screenshot(str(screenshot_path / f"error_{datetime.now().strftime('%H%M%S')}.png"))
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         MATCH-TRADER TEST TRADE                               ║
    ║         Will place a small 0.01 lot BUY on EURUSD             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    mt_config = config['match_trader']

    # Create browser client
    client = MatchTraderBrowser(
        headless=False,
        platform_url=mt_config['base_url']
    )

    try:
        # Login
        print("[1/3] Logging in...")
        if not client.login(mt_config['email'], mt_config['password']):
            print("ERROR: Login failed!")
            input("Press Enter to exit...")
            return

        print("       Login successful!")
        time.sleep(2)

        # Place test trade
        print("\n[2/3] Placing test trade...")
        success = place_test_trade(
            client=client,
            symbol="EURUSD",
            side="BUY",
            volume=0.01  # Smallest lot size
        )

        if success:
            print("\n[3/3] Trade placed! Check the platform.")
        else:
            print("\n[3/3] Trade may have failed. Check screenshots folder.")

        # Keep browser open
        print("\n" + "="*60)
        print("  Browser will stay open for 60 seconds")
        print("  Check the platform to verify the trade")
        print("="*60)

        time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nCancelled by user")

    finally:
        print("\nClosing browser...")
        client.close_browser()
        print("Done!")


if __name__ == "__main__":
    main()
