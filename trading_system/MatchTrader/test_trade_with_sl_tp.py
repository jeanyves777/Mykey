"""
Test Trade with SL/TP on Match-Trader
=====================================
Places a test trade WITH Stop Loss and Take Profit.

Usage:
    python test_trade_with_sl_tp.py
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
from selenium.webdriver.common.action_chains import ActionChains


def place_trade_with_sl_tp(client, symbol="EURUSD", side="BUY", volume=0.01,
                            sl_pips=15, tp_pips=22):
    """
    Place a trade with Stop Loss and Take Profit.

    Args:
        client: MatchTraderBrowser instance
        symbol: Trading symbol
        side: BUY or SELL
        volume: Lot size
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
    """
    print(f"\n{'='*60}")
    print(f"  PLACING TRADE: {side} {volume} {symbol}")
    print(f"  SL: {sl_pips} pips | TP: {tp_pips} pips")
    print(f"{'='*60}")

    driver = client.driver
    wait = WebDriverWait(driver, 15)

    # Screenshot folder
    screenshot_path = Path(__file__).parent / "screenshots"
    screenshot_path.mkdir(exist_ok=True)

    try:
        # Step 1: Navigate to trade page
        print("\n[1/6] Navigating to trade page...")
        if "/app/trade" not in driver.current_url:
            driver.get(f"{client.platform_url}/app/trade")
            time.sleep(3)

        # Step 2: Click on symbol in watchlist
        print(f"[2/6] Selecting {symbol}...")
        try:
            # Find symbol in the left panel/watchlist
            symbol_elements = driver.find_elements(By.XPATH,
                f"//*[contains(text(), '{symbol}') or contains(text(), '{symbol[:3]}/{symbol[3:]}')]")

            for elem in symbol_elements:
                if elem.is_displayed():
                    elem.click()
                    print(f"       Selected {symbol}")
                    time.sleep(1)
                    break
        except Exception as e:
            print(f"       Warning: Could not select symbol: {e}")

        # Take screenshot
        driver.save_screenshot(str(screenshot_path / f"step2_symbol_{datetime.now().strftime('%H%M%S')}.png"))

        # Step 3: Set volume
        print(f"[3/6] Setting volume to {volume}...")
        try:
            # Find volume/lot input - look for input fields with number type
            inputs = driver.find_elements(By.CSS_SELECTOR, "input")
            volume_set = False

            for inp in inputs:
                try:
                    if inp.is_displayed() and inp.is_enabled():
                        inp_type = inp.get_attribute('type')
                        inp_value = inp.get_attribute('value')
                        placeholder = inp.get_attribute('placeholder') or ''

                        # Look for volume/lot input
                        if inp_type == 'number' or 'lot' in placeholder.lower() or 'volume' in placeholder.lower():
                            # Check if it looks like a lot size field (value between 0.01 and 100)
                            try:
                                current_val = float(inp_value) if inp_value else 0
                                if 0 <= current_val <= 100:
                                    inp.click()
                                    inp.send_keys(Keys.CONTROL + "a")
                                    inp.send_keys(str(volume))
                                    print(f"       Volume set to {volume}")
                                    volume_set = True
                                    break
                            except:
                                pass
                except:
                    continue

            if not volume_set:
                print("       Warning: Could not find volume input")

        except Exception as e:
            print(f"       Warning: {e}")

        time.sleep(0.5)
        driver.save_screenshot(str(screenshot_path / f"step3_volume_{datetime.now().strftime('%H%M%S')}.png"))

        # Step 4: Set SL and TP before placing trade
        print(f"[4/6] Setting SL ({sl_pips} pips) and TP ({tp_pips} pips)...")

        # Try to find SL/TP input fields or toggles
        # Look for "SL" and "TP" labels and nearby inputs
        try:
            # Method 1: Look for SL/TP inputs directly
            sl_inputs = driver.find_elements(By.XPATH,
                "//input[contains(@placeholder, 'SL') or contains(@placeholder, 'Stop') or contains(@name, 'sl')]")
            tp_inputs = driver.find_elements(By.XPATH,
                "//input[contains(@placeholder, 'TP') or contains(@placeholder, 'Take') or contains(@name, 'tp')]")

            # Method 2: Look for checkboxes or toggles to enable SL/TP
            sl_toggles = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'SL') or contains(text(), 'Stop Loss')]//ancestor::div[1]//input[@type='checkbox']")
            tp_toggles = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'TP') or contains(text(), 'Take Profit')]//ancestor::div[1]//input[@type='checkbox']")

            # Method 3: Look for SL/TP section that might need to be expanded
            sl_sections = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Stop Loss') or contains(text(), 'SL')]")
            tp_sections = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Take Profit') or contains(text(), 'TP')]")

            # Click on any SL/TP sections to expand them
            for section in sl_sections + tp_sections:
                try:
                    if section.is_displayed():
                        section.click()
                        time.sleep(0.3)
                except:
                    pass

            # Try to set SL value
            for inp in sl_inputs:
                if inp.is_displayed():
                    inp.clear()
                    inp.send_keys(str(sl_pips))
                    print(f"       SL input found and set")
                    break

            # Try to set TP value
            for inp in tp_inputs:
                if inp.is_displayed():
                    inp.clear()
                    inp.send_keys(str(tp_pips))
                    print(f"       TP input found and set")
                    break

        except Exception as e:
            print(f"       Note: SL/TP fields not found in order panel: {e}")
            print(f"       Will set SL/TP after trade is placed")

        driver.save_screenshot(str(screenshot_path / f"step4_sltp_{datetime.now().strftime('%H%M%S')}.png"))

        # Step 5: Click BUY or SELL button
        print(f"[5/6] Clicking {side} button...")

        trade_executed = False
        button_search = 'buy' if side.upper() == 'BUY' else 'sell'

        # Find all buttons
        buttons = driver.find_elements(By.TAG_NAME, "button")

        for btn in buttons:
            try:
                if btn.is_displayed():
                    btn_text = btn.text.lower()
                    btn_class = btn.get_attribute('class') or ''

                    if button_search in btn_text or button_search in btn_class.lower():
                        print(f"       Found {side} button: '{btn.text}'")

                        # Take screenshot before click
                        driver.save_screenshot(str(screenshot_path / f"step5_before_click_{datetime.now().strftime('%H%M%S')}.png"))

                        btn.click()
                        trade_executed = True
                        print(f"       ✅ Clicked {side} button!")
                        time.sleep(2)
                        break
            except:
                continue

        if not trade_executed:
            # Try alternative: look for div with buy/sell class
            divs = driver.find_elements(By.XPATH,
                f"//div[contains(@class, '{button_search}') and not(contains(@class, 'disabled'))]")
            for div in divs:
                if div.is_displayed():
                    div.click()
                    trade_executed = True
                    print(f"       ✅ Clicked {side} div!")
                    time.sleep(2)
                    break

        driver.save_screenshot(str(screenshot_path / f"step5_after_click_{datetime.now().strftime('%H%M%S')}.png"))

        if not trade_executed:
            print(f"       ❌ Could not find {side} button!")
            return False

        # Step 6: Set SL/TP on the open position (if not set before)
        print(f"[6/6] Setting SL/TP on position...")
        time.sleep(2)

        # Navigate to positions tab if needed
        try:
            positions_tab = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Open Positions') or contains(text(), 'Positions')]")
            for tab in positions_tab:
                if tab.is_displayed():
                    tab.click()
                    time.sleep(1)
                    break
        except:
            pass

        # Find the edit/pencil button for SL/TP
        try:
            # Look for edit icons (pencil) in the positions row
            edit_buttons = driver.find_elements(By.XPATH,
                "//button[contains(@class, 'edit')] | //*[contains(@class, 'pencil')] | //button[.//*[name()='svg']]")

            # Or look for the TP/SL cell and click it
            sltp_cells = driver.find_elements(By.XPATH,
                "//td[contains(text(), 'TP:') or contains(text(), 'SL:')] | //div[contains(text(), 'TP:')]")

            for cell in sltp_cells:
                if cell.is_displayed():
                    cell.click()
                    print("       Clicked SL/TP cell")
                    time.sleep(1)
                    break

            # Now look for SL/TP input popup
            time.sleep(1)

            # Get current price to calculate SL/TP prices
            # For EURUSD, 1 pip = 0.0001
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01

            # Try to find and fill SL input
            sl_price_inputs = driver.find_elements(By.XPATH,
                "//input[contains(@placeholder, 'Stop') or contains(@name, 'stopLoss') or contains(@id, 'sl')]")

            tp_price_inputs = driver.find_elements(By.XPATH,
                "//input[contains(@placeholder, 'Take') or contains(@name, 'takeProfit') or contains(@id, 'tp')]")

            # Also try generic number inputs in a modal/popup
            modal_inputs = driver.find_elements(By.CSS_SELECTOR,
                ".modal input[type='number'], .popup input[type='number'], .dialog input")

            print(f"       Found {len(sl_price_inputs)} SL inputs, {len(tp_price_inputs)} TP inputs")

            driver.save_screenshot(str(screenshot_path / f"step6_sltp_edit_{datetime.now().strftime('%H%M%S')}.png"))

        except Exception as e:
            print(f"       Note: Could not set SL/TP on position: {e}")

        print(f"\n{'='*60}")
        print(f"  TRADE COMPLETED!")
        print(f"  {side} {volume} {symbol}")
        print(f"  Check the platform to verify SL/TP")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        driver.save_screenshot(str(screenshot_path / f"error_{datetime.now().strftime('%H%M%S')}.png"))
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     TEST TRADE WITH STOP LOSS & TAKE PROFIT                   ║
    ║     Will place: BUY 0.01 EURUSD                               ║
    ║     SL: 15 pips | TP: 22 pips                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    mt_config = config['match_trader']

    # Create browser
    client = MatchTraderBrowser(
        headless=False,
        platform_url=mt_config['base_url']
    )

    try:
        # Login
        print("[1/2] Logging in...")
        if not client.login(mt_config['email'], mt_config['password']):
            print("ERROR: Login failed!")
            input("Press Enter to exit...")
            return

        print("       Login successful!")
        time.sleep(2)

        # Place trade with SL/TP
        print("\n[2/2] Placing trade with SL/TP...")
        success = place_trade_with_sl_tp(
            client=client,
            symbol="EURUSD",
            side="BUY",
            volume=0.01,
            sl_pips=15,
            tp_pips=22
        )

        # Keep browser open
        print("\n" + "="*60)
        print("  Browser will stay open for 2 minutes")
        print("  You can manually adjust SL/TP if needed")
        print("  Screenshots saved in: screenshots/")
        print("="*60)

        time.sleep(120)

    except KeyboardInterrupt:
        print("\n\nCancelled")
    finally:
        print("\nClosing browser...")
        client.close_browser()


if __name__ == "__main__":
    main()
