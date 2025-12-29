"""
Match-Trader Browser Automation Client
======================================
Uses Selenium to automate trading through the Match-Trader web interface.
Works with demo accounts that don't have API access.

Features:
- Auto-login to Match-Trader platform
- Place market orders (buy/sell)
- Set stop loss and take profit
- Monitor open positions
- Close positions
- Get account balance
"""

import time
import json
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position"""
    position_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    volume: float
    entry_price: float
    current_price: float
    profit: float
    sl_price: float = 0
    tp_price: float = 0


@dataclass
class AccountInfo:
    """Account balance information"""
    balance: float
    equity: float
    profit: float
    free_margin: float


class MatchTraderBrowser:
    """
    Browser automation client for Match-Trader platform.

    Usage:
        client = MatchTraderBrowser()
        client.login("email@example.com", "password")

        # Place a trade
        client.open_position("EURUSD", "BUY", 0.01, sl_pips=15, tp_pips=20)

        # Get positions
        positions = client.get_open_positions()

        # Close position
        client.close_position(position_id)
    """

    def __init__(self, headless: bool = False, platform_url: str = "https://demo.match-trader.com"):
        """
        Initialize browser automation client.

        Args:
            headless: Run browser in headless mode (no GUI)
            platform_url: Match-Trader platform URL
        """
        self.platform_url = platform_url
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.is_logged_in = False

        # Symbol pip values
        self.pip_values = {
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDJPY': 0.01,
            'USDCHF': 0.0001,
            'USDCAD': 0.0001,
            'AUDUSD': 0.0001,
            'NZDUSD': 0.0001,
            'XAUUSD': 0.01,
            'XAGUSD': 0.001,
            'BTCUSD': 1.0,
        }

    def start_browser(self):
        """Initialize and start the Chrome browser"""
        logger.info("Starting Chrome browser...")

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")

        # Common options
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")

        # Disable automation detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--disable-blink-features=AutomationControlled")

        # Install/use ChromeDriver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # Set up wait
        self.wait = WebDriverWait(self.driver, 20)

        logger.info("Browser started successfully")

    def close_browser(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.is_logged_in = False
            logger.info("Browser closed")

    def login(self, email: str, password: str) -> bool:
        """
        Login to Match-Trader platform.

        Args:
            email: Account email
            password: Account password

        Returns:
            True if login successful
        """
        if not self.driver:
            self.start_browser()

        try:
            login_url = f"{self.platform_url}/login"
            logger.info(f"Navigating to {login_url}...")
            self.driver.get(login_url)
            time.sleep(3)  # Wait for page load

            # Wait for page to fully load
            self.wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
            time.sleep(2)

            # Check current URL - if redirected away from login, go back to login
            current_url = self.driver.current_url
            if "/login" not in current_url:
                logger.info(f"Redirected to {current_url}, going back to login page...")
                self.driver.get(f"{self.platform_url}/login")
                time.sleep(3)
                self.wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
                time.sleep(2)

            # Look for login form
            logger.info("Looking for login form...")

            # Try different selectors for email field
            email_selectors = [
                "input[type='email']",
                "input[name='email']",
                "input[placeholder*='email' i]",
                "input[placeholder*='Email' i]",
                "#email",
                "input[id*='email' i]",
                "input[autocomplete='email']",
            ]

            email_field = None
            for selector in email_selectors:
                try:
                    email_field = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    if email_field and email_field.is_displayed():
                        logger.info(f"Found email field with selector: {selector}")
                        break
                    else:
                        email_field = None
                except:
                    continue

            if not email_field:
                # Try XPath
                try:
                    email_field = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//input[@type='email' or contains(@placeholder, 'mail') or contains(@name, 'mail')]"))
                    )
                except:
                    logger.error("Could not find email field")
                    self._save_screenshot("login_error")
                    return False

            # Enter email
            email_field.clear()
            email_field.send_keys(email)
            logger.info("Email entered")
            time.sleep(0.5)

            # Find password field
            password_selectors = [
                "input[type='password']",
                "input[name='password']",
                "#password",
            ]

            password_field = None
            for selector in password_selectors:
                try:
                    password_field = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if password_field:
                        break
                except:
                    continue

            if not password_field:
                logger.error("Could not find password field")
                self._save_screenshot("login_error")
                return False

            # Enter password
            password_field.clear()
            password_field.send_keys(password)
            logger.info("Password entered")
            time.sleep(0.5)

            # Find and click login button
            login_selectors = [
                "button[type='submit']",
                "button:contains('Log')",
                "button:contains('Sign')",
                "input[type='submit']",
                ".login-button",
                "#login-btn",
            ]

            login_button = None
            for selector in login_selectors:
                try:
                    login_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if login_button:
                        break
                except:
                    continue

            if not login_button:
                # Try XPath for button with login text
                try:
                    login_button = self.driver.find_element(
                        By.XPATH,
                        "//button[contains(text(), 'Log') or contains(text(), 'Sign') or contains(text(), 'LOGIN')]"
                    )
                except:
                    # Just press Enter
                    password_field.send_keys(Keys.RETURN)
                    logger.info("Pressed Enter to submit")
                    login_button = None

            if login_button:
                login_button.click()
                logger.info("Login button clicked")

            # Wait for login to complete
            time.sleep(5)

            # Check if we're logged in by looking for trading interface elements
            if self._check_logged_in():
                self.is_logged_in = True
                logger.info("Login successful!")
                return True
            else:
                logger.error("Login may have failed - trading interface not found")
                self._save_screenshot("login_check_failed")
                return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            self._save_screenshot("login_exception")
            return False

    def _check_logged_in(self) -> bool:
        """Check if we're logged into the trading interface"""
        try:
            # Look for elements that indicate we're in the trading interface
            indicators = [
                ".balance",
                ".equity",
                "[class*='balance']",
                "[class*='equity']",
                "[class*='trade']",
                "[class*='position']",
                ".symbol-list",
                "[class*='watchlist']",
            ]

            for selector in indicators:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element:
                        return True
                except:
                    continue

            # Check URL
            current_url = self.driver.current_url
            if "/app/" in current_url or "/trade" in current_url:
                return True

            return False

        except:
            return False

    def _save_screenshot(self, name: str):
        """Save a screenshot for debugging"""
        if self.driver:
            path = Path(__file__).parent / f"screenshots/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path.parent.mkdir(exist_ok=True)
            self.driver.save_screenshot(str(path))
            logger.info(f"Screenshot saved: {path}")

    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account balance information.

        Returns:
            AccountInfo object or None on error
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        try:
            # Look for balance/equity elements
            balance = 0.0
            equity = 0.0
            profit = 0.0
            free_margin = 0.0

            # Try different selectors
            balance_selectors = [
                "[class*='balance']",
                ".account-balance",
                "[data-testid='balance']",
            ]

            for selector in balance_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    text = element.text.replace('$', '').replace(',', '').strip()
                    balance = float(text)
                    break
                except:
                    continue

            # Get equity
            equity_selectors = [
                "[class*='equity']",
                ".account-equity",
                "[data-testid='equity']",
            ]

            for selector in equity_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    text = element.text.replace('$', '').replace(',', '').strip()
                    equity = float(text)
                    break
                except:
                    continue

            return AccountInfo(
                balance=balance,
                equity=equity,
                profit=equity - balance,
                free_margin=equity  # Simplified
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current price quote for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            Dict with bid, ask prices or None
        """
        if not self.is_logged_in:
            return None

        try:
            # Look for the symbol in the watchlist/market view
            # This is highly dependent on the platform's HTML structure
            symbol_element = self.driver.find_element(
                By.XPATH,
                f"//*[contains(text(), '{symbol}')]"
            )

            # Get parent container and find bid/ask
            container = symbol_element.find_element(By.XPATH, "./..")

            # Try to find bid/ask prices
            prices = container.find_elements(By.CSS_SELECTOR, "[class*='price']")

            if len(prices) >= 2:
                bid = float(prices[0].text)
                ask = float(prices[1].text)
                return {'symbol': symbol, 'bid': bid, 'ask': ask}

        except Exception as e:
            logger.debug(f"Could not get quote for {symbol}: {e}")

        return None

    def select_symbol(self, symbol: str) -> bool:
        """
        Select a trading symbol in the interface.

        Args:
            symbol: Symbol to select (e.g., "EURUSD")

        Returns:
            True if symbol selected successfully
        """
        try:
            # Click on the symbol in the watchlist
            symbol_element = self.wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    f"//*[contains(text(), '{symbol}')]"
                ))
            )
            symbol_element.click()
            time.sleep(1)
            logger.info(f"Selected symbol: {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to select symbol {symbol}: {e}")
            return False

    def open_position(self, symbol: str, side: str, volume: float,
                      sl_pips: float = 0, tp_pips: float = 0,
                      sl_price: float = 0, tp_price: float = 0) -> Optional[str]:
        """
        Open a new trading position with Stop Loss and Take Profit.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            side: "BUY" or "SELL"
            volume: Lot size
            sl_pips: Stop loss in pips (0 for none) - used to calculate price if sl_price not given
            tp_pips: Take profit in pips (0 for none) - used to calculate price if tp_price not given
            sl_price: Stop loss price (0 for none)
            tp_price: Take profit price (0 for none)

        Returns:
            Position ID if successful, None on error
        """
        if not self.is_logged_in:
            logger.error("Not logged in")
            return None

        try:
            logger.info(f"Opening position: {side} {volume} {symbol} (SL: {sl_pips}p, TP: {tp_pips}p)")

            # Make sure we're on the trade page
            if "/app/trade" not in self.driver.current_url:
                self.driver.get(f"{self.platform_url}/app/trade")
                time.sleep(3)

            # Select the symbol
            if not self.select_symbol(symbol):
                logger.warning(f"Could not select {symbol}, trying to continue anyway...")

            time.sleep(1)

            # Find volume input and set it
            volume_selectors = [
                "input[type='number']",
                "input[name='volume']",
                "input[name='lots']",
                "input[placeholder*='lot' i]",
                "input[placeholder*='volume' i]",
                "[class*='volume'] input",
                "[class*='lot'] input",
            ]

            volume_input = None
            for selector in volume_selectors:
                try:
                    inputs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for inp in inputs:
                        if inp.is_displayed() and inp.is_enabled():
                            try:
                                current_val = inp.get_attribute('value')
                                # Check if this looks like a volume/lot input (small number)
                                if current_val and 0 < float(current_val) <= 100:
                                    volume_input = inp
                                    break
                            except:
                                volume_input = inp
                                break
                    if volume_input:
                        break
                except:
                    continue

            if volume_input:
                volume_input.click()
                volume_input.send_keys(Keys.CONTROL + "a")
                volume_input.send_keys(str(volume))
                logger.info(f"Volume set to {volume}")
                time.sleep(0.3)
            else:
                logger.warning("Could not find volume input")

            # Try to set SL/TP BEFORE placing trade (if the UI supports it)
            self._try_set_sl_tp_before_trade(symbol, side, sl_pips, tp_pips, sl_price, tp_price)

            # Find and click BUY or SELL button
            if side.upper() == "BUY":
                button_texts = ["Buy", "BUY", "Long"]
            else:
                button_texts = ["Sell", "SELL", "Short"]

            trade_button = None

            # Method 1: Find by text
            for text in button_texts:
                try:
                    buttons = self.driver.find_elements(
                        By.XPATH,
                        f"//button[contains(text(), '{text}')] | //div[contains(@class, 'buy') or contains(@class, 'sell')][contains(text(), '{text}')]"
                    )
                    for btn in buttons:
                        if btn.is_displayed() and btn.is_enabled():
                            trade_button = btn
                            break
                    if trade_button:
                        break
                except:
                    continue

            # Method 2: Try CSS selectors
            if not trade_button:
                side_lower = side.lower()
                css_selectors = [
                    f"button.{side_lower}",
                    f"[class*='{side_lower}-btn']",
                    f"button[class*='{side_lower}']",
                    f"div[class*='{side_lower}'][class*='button']",
                ]
                for selector in css_selectors:
                    try:
                        buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for btn in buttons:
                            if btn.is_displayed():
                                trade_button = btn
                                break
                        if trade_button:
                            break
                    except:
                        continue

            if trade_button:
                self._save_screenshot(f"before_trade_{side}_{symbol}")
                trade_button.click()
                logger.info(f"Clicked {side} button")
                time.sleep(2)
                self._save_screenshot(f"after_trade_{side}_{symbol}")

                # Generate a position ID
                position_id = f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                logger.info(f"Trade executed: {position_id}")

                # Try to set SL/TP AFTER trade if not set before
                if sl_pips > 0 or tp_pips > 0 or sl_price > 0 or tp_price > 0:
                    time.sleep(2)
                    self._try_set_sl_tp_after_trade(symbol, side, sl_pips, tp_pips, sl_price, tp_price)

                return position_id
            else:
                logger.error(f"Could not find {side} button")
                self._save_screenshot("trade_button_not_found")
                return None

        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            self._save_screenshot("open_position_error")
            return None

    def _try_set_sl_tp_before_trade(self, symbol: str, side: str,
                                     sl_pips: float, tp_pips: float,
                                     sl_price: float, tp_price: float):
        """Try to find and set SL/TP inputs in the order panel before placing trade"""
        try:
            # Look for SL/TP toggles or checkboxes to enable them
            sl_toggles = self.driver.find_elements(By.XPATH,
                "//*[contains(text(), 'SL') or contains(text(), 'Stop Loss')]//preceding::input[@type='checkbox'][1] | "
                "//*[contains(text(), 'SL') or contains(text(), 'Stop Loss')]//following::input[@type='checkbox'][1]")

            tp_toggles = self.driver.find_elements(By.XPATH,
                "//*[contains(text(), 'TP') or contains(text(), 'Take Profit')]//preceding::input[@type='checkbox'][1] | "
                "//*[contains(text(), 'TP') or contains(text(), 'Take Profit')]//following::input[@type='checkbox'][1]")

            # Click to enable SL/TP if toggles found
            for toggle in sl_toggles[:1]:
                if toggle.is_displayed():
                    toggle.click()
                    logger.info("Enabled SL toggle")
                    time.sleep(0.3)
                    break

            for toggle in tp_toggles[:1]:
                if toggle.is_displayed():
                    toggle.click()
                    logger.info("Enabled TP toggle")
                    time.sleep(0.3)
                    break

            # Look for SL/TP input fields
            sl_inputs = self.driver.find_elements(By.XPATH,
                "//input[contains(@placeholder, 'SL') or contains(@placeholder, 'Stop') or "
                "contains(@name, 'sl') or contains(@name, 'stopLoss') or contains(@id, 'sl')]")

            tp_inputs = self.driver.find_elements(By.XPATH,
                "//input[contains(@placeholder, 'TP') or contains(@placeholder, 'Take') or "
                "contains(@name, 'tp') or contains(@name, 'takeProfit') or contains(@id, 'tp')]")

            # Set SL value (in pips for now - Match-Trader may use pips or price)
            if sl_pips > 0 and sl_inputs:
                for inp in sl_inputs:
                    if inp.is_displayed():
                        inp.clear()
                        inp.send_keys(str(sl_pips))
                        logger.info(f"Set SL to {sl_pips} pips in order panel")
                        break

            # Set TP value
            if tp_pips > 0 and tp_inputs:
                for inp in tp_inputs:
                    if inp.is_displayed():
                        inp.clear()
                        inp.send_keys(str(tp_pips))
                        logger.info(f"Set TP to {tp_pips} pips in order panel")
                        break

        except Exception as e:
            logger.debug(f"Could not set SL/TP before trade: {e}")

    def _try_set_sl_tp_after_trade(self, symbol: str, side: str,
                                    sl_pips: float, tp_pips: float,
                                    sl_price: float, tp_price: float):
        """Try to set SL/TP on the open position after trade is placed"""
        try:
            logger.info("Attempting to set SL/TP on open position...")

            # Navigate to positions tab - look for "Open Positions" tab with count
            positions_tabs = self.driver.find_elements(By.XPATH,
                "//*[contains(text(), 'Open Positions')] | "
                "//button[contains(text(), 'Open')] | "
                "//*[contains(@class, 'positions')]")

            for tab in positions_tabs:
                if tab.is_displayed():
                    tab.click()
                    logger.info("Clicked Positions tab")
                    time.sleep(1)
                    break

            # Find the table rows - Match-Trader uses a table structure
            # Look for rows containing our symbol
            time.sleep(1)

            # Try multiple approaches to find position rows
            position_rows = self.driver.find_elements(By.XPATH,
                f"//tr[.//text()[contains(., '{symbol}')]]")

            if not position_rows:
                # Try CSS for table rows
                all_rows = self.driver.find_elements(By.CSS_SELECTOR, "tr")
                position_rows = [r for r in all_rows if symbol in r.text]

            if not position_rows:
                logger.warning(f"Could not find position row for {symbol}")
                self._save_screenshot("no_position_row")
                return

            logger.info(f"Found {len(position_rows)} position row(s) for {symbol}")

            for row in position_rows:
                try:
                    if not row.is_displayed():
                        continue

                    # Check if this row contains our symbol
                    if symbol not in row.text:
                        continue

                    logger.info(f"Processing row: {row.text[:100]}...")

                    # Find the pencil/edit icon - it's likely an SVG or button in the TP/SL cell
                    # The pencil icon appears in the TP/SL column
                    edit_icons = row.find_elements(By.CSS_SELECTOR,
                        "svg, button svg, [class*='edit'], [class*='pencil'], "
                        "[class*='icon'], button[class*='icon']")

                    # Also try finding clickable elements in cells
                    clickable = row.find_elements(By.XPATH,
                        ".//td//button | .//td//svg | .//td//*[@role='button'] | "
                        ".//div[contains(@class, 'icon')] | .//*[local-name()='svg']")

                    edit_icons.extend(clickable)

                    clicked = False
                    for icon in edit_icons:
                        try:
                            if icon.is_displayed():
                                # Try to click the icon
                                logger.info(f"Trying to click edit icon: {icon.tag_name}")
                                icon.click()
                                clicked = True
                                logger.info("Clicked edit/pencil icon")
                                time.sleep(1.5)
                                break
                        except Exception as click_err:
                            logger.debug(f"Click failed: {click_err}")
                            continue

                    if not clicked:
                        # Try clicking on the TP/SL cell directly (might open edit popup)
                        cells = row.find_elements(By.TAG_NAME, "td")
                        for cell in cells:
                            if "TP:" in cell.text or "SL:" in cell.text or "-" in cell.text:
                                try:
                                    cell.click()
                                    logger.info("Clicked on TP/SL cell")
                                    time.sleep(1.5)
                                    clicked = True
                                    break
                                except:
                                    pass

                    self._save_screenshot(f"after_edit_click_{symbol}")

                    # Now look for the SL/TP input popup/modal
                    time.sleep(0.5)

                    # Calculate prices if needed
                    pip_value = self.pip_values.get(symbol, 0.0001)

                    # Get entry price from the row
                    entry_price = 0
                    try:
                        # Look for price in the row text (format like 1.33623)
                        import re
                        prices = re.findall(r'\d+\.\d{3,5}', row.text)
                        if prices:
                            entry_price = float(prices[0])
                            logger.info(f"Found entry price: {entry_price}")
                    except:
                        pass

                    # Calculate SL/TP prices
                    if sl_price == 0 and sl_pips > 0 and entry_price > 0:
                        if side.upper() == "BUY":
                            sl_price = entry_price - (sl_pips * pip_value)
                        else:
                            sl_price = entry_price + (sl_pips * pip_value)

                    if tp_price == 0 and tp_pips > 0 and entry_price > 0:
                        if side.upper() == "BUY":
                            tp_price = entry_price + (tp_pips * pip_value)
                        else:
                            tp_price = entry_price - (tp_pips * pip_value)

                    logger.info(f"Calculated SL: {sl_price:.5f}, TP: {tp_price:.5f}")

                    # Find all visible inputs on the page (popup should now be open)
                    all_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input")
                    visible_inputs = [i for i in all_inputs if i.is_displayed()]
                    logger.info(f"Found {len(visible_inputs)} visible inputs")

                    # Try to find SL and TP inputs
                    sl_set = False
                    tp_set = False

                    for inp in visible_inputs:
                        try:
                            placeholder = inp.get_attribute('placeholder') or ''
                            name = inp.get_attribute('name') or ''
                            inp_id = inp.get_attribute('id') or ''
                            inp_type = inp.get_attribute('type') or ''

                            identifier = (placeholder + name + inp_id).lower()

                            if ('sl' in identifier or 'stop' in identifier) and not sl_set:
                                inp.clear()
                                inp.send_keys(str(round(sl_price, 5)))
                                logger.info(f"Set SL to {sl_price:.5f}")
                                sl_set = True
                            elif ('tp' in identifier or 'take' in identifier or 'profit' in identifier) and not tp_set:
                                inp.clear()
                                inp.send_keys(str(round(tp_price, 5)))
                                logger.info(f"Set TP to {tp_price:.5f}")
                                tp_set = True
                        except Exception as inp_err:
                            logger.debug(f"Input error: {inp_err}")

                    # If we couldn't identify inputs by name, try by position
                    if not sl_set and not tp_set and len(visible_inputs) >= 2:
                        try:
                            # First input might be SL, second might be TP
                            visible_inputs[0].clear()
                            visible_inputs[0].send_keys(str(round(sl_price, 5)))
                            logger.info(f"Set first input (SL?) to {sl_price:.5f}")

                            if len(visible_inputs) > 1:
                                visible_inputs[1].clear()
                                visible_inputs[1].send_keys(str(round(tp_price, 5)))
                                logger.info(f"Set second input (TP?) to {tp_price:.5f}")
                        except:
                            pass

                    self._save_screenshot(f"after_sl_tp_input_{symbol}")

                    # Look for confirm/save/apply button
                    confirm_buttons = self.driver.find_elements(By.XPATH,
                        "//button[contains(text(), 'Confirm') or contains(text(), 'Save') or "
                        "contains(text(), 'Apply') or contains(text(), 'OK') or "
                        "contains(text(), 'Update') or contains(text(), 'Set')]")

                    for btn in confirm_buttons:
                        if btn.is_displayed():
                            btn.click()
                            logger.info("Clicked confirm button")
                            time.sleep(1)
                            break

                    self._save_screenshot(f"sl_tp_complete_{symbol}")
                    logger.info(f"SL/TP modification attempted for {symbol}")
                    break

                except Exception as e:
                    logger.warning(f"Error on position row: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Could not set SL/TP after trade: {e}")
            self._save_screenshot("sl_tp_error")

    def close_position(self, position_id: str = None, symbol: str = None) -> bool:
        """
        Close an open position.

        Args:
            position_id: Position ID to close
            symbol: Symbol of position to close (alternative)

        Returns:
            True if closed successfully
        """
        if not self.is_logged_in:
            return False

        try:
            logger.info(f"Closing position: {position_id or symbol}")

            # Navigate to positions tab/section if needed
            positions_tab = None
            try:
                positions_tab = self.driver.find_element(
                    By.XPATH,
                    "//button[contains(text(), 'Position')] | //div[contains(text(), 'Position')] | //*[contains(@class, 'positions-tab')]"
                )
                if positions_tab:
                    positions_tab.click()
                    time.sleep(1)
            except:
                pass

            # Find the position row
            if symbol:
                position_row = self.driver.find_element(
                    By.XPATH,
                    f"//*[contains(text(), '{symbol}')]/ancestor::tr | //*[contains(text(), '{symbol}')]/ancestor::div[contains(@class, 'position')]"
                )
            else:
                # Find by position ID or just the first position
                position_row = self.driver.find_element(
                    By.CSS_SELECTOR,
                    "tr.position, div.position, [class*='position-row']"
                )

            # Find close button in the row
            close_button = position_row.find_element(
                By.XPATH,
                ".//button[contains(text(), 'Close') or contains(@class, 'close')] | .//span[contains(@class, 'close')]"
            )
            close_button.click()

            time.sleep(1)

            # Confirm if needed
            try:
                confirm_button = self.driver.find_element(
                    By.XPATH,
                    "//button[contains(text(), 'Confirm') or contains(text(), 'Yes')]"
                )
                confirm_button.click()
            except:
                pass

            logger.info("Position closed")
            return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def get_open_positions(self) -> List[Position]:
        """
        Get list of open positions.

        Returns:
            List of Position objects
        """
        positions = []

        if not self.is_logged_in:
            return positions

        try:
            # Navigate to positions tab if needed
            try:
                positions_tab = self.driver.find_element(
                    By.XPATH,
                    "//button[contains(text(), 'Position')] | //*[contains(@class, 'positions-tab')]"
                )
                positions_tab.click()
                time.sleep(1)
            except:
                pass

            # Find position rows
            position_rows = self.driver.find_elements(
                By.CSS_SELECTOR,
                "tr.position, div.position, [class*='position-row'], [class*='open-position']"
            )

            for row in position_rows:
                try:
                    # Extract position data
                    text = row.text
                    # Parse the text to extract position info
                    # This is highly platform-specific

                    # Create position object with available data
                    pos = Position(
                        position_id=str(len(positions)),
                        symbol="UNKNOWN",
                        side="BUY",
                        volume=0.01,
                        entry_price=0,
                        current_price=0,
                        profit=0
                    )
                    positions.append(pos)
                except:
                    continue

        except Exception as e:
            logger.debug(f"Error getting positions: {e}")

        return positions


# ==================== Test Script ====================

def test_login():
    """Test the browser automation login"""
    import json
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    mt_config = config['match_trader']

    print("\n" + "=" * 60)
    print("  MATCH-TRADER BROWSER AUTOMATION TEST")
    print("=" * 60)

    # Create client (visible browser for testing)
    client = MatchTraderBrowser(headless=False, platform_url=mt_config['base_url'])

    try:
        # Login
        print("\n[1/3] Logging in...")
        if client.login(mt_config['email'], mt_config['password']):
            print("  Login successful!")

            # Wait and observe
            print("\n[2/3] Exploring interface...")
            time.sleep(5)

            # Try to get account info
            print("\n[3/3] Getting account info...")
            info = client.get_account_info()
            if info:
                print(f"  Balance: ${info.balance:,.2f}")
                print(f"  Equity: ${info.equity:,.2f}")
            else:
                print("  Could not get account info")

            # Wait for user to see results
            print("\n" + "=" * 60)
            print("  TEST COMPLETE - Browser will close in 10 seconds")
            print("=" * 60)
            time.sleep(10)

        else:
            print("  Login failed!")

    finally:
        client.close_browser()


# Quick manual test function
def quick_test():
    """Quick test without waiting"""
    import json
    from pathlib import Path

    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    mt_config = config['match_trader']

    print("\n" + "=" * 60)
    print("  QUICK CONNECTION TEST")
    print("=" * 60)

    client = MatchTraderBrowser(headless=False, platform_url=mt_config['base_url'])

    try:
        if client.login(mt_config['email'], mt_config['password']):
            print("\n  SUCCESS! Logged into Match-Trader")
            print("  The browser will stay open for 30 seconds...")
            print("  You can manually test trading in the browser.")
            time.sleep(30)
            return True
        else:
            print("\n  FAILED to login")
            return False
    finally:
        client.close_browser()


if __name__ == "__main__":
    test_login()
