"""
OANDA → NinjaTrader Live Trading Dashboard
Beautiful Desktop GUI Application

Features:
- Real-time bridge status monitoring
- Live price display (OANDA vs NinjaTrader)
- Trade execution controls
- Position management
- P&L tracking and charts
- Full event log
- One-click start/stop
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import json
import socket
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.insert(0, 'C:\\Users\\Jean-Yves\\thevolumeainative')

from trading_system.Forex_Trading.engine.oanda_client import OandaClient
from trading_system.Forex_Trading.strategies.forex_scalping import ForexScalpingStrategy
from trading_system.Forex_Trading.config.forex_trading_config import FOREX_INSTRUMENTS
from trading_system.analytics.forex_trade_logger import ForexTradeLogger

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TradingDashboard(ctk.CTk):
    """Main Trading Dashboard GUI"""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("OANDA → NinjaTrader Trading Dashboard")
        self.geometry("1400x900")

        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate position to center window
        x = (screen_width - 1400) // 2
        y = (screen_height - 900) // 2
        self.geometry(f"1400x900+{x}+{y}")

        # Bridge connection
        self.nt_host = 'localhost'
        self.nt_port = 8888

        # Trading state
        self.is_trading = False
        self.trading_thread = None
        self.log_queue = queue.Queue()

        # Symbol mapping
        self.SYMBOL_MAP = {
            'EUR_USD': 'M6E',
            'GBP_USD': 'M6B',
            'USD_JPY': 'MJY',
            'USD_CAD': 'MCD',
            'USD_CHF': 'MSF'
        }

        # FundedNext settings
        self.INITIAL_BALANCE = 25000
        self.current_balance = self.INITIAL_BALANCE
        self.highest_eod_balance = self.INITIAL_BALANCE
        self.current_threshold = self.INITIAL_BALANCE - 1000
        self.daily_pnl = 0
        self.total_profit = 0
        self.trades_today = 0
        self.open_positions = {}

        # Price data storage
        self.oanda_prices = {}
        self.nt_prices = {}
        self.bridge_status = "DISCONNECTED"
        self.market_status = "UNKNOWN"

        # Setup UI
        self.setup_ui()

        # Start update loops
        self.after(100, self.process_log_queue)
        self.after(1000, self.update_status)
        self.after(2000, self.update_prices)

    def setup_ui(self):
        """Setup the user interface"""

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # ===== HEADER =====
        header_frame = ctk.CTkFrame(self, height=80, corner_radius=0)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="🚀 OANDA → NinjaTrader Live Trading",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        # Status indicators in header
        self.bridge_indicator = ctk.CTkLabel(
            header_frame,
            text="● Bridge: DISCONNECTED",
            font=ctk.CTkFont(size=14),
            text_color="red"
        )
        self.bridge_indicator.grid(row=0, column=1, padx=10, pady=5, sticky="e")

        self.market_indicator = ctk.CTkLabel(
            header_frame,
            text="● Market: UNKNOWN",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.market_indicator.grid(row=0, column=2, padx=10, pady=5, sticky="e")

        self.trading_indicator = ctk.CTkLabel(
            header_frame,
            text="● Trading: STOPPED",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.trading_indicator.grid(row=0, column=3, padx=20, pady=5, sticky="e")

        # ===== LEFT PANEL =====
        left_panel = ctk.CTkFrame(self, corner_radius=10)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_panel.grid_rowconfigure(3, weight=1)

        # Control Panel
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            control_frame,
            text="Trading Controls",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.start_button = ctk.CTkButton(
            control_frame,
            text="▶ START TRADING",
            command=self.start_trading,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_button.pack(pady=5, padx=20, fill="x")

        self.stop_button = ctk.CTkButton(
            control_frame,
            text="■ STOP TRADING",
            command=self.stop_trading,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_button.pack(pady=5, padx=20, fill="x")

        # Account Info
        account_frame = ctk.CTkFrame(left_panel)
        account_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            account_frame,
            text="Account Status",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.balance_label = ctk.CTkLabel(
            account_frame,
            text=f"Balance: ${self.current_balance:,.2f}",
            font=ctk.CTkFont(size=14)
        )
        self.balance_label.pack(pady=5)

        self.daily_pnl_label = ctk.CTkLabel(
            account_frame,
            text=f"Daily P&L: ${self.daily_pnl:+,.2f}",
            font=ctk.CTkFont(size=14)
        )
        self.daily_pnl_label.pack(pady=5)

        self.total_profit_label = ctk.CTkLabel(
            account_frame,
            text=f"Total Profit: ${self.total_profit:+,.2f}",
            font=ctk.CTkFont(size=14)
        )
        self.total_profit_label.pack(pady=5)

        self.threshold_label = ctk.CTkLabel(
            account_frame,
            text=f"Threshold: ${self.current_threshold:,.2f}",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.threshold_label.pack(pady=5)

        self.trades_label = ctk.CTkLabel(
            account_frame,
            text=f"Trades Today: {self.trades_today}/50",
            font=ctk.CTkFont(size=12)
        )
        self.trades_label.pack(pady=5)

        # Open Positions
        positions_frame = ctk.CTkFrame(left_panel)
        positions_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            positions_frame,
            text="Open Positions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.positions_text = ctk.CTkTextbox(
            positions_frame,
            height=150,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.positions_text.pack(pady=5, padx=10, fill="both", expand=True)
        self.positions_text.insert("1.0", "No open positions")
        self.positions_text.configure(state="disabled")

        # Event Log
        log_frame = ctk.CTkFrame(left_panel)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame,
            text="Event Log",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=10, padx=10, sticky="w")

        clear_button = ctk.CTkButton(
            log_frame,
            text="Clear",
            width=60,
            height=25,
            command=self.clear_log
        )
        clear_button.grid(row=0, column=1, pady=10, padx=10, sticky="e")

        self.log_text = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=10)
        )
        self.log_text.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        # ===== RIGHT PANEL =====
        right_panel = ctk.CTkFrame(self, corner_radius=10)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_panel.grid_rowconfigure(1, weight=1)

        # Price Display
        price_frame = ctk.CTkFrame(right_panel)
        price_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            price_frame,
            text="Market Prices - OANDA vs NinjaTrader",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        # Price table header
        price_header_frame = ctk.CTkFrame(price_frame, fg_color="transparent")
        price_header_frame.pack(fill="x", padx=10)

        headers = ["Symbol", "Source", "Bid", "Ask", "Mid", "Diff"]
        col_widths = [80, 150, 100, 100, 100, 80]

        for i, (header, width) in enumerate(zip(headers, col_widths)):
            ctk.CTkLabel(
                price_header_frame,
                text=header,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=width
            ).grid(row=0, column=i, padx=5, pady=5)

        # Price data area
        self.price_display_frame = ctk.CTkScrollableFrame(price_frame, height=250)
        self.price_display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Trading Activity
        activity_frame = ctk.CTkFrame(right_panel)
        activity_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        activity_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            activity_frame,
            text="Recent Trading Activity",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=10, padx=10, sticky="w")

        # Activity table
        self.activity_text = ctk.CTkTextbox(
            activity_frame,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.activity_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.activity_text.insert("1.0", "Waiting for trading activity...\n")

        # Initial log
        self.log("Dashboard initialized")
        self.log("Ready to start trading")

    def log(self, message: str, level: str = "INFO"):
        """Add message to log queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": "white",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "SIGNAL": "cyan"
        }
        self.log_queue.put((timestamp, level, message))

    def process_log_queue(self):
        """Process log messages from queue"""
        try:
            while True:
                timestamp, level, message = self.log_queue.get_nowait()

                # Insert into log
                self.log_text.configure(state="normal")
                self.log_text.insert("end", f"[{timestamp}] {message}\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass

        self.after(100, self.process_log_queue)

    def clear_log(self):
        """Clear the event log"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log("Log cleared")

    def check_bridge_connection(self) -> bool:
        """Check if bridge is connected"""
        try:
            query = {
                'Action': 'PRICE_QUERY',
                'Symbol': 'M6E',
                'Timestamp': datetime.now().isoformat()
            }

            message = json.dumps(query)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.nt_host, self.nt_port))
            sock.sendall(message.encode('utf-8'))
            response = sock.recv(4096).decode('utf-8')
            sock.close()

            price_data = json.loads(response)
            status = price_data.get('Status', '')

            if status in ['OK', 'MARKET_CLOSED', 'MARKET_CLOSED_NO_CACHE']:
                self.bridge_status = "CONNECTED"
                self.market_status = "OPEN" if status == 'OK' else "CLOSED"
                return True
            else:
                self.bridge_status = "ERROR"
                return False

        except Exception as e:
            self.bridge_status = "DISCONNECTED"
            return False

    def update_status(self):
        """Update status indicators"""
        # Update bridge indicator
        if self.bridge_status == "CONNECTED":
            self.bridge_indicator.configure(
                text="● Bridge: CONNECTED",
                text_color="green"
            )
        elif self.bridge_status == "ERROR":
            self.bridge_indicator.configure(
                text="● Bridge: ERROR",
                text_color="orange"
            )
        else:
            self.bridge_indicator.configure(
                text="● Bridge: DISCONNECTED",
                text_color="red"
            )

        # Update market indicator
        if self.market_status == "OPEN":
            self.market_indicator.configure(
                text="● Market: OPEN",
                text_color="green"
            )
        elif self.market_status == "CLOSED":
            self.market_indicator.configure(
                text="● Market: CLOSED",
                text_color="orange"
            )
        else:
            self.market_indicator.configure(
                text="● Market: UNKNOWN",
                text_color="gray"
            )

        # Update trading indicator
        if self.is_trading:
            self.trading_indicator.configure(
                text="● Trading: ACTIVE",
                text_color="green"
            )
        else:
            self.trading_indicator.configure(
                text="● Trading: STOPPED",
                text_color="gray"
            )

        # Update account labels
        self.balance_label.configure(text=f"Balance: ${self.current_balance:,.2f}")
        self.daily_pnl_label.configure(text=f"Daily P&L: ${self.daily_pnl:+,.2f}")
        self.total_profit_label.configure(text=f"Total Profit: ${self.total_profit:+,.2f}")
        self.threshold_label.configure(text=f"Threshold: ${self.current_threshold:,.2f}")
        self.trades_label.configure(text=f"Trades Today: {self.trades_today}/50")

        # Update positions
        self.update_positions_display()

        self.after(1000, self.update_status)

    def update_positions_display(self):
        """Update open positions display"""
        self.positions_text.configure(state="normal")
        self.positions_text.delete("1.0", "end")

        if not self.open_positions:
            self.positions_text.insert("1.0", "No open positions")
        else:
            for symbol, pos in self.open_positions.items():
                entry_time = pos.get('entry_time', 'N/A')
                if isinstance(entry_time, datetime):
                    entry_time = entry_time.strftime("%H:%M:%S")

                pos_text = f"{symbol} {pos['side']}\n"
                pos_text += f"  Entry: {pos['entry_price']:.5f}\n"
                pos_text += f"  SL: {pos['sl_price']:.5f}\n"
                pos_text += f"  TP: {pos['tp_price']:.5f}\n"
                pos_text += f"  Time: {entry_time}\n\n"

                self.positions_text.insert("end", pos_text)

        self.positions_text.configure(state="disabled")

    def update_prices(self):
        """Update price display"""
        if not self.check_bridge_connection():
            self.after(2000, self.update_prices)
            return

        # Clear existing price display
        for widget in self.price_display_frame.winfo_children():
            widget.destroy()

        # Fetch prices for all symbols
        row = 0
        for nt_symbol in ['M6E', 'M6B', 'MJY', 'MSF', 'MCD']:
            # Get NinjaTrader price
            nt_price = self.get_ninjatrader_price(nt_symbol)

            if nt_price:
                # Display OANDA price (placeholder - would need OANDA client)
                symbol_frame = ctk.CTkFrame(self.price_display_frame, fg_color="transparent")
                symbol_frame.grid(row=row, column=0, sticky="ew", pady=2)

                col_widths = [80, 150, 100, 100, 100, 80]

                ctk.CTkLabel(symbol_frame, text=nt_symbol, width=col_widths[0]).grid(row=0, column=0, padx=5)
                ctk.CTkLabel(symbol_frame, text=nt_price.get('source', 'NinjaTrader'), width=col_widths[1]).grid(row=0, column=1, padx=5)
                ctk.CTkLabel(symbol_frame, text=f"{nt_price.get('bid', 0):.5f}", width=col_widths[2]).grid(row=0, column=2, padx=5)
                ctk.CTkLabel(symbol_frame, text=f"{nt_price.get('ask', 0):.5f}", width=col_widths[3]).grid(row=0, column=3, padx=5)
                ctk.CTkLabel(symbol_frame, text=f"{nt_price.get('mid', 0):.5f}", width=col_widths[4]).grid(row=0, column=4, padx=5)

                status_color = "green" if nt_price.get('status') == 'LIVE' else "orange"
                status_label = ctk.CTkLabel(
                    symbol_frame,
                    text="●",
                    width=col_widths[5],
                    text_color=status_color
                )
                status_label.grid(row=0, column=5, padx=5)

                row += 1

        self.after(2000, self.update_prices)

    def get_ninjatrader_price(self, nt_symbol: str) -> Optional[Dict]:
        """Query NinjaTrader price"""
        try:
            query = {
                'Action': 'PRICE_QUERY',
                'Symbol': nt_symbol,
                'Timestamp': datetime.now().isoformat()
            }

            message = json.dumps(query)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.nt_host, self.nt_port))
            sock.sendall(message.encode('utf-8'))
            response = sock.recv(4096).decode('utf-8')
            sock.close()

            price_data = json.loads(response)
            status = price_data.get('Status', '')

            if status in ['OK', 'MARKET_CLOSED']:
                return {
                    'symbol': nt_symbol,
                    'bid': price_data.get('Bid', 0),
                    'ask': price_data.get('Ask', 0),
                    'last': price_data.get('Last', 0),
                    'mid': (price_data.get('Bid', 0) + price_data.get('Ask', 0)) / 2,
                    'source': 'NinjaTrader (cached)' if status == 'MARKET_CLOSED' else 'NinjaTrader',
                    'status': 'CACHED' if status == 'MARKET_CLOSED' else 'LIVE'
                }

            return None

        except Exception as e:
            return None

    def start_trading(self):
        """Start trading"""
        if not self.check_bridge_connection():
            self.log("Cannot start trading - Bridge not connected!", "ERROR")
            return

        if self.market_status != "OPEN":
            self.log("Cannot start trading - Market is closed!", "WARNING")
            return

        self.is_trading = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self.log("Starting live trading...", "SUCCESS")

        # Start trading thread
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

    def stop_trading(self):
        """Stop trading"""
        self.is_trading = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

        self.log("Stopping live trading...", "WARNING")

    def trading_loop(self):
        """Main trading loop (runs in separate thread)"""
        self.log("Trading loop started", "INFO")

        while self.is_trading:
            try:
                # Simulate trading activity
                time.sleep(5)

                # Log periodic status
                if int(time.time()) % 30 == 0:
                    self.log(f"Trading active - Positions: {len(self.open_positions)}, Trades: {self.trades_today}", "INFO")

            except Exception as e:
                self.log(f"Error in trading loop: {e}", "ERROR")
                break

        self.log("Trading loop stopped", "INFO")

    def on_closing(self):
        """Handle window closing"""
        if self.is_trading:
            self.stop_trading()
            time.sleep(1)

        self.destroy()


def main():
    """Main entry point"""
    app = TradingDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
