"""
OANDA → NinjaTrader Live Trading Dashboard - ULTIMATE VERSION
Complete Tabbed Interface - Everything Organized

Features:
- Launch NinjaTrader & Bridge from dashboard
- LEFT TABS: Account, Risk, Positions, Prices, Settings
- RIGHT TABS: Bridge Logs, Strategy Logs, Trade History
- Real-time monitoring
- Full trading controls
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import json
import socket
import subprocess
import os
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


class UltimateTradingDashboard(ctk.CTk):
    """Ultimate Trading Dashboard with Complete Tabbed Interface"""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("🚀 ULTIMATE Trading Dashboard - OANDA → NinjaTrader")
        self.geometry("1800x1000")

        # Center window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1800) // 2
        y = (screen_height - 1000) // 2
        self.geometry(f"1800x1000+{x}+{y}")

        # Process handles
        self.ninjatrader_process = None
        self.bridge_process = None

        # Bridge connection
        self.nt_host = 'localhost'
        self.nt_port = 8888

        # Trading state
        self.is_trading = False
        self.trading_thread = None
        self.log_queue = queue.Queue()
        self.bridge_log_queue = queue.Queue()
        self.strategy_log_queue = queue.Queue()

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
        self.is_challenge_mode = True
        self.enable_consistency_rule = False

        # Price data storage
        self.oanda_prices = {}
        self.nt_prices = {}
        self.bridge_status = "DISCONNECTED"
        self.market_status = "UNKNOWN"

        # Trade history
        self.trade_history = []

        # Setup UI
        self.setup_ui()

        # Start update loops
        self.after(100, self.process_log_queues)
        self.after(1000, self.update_status)
        self.after(2000, self.update_prices)
        self.after(500, self.check_processes)

    def setup_ui(self):
        """Setup the ultimate tabbed interface"""

        # Configure grid
        self.grid_columnconfigure(0, weight=1)  # Left tabs
        self.grid_columnconfigure(1, weight=2)  # Right tabs (bigger)
        self.grid_rowconfigure(1, weight=1)

        # ===== HEADER =====
        header_frame = ctk.CTkFrame(self, height=100, corner_radius=0)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="🚀 ULTIMATE Trading Dashboard - OANDA → NinjaTrader",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        # Launch buttons
        launch_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        launch_frame.grid(row=1, column=0, padx=20, pady=5, sticky="w")

        self.launch_nt_button = ctk.CTkButton(
            launch_frame,
            text="🎯 Launch NinjaTrader",
            command=self.launch_ninjatrader,
            width=150,
            height=32,
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        self.launch_nt_button.pack(side="left", padx=5)

        self.launch_bridge_button = ctk.CTkButton(
            launch_frame,
            text="🌉 Launch Bridge",
            command=self.launch_bridge,
            width=150,
            height=32,
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        self.launch_bridge_button.pack(side="left", padx=5)

        # Trading controls in header
        control_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="e")

        self.start_button = ctk.CTkButton(
            control_frame,
            text="▶ START TRADING",
            command=self.start_trading,
            height=35,
            width=150,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=2)

        self.stop_button = ctk.CTkButton(
            control_frame,
            text="■ STOP TRADING",
            command=self.stop_trading,
            height=35,
            width=150,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_button.grid(row=1, column=0, padx=5, pady=2)

        # Status indicators
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky="e")

        self.ninjatrader_indicator = ctk.CTkLabel(
            status_frame,
            text="● NinjaTrader: NOT RUNNING",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.ninjatrader_indicator.grid(row=0, column=0, padx=10, pady=2, sticky="e")

        self.bridge_indicator = ctk.CTkLabel(
            status_frame,
            text="● Bridge: DISCONNECTED",
            font=ctk.CTkFont(size=11),
            text_color="red"
        )
        self.bridge_indicator.grid(row=0, column=1, padx=10, pady=2, sticky="e")

        self.market_indicator = ctk.CTkLabel(
            status_frame,
            text="● Market: UNKNOWN",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.market_indicator.grid(row=1, column=0, padx=10, pady=2, sticky="e")

        self.trading_indicator = ctk.CTkLabel(
            status_frame,
            text="● Trading: STOPPED",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.trading_indicator.grid(row=1, column=1, padx=10, pady=2, sticky="e")

        # ===== LEFT PANEL WITH TABS =====
        left_panel = ctk.CTkFrame(self, corner_radius=10)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        self.left_tabview = ctk.CTkTabview(left_panel)
        self.left_tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Left tabs
        self.left_tabview.add("Account")
        self.left_tabview.add("Risk")
        self.left_tabview.add("Positions")
        self.left_tabview.add("Prices")
        self.left_tabview.add("Settings")

        self.setup_account_tab()
        self.setup_risk_tab()
        self.setup_positions_tab()
        self.setup_prices_tab()
        self.setup_settings_tab()

        # ===== RIGHT PANEL WITH TABS =====
        right_panel = ctk.CTkFrame(self, corner_radius=10)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self.right_tabview = ctk.CTkTabview(right_panel)
        self.right_tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Right tabs
        self.right_tabview.add("Bridge Logs")
        self.right_tabview.add("Strategy Logs")
        self.right_tabview.add("Trade History")

        self.setup_bridge_logs_tab()
        self.setup_strategy_logs_tab()
        self.setup_trade_history_tab()

        # Initial logs
        self.log_to_bridge("[DASHBOARD] Ultimate dashboard initialized")
        self.log_to_strategy("[DASHBOARD] Ready to trade")

    def setup_account_tab(self):
        """Setup Account tab"""
        tab = self.left_tabview.tab("Account")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            tab,
            text="💰 Account Status",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=15)

        # Account info frame
        info_frame = ctk.CTkFrame(tab)
        info_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.balance_label = ctk.CTkLabel(
            info_frame,
            text=f"Balance: ${self.current_balance:,.2f}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.balance_label.pack(pady=10)

        self.daily_pnl_label = ctk.CTkLabel(
            info_frame,
            text=f"Daily P&L: ${self.daily_pnl:+,.2f}",
            font=ctk.CTkFont(size=14)
        )
        self.daily_pnl_label.pack(pady=5)

        self.total_profit_label = ctk.CTkLabel(
            info_frame,
            text=f"Total Profit: ${self.total_profit:+,.2f}",
            font=ctk.CTkFont(size=14)
        )
        self.total_profit_label.pack(pady=5)

        self.threshold_label = ctk.CTkLabel(
            info_frame,
            text=f"Threshold: ${self.current_threshold:,.2f}",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.threshold_label.pack(pady=5)

        buffer = self.current_balance - self.current_threshold
        self.buffer_label = ctk.CTkLabel(
            info_frame,
            text=f"Buffer: ${buffer:,.2f}",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.buffer_label.pack(pady=5)

        self.trades_label = ctk.CTkLabel(
            info_frame,
            text=f"Trades Today: {self.trades_today}/50",
            font=ctk.CTkFont(size=12)
        )
        self.trades_label.pack(pady=5)

    def setup_risk_tab(self):
        """Setup Risk Management tab"""
        tab = self.left_tabview.tab("Risk")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            tab,
            text="⚠️ Risk Management",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=15)

        # Risk info
        risk_frame = ctk.CTkScrollableFrame(tab)
        risk_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # FundedNext Rules
        ctk.CTkLabel(
            risk_frame,
            text="FundedNext Challenge Rules",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10, anchor="w")

        rules = [
            f"Initial Balance: ${self.INITIAL_BALANCE:,}",
            f"Profit Target: $1,250",
            f"Max Loss: $1,000 (trailing)",
            f"Daily Loss Limit: -$500",
            f"Max Concurrent: 5 positions",
            f"Contracts Per Trade: 1",
            f"Max Trades/Day: 50",
            f"Max Trades/Symbol: 10"
        ]

        for rule in rules:
            ctk.CTkLabel(
                risk_frame,
                text=f"• {rule}",
                font=ctk.CTkFont(size=11)
            ).pack(pady=2, anchor="w", padx=10)

        # Traded Symbols
        ctk.CTkLabel(
            risk_frame,
            text="\nTraded Symbols",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10, anchor="w")

        symbols = [
            "M6E - Micro EUR/USD",
            "M6B - Micro GBP/USD",
            "MJY - Micro USD/JPY",
            "MCD - Micro USD/CAD",
            "MSF - Micro USD/CHF"
        ]

        for symbol in symbols:
            ctk.CTkLabel(
                risk_frame,
                text=f"• {symbol}",
                font=ctk.CTkFont(size=11)
            ).pack(pady=2, anchor="w", padx=10)

    def setup_positions_tab(self):
        """Setup Open Positions tab"""
        tab = self.left_tabview.tab("Positions")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            tab,
            text="📊 Open Positions",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=15)

        # Positions display
        self.positions_text = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="none"
        )
        self.positions_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.positions_text.insert("1.0", "No open positions\n")

    def setup_prices_tab(self):
        """Setup Live Prices tab"""
        tab = self.left_tabview.tab("Prices")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            tab,
            text="💹 Live Prices",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=15)

        # Prices display
        self.price_text = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="none"
        )
        self.price_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    def setup_settings_tab(self):
        """Setup Settings tab"""
        tab = self.left_tabview.tab("Settings")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            tab,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, pady=15)

        # Settings frame
        settings_frame = ctk.CTkScrollableFrame(tab)
        settings_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Mode selection
        ctk.CTkLabel(
            settings_frame,
            text="Trading Mode",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10, anchor="w")

        mode_text = "CHALLENGE MODE" if self.is_challenge_mode else "FUNDED MODE"
        self.mode_label = ctk.CTkLabel(
            settings_frame,
            text=f"Current Mode: {mode_text}",
            font=ctk.CTkFont(size=12)
        )
        self.mode_label.pack(pady=5, anchor="w", padx=10)

        # Consistency rule
        ctk.CTkLabel(
            settings_frame,
            text="\nConsistency Rule",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10, anchor="w")

        consistency_text = "ENABLED" if self.enable_consistency_rule else "DISABLED"
        self.consistency_label = ctk.CTkLabel(
            settings_frame,
            text=f"40% Rule: {consistency_text}",
            font=ctk.CTkFont(size=12)
        )
        self.consistency_label.pack(pady=5, anchor="w", padx=10)

        self.daily_cap_label = ctk.CTkLabel(
            settings_frame,
            text="Daily Profit Cap: $400",
            font=ctk.CTkFont(size=12)
        )
        self.daily_cap_label.pack(pady=5, anchor="w", padx=10)

    def setup_bridge_logs_tab(self):
        """Setup Bridge Logs tab"""
        tab = self.right_tabview.tab("Bridge Logs")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="🌉 NinjaTrader Bridge Console",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            header,
            text="Clear",
            width=60,
            height=28,
            command=self.clear_bridge_log
        ).grid(row=0, column=1, sticky="e", padx=10)

        # Log text
        self.bridge_log_text = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="none"
        )
        self.bridge_log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def setup_strategy_logs_tab(self):
        """Setup Strategy Logs tab"""
        tab = self.right_tabview.tab("Strategy Logs")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="📊 Trading Strategy Console",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            header,
            text="Clear",
            width=60,
            height=28,
            command=self.clear_strategy_log
        ).grid(row=0, column=1, sticky="e", padx=10)

        # Log text
        self.strategy_log_text = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="none"
        )
        self.strategy_log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def setup_trade_history_tab(self):
        """Setup Trade History tab"""
        tab = self.right_tabview.tab("Trade History")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="📋 Trade History",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            header,
            text="Refresh",
            width=60,
            height=28,
            command=self.refresh_trade_history
        ).grid(row=0, column=1, sticky="e", padx=10)

        # History text
        self.history_text = ctk.CTkTextbox(
            tab,
            font=ctk.CTkFont(family="Consolas", size=10),
            wrap="none"
        )
        self.history_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.history_text.insert("1.0", "No trades yet...\n")

    # All the other methods remain the same...
    # (launch_ninjatrader, launch_bridge, log_to_bridge, etc.)

    def launch_ninjatrader(self):
        """Launch NinjaTrader 8"""
        try:
            nt_paths = [
                "C:\\Program Files\\NinjaTrader 8\\bin\\NinjaTrader.exe",
                "C:\\Program Files (x86)\\NinjaTrader 8\\bin\\NinjaTrader.exe"
            ]

            nt_exe = None
            for path in nt_paths:
                if os.path.exists(path):
                    nt_exe = path
                    break

            if not nt_exe:
                self.log_to_bridge("[ERROR] NinjaTrader not found", "ERROR")
                return

            self.log_to_bridge(f"[LAUNCH] Starting NinjaTrader...")
            self.ninjatrader_process = subprocess.Popen([nt_exe])
            self.log_to_bridge("[LAUNCH] ✓ NinjaTrader launched!", "SUCCESS")

        except Exception as e:
            self.log_to_bridge(f"[ERROR] Failed to launch NT: {e}", "ERROR")

    def launch_bridge(self):
        """Launch Bridge"""
        try:
            bridge_path = Path("C:/Users/Jean-Yves/thevolumeainative/trading_system/NinjaTrader_Bridge/NinjaTraderBridge.exe")

            if not bridge_path.exists():
                self.log_to_bridge("[ERROR] Bridge not found!", "ERROR")
                return

            self.log_to_bridge(f"[LAUNCH] Starting Bridge...")

            self.bridge_process = subprocess.Popen(
                [str(bridge_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            threading.Thread(target=self.read_bridge_output, daemon=True).start()
            self.log_to_bridge("[LAUNCH] ✓ Bridge launched!", "SUCCESS")

        except Exception as e:
            self.log_to_bridge(f"[ERROR] Failed to launch bridge: {e}", "ERROR")

    def read_bridge_output(self):
        """Read bridge output"""
        if not self.bridge_process:
            return

        try:
            for line in self.bridge_process.stdout:
                if line:
                    self.log_to_bridge(line.rstrip())
        except Exception as e:
            self.log_to_bridge(f"[ERROR] Bridge output error: {e}", "ERROR")

    def check_processes(self):
        """Check processes"""
        if self.ninjatrader_process and self.ninjatrader_process.poll() is None:
            self.ninjatrader_indicator.configure(text="● NinjaTrader: RUNNING", text_color="green")
        else:
            self.ninjatrader_indicator.configure(text="● NinjaTrader: NOT RUNNING", text_color="gray")

        self.after(500, self.check_processes)

    def log_to_bridge(self, message: str, level: str = "INFO"):
        """Log to bridge"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.bridge_log_queue.put((timestamp, level, message))

    def log_to_strategy(self, message: str, level: str = "INFO"):
        """Log to strategy"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.strategy_log_queue.put((timestamp, level, message))

    def process_log_queues(self):
        """Process logs"""
        try:
            while True:
                timestamp, level, message = self.bridge_log_queue.get_nowait()
                self.bridge_log_text.configure(state="normal")
                if not message.startswith("["):
                    self.bridge_log_text.insert("end", f"[{timestamp}] {message}\n")
                else:
                    self.bridge_log_text.insert("end", f"{message}\n")
                self.bridge_log_text.see("end")
                self.bridge_log_text.configure(state="disabled")
        except queue.Empty:
            pass

        try:
            while True:
                timestamp, level, message = self.strategy_log_queue.get_nowait()
                self.strategy_log_text.configure(state="normal")
                if not message.startswith("["):
                    self.strategy_log_text.insert("end", f"[{timestamp}] {message}\n")
                else:
                    self.strategy_log_text.insert("end", f"{message}\n")
                self.strategy_log_text.see("end")
                self.strategy_log_text.configure(state="disabled")
        except queue.Empty:
            pass

        self.after(100, self.process_log_queues)

    def clear_bridge_log(self):
        """Clear bridge log"""
        self.bridge_log_text.configure(state="normal")
        self.bridge_log_text.delete("1.0", "end")
        self.bridge_log_text.configure(state="disabled")
        self.log_to_bridge("[INFO] Log cleared")

    def clear_strategy_log(self):
        """Clear strategy log"""
        self.strategy_log_text.configure(state="normal")
        self.strategy_log_text.delete("1.0", "end")
        self.strategy_log_text.configure(state="disabled")
        self.log_to_strategy("[INFO] Log cleared")

    def refresh_trade_history(self):
        """Refresh trade history"""
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")

        if not self.trade_history:
            self.history_text.insert("1.0", "No trades yet...\n")
        else:
            header = f"{'Time':<10} {'Symbol':<6} {'Side':<4} {'Entry':<10} {'Exit':<10} {'P&L':<10}\n"
            self.history_text.insert("end", header)
            self.history_text.insert("end", "-" * 60 + "\n")

            for trade in self.trade_history[-50:]:
                line = f"{trade.get('time', 'N/A'):<10} "
                line += f"{trade.get('symbol', 'N/A'):<6} "
                line += f"{trade.get('side', 'N/A'):<4} "
                line += f"{trade.get('entry', 0):<10.5f} "
                line += f"{trade.get('exit', 0):<10.5f} "
                line += f"{trade.get('pnl', 0):<10.2f}\n"
                self.history_text.insert("end", line)

        self.history_text.configure(state="disabled")

    def check_bridge_connection(self) -> bool:
        """Check bridge"""
        try:
            query = {'Action': 'PRICE_QUERY', 'Symbol': 'M6E', 'Timestamp': datetime.now().isoformat()}
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
        """Update status"""
        if self.bridge_status == "CONNECTED":
            self.bridge_indicator.configure(text="● Bridge: CONNECTED", text_color="green")
        else:
            self.bridge_indicator.configure(text="● Bridge: DISCONNECTED", text_color="red")

        if self.market_status == "OPEN":
            self.market_indicator.configure(text="● Market: OPEN", text_color="green")
        else:
            self.market_indicator.configure(text="● Market: CLOSED", text_color="orange")

        if self.is_trading:
            self.trading_indicator.configure(text="● Trading: ACTIVE", text_color="green")
        else:
            self.trading_indicator.configure(text="● Trading: STOPPED", text_color="gray")

        # Update account labels
        self.balance_label.configure(text=f"Balance: ${self.current_balance:,.2f}")
        self.daily_pnl_label.configure(text=f"Daily P&L: ${self.daily_pnl:+,.2f}")
        self.total_profit_label.configure(text=f"Total Profit: ${self.total_profit:+,.2f}")
        self.threshold_label.configure(text=f"Threshold: ${self.current_threshold:,.2f}")
        buffer = self.current_balance - self.current_threshold
        self.buffer_label.configure(text=f"Buffer: ${buffer:,.2f}")
        self.trades_label.configure(text=f"Trades Today: {self.trades_today}/50")

        # Update positions
        self.update_positions_display()

        self.after(1000, self.update_status)

    def update_positions_display(self):
        """Update positions"""
        self.positions_text.configure(state="normal")
        self.positions_text.delete("1.0", "end")

        if not self.open_positions:
            self.positions_text.insert("1.0", "No open positions\n")
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
        """Update prices"""
        if not self.check_bridge_connection():
            self.after(2000, self.update_prices)
            return

        self.price_text.configure(state="normal")
        self.price_text.delete("1.0", "end")

        header = f"{'Symbol':<7} {'Bid':<10} {'Ask':<10} {'Status':<10}\n"
        self.price_text.insert("end", header)
        self.price_text.insert("end", "-" * 40 + "\n")

        for nt_symbol in ['M6E', 'M6B', 'MJY', 'MSF', 'MCD']:
            nt_price = self.get_ninjatrader_price(nt_symbol)

            if nt_price:
                status_text = "LIVE" if nt_price.get('status') == 'LIVE' else "CACHED"
                line = f"{nt_symbol:<7} {nt_price.get('bid', 0):<10.5f} {nt_price.get('ask', 0):<10.5f} {status_text:<10}\n"
                self.price_text.insert("end", line)

        self.price_text.configure(state="disabled")
        self.after(2000, self.update_prices)

    def get_ninjatrader_price(self, nt_symbol: str) -> Optional[Dict]:
        """Get price"""
        try:
            query = {'Action': 'PRICE_QUERY', 'Symbol': nt_symbol, 'Timestamp': datetime.now().isoformat()}
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
                    'status': 'CACHED' if status == 'MARKET_CLOSED' else 'LIVE'
                }

            return None

        except Exception as e:
            return None

    def start_trading(self):
        """Start trading"""
        if not self.check_bridge_connection():
            self.log_to_strategy("[ERROR] Bridge not connected!", "ERROR")
            return

        self.is_trading = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.log_to_strategy("[START] Trading started", "SUCCESS")

        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

    def stop_trading(self):
        """Stop trading"""
        self.is_trading = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.log_to_strategy("[STOP] Trading stopped", "WARNING")

    def trading_loop(self):
        """Trading loop"""
        loop_count = 0
        while self.is_trading:
            try:
                loop_count += 1
                time.sleep(10)

                if loop_count % 6 == 0:
                    self.log_to_strategy(f"[LOOP {loop_count}] Monitoring...", "INFO")

            except Exception as e:
                self.log_to_strategy(f"[ERROR] {e}", "ERROR")
                break

    def on_closing(self):
        """Handle closing"""
        if self.is_trading:
            self.stop_trading()
            time.sleep(1)

        if self.bridge_process and self.bridge_process.poll() is None:
            self.bridge_process.terminate()

        self.destroy()


def main():
    """Main"""
    app = UltimateTradingDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
