"""
OANDA → NinjaTrader Live Trading Dashboard - ENHANCED
Beautiful Desktop GUI Application with Full Control

Features:
- Launch NinjaTrader & Bridge from dashboard
- Tabbed interface (Bridge Logs, Strategy Logs, Trade History)
- Real-time log viewing for both Bridge and Strategy
- Live price display (OANDA vs NinjaTrader)
- Trade execution controls
- Position management
- P&L tracking
- Full event log
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

# Local config manager for standalone app
from app_config import AppConfig, get_config

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EnhancedTradingDashboard(ctk.CTk):
    """Enhanced Trading Dashboard with Launch Controls and Tabbed Logs"""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("🚀 OANDA → NinjaTrader Trading Dashboard - ENHANCED")
        self.geometry("1600x950")

        # Center window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1600) // 2
        y = (screen_height - 950) // 2
        self.geometry(f"1600x950+{x}+{y}")

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

        # Win/Loss tracking
        self.wins_today = 0
        self.losses_today = 0
        self.total_wins = 0
        self.total_losses = 0

        # Risk limits
        self.DAILY_LOSS_LIMIT = -500  # $500 daily loss limit
        self.PROFIT_TARGET = 1250     # $1,250 profit target
        self.MAX_DRAWDOWN = 1000      # $1,000 max drawdown (trailing)

        # Consistency tracking
        self.consistency_enabled = False
        self.best_day_profit = 0
        self.consistency_limit = 0  # 40% of profit target

        # Trading Parameters per Symbol (from pair_specific_settings.py)
        self.PAIR_SETTINGS = {
            'M6E': {'tp_pips': 20, 'sl_pips': 16, 'ts_trigger': 12, 'ts_distance': 6},   # EUR_USD
            'M6B': {'tp_pips': 30, 'sl_pips': 25, 'ts_trigger': 18, 'ts_distance': 8},   # GBP_USD
            'MJY': {'tp_pips': 18, 'sl_pips': 15, 'ts_trigger': 12, 'ts_distance': 6},   # USD_JPY
            'MCD': {'tp_pips': 20, 'sl_pips': 16, 'ts_trigger': 12, 'ts_distance': 6},   # USD_CAD
            'MSF': {'tp_pips': 15, 'sl_pips': 12, 'ts_trigger': 10, 'ts_distance': 5}    # USD_CHF
        }
        self.contract_size = 1
        self.running_mode = "Challenge"  # Challenge or Funded

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
        self.after(500, self.check_processes)

    def setup_ui(self):
        """Setup the enhanced user interface"""

        # Configure grid - Give more space to right side with tabs
        self.grid_columnconfigure(0, weight=0, minsize=480)  # Left panel fixed width - WIDER
        self.grid_columnconfigure(1, weight=1)  # Right panel gets all remaining space
        self.grid_rowconfigure(1, weight=1)

        # ===== HEADER =====
        header_frame = ctk.CTkFrame(self, height=100, corner_radius=0)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="🚀 OANDA → NinjaTrader Live Trading - ENHANCED",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        # Launch buttons in header
        launch_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        launch_frame.grid(row=1, column=0, padx=20, pady=5, sticky="w")

        self.launch_nt_button = ctk.CTkButton(
            launch_frame,
            text="🎯 Launch NinjaTrader",
            command=self.toggle_ninjatrader,
            width=160,
            height=32,
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        self.launch_nt_button.pack(side="left", padx=5)

        self.launch_bridge_button = ctk.CTkButton(
            launch_frame,
            text="🌉 Launch Bridge",
            command=self.toggle_bridge,
            width=160,
            height=32,
            fg_color="#1f538d",
            hover_color="#14375e"
        )
        self.launch_bridge_button.pack(side="left", padx=5)

        # Status indicators in header
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="e")

        self.ninjatrader_indicator = ctk.CTkLabel(
            status_frame,
            text="● NinjaTrader: NOT RUNNING",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.ninjatrader_indicator.grid(row=0, column=0, padx=10, pady=2, sticky="e")

        self.bridge_indicator = ctk.CTkLabel(
            status_frame,
            text="● Bridge: DISCONNECTED",
            font=ctk.CTkFont(size=13),
            text_color="red"
        )
        self.bridge_indicator.grid(row=0, column=1, padx=10, pady=2, sticky="e")

        self.market_indicator = ctk.CTkLabel(
            status_frame,
            text="● Market: UNKNOWN",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.market_indicator.grid(row=1, column=0, padx=10, pady=2, sticky="e")

        self.trading_indicator = ctk.CTkLabel(
            status_frame,
            text="● Trading: STOPPED",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.trading_indicator.grid(row=1, column=1, padx=10, pady=2, sticky="e")

        # ===== LEFT PANEL (SCROLLABLE) =====
        left_panel_container = ctk.CTkFrame(self, corner_radius=10)
        left_panel_container.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_panel_container.grid_rowconfigure(0, weight=1)
        left_panel_container.grid_columnconfigure(0, weight=1)

        # Create scrollable frame inside left panel
        left_panel = ctk.CTkScrollableFrame(left_panel_container, corner_radius=0)
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_columnconfigure(0, weight=1)  # Full width

        # Control Panel
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            control_frame,
            text="Trading Controls",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        # Single toggle button for Start/Stop Trading
        self.trading_button = ctk.CTkButton(
            control_frame,
            text="▶ START TRADING",
            command=self.toggle_trading,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.trading_button.pack(pady=10, padx=20, fill="x")

        # Account Info - Reorganized with grid layout
        account_frame = ctk.CTkFrame(left_panel)
        account_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        account_frame.grid_columnconfigure(0, weight=1)
        account_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            account_frame,
            text="💰 Account Status",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(10, 5))

        # Row 1: Balance and Total Profit
        ctk.CTkLabel(account_frame, text="Balance:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        self.balance_label = ctk.CTkLabel(account_frame, text=f"${self.current_balance:,.2f}", font=ctk.CTkFont(size=12, weight="bold"))
        self.balance_label.grid(row=1, column=1, sticky="e", padx=10, pady=2)

        # Row 2: Daily P&L with color coding
        ctk.CTkLabel(account_frame, text="Daily P&L:", font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=10, pady=2)
        pnl_color = "#51CF66" if self.daily_pnl >= 0 else "#FF6B6B"
        self.daily_pnl_label = ctk.CTkLabel(account_frame, text=f"${self.daily_pnl:+,.2f}", font=ctk.CTkFont(size=12, weight="bold"), text_color=pnl_color)
        self.daily_pnl_label.grid(row=2, column=1, sticky="e", padx=10, pady=2)

        # Row 3: Daily Loss Limit
        ctk.CTkLabel(account_frame, text="Daily Loss Limit:", font=ctk.CTkFont(size=11), text_color="gray").grid(row=3, column=0, sticky="w", padx=10, pady=2)
        self.daily_limit_label = ctk.CTkLabel(account_frame, text=f"${self.DAILY_LOSS_LIMIT:,.0f}", font=ctk.CTkFont(size=11), text_color="#FF6B6B")
        self.daily_limit_label.grid(row=3, column=1, sticky="e", padx=10, pady=2)

        # Row 4: Total Profit / Target
        ctk.CTkLabel(account_frame, text="Total Profit:", font=ctk.CTkFont(size=12)).grid(row=4, column=0, sticky="w", padx=10, pady=2)
        profit_color = "#51CF66" if self.total_profit >= 0 else "#FF6B6B"
        self.total_profit_label = ctk.CTkLabel(account_frame, text=f"${self.total_profit:+,.2f} / ${self.PROFIT_TARGET}", font=ctk.CTkFont(size=12, weight="bold"), text_color=profit_color)
        self.total_profit_label.grid(row=4, column=1, sticky="e", padx=10, pady=2)

        # Row 5: Trailing Threshold
        ctk.CTkLabel(account_frame, text="Trailing Threshold:", font=ctk.CTkFont(size=11), text_color="gray").grid(row=5, column=0, sticky="w", padx=10, pady=2)
        self.threshold_label = ctk.CTkLabel(account_frame, text=f"${self.current_threshold:,.2f}", font=ctk.CTkFont(size=11), text_color="#FFD43B")
        self.threshold_label.grid(row=5, column=1, sticky="e", padx=10, pady=2)

        # Separator
        ctk.CTkFrame(account_frame, height=2, fg_color="gray").grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

        # Row 7: Win/Loss Today
        ctk.CTkLabel(account_frame, text="📊 Today's Trades:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        # Row 8: Wins/Losses
        win_loss_frame = ctk.CTkFrame(account_frame, fg_color="transparent")
        win_loss_frame.grid(row=8, column=0, columnspan=2, sticky="ew", padx=10, pady=2)
        win_loss_frame.grid_columnconfigure(0, weight=1)
        win_loss_frame.grid_columnconfigure(1, weight=1)
        win_loss_frame.grid_columnconfigure(2, weight=1)

        self.wins_label = ctk.CTkLabel(win_loss_frame, text=f"✅ {self.wins_today}W", font=ctk.CTkFont(size=12, weight="bold"), text_color="#51CF66")
        self.wins_label.grid(row=0, column=0, padx=5)

        self.losses_label = ctk.CTkLabel(win_loss_frame, text=f"❌ {self.losses_today}L", font=ctk.CTkFont(size=12, weight="bold"), text_color="#FF6B6B")
        self.losses_label.grid(row=0, column=1, padx=5)

        win_rate = (self.wins_today / max(1, self.wins_today + self.losses_today)) * 100
        self.winrate_label = ctk.CTkLabel(win_loss_frame, text=f"({win_rate:.0f}%)", font=ctk.CTkFont(size=11), text_color="gray")
        self.winrate_label.grid(row=0, column=2, padx=5)

        # Row 9: Total trades count
        ctk.CTkLabel(account_frame, text="Trades Today:", font=ctk.CTkFont(size=11)).grid(row=9, column=0, sticky="w", padx=10, pady=2)
        self.trades_label = ctk.CTkLabel(account_frame, text=f"{self.trades_today}/50", font=ctk.CTkFont(size=11, weight="bold"))
        self.trades_label.grid(row=9, column=1, sticky="e", padx=10, pady=2)

        # Separator
        ctk.CTkFrame(account_frame, height=2, fg_color="gray").grid(row=10, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

        # Row 11: Consistency Rule
        ctk.CTkLabel(account_frame, text="📏 Consistency Rule:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=11, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        consistency_status = "ENABLED" if self.consistency_enabled else "DISABLED"
        consistency_color = "#51CF66" if self.consistency_enabled else "gray"
        self.consistency_label = ctk.CTkLabel(account_frame, text=consistency_status, font=ctk.CTkFont(size=11, weight="bold"), text_color=consistency_color)
        self.consistency_label.grid(row=12, column=0, sticky="w", padx=10, pady=2)

        # Best day profit (40% rule)
        self.best_day_label = ctk.CTkLabel(account_frame, text=f"Best Day: ${self.best_day_profit:,.2f}", font=ctk.CTkFont(size=10), text_color="gray")
        self.best_day_label.grid(row=12, column=1, sticky="e", padx=10, pady=(2, 10))

        # Trading Parameters Section - Per Symbol
        params_frame = ctk.CTkFrame(left_panel)
        params_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        params_frame.grid_columnconfigure(0, weight=1)
        params_frame.grid_columnconfigure(1, weight=1)
        params_frame.grid_columnconfigure(2, weight=1)
        params_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(
            params_frame,
            text="⚙️ Trading Parameters (per Symbol)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=4, pady=10, padx=10)

        # Header row
        ctk.CTkLabel(params_frame, text="Symbol", font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=3, pady=3)
        ctk.CTkLabel(params_frame, text="TP", font=ctk.CTkFont(size=11, weight="bold"), text_color="#51CF66").grid(row=1, column=1, padx=3, pady=3)
        ctk.CTkLabel(params_frame, text="SL", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FF6B6B").grid(row=1, column=2, padx=3, pady=3)
        ctk.CTkLabel(params_frame, text="TS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#FFD43B").grid(row=1, column=3, padx=3, pady=3)

        # Per-symbol settings
        row = 2
        for symbol, settings in self.PAIR_SETTINGS.items():
            ctk.CTkLabel(params_frame, text=symbol, font=ctk.CTkFont(size=11, weight="bold"), text_color="#74C0FC").grid(row=row, column=0, padx=3, pady=2)
            ctk.CTkLabel(params_frame, text=f"{settings['tp_pips']}p", font=ctk.CTkFont(size=11), text_color="#51CF66").grid(row=row, column=1, padx=3, pady=2)
            ctk.CTkLabel(params_frame, text=f"{settings['sl_pips']}p", font=ctk.CTkFont(size=11), text_color="#FF6B6B").grid(row=row, column=2, padx=3, pady=2)
            ctk.CTkLabel(params_frame, text=f"{settings['ts_trigger']}p/{settings['ts_distance']}p", font=ctk.CTkFont(size=10), text_color="#FFD43B").grid(row=row, column=3, padx=3, pady=2)
            row += 1

        # Contract Size
        ctk.CTkLabel(params_frame, text="Contract Size:", font=ctk.CTkFont(size=13)).grid(row=row, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 3))
        self.contract_label = ctk.CTkLabel(params_frame, text=f"{self.contract_size}", font=ctk.CTkFont(size=13, weight="bold"), text_color="#74C0FC")
        self.contract_label.grid(row=row, column=3, sticky="e", padx=15, pady=(10, 3))

        # Running Mode Section
        mode_frame = ctk.CTkFrame(left_panel)
        mode_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        mode_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            mode_frame,
            text="🎯 Running Mode",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        ctk.CTkLabel(mode_frame, text="Current Mode:", font=ctk.CTkFont(size=13)).grid(row=1, column=0, sticky="w", padx=15, pady=5)
        mode_color = "#FFD43B" if self.running_mode == "Challenge" else "#51CF66"
        self.mode_label = ctk.CTkLabel(mode_frame, text=f"{self.running_mode}", font=ctk.CTkFont(size=14, weight="bold"), text_color=mode_color)
        self.mode_label.grid(row=1, column=1, sticky="e", padx=15, pady=5)

        # Symbols being traded
        ctk.CTkLabel(mode_frame, text="Symbols:", font=ctk.CTkFont(size=13)).grid(row=2, column=0, sticky="w", padx=15, pady=5)
        symbols_text = "M6E, M6B, MJY, MCD, MSF"
        self.symbols_label = ctk.CTkLabel(mode_frame, text=symbols_text, font=ctk.CTkFont(size=11), text_color="gray")
        self.symbols_label.grid(row=2, column=1, sticky="e", padx=15, pady=5)

        # ===== RIGHT PANEL WITH TABS =====
        right_panel = ctk.CTkFrame(self, corner_radius=10)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # Create tabview - FILL ALL SPACE
        self.tabview = ctk.CTkTabview(right_panel)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Add tabs
        self.tabview.add("Bridge Logs")
        self.tabview.add("Strategy Logs")
        self.tabview.add("Open Positions")
        self.tabview.add("Trade History")
        self.tabview.add("Settings")

        # === BRIDGE LOGS TAB ===
        bridge_tab = self.tabview.tab("Bridge Logs")
        bridge_tab.grid_rowconfigure(1, weight=1)
        bridge_tab.grid_columnconfigure(0, weight=1)

        bridge_header = ctk.CTkFrame(bridge_tab, fg_color="transparent", height=40)
        bridge_header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        bridge_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            bridge_header,
            text="🌉 NinjaTrader Bridge Console",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            bridge_header,
            text="Clear",
            width=60,
            height=28,
            command=self.clear_bridge_log
        ).grid(row=0, column=1, sticky="e", padx=10)

        self.bridge_log_text = ctk.CTkTextbox(
            bridge_tab,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="none"
        )
        self.bridge_log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.bridge_log_text.insert("1.0", "[BRIDGE] Waiting for bridge to start...\n")

        # === STRATEGY LOGS TAB ===
        strategy_tab = self.tabview.tab("Strategy Logs")
        strategy_tab.grid_rowconfigure(1, weight=1)
        strategy_tab.grid_columnconfigure(0, weight=1)

        strategy_header = ctk.CTkFrame(strategy_tab, fg_color="transparent", height=40)
        strategy_header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        strategy_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            strategy_header,
            text="📊 Trading Strategy Console",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            strategy_header,
            text="Clear",
            width=60,
            height=28,
            command=self.clear_strategy_log
        ).grid(row=0, column=1, sticky="e", padx=10)

        self.strategy_log_text = ctk.CTkTextbox(
            strategy_tab,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="none"
        )
        self.strategy_log_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.strategy_log_text.insert("1.0", "[STRATEGY] Waiting for trading to start...\n")

        # === OPEN POSITIONS TAB ===
        positions_tab = self.tabview.tab("Open Positions")
        positions_tab.grid_rowconfigure(1, weight=1)
        positions_tab.grid_columnconfigure(0, weight=1)

        positions_header = ctk.CTkFrame(positions_tab, fg_color="transparent", height=40)
        positions_header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        positions_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            positions_header,
            text="📊 Open Positions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            positions_header,
            text="Refresh",
            width=60,
            height=28,
            command=self.update_positions
        ).grid(row=0, column=1, sticky="e", padx=10)

        self.positions_text = ctk.CTkTextbox(
            positions_tab,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="none"
        )
        self.positions_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.positions_text.insert("1.0", "No open positions\n")

        # === TRADE HISTORY TAB ===
        history_tab = self.tabview.tab("Trade History")
        history_tab.grid_rowconfigure(1, weight=1)
        history_tab.grid_columnconfigure(0, weight=1)

        history_header = ctk.CTkFrame(history_tab, fg_color="transparent", height=40)
        history_header.grid(row=0, column=0, sticky="ew", pady=(5, 5))
        history_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            history_header,
            text="📋 Trade History",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        ctk.CTkButton(
            history_header,
            text="Refresh",
            width=60,
            height=28,
            command=self.refresh_trade_history
        ).grid(row=0, column=1, sticky="e", padx=10)

        self.history_text = ctk.CTkTextbox(
            history_tab,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="none"
        )
        self.history_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.history_text.insert("1.0", "No trades yet...\n")

        # === SETTINGS TAB ===
        self.setup_settings_tab()

        # Initial log
        self.log_to_bridge("[DASHBOARD] Dashboard initialized and ready")
        self.log_to_strategy("[DASHBOARD] Strategy engine ready to start")

    def setup_settings_tab(self):
        """Setup the Settings tab with configuration options"""
        settings_tab = self.tabview.tab("Settings")
        settings_tab.grid_columnconfigure(0, weight=1)
        settings_tab.grid_rowconfigure(0, weight=1)

        # Load config
        self.config = get_config()

        # Create scrollable frame for settings
        settings_scroll = ctk.CTkScrollableFrame(settings_tab)
        settings_scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        settings_scroll.grid_columnconfigure(0, weight=1)

        # ===== OANDA API CONFIGURATION =====
        oanda_frame = ctk.CTkFrame(settings_scroll)
        oanda_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        oanda_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            oanda_frame,
            text="🔑 OANDA API Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        # Mode toggle (Practice/Live)
        ctk.CTkLabel(oanda_frame, text="Trading Mode:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", padx=15, pady=5)
        self.mode_switch_var = ctk.StringVar(value="Live" if self.config.oanda.use_live else "Practice")
        self.mode_switch = ctk.CTkSegmentedButton(
            oanda_frame,
            values=["Practice", "Live"],
            variable=self.mode_switch_var,
            command=self.on_mode_change
        )
        self.mode_switch.grid(row=1, column=1, sticky="e", padx=15, pady=5)

        # Practice Account Settings
        ctk.CTkLabel(
            oanda_frame,
            text="Practice Account",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFD43B"
        ).grid(row=2, column=0, columnspan=2, pady=(15, 5), padx=15, sticky="w")

        ctk.CTkLabel(oanda_frame, text="API Key:", font=ctk.CTkFont(size=12)).grid(row=3, column=0, sticky="w", padx=15, pady=3)
        self.practice_api_key_entry = ctk.CTkEntry(oanda_frame, placeholder_text="Enter Practice API Key", show="*", width=350)
        self.practice_api_key_entry.grid(row=3, column=1, sticky="ew", padx=15, pady=3)
        if self.config.oanda.practice_api_key:
            self.practice_api_key_entry.insert(0, self.config.oanda.practice_api_key)

        ctk.CTkLabel(oanda_frame, text="Account ID:", font=ctk.CTkFont(size=12)).grid(row=4, column=0, sticky="w", padx=15, pady=3)
        self.practice_account_entry = ctk.CTkEntry(oanda_frame, placeholder_text="Enter Practice Account ID", width=350)
        self.practice_account_entry.grid(row=4, column=1, sticky="ew", padx=15, pady=3)
        if self.config.oanda.practice_account_id:
            self.practice_account_entry.insert(0, self.config.oanda.practice_account_id)

        # Live Account Settings
        ctk.CTkLabel(
            oanda_frame,
            text="Live Account",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FF6B6B"
        ).grid(row=5, column=0, columnspan=2, pady=(15, 5), padx=15, sticky="w")

        ctk.CTkLabel(oanda_frame, text="API Key:", font=ctk.CTkFont(size=12)).grid(row=6, column=0, sticky="w", padx=15, pady=3)
        self.live_api_key_entry = ctk.CTkEntry(oanda_frame, placeholder_text="Enter Live API Key", show="*", width=350)
        self.live_api_key_entry.grid(row=6, column=1, sticky="ew", padx=15, pady=3)
        if self.config.oanda.live_api_key:
            self.live_api_key_entry.insert(0, self.config.oanda.live_api_key)

        ctk.CTkLabel(oanda_frame, text="Account ID:", font=ctk.CTkFont(size=12)).grid(row=7, column=0, sticky="w", padx=15, pady=3)
        self.live_account_entry = ctk.CTkEntry(oanda_frame, placeholder_text="Enter Live Account ID", width=350)
        self.live_account_entry.grid(row=7, column=1, sticky="ew", padx=15, pady=3)
        if self.config.oanda.live_account_id:
            self.live_account_entry.insert(0, self.config.oanda.live_account_id)

        # ===== RISK MANAGEMENT =====
        risk_frame = ctk.CTkFrame(settings_scroll)
        risk_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        risk_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            risk_frame,
            text="⚠️ Risk Management",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        ctk.CTkLabel(risk_frame, text="Initial Balance ($):", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", padx=15, pady=3)
        self.initial_balance_entry = ctk.CTkEntry(risk_frame, width=150)
        self.initial_balance_entry.grid(row=1, column=1, sticky="w", padx=15, pady=3)
        self.initial_balance_entry.insert(0, str(self.config.risk.initial_balance))

        ctk.CTkLabel(risk_frame, text="Daily Loss Limit ($):", font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=15, pady=3)
        self.daily_loss_entry = ctk.CTkEntry(risk_frame, width=150)
        self.daily_loss_entry.grid(row=2, column=1, sticky="w", padx=15, pady=3)
        self.daily_loss_entry.insert(0, str(self.config.risk.daily_loss_limit))

        ctk.CTkLabel(risk_frame, text="Profit Target ($):", font=ctk.CTkFont(size=12)).grid(row=3, column=0, sticky="w", padx=15, pady=3)
        self.profit_target_entry = ctk.CTkEntry(risk_frame, width=150)
        self.profit_target_entry.grid(row=3, column=1, sticky="w", padx=15, pady=3)
        self.profit_target_entry.insert(0, str(self.config.risk.profit_target))

        ctk.CTkLabel(risk_frame, text="Max Drawdown ($):", font=ctk.CTkFont(size=12)).grid(row=4, column=0, sticky="w", padx=15, pady=3)
        self.max_drawdown_entry = ctk.CTkEntry(risk_frame, width=150)
        self.max_drawdown_entry.grid(row=4, column=1, sticky="w", padx=15, pady=3)
        self.max_drawdown_entry.insert(0, str(self.config.risk.max_drawdown))

        ctk.CTkLabel(risk_frame, text="Max Trades/Day:", font=ctk.CTkFont(size=12)).grid(row=5, column=0, sticky="w", padx=15, pady=3)
        self.max_trades_entry = ctk.CTkEntry(risk_frame, width=150)
        self.max_trades_entry.grid(row=5, column=1, sticky="w", padx=15, pady=3)
        self.max_trades_entry.insert(0, str(self.config.risk.max_trades_per_day))

        # Consistency Rule Toggle
        ctk.CTkLabel(risk_frame, text="Consistency Rule:", font=ctk.CTkFont(size=12)).grid(row=6, column=0, sticky="w", padx=15, pady=3)
        self.consistency_switch = ctk.CTkSwitch(risk_frame, text="Enabled" if self.config.risk.consistency_rule_enabled else "Disabled")
        self.consistency_switch.grid(row=6, column=1, sticky="w", padx=15, pady=3)
        if self.config.risk.consistency_rule_enabled:
            self.consistency_switch.select()

        # ===== PATH CONFIGURATION =====
        paths_frame = ctk.CTkFrame(settings_scroll)
        paths_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        paths_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            paths_frame,
            text="📁 Path Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        ctk.CTkLabel(paths_frame, text="NinjaTrader Path:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w", padx=15, pady=3)
        self.nt_path_entry = ctk.CTkEntry(paths_frame, width=400)
        self.nt_path_entry.grid(row=1, column=1, sticky="ew", padx=15, pady=3)
        self.nt_path_entry.insert(0, self.config.paths.ninjatrader_path)

        ctk.CTkLabel(paths_frame, text="Bridge Path:", font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=15, pady=3)
        self.bridge_path_entry = ctk.CTkEntry(paths_frame, width=400)
        self.bridge_path_entry.grid(row=2, column=1, sticky="ew", padx=15, pady=3)
        self.bridge_path_entry.insert(0, self.config.paths.bridge_path)

        ctk.CTkLabel(paths_frame, text="Logs Directory:", font=ctk.CTkFont(size=12)).grid(row=3, column=0, sticky="w", padx=15, pady=3)
        self.logs_path_entry = ctk.CTkEntry(paths_frame, width=400)
        self.logs_path_entry.grid(row=3, column=1, sticky="ew", padx=15, pady=3)
        self.logs_path_entry.insert(0, self.config.paths.logs_directory)

        # ===== AUTO-START OPTIONS =====
        autostart_frame = ctk.CTkFrame(settings_scroll)
        autostart_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        autostart_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            autostart_frame,
            text="🚀 Auto-Start Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        self.auto_nt_switch = ctk.CTkSwitch(autostart_frame, text="Auto-launch NinjaTrader")
        self.auto_nt_switch.grid(row=1, column=0, columnspan=2, sticky="w", padx=15, pady=5)
        if self.config.auto_start.auto_launch_ninjatrader:
            self.auto_nt_switch.select()

        self.auto_bridge_switch = ctk.CTkSwitch(autostart_frame, text="Auto-launch Bridge")
        self.auto_bridge_switch.grid(row=2, column=0, columnspan=2, sticky="w", padx=15, pady=5)
        if self.config.auto_start.auto_launch_bridge:
            self.auto_bridge_switch.select()

        self.auto_trading_switch = ctk.CTkSwitch(autostart_frame, text="Auto-start Trading")
        self.auto_trading_switch.grid(row=3, column=0, columnspan=2, sticky="w", padx=15, pady=5)
        if self.config.auto_start.auto_start_trading:
            self.auto_trading_switch.select()

        # ===== SAVE BUTTON =====
        button_frame = ctk.CTkFrame(settings_scroll, fg_color="transparent")
        button_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=20)
        button_frame.grid_columnconfigure(0, weight=1)

        self.save_settings_button = ctk.CTkButton(
            button_frame,
            text="💾 Save Settings",
            command=self.save_settings,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#51CF66",
            hover_color="#40A050"
        )
        self.save_settings_button.grid(row=0, column=0, pady=10)

        # Status label
        self.settings_status_label = ctk.CTkLabel(
            button_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.settings_status_label.grid(row=1, column=0, pady=5)

    def on_mode_change(self, value):
        """Handle trading mode change"""
        self.config.oanda.use_live = (value == "Live")

    def save_settings(self):
        """Save all settings to config file"""
        try:
            # OANDA Settings
            self.config.oanda.practice_api_key = self.practice_api_key_entry.get()
            self.config.oanda.practice_account_id = self.practice_account_entry.get()
            self.config.oanda.live_api_key = self.live_api_key_entry.get()
            self.config.oanda.live_account_id = self.live_account_entry.get()
            self.config.oanda.use_live = (self.mode_switch_var.get() == "Live")

            # Risk Settings
            self.config.risk.initial_balance = float(self.initial_balance_entry.get())
            self.config.risk.daily_loss_limit = float(self.daily_loss_entry.get())
            self.config.risk.profit_target = float(self.profit_target_entry.get())
            self.config.risk.max_drawdown = float(self.max_drawdown_entry.get())
            self.config.risk.max_trades_per_day = int(self.max_trades_entry.get())
            self.config.risk.consistency_rule_enabled = self.consistency_switch.get()

            # Path Settings
            self.config.paths.ninjatrader_path = self.nt_path_entry.get()
            self.config.paths.bridge_path = self.bridge_path_entry.get()
            self.config.paths.logs_directory = self.logs_path_entry.get()

            # Auto-start Settings
            self.config.auto_start.auto_launch_ninjatrader = self.auto_nt_switch.get()
            self.config.auto_start.auto_launch_bridge = self.auto_bridge_switch.get()
            self.config.auto_start.auto_start_trading = self.auto_trading_switch.get()

            # Save to file
            if self.config.save():
                self.settings_status_label.configure(text="✅ Settings saved successfully!", text_color="#51CF66")
                self.log_to_strategy("[CONFIG] Settings saved successfully", "SUCCESS")

                # Update dashboard values from config
                self.INITIAL_BALANCE = self.config.risk.initial_balance
                self.DAILY_LOSS_LIMIT = -self.config.risk.daily_loss_limit
                self.PROFIT_TARGET = self.config.risk.profit_target
                self.MAX_DRAWDOWN = self.config.risk.max_drawdown
            else:
                self.settings_status_label.configure(text="❌ Failed to save settings", text_color="#FF6B6B")

        except ValueError as e:
            self.settings_status_label.configure(text=f"❌ Invalid value: {e}", text_color="#FF6B6B")
        except Exception as e:
            self.settings_status_label.configure(text=f"❌ Error: {e}", text_color="#FF6B6B")

    def toggle_ninjatrader(self):
        """Toggle NinjaTrader - Launch or Stop"""
        if self.ninjatrader_process and self.ninjatrader_process.poll() is None:
            # Process is running - stop it
            self.stop_ninjatrader()
        else:
            # Process not running - launch it
            self.launch_ninjatrader()

    def toggle_bridge(self):
        """Toggle Bridge - Launch or Stop"""
        if self.bridge_process and self.bridge_process.poll() is None:
            # Process is running - stop it
            self.stop_bridge()
        else:
            # Process not running - launch it
            self.launch_bridge()

    def launch_ninjatrader(self):
        """Launch NinjaTrader 8"""
        try:
            # Common NinjaTrader installation paths
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
                self.log_to_bridge("[ERROR] NinjaTrader not found in default locations", "ERROR")
                self.log_to_bridge("[ERROR] Please launch NinjaTrader manually", "ERROR")
                return

            self.log_to_bridge(f"[LAUNCH] Starting NinjaTrader from {nt_exe}...")
            self.ninjatrader_process = subprocess.Popen([nt_exe])
            self.log_to_bridge("[LAUNCH] ✓ NinjaTrader launched successfully!", "SUCCESS")

            # Update button to show Stop
            self.launch_nt_button.configure(text="⏹ Stop NinjaTrader", fg_color="#8B0000", hover_color="#5C0000")

        except Exception as e:
            self.log_to_bridge(f"[ERROR] Failed to launch NinjaTrader: {e}", "ERROR")

    def stop_ninjatrader(self):
        """Stop NinjaTrader process"""
        try:
            if self.ninjatrader_process:
                self.log_to_bridge("[STOP] Stopping NinjaTrader...")
                self.ninjatrader_process.terminate()
                self.ninjatrader_process = None
                self.log_to_bridge("[STOP] ✓ NinjaTrader stopped", "SUCCESS")

                # Update button to show Launch
                self.launch_nt_button.configure(text="🎯 Launch NinjaTrader", fg_color="#1f538d", hover_color="#14375e")
        except Exception as e:
            self.log_to_bridge(f"[ERROR] Failed to stop NinjaTrader: {e}", "ERROR")

    def launch_bridge(self):
        """Launch NinjaTrader Bridge"""
        try:
            bridge_path = Path("C:/Users/Jean-Yves/thevolumeainative/trading_system/NinjaTrader_Bridge/NinjaTraderBridge.exe")

            if not bridge_path.exists():
                self.log_to_bridge("[ERROR] Bridge executable not found!", "ERROR")
                self.log_to_bridge(f"[ERROR] Expected: {bridge_path}", "ERROR")
                return

            self.log_to_bridge(f"[LAUNCH] Starting NinjaTrader Bridge...")

            # Launch bridge and capture output
            self.bridge_process = subprocess.Popen(
                [str(bridge_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Start thread to read bridge output
            bridge_output_thread = threading.Thread(
                target=self.read_bridge_output,
                daemon=True
            )
            bridge_output_thread.start()

            self.log_to_bridge("[LAUNCH] ✓ Bridge launched successfully!", "SUCCESS")
            self.log_to_bridge("[LAUNCH] Monitoring bridge console output...", "INFO")

            # Update button to show Stop
            self.launch_bridge_button.configure(text="⏹ Stop Bridge", fg_color="#8B0000", hover_color="#5C0000")

        except Exception as e:
            self.log_to_bridge(f"[ERROR] Failed to launch bridge: {e}", "ERROR")

    def stop_bridge(self):
        """Stop Bridge process"""
        try:
            if self.bridge_process:
                self.log_to_bridge("[STOP] Stopping Bridge...")
                self.bridge_process.terminate()
                self.bridge_process = None
                self.log_to_bridge("[STOP] ✓ Bridge stopped", "SUCCESS")

                # Update button to show Launch
                self.launch_bridge_button.configure(text="🌉 Launch Bridge", fg_color="#1f538d", hover_color="#14375e")
        except Exception as e:
            self.log_to_bridge(f"[ERROR] Failed to stop bridge: {e}", "ERROR")

    def read_bridge_output(self):
        """Read bridge process output in real-time"""
        if not self.bridge_process:
            return

        try:
            for line in self.bridge_process.stdout:
                if line:
                    stripped = line.rstrip()
                    if stripped:  # Skip empty lines
                        self.log_to_bridge(stripped)
        except Exception as e:
            self.log_to_bridge(f"[ERROR] Bridge output stream error: {e}", "ERROR")

    def check_processes(self):
        """Check if launched processes are still running"""
        # Check NinjaTrader
        if self.ninjatrader_process and self.ninjatrader_process.poll() is None:
            # Process is running
            self.ninjatrader_indicator.configure(
                text="● NinjaTrader: RUNNING",
                text_color="green"
            )
            # Make sure button shows Stop
            self.launch_nt_button.configure(text="⏹ Stop NinjaTrader", fg_color="#8B0000", hover_color="#5C0000")
        elif self.ninjatrader_process:
            # Process exited
            self.ninjatrader_indicator.configure(
                text="● NinjaTrader: STOPPED",
                text_color="gray"
            )
            # Reset button to Launch
            self.launch_nt_button.configure(text="🎯 Launch NinjaTrader", fg_color="#1f538d", hover_color="#14375e")
            self.ninjatrader_process = None
        else:
            # Not launched from dashboard
            self.ninjatrader_indicator.configure(
                text="● NinjaTrader: NOT RUNNING",
                text_color="gray"
            )
            # Ensure button shows Launch
            self.launch_nt_button.configure(text="🎯 Launch NinjaTrader", fg_color="#1f538d", hover_color="#14375e")

        # Check Bridge
        if self.bridge_process and self.bridge_process.poll() is None:
            # Process is running
            # Make sure button shows Stop
            self.launch_bridge_button.configure(text="⏹ Stop Bridge", fg_color="#8B0000", hover_color="#5C0000")
            # Check if bridge is actually connected (can communicate)
            if self.check_bridge_connection():
                self.bridge_status = "CONNECTED"
            else:
                self.bridge_status = "RUNNING"  # Process running but not yet connected
        elif self.bridge_process:
            # Process exited
            self.bridge_status = "STOPPED"
            self.log_to_bridge("[WARNING] Bridge process stopped!", "WARNING")
            # Reset button to Launch
            self.launch_bridge_button.configure(text="🌉 Launch Bridge", fg_color="#1f538d", hover_color="#14375e")
            self.bridge_process = None
        else:
            # Not launched
            self.launch_bridge_button.configure(text="🌉 Launch Bridge", fg_color="#1f538d", hover_color="#14375e")
            self.bridge_status = "DISCONNECTED"

        self.after(500, self.check_processes)

    def log_to_bridge(self, message: str, level: str = "INFO"):
        """Add message to bridge log queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.bridge_log_queue.put((timestamp, level, message))

    def log_to_strategy(self, message: str, level: str = "INFO"):
        """Add message to strategy log queue"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.strategy_log_queue.put((timestamp, level, message))

    def process_log_queues(self):
        """Process all log queues"""
        # Process bridge logs
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

        # Process strategy logs
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
        """Refresh trade history display"""
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")

        if not self.trade_history:
            self.history_text.insert("1.0", "No trades yet...\n")
        else:
            header = f"{'Time':<10} {'Symbol':<6} {'Side':<4} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'Status':<10}\n"
            self.history_text.insert("end", header)
            self.history_text.insert("end", "-" * 70 + "\n")

            for trade in self.trade_history[-50:]:  # Last 50 trades
                line = f"{trade.get('time', 'N/A'):<10} "
                line += f"{trade.get('symbol', 'N/A'):<6} "
                line += f"{trade.get('side', 'N/A'):<4} "
                line += f"{trade.get('entry', 0):<10.5f} "
                line += f"{trade.get('exit', 0):<10.5f} "
                line += f"{trade.get('pnl', 0):<10.2f} "
                line += f"{trade.get('status', 'N/A'):<10}\n"
                self.history_text.insert("end", line)

        self.history_text.configure(state="disabled")

    def update_positions(self):
        """Update open positions display"""
        self.positions_text.configure(state="normal")
        self.positions_text.delete("1.0", "end")

        if not self.open_positions:
            self.positions_text.insert("1.0", "No open positions\n")
        else:
            header = f"{'Symbol':<8} {'Side':<5} {'Size':<6} {'Entry':<12} {'Current':<12} {'P&L':<10}\n"
            self.positions_text.insert("end", header)
            self.positions_text.insert("end", "-" * 55 + "\n")

            for pos in self.open_positions:
                line = f"{pos.get('symbol', 'N/A'):<8} "
                line += f"{pos.get('side', 'N/A'):<5} "
                line += f"{pos.get('size', 0):<6} "
                line += f"{pos.get('entry', 0):<12.5f} "
                line += f"{pos.get('current', 0):<12.5f} "
                pnl = pos.get('pnl', 0)
                pnl_color = "+" if pnl >= 0 else ""
                line += f"{pnl_color}{pnl:<10.2f}\n"
                self.positions_text.insert("end", line)

        self.positions_text.configure(state="disabled")

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
        elif self.bridge_status == "RUNNING":
            self.bridge_indicator.configure(
                text="● Bridge: RUNNING",
                text_color="#FFD43B"  # Yellow - process running, waiting for connection
            )
        elif self.bridge_status == "ERROR":
            self.bridge_indicator.configure(
                text="● Bridge: ERROR",
                text_color="orange"
            )
        elif self.bridge_status == "STOPPED":
            self.bridge_indicator.configure(
                text="● Bridge: STOPPED",
                text_color="gray"
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
                    'status': 'CACHED' if status == 'MARKET_CLOSED' else 'LIVE'
                }

            return None

        except Exception as e:
            return None

    def toggle_trading(self):
        """Toggle trading on/off"""
        if self.is_trading:
            self.stop_trading()
        else:
            self.start_trading()

    def start_trading(self):
        """Start trading"""
        if not self.check_bridge_connection():
            self.log_to_strategy("[ERROR] Cannot start - Bridge not connected!", "ERROR")
            return

        if self.market_status != "OPEN":
            self.log_to_strategy("[WARNING] Cannot start - Market is closed!", "WARNING")
            return

        self.is_trading = True

        # Update button to Stop
        self.trading_button.configure(
            text="■ STOP TRADING",
            fg_color="#8B0000",
            hover_color="#5C0000"
        )

        self.log_to_strategy("[START] Starting live trading...", "SUCCESS")
        self.log_to_strategy("[START] Forex Scalping Strategy initialized", "INFO")

        # Start trading thread
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

    def stop_trading(self):
        """Stop trading"""
        self.is_trading = False

        # Update button to Start
        self.trading_button.configure(
            text="▶ START TRADING",
            fg_color="green",
            hover_color="darkgreen"
        )

        self.log_to_strategy("[STOP] Stopping live trading...", "WARNING")

    def trading_loop(self):
        """Main trading loop (simulated for now)"""
        self.log_to_strategy("[TRADING] Trading loop started", "INFO")

        loop_count = 0
        while self.is_trading:
            try:
                loop_count += 1
                time.sleep(10)

                if loop_count % 6 == 0:  # Every minute
                    self.log_to_strategy(f"[LOOP {loop_count}] Checking symbols...", "INFO")
                    self.log_to_strategy(f"[LOOP {loop_count}] Positions: {len(self.open_positions)}, Trades: {self.trades_today}", "INFO")

            except Exception as e:
                self.log_to_strategy(f"[ERROR] Trading loop error: {e}", "ERROR")
                break

        self.log_to_strategy("[TRADING] Trading loop stopped", "INFO")

    def on_closing(self):
        """Handle window closing"""
        if self.is_trading:
            self.stop_trading()
            time.sleep(1)

        # Terminate launched processes
        if self.bridge_process and self.bridge_process.poll() is None:
            self.bridge_process.terminate()

        self.destroy()


def main():
    """Main entry point"""
    app = EnhancedTradingDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
