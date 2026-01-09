# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for OANDA → NinjaTrader Trading Dashboard
Creates a standalone executable with all dependencies bundled
"""

import os
import sys
from pathlib import Path

# Project paths
project_root = r'C:\Users\Jean-Yves\thevolumeainative'
bridge_dir = os.path.join(project_root, 'trading_system', 'NinjaTrader_Bridge')
forex_dir = os.path.join(project_root, 'trading_system', 'Forex_Trading')
analytics_dir = os.path.join(project_root, 'trading_system', 'analytics')

# Main script
main_script = os.path.join(bridge_dir, 'trading_dashboard_enhanced.py')

# Data files to bundle
datas = [
    # NinjaTrader Bridge executable and DLLs
    (os.path.join(bridge_dir, 'NinjaTraderBridge.exe'), '.'),
    (os.path.join(bridge_dir, 'Newtonsoft.Json.dll'), '.'),
    (os.path.join(bridge_dir, 'NinjaTrader.Client.dll'), '.'),

    # Bridge source (in case recompilation is needed)
    (os.path.join(bridge_dir, 'NinjaTraderBridge.cs'), '.'),

    # Config files
    (os.path.join(bridge_dir, 'app_config.py'), '.'),

    # Forex Trading module
    (forex_dir, 'trading_system/Forex_Trading'),

    # Analytics module
    (analytics_dir, 'trading_system/analytics'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'customtkinter',
    'tkinter',
    'tkinter.ttk',
    'oandapyV20',
    'oandapyV20.contrib.requests',
    'oandapyV20.endpoints.pricing',
    'oandapyV20.endpoints.orders',
    'oandapyV20.endpoints.positions',
    'oandapyV20.endpoints.accounts',
    'oandapyV20.endpoints.trades',
    'pandas',
    'numpy',
    'requests',
    'json',
    'threading',
    'queue',
    'socket',
    'subprocess',
    'datetime',
    'pathlib',
    'dataclasses',
]

a = Analysis(
    [main_script],
    pathex=[project_root, bridge_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TradingDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you need console output for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
