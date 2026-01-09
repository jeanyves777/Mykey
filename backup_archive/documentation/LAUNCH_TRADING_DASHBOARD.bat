@echo off
REM ============================================================================
REM OANDA → NinjaTrader Trading Dashboard Launcher
REM ============================================================================

echo.
echo ================================================================================
echo   OANDA → NinjaTrader Trading Dashboard
echo ================================================================================
echo.
echo Starting beautiful desktop trading dashboard...
echo.

cd "C:\Users\Jean-Yves\thevolumeainative"

.venv\Scripts\python.exe trading_system\NinjaTrader_Bridge\trading_dashboard_gui.py

pause
