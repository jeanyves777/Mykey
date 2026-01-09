@echo off
REM ============================================================================
REM OANDA → NinjaTrader ENHANCED Trading Dashboard Launcher
REM ============================================================================

echo.
echo ================================================================================
echo   OANDA → NinjaTrader ENHANCED Trading Dashboard
echo ================================================================================
echo.
echo Features:
echo   - Launch NinjaTrader and Bridge from dashboard
echo   - Tabbed logs (Bridge, Strategy, Trade History)
echo   - Real-time monitoring
echo   - Full trading controls
echo.
echo Starting dashboard...
echo.

cd "C:\Users\Jean-Yves\thevolumeainative"

.venv\Scripts\python.exe trading_system\NinjaTrader_Bridge\trading_dashboard_enhanced.py

pause
