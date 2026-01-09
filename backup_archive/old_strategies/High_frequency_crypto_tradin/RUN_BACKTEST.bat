@echo off
echo ============================================================
echo HIGH-FREQUENCY CRYPTO TRADING - BACKTEST
echo ============================================================
echo.

cd /d "%~dp0"
cd ..\..

echo Current directory: %CD%
echo.

echo Running backtest on historical data...
echo.

python trading_system/High_frequency_crypto_tradin/run_backtest.py

echo.
pause
