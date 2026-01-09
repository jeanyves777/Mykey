@echo off
echo ============================================================
echo HIGH-FREQUENCY CRYPTO TRADING - PAPER TRADING
echo ============================================================
echo.

cd /d "%~dp0"
cd ..\..

echo Current directory: %CD%
echo.

echo Starting paper trading with Alpaca...
echo Press Ctrl+C to stop trading at any time.
echo.

python trading_system/High_frequency_crypto_tradin/run_paper_trading.py

pause
