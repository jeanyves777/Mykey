@echo off
echo ============================================================
echo HIGH-FREQUENCY CRYPTO TRADING - FAST MODEL TRAINING
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo Python not found in PATH. Trying py launcher...
    py --version 2>nul
    if errorlevel 1 (
        echo ERROR: Python not found. Please install Python 3.8+
        pause
        exit /b 1
    )
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo.
echo Step 1: Installing dependencies (if needed)...
%PYTHON_CMD% -m pip install numpy pandas scikit-learn xgboost lightgbm catboost tensorflow alpaca-py python-dotenv joblib --quiet 2>nul

echo.
echo Step 2: Starting FAST training...
echo Expected time: 3-10 minutes (optimized with early stopping)
echo.

cd ..\..
%PYTHON_CMD% trading_system/High_frequency_crypto_tradin/train_ensemble.py

echo.
echo ============================================================
if errorlevel 1 (
    echo TRAINING FAILED - Check errors above
) else (
    echo TRAINING COMPLETE!
    echo.
    echo Next steps:
    echo   1. Run backtest: RUN_BACKTEST.bat
    echo   2. Run paper trading: RUN_PAPER_TRADING.bat
)
echo ============================================================
echo.
pause
