@echo off
echo ============================================================
echo HIGH-FREQUENCY CRYPTO TRADING - INSTALL DEPENDENCIES
echo ============================================================
echo.

echo Installing required Python packages...
echo This may take a few minutes.
echo.

pip install numpy pandas scipy scikit-learn
pip install xgboost lightgbm catboost
pip install tensorflow
pip install alpaca-py
pip install python-dotenv joblib
pip install matplotlib seaborn

echo.
echo ============================================================
echo INSTALLATION COMPLETE
echo ============================================================
echo.
echo Now run TRAIN_MODEL.bat to start training!
echo.
pause
