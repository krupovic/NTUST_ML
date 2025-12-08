@echo off
REM Quick Start Script (Batch version - works without ExecutionPolicy changes)

echo.
echo ========================================
echo   Taxi Trajectory Model Training Setup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo WARNING: Virtual environment not found!
    echo Creating virtual environment...
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Please ensure Python is installed.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Check if requirements are installed
echo Checking dependencies...
pip show xgboost >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    echo Dependencies installed successfully.
) else (
    echo Dependencies already installed.
)
echo.

REM Check if processed data exists
if not exist "train_processed.csv" (
    echo WARNING: Processed data not found!
    echo.
    echo Please run data preparation first:
    echo   python data_preparation.py
    echo.
    set /p RUN_PREP="Run data preparation now? (y/N): "
    if /i "%RUN_PREP%"=="y" (
        python data_preparation.py
    )
    echo.
)

REM Display menu
echo ========================================
echo   What would you like to do?
echo ========================================
echo.
echo 1. Train ALL models (recommended first run)
echo    - Trains XGBoost, LightGBM, Neural Network, Ensemble
echo    - Time: approximately 25-35 minutes
echo.
echo 2. Train individual model
echo    - Choose specific model to train
echo.
echo 3. Generate predictions (requires trained model)
echo    - Creates submission file
echo.
echo 4. View model summary
echo.
echo 5. Exit
echo.

set /p CHOICE="Enter your choice (1-5): "

if "%CHOICE%"=="1" (
    echo.
    echo Training ALL models...
    echo This will take approximately 25-35 minutes.
    echo.
    python train_all_models.py
) else if "%CHOICE%"=="2" (
    echo.
    echo Available models:
    echo 1. XGBoost Advanced (5-10 min)
    echo 2. LightGBM (3-6 min, fastest)
    echo 3. Neural Network (10-20 min, requires PyTorch)
    echo 4. Ensemble (requires base models trained first)
    echo.
    set /p MODEL="Enter model number (1-4): "
    
    if "%MODEL%"=="1" (
        echo.
        echo Training XGBoost...
        python models\xgboost\train_xgboost_advanced.py
    ) else if "%MODEL%"=="2" (
        echo.
        echo Training LightGBM...
        python models\lightgbm\train_lightgbm.py
    ) else if "%MODEL%"=="3" (
        echo.
        echo Training Neural Network...
        python models\neural_network\train_neural_network.py
    ) else if "%MODEL%"=="4" (
        echo.
        echo Training Ensemble...
        python models\ensemble\train_ensemble.py
    ) else (
        echo Invalid choice!
    )
) else if "%CHOICE%"=="3" (
    echo.
    echo Generating predictions...
    python generate_predictions.py
) else if "%CHOICE%"=="4" (
    echo.
    python model_summary.py
) else if "%CHOICE%"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo.
    echo Invalid choice!
)

echo.
echo ========================================
echo   Complete!
echo ========================================
echo.
echo Virtual environment is still active.
echo To deactivate, type: deactivate
echo.

pause
