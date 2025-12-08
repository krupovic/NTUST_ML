# Quick Start Guide - Run This First!

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Taxi Trajectory Model Training Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "⚠ Virtual environment not found!" -ForegroundColor Yellow
    Write-Host "Creating virtual environment...`n" -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to create virtual environment!" -ForegroundColor Red
        Write-Host "Please ensure Python is installed and accessible." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Virtual environment created`n" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "→ Activating virtual environment..." -ForegroundColor Cyan
try {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    Write-Host "  Location: $env:VIRTUAL_ENV`n" -ForegroundColor Gray
} catch {
    Write-Host "✗ Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "`nTry running: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}

# Check if requirements are installed
Write-Host "→ Checking dependencies..." -ForegroundColor Cyan
$packages = pip list 2>$null
if ($packages -notmatch "xgboost" -or $packages -notmatch "lightgbm") {
    Write-Host "⚠ Installing required packages..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed`n" -ForegroundColor Green
} else {
    Write-Host "✓ Dependencies already installed`n" -ForegroundColor Green
}

# Check if processed data exists
if (-not (Test-Path "train_processed.csv") -or -not (Test-Path "test_processed.csv")) {
    Write-Host "⚠ Processed data not found!" -ForegroundColor Yellow
    Write-Host "Running data preparation...`n" -ForegroundColor Yellow
    python data_preparation.py
    Write-Host "`n✓ Data preparation complete`n" -ForegroundColor Green
} else {
    Write-Host "✓ Processed data files found`n" -ForegroundColor Green
}

# Display menu
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  What would you like to do?" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Train ALL models (recommended first run)" -ForegroundColor White
Write-Host "   - Trains XGBoost, LightGBM, Neural Network, Ensemble" -ForegroundColor Gray
Write-Host "   - Generates comparison report" -ForegroundColor Gray
Write-Host "   - Time: ~25-35 minutes" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Train individual model" -ForegroundColor White
Write-Host "   - Choose specific model to train" -ForegroundColor Gray
Write-Host "   - Time: ~3-20 minutes per model" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Generate predictions (requires trained model)" -ForegroundColor White
Write-Host "   - Creates submission file" -ForegroundColor Gray
Write-Host "   - Time: ~1-2 minutes" -ForegroundColor Gray
Write-Host ""
Write-Host "4. View model summary" -ForegroundColor White
Write-Host "   - Display model information" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host "`n→ Training ALL models..." -ForegroundColor Cyan
        Write-Host "This will take approximately 25-35 minutes.`n" -ForegroundColor Yellow
        python train_all_models.py
    }
    "2" {
        Write-Host "`nAvailable models:" -ForegroundColor Cyan
        Write-Host "1. XGBoost Advanced (~5-10 min)" -ForegroundColor White
        Write-Host "2. LightGBM (~3-6 min, fastest)" -ForegroundColor White
        Write-Host "3. Neural Network (~10-20 min, requires PyTorch)" -ForegroundColor White
        Write-Host "4. Ensemble (~sum of base models)" -ForegroundColor White
        
        $model_choice = Read-Host "`nEnter model number (1-4)"
        
        switch ($model_choice) {
            "1" {
                Write-Host "`n→ Training XGBoost...`n" -ForegroundColor Cyan
                python models/xgboost/train_xgboost_advanced.py
            }
            "2" {
                Write-Host "`n→ Training LightGBM...`n" -ForegroundColor Cyan
                python models/lightgbm/train_lightgbm.py
            }
            "3" {
                Write-Host "`n→ Training Neural Network...`n" -ForegroundColor Cyan
                python models/neural_network/train_neural_network.py
            }
            "4" {
                Write-Host "`n→ Training Ensemble...`n" -ForegroundColor Cyan
                python models/ensemble/train_ensemble.py
            }
            default {
                Write-Host "Invalid choice!" -ForegroundColor Red
            }
        }
    }
    "3" {
        Write-Host "`n→ Generating predictions...`n" -ForegroundColor Cyan
        python generate_predictions.py
    }
    "4" {
        Write-Host "`n" -ForegroundColor Cyan
        python model_summary.py
    }
    "5" {
        Write-Host "`nGoodbye!`n" -ForegroundColor Cyan
        exit
    }
    default {
        Write-Host "`nInvalid choice!`n" -ForegroundColor Red
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Complete!" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
