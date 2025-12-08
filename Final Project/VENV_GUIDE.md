# 🚀 Quick Reference - Virtual Environment Setup

## ⚠️ IMPORTANT: Always Use Virtual Environment

All Python commands **must** be run inside the `.venv` virtual environment to ensure correct package versions and avoid conflicts.

---

## 🔧 Initial Setup (One Time)

```powershell
# 1. Create virtual environment (only needed once)
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install all dependencies
pip install -r requirements.txt
```

You should see `(.venv)` at the beginning of your command prompt when activated.

---

## ✅ Every Time You Open a New Terminal

**ALWAYS run this first:**
```powershell
.\.venv\Scripts\Activate.ps1
```

You'll know it worked when you see `(.venv)` in your prompt:
```
(.venv) PS G:\Final Project>
```

---

## 📋 Common Commands (With Virtual Environment)

### Training Models

```powershell
# Activate venv first!
.\.venv\Scripts\Activate.ps1

# Train all models (recommended)
python train_all_models.py

# Or train individual models
python models/xgboost/train_xgboost_advanced.py
python models/lightgbm/train_lightgbm.py
python models/neural_network/train_neural_network.py
python models/ensemble/train_ensemble.py
```

### Generate Predictions

```powershell
# Activate venv first!
.\.venv\Scripts\Activate.ps1

# Auto-select best model
python generate_predictions.py

# Or specify a model
python generate_predictions.py "XGBoost Advanced"
python generate_predictions.py "LightGBM"
```

### Data Preparation

```powershell
# Activate venv first!
.\.venv\Scripts\Activate.ps1

# Process raw data
python data_preparation.py

# Run visualizations
python data_visualization.py
```

---

## 🎯 Easiest Way: Use start.ps1

The `start.ps1` script automatically handles virtual environment activation:

```powershell
.\start.ps1
```

This interactive menu will:
1. ✅ Activate `.venv` automatically
2. ✅ Check and install dependencies
3. ✅ Guide you through training and predictions

---

## 🐛 Troubleshooting

### "Scripts execution is disabled"
```powershell
# Run as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Virtual environment not activating
```powershell
# Try full path
& "G:\Final Project\.venv\Scripts\Activate.ps1"
```

### Packages not found after installation
```powershell
# Make sure venv is activated, then:
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Wrong Python version
```powershell
# Check Python version (should be 3.8+)
python --version

# If needed, specify Python explicitly when creating venv
py -3.10 -m venv .venv
```

---

## 🔍 Verify Virtual Environment

Check if you're in the virtual environment:

```powershell
# Windows PowerShell
$env:VIRTUAL_ENV
# Should output: G:\Final Project\.venv

# Or check Python path
python -c "import sys; print(sys.prefix)"
# Should contain ".venv"
```

---

## 📦 Package Management

### Install new packages (inside venv)
```powershell
.\.venv\Scripts\Activate.ps1
pip install package-name
```

### Update requirements.txt
```powershell
.\.venv\Scripts\Activate.ps1
pip freeze > requirements.txt
```

### Install from requirements.txt
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🎓 Best Practices

1. **Always activate** `.venv` before running Python commands
2. **Use start.ps1** for guided, automatic setup
3. **Check for `(.venv)` prefix** in your terminal prompt
4. **Never install packages globally** when working on this project
5. **Keep requirements.txt updated** when adding new packages

---

## 📝 Complete Workflow Example

```powershell
# 1. Open PowerShell in project directory
cd "G:\Final Project"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify activation (you should see .venv in path)
python -c "import sys; print(sys.prefix)"

# 4. Run your commands
python train_all_models.py
python generate_predictions.py

# 5. Deactivate when done (optional)
deactivate
```

---

## 🆘 Quick Help

**Problem**: Command not working  
**Solution**: Did you activate `.venv`? Run `.\.venv\Scripts\Activate.ps1`

**Problem**: Package not found  
**Solution**: Activate `.venv`, then run `pip install -r requirements.txt`

**Problem**: Wrong Python version  
**Solution**: Use `python --version` to check. Recreate venv if needed.

**Problem**: Permission denied  
**Solution**: Run PowerShell as Administrator for ExecutionPolicy changes

---

**Remember**: When in doubt, always activate the virtual environment first! ✅

```powershell
.\.venv\Scripts\Activate.ps1
```
