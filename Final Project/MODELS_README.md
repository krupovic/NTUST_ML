# 🚕 Taxi Trajectory Destination Prediction

Advanced machine learning models for predicting taxi trip destinations from partial trajectories (ECML/PKDD 2015 Porto dataset).

## 🎯 Project Overview

This project implements multiple state-of-the-art machine learning models to predict the final destination (latitude, longitude) of taxi trips based on:
- Partial GPS trajectory data
- Temporal features (hour, day, month)
- Trip metadata (call type, origin)
- Derived geospatial features

## 🏗️ Project Structure

```
├── data_understanding.py          # Data loading and preprocessing
├── data_visualization.py          # EDA and visualization generation
├── data_preparation.py            # Feature engineering pipeline
├── prepare_ml_data.py             # ML-ready dataset preparation
├── train_all_models.py            # Master training pipeline ⭐
├── generate_predictions.py        # Prediction generation for test set
├── requirements.txt               # Python dependencies
│
├── models/                        # Model implementations
│   ├── xgboost/
│   │   └── train_xgboost_advanced.py    # XGBoost with advanced features
│   ├── lightgbm/
│   │   └── train_lightgbm.py            # LightGBM model
│   ├── neural_network/
│   │   └── train_neural_network.py      # Deep learning (PyTorch)
│   └── ensemble/
│       └── train_ensemble.py            # Ensemble combining all models
│
└── pkdd-15-predict-taxi-service-trajectory-i/
    └── [Kaggle competition data]
```

## 🤖 Implemented Models

### 1. **XGBoost Advanced** 🌳
- Optimized gradient boosting with 500 trees
- Advanced geospatial feature engineering
- Features: bearing, distance, temporal cycles
- **Best for**: Interpretability and robust performance

### 2. **LightGBM** ⚡
- Fast gradient boosting optimized for large datasets
- Handles categorical features natively
- Efficient memory usage with histogram-based learning
- **Best for**: Speed and scalability

### 3. **Deep Neural Network** 🧠
- Multi-layer perceptron (256→128→64→32→2)
- Custom Haversine loss function for geographic accuracy
- Batch normalization and dropout regularization
- **Best for**: Capturing complex non-linear patterns
- **Requires**: PyTorch (`pip install torch`)

### 4. **Ensemble Model** 🏆
- Combines predictions from all base models
- Two strategies: Weighted averaging & Stacking
- Automatically selects optimal combination
- **Best for**: Maximum prediction accuracy

## 📊 Feature Engineering

The models use **advanced geospatial and temporal features**:

### Geospatial Features
- `delta_lon`, `delta_lat`: Displacement from start to current position
- `euclidean_dist`: Straight-line distance traveled
- `bearing`: Direction of travel (radians)
- `manhattan_dist`: City-block distance
- `lat_lon_interaction`: Coordinate interactions
- `start_quadrant`: Geographic region classification

### Temporal Features (Cyclic Encoding)
- `hour_sin`, `hour_cos`: Time of day (circular)
- `weekday_sin`, `weekday_cos`: Day of week (circular)
- `month_sin`, `month_cos`: Month of year (circular)
- `is_rush_hour`: Peak traffic hours (7-9 AM, 5-7 PM)
- `is_night`: Night hours (10 PM - 6 AM)
- `is_weekend`: Saturday/Sunday indicator

### Metadata Features
- Call type encoding (A/B/C)
- Origin call/stand flags
- Missing data indicator

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment (if not exists)
python -m venv .venv

# Activate (Windows PowerShell) - REQUIRED for all commands below
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

⚠️ **IMPORTANT**: Always activate the virtual environment before running any commands:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Prepare Data

Place Kaggle competition files in `./pkdd-15-predict-taxi-service-trajectory-i/`:
- `train.csv.zip`
- `test.csv.zip`
- `metaData_taxistandsID_name_GPSlocation.csv.zip`

```powershell
# Ensure venv is activated first!
.\.venv\Scripts\Activate.ps1

# Process raw data
python data_preparation.py
```

This generates:
- `train_processed.csv` - Engineered training features
- `test_processed.csv` - Engineered test features

### 3. Train Models

**Option A: Train all models at once (recommended)**
```powershell
# Ensure venv is activated
.\.venv\Scripts\Activate.ps1

# Train all models
python train_all_models.py
```
This will:
1. Train XGBoost, LightGBM, Neural Network sequentially
2. Train ensemble model combining all three
3. Generate comprehensive comparison report
4. Identify the best performing model

**Option B: Train individual models**
```powershell
# Ensure venv is activated
.\.venv\Scripts\Activate.ps1

# XGBoost
python models/xgboost/train_xgboost_advanced.py

# LightGBM
python models/lightgbm/train_lightgbm.py

# Neural Network (requires PyTorch)
python models/neural_network/train_neural_network.py

# Ensemble (requires base models trained first)
python models/ensemble/train_ensemble.py
```

### 4. Generate Predictions

```powershell
# Ensure venv is activated
.\.venv\Scripts\Activate.ps1

# Auto-select best model
python generate_predictions.py

# Or specify a model
python generate_predictions.py "XGBoost Advanced"
python generate_predictions.py "LightGBM"
```

Output: `submission_<model_name>.csv` with format:
```
TRIP_ID,LATITUDE,LONGITUDE
T1,41.146504,-8.611317
T2,42.230000,-8.629454
```

## 📈 Evaluation Metrics

All models are evaluated using:

### Primary Metric
- **Mean Haversine Distance (km)**: Geographic distance between predicted and actual destinations

### Additional Metrics
- **Median/P90/P95 Haversine**: Distribution of errors
- **Within 1km/2km/5km**: Accuracy thresholds
- **Longitude/Latitude R²**: Coordinate-wise performance
- **MSE/MAE**: Standard regression metrics

## 🎓 Model Comparison

After training all models, view the comparison report:

```bash
# JSON format
cat model_comparison_report.json

# CSV format (can open in Excel)
model_comparison_report.csv
```

Example output:
```
Model                Mean Haversine (km)  Within 1km (%)  Training Time (s)
Ensemble             2.1234               45.2            850
XGBoost Advanced     2.3456               42.1            320
LightGBM             2.4567               40.8            180
Neural Network       2.5678               39.5            620
```

## 💡 Best Practices

### For Best Performance
1. **Use Ensemble**: Combines strengths of all models
2. **Tune sample size**: Balance between accuracy and training time
3. **More data**: Increase `sample_size` parameter in training scripts
4. **GPU acceleration**: For Neural Network, use CUDA if available

### For Fast Experimentation
1. **Use LightGBM**: Fastest training
2. **Reduce sample size**: Set to 50,000-100,000 samples
3. **Skip Neural Network**: Requires PyTorch and longer training

### For Production
1. **Train on full dataset**: Remove `sample_size` limits
2. **Cross-validation**: Implement k-fold CV for robust estimates
3. **Hyperparameter tuning**: Use grid/random search
4. **Model monitoring**: Track performance over time

## 🔧 Customization

### Adjust Model Hyperparameters

Edit the training scripts:
- `models/xgboost/train_xgboost_advanced.py`: XGBoost params
- `models/lightgbm/train_lightgbm.py`: LightGBM params
- `models/neural_network/train_neural_network.py`: Network architecture

### Add New Features

Modify `add_advanced_features()` function in each training script to add custom features.

### Change Sample Size

In each training script, adjust:
```python
sample_size=200000  # Increase for more data, decrease for faster training
```

## 📦 Dependencies

### Core (Required)
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- joblib

### Optional
- torch (for Neural Network model)
- folium, seaborn, plotly, matplotlib (for visualizations)

## 🐛 Troubleshooting

### "PyTorch not available"
```bash
pip install torch
# Or skip neural network model
```

### Out of memory
- Reduce `sample_size` in training scripts
- Use LightGBM (most memory efficient)
- Close other applications

### Models not found
- Ensure models are trained before ensemble/prediction
- Check for error messages in training output

## 📚 References

- **Competition**: [ECML/PKDD 2015 Taxi Trajectory Prediction](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)
- **XGBoost**: [Documentation](https://xgboost.readthedocs.io/)
- **LightGBM**: [Documentation](https://lightgbm.readthedocs.io/)
- **PyTorch**: [Documentation](https://pytorch.org/docs/)

## 📄 License

This project is for educational purposes. Dataset from Kaggle ECML/PKDD 2015 competition.

---

**🎯 Ready to train? Run:** `python train_all_models.py`
