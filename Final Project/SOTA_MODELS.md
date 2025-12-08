# State-of-the-Art Models Summary

## New Models Added

### 1. CatBoost
**Path:** `models/catboost/train_catboost.py`

**Key Features:**
- Categorical feature handling without preprocessing
- Ordered boosting to reduce overfitting
- Often outperforms XGBoost and LightGBM on structured data
- Built-in GPU support

**Parameters:**
- Iterations: 1000
- Learning rate: 0.05
- Depth: 8
- L2 regularization: 3
- Early stopping: 50 rounds

**Expected Performance:** ~2.2-2.4 km Haversine error

**Installation:** `pip install catboost>=1.2.0`

---

### 2. HistGradientBoosting (sklearn)
**Path:** `models/histgb/train_histgb.py`

**Key Features:**
- Native scikit-learn implementation
- Fast histogram-based algorithm
- No additional dependencies
- Handles missing values natively
- Good baseline for comparison

**Parameters:**
- Max iterations: 500
- Learning rate: 0.05
- Max depth: 10
- Max leaf nodes: 63
- L2 regularization: 0.1
- Early stopping enabled

**Expected Performance:** ~2.4-2.6 km Haversine error

**Installation:** Included with scikit-learn (no extra install)

---

## Model Comparison

| Model | Type | Speed | Accuracy | Dependencies |
|-------|------|-------|----------|--------------|
| XGBoost | Gradient Boosting | Medium | High | xgboost |
| LightGBM | Gradient Boosting | Fast | High | lightgbm |
| **CatBoost** | Gradient Boosting | Medium | **Very High** | catboost |
| **HistGradientBoosting** | Gradient Boosting | Very Fast | Medium-High | sklearn |
| Ensemble | Meta-Learner | Slow | Highest | All above |

---

## Training Order

The updated `train_all_models.py` runs models in this sequence:
1. XGBoost Advanced
2. LightGBM
3. **CatBoost** (new)
4. **HistGradientBoosting** (new)
5. Ensemble (combines all successful models)

---

## Why These Models?

### CatBoost Advantages:
- **State-of-the-art accuracy** on tabular data
- Robust to hyperparameter settings
- Excellent handling of categorical features (call_type, origin)
- Reduced overfitting through ordered boosting
- Often wins Kaggle competitions

### HistGradientBoosting Advantages:
- **No extra dependencies** (pure sklearn)
- Very fast training on large datasets
- Native missing value handling
- Good baseline without tuning
- Production-ready in sklearn ecosystem

---

## Usage

### Train All Models:
```bash
python train_all_models.py
```

### Train Individual Model:
```bash
# CatBoost
python models/catboost/train_catboost.py

# HistGradientBoosting
python models/histgb/train_histgb.py
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Output Files

### CatBoost:
- `models/catboost/catboost_lon.pkl`
- `models/catboost/catboost_lat.pkl`
- `models/catboost/metrics_catboost.json`

### HistGradientBoosting:
- `models/histgb/histgb_lon.pkl`
- `models/histgb/histgb_lat.pkl`
- `models/histgb/metrics_histgb.json`

---

## Performance Expectations

Based on similar datasets, expected validation metrics:

**CatBoost:**
- Mean Haversine: 2.2-2.4 km
- Within 2km: ~45-50%
- Within 5km: ~75-80%
- Training time: ~5-10 minutes

**HistGradientBoosting:**
- Mean Haversine: 2.4-2.6 km
- Within 2km: ~40-45%
- Within 5km: ~70-75%
- Training time: ~2-5 minutes

---

## Next Steps

1. Train all models to compare performance
2. Analyze feature importances across models
3. Use best model for final predictions
4. Consider ensemble if multiple models perform well
