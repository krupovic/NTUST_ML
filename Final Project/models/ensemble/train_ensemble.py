"""
Ensemble Model - Combining XGBoost, LightGBM, CatBoost, and HistGradientBoosting
- Weighted averaging ensemble
- Stacking with meta-learner
- Best performance through model diversity
"""
import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def add_advanced_features(df):
    """Add advanced geospatial and temporal features"""
    df = df.copy()
    
    df["delta_lon"] = df["lon_last"] - df["lon_1st"]
    df["delta_lat"] = df["lat_last"] - df["lat_1st"]
    df["euclidean_dist"] = np.sqrt(df["delta_lon"]**2 + df["delta_lat"]**2)
    df["bearing"] = np.arctan2(df["delta_lat"], df["delta_lon"])
    df["manhattan_dist"] = np.abs(df["delta_lon"]) + np.abs(df["delta_lat"])
    
    if "hour" in df.columns:
        df["is_rush_hour"] = df["hour"].apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)
        df["is_night"] = df["hour"].apply(lambda x: 1 if (22 <= x) or (x <= 6) else 0)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["time_of_day"] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True)
    
    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter"] = ((df["month"] - 1) // 3) + 1
    
    df["lat_lon_interaction"] = df["lon_1st"] * df["lat_1st"]
    df["start_quadrant"] = ((df["lat_1st"] > 41.15).astype(int) * 2 + 
                            (df["lon_1st"] > -8.62).astype(int))
    
    return df

def prepare_features(train_path, test_path, sample_size=None, test_split=0.2, random_state=42):
    """Prepare features for ensemble"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train = add_advanced_features(train)
    test = add_advanced_features(test)
    
    if sample_size and sample_size < len(train):
        train = train.sample(sample_size, random_state=random_state)
    
    # XGBoost/LightGBM features
    base_features = [
        "call_type_a", "call_type_b", "call_type_c",
        "lon_1st", "lat_1st", "delta_lon", "delta_lat",
        "euclidean_dist", "bearing", "manhattan_dist",
        "lat_lon_interaction", "start_quadrant"
    ]
    
    temporal_features = []
    if "hour" in train.columns:
        temporal_features.extend(["hour", "is_rush_hour", "is_night", "hour_sin", "hour_cos"])
        if "time_of_day" in train.columns:
            temporal_features.append("time_of_day")
    if "weekday" in train.columns:
        temporal_features.extend(["weekday", "is_weekend", "weekday_sin", "weekday_cos"])
    if "month" in train.columns:
        temporal_features.extend(["month", "month_sin", "month_cos"])
        if "quarter" in train.columns:
            temporal_features.append("quarter")
    
    metadata_features = []
    if "ORIGIN_CALL" in train.columns:
        if train["ORIGIN_CALL"].dtype in [np.float64, np.int64]:
            metadata_features.append("ORIGIN_CALL")
        else:
            train["ORIGIN_CALL_flag"] = train["ORIGIN_CALL"].notna().astype(int)
            test["ORIGIN_CALL_flag"] = test["ORIGIN_CALL"].notna().astype(int)
            metadata_features.append("ORIGIN_CALL_flag")
    
    if "ORIGIN_STAND" in train.columns:
        if train["ORIGIN_STAND"].dtype in [np.float64, np.int64]:
            metadata_features.append("ORIGIN_STAND")
        else:
            train["ORIGIN_STAND_flag"] = train["ORIGIN_STAND"].notna().astype(int)
            test["ORIGIN_STAND_flag"] = test["ORIGIN_STAND"].notna().astype(int)
            metadata_features.append("ORIGIN_STAND_flag")
    
    if "MISSING_DATA" in train.columns:
        if train["MISSING_DATA"].dtype in [np.float64, np.int64]:
            metadata_features.append("MISSING_DATA")
    
    feature_cols = base_features + temporal_features + metadata_features
    feature_cols = [f for f in feature_cols if f in train.columns]
    
    target_cols = ["lon_last", "lat_last"]
    
    X = train[feature_cols]
    y = train[target_cols]
    X_test_full = test[feature_cols]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, X_test_full, feature_cols

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation metrics"""
    metrics = {}
    
    if hasattr(y_true, 'values'):
        y_true_np = y_true.values
    else:
        y_true_np = y_true
    
    if hasattr(y_pred, 'values'):
        y_pred_np = y_pred.values
    else:
        y_pred_np = y_pred
    
    metrics["lon_mse"] = mean_squared_error(y_true_np[:, 0], y_pred_np[:, 0])
    metrics["lat_mse"] = mean_squared_error(y_true_np[:, 1], y_pred_np[:, 1])
    metrics["lon_mae"] = mean_absolute_error(y_true_np[:, 0], y_pred_np[:, 0])
    metrics["lat_mae"] = mean_absolute_error(y_true_np[:, 1], y_pred_np[:, 1])
    metrics["lon_r2"] = r2_score(y_true_np[:, 0], y_pred_np[:, 0])
    metrics["lat_r2"] = r2_score(y_true_np[:, 1], y_pred_np[:, 1])
    
    distances = haversine_distance(
        y_true_np[:, 1], y_true_np[:, 0],
        y_pred_np[:, 1], y_pred_np[:, 0]
    )
    
    metrics["mean_haversine_km"] = np.mean(distances)
    metrics["median_haversine_km"] = np.median(distances)
    metrics["p90_haversine_km"] = np.percentile(distances, 90)
    metrics["p95_haversine_km"] = np.percentile(distances, 95)
    metrics["within_1km"] = np.mean(distances < 1.0)
    metrics["within_2km"] = np.mean(distances < 2.0)
    metrics["within_5km"] = np.mean(distances < 5.0)
    
    return metrics

def load_base_models():
    """Load pre-trained base models with their feature configurations"""
    models = {}
    feature_configs = {}
    
    # Try to load XGBoost
    try:
        models['xgb_lon'] = joblib.load("models/xgboost/xgboost_lon_advanced.pkl")
        models['xgb_lat'] = joblib.load("models/xgboost/xgboost_lat_advanced.pkl")
        with open("models/xgboost/metrics_xgboost_advanced.json", 'r') as f:
            metrics = json.load(f)
            feature_configs['xgb'] = metrics.get('features', [])
        print("✓ Loaded XGBoost models")
    except Exception as e:
        print(f"✗ XGBoost models not found: {e}")
    
    # Try to load LightGBM
    try:
        models['lgb_lon'] = joblib.load("models/lightgbm/lightgbm_lon.pkl")
        models['lgb_lat'] = joblib.load("models/lightgbm/lightgbm_lat.pkl")
        with open("models/lightgbm/metrics_lightgbm.json", 'r') as f:
            metrics = json.load(f)
            feature_configs['lgb'] = metrics.get('features', [])
        print("✓ Loaded LightGBM models")
    except Exception as e:
        print(f"✗ LightGBM models not found: {e}")
    
    # Try to load CatBoost
    try:
        models['cat_lon'] = joblib.load("models/catboost/catboost_lon.pkl")
        models['cat_lat'] = joblib.load("models/catboost/catboost_lat.pkl")
        with open("models/catboost/metrics_catboost.json", 'r') as f:
            metrics = json.load(f)
            feature_configs['cat'] = metrics.get('features', [])
        print("✓ Loaded CatBoost models")
    except Exception as e:
        print(f"✗ CatBoost models not found: {e}")
    
    # Try to load HistGradientBoosting
    try:
        models['hgb_lon'] = joblib.load("models/histgb/histgb_lon.pkl")
        models['hgb_lat'] = joblib.load("models/histgb/histgb_lat.pkl")
        with open("models/histgb/metrics_histgb.json", 'r') as f:
            metrics = json.load(f)
            feature_configs['hgb'] = metrics.get('features', [])
        print("✓ Loaded HistGradientBoosting models")
    except Exception as e:
        print(f"✗ HistGradientBoosting models not found: {e}")
    
    return models, feature_configs

def train_ensemble_model():
    """Train ensemble model combining all base models"""
    print("=" * 60)
    print("Ensemble Model Training")
    print("=" * 60)
    
    # Load base models
    print("\n1. Loading base models...")
    models, feature_configs = load_base_models()
    
    if len(models) < 2:
        print("\nError: Need at least 2 base models trained!")
        print("Please train base models first:")
        print("  - python models/xgboost/train_xgboost_advanced.py")
        print("  - python models/lightgbm/train_lightgbm.py")
        print("  - python models/neural_network/train_neural_network.py")
        return None
    
    # Find common features across all models
    if feature_configs:
        all_features = [set(features) for features in feature_configs.values()]
        common_features = list(set.intersection(*all_features))
        print(f"\n   Using {len(common_features)} common features across all models")
    else:
        common_features = None
    
    # Load and prepare data
    print("\n2. Loading and preparing features...")
    X_train, X_val, y_train, y_val, X_test, feature_cols = prepare_features(
        "train_processed.csv",
        "test_processed.csv",
        sample_size=150000,
        test_split=0.2,
        random_state=42
    )
    
    # Use common features if available
    if common_features:
        feature_cols = [f for f in common_features if f in X_train.columns]
        X_train = X_train[feature_cols]
        X_val = X_val[feature_cols]
        print(f"   Filtered to {len(feature_cols)} compatible features")
    
    print(f"   Train size: {X_train.shape}")
    print(f"   Validation size: {X_val.shape}")
    print(f"   Features: {feature_cols[:5]}... (showing first 5)")
    
    # Get predictions from base models
    print("\n3. Generating base model predictions...")
    train_preds = []
    val_preds = []
    model_names = []
    
    # XGBoost predictions
    if 'xgb_lon' in models and 'xgb_lat' in models:
        xgb_train = np.column_stack([
            models['xgb_lon'].predict(X_train),
            models['xgb_lat'].predict(X_train)
        ])
        xgb_val = np.column_stack([
            models['xgb_lon'].predict(X_val),
            models['xgb_lat'].predict(X_val)
        ])
        train_preds.append(xgb_train)
        val_preds.append(xgb_val)
        model_names.append("XGBoost")
        print("   ✓ XGBoost predictions generated")
    
    # LightGBM predictions
    if 'lgb_lon' in models and 'lgb_lat' in models:
        lgb_train = np.column_stack([
            models['lgb_lon'].predict(X_train),
            models['lgb_lat'].predict(X_train)
        ])
        lgb_val = np.column_stack([
            models['lgb_lon'].predict(X_val),
            models['lgb_lat'].predict(X_val)
        ])
        train_preds.append(lgb_train)
        val_preds.append(lgb_val)
        model_names.append("LightGBM")
        print("   ✓ LightGBM predictions generated")
    
    # CatBoost predictions
    if 'cat_lon' in models and 'cat_lat' in models:
        cat_train = np.column_stack([
            models['cat_lon'].predict(X_train),
            models['cat_lat'].predict(X_train)
        ])
        cat_val = np.column_stack([
            models['cat_lon'].predict(X_val),
            models['cat_lat'].predict(X_val)
        ])
        train_preds.append(cat_train)
        val_preds.append(cat_val)
        model_names.append("CatBoost")
        print("   ✓ CatBoost predictions generated")
    
    # HistGradientBoosting predictions
    if 'hgb_lon' in models and 'hgb_lat' in models:
        hgb_train = np.column_stack([
            models['hgb_lon'].predict(X_train),
            models['hgb_lat'].predict(X_train)
        ])
        hgb_val = np.column_stack([
            models['hgb_lon'].predict(X_val),
            models['hgb_lat'].predict(X_val)
        ])
        train_preds.append(hgb_train)
        val_preds.append(hgb_val)
        model_names.append("HistGradientBoosting")
        print("   ✓ HistGradientBoosting predictions generated")
    
    # Evaluate individual models
    print("\n4. Individual Model Performance on Validation Set:")
    individual_metrics = {}
    for i, name in enumerate(model_names):
        metrics = evaluate_predictions(y_val, val_preds[i], name)
        individual_metrics[name] = metrics
        print(f"\n   {name}:")
        print(f"     Mean Haversine: {metrics['mean_haversine_km']:.4f} km")
        print(f"     Within 1km: {metrics['within_1km']:.4f}")
        print(f"     Lon R²: {metrics['lon_r2']:.4f}, Lat R²: {metrics['lat_r2']:.4f}")
    
    # Method 1: Simple Weighted Average
    print("\n5. Training Weighted Average Ensemble...")
    
    # Find optimal weights using validation set
    best_weights = None
    best_score = float('inf')
    
    # Grid search for weights
    from itertools import product
    weight_options = [0.2, 0.3, 0.4, 0.5]
    
    for weights in product(weight_options, repeat=len(model_names)):
        if abs(sum(weights) - 1.0) > 0.01:
            continue
        
        weighted_val = sum(w * pred for w, pred in zip(weights, val_preds))
        score = np.mean(haversine_distance(
            y_val.values[:, 1], y_val.values[:, 0],
            weighted_val[:, 1], weighted_val[:, 0]
        ))
        
        if score < best_score:
            best_score = score
            best_weights = weights
    
    print(f"   Optimal weights: {dict(zip(model_names, best_weights))}")
    
    # Generate weighted average predictions
    weighted_train = sum(w * pred for w, pred in zip(best_weights, train_preds))
    weighted_val = sum(w * pred for w, pred in zip(best_weights, val_preds))
    
    # Method 2: Stacking with meta-learner
    print("\n6. Training Stacking Ensemble...")
    
    # Stack predictions as features for meta-learner
    train_meta_features = np.hstack([pred for pred in train_preds])
    val_meta_features = np.hstack([pred for pred in val_preds])
    
    # Train meta-learner (Ridge regression for each output)
    meta_lon = Ridge(alpha=1.0)
    meta_lat = Ridge(alpha=1.0)
    
    meta_lon.fit(train_meta_features, y_train.values[:, 0])
    meta_lat.fit(train_meta_features, y_train.values[:, 1])
    
    # Stacking predictions
    stacking_train = np.column_stack([
        meta_lon.predict(train_meta_features),
        meta_lat.predict(train_meta_features)
    ])
    stacking_val = np.column_stack([
        meta_lon.predict(val_meta_features),
        meta_lat.predict(val_meta_features)
    ])
    
    # Evaluate ensemble methods
    print("\n7. Ensemble Performance Comparison:")
    
    print("\n   Weighted Average Ensemble:")
    weighted_metrics = evaluate_predictions(y_val, weighted_val, "Weighted")
    for k, v in weighted_metrics.items():
        print(f"     {k}: {v:.6f}")
    
    print("\n   Stacking Ensemble:")
    stacking_metrics = evaluate_predictions(y_val, stacking_val, "Stacking")
    for k, v in stacking_metrics.items():
        print(f"     {k}: {v:.6f}")
    
    # Choose best ensemble
    if weighted_metrics['mean_haversine_km'] < stacking_metrics['mean_haversine_km']:
        best_method = "weighted"
        best_metrics = weighted_metrics
        print("\n   → Best method: Weighted Average")
    else:
        best_method = "stacking"
        best_metrics = stacking_metrics
        print("\n   → Best method: Stacking")
    
    # Save ensemble configuration
    print("\n8. Saving ensemble configuration...")
    
    ensemble_config = {
        "method": best_method,
        "base_models": model_names,
        "weights": {name: float(w) for name, w in zip(model_names, best_weights)} if best_method == "weighted" else None,
        "individual_metrics": {k: {m: float(v) for m, v in metrics.items()} 
                              for k, metrics in individual_metrics.items()},
        "ensemble_metrics": {
            "weighted": {k: float(v) for k, v in weighted_metrics.items()},
            "stacking": {k: float(v) for k, v in stacking_metrics.items()}
        },
        "best_ensemble_metrics": {k: float(v) for k, v in best_metrics.items()}
    }
    
    with open("models/ensemble/ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save meta-learners if stacking
    if best_method == "stacking":
        joblib.dump(meta_lon, "models/ensemble/meta_lon.pkl")
        joblib.dump(meta_lat, "models/ensemble/meta_lat.pkl")
    
    print("\n✓ Ensemble configuration saved successfully!")
    print("=" * 60)
    
    return ensemble_config

if __name__ == "__main__":
    train_ensemble_model()
