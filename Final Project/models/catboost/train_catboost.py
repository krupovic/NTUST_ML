"""
CatBoost Model for Taxi Trajectory Prediction
- State-of-the-art gradient boosting
- Excellent handling of categorical features
- Often outperforms XGBoost and LightGBM
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers"""
    R = 6371
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
    
    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    df["lat_lon_interaction"] = df["lon_1st"] * df["lat_1st"]
    
    return df

def prepare_features(train_path, test_path, sample_size=None, test_split=0.2, random_state=42):
    """Prepare features with advanced engineering"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train = add_advanced_features(train)
    test = add_advanced_features(test)
    
    if sample_size and sample_size < len(train):
        train = train.sample(sample_size, random_state=random_state)
    
    base_features = [
        "call_type_a", "call_type_b", "call_type_c",
        "lon_1st", "lat_1st", "delta_lon", "delta_lat",
        "euclidean_dist", "bearing", "manhattan_dist", "lat_lon_interaction"
    ]
    
    temporal_features = []
    if "hour" in train.columns:
        temporal_features.extend(["hour", "is_rush_hour", "is_night", "hour_sin", "hour_cos"])
    if "weekday" in train.columns:
        temporal_features.extend(["weekday", "is_weekend", "weekday_sin", "weekday_cos"])
    if "month" in train.columns:
        temporal_features.extend(["month", "month_sin", "month_cos"])
    
    metadata_features = []
    if "ORIGIN_CALL" in train.columns:
        if train["ORIGIN_CALL"].dtype in [np.float64, np.int64]:
            metadata_features.append("ORIGIN_CALL")
    if "ORIGIN_STAND" in train.columns:
        if train["ORIGIN_STAND"].dtype in [np.float64, np.int64]:
            metadata_features.append("ORIGIN_STAND")
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
    
    metrics["lon_mse"] = mean_squared_error(y_true.iloc[:, 0], y_pred[:, 0])
    metrics["lat_mse"] = mean_squared_error(y_true.iloc[:, 1], y_pred[:, 1])
    metrics["lon_mae"] = mean_absolute_error(y_true.iloc[:, 0], y_pred[:, 0])
    metrics["lat_mae"] = mean_absolute_error(y_true.iloc[:, 1], y_pred[:, 1])
    metrics["lon_r2"] = r2_score(y_true.iloc[:, 0], y_pred[:, 0])
    metrics["lat_r2"] = r2_score(y_true.iloc[:, 1], y_pred[:, 1])
    
    distances = haversine_distance(
        y_true.iloc[:, 1].values, y_true.iloc[:, 0].values,
        y_pred[:, 1], y_pred[:, 0]
    )
    
    metrics["mean_haversine_km"] = np.mean(distances)
    metrics["median_haversine_km"] = np.median(distances)
    metrics["p90_haversine_km"] = np.percentile(distances, 90)
    metrics["p95_haversine_km"] = np.percentile(distances, 95)
    metrics["within_1km"] = np.mean(distances < 1.0)
    metrics["within_2km"] = np.mean(distances < 2.0)
    metrics["within_5km"] = np.mean(distances < 5.0)
    
    return metrics

def train_catboost_model():
    """Train CatBoost models for longitude and latitude"""
    if not CATBOOST_AVAILABLE:
        print("CatBoost is required. Install with: pip install catboost")
        return None, None, None
    
    print("=" * 60)
    print("CatBoost Model Training")
    print("=" * 60)
    
    print("\n1. Loading and preparing features...")
    X_train, X_val, y_train, y_val, X_test, feature_cols = prepare_features(
        "train_processed.csv",
        "test_processed.csv",
        sample_size=250000,
        test_split=0.2,
        random_state=42
    )
    
    print(f"   Train size: {X_train.shape}")
    print(f"   Validation size: {X_val.shape}")
    print(f"   Test size: {X_test.shape}")
    print(f"   Features: {len(feature_cols)}")
    
    # CatBoost parameters
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50,
        'task_type': 'CPU'
    }
    
    # Train longitude model
    print("\n2. Training Longitude Model...")
    model_lon = CatBoostRegressor(**cat_params)
    model_lon.fit(
        X_train, y_train.iloc[:, 0],
        eval_set=(X_val, y_val.iloc[:, 0]),
        use_best_model=True
    )
    
    # Train latitude model
    print("\n3. Training Latitude Model...")
    model_lat = CatBoostRegressor(**cat_params)
    model_lat.fit(
        X_train, y_train.iloc[:, 1],
        eval_set=(X_val, y_val.iloc[:, 1]),
        use_best_model=True
    )
    
    # Predictions
    print("\n4. Generating Predictions...")
    y_train_pred = np.column_stack([
        model_lon.predict(X_train),
        model_lat.predict(X_train)
    ])
    
    y_val_pred = np.column_stack([
        model_lon.predict(X_val),
        model_lat.predict(X_val)
    ])
    
    # Evaluate
    print("\n5. Evaluation Results:")
    print("\n--- Training Set ---")
    train_metrics = evaluate_predictions(y_train, y_train_pred, "CatBoost")
    for k, v in train_metrics.items():
        print(f"   {k}: {v:.6f}")
    
    print("\n--- Validation Set ---")
    val_metrics = evaluate_predictions(y_val, y_val_pred, "CatBoost")
    for k, v in val_metrics.items():
        print(f"   {k}: {v:.6f}")
    
    # Feature importance
    print("\n6. Top 10 Feature Importances (Longitude):")
    lon_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_lon.feature_importances_
    }).sort_values('importance', ascending=False)
    print(lon_importance.head(10).to_string(index=False))
    
    # Save models
    print("\n7. Saving models...")
    joblib.dump(model_lon, "models/catboost/catboost_lon.pkl")
    joblib.dump(model_lat, "models/catboost/catboost_lat.pkl")
    
    # Save metrics
    all_metrics = {
        "train": {k: float(v) for k, v in train_metrics.items()},
        "validation": {k: float(v) for k, v in val_metrics.items()},
        "params": cat_params,
        "features": feature_cols
    }
    
    with open("models/catboost/metrics_catboost.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n✓ CatBoost models saved successfully!")
    print("=" * 60)
    
    return model_lon, model_lat, val_metrics

if __name__ == "__main__":
    train_catboost_model()
