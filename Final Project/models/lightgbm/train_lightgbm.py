"""
LightGBM Model for Taxi Trajectory Prediction
- Fast gradient boosting with categorical feature support
- Optimized for large-scale geospatial data
- Built-in feature importance and early stopping
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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
    
    # Existing delta features
    df["delta_lon"] = df["lon_last"] - df["lon_1st"]
    df["delta_lat"] = df["lat_last"] - df["lat_1st"]
    
    # Distance and bearing features
    df["euclidean_dist"] = np.sqrt(df["delta_lon"]**2 + df["delta_lat"]**2)
    df["bearing"] = np.arctan2(df["delta_lat"], df["delta_lon"])
    df["manhattan_dist"] = np.abs(df["delta_lon"]) + np.abs(df["delta_lat"])
    
    # Speed estimation (assuming each point is 15 seconds apart)
    df["estimated_speed"] = df["euclidean_dist"] / 0.00417  # Convert to km/h approximation
    
    # Temporal features
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
    
    # Geospatial features
    df["lat_lon_interaction"] = df["lon_1st"] * df["lat_1st"]
    df["lat_last_lon_last_interaction"] = df["lon_last"] * df["lat_last"]
    
    # Quadrant features
    df["start_quadrant"] = ((df["lat_1st"] > 41.15).astype(int) * 2 + 
                            (df["lon_1st"] > -8.62).astype(int))
    
    return df

def prepare_features(train_path, test_path, sample_size=None, test_split=0.2, random_state=42):
    """Prepare features with advanced engineering"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Add advanced features
    train = add_advanced_features(train)
    test = add_advanced_features(test)
    
    # Sample if needed
    if sample_size and sample_size < len(train):
        train = train.sample(sample_size, random_state=random_state)
    
    # Feature selection
    base_features = [
        "call_type_a", "call_type_b", "call_type_c",
        "lon_1st", "lat_1st", "delta_lon", "delta_lat",
        "euclidean_dist", "bearing", "manhattan_dist", "estimated_speed",
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
    # Filter only available features
    feature_cols = [f for f in feature_cols if f in train.columns]
    
    # Categorical features for LightGBM
    categorical_features = []
    if "time_of_day" in feature_cols:
        categorical_features.append("time_of_day")
    if "quarter" in feature_cols:
        categorical_features.append("quarter")
    if "start_quadrant" in feature_cols:
        categorical_features.append("start_quadrant")
    
    target_cols = ["lon_last", "lat_last"]
    
    X = train[feature_cols]
    y = train[target_cols]
    X_test_full = test[feature_cols]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val, X_test_full, feature_cols, categorical_features

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """Comprehensive evaluation metrics"""
    metrics = {}
    
    # MSE, MAE, R2 for longitude and latitude
    metrics["lon_mse"] = mean_squared_error(y_true.iloc[:, 0], y_pred[:, 0])
    metrics["lat_mse"] = mean_squared_error(y_true.iloc[:, 1], y_pred[:, 1])
    metrics["lon_mae"] = mean_absolute_error(y_true.iloc[:, 0], y_pred[:, 0])
    metrics["lat_mae"] = mean_absolute_error(y_true.iloc[:, 1], y_pred[:, 1])
    metrics["lon_r2"] = r2_score(y_true.iloc[:, 0], y_pred[:, 0])
    metrics["lat_r2"] = r2_score(y_true.iloc[:, 1], y_pred[:, 1])
    
    # Haversine distances
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

def train_lightgbm_model():
    """Train LightGBM models for longitude and latitude"""
    print("=" * 60)
    print("LightGBM Model Training")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing features...")
    X_train, X_val, y_train, y_val, X_test, feature_cols, cat_features = prepare_features(
        "train_processed.csv",
        "test_processed.csv",
        sample_size=250000,  # LightGBM can handle more data efficiently
        test_split=0.2,
        random_state=42
    )
    
    print(f"   Train size: {X_train.shape}")
    print(f"   Validation size: {X_val.shape}")
    print(f"   Test size: {X_test.shape}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Categorical features: {cat_features}")
    
    # LightGBM parameters optimized for geospatial regression
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Train longitude model
    print("\n2. Training Longitude Model...")
    model_lon = lgb.LGBMRegressor(**lgb_params)
    model_lon.fit(
        X_train, y_train.iloc[:, 0],
        eval_set=[(X_val, y_val.iloc[:, 0])],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)]
    )
    
    # Train latitude model
    print("\n3. Training Latitude Model...")
    model_lat = lgb.LGBMRegressor(**lgb_params)
    model_lat.fit(
        X_train, y_train.iloc[:, 1],
        eval_set=[(X_val, y_val.iloc[:, 1])],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)]
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
    train_metrics = evaluate_predictions(y_train, y_train_pred, "LightGBM")
    for k, v in train_metrics.items():
        print(f"   {k}: {v:.6f}")
    
    print("\n--- Validation Set ---")
    val_metrics = evaluate_predictions(y_val, y_val_pred, "LightGBM")
    for k, v in val_metrics.items():
        print(f"   {k}: {v:.6f}")
    
    # Feature importance
    print("\n6. Top 10 Feature Importances (Longitude):")
    lon_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_lon.feature_importances_
    }).sort_values('importance', ascending=False)
    print(lon_importance.head(10).to_string(index=False))
    
    print("\n7. Top 10 Feature Importances (Latitude):")
    lat_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_lat.feature_importances_
    }).sort_values('importance', ascending=False)
    print(lat_importance.head(10).to_string(index=False))
    
    # Save models
    print("\n8. Saving models...")
    joblib.dump(model_lon, "models/lightgbm/lightgbm_lon.pkl")
    joblib.dump(model_lat, "models/lightgbm/lightgbm_lat.pkl")
    
    # Save metrics
    all_metrics = {
        "train": {k: float(v) for k, v in train_metrics.items()},
        "validation": {k: float(v) for k, v in val_metrics.items()},
        "params": lgb_params,
        "features": feature_cols,
        "categorical_features": cat_features
    }
    
    with open("models/lightgbm/metrics_lightgbm.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n✓ LightGBM models saved successfully!")
    print("=" * 60)
    
    return model_lon, model_lat, val_metrics

if __name__ == "__main__":
    train_lightgbm_model()
