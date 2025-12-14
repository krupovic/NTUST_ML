"""Shared evaluation helpers for baseline destination models."""


import time
from typing import Dict, Tuple


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score



def haversine_km(lon1, lat1, lon2, lat2):
    """Compute Haversine distance in kilometers between two arrays of lon/lat."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def accuracy_within_km(pred_lon, pred_lat, true_lon, true_lat, km=1.0):
    dists = haversine_km(pred_lon, pred_lat, true_lon, true_lat)
    return np.mean(dists <= km) * 100, dists

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """Fit diagnostics for a trained multi-output regressor, with advanced metrics and timing."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Haversine metrics
    hav_val = haversine_km(val_pred[:,0], val_pred[:,1], y_val.iloc[:,0].values, y_val.iloc[:,1].values)
    mean_hav = float(np.mean(hav_val))
    median_hav = float(np.percentile(hav_val, 50))
    p90_hav = float(np.percentile(hav_val, 90))
    within_1km, _ = accuracy_within_km(val_pred[:,0], val_pred[:,1], y_val.iloc[:,0].values, y_val.iloc[:,1].values, km=1.0)
    within_2km, _ = accuracy_within_km(val_pred[:,0], val_pred[:,1], y_val.iloc[:,0].values, y_val.iloc[:,1].values, km=2.0)

    metrics = {
        "mse_train": mean_squared_error(y_train, train_pred),
        "mse_val": mean_squared_error(y_val, val_pred),
        "r2_train": r2_score(y_train, train_pred),
        "r2_val": r2_score(y_val, val_pred),
        "mean_haversine_km": mean_hav,
        "median_haversine_km": median_hav,
        "p90_haversine_km": p90_hav,
        "within_1km": within_1km,
        "within_2km": within_2km,
        "train_time_s": train_time,
    }
    return metrics, train_pred, val_pred, train_time


def predict_for_submission(model, X_full: pd.DataFrame) -> np.ndarray:
    """Predict destinations for the competition test set."""
    return model.predict(X_full)
