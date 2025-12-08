"""
Generate predictions using the best trained model
"""
import pandas as pd
import numpy as np
import json
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

def check_virtual_environment():
    """Check if running inside virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("\n" + "=" * 60)
        print("  ⚠️  WARNING: Virtual Environment Not Activated!")
        print("=" * 60)
        print("\nPlease activate the virtual environment first:")
        print("  .venv\\Scripts\\Activate.ps1")
        print("\nThen run this script again.")
        print("=" * 60 + "\n")
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("\nExiting. Please activate virtual environment first.")
            sys.exit(1)
        print("\n  Proceeding without virtual environment...\n")
    else:
        print("✓ Virtual environment detected\n")

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

def prepare_test_features(test_path, feature_cols):
    """Prepare test features matching training format"""
    test = pd.read_csv(test_path)
    test = add_advanced_features(test)
    
    # Ensure all required features exist
    for col in feature_cols:
        if col not in test.columns:
            test[col] = 0  # Add missing features with default value
    
    return test[feature_cols], test["TRIP_ID"] if "TRIP_ID" in test.columns else None

def generate_predictions(model_name="auto"):
    """Generate predictions using specified model"""
    check_virtual_environment()
    
    print("=" * 60)
    print("Prediction Generation")
    print("=" * 60)
    
    # Auto-select best model
    if model_name == "auto":
        print("\n1. Auto-selecting best model...")
        
        try:
            with open("model_comparison_report.json", 'r') as f:
                report = json.load(f)
                model_name = report["best_model"]
                print(f"   Selected: {model_name}")
        except:
            print("   No comparison report found, defaulting to XGBoost")
            model_name = "XGBoost Advanced"
    
    # Load model and generate predictions
    print(f"\n2. Loading {model_name} model...")
    
    if model_name == "XGBoost Advanced":
        try:
            model_lon = joblib.load("models/xgboost/xgboost_lon_advanced.pkl")
            model_lat = joblib.load("models/xgboost/xgboost_lat_advanced.pkl")
            
            with open("models/xgboost/metrics_xgboost_advanced.json", 'r') as f:
                metrics = json.load(f)
                feature_cols = metrics["features"]
            
            print("   ✓ Loaded XGBoost models")
            
        except Exception as e:
            print(f"   ✗ Error loading XGBoost: {e}")
            return None
    
    elif model_name == "LightGBM":
        try:
            model_lon = joblib.load("models/lightgbm/lightgbm_lon.pkl")
            model_lat = joblib.load("models/lightgbm/lightgbm_lat.pkl")
            
            with open("models/lightgbm/metrics_lightgbm.json", 'r') as f:
                metrics = json.load(f)
                feature_cols = metrics["features"]
            
            print("   ✓ Loaded LightGBM models")
            
        except Exception as e:
            print(f"   ✗ Error loading LightGBM: {e}")
            return None
    
    else:
        print(f"   ✗ Model '{model_name}' not supported for direct prediction")
        print("   Supported: 'XGBoost Advanced', 'LightGBM'")
        return None
    
    # Prepare test data
    print("\n3. Preparing test data...")
    X_test, trip_ids = prepare_test_features("test_processed.csv", feature_cols)
    print(f"   Test samples: {len(X_test)}")
    
    # Generate predictions
    print("\n4. Generating predictions...")
    predictions_lon = model_lon.predict(X_test)
    predictions_lat = model_lat.predict(X_test)
    
    # Create submission file
    print("\n5. Creating submission file...")
    
    if trip_ids is not None:
        submission = pd.DataFrame({
            "TRIP_ID": trip_ids,
            "LATITUDE": predictions_lat,
            "LONGITUDE": predictions_lon
        })
    else:
        # Generate dummy trip IDs if not available
        submission = pd.DataFrame({
            "TRIP_ID": [f"T{i}" for i in range(len(predictions_lon))],
            "LATITUDE": predictions_lat,
            "LONGITUDE": predictions_lon
        })
    
    # Save submission
    output_file = f"submission_{model_name.lower().replace(' ', '_')}.csv"
    submission.to_csv(output_file, index=False)
    
    print(f"   ✓ Submission saved: {output_file}")
    print(f"   Format: TRIP_ID, LATITUDE, LONGITUDE")
    print(f"   Rows: {len(submission)}")
    
    # Show sample predictions
    print("\n6. Sample predictions:")
    print(submission.head(10).to_string(index=False))
    
    print("\n✓ Prediction generation complete!")
    print("=" * 60)
    
    return submission

if __name__ == "__main__":
    import sys
    
    # Allow command line argument for model selection
    model = sys.argv[1] if len(sys.argv) > 1 else "auto"
    generate_predictions(model)
