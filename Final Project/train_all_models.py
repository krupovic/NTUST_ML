"""
Model Comparison and Training Pipeline
Run all models and generate comprehensive comparison report
"""
import subprocess
import json
import os
import sys
import time
from datetime import datetime
import pandas as pd

def check_virtual_environment():
    """Check if running inside virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("\n" + "=" * 70)
        print("  ⚠️  WARNING: Virtual Environment Not Activated!")
        print("=" * 70)
        print("\nYou should run this script inside the .venv virtual environment.")
        print("\nTo activate the virtual environment, run:")
        print("  .venv\\Scripts\\Activate.ps1")
        print("\nThen run this script again:")
        print("  python train_all_models.py")
        print("\n" + "=" * 70)
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("\nExiting. Please activate virtual environment first.")
            sys.exit(1)
        print("\n⚠️  Proceeding without virtual environment...\n")
    else:
        print("\n✓ Virtual environment detected")
        print(f"  Python: {sys.executable}")

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def run_model(script_path, model_name):
    """Run a model training script with real-time output"""
    print(f"\n→ Training {model_name}...")
    print(f"   Using: {sys.executable}")
    print("-" * 70)
    start_time = time.time()
    
    # Use the same Python interpreter that's running this script
    python_executable = sys.executable
    
    try:
        # Run with real-time output instead of capturing
        result = subprocess.run(
            [python_executable, script_path],
            timeout=1800,  # 30 minute timeout
            check=False  # Don't raise exception on non-zero return
        )
        
        elapsed_time = time.time() - start_time
        
        print("-" * 70)
        if result.returncode == 0:
            print(f"✓ {model_name} completed in {elapsed_time:.1f} seconds")
            return True, elapsed_time, "Success"
        else:
            print(f"✗ {model_name} failed with return code {result.returncode}")
            return False, elapsed_time, f"Failed with code {result.returncode}"
    
    except subprocess.TimeoutExpired:
        print("-" * 70)
        print(f"✗ {model_name} timed out after 30 minutes!")
        return False, 1800, "Timeout"
    except Exception as e:
        print("-" * 70)
        print(f"✗ {model_name} error: {e}")
        return False, 0, str(e)

def load_metrics(metrics_path):
    """Load metrics from JSON file"""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except:
        return None

def create_comparison_report(training_results):
    """Create comprehensive comparison report"""
    print_header("MODEL COMPARISON REPORT")
    
    # Collect all metrics
    comparison_data = []
    
    for model_name, metrics_path in [
        ("XGBoost Advanced", "models/xgboost/metrics_xgboost_advanced.json"),
        ("LightGBM", "models/lightgbm/metrics_lightgbm.json"),
        ("CatBoost", "models/catboost/metrics_catboost.json"),
        ("HistGradientBoosting", "models/histgb/metrics_histgb.json"),
        ("Ensemble", "models/ensemble/ensemble_config.json")
    ]:
        metrics = load_metrics(metrics_path)
        if metrics:
            if model_name == "Ensemble":
                val_metrics = metrics.get("best_ensemble_metrics", {})
            else:
                val_metrics = metrics.get("validation", {})
            
            training_time = training_results.get(model_name, {}).get("time", 0)
            
            comparison_data.append({
                "Model": model_name,
                "Mean Haversine (km)": val_metrics.get("mean_haversine_km", float('inf')),
                "Median Haversine (km)": val_metrics.get("median_haversine_km", float('inf')),
                "P90 Haversine (km)": val_metrics.get("p90_haversine_km", float('inf')),
                "Within 1km (%)": val_metrics.get("within_1km", 0) * 100,
                "Within 2km (%)": val_metrics.get("within_2km", 0) * 100,
                "Lon R²": val_metrics.get("lon_r2", 0),
                "Lat R²": val_metrics.get("lat_r2", 0),
                "Training Time (s)": training_time
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by Mean Haversine (lower is better)
    df = df.sort_values("Mean Haversine (km)")
    
    # Print comparison table
    print("\n📊 Validation Set Performance:\n")
    print(df.to_string(index=False))
    
    # Identify best model
    best_model = df.iloc[0]["Model"]
    best_haversine = df.iloc[0]["Mean Haversine (km)"]
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Mean Haversine Distance: {best_haversine:.4f} km")
    
    # Save comparison report
    report = {
        "generated_at": datetime.now().isoformat(),
        "models": comparison_data,
        "best_model": best_model,
        "best_haversine_km": float(best_haversine),
        "training_results": training_results
    }
    
    with open("model_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save as CSV
    df.to_csv("model_comparison_report.csv", index=False)
    
    print("\n✓ Comparison report saved:")
    print("  - model_comparison_report.json")
    print("  - model_comparison_report.csv")
    
    return df, best_model

def main():
    """Main training pipeline"""
    # Check virtual environment first
    check_virtual_environment()
    
    print_header("TAXI TRAJECTORY PREDICTION - MODEL TRAINING PIPELINE")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if processed data exists
    if not os.path.exists("train_processed.csv") or not os.path.exists("test_processed.csv"):
        print("\n⚠ Processed data not found!")
        print("Please run data preparation first:")
        print("  python data_preparation.py")
        return
    
    # Model training configuration
    models_to_train = [
        ("XGBoost Advanced", "models/xgboost/train_xgboost_advanced.py"),
        ("LightGBM", "models/lightgbm/train_lightgbm.py"),
        ("CatBoost", "models/catboost/train_catboost.py"),
        ("HistGradientBoosting", "models/histgb/train_histgb.py"),
        ("Ensemble", "models/ensemble/train_ensemble.py")
    ]
    
    training_results = {}
    
    # Train each model
    print_header("TRAINING MODELS")
    
    for model_name, script_path in models_to_train:
        success, elapsed_time, output = run_model(script_path, model_name)
        
        training_results[model_name] = {
            "success": success,
            "time": elapsed_time,
            "script": script_path
        }
        
        # Continue training even if a model fails
        if not success:
            print(f"\n⚠ {model_name} failed, but continuing with remaining models...")
    
    # Create comparison report
    if any(result["success"] for result in training_results.values()):
        comparison_df, best_model = create_comparison_report(training_results)
        
        # Print recommendations
        print_header("RECOMMENDATIONS")
        print(f"\n✓ Best performing model: {best_model}")
        print(f"\n📝 Next steps:")
        print(f"  1. Review detailed metrics in models/*/metrics_*.json")
        print(f"  2. Analyze feature importances")
        print(f"  3. Generate predictions on test set")
        print(f"  4. Create submission file for Kaggle")
        
        print(f"\n💡 Model characteristics:")
        print(f"  • XGBoost: Fast, interpretable, good baseline")
        print(f"  • LightGBM: Fastest training, handles large data well")
        print(f"  • CatBoost: State-of-the-art accuracy on tabular data")
        print(f"  • HistGradientBoosting: sklearn native, very fast")
        print(f"  • Ensemble: Best performance, combines all models")
    
    else:
        print("\n✗ All models failed to train!")
        print("Please check error messages above and ensure:")
        print("  1. All required packages are installed")
        print("  2. Processed data files exist")
        print("  3. Sufficient memory and disk space")
    
    print("\n" + "=" * 70)
    print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
