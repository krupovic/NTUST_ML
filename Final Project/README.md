# Taxi Trajectory Destination Prediction (ECML/PKDD 2015, Porto)

Predict final taxi trip destination (latitude, longitude) from partial trajectories and metadata.

## 1. Repository Structure
```
data_understanding.py    # Data loading & basic cleaning (train/test/meta)
data_visualization.py    # EDA visualizations (counts, maps, trajectories)
data_preparation.py      # Build training (prep_X.csv, prep_y.csv) & test feature tables
run_all.py               # Orchestrates loading, visualization, and training feature build
models/xgb/train_xgboost.py  # Train XGBoost regressors for lon/lat
models/xgb/infer_xgboost.py  # Generate submission from test set
prep_X.csv, prep_y.csv   # Generated training features & targets
submission_xgb_cut10.csv # Example generated submission file
requirements.txt         # Project dependencies
```

## 2. Setup
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data Acquisition
Place the Kaggle competition zip files in `./pkdd-15-predict-taxi-service-trajectory-i/`:
```
sampleSubmission.csv.zip
train.csv.zip
test.csv.zip
metaData_taxistandsID_name_GPSlocation.csv.zip
```

## 4. Data Understanding & Visualization
Run visualizations (PNG files saved in project root):
```bash
python data_visualization.py
```
Generates:
- `trip_counts_distribution.png`
- `call_type_distribution.png`
- `trajectory_length_distribution.png`
- `start_locations_map.png`
- `trajectories_sample.png`
- `destinations_map.png`

## 5. Feature Preparation
Build prefix-based training features (uses first `cut_len` points per trajectory):
```bash
python run_all.py 10        # produces prep_X.csv & prep_y.csv
```
Or directly:
```bash
python data_preparation.py  # default cut_len=10 inside main()
```
Outputs:
- `prep_X.csv` – feature matrix (metadata, time features, prefix geometry)
- `prep_y.csv` – destination coordinates (`dest_lon`, `dest_lat`)

## 6. Model Training (XGBoost Baseline)
Train longitude and latitude regressors:
```bash
python models/xgb/train_xgboost.py
```
Artifacts saved to `models/xgb/`:
- `xgb_lon.joblib`
- `xgb_lat.joblib`
- `metrics_xgb.json` (contains mean/p50/p90 Haversine, within 1km/2km, counts)

### Evaluation Metrics (reported)
- Mean Haversine Distance (km)
- P50 / P90 Haversine
- Within 1 km / Within 2 km accuracy

## 7. Inference & Submission Generation
Generate a submission file for the test set:
```bash
python models/xgb/infer_xgboost.py      # default cut_len=10
```
Outputs:
- `submission_xgb_cut10.csv` with columns:
  - `TRIP_ID`
  - `LATITUDE`
  - `LONGITUDE`

Adjust prefix length:
```bash
python models/xgb/infer_xgboost.py      # (edit cut_len in code or modify main())
```

## 8. Submission Format Example
```
TRIP_ID,LATITUDE,LONGITUDE
T1,41.146504,-8.611317
T2,42.230000,-8.629454
T10,42.110000,-8.721111
```

## 9. Extending the Pipeline
Potential improvements:
- Time-aware validation (train earlier weeks → validate later weeks).
- Multiple prefix cuts per trajectory (e.g., lengths 5, 10, 20) to expand training data.
- Additional features: average speed, bearing changes, curvature, H3/geohash cells, destination clustering.
- Uncertainty estimation (predict error radius or quantile regression).
- Ensemble: Combine kNN, cluster-centroid classifier, boosted trees, and sequence models.

## 10. Reproducibility Notes
- Prefix duration assumes 15s sampling interval between points.
- Frequency encoding used for high-cardinality fields (`ORIGIN_CALL`, `TAXI_ID`) to control memory.
- One-hot kept for low-cardinality fields (`CALL_TYPE`, `ORIGIN_STAND`).
- No data leakage: destination only used in `prep_y.csv` (labels), prefix features computed from partial points.

## 11. Quick Command Summary
```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. EDA
python data_visualization.py

# 3. Feature prep (cut_len=10)
python run_all.py 10

# 4. Train XGBoost
python models/xgb/train_xgboost.py

# 5. Inference / submission
python models/xgb/infer_xgboost.py
```
