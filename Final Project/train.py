"""Run baseline models and generate comparison metrics."""

import re
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from prepare_ml_data import prepare_ml_data
from models.baselines import (
    build_decision_tree,
    build_elastic_net,
    build_extra_trees,
    build_gradient_boosting,
    build_knn,
    build_linear_regression,
    build_random_forest,
    build_ridge,
    build_svr,
)
from models.evaluation import evaluate_model, predict_for_submission
from submission_writer import save_submission

TRAIN_PATH = "train_processed.csv"
TEST_PATH = "test_processed.csv"

# Model registry: name -> factory callable
TRAINERS: List[Tuple[str, Callable]] = [
    ("Linear Regression", build_linear_regression),
    ("Ridge Regression", build_ridge),
    ("Elastic Net", build_elastic_net),
    ("Support Vector Regression", build_svr),
    ("K-Nearest Neighbors", build_knn),
    ("Decision Tree", build_decision_tree),
    ("Random Forest", build_random_forest),
    ("Extra Trees", build_extra_trees),
    ("Gradient Boosting", build_gradient_boosting),
]


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics for a model."""
    print(f"{name}")
    print(f"  MSE train: {metrics['mse_train']:.6f}")
    print(f"  MSE val:   {metrics['mse_val']:.6f}")
    print(f"  R2 train:  {metrics['r2_train']:.6f}")
    print(f"  R2 val:    {metrics['r2_val']:.6f}")


def main(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH) -> None:
    """Train all baseline models, report metrics, and write a submission file."""
    X_train, X_val, y_train, y_val, X_test = prepare_ml_data(train_path, test_path)
    test_df = pd.read_csv(test_path)

    metrics_summary: List[Dict[str, Any]] = []

    for name, factory in TRAINERS:
        model = factory()
        model.fit(X_train, y_train)
        # Evaluate and fit inside evaluation to get timing and all metrics
        metrics, train_pred, val_pred, train_time = evaluate_model(model, X_train, X_val, y_train, y_val)
        print_metrics(name, metrics)
        metrics_with_name = {"model": name, **metrics}
        metrics_summary.append(metrics_with_name)

        predictions = predict_for_submission(model, X_test)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        submission_path = f"submission_{slug}.csv"
        save_submission(
            trip_ids=test_df["TRIP_ID"],
            predictions=predictions,
            output_path=submission_path,
        )
        print(f"Saved submission: {submission_path}")

    if metrics_summary:
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_path = "baseline_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()