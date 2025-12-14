"""
Run the end-to-end workflow: data load, visualization, and data preparation.

Usage:
  python run_all.py              # default cut_len=10
  python run_all.py 25           # use cut_len=25 for prefixes
"""

from project_imports import *
import sys

from data_understanding import load_and_preprocess_data
from data_visualization import (
    plot_trip_counts,
    plot_call_type_distribution,
    plot_trajectory_lengths,
    plot_start_locations,
    plot_trajectories_sample,
    plot_destinations_map,
    plot_missing_rate_table,
    plot_correlation_heatmap
)
from data_preparation import main as prepare_data


def main(cut_len: int = 10) -> None:
    print("[1/2] Generating visualizations...")
    train, test, sample, location = load_and_preprocess_data()
    plot_trip_counts(train.copy())
    plot_call_type_distribution(train.copy())
    plot_trajectory_lengths(train.copy())
    plot_start_locations(train.copy())
    plot_trajectories_sample(train.copy())
    plot_destinations_map(train.copy())
    plot_missing_rate_table(train.copy())
    # X is not available in new workflow, skip correlation heatmap

    print("[2/2] Preparing processed data files...")
    prepare_data()
    print("Saved: train_processed.csv and test_processed.csv")

    # Correlation study for processed training data
    print("[Post] Correlation study for processed training data...")
    processed_train = pd.read_csv("train_processed.csv")
    print("[Post] Rendering missing value table for processed training data...")
    # plot_missing_rate_table(processed_train)
    plot_correlation_heatmap(processed_train)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            cut_len = int(sys.argv[1])
        except ValueError:
            raise SystemExit("cut_len must be an integer, e.g., 10 or 25")
    else:
        cut_len = 10
    main(cut_len=cut_len)
