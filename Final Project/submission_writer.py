"""Utilities for producing competition submission CSV files."""

from typing import Iterable

import numpy as np
import pandas as pd


def save_submission(
    trip_ids: Iterable,
    predictions,
    output_path: str = "submission.csv",
) -> str:
    """Persist latitude/longitude predictions to CSV.

    Mirrors the formatting logic used in the legacy ``ref_solution`` script.
    ``predictions`` must be array-like with shape (n_samples, 2) where column 0
    is the predicted longitude and column 1 is the predicted latitude.
    """
    preds = np.asarray(predictions)
    if preds.ndim != 2 or preds.shape[1] != 2:
        raise ValueError("predictions must have shape (n_samples, 2)")

    submission = pd.DataFrame(
        {
            "TRIP_ID": list(trip_ids),
            "LATITUDE": preds[:, 1],
            "LONGITUDE": preds[:, 0],
        }
    )
    submission.to_csv(output_path, index=False)
    return output_path
