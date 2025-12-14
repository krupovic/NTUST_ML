"""Baseline k-nearest neighbors regressor for multi-output destination prediction."""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output k-nearest neighbors estimator."""
    base = KNeighborsRegressor()
    return MultiOutputRegressor(base)
