"""Baseline ridge regression head for multi-output destination prediction."""

from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output ridge regression estimator."""
    base = linear_model.Ridge()
    return MultiOutputRegressor(base)
