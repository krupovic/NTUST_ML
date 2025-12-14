"""Baseline gradient boosting regressor for multi-output destination prediction."""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output gradient boosting estimator with default settings."""
    base = GradientBoostingRegressor(random_state=0)
    return MultiOutputRegressor(base)
