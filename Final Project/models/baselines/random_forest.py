"""Baseline random forest regressor for multi-output destination prediction."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output random forest estimator with default settings."""
    base = RandomForestRegressor(n_estimators=100, random_state=1)
    return MultiOutputRegressor(base)
