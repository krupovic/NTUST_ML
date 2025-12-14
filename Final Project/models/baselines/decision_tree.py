"""Baseline decision tree regressor for multi-output destination prediction."""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output decision tree estimator."""
    base = DecisionTreeRegressor(max_depth=50, random_state=1)
    return MultiOutputRegressor(base)
