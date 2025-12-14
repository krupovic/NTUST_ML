"""Baseline linear regression head for multi-output destination prediction."""

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output linear regression estimator."""
    try:
        base = LinearRegression(n_jobs=1)
    except TypeError:
        base = LinearRegression()
    return MultiOutputRegressor(base)
