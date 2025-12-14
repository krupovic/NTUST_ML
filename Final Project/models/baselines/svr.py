"""Baseline support vector regressor for multi-output destination prediction."""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


def build_model() -> MultiOutputRegressor:
    """Return a multi-output support vector regression estimator."""
    base = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    return MultiOutputRegressor(base)
