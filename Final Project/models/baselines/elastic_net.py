"""Baseline elastic net regressor for multi-output destination prediction."""

from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output elastic net estimator."""
    base = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=1)
    return MultiOutputRegressor(base)
