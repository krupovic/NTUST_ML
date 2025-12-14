"""Baseline extra-trees regressor for multi-output destination prediction."""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> MultiOutputRegressor:
    """Return a multi-output extra-trees estimator."""
    base = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=1,
    )
    return MultiOutputRegressor(base)
