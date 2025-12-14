"""Factory functions for baseline multi-output regressors."""

from .linear_regression import build_model as build_linear_regression
from .ridge import build_model as build_ridge
from .elastic_net import build_model as build_elastic_net
from .svr import build_model as build_svr
from .knn import build_model as build_knn
from .decision_tree import build_model as build_decision_tree
from .random_forest import build_model as build_random_forest
from .extra_trees import build_model as build_extra_trees
from .gradient_boosting import build_model as build_gradient_boosting

__all__ = [
    "build_linear_regression",
    "build_ridge",
    "build_elastic_net",
    "build_svr",
    "build_knn",
    "build_decision_tree",
    "build_random_forest",
    "build_extra_trees",
    "build_gradient_boosting",
]
