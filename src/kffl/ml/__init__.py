"""ML model definitions and local training utilities for KFFL."""

from .base import TrainFn, TrainResult, get_weights, set_weights
from .factory import create_model, get_train_fn
from .logistic import LogisticRegression

__all__ = [
    # base
    "TrainResult",
    "TrainFn",
    "get_weights",
    "set_weights",
    # factory
    "create_model",
    "get_train_fn",
    # models
    "LogisticRegression",
]
