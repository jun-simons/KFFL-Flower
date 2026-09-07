"""Model / optimisation layer."""

from kffl.ml.local import solve_local
from kffl.ml.losses import build_loss, predict
from kffl.ml.models import MODELS, build_model

__all__ = ["MODELS", "build_model", "build_loss", "predict", "solve_local"]
