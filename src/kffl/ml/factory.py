"""Model factory for KFFL.

Adding a new model requires only:
  1. Create ``src/kffl/ml/<name>.py`` with a ``Model`` class (``nn.Module``)
     and a ``train()`` function matching the ``TrainFn`` protocol.
  2. Add one entry to ``_MODEL_REGISTRY`` below.

Usage
-----
>>> model = create_model("logistic", input_dim=95)
>>> train_fn = get_train_fn("logistic")
>>> result = train_fn(model, loader, lr=0.01, num_epochs=3, ...)
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

import torch.nn as nn

from .base import TrainFn

# ---------------------------------------------------------------------------
# Registry: name → (module_path, model_class_name, train_fn_name)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "logistic": ("kffl.ml.logistic", "LogisticRegression", "train"),
}


def _get_module(name: str):
    """Import and return the module for the registered model *name*."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}."
        )
    module_path, _, _ = _MODEL_REGISTRY[name]
    return importlib.import_module(module_path)


def create_model(name: str, input_dim: int, **kwargs: Any) -> nn.Module:
    """Instantiate a model by registry name.

    Parameters
    ----------
    name:
        Key in ``_MODEL_REGISTRY`` (e.g. ``"logistic"``).
    input_dim:
        Number of input features; passed as the first positional argument to
        the model constructor.
    **kwargs:
        Additional keyword arguments forwarded to the model constructor
        (e.g. ``num_classes=2``).

    Returns
    -------
    nn.Module
        An initialised, untrained model.
    """
    module = _get_module(name)
    _, class_name, _ = _MODEL_REGISTRY[name]
    cls = getattr(module, class_name)
    return cls(input_dim, **kwargs)


def get_train_fn(name: str) -> TrainFn:
    """Return the training function for the model registered under *name*.

    Parameters
    ----------
    name:
        Key in ``_MODEL_REGISTRY`` (e.g. ``"logistic"``).

    Returns
    -------
    TrainFn
        A callable matching the ``TrainFn`` protocol.
    """
    module = _get_module(name)
    _, _, fn_name = _MODEL_REGISTRY[name]
    return getattr(module, fn_name)
