"""Shared contracts and utilities for all KFFL ML models.

Contents:
  - ``TrainResult``     — typed return value from every ``train()`` function
  - ``get_weights``     — extract model parameters as a list of numpy arrays
  - ``set_weights``     — load parameters back into a model in-place
  - ``TrainFn``         — Protocol describing the ``train()`` signature that
                          every model-specific module must satisfy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Structured return value from a model's ``train()`` function.

    Attributes:
        loss_history: Per-epoch mean loss over all batches.
        num_examples: Total number of training samples seen.
        num_epochs: Number of epochs completed.
    """

    loss_history: List[float]
    num_examples: int
    num_epochs: int

    @property
    def final_loss(self) -> float:
        """Mean loss of the last epoch."""
        return self.loss_history[-1] if self.loss_history else float("nan")


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Return all trainable parameters as a list of numpy arrays.

    The order matches ``model.parameters()``, which is deterministic for a
    given model architecture.  This is the format used by the KFFL server to
    communicate model updates over Flower messages.
    """
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Load parameter values from *weights* into *model* in-place.

    Parameters
    ----------
    model:
        The model whose parameters will be overwritten.
    weights:
        List of arrays in the same order as ``model.parameters()``.

    Raises
    ------
    ValueError
        If the number of weight arrays does not match the model's parameters.
    """
    params = list(model.parameters())
    if len(weights) != len(params):
        raise ValueError(
            f"Expected {len(params)} weight arrays, got {len(weights)}."
        )
    with torch.no_grad():
        for param, w in zip(params, weights):
            param.copy_(torch.as_tensor(w, dtype=param.dtype, device=param.device))


# ---------------------------------------------------------------------------
# Trainer Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainFn(Protocol):
    """Protocol that every model-specific ``train`` function must satisfy.

    This lets the factory return a typed callable without coupling to any
    specific model class.
    """

    def __call__(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        *,
        lr: float,
        num_epochs: int,
        weight_decay: float,
        proximal_mu: float,
        proximal_center: Optional[List[np.ndarray]],
        device: str,
    ) -> TrainResult:
        ...
