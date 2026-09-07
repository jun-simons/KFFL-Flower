"""Data-fitting terms ``l^i`` of the composite objective (Eq. 10)."""

from __future__ import annotations

import torch
from torch import nn

_LOSSES = {
    # "cross-entropy loss for classification tasks or the mean squared error
    # (MSE) for regression tasks" (Sec. 2.3).
    "classification": nn.BCEWithLogitsLoss,
    "regression": nn.MSELoss,
}


def build_loss(task: str) -> nn.Module:
    try:
        return _LOSSES[task]()
    except KeyError:
        raise ValueError(
            f"unknown task {task!r}; available: {sorted(_LOSSES)}"
        ) from None


def predict(scores: torch.Tensor, task: str) -> torch.Tensor:
    """Map raw model scores to predictions used by the fairness metrics."""
    if task == "classification":
        return (scores > 0).float()  # yhat = 1[sigmoid(score) > 0.5]
    return scores
