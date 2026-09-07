"""Models used for local training (Appendix B.4).

Both return a raw score (logit for classification, prediction for regression)
of shape ``(n, 1)``; the fairness regulariser feeds this score to the RFM
``phi_f`` (Appendix G: "the model output (e.g., logits)")
"""

from __future__ import annotations

import torch
from torch import nn


class LogisticRegression(nn.Module):
    """"binary logistic regression model with a sigmoid activation" (App. B.4).

    The sigmoid lives in the loss (``BCEWithLogitsLoss``) so that ``forward``
    returns the logit
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    """"a single hidden layer with 100 neurons [...] ReLU activation" (App. B.4)."""

    def __init__(self, num_features: int, hidden: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODELS = {
    "logistic-regression": LogisticRegression,
    "mlp": MLP,
    # "linear-regression" is LogisticRegression's architecture under an MSE
    # loss (App. B.4: "a linear model for regression"); select it via `task`.
    "linear-regression": LogisticRegression,
}


def build_model(name: str, num_features: int) -> nn.Module:
    try:
        cls = MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODELS)}"
        ) from None
    return cls(num_features)
