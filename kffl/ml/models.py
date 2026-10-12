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


def build_model(name: str, num_features: int, init: str = "default") -> nn.Module:
    """Construct a model; ``init="zeros"`` zeroes every parameter.

    The paper never specifies the initial model ``w_0``.  For the linear models,
    zeros make ``f(X; w_0)`` constant, so the predictions carry no information
    about ``s``, ``G(w_0) = 0``, and the first fairness step vanishes instead of
    being taken on an arbitrary random model.  Do not use it for the MLP: zero
    hidden weights receive no gradient.
    """
    try:
        cls = MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODELS)}"
        ) from None
    model = cls(num_features)
    if init == "zeros":
        for param in model.parameters():
            nn.init.zeros_(param)
    elif init != "default":
        raise ValueError(f"unknown model init {init!r}; use 'default' or 'zeros'")
    return model
