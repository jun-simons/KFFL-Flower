"""Local solver for the proximal subproblem (Algorithm 1, line 5).

    w^i_{t+1} = argmin_w  l^i(w) + (1 / 2 alpha) ||w - w_{t+1/2}||^2

Definition 3 only requires the solver to return a gamma-suboptimal point, so
the paper's choice of solver is a hyperparameter: Appendix B.1 specifies Adam,
learning rate 0.01, 5 local epochs, batch size 64.

"""

from __future__ import annotations

import torch
from torch import nn

_OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def solve_local(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str = "adam",
    prox_mu: float = 0.0,
    generator: torch.Generator | None = None,
) -> float:
    """Minimise ``l^i(w) + prox_mu * ||w - w_center||^2`` in place

    ``model`` must already hold ``w_{t+1/2}``, which is also taken as the
    proximal center, 
    Returns the mean data-fitting loss over the last epoch
    """
    try:
        opt_cls = _OPTIMIZERS[optimizer]
    except KeyError:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; available: {sorted(_OPTIMIZERS)}"
        ) from None

    params = list(model.parameters())
    # w_{t+1/2}: detached copy of the incoming model, held fixed for the round
    center = [p.detach().clone() for p in params]
    opt = opt_cls(params, lr=lr)

    n = X.shape[0]
    running, batches = 0.0, 0
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            fit = loss_fn(model(X[idx]).squeeze(-1), y[idx])
            objective = fit
            if prox_mu > 0.0:
                prox = sum(
                    ((p - c) ** 2).sum() for p, c in zip(params, center)
                )
                objective = objective + prox_mu * prox
            objective.backward()
            opt.step()
            running += float(fit.detach())
            batches += 1

    return running / max(batches, 1)
