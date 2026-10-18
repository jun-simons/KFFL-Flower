"""Randomised CDF regulariser for equalised odds (extension; see notes/cdf_idea.tex).

Not in the paper.  Within each true label y it compares the two sensitive
groups' CDFs of the predicted probability r = sigmoid(f(x; w)):

    F_{s,y}(t)  = (1 / n_{s,y}) sum_{i: S_i = s, Y_i = y} 1{r_i <= t}
    Delta_y(t)  = F_{0,y}(t) - F_{1,y}(t)
    R(w)        = sum_y w_y [ (1 - beta) Delta_y(tau)^2 + (beta / M) sum_j Delta_y(T_j)^2 ]

At the decision threshold tau, Delta_1(tau) = TPR_1 - TPR_0 is the EO gap of
Eq. (2) and Delta_0(tau) the matching FPR gap, so beta = 0 targets equalised odds
at tau.  With thresholds T_j drawn
from Uniform(0, 1), beta = 1 estimates the integrated squared CDF gap.  The
indicator is smoothed as sigmoid((t - r) / h).

The penalty uses the same FAIR1 / FAIR2 rounds as the KHSIC:

    FAIR1  client  A_{i,s,y,j} = sum_{k: S=s, Y=y} sigmoid((T_j - r_k) / h),  n_{i,s,y}
           server  A = sum_i A_i, n = sum_i n_i,  Delta_{y,j} = A_{0,y,j}/n_{0,y} - A_{1,y,j}/n_{1,y}
    FAIR2  server  b_{s,y,j} = dR / dA_{s,y,j}, held fixed on the clients
           client  g^i = grad_w sum_{s,y,j} b_{s,y,j} A_{i,s,y,j}(w)

Because A is a sum over clients, sum_i g^i = grad R exactly.  The CDFs are
normalised by the *global* cell counts, which keeps them correct when clients
are non-IID.
"""

from __future__ import annotations

import numpy as np
import torch


def thresholds(tau: float, num_random: int, zeta: int) -> torch.Tensor:
    """``(T_0, ..., T_M)`` with ``T_0 = tau`` and ``T_j ~ Uniform(0, 1)``.

    Drawn from the round's shared seed, so every client and both FAIR phases
    use the same thresholds.
    """
    rng = np.random.default_rng([zeta, 1])  # independent of the RFM draws
    return torch.tensor([tau, *rng.uniform(0.0, 1.0, num_random)], dtype=torch.float32)


def threshold_weights(beta: float, num_random: int) -> torch.Tensor:
    """``a_0 = 1 - beta`` for tau and ``a_j = beta / M`` for the random thresholds."""
    if num_random == 0:
        if beta != 0.0:
            raise ValueError("cdf-beta > 0 needs cdf-num-thresholds > 0")
        return torch.ones(1)
    return torch.tensor([1.0 - beta] + [beta / num_random] * num_random)


def label_weights(n_y: torch.Tensor, kind: str) -> torch.Tensor:
    """``w_y``: ``"uniform"`` (1 / C) or ``"prevalence"`` (n_y / n)."""
    if kind == "uniform":
        return torch.full_like(n_y, 1.0 / len(n_y))
    if kind == "prevalence":
        return n_y / n_y.sum()
    raise ValueError(f"unknown cdf label weights {kind!r}")


def local_sums(
    r: torch.Tensor, s: torch.Tensor, y: torch.Tensor, T: torch.Tensor, h: float, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothed CDF sums ``A`` of shape (2, C, J) and cell counts ``n`` of shape (2, C)."""
    q = torch.sigmoid((T[None, :] - r[:, None]) / h)  # ~ 1{r <= T_j}
    S = torch.nn.functional.one_hot(s.long(), 2).to(q.dtype)
    Y = torch.nn.functional.one_hot(y.long(), num_classes).to(q.dtype)
    return torch.einsum("nk,nc,nj->kcj", S, Y, q), S.T @ Y


def objective(
    A: torch.Tensor, n: torch.Tensor, w_y: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``R`` and ``Delta`` from global sums.  A label with an empty group is skipped."""
    F = A / n.clamp(min=1.0)[..., None]
    present = (n > 0).all(dim=0).to(A.dtype)  # both groups observed for label y
    delta = (F[0] - F[1]) * present[:, None]
    return (w_y[:, None] * a[None, :] * delta**2).sum(), delta


# --------------------------------------------------------------------------
# Regulariser interface, called by the client and server apps
# --------------------------------------------------------------------------


def _probabilities(model, X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model(X).squeeze(-1))


def _local(cfg, model, X, s, y, num_groups, num_classes, zeta):
    if num_groups != 2 or num_classes < 2:
        raise ValueError("the cdf regulariser needs a binary sensitive attribute and discrete labels")
    T = thresholds(cfg.cdf_tau, cfg.cdf_num_thresholds, zeta)
    return local_sums(_probabilities(model, X), s, y, T, cfg.cdf_bandwidth, num_classes)


def client_stats(cfg, model, X, s, y, num_groups, num_classes, zeta) -> dict[str, torch.Tensor]:
    """FAIR1: the client's additive CDF sums and cell counts."""
    A, n = _local(cfg, model, X, s, y, num_groups, num_classes, zeta)
    return {"A": A, "n": n}


def server_combine(cfg, stats: list[dict[str, torch.Tensor]]) -> tuple[dict, dict]:
    """Global ``Delta`` and the coefficients ``b = dR / dA`` sent back in FAIR2."""
    A, n = sum(st["A"] for st in stats), sum(st["n"] for st in stats)
    w_y = label_weights(n.sum(dim=0), cfg.cdf_label_weights).to(A.dtype)
    a = threshold_weights(cfg.cdf_beta, cfg.cdf_num_thresholds).to(A.dtype)
    R, delta = objective(A, n, w_y, a)
    dR_ddelta = 2.0 * w_y[:, None] * a[None, :] * delta
    b = torch.stack(
        [dR_ddelta / n[0].clamp(min=1.0)[:, None], -dR_ddelta / n[1].clamp(min=1.0)[:, None]]
    )
    info = {"cdf_R": float(R), "n": int(n.sum())}
    info.update({f"cdf_delta_tau_y{c}": float(v) for c, v in enumerate(delta[:, 0])})
    return {"b": b}, info


def client_surrogate(cfg, model, X, s, y, num_groups, num_classes, zeta, broadcast) -> torch.Tensor:
    """FAIR2: ``sum b A_i(w)``, whose gradient summed over clients is ``grad R``."""
    A, _ = _local(cfg, model, X, s, y, num_groups, num_classes, zeta)
    return (broadcast["b"] * A).sum()
