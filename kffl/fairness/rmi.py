"""Order-2 matrix-based Renyi mutual information regulariser (extension; notes/renyi_idea.tex).

Not in the paper.  For a PSD Gram matrix K with constant diagonal, the order-2
matrix-based entropy is  S_2(K) = -log2( ||K||_F^2 / tr(K)^2 ), and

    I_2(S; R) = S_2(K_S) + S_2(K_R) - S_2(K_S o K_R)

with the Hadamard product as the joint kernel.  K_S is the indicator kernel on
a binary S, and K_R ~ Z Z^T uses paired sine/cosine random features of the score
with ||z(r)|| = 1.  The joint feature map e_s (x) z(r) then has a block-diagonal
second moment, and everything reduces to per-group D x D matrices
A_{s,y} = sum_{i: S=s, Y=y} z(r_i) z(r_i)^T:

    a_y = sum_s n_{s,y}^2,   b_y = ||A_{0,y} + A_{1,y}||_F^2,   c_y = sum_s ||A_{s,y}||_F^2
    I_y = log2( c_y n_y^2 / (a_y b_y) ),           R(w) = sum_y w_y I_y

Group weighting (``rmi-group-weighting``).  With ``"sample"`` weights, as in the
note, I_y = log2[ sum_s pi_s^2 ||m_s||^2 / ((sum_s pi_s^2) ||sum_s pi_s m_s||^2) ]
for group shares pi_s and per-group mean second moments m_s = A_{s,y} / n_{s,y}.
That can be *negative* when the groups are imbalanced (e.g. pi = (0.9, 0.1) and
the minority's scores 20% more concentrated gives -0.05 bits), so its minimum
is not at independence.  ``"balanced"`` weights every sample by 1 / n_{s,y}, the
weighted matrix-based information, which gives
    I_y = log2( 2 (||m_0||^2 + ||m_1||^2) / ||m_0 + m_1||^2 )  in  [0, 1],
zero exactly when the two groups' (kernel-embedded) score distributions agree.
Either way a constant score gives I_y = 0, so the penalty does not by itself
prevent collapse.

Federated computation follows the same FAIR1 / FAIR2 pattern as the KHSIC: the
clients send A_{k,s,y} and n_{k,s,y}; the server forms I_y and sends back
B_{s,y} = dR/dA_{s,y} = (2 w_y / ln 2) (A_{s,y} / c_y - A_y / b_y); each client
returns grad_w sum <B_{s,y}, A_{k,s,y}(w)>_F, which sums to grad R exactly.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def features(r: torch.Tensor, num_freq: int, gamma: float, zeta: int) -> torch.Tensor:
    """Unit-norm random features ``z(r) = [cos(w_j r), sin(w_j r)] / sqrt(M)``.

    ``z(r)^T z(r') = mean_j cos(w_j (r - r'))`` estimates exp(-gamma (r - r')^2)
    for w_j ~ N(0, 2 gamma); drawn from the round's shared seed zeta.
    """
    rng = np.random.default_rng([zeta, 2])  # independent of the other draws
    w = torch.tensor(rng.normal(0.0, math.sqrt(2.0 * gamma), num_freq), dtype=r.dtype)
    rw = r[:, None] * w[None, :]
    return torch.cat([torch.cos(rw), torch.sin(rw)], dim=1) / math.sqrt(num_freq)


def strata(y: torch.Tensor, conditioning: str, num_classes: int) -> torch.Tensor:
    """One stratum per label (``"label"``) or a single stratum (``"none"``)."""
    if conditioning == "label":
        return torch.nn.functional.one_hot(y.long(), num_classes).float()
    if conditioning == "none":
        return torch.ones(len(y), 1)
    raise ValueError(f"unknown rmi conditioning {conditioning!r}")


def label_weights(n_y: torch.Tensor, kind: str) -> torch.Tensor:
    """``w_y``: ``"uniform"`` (1 / C) or ``"prevalence"`` (n_y / n)."""
    if kind == "uniform":
        return torch.full_like(n_y, 1.0 / len(n_y))
    if kind == "prevalence":
        return n_y / n_y.sum()
    raise ValueError(f"unknown rmi label weights {kind!r}")


def local_sums(Z: torch.Tensor, s: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``A`` of shape (2, C, D, D) and counts ``n`` of shape (2, C)."""
    S = torch.nn.functional.one_hot(s.long(), 2).to(Z.dtype)
    R = R.to(Z.dtype)
    return torch.einsum("nk,nc,nd,ne->kcde", S, R, Z, Z), S.T @ R


def _moments(A: torch.Tensor, n: torch.Tensor, weighting: str):
    """Sample weights ``u_s`` (1, or 1 / n_s) and the weighted ``A~_s = u_s A_s``, a, b, c, N."""
    if weighting == "sample":
        u = torch.ones_like(n)
    elif weighting == "balanced":
        u = 1.0 / n.clamp(min=1.0)
    else:
        raise ValueError(f"unknown rmi group weighting {weighting!r}")
    At = A * u[..., None, None]
    mass = u * n  # total weight of each group (0 for an empty group)
    N, a = mass.sum(dim=0), (mass**2).sum(dim=0)
    b = (At.sum(dim=0) ** 2).sum(dim=(1, 2))
    c = (At**2).sum(dim=(0, 2, 3))
    valid = (n.sum(dim=0) >= 2) & (b > 0)
    return u, At, N, a, b, c, valid


def objective(A: torch.Tensor, n: torch.Tensor, w_y: torch.Tensor, weighting: str = "balanced"):
    """``R = sum_y w_y I_y`` and the per-label ``I_y``.  Labels with n_y < 2 are skipped."""
    _, _, N, a, b, c, valid = _moments(A, n, weighting)
    tiny = torch.finfo(A.dtype).tiny
    ratio = c * N**2 / (a * b).clamp(min=tiny)
    I_y = torch.where(valid, torch.log2(ratio.clamp(min=tiny)), torch.zeros_like(ratio))
    return (w_y * I_y).sum(), I_y


# --------------------------------------------------------------------------
# Regulariser interface, called by the client and server apps
# --------------------------------------------------------------------------


def _local(cfg, model, X, s, y, num_groups, num_classes, zeta):
    if num_groups != 2:
        raise ValueError("the rmi regulariser needs a binary sensitive attribute")
    if cfg.rfm_dim % 2:
        raise ValueError("rmi uses paired sine/cosine features: rfm-dim must be even")
    scores = model(X).squeeze(-1)
    r = torch.sigmoid(scores) if cfg.fairness_output == "prob" else scores
    Z = features(r, cfg.rfm_dim // 2, cfg.kernel_gamma_f, zeta)
    return local_sums(Z, s, strata(y, cfg.rmi_conditioning, num_classes))


def client_stats(cfg, model, X, s, y, num_groups, num_classes, zeta) -> dict[str, torch.Tensor]:
    """FAIR1: per-group, per-label feature second moments ``A_{k,s,y}`` and counts."""
    A, n = _local(cfg, model, X, s, y, num_groups, num_classes, zeta)
    return {"A": A, "n": n}


def server_combine(cfg, stats: list[dict[str, torch.Tensor]]) -> tuple[dict, dict]:
    """``I_y`` from the global sums, and ``B_{s,y} = dR / dA_{s,y}`` for FAIR2."""
    A, n = sum(st["A"] for st in stats), sum(st["n"] for st in stats)
    w_y = label_weights(n.sum(dim=0), cfg.rmi_label_weights).to(A.dtype)
    R, I_y = objective(A, n, w_y, cfg.rmi_group_weighting)
    # dI_y/dA_{s,y} = (2 u_s / ln 2) (A~_{s,y} / c_y - A~_y / b_y)
    u, At, _, _, b, c, valid = _moments(A, n, cfg.rmi_group_weighting)
    tiny = torch.finfo(A.dtype).tiny
    coef = (2.0 / math.log(2.0)) * w_y * valid.to(A.dtype)
    B = (coef[None, :] * u)[..., None, None] * (
        At / c.clamp(min=tiny)[None, :, None, None]
        - At.sum(dim=0, keepdim=True) / b.clamp(min=tiny)[None, :, None, None]
    )
    info = {"rmi": float(R), "n": int(n.sum())}
    info.update({f"rmi_y{k}": float(v) for k, v in enumerate(I_y)})
    return {"B": B}, info


def client_surrogate(cfg, model, X, s, y, num_groups, num_classes, zeta, broadcast) -> torch.Tensor:
    """FAIR2: ``sum <B_{s,y}, A_{k,s,y}(w)>_F``, whose gradient summed over clients is ``grad R``."""
    A, _ = _local(cfg, model, X, s, y, num_groups, num_classes, zeta)
    return (broadcast["B"] * A).sum()
