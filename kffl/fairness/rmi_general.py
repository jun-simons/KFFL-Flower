"""General matrix-based Renyi dependence via random features (extension; notes/renyi_expanded.tex).

Not in the paper.  Unlike ``rmi.py`` it does not assume a binary or one-hot
sensitive attribute.  With feature maps v(s) for the sensitive attribute and z(r)
for the model score, and a joint map u(s, r), the three feature covariances

    C_S = sum_i v_i v_i^T,   C_R = sum_i z_i z_i^T,   C_SR = sum_i u_i u_i^T

have the same nonzero spectra as the n x n Gram matrices (Lemma 1), so

    H_alpha(C) = log2( tr(C^alpha) / tr(C)^alpha ) / (1 - alpha)
    I_alpha    = H_alpha(C_S) + H_alpha(C_R) - H_alpha(C_SR)        (eq. MI)

using the trace normalisation of eq. (nonunit), which is valid for any feature
norms.  Options (``rmig-*`` in pyproject.toml):

  sensitive kernel  "equality": v(s) = e_s, exact; "rbf": paired RFF on the
                    encoded attribute (sensitive-encoding), D_S = rmig-sensitive-dim
  joint features    "tensor": u = v (x) z, so U U^T = K_S o K_R exactly (Sec. 3);
                    "direct": paired RFF with joint Gaussian frequencies (Sec. 11),
                    D_J = rmig-joint-dim.  Its joint entropy is capped at log2 D_J,
                    so D_J must be large (see scripts/check_renyi_expanded.py).
  order             rmig-alpha > 0, alpha != 1
  conditioning      "none", or per true label y with weights w_y (Sec. 13)

Federated computation (Sec. 12): clients send C_{S,k}, C_{R,k}, C_{SR,k}; the
server sums them, evaluates I, and returns B_R = dI/dC_R and B_SR = dI/dC_SR
(C_S does not depend on w); each client returns the gradient of
<B_R, C_{R,k}(w)> + <B_SR, C_{SR,k}(w)>, which sums to grad I exactly.

Like ``rmi.py`` with sample weights, the value can be negative, so its minimum is
not guaranteed to be at independence.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Seed tags: each feature map gets its own independent draw from the round's zeta.
_SEED_S, _SEED_R, _SEED_J = 10, 11, 12


def rff(x: torch.Tensor, num_freq: int, gamma: float, seed: list[int]) -> torch.Tensor:
    """Unit-norm paired features for exp(-gamma ||x - x'||^2): frequencies ~ N(0, 2 gamma I)."""
    rng = np.random.default_rng(seed)
    W = torch.tensor(rng.normal(0.0, math.sqrt(2.0 * gamma), (x.shape[1], num_freq)), dtype=x.dtype)
    P = x @ W
    return torch.cat([torch.cos(P), torch.sin(P)], dim=1) / math.sqrt(num_freq)


def direct_features(
    s_in: torch.Tensor, r_in: torch.Tensor, num_freq: int, gamma_s: float, gamma_r: float, seed: list[int]
) -> torch.Tensor:
    """Joint paired RFF q(s, r) for k_S k_R with independent (xi_j, eta_j) (eq. directfeat)."""
    rng = np.random.default_rng(seed)
    xi = rng.normal(0.0, math.sqrt(2.0 * gamma_s), (s_in.shape[1], num_freq))
    eta = rng.normal(0.0, math.sqrt(2.0 * gamma_r), (r_in.shape[1], num_freq))
    P = s_in @ torch.tensor(xi, dtype=s_in.dtype) + r_in @ torch.tensor(eta, dtype=r_in.dtype)
    return torch.cat([torch.cos(P), torch.sin(P)], dim=1) / math.sqrt(num_freq)


def tensor_features(V: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Rows u_i = v_i (x) z_i (eq. tensor)."""
    return (V[:, :, None] * Z[:, None, :]).reshape(V.shape[0], -1)


def strata(y: torch.Tensor, conditioning: str, num_classes: int) -> torch.Tensor:
    if conditioning == "none":
        return torch.ones(len(y), 1)
    if conditioning == "label":
        if num_classes < 2:
            raise ValueError("label conditioning needs discrete labels (a classification task)")
        return torch.nn.functional.one_hot(y.long(), num_classes).float()
    raise ValueError(f"unknown rmig conditioning {conditioning!r}")


def label_weights(n_y: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "uniform":
        return torch.full_like(n_y, 1.0 / len(n_y))
    if kind == "prevalence":
        return n_y / n_y.sum()
    raise ValueError(f"unknown rmig label weights {kind!r}")


def second_moments(F: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Per-stratum feature covariances ``sum_{i in y} f_i f_i^T``, shape (C, D, D)."""
    return torch.einsum("nc,nd,ne->cde", R.to(F.dtype), F, F)


def _spectrum(C: torch.Tensor) -> torch.Tensor:
    ev = torch.linalg.eigvalsh(C).clamp(min=0.0)
    return ev * (ev > 1e-12 * ev.amax(dim=-1, keepdim=True).clamp(min=1e-300))


def power_trace(C: torch.Tensor, alpha: float) -> torch.Tensor:
    """``tr(C^alpha)`` over the nonzero spectrum (exact Frobenius form at alpha = 2)."""
    if alpha == 2.0:
        return (C**2).sum(dim=(-2, -1))
    ev = _spectrum(C)
    return torch.where(ev > 0, ev.clamp(min=1e-300) ** alpha, torch.zeros_like(ev)).sum(dim=-1)


def entropy(C: torch.Tensor, alpha: float) -> torch.Tensor:
    """``H_alpha`` with trace normalisation, batched over the leading axis."""
    T = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=-1)
    return (torch.log2(power_trace(C, alpha)) - alpha * torch.log2(T)) / (1.0 - alpha)


def entropy_grad(C: torch.Tensor, alpha: float) -> torch.Tensor:
    """``dH_alpha / dC = alpha / ((1 - alpha) ln 2) (C^(alpha-1) / tr C^alpha - I / tr C)``."""
    T = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]
    Q = power_trace(C, alpha)[..., None, None]
    if alpha == 2.0:
        P = C
    else:  # C^(alpha - 1) on the nonzero spectrum
        ev, vecs = torch.linalg.eigh(C)
        ev = ev.clamp(min=0.0)
        keep = ev > 1e-12 * ev.amax(dim=-1, keepdim=True).clamp(min=1e-300)
        powered = torch.where(keep, ev.clamp(min=1e-300) ** (alpha - 1.0), torch.zeros_like(ev))
        P = vecs @ torch.diag_embed(powered) @ vecs.transpose(-2, -1)
    eye = torch.eye(C.shape[-1], dtype=C.dtype)
    return alpha / ((1.0 - alpha) * math.log(2.0)) * (P / Q - eye / T)


def objective(C_S, C_R, C_SR, n, w_y, alpha: float):
    """``R = sum_y w_y I_y`` and the per-stratum ``I_y``; strata with fewer than 2 samples are skipped."""
    valid = n >= 2
    I_y = entropy(C_S, alpha) + entropy(C_R, alpha) - entropy(C_SR, alpha)
    I_y = torch.where(valid, I_y, torch.zeros_like(I_y))
    return (w_y * I_y).sum(), I_y


# --------------------------------------------------------------------------
# Regulariser interface, called by the client and server apps
# --------------------------------------------------------------------------


def _features(cfg, model, X, s, y, num_groups, num_classes, zeta):
    for name in ("rfm_dim", "rmig_sensitive_dim", "rmig_joint_dim"):
        if getattr(cfg, name) % 2:
            raise ValueError(f"{name} must be even: the features come in sine/cosine pairs")
    scores = model(X).squeeze(-1)
    r_in = (torch.sigmoid(scores) if cfg.fairness_output == "prob" else scores)[:, None]
    if cfg.sensitive_encoding == "one-hot":
        s_in = torch.nn.functional.one_hot(s.long(), num_groups).to(r_in.dtype)
    else:
        s_in = s.to(r_in.dtype)[:, None]
    if cfg.rmig_sensitive_kernel == "equality":
        V = torch.nn.functional.one_hot(s.long(), num_groups).to(r_in.dtype)
    elif cfg.rmig_sensitive_kernel == "rbf":
        V = rff(s_in, cfg.rmig_sensitive_dim // 2, cfg.kernel_gamma_s, [zeta, _SEED_S])
    else:
        raise ValueError(f"unknown rmig sensitive kernel {cfg.rmig_sensitive_kernel!r}")
    Z = rff(r_in, cfg.rfm_dim // 2, cfg.kernel_gamma_f, [zeta, _SEED_R])
    if cfg.rmig_joint == "tensor":
        J = tensor_features(V, Z)
    elif cfg.rmig_joint == "direct":
        if cfg.rmig_sensitive_kernel != "rbf":
            raise ValueError("direct joint features approximate a Gaussian product kernel: use rmig-sensitive-kernel='rbf'")
        J = direct_features(s_in, r_in, cfg.rmig_joint_dim // 2, cfg.kernel_gamma_s, cfg.kernel_gamma_f,
                            [zeta, _SEED_J])
    else:
        raise ValueError(f"unknown rmig joint {cfg.rmig_joint!r}")
    return V, Z, J, strata(y, cfg.rmig_conditioning, num_classes)


def client_stats(cfg, model, X, s, y, num_groups, num_classes, zeta) -> dict[str, torch.Tensor]:
    """FAIR1: per-stratum feature covariances ``C_S``, ``C_R``, ``C_SR`` and counts."""
    V, Z, J, R = _features(cfg, model, X, s, y, num_groups, num_classes, zeta)
    return {"C_S": second_moments(V, R), "C_R": second_moments(Z, R), "C_SR": second_moments(J, R),
            "n": R.sum(dim=0).to(Z.dtype)}


def server_combine(cfg, stats: list[dict[str, torch.Tensor]]) -> tuple[dict, dict]:
    """``I_y`` from the summed covariances and the FAIR2 coefficients ``B_R``, ``B_SR`` (Sec. 12)."""
    C_S, C_R, C_SR, n = (sum(st[k] for st in stats) for k in ("C_S", "C_R", "C_SR", "n"))
    w_y = label_weights(n, cfg.rmig_label_weights).to(C_R.dtype)
    R, I_y = objective(C_S, C_R, C_SR, n, w_y, cfg.rmig_alpha)
    scale = (w_y * (n >= 2).to(C_R.dtype))[:, None, None]
    B_R = scale * entropy_grad(C_R, cfg.rmig_alpha)  # dI/dC_R  = + dH(C_R)/dC_R
    B_SR = -scale * entropy_grad(C_SR, cfg.rmig_alpha)  # dI/dC_SR = - dH(C_SR)/dC_SR
    info = {"rmig": float(R), "n": int(n.sum())}
    info.update({f"rmig_y{c}": float(v) for c, v in enumerate(I_y)})
    return {"B_R": B_R, "B_SR": B_SR}, info


def client_surrogate(cfg, model, X, s, y, num_groups, num_classes, zeta, broadcast) -> torch.Tensor:
    """FAIR2: ``<B_R, C_{R,k}(w)> + <B_SR, C_{SR,k}(w)>``, summing over clients to ``grad I``."""
    _, Z, J, R = _features(cfg, model, X, s, y, num_groups, num_classes, zeta)
    return (broadcast["B_R"] * second_moments(Z, R)).sum() + (broadcast["B_SR"] * second_moments(J, R)).sum()
