"""Communication-efficient KHSIC regulariser (Section 4, Eqs. 13-20).

Theorem 2 replaces the two n x n kernel matrices of Eq. (9) by the D x D
feature-interaction matrix

    G(w) = Z_s^T Z_f(w) - n mu_s mu_f(w)^T = Z_s^T H Z_f(w)          (Eq. 13)

with  psi(w) = E ||G(w)||_F^2 / (n-1)^2, and Lemma 1 splits it over clients.
Lemma 2 then splits its gradient over clients as  g = sum_i J_{Omega_i}^T G.

Extension (not in the paper, which names it as future work).  Every quantity
below carries a leading stratum axis of length C.  C = 1 is the paper's KHSIC.
With one stratum per label y the penalty is the label-conditional KHSIC

    psi_cond(w) = sum_y w_y E ||G_y(w)||_F^2 / (n_y - 1)^2,

which vanishes when f(x; w) is independent of s given y (equalised odds).
Lemmas 1 and 2 hold inside each stratum unchanged, since each psi_y is Eq. (9)
on the sub-dataset {j : y_j = y}.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalTerms:
    """``Phi^i(w_t) = {M^{i,y}, mu_{s,i,y}, mu_{f,i,y}}`` (Algorithm 2, line 3)."""

    M: torch.Tensor  # (C, D, D)  Eq. (14), per stratum
    mu_s: torch.Tensor  # (C, D)  Eq. (19)
    mu_f: torch.Tensor  # (C, D)  Eq. (20)
    n: torch.Tensor  # (C,)  n_{i,y}, the client's sample count per stratum


def strata_onehot(y: torch.Tensor, conditioning: str, num_classes: int) -> torch.Tensor:
    """Stratum membership ``R`` of shape (n, C).

    ``"none"``  one stratum holding every sample: the paper's KHSIC.
    ``"label"`` one stratum per true label y: the conditional KHSIC.
    """
    if conditioning == "none":
        return torch.ones(len(y), 1)
    if conditioning == "label":
        if num_classes < 2:
            raise ValueError(
                "label-conditional KHSIC needs discrete labels (a classification task)"
            )
        return torch.nn.functional.one_hot(y.long(), num_classes).float()
    raise ValueError(f"unknown khsic conditioning {conditioning!r}")


def local_terms(Z_s: torch.Tensor, Z_f: torch.Tensor, R: torch.Tensor) -> LocalTerms:
    """FAIR1: the client's contribution to each ``G_y`` (Eqs. 14, 19, 20)."""
    R = R.to(Z_s.dtype)
    n = R.sum(dim=0)
    safe = n.clamp(min=1.0).unsqueeze(-1)  # an empty stratum contributes zeros
    return LocalTerms(
        # Eq. (14) on the rows of stratum y: M^{i,y} = Z_{s,i,y}^T Z_{f,i,y}
        M=torch.stack([(Z_s * R[:, c : c + 1]).T @ Z_f for c in range(R.shape[1])]),
        mu_s=(R.T @ Z_s) / safe,  # Eq. (19): row mean of Z_{s,i,y}
        mu_f=(R.T @ Z_f) / safe,  # Eq. (20)
        n=n,
    )


def global_interaction(
    terms: list[LocalTerms],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Server side of FAIR1: ``G_y``, ``mu_{s,y}``, ``mu_{f,y}``, ``n_y`` (Eqs. 13, 17, 18).

    Lemma 1 within stratum y:
        G_y(w) = sum_i M^{i,y} - n_y mu_{s,y} mu_{f,y}^T
    """
    n = sum(t.n for t in terms)
    safe = n.clamp(min=1.0).unsqueeze(-1)
    mu_s = sum(t.n.unsqueeze(-1) * t.mu_s for t in terms) / safe  # Eq. (17)
    mu_f = sum(t.n.unsqueeze(-1) * t.mu_f for t in terms) / safe  # Eq. (18)
    G = sum(t.M for t in terms) - n[:, None, None] * torch.einsum(  # Eq. (13)
        "cd,ce->cde", mu_s, mu_f
    )
    return G, mu_s, mu_f, n


def local_interaction(
    M: torch.Tensor,
    mu_s: torch.Tensor,
    mu_f_local: torch.Tensor,
    mu_f_global: torch.Tensor,
    n_i: torch.Tensor,
    variant: str = "exact",
) -> torch.Tensor:
    """``Omega_{i,y}(w)``, the client's differentiable share of ``G_y`` (Eq. 15).

    variant="exact"
        ``Omega_{i,y} = M^{i,y}(w) - n_{i,y} mu_{s,y} mu_{f,i,y}(w)^T``.  Since
        ``Z_s`` does not depend on ``w``, ``mu_{s,y}`` is genuinely constant
        while ``mu_{f,i,y}`` is not, and ``sum_i Omega_{i,y} = G_y`` holds
        *identically in w*.  Only then does
        ``sum_i J_{Omega_{i,y}}^T G_y = J_{G_y}^T G_y``, i.e. Lemma 2 and the
        unbiasedness of Corollary 1.

    variant="paper-literal"
        Eq. (15) as printed, with the *global* ``mu_s mu_f^T`` that Algorithm 3
        line 11 ships as one constant D x D block.  Its Jacobian is then just
        ``J_{M^i}``, which drops the ``-n mu_s d(mu_f)/dw`` term and yields a
        biased gradient estimate.
    """
    n_i = n_i.to(M.dtype)[:, None, None]
    if variant == "exact":
        return M - n_i * torch.einsum("cd,ce->cde", mu_s, mu_f_local)
    if variant == "paper-literal":
        return M - n_i * torch.einsum("cd,ce->cde", mu_s, mu_f_global).detach()
    raise ValueError(f"unknown omega variant {variant!r}")


def _normaliser(n: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-stratum ``1/(n_y - 1)^2`` of Eq. (9); strata with n_y < 2 are dropped."""
    n = n.double()
    if mode == "n-minus-1-sq":
        value = 1.0 / (n - 1.0).clamp(min=1.0) ** 2
    elif mode == "n-sq":
        value = 1.0 / n.clamp(min=1.0) ** 2
    elif mode == "none":
        value = torch.ones_like(n)
    else:
        raise ValueError(f"unknown psi normalization {mode!r}")
    return torch.where(n >= 2, value, torch.zeros_like(value))


def stratum_weights(n: torch.Tensor, weights: str) -> torch.Tensor:
    """The weights ``w_y`` of psi_cond; both choices give 1 when C = 1.

    ``"prevalence"``  w_y = n_y / n, so psi_cond is the average conditional
                      dependence over the empirical distribution of Y.
    ``"uniform"``     w_y = 1 / C, every label counts equally.
    """
    n = n.double()
    if weights == "prevalence":
        return n / n.sum()
    if weights == "uniform":
        return torch.full_like(n, 1.0 / len(n))
    raise ValueError(f"unknown khsic stratum weights {weights!r}")


def psi_values(G: torch.Tensor, n: torch.Tensor, mode: str) -> torch.Tensor:
    """Monitoring only: ``psi_y = ||G_y||_F^2 / (n_y - 1)^2`` for each stratum."""
    return (G.double() ** 2).sum(dim=(1, 2)) * _normaliser(n, mode)


def stratum_scales(n: torch.Tensor, mode: str, weights: str) -> torch.Tensor:
    """``c_y`` such that ``grad psi = sum_y c_y J_{G_y}^T G_y``.

    Eq. (9) carries ``1/(n_y - 1)^2`` and Appendix A.6 a factor 2 from
    differentiating the squared Frobenius norm; Eqs. (13)-(16) print neither, and
    ``mode="none"`` reproduces that literal reading.  The scales are applied on
    the client, inside one backward pass, so that the uplink stays one gradient
    of size |w| whatever the number of strata.
    """
    factor = 1.0 if mode == "none" else 2.0
    return factor * stratum_weights(n, weights) * _normaliser(n, mode)
