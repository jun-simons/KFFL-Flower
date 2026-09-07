"""Communication-efficient KHSIC regulariser (Section 4, Eqs. 13-20).

Theorem 2 replaces the two n x n kernel matrices of Eq. (9) by the D x D
feature-interaction matrix

    G(w) = Z_s^T Z_f(w) - n mu_s mu_f(w)^T = Z_s^T H Z_f(w)          (Eq. 13)

with  psi(w) = E ||G(w)||_F^2 / (n-1)^2, and Lemma 1 splits it over clients.
Lemma 2 then splits its gradient over clients as  g = sum_i J_{Omega_i}^T G.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalTerms:
    """``Phi^i(w_t) = {M^i(w_t), mu_{s,i}, mu_{f,i}}`` (Algorithm 2, line 3)."""

    M: torch.Tensor  # (D, D)  Eq. (14)
    mu_s: torch.Tensor  # (D,)  Eq. (19)
    mu_f: torch.Tensor  # (D,)  Eq. (20)
    n_i: int


def local_terms(Z_s: torch.Tensor, Z_f: torch.Tensor) -> LocalTerms:
    """FAIR1: the client's contribution to ``G`` (Eqs. 14, 19, 20)."""
    return LocalTerms(
        M=Z_s.T @ Z_f,  # Eq. (14)
        mu_s=Z_s.mean(dim=0),  # Eq. (19): mu_{s,i}^T = (1/n_i) 1^T Z_{s,i}
        mu_f=Z_f.mean(dim=0),  # Eq. (20)
        n_i=int(Z_s.shape[0]),
    )


def global_interaction(
    terms: list[LocalTerms],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Server side of FAIR1: assemble ``G``, ``mu_s``, ``mu_f`` (Eqs. 13, 17, 18).

    Lemma 1:
        G(w) = sum_i Z_{s,i}^T Z_{f,i} - n (1/n sum_i n_i mu_{s,i})(...)^T
    """
    n = sum(t.n_i for t in terms)
    mu_s = sum(t.n_i * t.mu_s for t in terms) / n  # Eq. (17)
    mu_f = sum(t.n_i * t.mu_f for t in terms) / n  # Eq. (18)
    G = sum(t.M for t in terms) - n * torch.outer(mu_s, mu_f)  # Eq. (13)
    return G, mu_s, mu_f, n


def local_interaction(
    M: torch.Tensor,
    mu_s: torch.Tensor,
    mu_f_local: torch.Tensor,
    mu_f_global: torch.Tensor,
    n_i: int,
    variant: str = "exact",
) -> torch.Tensor:
    """``Omega_i(w)``, the client's differentiable share of ``G`` (Eq. 15).

    variant="exact"
        ``Omega_i = M^i(w) - n_i mu_s mu_{f,i}(w)^T``.  Since ``Z_s`` does not
        depend on ``w``, ``mu_s`` is genuinely constant while ``mu_{f,i}`` is
        not, and ``sum_i Omega_i = G`` holds *identically in w*.  Only then does
        ``sum_i J_{Omega_i}^T G = J_G^T G``, i.e. Lemma 2 and the unbiasedness
        of Corollary 1.

    variant="paper-literal"
        Eq. (15) as printed, ``Omega_i = M^i(w) - n_i mu_s mu_f^T``, with the
        *global* ``mu_s mu_f^T`` that Algorithm 3 line 11 ships as one constant
        D x D block.  Its Jacobian is then just ``J_{M^i}``, which drops the
        ``-n mu_s d(mu_f)/dw`` term and yields a biased gradient estimate.
    """
    if variant == "exact":
        return M - n_i * torch.outer(mu_s, mu_f_local)
    if variant == "paper-literal":
        return M - n_i * torch.outer(mu_s, mu_f_global).detach()
    raise ValueError(f"unknown omega variant {variant!r}")


def psi_scale(n: int, mode: str) -> float:
    """Constant multiplying ``grad ||G||_F^2`` to obtain ``grad psi``.

    Eq. (9) defines psi with a ``1/(n-1)^2`` factor and Appendix A.6 carries a
    factor 2 from differentiating the squared Frobenius norm; Eqs. (13)-(16)
    print neither.  ``"none"`` reproduces the latter, literal reading.
    """
    if mode == "n-minus-1-sq":
        return 2.0 / float(n - 1) ** 2
    if mode == "n-sq":
        return 2.0 / float(n) ** 2
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown psi normalization {mode!r}")


def psi_value(G: torch.Tensor, n: int, mode: str) -> float:
    """Monitoring only: the KHSIC estimate itself.

    ``psi_scale`` is the *gradient* scale and carries the extra factor 2 from
    differentiating a squared norm, so psi is written out separately here.
    """
    norm = float((G**2).sum())
    if mode == "n-minus-1-sq":
        return norm / float(n - 1) ** 2  # Eq. (9)
    if mode == "n-sq":
        return norm / float(n) ** 2
    return norm  # "none": Eq. (13) literal
