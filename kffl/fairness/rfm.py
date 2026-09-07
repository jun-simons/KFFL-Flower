"""Random feature maps for the RBF kernel (Appendix G).

Eq. (30):   phi(x) = sqrt(2/D) [cos(w_1^T x + b_1), ..., cos(w_D^T x + b_D)]
Eq. (33):   w_k ~ N(0, sigma^{-2} I),  b_k ~ U[0, 2 pi]

so that  kappa(x, y) = E <phi(x), phi(y)>  (Eq. 31).  We parameterise by
``gamma`` with kappa(x, y) = exp(-gamma ||x - y||^2), i.e. w ~ N(0, 2 gamma I).

Section 4 specifies the *Orthogonal* Random Feature map of Yu et al. (2016),
which replaces the i.i.d. Gaussian rows by ``W = S Q`` with ``Q`` Haar-orthogonal
and ``S`` diagonal with chi(d)-distributed entries, stacked in ``ceil(D/d)``
blocks.  (For d = 1 - the scalar model output feeding phi_f - there is nothing
to orthogonalise and ORF coincides with plain RFF.)

The map is a torch module, which meaning ``Z_f = phi_f(f(X; w))`` stays differentiable
with respect to ``w``, which is needed for the Jacobian in Lemma 2
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import torch


def _orthogonal_rows(d: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """``D x d`` matrix of orthogonal random features (Yu et al., 2016)."""
    blocks = []
    for _ in range(math.ceil(D / d)):
        Q, R = np.linalg.qr(rng.standard_normal((d, d)))
        Q *= np.sign(np.diag(R))  # make Q Haar-distributed
        # Rescale rows so their norms match those of a Gaussian matrix.
        S = np.sqrt(rng.chisquare(df=d, size=d))
        blocks.append(S[:, None] * Q)
    return np.vstack(blocks)[:D]


class RandomFeatureMap(torch.nn.Module):
    """``phi: R^d -> R^D`` approximating an RBF kernel.

    The draws are fully determined by ``seed``, which the server broadcasts as
    the shared seed ``zeta`` so that every client's ``Z_{s,i}`` and ``Z_{f,i}``
    live in the same feature space (Algorithm 2, line 2).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gamma: float,
        seed: int,
        kind: str = "orf",
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        if kind == "orf":
            W = _orthogonal_rows(in_dim, out_dim, rng)
        elif kind == "rff":
            W = rng.standard_normal((out_dim, in_dim))
        else:
            raise ValueError(f"unknown rfm kind {kind!r}; use 'orf' or 'rff'")
        W *= math.sqrt(2.0 * gamma)  # w ~ N(0, 2 gamma I)
        b = rng.uniform(0.0, 2.0 * math.pi, size=out_dim)

        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))
        self.scale = math.sqrt(2.0 / out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (n, d) -> ``Z``: (n, D)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.scale * torch.cos(x @ self.W.T + self.b)


def derive_seed(zeta: int, tag: str) -> int:
    """Derive independent-but-reproducible seeds for phi_s and phi_f from zeta.

    Uses CRC32 rather than ``hash()``: the seed must agree across client
    processes, and Python randomises string hashes per interpreter.
    """
    return int((zeta * 0x9E3779B1 + zlib.crc32(tag.encode())) % (2**31 - 1))
