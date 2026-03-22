#src/kffl/fairness/orf.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


ArrayLike = Union[np.ndarray]


def _check_X(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D. Got shape {X.shape}.")
    return X


def _rng(random_state: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(int(random_state))


def _orthonormal_block(rng: np.random.Generator, d: int) -> np.ndarray:
    """
    Sample a random orthonormal matrix Q \in R^{d x d}
    by QR factorization of a standard normal matrix.
    """
    A = rng.standard_normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs  # broadcast across columns
    return Q


def _sample_orf_weights(
    rng: np.random.Generator,
    d: int,
    n_components: int,
    gamma: float,
) -> np.ndarray:
    """
    ORF for Gaussian RBF kernel:
      w ~ N(0, 2*gamma I)
    and pick orthogonal directions and chi-distributed radii.
    Generate blocks:
      W_block = diag(r) @ Q
    where Q is orthonormal, and r_i = ||g_i|| with g_i ~ N(0, I_d).
    Then scale by sqrt(2*gamma).
    Returns W with shape (n_components, d).
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0 for RBF/Gaussian features.")

    W = np.empty((n_components, d), dtype=np.float64)
    filled = 0

    while filled < n_components:
        Q = _orthonormal_block(rng, d)  # (d, d)

        # radii: each is norm of N(0, I_d), i.e. chi distribution with d dof
        G = rng.standard_normal(size=(d, d))
        r = np.linalg.norm(G, axis=1)  # (d,)

        # W_block rows: r_i * q_i^T  (if Q is (d,d) with rows orthonormal too)
        # Using rows of Q gives an orthonormal set as well.
        W_block = (r[:, None] * Q)  # (d, d)

        take = min(d, n_components - filled)
        W[filled : filled + take, :] = W_block[:take, :]
        filled += take

    # Scale to match RBF kernel parameterization
    W *= np.sqrt(2.0 * gamma)
    return W


@dataclass
class OrthogonalRandomFeaturesRBF:
    """
    Orthogonal Random Features for the Gaussian/RBF kernel.

    Approximates:
      k(x, y) = exp(-gamma * ||x - y||^2)

    Feature map:
      z(x) = sqrt(2/D) * cos(W x + b)
    with:
      b ~ Uniform(0, 2pi)
      W sampled via ORF to match N(0, 2*gamma I) marginally.
    """
    gamma: float = 1.0
    n_components: int = 256
    random_state: Optional[Union[int, np.random.Generator]] = None

    # learned / sampled at fit time
    W_: Optional[np.ndarray] = None
    b_: Optional[np.ndarray] = None
    n_features_in_: Optional[int] = None

    def fit(self, X: ArrayLike, y: None = None) -> "OrthogonalRandomFeaturesRBF":
        X = _check_X(X)
        n, d = X.shape
        self.n_features_in_ = d

        rng = _rng(self.random_state)
        self.W_ = _sample_orf_weights(rng, d=d, n_components=int(self.n_components), gamma=float(self.gamma))
        self.b_ = rng.uniform(0.0, 2.0 * np.pi, size=(int(self.n_components),))
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.W_ is None or self.b_ is None or self.n_features_in_ is None:
            raise RuntimeError("Call fit(X) before transform(X).")

        X = _check_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this was fit with {self.n_features_in_}."
            )

        # (n, d) @ (d, D)^T -> (n, D)
        projection = X @ self.W_.T
        projection += self.b_[None, :]

        Z = np.cos(projection)
        Z *= np.sqrt(2.0 / self.W_.shape[0])
        return Z

    def fit_transform(self, X: ArrayLike, y: None = None) -> np.ndarray:
        return self.fit(X).transform(X)


