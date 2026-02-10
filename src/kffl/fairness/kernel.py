# src/kffl/fairness/orfm.py
from __future__ import annotations

import numpy as np
from pyrfm.random_feature import OrthogonalRandomFeature


def _make_orf(D: int, gamma: float, seed: int) -> OrthogonalRandomFeature:
    # use_offset=False gives cos/sin pairing 
    orf = OrthogonalRandomFeature(
        n_components=D,
        gamma=gamma,
        distribution="gaussian",
        random_fourier=True,
        use_offset=False,
        random_state=seed,
    )
    # Fit only needs to know n_features; we use 1D inputs (s and f are scalars)
    orf.fit(np.zeros((1, 1), dtype=np.float32))
    return orf


def orf_transform_1d(x: np.ndarray, D: int, gamma: float, seed: int) -> np.ndarray:
    """
    x: shape (n,) or (n,1) -> returns Z: shape (n, D)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    orf = _make_orf(D, gamma, seed)
    Z = orf.transform(x)
    return np.asarray(Z, dtype=np.float32)

def orf_transform(X: np.ndarray, *, D: int, gamma: float, seed: int) -> np.ndarray:
    """
    Orthogonal Random Features for RBF kernel.
    X: (n, d) float32/float64
    returns Z: (n, D) float32
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X.astype(np.float32, copy=False)

    d_in = X.shape[1]

    rff = OrthogonalRandomFeature(
        n_components=D,
        gamma=gamma,
        distribution="gaussian",
        random_fourier=True, # RBF
        use_offset=False, #cos/sin form 
        random_state=seed,
    )

    # Fit only to set up weights for the correct input dimension
    rff.fit(np.zeros((1, d_in), dtype=np.float32))

    Z = rff.transform(X).astype(np.float32, copy=False)
    return Z
