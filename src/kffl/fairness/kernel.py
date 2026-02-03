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
