"""Common containers for the tabular datasets used in the paper (App. B.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    """One split of a dataset, in the paper's ``{x_i, s_i, y_i}`` notation.

    Attributes
    ----------
    X : (n, d) float32
        Non-sensitive covariates fed to ``f(x; w)``.
    s : (n,) int64
        Sensitive covariate, encoded as a group index in ``{0, ..., K-1}``.
    y : (n,) float32
        Ground-truth label (binary 0/1 for classification, real for regression).
    s_partition : (n,) int64, optional
        Binary group used only to build the non-IID client split.
    """

    X: np.ndarray
    s: np.ndarray
    y: np.ndarray
    s_partition: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def partition_key(self) -> np.ndarray:
        """Binary attribute that drives the non-IID split (Appendix B.3).

        B.3 only defines the split for a binary protected group, so datasets
        with K > 2 groups supply a binary ``s_partition``; otherwise it is ``s``.
        """
        return self.s if self.s_partition is None else self.s_partition

    def subset(self, idx: np.ndarray) -> "Split":
        return Split(
            X=self.X[idx],
            s=self.s[idx],
            y=self.y[idx],
            s_partition=None if self.s_partition is None else self.s_partition[idx],
        )


@dataclass(frozen=True)
class Dataset:
    """A train/test dataset plus the metadata the model and kernels need."""

    name: str
    task: str  # "classification" | "regression"
    train: Split
    test: Split
    num_features: int  # d, the input width of f(.; w)
    num_groups: int  # K, the number of sensitive groups
    sensitive_attribute: str
    num_classes: int = 2  # C, the number of label classes (0 for regression)
