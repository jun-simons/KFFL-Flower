"""Group-fairness metrics (Section 2.1)."""

from __future__ import annotations

import numpy as np


def statistical_parity_difference(y_hat: np.ndarray, s: np.ndarray) -> float:
    """SPD, Eq. (1):  Pr(yhat=1 | s=1) - Pr(yhat=1 | s=0)."""
    r1, r0 = y_hat[s == 1], y_hat[s == 0]
    if len(r1) == 0 or len(r0) == 0:
        return float("nan")
    return float(r1.mean() - r0.mean())


def equality_of_opportunity(
    y_hat: np.ndarray, y: np.ndarray, s: np.ndarray
) -> float:
    """EO, Eq. (2):  Pr(yhat=1 | y=1, s=1) - Pr(yhat=1 | y=1, s=0)."""
    pos = y == 1
    return statistical_parity_difference(y_hat[pos], s[pos])


def _rates(y_hat: np.ndarray, s: np.ndarray) -> list[float]:
    return [float(y_hat[s == g].mean()) for g in np.unique(s) if (s == g).any()]


def multigroup_spd(y_hat: np.ndarray, s: np.ndarray) -> float:
    """SPD_multi, Definition 1.

    NOTE: Eq. (3) as printed reads ``max_s |r_s - max_s' |r_s - r_s'||``, which
    is malformed.  Definition 2's EO_multi (Eq. 4) is the clean max-min gap
    ``max_{s,s'} |r_s - r_s'|``, and the text says both "reduce to Eqs. (1)-(2)
    when K = 2", which fixes the intended form; that is what we compute.
    """
    r = _rates(y_hat, s)
    return float(max(r) - min(r)) if r else float("nan")


def multigroup_eo(y_hat: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    """EO_multi, Eq. (4): ``max_{s,s'} |TPR_s - TPR_s'|``."""
    pos = y == 1
    return multigroup_spd(y_hat[pos], s[pos])


def ks_distance(scores: np.ndarray, s: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance for regression, Eq. (6).

    ``max_{s,s'} sup_t |F^s(t) - F^{s'}(t)|`` over the empirical CDFs (Eq. 5).
    """
    groups = [scores[s == g] for g in np.unique(s) if (s == g).any()]
    if len(groups) < 2:
        return float("nan")
    grid = np.sort(np.unique(scores))
    cdfs = np.stack([np.searchsorted(np.sort(g), grid, "right") / len(g) for g in groups])
    return float(np.abs(cdfs[:, None, :] - cdfs[None, :, :]).max())


def classification_report(
    scores: np.ndarray, y: np.ndarray, s: np.ndarray
) -> dict[str, float]:
    """Accuracy plus the fairness gaps the paper reports for classifiers."""
    y_hat = (scores > 0).astype(np.float64)  # yhat = 1[sigmoid(score) > 0.5]
    return {
        "accuracy": float((y_hat == y).mean()),
        "spd": statistical_parity_difference(y_hat, s),
        "eo": equality_of_opportunity(y_hat, y, s),
        "spd_multi": multigroup_spd(y_hat, s),
        "eo_multi": multigroup_eo(y_hat, y, s),
    }


def regression_report(
    scores: np.ndarray, y: np.ndarray, s: np.ndarray
) -> dict[str, float]:
    """RMSE plus the KS distance (Section 6.4)."""
    return {
        "rmse": float(np.sqrt(((scores - y) ** 2).mean())),
        "ks": ks_distance(scores, s),
    }
