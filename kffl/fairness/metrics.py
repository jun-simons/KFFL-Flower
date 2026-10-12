"""Group-fairness metrics (Section 2.1)."""

from __future__ import annotations

import numpy as np

# Section 6.5's six-race COMPAS setting has test groups of 7 (Asian) and 2
# (Native American) people, where one prediction moves a group's rate by 0.14-0.5
# and EO_multi is routinely 1.0.  The ``*_min20`` gaps repeat Eqs. (3)-(4) over
# groups with at least MIN_GROUP test samples (SPD) or positives (EO).  They are
# a supplementary diagnostic, not a metric from the paper.
MIN_GROUP = 20


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


def false_positive_gap(y_hat: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    """FPR gap:  Pr(yhat=1 | y=0, s=1) - Pr(yhat=1 | y=0, s=0).

    Together with EO this is equalised odds (Hardt et al., 2016), the criterion
    the label-conditional KHSIC targets.  Not a metric from the paper.
    """
    neg = y == 0
    return statistical_parity_difference(y_hat[neg], s[neg])


def _rates(y_hat: np.ndarray, s: np.ndarray, min_count: int = 1) -> list[float]:
    return [
        float(y_hat[s == g].mean()) for g in np.unique(s) if (s == g).sum() >= min_count
    ]


def multigroup_spd(y_hat: np.ndarray, s: np.ndarray, min_count: int = 1) -> float:
    """SPD_multi, Definition 1.

    NOTE: Eq. (3) as printed reads ``max_s |r_s - max_s' |r_s - r_s'||``, which
    is malformed.  Definition 2's EO_multi (Eq. 4) is the clean max-min gap
    ``max_{s,s'} |r_s - r_s'|``, and the text says both "reduce to Eqs. (1)-(2)
    when K = 2", which fixes the intended form; that is what we compute.
    """
    r = _rates(y_hat, s, min_count)
    return float(max(r) - min(r)) if r else float("nan")


def multigroup_eo(
    y_hat: np.ndarray, y: np.ndarray, s: np.ndarray, min_count: int = 1
) -> float:
    """EO_multi, Eq. (4): ``max_{s,s'} |TPR_s - TPR_s'|``."""
    pos = y == 1
    return multigroup_spd(y_hat[pos], s[pos], min_count)


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
    """Accuracy, the paper's fairness gaps, and positive-prediction rates.

    The binary gaps (Eqs. 1-2) are only defined for K = 2 and are NaN otherwise.
    The positive-prediction and true-positive rates, overall and per group, make
    a degenerate single-class classifier visible: it scores SPD = EO = 0.
    """
    y_hat = (scores > 0).astype(np.float64)  # yhat = 1[sigmoid(score) > 0.5]
    groups = np.unique(s)
    binary = len(groups) == 2
    report = {
        "accuracy": float((y_hat == y).mean()),
        "pos_rate": float(y_hat.mean()),
        "spd": statistical_parity_difference(y_hat, s) if binary else float("nan"),
        "eo": equality_of_opportunity(y_hat, y, s) if binary else float("nan"),
        "fpr_gap": false_positive_gap(y_hat, y, s) if binary else float("nan"),
        "spd_multi": multigroup_spd(y_hat, s),
        "eo_multi": multigroup_eo(y_hat, y, s),
        "spd_multi_min20": multigroup_spd(y_hat, s, MIN_GROUP),
        "eo_multi_min20": multigroup_eo(y_hat, y, s, MIN_GROUP),
    }
    # Equalised-odds gap: the larger of the TPR and FPR gaps.
    report["eodds"] = max(abs(report["eo"]), abs(report["fpr_gap"]))
    for g in groups:
        in_group = s == g
        positives = in_group & (y == 1)
        report[f"n_test_g{int(g)}"] = int(in_group.sum())
        report[f"pos_rate_g{int(g)}"] = float(y_hat[in_group].mean())
        report[f"tpr_g{int(g)}"] = (
            float(y_hat[positives].mean()) if positives.any() else float("nan")
        )
    return report


def regression_report(
    scores: np.ndarray, y: np.ndarray, s: np.ndarray
) -> dict[str, float]:
    """RMSE plus the KS distance (Section 6.4)."""
    return {
        "rmse": float(np.sqrt(((scores - y) ** 2).mean())),
        "ks": ks_distance(scores, s),
    }
