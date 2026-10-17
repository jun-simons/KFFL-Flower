"""Equalised-odds baseline: FedAvg followed by group-specific decision thresholds.

sort of simplified version of Hardt et al. (2016).  For each equalised-odds budget eps, the
pair of per-group thresholds with the best *training* accuracy among those whose
training equalised-odds gap is at most eps is applied to the test set.

Unlike KFFL this uses the sensitive attribute at prediction time, and choosing
the thresholds needs per-group score statistics from the clients, so it is a
strong reference point rather than a like-for-like competitor.

FedAvg models are trained on demand and cached under results/models/, so
re-running resumes.  The CSV has the columns of scripts/sweep.py (the budget
eps is stored as ``fairness_lambda``), so scripts/plot_tradeoff.py can overlay it.

    PYTHONPATH=. python scripts/eo_baseline.py --dataset adult --partition non-iid --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import torch

from kffl.dataset import get_dataset
from kffl.fairness.metrics import classification_report
from kffl.ml import build_model

SENSITIVE = {"adult": "sex", "compas": "african-american"}
BUDGETS = [1.0, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
COLUMNS = ["algorithm", "khsic_conditioning", "khsic_stratum_weights", "dataset",
           "sensitive_attribute", "partition", "seed", "fairness_lambda",
           "accuracy", "pos_rate", "spd", "eo", "fpr_gap", "eodds"]


def fedavg_scores(dataset: str, partition: str, seed: int, ds) -> tuple[np.ndarray, np.ndarray]:
    path = Path(f"results/models/fedavg_{dataset}_{partition}_s{seed}.pt")
    if not path.exists():
        cfg = (f'algorithm="fedavg" dataset="{dataset}" sensitive-attribute="{SENSITIVE[dataset]}" '
               f'partition="{partition}" seed={seed} save-model-path="{path.as_posix()}" '
               f'results-path="results/models/fedavg_{dataset}_{partition}_s{seed}.json"')
        subprocess.run(["flwr", "run", ".", "--run-config", cfg], check=True, capture_output=True)
    model = build_model("logistic-regression", ds.num_features)
    model.load_state_dict(torch.load(path))
    with torch.no_grad():
        return (model(torch.from_numpy(ds.train.X)).squeeze(-1).numpy(),
                model(torch.from_numpy(ds.test.X)).squeeze(-1).numpy())


def rates(scores, y, cuts):
    """For each cut: true positives, false positives (counts) when predicting score > cut."""
    pred = scores[None, :] > cuts[:, None]
    return (pred & (y == 1)).sum(1), (pred & (y == 0)).sum(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="adult", choices=sorted(SENSITIVE))
    p.add_argument("--partition", default="iid", choices=["iid", "non-iid"])
    p.add_argument("--seeds", default="0,1,2,3,4")
    args = p.parse_args()

    ds = get_dataset(args.dataset, SENSITIVE[args.dataset], True)
    s, y = ds.train.s, ds.train.y
    rows = []
    for seed in [int(v) for v in args.seeds.split(",")]:
        train, test = fedavg_scores(args.dataset, args.partition, seed, ds)
        cuts = {g: np.quantile(train[s == g], np.linspace(0.0, 1.0, 201)) for g in (0, 1)}
        tp, fp, pos, neg = {}, {}, {}, {}
        for g in (0, 1):
            tp[g], fp[g] = rates(train[s == g], y[s == g], cuts[g])
            pos[g], neg[g] = (y[s == g] == 1).sum(), (y[s == g] == 0).sum()
        # Every pair of cuts (group 0 down the rows, group 1 across the columns).
        correct = (tp[0] + neg[0] - fp[0])[:, None] + (tp[1] + neg[1] - fp[1])[None, :]
        gap = np.maximum(np.abs(tp[1] / pos[1] - (tp[0] / pos[0])[:, None]),
                         np.abs(fp[1] / neg[1] - (fp[0] / neg[0])[:, None]))
        for eps in BUDGETS:
            i, j = np.unravel_index(np.argmax(np.where(gap <= eps, correct, -1)), gap.shape)
            shifted = test - np.where(ds.test.s == 1, cuts[1][j], cuts[0][i])  # > 0 means positive
            r = classification_report(shifted.astype(np.float64), ds.test.y.astype(np.float64), ds.test.s)
            rows.append({"algorithm": "fedavg+thresholds", "khsic_conditioning": "none",
                         "khsic_stratum_weights": "", "dataset": args.dataset,
                         "sensitive_attribute": SENSITIVE[args.dataset], "partition": args.partition,
                         "seed": seed, "fairness_lambda": eps, **{k: r[k] for k in COLUMNS[8:]}})
        print(f"seed {seed}: " + "  ".join(f"eps={r['fairness_lambda']:g}: {r['accuracy']:.3f}/{r['eodds']:.3f}"
                                            for r in rows[-len(BUDGETS):]), flush=True)

    out = Path(f"results/eo_baseline_{args.dataset}_{args.partition}.csv")
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  (per budget: test accuracy / test eq.odds gap)")


if __name__ == "__main__":
    main()
