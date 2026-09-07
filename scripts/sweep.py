"""Sweep the fairness weight lambda and collect the accuracy-fairness curve.

Reproduces the trade-off plots of Section 6.1 (Figure 3 for ADULT).  The
lambda grids are Appendix B.5; the ADULT one is `linspace(20, 1000, 20)` with
0 prepended (B.5 lists exactly that, minus a dropped 638.95).

Usage
-----
    PYTHONPATH=. python scripts/sweep.py --partition iid
    PYTHONPATH=. python scripts/sweep.py --lambdas 0,1,10,100 --extra "alpha=0.05"
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Appendix B.5, "Fairness weights".
LAMBDA_GRIDS = {
    "adult": [0.0] + list(np.linspace(20.0, 1000.0, 20)),
    "compas": [0.0, 0.01, 0.10, 20.00, 71.58, 123.16],
    "bank": [0.0, 0.01, 0.10, 1.00],
    "german": [0.0, 0.01, 0.10, 1.00],
    "acs": [0.0, 0.01, 0.10, 1.00],
}

FIELDS = [
    "algorithm", "dataset", "partition", "seed", "fairness_lambda",
    "accuracy", "spd", "eo", "spd_multi", "eo_multi", "rmse", "ks",
    "train_loss", "psi",
]


def run_one(lam: float, args, out: Path) -> dict:
    cfg = (
        f'algorithm="{args.algorithm}" dataset="{args.dataset}" '
        f'partition="{args.partition}" seed={args.seed} '
        f'fairness-lambda={lam} results-path="{out}"'
    )
    if args.extra:
        cfg += " " + args.extra
    print(f"--- lambda={lam:<10.4g} {cfg}", flush=True)
    proc = subprocess.run(
        ["flwr", "run", str(args.app), "--run-config", cfg],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.exists():
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"run failed for lambda={lam}")
    payload = json.loads(out.read_text())
    row = {"fairness_lambda": lam, "algorithm": args.algorithm,
           "dataset": args.dataset, "partition": args.partition, "seed": args.seed}
    row.update({k: v for k, v in payload["final"].items() if k in FIELDS})
    print(f"    -> " + "  ".join(f"{k}={row[k]:.4f}" for k in
                                 ("accuracy", "spd", "eo") if k in row), flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default=".")
    p.add_argument("--algorithm", default="kffl")
    p.add_argument("--dataset", default="adult")
    p.add_argument("--partition", default="iid", choices=["iid", "non-iid"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambdas", default=None,
                   help="comma-separated; defaults to the Appendix B.5 grid")
    p.add_argument("--extra", default="", help="extra --run-config pairs")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    lambdas = (
        [float(x) for x in args.lambdas.split(",")]
        if args.lambdas
        else LAMBDA_GRIDS[args.dataset]
    )
    out_csv = Path(
        args.out
        or f"results/sweep_{args.algorithm}_{args.dataset}_{args.partition}_s{args.seed}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scratch = out_csv.with_suffix(".tmp.json")

    rows = [run_one(lam, args, scratch) for lam in lambdas]
    scratch.unlink(missing_ok=True)

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
