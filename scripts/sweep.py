"""Sweep the fairness weight lambda over one or more seeds.

Each (lambda, seed) pair is one ``flwr run``.  Rows are appended to a single
CSV as runs finish, and pairs already present in that CSV are skipped, so an
interrupted sweep resumes where it stopped.

The lambda grids are Appendix B.5 (Figures 3-4) and Section 6.5 (Figures 5-6).
The ADULT grid is `linspace(20, 1000, 20)` with 0 prepended; B.5 prints exactly
that, minus a dropped 638.95.

Usage
-----
    PYTHONPATH=. python scripts/sweep.py --partition iid --seeds 0,1,2,3,4 --jobs 3
    PYTHONPATH=. python scripts/sweep.py --dataset compas --sensitive race-sex \\
        --grid compas-multi --extra "rfm-dim=16" --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

LAMBDA_GRIDS = {
    # Appendix B.5, "Fairness weights".
    "adult": [0.0] + list(np.linspace(20.0, 1000.0, 20)),
    "compas": [0.0, 0.01, 0.10, 20.00, 71.58, 123.16],
    # B.5's COMPAS grid barely moves SPD under our step-size convention (README),
    # so it is extended to where the fairness term acts.
    "compas-extended": [0.0, 0.01, 0.10, 20.00, 71.58, 123.16, 250.0, 500.0,
                        1000.0, 2000.0, 5000.0],
    "bank": [0.0, 0.01, 0.10, 1.00],
    "german": [0.0, 0.01, 0.10, 1.00],
    "acs": [0.0, 0.01, 0.10, 1.00],
    # Section 6.5: lambda in {0, 1000, 1100, ..., 2000}.
    "compas-multi": [0.0] + list(np.arange(1000.0, 2001.0, 100.0)),
}

FIELDS = [
    "algorithm", "khsic_conditioning", "khsic_stratum_weights", "dataset",
    "sensitive_attribute", "partition", "seed",
    "fairness_lambda", "accuracy", "pos_rate", "spd", "eo", "fpr_gap", "eodds", "spd_multi",
    "eo_multi", "spd_multi_min20", "eo_multi_min20", "rmse", "ks", "train_loss", "psi",
]


# Several Ray-backed simulations starting in the same second occasionally fail to
# register their workers; space out launches and retry a failed run once.
STAGGER_SECONDS = 6.0
ATTEMPTS = 4
_launch_lock = threading.Lock()


def _flwr_run(cmd: list[str], stagger: float) -> tuple[int, str, str]:
    with _launch_lock:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(stagger)
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def _key(seed, lam) -> tuple[int, float]:
    return int(seed), round(float(lam), 6)


def run_one(lam: float, seed: int, args, run_dir: Path) -> dict:
    # Relative to the app directory, where the ServerApp writes it.
    out = run_dir / f"lambda{lam:.6g}_seed{seed}.json"
    pairs = [
        f'algorithm="{args.algorithm}"',
        f'dataset="{args.dataset}"',
        f'partition="{args.partition}"',
        f"seed={seed}",
        f"fairness-lambda={lam}",
        f'results-path="{out.as_posix()}"',
    ]
    if args.sensitive:
        pairs.append(f'sensitive-attribute="{args.sensitive}"')
    if args.extra:
        pairs.append(args.extra)

    result = Path(args.app) / out
    for _ in range(ATTEMPTS):
        if args.inprocess:  # same apps, no Ray: see scripts/run_inprocess.py
            cmd = [sys.executable, "scripts/run_inprocess.py", "--run-config", " ".join(pairs)]
        else:
            cmd = ["flwr", "run", args.app, "--run-config", " ".join(pairs)]
        code, stdout, stderr = _flwr_run(cmd, 0.0 if args.inprocess else STAGGER_SECONDS)
        if code == 0 and result.exists():
            break
    else:
        raise RuntimeError(
            f"lambda={lam} seed={seed} failed {ATTEMPTS} times\n"
            f"{stdout[-1500:]}\n{stderr[-1500:]}"
        )
    payload = json.loads(result.read_text())
    result.unlink()

    cfg = payload["config"]
    row = {
        "algorithm": cfg["algorithm"],
        "dataset": cfg["dataset"],
        "sensitive_attribute": cfg["sensitive_attribute"],
        "partition": cfg["partition"],
        "khsic_conditioning": cfg.get("khsic_conditioning", "none"),
        "khsic_stratum_weights": cfg.get("khsic_stratum_weights", "prevalence"),
        "seed": seed,
        "fairness_lambda": lam,
    }
    row.update({k: v for k, v in payload["final"].items() if k in FIELDS})
    return row


def _write(path: Path, rows: dict) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))


def _describe(row: dict) -> str:
    def fmt(k):
        v = float(row.get(k, "nan"))
        return f"{k}={v:+.4f}" if v == v else None

    gaps = [fmt("spd"), fmt("eo")] if fmt("spd") else [fmt("spd_multi"), fmt("eo_multi")]
    return "  ".join(x for x in [fmt("accuracy"), *gaps, fmt("pos_rate")] if x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default=".")
    p.add_argument("--algorithm", default="kffl")
    p.add_argument("--dataset", default="adult")
    p.add_argument("--sensitive", default=None, help="sensitive-attribute override")
    p.add_argument("--partition", default="iid", choices=["iid", "non-iid"])
    p.add_argument("--seeds", default="0", help="comma-separated")
    p.add_argument("--grid", default=None, help=f"one of {sorted(LAMBDA_GRIDS)}")
    p.add_argument("--lambdas", default=None, help="comma-separated; overrides --grid")
    p.add_argument("--extra", default="", help="extra --run-config pairs")
    p.add_argument("--jobs", type=int, default=1, help="concurrent runs")
    p.add_argument("--inprocess", action="store_true",
                   help="run each simulation in one process without Ray (much faster)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    lambdas = (
        [float(x) for x in args.lambdas.split(",")]
        if args.lambdas
        else LAMBDA_GRIDS[args.grid or args.dataset]
    )
    seeds = [int(x) for x in args.seeds.split(",")]
    sens = f"_{args.sensitive}" if args.sensitive else ""
    out_csv = Path(
        args.out
        or f"results/sweep_{args.algorithm}_{args.dataset}{sens}_{args.partition}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path("results/.runs") / out_csv.stem

    rows: dict = {}
    if out_csv.exists():
        with out_csv.open() as fh:
            rows = {_key(r["seed"], r["fairness_lambda"]): r for r in csv.DictReader(fh)}
    todo = [(l, s) for s in seeds for l in lambdas if _key(s, l) not in rows]
    print(f"{out_csv}: {len(todo)} runs to do, {len(rows)} already done, "
          f"{args.jobs} in parallel", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, l, s, args, run_dir): (l, s) for l, s in todo}
        for i, future in enumerate(as_completed(futures), 1):
            lam, seed = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep the sweep going; report at the end
                failures += 1
                print(f"[{i}/{len(todo)}] FAILED {exc}", file=sys.stderr, flush=True)
                continue
            rows[_key(seed, lam)] = row
            _write(out_csv, rows)
            print(f"[{i}/{len(todo)}] seed={seed} lambda={lam:<9.4g} {_describe(row)}",
                  flush=True)

    print(f"wrote {out_csv} ({len(rows)} rows, {failures} failed)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
