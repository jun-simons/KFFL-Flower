"""Plot the accuracy-fairness trade-off curves swept by ``scripts/sweep.py``.

Reproduces the layout of Figure 3: accuracy against |SPD| and against |EO|, one
marker per fairness weight lambda, with the lambda = 0 point ("completely
unfair", i.e. FedAvg) highlighted.

    PYTHONPATH=. python scripts/plot_tradeoff.py results/sweep_*.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = [
            {k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()
             if k not in ("algorithm", "dataset", "partition")}
            | {k: r[k] for k in ("algorithm", "dataset", "partition")}
            for r in csv.DictReader(fh)
        ]
    return sorted(rows, key=lambda r: r["fairness_lambda"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/tradeoff.png"))
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for path in args.csvs:
        rows = read(path)
        label = f"{rows[0]['algorithm'].upper()} ({rows[0]['partition']})"
        for ax, metric in zip(axes, ("spd", "eo")):
            gaps = [abs(r[metric]) for r in rows]
            accs = [r["accuracy"] for r in rows]
            ax.plot(gaps, accs, "-o", ms=4, alpha=0.85, label=label)
            ax.scatter(gaps[0], accs[0], marker="*", s=200, zorder=5,
                       edgecolor="k", linewidth=0.6)

    for ax, name in zip(axes, ("SPD", "EO")):
        ax.axvspan(0, 0.05, color="tab:blue", alpha=0.08)
        ax.set_xlabel(f"|{name}| gap  (lower = fairer)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Accuracy vs {name}   (star = lambda 0)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
