"""Plot accuracy-fairness trade-off curves from ``scripts/sweep.py`` CSVs.

Each point is one fairness weight lambda averaged over the seeds in the CSV,
with one-standard-deviation error bars on both axes; the lambda = 0 point is
starred.  Gaps are plotted as absolute values, as in Figures 3-6.  A summary
table is printed for every CSV.

    PYTHONPATH=. python scripts/plot_tradeoff.py results/sweep_kffl_adult_*.csv
    PYTHONPATH=. python scripts/plot_tradeoff.py --gaps spd_multi,eo_multi \\
        results/sweep_kffl_compas_race-sex_*.csv --out results/compas_race_sex.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarize(rows: list[dict], gaps: tuple[str, str]) -> list[dict]:
    """Mean and sample standard deviation per lambda, over seeds."""
    by_lambda = defaultdict(list)
    for row in rows:
        by_lambda[float(row["fairness_lambda"])].append(row)

    summary = []
    for lam in sorted(by_lambda):
        group = by_lambda[lam]
        entry = {"lambda": lam, "n": len(group)}
        for metric in ("accuracy", *gaps):
            values = np.array([float(r[metric]) for r in group])
            if metric != "accuracy":
                values = np.abs(values)
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            entry[metric] = (float(np.mean(values)), std)
        summary.append(entry)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("--gaps", default="spd,eo",
                   help="two gap columns, e.g. spd_multi,eo_multi (Eqs. 3-4)")
    p.add_argument("--out", type=Path, default=Path("results/tradeoff.png"))
    args = p.parse_args()

    gaps = tuple(args.gaps.split(","))
    names = tuple(g.upper().replace("_MULTI", "_multi") for g in gaps)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for path in args.csvs:
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        first = rows[0]
        method = first["algorithm"].upper()
        if (first.get("khsic_conditioning") or "none") != "none":
            method += f"-cond[{first['khsic_stratum_weights']}]"
        label = (f"{method} {first['dataset']}/"
                 f"{first['sensitive_attribute']} ({first['partition']})")
        summary = summarize(rows, gaps)

        print(f"\n{label}   [{path}]")
        print(f"{'lambda':>9} {'seeds':>5}  {'accuracy':>15}  "
              f"{'|' + names[0] + '|':>15}  {'|' + names[1] + '|':>15}")
        for e in summary:
            cells = "  ".join(f"{e[m][0]:.4f} ± {e[m][1]:.4f}" for m in ("accuracy", *gaps))
            print(f"{e['lambda']:9.2f} {e['n']:5d}  {cells}")

        acc = np.array([e["accuracy"] for e in summary])
        for ax, metric in zip(axes, gaps):
            gap = np.array([e[metric] for e in summary])
            line = ax.errorbar(gap[:, 0], acc[:, 0], xerr=gap[:, 1], yerr=acc[:, 1],
                               fmt="-o", ms=4, capsize=2, alpha=0.85, label=label)
            if summary[0]["lambda"] == 0:  # star the unregularised point, if swept
                ax.scatter(gap[0, 0], acc[0, 0], marker="*", s=200, zorder=5,
                           color=line[0].get_color(), edgecolor="k", linewidth=0.6)

    for ax, name in zip(axes, names):
        ax.axvspan(0, 0.05, color="tab:blue", alpha=0.08)
        ax.set_xlabel(f"|{name}|  (lower = fairer)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Accuracy vs {name}   (star = lambda 0, bars = 1 sd)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
